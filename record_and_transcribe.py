import argparse
import subprocess
import sys
import time
from pathlib import Path
import shutil
import os
import math
import numpy as np
import soundfile as sf
from playwright.sync_api import sync_playwright
import json


def get_ffmpeg_path():
    p = shutil.which("ffmpeg")
    if p:
        return p
    alt = Path(os.environ.get("LOCALAPPDATA", "")) / "ms-playwright" / "ffmpeg-1010" / "ffmpeg-win64.exe"
    if alt.exists():
        return str(alt)
    return None


def try_record_ffmpeg(out_wav: Path, duration: int, audio_device: str = None):
    ff = get_ffmpeg_path()
    if not ff:
        raise RuntimeError("未检测到ffmpeg，请安装或确保Playwright的ffmpeg可用")
    cmds = []
    if audio_device:
        cmds.append([
            ff, "-hide_banner", "-y",
            "-f", "dshow", "-i", f"audio={audio_device}",
            "-t", str(duration),
            "-ac", "1", "-ar", "16000", "-vn",
            str(out_wav),
        ])
    cmds.append([
        ff, "-hide_banner", "-y",
        "-f", "dshow", "-i", "audio=virtual-audio-capturer",
        "-t", str(duration),
        "-ac", "1", "-ar", "16000", "-vn",
        str(out_wav),
    ])
    cmds.append([
        ff, "-hide_banner", "-y",
        "-f", "dshow", "-i", "audio=Stereo Mix (Realtek(R) Audio)",
        "-t", str(duration),
        "-ac", "1", "-ar", "16000", "-vn",
        str(out_wav),
    ])
    cmds.append([
        ff, "-hide_banner", "-y",
        "-f", "wasapi", "-i", "default",
        "-t", str(duration),
        "-ac", "1", "-ar", "16000", "-vn",
        str(out_wav),
    ])
    last_error = None
    for cmd in cmds:
        try:
            subprocess.run(cmd, check=True)
            return True
        except Exception as e:
            last_error = e
    if last_error:
        raise last_error
    return False


def try_record_sounddevice(out_wav: Path, duration: int, samplerate: int = 16000):
    import sounddevice as sd
    # Prefer WASAPI loopback
    extra = None
    try:
        extra = sd.WasapiSettings(exclusive=False, loopback=True)
    except Exception:
        extra = None
    # pick a WASAPI output device if available
    dev_index = None
    try:
        hostapis = sd.query_hostapis()
        wasapi_index = None
        for i, ha in enumerate(hostapis):
            if "WASAPI" in ha.get("name", ""):
                wasapi_index = i
                break
        devices = sd.query_devices()
        for i, d in enumerate(devices):
            if wasapi_index is not None and d.get("hostapi") == wasapi_index and d.get("max_output_channels", 0) > 0:
                dev_index = i
                break
    except Exception:
        dev_index = None
    frames = int(duration * samplerate)
    data = sd.rec(frames, samplerate=samplerate, channels=1, dtype='float32', device=dev_index, extra_settings=extra)
    sd.wait()
    sf.write(str(out_wav), data.squeeze(), samplerate, subtype='PCM_16')
    return True

def try_record_soundcard(out_wav: Path, duration: int, samplerate: int = 16000):
    import soundcard as sc
    import numpy as np
    spk = sc.default_speaker()
    mic = sc.get_microphone(spk.name, include_loopback=True)
    with mic.recorder(samplerate=samplerate) as rec:
        data = rec.record(duration * samplerate)
    mono = data.mean(axis=1)
    sf.write(str(out_wav), mono, samplerate, subtype='PCM_16')
    return True

def get_video_duration_and_play(url: str) -> tuple:
    with sync_playwright() as p:
        chrome_ud = Path(os.environ.get("LOCALAPPDATA", "")) / "Google" / "Chrome" / "User Data"
        context = None
        page = None
        try:
            context = p.chromium.launch_persistent_context(
                user_data_dir=str(chrome_ud), channel="chrome", headless=False
            )
            page = context.new_page()
        except Exception:
            browser = p.chromium.launch(channel="chrome", headless=False)
            context = browser.new_context()
            page = context.new_page()
        page.goto(url, wait_until="domcontentloaded")
        dur = 0
        try:
            page.wait_for_selector("video", timeout=15000)
            page.evaluate("document.querySelector('video') && document.querySelector('video').play()")
            page.wait_for_function(
                "document.querySelector('video') && document.querySelector('video').duration && document.querySelector('video').duration > 0",
                timeout=20000,
            )
            dur = page.evaluate("document.querySelector('video').duration")
        except Exception:
            try:
                dur = page.evaluate(
                    "(() => { const s = window.__INITIAL_STATE__ || {}; const vd = s.videoData || {}; return vd.duration || 0; })()"
                )
            except Exception:
                dur = 0
        seconds = int(math.ceil(dur)) + 5 if dur and dur > 0 else 0
        return seconds, context


def record_stream_soundcard(out_wav: Path, seconds: int, samplerate: int = 16000):
    import soundcard as sc
    spk = sc.default_speaker()
    mic = sc.get_microphone(spk.name, include_loopback=True)
    frames_per_chunk = samplerate
    chunks = seconds
    with mic.recorder(samplerate=samplerate) as rec, sf.SoundFile(str(out_wav), mode='w', samplerate=samplerate, channels=1, subtype='PCM_16') as sfw:
        for i in range(chunks):
            data = rec.record(frames_per_chunk)
            mono = data.mean(axis=1)
            sfw.write(mono)
    return True


def main():
    parser = argparse.ArgumentParser(description="播放B站页面并录制系统音频，随后离线转写")
    parser.add_argument("url", help="视频页面URL")
    parser.add_argument("--out", default="outputs", help="输出目录")
    parser.add_argument("--duration", type=int, default=300, help="录制时长（秒）")
    parser.add_argument("--auto", action="store_true", help="自动获取视频时长并录制全片")
    parser.add_argument("--until-ended", action="store_true", help="持续录制直到视频播放结束")
    parser.add_argument("--max-seconds", type=int, default=None, help="录制硬上限秒数")
    parser.add_argument("--audio-device", default=None, help="指定dshow音频设备名，如 'virtual-audio-capturer' 或 'Stereo Mix ...'")
    parser.add_argument("--model", default="small", help="whisper模型：tiny/base/small/medium/large-v3")
    parser.add_argument("--lang", default="zh", help="语言代码，如zh/en")
    parser.add_argument("--engine", default=os.getenv("ASR_ENGINE", "whisper"), help="ASR引擎：whisper/funasr/vosk/groq/qwen")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    out_wav = out_dir / f"record_{ts}.wav"

    ctx = None
    page = None
    total_secs = args.duration
    if args.auto:
        total_secs, ctx = get_video_duration_and_play(args.url)
        if not total_secs:
            total_secs = args.duration
    if args.until_ended and not ctx:
        total_secs, ctx = get_video_duration_and_play(args.url)
    if args.until_ended:
        print("开始录制，直到视频结束")
    else:
        print("开始录制 ", total_secs, "秒")

    ok = False
    hard_limit = args.max_seconds if args.max_seconds else (total_secs + 120 if total_secs else None)
    if args.until_ended and ctx:
        try:
            from soundcard import get_microphone, default_speaker
            import soundcard as sc
            spk = sc.default_speaker()
            mic = get_microphone(spk.name, include_loopback=True)
            samplerate = 16000
            frames = samplerate
            with mic.recorder(samplerate=samplerate) as rec, sf.SoundFile(str(out_wav), mode='w', samplerate=samplerate, channels=1, subtype='PCM_16') as sfw:
                elapsed = 0
                silent_ticks = 0
                while True:
                    data = rec.record(frames)
                    mono = data.mean(axis=1)
                    sfw.write(mono)
                    elapsed += 1
                    try:
                        stat = ctx.pages[0].evaluate("(() => { const v=document.querySelector('video'); if(!v) return {ct:0,dur:0,paused:false,ended:false}; return {ct:v.currentTime||0,dur:v.duration||0,paused:!!v.paused,ended:!!v.ended}; })()")
                    except Exception:
                        stat = {"ct":0,"dur":0,"paused":False,"ended":False}
                    energy = float(np.sqrt(np.mean(mono.astype(np.float32)**2))) if mono.size else 0.0
                    if energy < 1e-3:
                        silent_ticks += 1
                    else:
                        silent_ticks = 0
                    # write status
                    try:
                        status = {
                            "ts": int(time.time()),
                            "phase": "recording",
                            "record_secs": elapsed,
                            "video_ct": stat.get("ct", 0),
                            "video_dur": stat.get("dur", 0),
                            "rms": energy,
                            "silent_secs": silent_ticks,
                            "eta_secs": max(0, int(stat.get("dur", 0) - stat.get("ct", 0))) if stat.get("dur", 0) > 0 else None,
                        }
                        with open(str(Path(args.out) / "status.json"), "w", encoding="utf-8") as f:
                            json.dump(status, f, ensure_ascii=False)
                    except Exception:
                        pass
                    if stat.get("ended"):
                        break
                    if stat.get("dur",0) > 0 and stat.get("ct",0) >= max(0, stat.get("dur",0) - 1):
                        break
                    if hard_limit and elapsed >= hard_limit:
                        break
                    if stat.get("dur",0) > 0 and stat.get("ct",0) > stat.get("dur",0)/2 and silent_ticks >= 20:
                        break
            ok = True
        except Exception:
            ok = False
    else:
        ok = False
        # 优先使用 soundcard 环路回录，随后再尝试 ffmpeg 和 sounddevice
        for rec in (
            lambda: record_stream_soundcard(out_wav, total_secs),
            lambda: try_record_ffmpeg(out_wav, total_secs, args.audio_device),
            lambda: try_record_sounddevice(out_wav, total_secs),
        ):
            try:
                ok = rec()
            except Exception:
                ok = False
            # 录完后检测静音，静音则认为失败并继续尝试下一种录制方式
            if ok:
                rms = _file_rms(out_wav)
                if rms < 1e-3:
                    try:
                        out_wav.unlink(missing_ok=True)
                    except Exception:
                        pass
                    ok = False
                else:
                    break
        if not ok:
            print("录制错误: 无法获取有效的系统音频，请检查扬声器/回录设备")
            if ctx:
                try:
                    ctx.close()
                except Exception:
                    pass
            sys.exit(1)

    print("录制完成：", str(out_wav))
    # 调用已有转写脚本
    cmd = [
        sys.executable, "transcribe_bilibili.py", str(out_wav),
        "--out", args.out, "--engine", args.engine, "--model", args.model, "--lang", args.lang,
        "--status-file", str(Path(args.out) / "status.json"),
    ]
    try:
        subprocess.run(cmd, check=True)
    except Exception as e:
        print("转写错误:", e)
        sys.exit(1)
    if ctx:
        try:
            ctx.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
def _file_rms(wav_path: Path) -> float:
    try:
        data, sr = sf.read(str(wav_path))
        if data is None or getattr(data, 'size', 0) == 0:
            return 0.0
        import numpy as _np
        mono = data if data.ndim == 1 else _np.mean(data, axis=1)
        return float(_np.sqrt(_np.mean(mono.astype('float32') ** 2)))
    except Exception:
        return 0.0
