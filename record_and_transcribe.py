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
import re


def sanitize_filename(name: str) -> str:
    name = re.sub(r'[\\/*?:"<>|]', '_', name)
    name = name.strip()
    if not name:
        return f"video_{int(time.time())}"
    return name


def get_ffmpeg_path():
    p = shutil.which("ffmpeg")
    if p:
        return p
    alt = Path(os.environ.get("LOCALAPPDATA", "")) / "ms-playwright" / "ffmpeg-1010" / "ffmpeg-win64.exe"
    if alt.exists():
        return str(alt)
    return None


def get_dshow_audio_devices(ffmpeg_path: str) -> list:
    """Run ffmpeg to list dshow audio devices and return a list of device names."""
    try:
        cmd = [ffmpeg_path, "-list_devices", "true", "-f", "dshow", "-i", "dummy"]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        output = result.stderr

        devices = []
        in_audio_section = False
        for line in output.splitlines():
            if "DirectShow audio devices" in line:
                in_audio_section = True
                continue
            if "DirectShow video devices" in line:
                in_audio_section = False
                continue
            if in_audio_section:
                match = re.search(r'\[dshow @ .*?\]\s+"([^"]+)"', line)
                if match:
                    devices.append(match.group(1))
        return devices
    except Exception as e:
        print(f"Error listing dshow devices: {e}")
        return []


def list_audio_devices():
    """List possible system audio loopback devices for reference."""
    ff = get_ffmpeg_path()
    if ff:
        dshow = get_dshow_audio_devices(ff)
        if dshow:
            print(f"[ffmpeg dshow] 可见音频设备: {dshow}")
    # PyAudioWPatch loopbacks
    try:
        import pyaudiowpatch as pyaudio
        p = pyaudio.PyAudio()
        loopbacks = []
        if hasattr(p, "get_loopback_device_info_generator"):
            loopbacks = list(p.get_loopback_device_info_generator())
        if loopbacks:
            names = [d["name"] for d in loopbacks]
            print(f"[PyAudioWPatch] Loopback 设备: {names}")
        p.terminate()
    except Exception:
        pass
    # sounddevice enumeration (WASAPI)
    try:
        import sounddevice as sd
        devs = sd.query_devices()
        wasapi_index = None
        for i, ha in enumerate(sd.query_hostapis()):
            if "WASAPI" in ha.get("name", ""):
                wasapi_index = i
                break
        if wasapi_index is not None:
            names = [d.get("name") for d in devs if d.get("hostapi") == wasapi_index]
            if names:
                print(f"[sounddevice WASAPI] 设备: {names}")
    except Exception:
        pass


def summarize_rms(wav_path: Path):
    """Compute RMS/peak in dB for quick quality feedback."""
    try:
        data, sr = sf.read(str(wav_path))
        if data.ndim > 1:
            data = data.mean(axis=1)
        rms = float(np.sqrt(np.mean(np.square(data))))
        peak = float(np.max(np.abs(data))) if data.size else 0.0
        rms_db = 20 * np.log10(rms) if rms > 0 else -96.0
        peak_db = 20 * np.log10(peak) if peak > 0 else -96.0
        return {"rms": rms, "rms_db": rms_db, "peak_db": peak_db, "sr": sr}
    except Exception as e:
        print(f"RMS 计算失败: {e}")
        return None


def precheck_device(ffmpeg_path: str, device: str, seconds: int, rms_threshold: float = 45.0):
    """Record a very short clip to gauge if the device has non-silent audio."""
    if not device or seconds <= 0:
        return
    tmp = Path(os.getenv("TEMP", ".")) / f"precheck_{int(time.time())}.wav"
    cmd = [
        ffmpeg_path, "-hide_banner", "-y",
        "-f", "dshow", "-i", f"audio={device}",
        "-t", str(seconds),
        "-ac", "1", "-ar", "16000",
        str(tmp)
    ]
    try:
        print(f"录前预检设备 {device}，时长 {seconds}s ...", flush=True)
        subprocess.run(cmd, check=True, capture_output=True)
        rpt = summarize_rms(tmp)
        if rpt:
            print(f"预检 RMS={rpt['rms_db']:.1f} dB, Peak={rpt['peak_db']:.1f} dB")
            if rpt["rms_db"] < rms_threshold:
                print(f"⚠️ 预检音量低于阈值 {rms_threshold} dB，可能无声或未播放，请确认系统输出设备/音量。")
    except Exception as e:
        print(f"预检失败（忽略继续）: {e}")
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def try_record_ffmpeg(out_wav: Path, duration: int, audio_device: str = None):
    ff = get_ffmpeg_path()
    if not ff:
        raise RuntimeError("未检测到ffmpeg，请安装或确保Playwright的ffmpeg可用")
    
    cmds = []
    
    # 1. Dynamic discovery of dshow devices
    print("正在探测可用音频设备...", flush=True)
    discovered_devices = get_dshow_audio_devices(ff)
    print(f"发现设备: {discovered_devices}", flush=True)

    # Priority list of keywords to look for in discovered devices
    priority_keywords = ["virtual-audio-capturer", "Stereo Mix", "立体声混音", "What U Hear"]
    
    # If user specified a device, try it first
    if audio_device:
         cmds.append([
            ff, "-hide_banner", "-y",
            "-f", "dshow", "-i", f"audio={audio_device}",
            "-t", str(duration),
            "-ac", "1", "-ar", "16000", "-vn",
            str(out_wav),
        ])

    # Try discovered devices that match keywords
    for keyword in priority_keywords:
        for dev in discovered_devices:
            if keyword.lower() in dev.lower():
                cmds.append([
                    ff, "-hide_banner", "-y",
                    "-f", "dshow", "-i", f"audio={dev}",
                    "-t", str(duration),
                    "-ac", "1", "-ar", "16000", "-vn",
                    str(out_wav),
                ])
    
    # Fallback 1: Try all other discovered devices (maybe dangerous if it picks a mic, but better than nothing?)
    # Let's skip this for now to avoid recording random mics unless we are desperate.
    # User feedback says "gracefully degrade to mic if user agrees". 
    # For now, let's stick to system audio attempts.
    
    # Fallback 2: Hardcoded "virtual-audio-capturer" if not found in list (sometimes list fails or hidden)
    cmds.append([
        ff, "-hide_banner", "-y",
        "-f", "dshow", "-i", "audio=virtual-audio-capturer",
        "-t", str(duration),
        "-ac", "1", "-ar", "16000", "-vn",
        str(out_wav),
    ])
    
    # Fallback 3: Hardcoded "Stereo Mix"
    cmds.append([
        ff, "-hide_banner", "-y",
        "-f", "dshow", "-i", "audio=Stereo Mix (Realtek(R) Audio)",
        "-t", str(duration),
        "-ac", "1", "-ar", "16000", "-vn",
        str(out_wav),
    ])
    
    # Fallback 4: WASAPI loopback (ffmpeg feature)
    cmds.append([
        ff, "-hide_banner", "-y",
        "-f", "wasapi", "-i", "default",
        "-t", str(duration),
        "-ac", "1", "-ar", "16000", "-vn",
        str(out_wav),
    ])

    last_error = None
    for i, cmd in enumerate(cmds):
        try:
            device_name = "unknown"
            for arg in cmd:
                if arg.startswith("audio="):
                    device_name = arg.split("=", 1)[1]
                elif arg == "default" and "wasapi" in cmd:
                    device_name = "WASAPI Default"
            
            print(f"尝试录制方案 {i+1}: {device_name}", flush=True)
            subprocess.run(cmd, check=True)
            report = summarize_rms(out_wav)
            if report:
                print(f"录制完成: RMS={report['rms_db']:.1f} dB, Peak={report['peak_db']:.1f} dB, SR={report['sr']}")
                # Check for silence (e.g. < -60dB is effectively silent)
                if report['rms_db'] < -60.0:
                    print(f"⚠️ 警告: 录制音量过低 ({report['rms_db']:.1f} dB)，判定为静音失败。", flush=True)
                    # Try next device/method
                    continue 
            return True
        except Exception as e:
            print(f"方案 {i+1} 失败: {e}", flush=True)
            last_error = e
            
    if last_error:
        raise last_error
    return False


def try_record_pyaudiowpatch(out_wav: Path, duration: int, samplerate: int = 16000):
    """
    Attempt to record system audio using PyAudioWPatch (WASAPI Loopback).
    Records at device native rate/channels to avoid artifacts.
    Includes silence detection and real-time RMS logging.
    """
    try:
        import pyaudiowpatch as pyaudio
        import wave
        import numpy as np
    except ImportError:
        print("PyAudioWPatch or numpy not installed. Skipping.")
        return False

    p = pyaudio.PyAudio()
    try:
        print("正在尝试使用 PyAudioWPatch (WASAPI Loopback) 录制...", flush=True)
        
        # Method 1: Try to get default WASAPI loopback device directly (Recommended)
        loopback_device = None
        
        # Helper to check if device is likely a monitor/HDMI (often silent)
        def is_monitor_audio(name):
            keywords = ["display", "monitor", "hdmi", "nvidia", "amd", "phl ", "dell", "aoc", "samsung", "lg"]
            name_lower = name.lower()
            return any(k in name_lower for k in keywords) and "speaker" not in name_lower

        try:
            # First, list all available loopback devices to make a smart choice
            all_loopbacks = []
            if hasattr(p, 'get_loopback_device_info_generator'):
                for info in p.get_loopback_device_info_generator():
                    all_loopbacks.append(info)
            
            if all_loopbacks:
                print(f"PyAudioWPatch: 发现 {len(all_loopbacks)} 个 Loopback 设备:", flush=True)
                for i, d in enumerate(all_loopbacks):
                    print(f"  [{i}] {d['name']}", flush=True)
                
                # Filter out monitors if possible
                candidates = [d for d in all_loopbacks if not is_monitor_audio(d['name'])]
                
                if candidates:
                    # Pick the first non-monitor device (usually Speakers or Headphones)
                    loopback_device = candidates[0]
                    print(f"PyAudioWPatch: 智能优选设备 -> {loopback_device['name']}", flush=True)
                else:
                    # Fallback to default if all look like monitors
                    print("PyAudioWPatch: 未发现常用音频设备，将尝试使用默认设备。", flush=True)

            # If smart selection didn't work, fallback to default method
            if not loopback_device and hasattr(p, 'get_default_wasapi_loopback'):
                loopback_device = p.get_default_wasapi_loopback()
                if is_monitor_audio(loopback_device['name']):
                    print(f"⚠️ 警告: 默认设备 '{loopback_device['name']}' 可能是显示器音频，可能导致录制无声！建议手动切换系统播放设备到扬声器/耳机。", flush=True)
                print(f"PyAudioWPatch: 获取默认 Loopback 设备: {loopback_device['name']}", flush=True)

        except Exception as e:
            print(f"PyAudioWPatch: 设备发现失败: {e}", flush=True)

        # Method 2: Manual discovery if Method 1 fails
        if not loopback_device:
            try:
                wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
                default_speakers = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
                print(f"默认输出设备: {default_speakers['name']}", flush=True)

                if not default_speakers["isLoopbackDevice"]:
                    for loopback in p.get_loopback_device_info_generator():
                        if default_speakers["name"] in loopback["name"]:
                            loopback_device = loopback
                            break
                    if not loopback_device:
                         print("PyAudioWPatch: 未找到完全匹配的 Loopback，尝试查找任意 Loopback...", flush=True)
                         for loopback in p.get_loopback_device_info_generator():
                             loopback_device = loopback
                             break
                else:
                    loopback_device = default_speakers
            except Exception as e:
                 print(f"PyAudioWPatch: 手动查找 Loopback 设备失败: {e}", flush=True)

        if not loopback_device:
            print("PyAudioWPatch: 未找到可用的 Loopback 设备")
            return False
            
        print(f"PyAudioWPatch: 最终选中设备 '{loopback_device['name']}'", flush=True)

        dev_channels = int(loopback_device["maxInputChannels"])
        dev_rate = int(loopback_device["defaultSampleRate"])
        
        print(f"录制参数: {dev_rate}Hz, {dev_channels}ch", flush=True)

        with p.open(format=pyaudio.paInt16,
                    channels=dev_channels,
                    rate=dev_rate,
                    input=True,
                    input_device_index=loopback_device["index"],
                    frames_per_buffer=dev_rate) as stream:
            
            with wave.open(str(out_wav), 'wb') as wf:
                wf.setnchannels(dev_channels)
                wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
                wf.setframerate(dev_rate)
                
                print(f"正在录制 {duration} 秒...", flush=True)
                
                # Chunk size = 1 second of audio
                chunk_size = dev_rate
                
                # Silence detection counters
                low_volume_count = 0
                total_chunks = 0
                
                for i in range(duration):
                    try:
                        data = stream.read(chunk_size)
                        wf.writeframes(data)
                        
                        # Real-time RMS logging
                        audio_data = np.frombuffer(data, dtype=np.int16)
                        if len(audio_data) > 0:
                            rms = np.sqrt(np.mean(audio_data.astype(np.float64)**2))
                            db = 20 * np.log10(rms) if rms > 0 else -96
                            
                            # Visual feedback
                            bar_len = int(min(db + 60, 40) / 2) # normalize roughly -60dB to -20dB
                            if bar_len < 0: bar_len = 0
                            bar = "|" * bar_len
                            
                            print(f"\r录制中 {i+1}/{duration}s [RMS:{rms:.1f} | dB:{db:.1f}] {bar}", end="", flush=True)
                            
                            if rms < 50:
                                low_volume_count += 1
                        total_chunks += 1
                        
                    except Exception as e:
                        print(f"\n录制中断: {e}")
                        break
                
                print("\n录制完成。")
                if total_chunks > 0 and (low_volume_count / total_chunks) > 0.8:
                     print("⚠️ 警告: 录制过程中大部分时间音量过低 (Silence Detected)！", flush=True)
                     return False

        return True

    except Exception as e:
        print(f"PyAudioWPatch 录制异常: {e}")
        return False
    finally:
        p.terminate()


def try_record_sounddevice(out_wav: Path, duration: int, samplerate: int = 16000):
    import sounddevice as sd
    # Prefer WASAPI loopback
    extra = None
    try:
        extra = sd.WasapiSettings(exclusive=False, loopback=True)
    except Exception as e:
        print(f"sounddevice: WASAPI loopback设置失败 ({e})，将尝试默认设置", flush=True)
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
        
        if wasapi_index is None:
             print("sounddevice: 未找到WASAPI Host API", flush=True)
             
        devices = sd.query_devices()
        # Find first WASAPI output device
        for i, d in enumerate(devices):
            if wasapi_index is not None and d.get("hostapi") == wasapi_index and d.get("max_output_channels", 0) > 0:
                dev_index = i
                print(f"sounddevice: 选中WASAPI设备 {d.get('name')} (Index: {i})", flush=True)
                break
    except Exception as e:
        print(f"sounddevice: 设备查询失败 ({e})", flush=True)
        dev_index = None
    
    print(f"正在使用 sounddevice 录制 {duration} 秒音频...", flush=True)
    frames = int(duration * samplerate)
    
    try:
        with sf.SoundFile(str(out_wav), mode='w', samplerate=samplerate, channels=1, subtype='PCM_16') as file:
            with sd.InputStream(samplerate=samplerate, device=dev_index, channels=1, dtype='float32', extra_settings=extra) as stream:
                total_frames = 0
                block_size = samplerate  # 1 second
                while total_frames < frames:
                    read_frames = min(block_size, frames - total_frames)
                    data, overflowed = stream.read(read_frames)
                    file.write(data.squeeze())
                    total_frames += read_frames
                    if (total_frames / samplerate) % 10 == 0:
                        print(f"已录制 {int(total_frames / samplerate)}/{duration} 秒...", flush=True)
        
        # Post-check silence
        rpt = summarize_rms(out_wav)
        if rpt and rpt['rms_db'] < -60.0:
            print(f"⚠️ sounddevice 录制音量过低 ({rpt['rms_db']:.1f} dB)，判定失败。", flush=True)
            return False
            
        return True
    except Exception as e:
        print(f"sounddevice 录制过程发生异常: {e}", flush=True)
        raise e

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
        
        # Args to ensure audio autoplay and stability
        launch_args = [
            "--autoplay-policy=no-user-gesture-required",
            "--disable-features=AudioServiceOutOfProcess",
        ]

        try:
            context = p.chromium.launch_persistent_context(
                user_data_dir=str(chrome_ud), 
                channel="chrome", 
                headless=False,
                args=launch_args
            )
            page = context.new_page()
        except Exception:
            browser = p.chromium.launch(
                channel="chrome", 
                headless=False,
                args=launch_args
            )
            context = browser.new_context()
            page = context.new_page()
            
        page.goto(url, wait_until="domcontentloaded")
        try:
            page.bring_to_front()
        except:
            pass
        
        # Extract title
        title = ""
        try:
            # Wait a bit for title to be available
            page.wait_for_selector("h1", timeout=3000)
        except Exception:
            pass

        if not title or "出错啦" in title:
            try:
                title = page.evaluate("(() => { const s = window.__INITIAL_STATE__ || {}; const vd = s.videoData || {}; return vd.title || ''; })()")
            except Exception:
                pass
        if not title or "出错啦" in title:
            try:
                title = page.title()
                title = title.replace("_哔哩哔哩_bilibili", "")
            except Exception:
                pass

        if not title or "出错啦" in title:
             # Try one more specific selector for title if window state failed
             try:
                 title = page.evaluate("document.querySelector('h1.video-title') ? document.querySelector('h1.video-title').innerText : ''")
             except Exception:
                 pass
        
        if not title or "出错啦" in title:
            # Fallback to yt-dlp for title
            try:
                # Use subprocess to call yt-dlp
                # We assume yt-dlp is installed or we can use the library if we import it
                # Since we want to avoid extra imports if possible, let's use subprocess if we can find it,
                # or just import yt_dlp if available.
                # Actually transcribe_bilibili.py uses yt_dlp library.
                from yt_dlp import YoutubeDL
                with YoutubeDL({"quiet": True, "noplaylist": True}) as ydl:
                    info = ydl.extract_info(url, download=False)
                    title = info.get("title", "")
            except Exception:
                pass

        dur = 0
        try:
            page.wait_for_selector("video", timeout=15000)
            # Ensure video is unmuted and volume is 100%
            page.evaluate("(() => { const v = document.querySelector('video'); if (v) { v.muted = false; v.volume = 1.0; v.play(); } })()")
            
            # Wait for playback to actually start (currentTime > 0)
            page.wait_for_function(
                "document.querySelector('video') && document.querySelector('video').currentTime > 0.1",
                timeout=10000
            )
            
            page.wait_for_function(
                "document.querySelector('video') && document.querySelector('video').duration && document.querySelector('video').duration > 0",
                timeout=20000,
            )
            dur = page.evaluate("document.querySelector('video').duration")
        except Exception as e:
            print(f"Warning: Error during video playback initialization: {e}")
            try:
                dur = page.evaluate(
                    "(() => { const s = window.__INITIAL_STATE__ || {}; const vd = s.videoData || {}; return vd.duration || 0; })()"
                )
            except Exception:
                dur = 0
        seconds = int(math.ceil(dur)) + 5 if dur and dur > 0 else 0
        return seconds, context, title


def record_stream_soundcard(out_wav: Path, seconds: int, samplerate: int = 16000):
    import soundcard as sc
    spk = sc.default_speaker()
    mic = sc.get_microphone(spk.name, include_loopback=True)
    frames_per_chunk = samplerate
    chunks = seconds
    print(f"正在使用 soundcard 录制 {seconds} 秒音频...", flush=True)
    with mic.recorder(samplerate=samplerate) as rec, sf.SoundFile(str(out_wav), mode='w', samplerate=samplerate, channels=1, subtype='PCM_16') as sfw:
        for i in range(chunks):
            data = rec.record(frames_per_chunk)
            mono = data.mean(axis=1)
            sfw.write(mono)
            if (i + 1) % 10 == 0:
                print(f"已录制 {i + 1}/{seconds} 秒...", flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description="播放B站页面并录制系统音频，随后离线转写")
    parser.add_argument("url", help="视频页面URL")
    parser.add_argument("--out", default="outputs", help="输出目录")
    parser.add_argument("--duration", type=int, default=300, help="录制时长（秒）")
    parser.add_argument("--auto", action="store_true", help="自动获取视频时长并录制全片")
    parser.add_argument("--until-ended", action="store_true", help="持续录制直到视频播放结束")
    parser.add_argument("--max-seconds", type=int, default=None, help="录制硬上限秒数")
    parser.add_argument("--audio-device", default=None, help="指定dshow音频设备名，如 'virtual-audio-capturer' 或 'Stereo Mix ...' (兼容旧参数)")
    parser.add_argument("--device-mode", choices=["auto", "list", "name"], default="auto", help="录音设备选择模式：auto自动优选，list仅列出设备，name使用 --device-name")
    parser.add_argument("--device-name", default=None, help="与 --device-mode name 搭配，指定设备名")
    parser.add_argument("--precheck-seconds", type=int, default=0, help="录前预检秒数（0 关闭）")
    parser.add_argument("--rms-threshold", type=float, default=45.0, help="录后RMS告警阈值(dB)，仅提示不影响流程")
    parser.add_argument("--model", default="small", help="whisper模型：tiny/base/small/medium/large-v3")
    parser.add_argument("--lang", default="zh", help="语言代码，如zh/en")
    parser.add_argument("--engine", default=os.getenv("ASR_ENGINE", "whisper"), help="ASR引擎：whisper/funasr/vosk/groq/qwen")
    args = parser.parse_args()

    # 仅列设备后退出
    if args.device_mode == "list":
        list_audio_devices()
        return

    selected_device = args.device_name or args.audio_device
    if args.device_mode == "name" and not selected_device:
        print("错误：device-mode 为 name 但未提供 --device-name。")
        sys.exit(1)
    if args.precheck_seconds > 0 and selected_device:
        ff = get_ffmpeg_path()
        if ff:
            precheck_device(ff, selected_device, args.precheck_seconds, args.rms_threshold)
        else:
            print("预检跳过：未找到 ffmpeg")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())

    ctx = None
    page = None
    total_secs = args.duration
    video_title = ""

    if args.auto:
        total_secs, ctx, video_title = get_video_duration_and_play(args.url)
        if not total_secs:
            total_secs = args.duration
    if args.until_ended and not ctx:
        total_secs, ctx, video_title = get_video_duration_and_play(args.url)

    # Apply max-seconds limit if specified (affects fixed duration recording)
    if args.max_seconds and total_secs > args.max_seconds:
        print(f"根据 max-seconds 限制，调整录制时长: {total_secs} -> {args.max_seconds} 秒")
        total_secs = args.max_seconds

    # Create title-based subfolder
    sanitized_title = sanitize_filename(video_title) if video_title else f"record_{ts}"
    out_subdir = out_dir / sanitized_title
    out_subdir.mkdir(parents=True, exist_ok=True)
    out_wav = out_subdir / f"record_{ts}.wav"
    status_file_path = out_subdir / "status.json"

    if args.until_ended:
        print(f"开始录制，直到视频结束。输出目录: {out_subdir}")
    else:
        print(f"开始录制 {total_secs} 秒。输出目录: {out_subdir}")
    
    if not video_title:
        print("Warning: 未能自动获取视频标题，使用时间戳命名。")

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
            print(f"正在使用 soundcard (until-ended) 录制中...", flush=True)
            with mic.recorder(samplerate=samplerate) as rec, sf.SoundFile(str(out_wav), mode='w', samplerate=samplerate, channels=1, subtype='PCM_16') as sfw:
                elapsed = 0
                silent_ticks = 0
                while True:
                    data = rec.record(frames)
                    mono = data.mean(axis=1)
                    sfw.write(mono)
                    elapsed += 1
                    if elapsed % 10 == 0:
                        print(f"已录制 {elapsed} 秒...", flush=True)
                    
                    # Check silence (simple RMS)
                    rms = np.sqrt(np.mean(mono**2))
                    if rms < 0.001:
                        silent_ticks += 1
                    else:
                        silent_ticks = 0
                    
                    if hard_limit and elapsed >= hard_limit:
                        print("达到最大录制时长限制，停止录制。")
                        break
                    
                    # Stop if silent for too long (e.g. 30s) indicating video ended
                    if silent_ticks > 30:
                        print("检测到长时间静音，停止录制。")
                        break
            ok = True
        except Exception as e:
            print(f"soundcard until-ended 录制失败: {e}")
            ok = False

    elif not args.until_ended:
        # 策略顺序：PyAudioWPatch (WASAPI Loopback) -> FFmpeg (dshow/wasapi) -> sounddevice -> soundcard
        
        # 1. PyAudioWPatch (Preferred for Windows Loopback)
        if not ok:
            ok = try_record_pyaudiowpatch(out_wav, total_secs)
            
        # 2. FFmpeg (dshow / stereo mix)
        if not ok:
            try:
                ok = try_record_ffmpeg(out_wav, total_secs, selected_device)
            except Exception as e:
                print(f"FFmpeg录制失败: {e}")
                ok = False

        # 3. sounddevice (WASAPI Loopback via PortAudio)
        if not ok:
            try:
                ok = try_record_sounddevice(out_wav, total_secs)
            except Exception:
                ok = False
                
        # 4. soundcard (Last resort)
        if not ok:
             try:
                 ok = record_stream_soundcard(out_wav, total_secs)
             except Exception:
                 ok = False

    if ctx:
        try:
            ctx.close()
        except:
            pass
            
    if not ok:
        print("所有录制方案均失败。")
        sys.exit(1)

    # 录后质量提示
    report = summarize_rms(out_wav)
    if report:
        print(f"录制完成: RMS={report['rms_db']:.1f} dB, Peak={report['peak_db']:.1f} dB, SR={report['sr']}")
        if report["rms_db"] < args.rms_threshold:
            print(f"⚠️ 音量低于阈值 {args.rms_threshold} dB，若转写效果差，请检查系统输出设备/音量。")

    # Post-processing: Normalize Audio (Optional but recommended)
    try:
        print("正在进行音频标准化处理...", flush=True)
        ff = get_ffmpeg_path()
        if ff and out_wav.exists():
            temp_wav = out_wav.with_suffix(".temp.wav")
            # Use ffmpeg-normalize or simple loudnorm filter
            # cmd: ffmpeg -i input.wav -af loudnorm=I=-16:TP=-1.5:LRA=11 -ar 16000 -y output.wav
            cmd = [
                ff, "-hide_banner", "-y",
                "-i", str(out_wav),
                "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
                "-ar", "16000",
                str(temp_wav)
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            shutil.move(str(temp_wav), str(out_wav))
            print("音频标准化完成。", flush=True)
    except Exception as e:
        print(f"音频标准化失败 (非致命): {e}")

    # Transcribe
    print(f"开始转写: {out_wav}")
    # 调用已有转写脚本
    cmd = [
        sys.executable, "transcribe_bilibili.py", str(out_wav),
        "--out", str(out_subdir), "--engine", args.engine, "--model", args.model, "--lang", args.lang,
        "--status-file", str(status_file_path),
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
