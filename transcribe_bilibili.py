import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

from yt_dlp import YoutubeDL


def ensure_ffmpeg():
    return shutil.which("ffmpeg") is not None


def sanitize_filename(name):
    return re.sub(r"[\\/:*?\"<>|]", "_", name)


def download_audio(url, out_dir, cookies_from_browser=None, cookies_file=None, proxy=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if os.path.exists(url) and os.path.isfile(url):
        return url
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / "%(title)s" / "%(title)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
    }
    if cookies_from_browser:
        browser = cookies_from_browser.strip().lower()
        if browser == "chrome":
            ydl_opts["cookiesfrombrowser"] = ("chrome", "Default")
        elif browser == "edge":
            ydl_opts["cookiesfrombrowser"] = ("edge", "Default")
        elif browser == "firefox":
            ydl_opts["cookiesfrombrowser"] = ("firefox", "default-release")
        else:
            raise RuntimeError("不支持的浏览器，使用 chrome/edge/firefox")
    if proxy:
        ydl_opts["proxy"] = proxy
    ydl_opts["http_headers"] = {
        "Referer": "https://www.bilibili.com/",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
    }
    if cookies_file:
        ydl_opts["cookiefile"] = cookies_file
    if ensure_ffmpeg():
        ydl_opts["postprocessors"] = [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ]
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
    
    # Determine where the file ended up
    # prepare_filename gives the path based on info and outtmpl
    downloaded_path_str = ydl.prepare_filename(info)
    downloaded_path = Path(downloaded_path_str)
    
    # The directory where the file should be
    search_dir = downloaded_path.parent
    title_stem = downloaded_path.stem
    
    audio_path = None

    if ensure_ffmpeg():
        # Look for wav in the specific directory
        # Note: stem might vary if sanitization differs, but search_dir is correct
        if search_dir.exists():
            audio_path = next(search_dir.glob(f"{title_stem}.wav"), None)
            if audio_path is None:
                # Fallback: find any wav in that subdirectory
                audio_path = next(search_dir.glob("*.wav"), None)
    else:
        # No ffmpeg
        if downloaded_path.exists():
            audio_path = downloaded_path
        elif search_dir.exists():
            # Try to find any audio file in that subdir
            for ext in ("m4a", "webm", "mp3", "aac", "ogg"):
                candidate = next(search_dir.glob(f"*.{ext}"), None)
                if candidate is not None:
                    audio_path = candidate
                    break
    
    # Last resort fallback: search recursively in out_dir
    if audio_path is None or not audio_path.exists():
        if ensure_ffmpeg():
            audio_path = next(out_dir.glob("**/*.wav"), None)
        else:
            audio_path = next(out_dir.glob("**/*.*"), None)

    if audio_path is None:
        raise RuntimeError("音频下载失败")
    return str(audio_path)


def format_timestamp(t):
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int((t - int(t)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_outputs(base_path, segments):
    base = Path(base_path)
    srt_path = base.with_suffix(".srt")
    txt_path = base.with_suffix(".txt")
    json_path = base.with_suffix(".json")
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            f.write(str(i) + "\n")
            f.write(format_timestamp(seg["start"]) + " --> " + format_timestamp(seg["end"]) + "\n")
            f.write(seg["text"].strip() + "\n\n")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(s["text"].strip() for s in segments))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"segments": segments}, f, ensure_ascii=False, indent=2)
    return str(srt_path), str(txt_path), str(json_path)


def _update_status(status_file, phase=None, segments_count=None, eta_secs=None, error=None):
    if not status_file:
        return
    try:
        import json, time
        payload = {"ts": int(time.time())}
        if phase is not None:
            payload["phase"] = phase
        if segments_count is not None:
            payload["segments"] = segments_count
        if eta_secs is not None:
            payload["eta_secs"] = eta_secs
        if error is not None:
            payload["error"] = str(error)
        with open(status_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
    except Exception:
        pass


def transcribe_whisper(audio_path, model_size, language, device, compute_type, status_file=None):
    from faster_whisper import WhisperModel
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    gen, _ = model.transcribe(audio_path, language=language, vad_filter=True)
    out = []
    _update_status(status_file, phase="transcribing", segments_count=0)
    for seg in gen:
        out.append({"start": seg.start or 0.0, "end": seg.end or 0.0, "text": seg.text})
        _update_status(status_file, phase="transcribing", segments_count=len(out))
    _update_status(status_file, phase="transcribed", segments_count=len(out))
    return out


def transcribe_vosk(audio_path, model_path):
    import subprocess
    import wave
    from vosk import Model, KaldiRecognizer
    if not os.path.isdir(model_path):
        raise RuntimeError("Vosk模型路径不存在")
    if not ensure_ffmpeg():
        raise RuntimeError("Vosk需要ffmpeg用于重采样，请安装后重试")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.close()
    subprocess.run([
        "ffmpeg",
        "-y",
        "-i",
        audio_path,
        "-ac",
        "1",
        "-ar",
        "16000",
        tmp.name,
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    wf = wave.open(tmp.name, "rb")
    model = Model(model_path)
    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)
    segments = []
    start = 0.0
    chunk_dur = 5.0
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            res = json.loads(rec.Result())
            text = res.get("text", "").strip()
            if text:
                end = start + chunk_dur
                segments.append({"start": start, "end": end, "text": text})
                start = end
        else:
            _ = rec.PartialResult()
    final = json.loads(rec.FinalResult())
    text = final.get("text", "").strip()
    if text:
        segments.append({"start": start, "end": start + chunk_dur, "text": text})
    wf.close()
    os.unlink(tmp.name)
    return segments


def main():
    parser = argparse.ArgumentParser(description="B站URL离线音频转写")
    parser.add_argument("url", help="B站视频页面URL")
    parser.add_argument("--out", default="downloads", help="下载输出目录")
    parser.add_argument("--engine", choices=["whisper", "vosk"], default="whisper")
    parser.add_argument("--model", default="small", help="whisper模型：tiny/base/small/medium/large-v2/large-v3")
    parser.add_argument("--lang", default=None, help="语言代码，如zh/en，留空自动检测")
    parser.add_argument("--device", default="cpu", help="设备：cpu/cuda")
    parser.add_argument("--compute-type", default="int8", help="faster-whisper计算类型：int8/float32/float16")
    parser.add_argument("--vosk-model-path", default="models/vosk/cn", help="Vosk模型路径")
    parser.add_argument("--status-file", default=None, help="状态文件路径，用于记录转写进度")
    parser.add_argument("--cookies-from-browser", default=None, help="从浏览器读取登录Cookie：chrome/edge/firefox")
    parser.add_argument("--proxy", default=None, help="HTTP/HTTPS代理，如 http://127.0.0.1:7890")
    parser.add_argument("--cookies-file", default=None, help="Cookie文本文件路径（Netscape格式）")
    args = parser.parse_args()

    audio_path = download_audio(
        args.url,
        args.out,
        cookies_from_browser=args.cookies_from_browser,
        cookies_file=args.cookies_file,
        proxy=args.proxy,
    )
    base = os.path.splitext(audio_path)[0]

    # Check if outputs exist to skip re-transcription
    final_txt = base + ".txt"
    if os.path.exists(final_txt) and os.path.getsize(final_txt) > 0:
        print("检测到已有转写结果，跳过转写步骤。")
        print("生成文件:")
        print(base + ".srt")
        print(final_txt)
        print(base + ".json")
        return

    if args.engine == "whisper":
        segments = transcribe_whisper(audio_path, args.model, args.lang, args.device, args.compute_type, status_file=args.status_file)
    else:
        segments = transcribe_vosk(audio_path, args.vosk_model_path)

    srt_path, txt_path, json_path = write_outputs(base, segments)
    _update_status(args.status_file, phase="done", segments_count=len(segments))
    print("生成文件:")
    print(srt_path)
    print(txt_path)
    print(json_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("错误:", e)
        sys.exit(1)
def build_cookie_header_from_netscape(path):
    jar = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) != 7:
                continue
            domain, tailmatch, cpath, secure, expiry, name, value = parts
            if "bilibili.com" in domain:
                jar[name] = value
    if not jar:
        return None
    return "; ".join([f"{k}={v}" for k, v in jar.items()])
