import argparse
import subprocess
import os
import sys
from pathlib import Path


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
            jar[name] = value
    if not jar:
        return None
    return "; ".join([f"{k}={v}" for k, v in jar.items()])


def main():
    parser = argparse.ArgumentParser(description="使用m3u8直链下载音频并离线转写")
    parser.add_argument("m3u8", help="m3u8直链")
    parser.add_argument("--out", default="outputs", help="输出目录")
    parser.add_argument("--cookies-file", default=None, help="Cookie文本文件（Netscape格式）")
    parser.add_argument("--model", default="small", help="whisper模型：tiny/base/small/medium/large-v3")
    parser.add_argument("--lang", default="zh", help="语言代码，如zh/en")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_wav = out_dir / "m3u8_audio.wav"

    headers = None
    if args.cookies_file:
        cookie_header = build_cookie_header_from_netscape(args.cookies_file)
        if cookie_header:
            headers = f"Cookie: {cookie_header}\r\n"

    ff = os.path.join(os.environ.get("LOCALAPPDATA", ""), "ms-playwright", "ffmpeg-1010", "ffmpeg-win64.exe")
    if not os.path.exists(ff):
        ff = "ffmpeg"
    cmd = [ff, "-hide_banner", "-y"]
    if headers:
        cmd += ["-headers", headers]
    cmd += [
        "-protocol_whitelist", "file,http,https,tcp,tls",
        "-i", args.m3u8,
        "-vn", "-ac", "1", "-ar", "16000",
        str(out_wav),
    ]
    try:
        subprocess.run(cmd, check=True)
    except Exception as e:
        print("下载/合并错误:", e)
        sys.exit(1)

    tr_cmd = [
        sys.executable, "transcribe_bilibili.py", str(out_wav),
        "--out", args.out, "--engine", "whisper", "--model", args.model, "--lang", args.lang,
    ]
    try:
        subprocess.run(tr_cmd, check=True)
    except Exception as e:
        print("转写错误:", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
