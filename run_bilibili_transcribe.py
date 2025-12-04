import argparse
import subprocess
import sys
import os
from pathlib import Path


def run(url: str, out: str, model: str, lang: str, proxy: str = None):
    cookies_file = Path("cookies") / "bilibili.txt"
    cookies_file.parent.mkdir(exist_ok=True)
    
    # Step 1: Export Cookies
    print(f">>> [1/3] 正在导出 Bilibili Cookies 到 {cookies_file} ...")
    export_script = "export_bilibili_cookies.py" if Path("export_bilibili_cookies.py").exists() else "export_chrome_cookies_direct.py"
    try:
        subprocess.run([sys.executable, export_script, "--out", str(cookies_file)], check=True)
    except Exception as e:
        print(f"Warning: Cookie导出失败 ({e})，将尝试无Cookie模式或使用旧Cookie")

    # Step 2: Try Direct Download & Transcribe (Fast Method)
    print(f">>> [2/3] 尝试直接下载并转写 (高速模式)...")
    status_file = Path(out) / "status.json"
    
    # Init status file for UI
    try:
        import json
        import time
        with open(status_file, "w", encoding="utf-8") as f:
            json.dump({"ts": int(time.time()), "phase": "downloading"}, f)
    except:
        pass

    try:
        cmd = [
            sys.executable,
            "transcribe_bilibili.py",
            url,
            "--out",
            out,
            "--engine",
            "whisper",
            "--model",
            model,
            "--lang",
            lang,
            "--cookies-file",
            str(cookies_file),
            "--status-file",
            str(status_file)
        ]
        if proxy:
            cmd += ["--proxy", proxy]
        
        # Set OMP_NUM_THREADS=1 to avoid mkl_malloc memory error on some systems
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"
        
        subprocess.run(cmd, check=True, env=env)
        print(">>> 成功: 视频已通过高速模式完成转写！")
        return
    except subprocess.CalledProcessError:
        print(">>> 警告: 直接下载失败 (可能是B站反爬或Cookie无效)")
        print(">>> [3/3] 切换到录制模式 (Fallback)...")

    # Step 3: Fallback to Recording
    print(">>> 开始录制模式 (请保持静音，脚本将自动播放视频)...")
    # Ensure previous status is cleared or marked
    try:
        cmd = [
            sys.executable,
            "record_and_transcribe.py",
            url,
            "--out",
            out,
            "--model",
            model,
            "--lang",
            lang,
            "--auto"
        ]
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: 录制模式也失败了 ({e})")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="一键导出Cookie并转写B站视频")
    parser.add_argument("url", help="B站视频页面URL")
    parser.add_argument("--out", default="outputs", help="输出目录")
    parser.add_argument("--model", default="small", help="whisper模型：tiny/base/small/medium/large-v3")
    parser.add_argument("--lang", default="zh", help="语言代码，如zh/en")
    parser.add_argument("--proxy", default=None, help="HTTP/HTTPS代理，如 http://127.0.0.1:7890")
    args = parser.parse_args()
    run(args.url, args.out, args.model, args.lang, args.proxy)


if __name__ == "__main__":
    main()
