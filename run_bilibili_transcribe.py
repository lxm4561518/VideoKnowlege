import argparse
import subprocess
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def run(url: str, out: str, model: str, lang: str, proxy: str = None, groq_key: str = None, qwen_key: str = None, llm_engine: str = None, asr_engine: str = "whisper"):
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
    
    # Init status file (optional, for logging)
    try:
        import json
        import time
        os.makedirs(out, exist_ok=True)
        with open(status_file, "w", encoding="utf-8") as f:
            json.dump({"ts": int(time.time()), "phase": "downloading"}, f)
    except:
        pass

    engine = asr_engine

    try:
        cmd = [
            sys.executable,
            "transcribe_bilibili.py",
            url,
            "--out",
            out,
            "--engine",
            engine,
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
        if groq_key:
            cmd += ["--groq-key", groq_key]
        if qwen_key:
            cmd += ["--qwen-key", qwen_key]
        if llm_engine:
            cmd += ["--llm-engine", llm_engine]
        
        # Set OMP_NUM_THREADS=1 to avoid mkl_malloc memory error on some systems
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"
        
        print(f"Executing command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, env=env)
        print(f">>> 成功: 视频已通过高速模式完成转写 (Engine: {engine})！")
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
        # TODO: Add Groq/Qwen support to recording mode if needed
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: 录制模式也失败了 ({e})")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="一键导出Cookie并转写B站视频 (Backend Mode)")
    
    # Argument defaults are pulled from environment variables
    parser.add_argument("url", nargs='?', help="B站视频页面URL (Optional if BILIBILI_URL is set in .env)")
    parser.add_argument("--out", default=os.getenv("OUTPUT_DIR", "outputs"), help="输出目录")
    parser.add_argument("--model", default=os.getenv("MODEL", "small"), help="whisper模型：tiny/base/small/medium/large-v3")
    parser.add_argument("--lang", default=os.getenv("LANG", "zh"), help="语言代码，如zh/en")
    parser.add_argument("--proxy", default=os.getenv("PROXY"), help="HTTP/HTTPS代理，如 http://127.0.0.1:7890")
    parser.add_argument("--groq-key", default=os.getenv("GROQ_API_KEY"), help="Groq API Key")
    parser.add_argument("--qwen-key", default=os.getenv("QWEN_API_KEY"), help="Qwen API Key")
    parser.add_argument("--llm-engine", choices=["groq", "qwen"], default=os.getenv("LLM_ENGINE"), help="LLM Engine")
    parser.add_argument("--asr-engine", choices=["whisper", "groq", "vosk", "qwen"], default=os.getenv("ASR_ENGINE", "whisper"), help="ASR Engine")
    
    args = parser.parse_args()

    target_url = args.url or os.getenv("BILIBILI_URL")
    if not target_url:
        print("Error: No URL provided. Please provide a URL as an argument or set BILIBILI_URL in .env file.")
        parser.print_help()
        sys.exit(1)

    print(f"Configuration Loaded:")
    print(f"  URL: {target_url}")
    print(f"  ASR Engine: {args.asr_engine}")
    print(f"  LLM Engine: {args.llm_engine}")
    print(f"  Output Dir: {args.out}")
    if args.qwen_key:
        print(f"  Qwen Key: ******{args.qwen_key[-4:]}")
    if args.groq_key:
        print(f"  Groq Key: ******{args.groq_key[-4:]}")

    run(target_url, args.out, args.model, args.lang, args.proxy, args.groq_key, args.qwen_key, args.llm_engine, args.asr_engine)


if __name__ == "__main__":
    main()
