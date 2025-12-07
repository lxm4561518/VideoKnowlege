import argparse
import subprocess
import sys
import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def run(url: str, out: str, model: str, lang: str, proxy: str = None, groq_key: str = None, qwen_key: str = None, llm_engine: str = None, asr_engine: str = "whisper", json_mode: bool = False, no_summary: bool = False, max_retries: int = 3, retry_intervals: list = [120, 300, 480]):
    
    def log(msg):
        if json_mode:
            print(msg, file=sys.stderr)
        else:
            print(msg)

    BASE_DIR = Path(__file__).parent.absolute()
    cookies_file = BASE_DIR / "cookies" / "bilibili.txt"
    cookies_file.parent.mkdir(exist_ok=True)
    
    # Step 1: Export Cookies
    log(f">>> [1/3] 正在导出 Bilibili Cookies 到 {cookies_file} ...")
    export_script_name = "export_bilibili_cookies.py" if (BASE_DIR / "export_bilibili_cookies.py").exists() else "export_chrome_cookies_direct.py"
    export_script = str(BASE_DIR / export_script_name)
    try:
        # If json_mode, redirect stdout to stderr to keep stdout clean for JSON output
        stdout_dest = sys.stderr if json_mode else None
        subprocess.run([sys.executable, export_script, "--out", str(cookies_file)], check=True, stdout=stdout_dest, stderr=sys.stderr)
    except Exception as e:
        log(f"Warning: Cookie导出失败 ({e})，将尝试无Cookie模式或使用旧Cookie")

    # Step 2: Try Direct Download & Transcribe (Fast Method)
    log(f">>> [2/3] 尝试直接下载并转写 (高速模式)...")
    status_file = Path(out) / "status.json"
    
    # Init status file (optional, for logging)
    try:
        import time
        os.makedirs(out, exist_ok=True)
        with open(status_file, "w", encoding="utf-8") as f:
            json.dump({"ts": int(time.time()), "phase": "downloading"}, f)
    except:
        pass

    engine = asr_engine
    captured_summary_path = None
    captured_optimized_path = None

    try:
        cmd = [
            sys.executable,
            str(BASE_DIR / "transcribe_bilibili.py"),
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
        if no_summary:
            cmd += ["--no-summary"]
        
        # Set OMP_NUM_THREADS=1 to avoid mkl_malloc memory error on some systems
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        
        # Retry Loop
        import time
        attempt = 0
        while True:
            log(f"Executing command (Attempt {attempt + 1}/{max_retries + 1}): {' '.join(cmd)}")
            
            # In JSON mode, we need to capture output to find result paths, 
            # and ensure no stray prints go to stdout.
            if json_mode:
                process = subprocess.Popen(
                    cmd, 
                    env=env, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT,
                    encoding='utf-8',
                    errors='replace'
                )
                
                captured_summary_path = None
                captured_optimized_path = None
                
                while True:
                    line = process.stdout.readline()
                    if not line and process.poll() is not None:
                        break
                    if line:
                        sys.stderr.write(line) # Echo to stderr
                        line_clean = line.strip()
                        if "AI 智能总结已生成:" in line_clean:
                            captured_summary_path = line_clean.split("AI 智能总结已生成:")[-1].strip()
                        elif "AI 优化文案已生成:" in line_clean:
                            captured_optimized_path = line_clean.split("AI 优化文案已生成:")[-1].strip()
                
                if process.returncode != 0:
                    attempt += 1
                    if attempt > max_retries:
                         raise subprocess.CalledProcessError(process.returncode, cmd)
                    
                    # Calculate wait time
                    idx = attempt - 1
                    wait_seconds = retry_intervals[idx] if idx < len(retry_intervals) else retry_intervals[-1]
                    log(f">>> 警告: 转写失败 (Attempt {attempt}/{max_retries})，将在 {wait_seconds} 秒后重试...")
                    time.sleep(wait_seconds)
                    continue # Retry loop
                else:
                    break # Success, exit loop

            else:
                try:
                    subprocess.run(cmd, check=True, env=env)
                    break # Success
                except subprocess.CalledProcessError as e:
                    attempt += 1
                    if attempt > max_retries:
                        raise e
                    
                    # Calculate wait time
                    idx = attempt - 1
                    wait_seconds = retry_intervals[idx] if idx < len(retry_intervals) else retry_intervals[-1]
                    log(f">>> 警告: 转写失败 (Attempt {attempt}/{max_retries})，将在 {wait_seconds} 秒后重试...")
                    time.sleep(wait_seconds)
                    continue # Retry loop
            
        log(f">>> 成功: 视频已通过高速模式完成转写 (Engine: {engine})！")
        
        if json_mode:
            result = {}
            # If summary is captured, use it to derive other paths
            if captured_summary_path and os.path.exists(captured_summary_path):
                # Extract title from filename: Title_summary.md
                filename = os.path.basename(captured_summary_path)
                # Assuming format: Title_summary.md
                title = filename.replace("_summary.md", "")
                result["title"] = title
                
                with open(captured_summary_path, "r", encoding="utf-8") as f:
                    result["summary"] = f.read()
                
                optimized_path = captured_summary_path.replace("_summary.md", "_optimized.txt")
                if os.path.exists(optimized_path):
                    with open(optimized_path, "r", encoding="utf-8") as f:
                        result["content"] = f.read()
            # If no summary captured (e.g. no_summary=True), try to find optimized text
            elif no_summary:
                 result["summary"] = None
                 
                 if captured_optimized_path and os.path.exists(captured_optimized_path):
                     # Derive title from filename: Title_optimized.txt
                     filename = os.path.basename(captured_optimized_path)
                     title = filename.replace("_optimized.txt", "")
                     result["title"] = title
                     
                     with open(captured_optimized_path, "r", encoding="utf-8") as f:
                        result["content"] = f.read()
                 else:
                     result["content"] = None
                     result["title"] = None
                     result["error"] = "Optimization skipped or file not found, but summary was also disabled."
            else:
                 # Try to guess path if not captured (e.g. skipped because existed)
                 # But for now, just report error or what we have
                 result["error"] = "Summary file not found or captured."

            print(json.dumps(result, ensure_ascii=False))
            
        return

    except subprocess.CalledProcessError:
        log(">>> 警告: 直接下载失败 (可能是B站反爬或Cookie无效)")
        log(">>> [3/3] 切换到录制模式 (Fallback)...")
        if json_mode:
             print(json.dumps({"error": "Download failed. Recording fallback not supported in JSON mode."}, ensure_ascii=False))
             sys.exit(1)

    # Step 3: Fallback to Recording
    log(">>> 开始录制模式 (请保持静音，脚本将自动播放视频)...")
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
    parser.add_argument("--json", action="store_true", help="启用JSON输出模式 (用于N8N集成)，只输出最终结果JSON到stdout，日志输出到stderr")
    parser.add_argument("--no-summary", action="store_true", help="跳过内容总结")
    parser.add_argument("--max-retries", type=int, default=int(os.getenv("MAX_RETRIES", 3)), help="最大重试次数")
    parser.add_argument("--retry-intervals", default=os.getenv("RETRY_INTERVALS", "120,300,480"), help="重试间隔(秒)，逗号分隔")
    
    args = parser.parse_args()

    target_url = args.url or os.getenv("BILIBILI_URL")
    if not target_url:
        print("Error: No URL provided. Please provide a URL as an argument or set BILIBILI_URL in .env file.", file=sys.stderr)
        parser.print_help(sys.stderr)
        sys.exit(1)

    # Determine output format
    env_format = os.getenv("OUTPUT_FORMAT", "json").lower()
    json_mode = args.json or (env_format == "json")

    if not json_mode:
        print(f"Configuration Loaded:")
        print(f"  URL: {target_url}")
        print(f"  ASR Engine: {args.asr_engine}")
        print(f"  LLM Engine: {args.llm_engine}")
        print(f"  Output Dir: {args.out}")
        print(f"  Output Format: {env_format}")
        print(f"  Max Retries: {args.max_retries}")
        print(f"  Retry Intervals: {args.retry_intervals}")
        if args.qwen_key:
            print(f"  Qwen Key: ******{args.qwen_key[-4:]}")
        if args.groq_key:
            print(f"  Groq Key: ******{args.groq_key[-4:]}")

    try:
        retry_intervals_list = [int(x.strip()) for x in args.retry_intervals.split(",")]
    except ValueError:
        print("Error: Invalid retry-intervals format. Use comma separated integers (e.g. 120,300,480)", file=sys.stderr)
        sys.exit(1)

    run(target_url, args.out, args.model, args.lang, args.proxy, args.groq_key, args.qwen_key, args.llm_engine, args.asr_engine, json_mode=json_mode, no_summary=args.no_summary, max_retries=args.max_retries, retry_intervals=retry_intervals_list)


if __name__ == "__main__":
    main()
