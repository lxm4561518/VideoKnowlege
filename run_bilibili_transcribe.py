import argparse
import subprocess
import sys
import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def run(url: str, out: str, model: str, lang: str, proxy: str = None, groq_key: str = None, qwen_key: str = None, llm_engine: str = None, asr_engine: str = "whisper", json_mode: bool = False, no_summary: bool = False, max_retries: int = 3, retry_intervals: list = [120, 300, 480], cookies_file_arg: str = None, cookies_from_browser_arg: str = None, skip_cookie_export: bool = False, cookie_min_ttl: int = 600, skip_recording_fallback: bool = False, force_cookie_export: bool = False, enable_m3u8_fallback: bool = False):
    
    def log(msg):
        if json_mode:
            print(msg, file=sys.stderr)
        else:
            print(msg)

    BASE_DIR = Path(__file__).parent.absolute()
    cookies_file = Path(cookies_file_arg).absolute() if cookies_file_arg else (BASE_DIR / "cookies" / "bilibili.txt")
    cookies_file.parent.mkdir(exist_ok=True)
    cookies_from_browser = cookies_from_browser_arg or os.getenv("COOKIES_FROM_BROWSER")
    log(">>> [0/3] 检查Cookie配置与有效性 ...")
    try:
        exists = cookies_file.exists()
        size = cookies_file.stat().st_size if exists else 0
        log(f"Cookie文件: {cookies_file} | exists={exists} | size={size}")
    except Exception:
        log(f"Cookie文件: {cookies_file} | exists=False")
    log(f"SKIP_COOKIE_EXPORT={skip_cookie_export} | COOKIE_MIN_TTL={cookie_min_ttl}")
    if cookies_from_browser:
        log(f"COOKIES_FROM_BROWSER={cookies_from_browser}")
    
    # Step 1: Export Cookies
    do_export = True
    def _netscape_cookie_valid(fp: Path, min_ttl: int):
        try:
            import time
            now = int(time.time())
            has_bili = False
            max_ttl = -1
            with open(fp, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    t = line.strip()
                    if not t or t.startswith("#"):
                        continue
                    parts = t.split("\t")
                    if len(parts) < 7:
                        parts = t.split()
                        if len(parts) < 7:
                            continue
                    domain, flag, path, secure, expires_str, name, value = parts[:7]
                    if "bilibili.com" in domain:
                        has_bili = True
                        try:
                            exp = int(expires_str)
                            ttl = exp - now
                            if ttl > max_ttl:
                                max_ttl = ttl
                        except Exception:
                            pass
            if not has_bili:
                return False, 0
            return max_ttl > min_ttl, max_ttl
        except Exception:
            return False, 0
    if force_cookie_export:
        do_export = True
        log(">>> [1/3] 已启用强制导出Cookie (忽略现有文件/TTL)")
    elif skip_cookie_export:
        do_export = False
    elif cookies_file.exists():
        try:
            valid, ttl = _netscape_cookie_valid(cookies_file, cookie_min_ttl)
            if valid:
                do_export = False
                log(f">>> [1/3] 跳过Cookie导出，复用已有文件 (TTL {int(ttl)} 秒): {cookies_file}")
            else:
                do_export = True
                log(f">>> [1/3] Cookie 文件已过期或无效，将重新导出: {cookies_file}")
        except Exception:
            do_export = True
    if do_export:
        log(f">>> [1/3] 正在导出 Bilibili Cookies 到 {cookies_file} ...")
        export_script_name = "export_bilibili_cookies.py" if (BASE_DIR / "export_bilibili_cookies.py").exists() else "export_chrome_cookies_direct.py"
        export_script = str(BASE_DIR / export_script_name)
        try:
            stdout_dest = sys.stderr if json_mode else None
            subprocess.run([sys.executable, export_script, "--out", str(cookies_file)], check=True, stdout=stdout_dest, stderr=sys.stderr)
            log(f">>> [1/3] Cookie导出完成: {cookies_file}")
        except Exception as e:
            log(f"Warning: Cookie导出失败 ({e})，将尝试无Cookie模式或使用旧Cookie")
            if not cookies_from_browser:
                cookies_from_browser = "chrome"
    else:
        try:
            valid, ttl = _netscape_cookie_valid(cookies_file, cookie_min_ttl)
            log(f">>> [1/3] 使用已有Cookie文件: {cookies_file} | TTL={int(ttl) if ttl else 0}")
        except Exception:
            log(f">>> [1/3] 使用已有Cookie文件: {cookies_file}")

    # Step 2: Try Direct Download & Transcribe (Fast Method)
    log(f">>> [2/3] 尝试直接下载并转写 (高速模式)...")
    status_file = Path(out) / "status.json"
    
    # Init status file (optional, for logging)
    try:
        import time
        os.makedirs(out, exist_ok=True)
        with open(status_file, "w", encoding="utf-8") as f:
            json.dump({"ts": int(time.time()), "phase": "downloading"}, f)
        log(f">>> 状态文件已初始化: {status_file}")
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
            "--status-file",
            str(status_file)
        ]
        log(f">>> 直下模式参数: engine={engine}, model={model}, lang={lang}, out={out}")
        if cookies_from_browser:
            log(f">>> 使用浏览器Cookie读取方式: {cookies_from_browser}")
            cmd += ["--cookies-from-browser", cookies_from_browser]
        elif cookies_file.exists():
            log(f">>> 使用Cookie文件: {cookies_file}")
            cmd += ["--cookies-file", str(cookies_file)]
        else:
            log(">>> 未使用任何Cookie：可能导致直下失败(412)")
        if proxy:
            log(f">>> 使用代理: {proxy}")
            cmd += ["--proxy", proxy]
        if groq_key:
            cmd += ["--groq-key", groq_key]
        if qwen_key:
            cmd += ["--qwen-key", qwen_key]
        if llm_engine:
            cmd += ["--llm-engine", llm_engine]
        # transcribe_bilibili.py 不支持 --no-summary 参数，跳过
        
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
                captured_txt_path = None
                
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
                        elif line_clean.endswith('.txt') and ("_optimized.txt" not in line_clean):
                            # capture base transcript txt path
                            captured_txt_path = line_clean
                
                if process.returncode != 0:
                    attempt += 1
                    if attempt > max_retries:
                         raise subprocess.CalledProcessError(process.returncode, cmd)
                    
                    # Calculate wait time
                    idx = attempt - 1
                    wait_seconds = retry_intervals[idx] if idx < len(retry_intervals) else retry_intervals[-1]
                    log(f">>> 警告: 直下失败 (Attempt {attempt}/{max_retries})，将在 {wait_seconds} 秒后重试...")
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
                    log(f">>> 警告: 直下失败 (Attempt {attempt}/{max_retries})，将在 {wait_seconds} 秒后重试...")
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
            elif no_summary or not llm_engine:
                 result["summary"] = None
                 # Prefer optimized if present, else base txt content
                 base_txt = captured_txt_path
                 if captured_optimized_path and os.path.exists(captured_optimized_path):
                     filename = os.path.basename(captured_optimized_path)
                     result["title"] = filename.replace("_optimized.txt", "")
                     with open(captured_optimized_path, "r", encoding="utf-8") as f:
                        result["content"] = f.read()
                 elif base_txt and os.path.exists(base_txt):
                     filename = os.path.basename(base_txt)
                     result["title"] = os.path.splitext(filename)[0]
                     with open(base_txt, "r", encoding="utf-8") as f:
                        result["content"] = f.read()
                 else:
                     result["error"] = "Transcript file not found."
            else:
                 # Try to guess path if not captured (e.g. skipped because existed)
                 # But for now, just report error or what we have
                 result["error"] = "Summary file not found or captured."

            print(json.dumps(result, ensure_ascii=False))
            
        return

    except subprocess.CalledProcessError:
        log(">>> 警告: 直接下载失败 (可能是B站反爬或Cookie无效)")
        if skip_recording_fallback:
            if enable_m3u8_fallback:
                log(">>> [3/3] 尝试m3u8直链抓取并转写 ...")
                try:
                    env2 = os.environ.copy()
                    env2["OMP_NUM_THREADS"] = "1"
                    env2["PYTHONIOENCODING"] = "utf-8"
                    cap_cmd = [
                        sys.executable,
                        str(BASE_DIR / "auto_m3u8_capture.py"),
                        url,
                        "--cookies-file",
                        str(cookies_file),
                        "--out",
                        out,
                        "--model",
                        model,
                        "--lang",
                        lang,
                    ]
                    subprocess.run(cap_cmd, check=True, env=env2)
                    latest_txt = None
                    latest_mtime = -1
                    for root, dirs, files in os.walk(out):
                        for fn in files:
                            if fn.lower().endswith(".txt") and not fn.endswith("_optimized.txt") and not fn.endswith("_summary.md"):
                                fp = os.path.join(root, fn)
                                try:
                                    mt = os.path.getmtime(fp)
                                except Exception:
                                    mt = 0
                                if mt > latest_mtime:
                                    latest_mtime = mt
                                    latest_txt = fp
                    if json_mode:
                        if latest_txt and os.path.exists(latest_txt):
                            try:
                                with open(latest_txt, "r", encoding="utf-8") as f:
                                    content = f.read()
                            except Exception:
                                content = None
                            title = os.path.splitext(os.path.basename(latest_txt))[0]
                            print(json.dumps({"success": True, "title": title, "summary": None, "content": content}, ensure_ascii=False))
                            return
                        print(json.dumps({"success": False, "error": "M3U8 fallback succeeded but result file not found."}, ensure_ascii=False))
                        return
                    return
                except Exception as e:
                    if json_mode:
                        print(json.dumps({"success": False, "error": f"M3U8 fallback failed: {e}"}, ensure_ascii=False))
                        return
                    else:
                        print(f"M3U8 fallback failed: {e}", file=sys.stderr)
                        sys.exit(1)
            if json_mode:
                print(json.dumps({"success": False, "error": "Direct download failed; recording fallback disabled."}, ensure_ascii=False))
                return
            else:
                print("直接下载失败；已禁用录制回退。", file=sys.stderr)
                sys.exit(1)
        log(">>> [3/3] 切换到录制模式 (Fallback)...")
        if json_mode:
            try:
                env = os.environ.copy()
                env["OMP_NUM_THREADS"] = "1"
                env["PYTHONIOENCODING"] = "utf-8"
                rec_cmd = [
                    sys.executable,
                    str(BASE_DIR / "record_and_transcribe.py"),
                    url,
                    "--out",
                    out,
                    "--model",
                    model,
                    "--lang",
                    lang,
                    "--engine",
                    engine,
                    "--auto",
                    "--until-ended",
                ]
                # 可选限制最大录制时长，避免过长阻塞
                max_secs_env = os.getenv("RECORD_MAX_SECONDS")
                if max_secs_env:
                    try:
                        rec_cmd += ["--max-seconds", str(int(max_secs_env))]
                    except Exception:
                        pass
                process = subprocess.run(rec_cmd, check=True, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
                latest_txt = None
                latest_mtime = -1
                for root, dirs, files in os.walk(out):
                    for fn in files:
                        if fn.lower().endswith(".txt") and not fn.endswith("_optimized.txt") and not fn.endswith("_summary.md"):
                            fp = os.path.join(root, fn)
                            try:
                                mt = os.path.getmtime(fp)
                            except Exception:
                                mt = 0
                            if mt > latest_mtime:
                                latest_mtime = mt
                                latest_txt = fp
                if latest_txt and os.path.exists(latest_txt):
                    try:
                        with open(latest_txt, "r", encoding="utf-8") as f:
                            content = f.read()
                    except Exception:
                        content = None
                    title = os.path.splitext(os.path.basename(latest_txt))[0]
                    print(json.dumps({"success": True, "title": title, "summary": None, "content": content}, ensure_ascii=False))
                    return
                print(json.dumps({"success": False, "error": "Recording succeeded but result file not found."}, ensure_ascii=False))
                sys.exit(1)
            except Exception as e:
                print(json.dumps({"success": False, "error": f"Recording fallback failed: {e}"}, ensure_ascii=False))
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
    parser.add_argument("--asr-engine", choices=["whisper", "groq", "vosk", "qwen", "funasr"], default=os.getenv("ASR_ENGINE", "whisper"), help="ASR Engine")
    parser.add_argument("--json", action="store_true", help="启用JSON输出模式 (用于N8N集成)，只输出最终结果JSON到stdout，日志输出到stderr")
    parser.add_argument("--no-summary", action="store_true", help="跳过内容总结")
    parser.add_argument("--max-retries", type=int, default=int(os.getenv("MAX_RETRIES", 3)), help="最大重试次数")
    parser.add_argument("--retry-intervals", default=os.getenv("RETRY_INTERVALS", "120,300,480"), help="重试间隔(秒)，逗号分隔")
    parser.add_argument("--cookies-file", default=os.getenv("COOKIES_FILE"), help="已有Cookie文件路径")
    parser.add_argument("--cookies-from-browser", default=os.getenv("COOKIES_FROM_BROWSER"), help="直接从浏览器读取Cookie：chrome/edge/firefox")
    parser.add_argument("--skip-cookie-export", action="store_true", default=(os.getenv("SKIP_COOKIE_EXPORT") in ("1","true","True")), help="跳过Cookie导出步骤")
    parser.add_argument("--skip-recording-fallback", action="store_true", default=(os.getenv("SKIP_RECORDING_FALLBACK") in ("1","true","True")), help="直下失败时跳过录制回退，直接返回错误")
    parser.add_argument("--cookie-min-ttl", type=int, default=int(os.getenv("COOKIE_MIN_TTL", 600)), help="Cookie 最小剩余有效期(秒)，低于该值则重新导出")
    parser.add_argument("--force-cookie-export", action="store_true", default=(os.getenv("FORCE_COOKIE_EXPORT") in ("1","true","True")), help="强制导出Cookie，忽略现有文件/TTL")
    parser.add_argument("--enable-m3u8-fallback", action="store_true", default=(os.getenv("ENABLE_M3U8_FALLBACK") in ("1","true","True")), help="启用m3u8抓取直链的回退方案")
    
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

    run(target_url, args.out, args.model, args.lang, args.proxy, args.groq_key, args.qwen_key, args.llm_engine, args.asr_engine, json_mode=json_mode, no_summary=args.no_summary, max_retries=args.max_retries, retry_intervals=retry_intervals_list, cookies_file_arg=args.cookies_file, cookies_from_browser_arg=args.cookies_from_browser, skip_cookie_export=args.skip_cookie_export, cookie_min_ttl=args.cookie_min_ttl, skip_recording_fallback=args.skip_recording_fallback, force_cookie_export=args.force_cookie_export, enable_m3u8_fallback=args.enable_m3u8_fallback)


if __name__ == "__main__":
    main()
