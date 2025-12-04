import streamlit as st
import subprocess
import sys
import os
import time
import json
import threading
from pathlib import Path

st.set_page_config(page_title="B站视频转写助手", layout="wide")

def get_status(status_file):
    try:
        if os.path.exists(status_file):
            with open(status_file, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return None

def run_process(url, model, lang, out_dir, groq_key=None, qwen_key=None, proxy=None, llm_engine=None, asr_engine="whisper"):
    cmd = [
        sys.executable,
        "run_bilibili_transcribe.py",
        url,
        "--out", out_dir,
        "--model", model,
        "--lang", lang,
        "--asr-engine", asr_engine
    ]
    if llm_engine:
        cmd += ["--llm-engine", llm_engine]
    
    if groq_key:
        cmd += ["--groq-key", groq_key]
        
    if qwen_key:
        cmd += ["--qwen-key", qwen_key]
    
    if proxy:
        cmd += ["--proxy", proxy]
    
    # Set environment variable for OMP
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        cwd=os.getcwd()
    )
    return process

def main():
    st.title("📺 B站视频自动转写助手")
    
    with st.sidebar:
        st.header("设置")
        
        # Proxy Settings
        use_proxy = st.checkbox("🌐 启用网络代理", value=False, help="如果无法访问 B站或 Groq API，请开启此选项")
        proxy_url = ""
        if use_proxy:
            proxy_url = st.text_input("代理地址 (HTTP/HTTPS)", value="http://127.0.0.1:7890", placeholder="http://127.0.0.1:7890")
        
        # Groq Acceleration
        st.subheader("🛠️ 引擎配置")
        
        # ASR Configuration
        asr_option = st.selectbox("语音转写引擎 (ASR)", ["Whisper (本地)", "Groq (云端/极速)", "Qwen (DashScope/云端)", "Vosk (离线)"], index=0)
        asr_engine = "whisper"
        if "Groq" in asr_option:
            asr_engine = "groq"
        elif "Qwen" in asr_option:
            asr_engine = "qwen"
        elif "Vosk" in asr_option:
            asr_engine = "vosk"
        
        # LLM Configuration
        llm_option = st.selectbox("AI 优化与总结 (LLM)", ["不使用", "Groq (Llama3)", "Qwen (通义千问)"], index=0)
        llm_engine = None
        if "Groq" in llm_option:
            llm_engine = "groq"
        elif "Qwen" in llm_option:
            llm_engine = "qwen"

        # API Keys
        groq_key = ""
        qwen_key = ""
        
        if asr_engine == "groq" or llm_engine == "groq":
            groq_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...", help="用于 Groq 转写或 Llama3 总结")
            st.caption("申请: https://console.groq.com/keys")
        
        if asr_engine == "qwen" or llm_engine == "qwen":
            qwen_key = st.text_input("Qwen API Key", type="password", placeholder="sk-...", help="用于通义千问转写或总结")
            st.caption("申请: https://dashscope.console.aliyun.com/")
        
        model = "small"
        if asr_engine == "whisper":
            model = st.selectbox("Whisper模型", ["tiny", "base", "small", "medium", "large-v3"], index=2)
        elif asr_engine == "groq":
            st.info("Groq 模式下默认使用 whisper-large-v3 模型")
            model = "large-v3"
        elif asr_engine == "qwen":
            st.info("Qwen 模式下使用 qwen3-asr-flash 模型 (云端)")
            model = "qwen3-asr-flash"
            
        lang = st.selectbox("语言", ["zh", "en", "ja"], index=0)
        st.info("说明：优先尝试高速下载，失败后自动切换录制模式。")

    url = st.text_input("请输入B站视频链接", placeholder="https://www.bilibili.com/video/BV...")
    
    # Initialize session state
    if "running" not in st.session_state:
        st.session_state.running = False
    if "start_time" not in st.session_state:
        st.session_state.start_time = 0
    if "output_files" not in st.session_state:
        st.session_state.output_files = None

    out_dir = "outputs"
    status_file = os.path.join(out_dir, "status.json")
    
    start_btn = st.button("开始转写", disabled=st.session_state.running, type="primary")

    if start_btn and url:
        if not url.strip():
            st.error("请输入有效的链接")
        elif (asr_engine == "groq" or llm_engine == "groq") and not groq_key:
            st.error("请在左侧侧边栏输入 Groq API Key")
        elif (asr_engine == "qwen" or llm_engine == "qwen") and not qwen_key:
            st.error("请在左侧侧边栏输入 Qwen API Key")
        else:
            st.session_state.running = True
            st.session_state.start_time = time.time()
            st.session_state.output_files = None
            
            # Clear old status
            if os.path.exists(status_file):
                try:
                    os.remove(status_file)
                except:
                    pass
            
            # Run in a separate thread is hard with streamlit rerun model, 
            # so we use Popen and monitor in a loop here.
            with st.spinner("正在启动任务..."):
                final_proxy = proxy_url if use_proxy and proxy_url else None
                process = run_process(url, model, lang, out_dir, groq_key, qwen_key, final_proxy, llm_engine, asr_engine)
                st.session_state.process = process
                st.rerun()

    if st.session_state.running:
        # Progress Area
        progress_bar = st.progress(0)
        status_text = st.empty()
        log_area = st.empty()
        
        process = st.session_state.process
        
        while True:
            # Check process status
            retcode = process.poll()
            
            # Read status.json
            status = get_status(status_file)
            if status:
                phase = status.get("phase", "init")
                ts = status.get("ts", 0)
                
                if phase == "downloading":
                    progress_bar.progress(10)
                    status_text.info(f"📥 正在下载音频... (最后更新: {time.strftime('%H:%M:%S', time.localtime(ts))})")
                
                elif phase == "recording":
                    dur = status.get("video_dur", 0)
                    curr = status.get("record_secs", 0)
                    eta = status.get("eta_secs", 0)
                    if dur > 0:
                        pct = min(80, int((curr / dur) * 70) + 10)
                        progress_bar.progress(pct)
                    else:
                        progress_bar.progress(20)
                    status_text.warning(f"🔴 正在录制中... 已录制: {int(curr)}s / 预计剩余: {eta}s")
                
                elif phase == "transcribing":
                    segments = status.get("segments", 0)
                    progress_bar.progress(50)
                    status_text.success(f"📝 正在转写中... 已生成 {segments} 句字幕")
                
                elif phase == "transcribed":
                    progress_bar.progress(80)
                    status_text.success("📝 转写完成，正在准备后处理...")
                    
                elif phase == "optimizing":
                    progress_bar.progress(90)
                    status_text.info("🧠 正在进行 AI 智能优化与总结...")
                
                elif phase == "done":
                    progress_bar.progress(100)
                    status_text.success("✅ 转写完成！")
                    break
            else:
                status_text.info("⏳ 正在初始化...")
            
            if retcode is not None:
                if retcode == 0:
                    progress_bar.progress(100)
                    status_text.success("✅ 任务结束")
                else:
                    status_text.error("❌ 任务异常退出")
                break
                
            time.sleep(1)
        
        st.session_state.running = False
        st.session_state.output_files = True
        st.rerun()

    # Display Results
    if st.session_state.output_files:
        st.success("🎉 处理完成！结果如下：")
        
        # Find files modified AFTER the task started
        try:
            start_time = st.session_state.start_time
            # Search recursively in subdirectories
            files = sorted(Path(out_dir).glob("**/*.txt"), key=os.path.getmtime, reverse=True)
            
            # Filter by modification time (allow 5s buffer)
            new_files = [f for f in files if f.stat().st_mtime > start_time - 5]
            
            if new_files:
                # Logic to group files by basename
                base_files = {}
                for f in new_files:
                    # Remove _optimized suffix to find the base name
                    if f.name.endswith("_optimized.txt"):
                        base_name = f.name.replace("_optimized.txt", "")
                    else:
                        base_name = f.stem
                    
                    if base_name not in base_files:
                        base_files[base_name] = []
                    base_files[base_name].append(f)
                
                # Pick the first group (most recent)
                latest_base = list(base_files.keys())[0]
                latest_group = base_files[latest_base]
                
                st.subheader(latest_base)
                
                # Define paths
                txt_file = Path(out_dir) / latest_group[0].parent / f"{latest_base}.txt"
                optimized_file = Path(out_dir) / latest_group[0].parent / f"{latest_base}_optimized.txt"
                summary_file = Path(out_dir) / latest_group[0].parent / f"{latest_base}_summary.md"
                srt_file = Path(out_dir) / latest_group[0].parent / f"{latest_base}.srt"

                tabs = ["📄 原始文案", "🎬 字幕文件 (SRT)"]
                if optimized_file.exists():
                    tabs.insert(0, "✨ AI 优化文案")
                if summary_file.exists():
                    tabs.insert(0, "💡 智能总结")
                
                st_tabs = st.tabs(tabs)
                
                tab_idx = 0
                
                # Summary Tab
                if summary_file.exists():
                    with st_tabs[tab_idx]:
                        with open(summary_file, "r", encoding="utf-8") as f:
                            content = f.read()
                        st.markdown(content)
                        st.download_button("下载总结 (.md)", content, file_name=summary_file.name)
                    tab_idx += 1

                # Optimized Tab
                if optimized_file.exists():
                    with st_tabs[tab_idx]:
                        with open(optimized_file, "r", encoding="utf-8") as f:
                            content = f.read()
                        st.text_area("优化后内容", content, height=400)
                        st.download_button("下载优化文案 (.txt)", content, file_name=optimized_file.name)
                    tab_idx += 1
                
                # Original Tab
                with st_tabs[tab_idx]:
                    if txt_file.exists():
                        with open(txt_file, "r", encoding="utf-8") as f:
                            content = f.read()
                        st.text_area("文案内容", content, height=400)
                        st.download_button("下载文案 (.txt)", content, file_name=txt_file.name)
                    else:
                        st.warning("原始文案文件未找到")
                tab_idx += 1
                
                # SRT Tab
                with st_tabs[tab_idx]:
                    if srt_file.exists():
                        with open(srt_file, "r", encoding="utf-8") as f:
                            srt_content = f.read()
                        st.text_area("字幕内容", srt_content, height=400)
                        st.download_button("下载字幕 (.srt)", srt_content, file_name=srt_file.name)
                    else:
                        st.warning("字幕文件未找到")
        except Exception as e:
            st.error(f"加载结果时出错: {e}")

if __name__ == "__main__":
    main()
