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

def run_process(url, model, lang, out_dir):
    cmd = [
        sys.executable,
        "run_bilibili_transcribe.py",
        url,
        "--out", out_dir,
        "--model", model,
        "--lang", lang
    ]
    
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
        model = st.selectbox("Whisper模型", ["tiny", "base", "small", "medium", "large-v3"], index=2)
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
                process = run_process(url, model, lang, out_dir)
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
                    progress_bar.progress(85)
                    status_text.success(f"📝 正在转写中... 已生成 {segments} 句字幕")
                
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
            files = sorted(Path(out_dir).glob("*.txt"), key=os.path.getmtime, reverse=True)
            
            # Filter by modification time (allow 5s buffer)
            new_files = [f for f in files if f.stat().st_mtime > start_time - 5]
            
            if new_files:
                latest_file = new_files[0]
                st.subheader(latest_file.stem)
                
                tab1, tab2 = st.tabs(["📄 纯文本", "🎬 字幕文件 (SRT)"])
                
                with tab1:
                    with open(latest_file, "r", encoding="utf-8") as f:
                        content = f.read()
                    st.text_area("文案内容", content, height=400)
                    st.download_button("下载文案 (.txt)", content, file_name=latest_file.name)
                
                with tab2:
                    srt_file = latest_file.with_suffix(".srt")
                    if srt_file.exists():
                        with open(srt_file, "r", encoding="utf-8") as f:
                            srt_content = f.read()
                        st.text_area("字幕内容", srt_content, height=400)
                        st.download_button("下载字幕 (.srt)", srt_content, file_name=srt_file.name)
            else:
                st.warning("未找到本次任务生成的输出文件。可能是因为：")
                st.write("1. 视频转写失败")
                st.write("2. 文件已存在且未被覆盖（跳过了转写）")
                st.write("3. 任务被意外终止")
                
                # Option to show older files
                if files:
                    st.info(f"找到 {len(files)} 个历史文件，最近的一个是: {files[0].name}")
                    if st.button("显示最近的历史文件"):
                         # This logic requires rerun to persist the choice, simplistic approach here
                         st.session_state.start_time = 0 # Reset time filter to show old files
                         st.rerun()

        except Exception as e:
            st.error(f"读取结果失败: {e}")

if __name__ == "__main__":
    main()
