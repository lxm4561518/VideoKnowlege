from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import subprocess
import json
import os
import logging
import sys
from pathlib import Path
from typing import Optional

# 初始化 FastAPI 应用
app = FastAPI(title="VideoKnowledge API", version="1.0.0")

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 定义请求数据模型
class TranscriptRequest(BaseModel):
    bilibili_url: str
    need_summary: bool = False # 默认不需要做内容总结

# 定义响应数据模型
class TranscriptResponse(BaseModel):
    success: bool
    title: Optional[str] = None
    content: Optional[str] = None
    summary: Optional[str] = None
    error: Optional[str] = None

@app.post("/transcript", response_model=TranscriptResponse)
async def create_transcript(request: TranscriptRequest):
    """
    接收 B 站 URL，调用原有逻辑处理视频并返回文案
    """
    bilibili_url = request.bilibili_url
    need_summary = request.need_summary
    logger.info(f"Received request for URL: {bilibili_url}, need_summary: {need_summary}")
    
    # 设定工作目录为当前脚本所在目录
    cwd = str(Path(__file__).parent.absolute())
    
    try:
        venv_python = Path(cwd) / ".venv" / "Scripts" / "python.exe"
        exe = str(venv_python) if venv_python.exists() else sys.executable
        cmd = [
            exe,
            'run_bilibili_transcribe.py',
            bilibili_url,
            '--json'
        ]
        if not need_summary:
            cmd.append('--no-summary')
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        logger.info(f"Using interpreter: {exe}")
        logger.info(f"Executing command: {' '.join(cmd)}")
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        while True:
            line = process.stderr.readline()
            if not line and process.poll() is not None:
                break
            if line:
                logger.info(line.rstrip())
        stdout_data = process.stdout.read() or ""
        ret = process.wait()
        if ret == 0:
            try:
                output_data = json.loads(stdout_data)
                if "error" in output_data and output_data["error"]:
                    return TranscriptResponse(success=False, error=output_data["error"])
                content_val = output_data.get('content')
                if content_val is None or (isinstance(content_val, str) and not content_val.strip()):
                    logger.error("Empty content returned; check previous stderr logs for stage failures")
                return TranscriptResponse(
                    success=True,
                    title=output_data.get('title'),
                    summary=output_data.get('summary'),
                    content=content_val
                )
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode JSON: {e}")
                logger.error(f"Stdout content: {stdout_data}")
                return TranscriptResponse(success=False, error=f"JSON Decode Error: {str(e)}. Raw output: {stdout_data[:500]}...")
        else:
            stderr_data = process.stderr.read() or ""
            if stdout_data:
                logger.error(f"Stdout: {stdout_data[:500]}")
            if stderr_data:
                logger.error(f"Stderr: {stderr_data[:500]}")
            try:
                parsed = json.loads(stdout_data)
                return TranscriptResponse(
                    success=False,
                    title=parsed.get('title'),
                    summary=parsed.get('summary'),
                    content=parsed.get('content'),
                    error=parsed.get('error') or f"Process failed. Stderr: {stderr_data}"
                )
            except Exception:
                return TranscriptResponse(success=False, error=f"Process failed. Stderr: {stderr_data} Stdout: {stdout_data[:500]}")
            
    except Exception as e:
        logger.error(f"Internal Server Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"处理视频时发生错误: {str(e)}")

@app.get("/health")
async def health_check():
    """健康检查端点，用于监控服务状态"""
    return {"status": "healthy"}

if __name__ == "__main__":
    # 启动服务，host='0.0.0.0' 允许外部访问
    # 端口 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
