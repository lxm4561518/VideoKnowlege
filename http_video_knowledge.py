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
        # 调用 run_bilibili_transcribe.py
        # 使用 sys.executable 确保使用当前 Python 环境
        cmd = [
            sys.executable, 
            'run_bilibili_transcribe.py', 
            bilibili_url, 
            '--json'
        ]
        
        if not need_summary:
            cmd.append('--no-summary')
        
        # 继承当前环境变量 (包括 PATH, 还有 .env 加载的那些如果已经被加载)
        # 并强制设置 PYTHONIOENCODING 为 utf-8，确保子进程输出为 UTF-8
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        
        logger.info(f"Executing command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            cwd=cwd,
            encoding='utf-8',
            env=env,
            errors='replace' # 防止编码错误导致 crash
        )
        
        if result.returncode == 0:
            # 解析原有脚本的 JSON 输出
            try:
                # stdout 应该只包含 JSON
                output_data = json.loads(result.stdout)
                
                # 检查是否有错误字段
                if "error" in output_data and output_data["error"]:
                     logger.error(f"Script returned error: {output_data['error']}")
                     return TranscriptResponse(
                        success=False,
                        error=output_data["error"]
                    )
                
                return TranscriptResponse(
                    success=True,
                    title=output_data.get('title'),
                    summary=output_data.get('summary'),
                    content=output_data.get('content')
                )
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode JSON: {e}")
                logger.error(f"Stdout content: {result.stdout}")
                return TranscriptResponse(
                    success=False,
                    error=f"JSON Decode Error: {str(e)}. Raw output: {result.stdout[:500]}..."
                )
        else:
            logger.error(f"Process failed with return code {result.returncode}")
            logger.error(f"Stderr: {result.stderr}")
            return TranscriptResponse(
                success=False,
                error=f"Process failed. Stderr: {result.stderr}"
            )
            
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
