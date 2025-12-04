from qwen3_asr_toolkit.qwen3asr import QwenASR
import inspect
import sys

try:
    print(f"INIT: {inspect.signature(QwenASR.__init__)}")
    print(f"ASR: {inspect.signature(QwenASR.asr)}")
except Exception as e:
    print(e)
