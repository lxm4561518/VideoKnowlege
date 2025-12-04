from qwen3_asr_toolkit.qwen3asr import QwenASR
import inspect

try:
    print(inspect.getsource(QwenASR.asr))
except Exception as e:
    print(e)
