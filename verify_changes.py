
import sys
import os

print("Testing imports...")
try:
    import qwen_service
    print("✅ qwen_service imported successfully")
except ImportError as e:
    print(f"❌ Failed to import qwen_service: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error importing qwen_service: {e}")
    sys.exit(1)

try:
    import transcribe_bilibili
    print("✅ transcribe_bilibili imported successfully")
except ImportError as e:
    print(f"❌ Failed to import transcribe_bilibili: {e}")
    sys.exit(1)

try:
    import web_ui
    print("✅ web_ui imported successfully")
except ImportError as e:
    print(f"❌ Failed to import web_ui: {e}")
    # web_ui might fail if streamlit is not installed or has issues in headless mode, but import should work if dependencies are met
    print("Warning: web_ui import failed, checking dependencies...")

# Check for QwenASR class existence
if hasattr(qwen_service, 'QwenASR'):
    print("✅ QwenASR class found in qwen_service")
else:
    print("❌ QwenASR class NOT found in qwen_service")

# Check for transcribe_audio_qwen function
if hasattr(qwen_service, 'transcribe_audio_qwen'):
    print("✅ transcribe_audio_qwen function found in qwen_service")
else:
    print("❌ transcribe_audio_qwen function NOT found in qwen_service")

# Check dependencies
try:
    import silero_vad
    print("✅ silero_vad imported successfully")
except ImportError:
    print("❌ silero_vad NOT installed. Please install it.")

try:
    import dashscope
    print("✅ dashscope imported successfully")
except ImportError:
    print("❌ dashscope NOT installed.")

print("Verification complete.")
