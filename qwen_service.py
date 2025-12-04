import os
import io
import time
import random
import json
import shutil
import subprocess
import requests
import concurrent.futures
import numpy as np
import soundfile as sf
import librosa
import dashscope
from http import HTTPStatus
from dashscope import Generation, MultiModalConversation
from pydub import AudioSegment
from silero_vad import load_silero_vad, get_speech_timestamps
from typing import List, Tuple, Any
from pathlib import Path

# Constants
WAV_SAMPLE_RATE = 16000
MAX_API_RETRY = 5
API_RETRY_SLEEP = (1, 2)

LANGUAGE_CODE_MAPPING = {
    "ar": "Arabic",
    "zh": "Chinese",
    "en": "English",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "pt": "Portuguese",
    "ru": "Russian",
    "es": "Spanish"
}

def get_client(api_key):
    if not api_key:
        raise ValueError("DashScope API Key is required")
    dashscope.api_key = api_key

# --- Qwen ASR Components (Adapted from Qwen3-ASR-Toolkit) ---

class QwenASR:
    def __init__(self, model: str = "qwen3-asr-flash"):
        self.model = model

    def post_text_process(self, text, threshold=20):
        def fix_char_repeats(s, thresh):
            res = []
            i = 0
            n = len(s)
            while i < n:
                count = 1
                while i + count < n and s[i + count] == s[i]:
                    count += 1

                if count > thresh:
                    res.append(s[i])
                    i += count
                else:
                    res.append(s[i:i + count])
                    i += count
            return ''.join(res)

        def fix_pattern_repeats(s, thresh, max_len=20):
            n = len(s)
            min_repeat_chars = thresh * 2
            if n < min_repeat_chars:
                return s

            i = 0
            result = []
            while i <= n - min_repeat_chars:
                found = False
                for k in range(1, max_len + 1):
                    if i + k * thresh > n:
                        break

                    pattern = s[i:i + k]

                    valid = True
                    for rep in range(1, thresh):
                        start_idx = i + rep * k
                        if s[start_idx:start_idx + k] != pattern:
                            valid = False
                            break

                    if valid:
                        total_rep = thresh
                        end_index = i + thresh * k
                        while end_index + k <= n and s[end_index:end_index + k] == pattern:
                            total_rep += 1
                            end_index += k

                        result.append(pattern)
                        result.append(fix_pattern_repeats(s[end_index:], thresh, max_len))
                        i = n
                        found = True
                        break

                if found:
                    break
                else:
                    result.append(s[i])
                    i += 1

            if not found:
                result.append(s[i:])
            return ''.join(result)

        text = fix_char_repeats(text, threshold)
        return fix_pattern_repeats(text, threshold)

    def asr(self, wav_url: str, context: str = ""):
        if not wav_url.startswith("http"):
            if not os.path.exists(wav_url):
                 raise FileNotFoundError(f"{wav_url} not exists!")
            file_path = wav_url
            file_size = os.path.getsize(file_path)

            # file size > 10M, convert to mp3 to save bandwidth/meet limit?
            # Actually API limit is 10MB.
            if file_size > 10 * 1024 * 1024:
                mp3_path = os.path.splitext(file_path)[0] + ".mp3"
                # Use pydub for conversion
                audio = AudioSegment.from_file(file_path)
                audio.export(mp3_path, format="mp3")
                wav_url = mp3_path

            # DashScope SDK on Windows seems to have issues with file:/// (3 slashes)
            # It likely strips file:// and gets /C:/... which fails.
            # We try to use file://C:/... (2 slashes) which results in C:/... after stripping.
            wav_url = Path(wav_url).absolute().as_uri().replace("file:///", "file://")

        last_exception = None
        for _ in range(MAX_API_RETRY):
            try:
                messages = [
                    {
                        "role": "system",
                        "content": [{"text": context}]
                    },
                    {
                        "role": "user",
                        "content": [{"audio": wav_url}]
                    }
                ]
                
                response = MultiModalConversation.call(
                    model=self.model,
                    messages=messages,
                    result_format="message",
                    asr_options={
                        "enable_lid": True,
                        "enable_itn": False
                    }
                )

                if response.status_code != 200:
                     raise Exception(f"API Error {response.status_code}: {response.message}")

                output = response['output']['choices'][0]
                recog_text = ""
                if output["message"]["content"]:
                    recog_text = output["message"]["content"][0].get("text", "")
                
                lang_code = None
                if "annotations" in output["message"]:
                    lang_code = output["message"]["annotations"][0].get("language")
                
                language = LANGUAGE_CODE_MAPPING.get(lang_code, "Not Supported")
                return language, self.post_text_process(recog_text)

            except Exception as e:
                last_exception = e
                print(f"Qwen ASR Retry {_ + 1}/{MAX_API_RETRY}... Error: {e}")
                # If response is available and has code, check it
                # But response might be undefined if call raised exception
                time.sleep(random.uniform(*API_RETRY_SLEEP))
        
        raise Exception(f"Qwen ASR task failed after retries. Last error: {last_exception}")


def load_audio(file_path: str) -> np.ndarray:
    try:
        if file_path.startswith(("http://", "https://")):
             raise ValueError("Using ffmpeg to load remote file not supported in this simplified version.")
        
        # Try librosa first
        wav_data, _ = librosa.load(file_path, sr=WAV_SAMPLE_RATE, mono=True)
        return wav_data
    except FileNotFoundError as e:
         if "WinError 2" in str(e) or e.errno == 2:
             raise RuntimeError("FFmpeg not found. Please install FFmpeg and add it to PATH, or place ffmpeg.exe in the project directory.") from e
         raise
    except Exception as e:
        print(f"Librosa load failed, trying ffmpeg directly: {e}")
        try:
            # Fallback: try using ffmpeg directly to decode to stdout
            # command = ['ffmpeg', '-i', file_path, '-f', 'wav', '-ar', str(WAV_SAMPLE_RATE), '-ac', '1', '-']
            # process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            # return np.frombuffer(process.stdout, dtype=np.int16).astype(np.float32) / 32768.0
            # The above is complex to handle headers. simpler to raise error if librosa fails and it's likely ffmpeg missing.
            if "No backend available" in str(e) or "WinError 2" in str(e):
                 raise RuntimeError("FFmpeg not found. Please install FFmpeg to process non-WAV audio files.") from e
            raise
        except Exception as ffmpeg_error:
            raise RuntimeError(f"Failed to load audio. Ensure FFmpeg is installed. Details: {e}") from ffmpeg_error
        try:
            command = [
                'ffmpeg',
                '-i', file_path,
                '-ar', str(WAV_SAMPLE_RATE),
                '-ac', '1',
                '-c:a', 'pcm_s16le',
                '-f', 'wav',
                '-'
            ]
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stdout_data, stderr_data = process.communicate()

            if process.returncode != 0:
                raise RuntimeError(f"FFmpeg error: {stderr_data.decode('utf-8', errors='ignore')}")

            with io.BytesIO(stdout_data) as data_io:
                wav_data, sr = sf.read(data_io, dtype='float32')

            return wav_data
        except Exception as ffmpeg_e:
            raise RuntimeError(f"Failed to load audio '{file_path}'. Error: {ffmpeg_e}")

def process_vad(wav: np.ndarray, worker_vad_model, segment_threshold_s: int = 120, max_segment_threshold_s: int = 180) -> List[Tuple[int, int, np.ndarray]]:
    try:
        vad_params = {
            'sampling_rate': WAV_SAMPLE_RATE,
            'return_seconds': False,
            'min_speech_duration_ms': 1500,
            'min_silence_duration_ms': 500
        }

        speech_timestamps = get_speech_timestamps(
            wav,
            worker_vad_model,
            **vad_params
        )

        if not speech_timestamps:
            # Fallback if no speech detected? Or just return whole?
            # The original raised ValueError. Let's treat as one big chunk if small enough, or force split.
             # If VAD fails to find speech, maybe it's music or silence.
             # Let's return empty or whole file?
             # Original raised ValueError.
             pass # Let exception handler handle it to force split.

        potential_split_points_s = {0.0, len(wav)}
        for i in range(len(speech_timestamps)):
            start_of_next_s = speech_timestamps[i]['start']
            potential_split_points_s.add(start_of_next_s)
        sorted_potential_splits = sorted(list(potential_split_points_s))

        final_split_points_s = {0.0, len(wav)}
        segment_threshold_samples = segment_threshold_s * WAV_SAMPLE_RATE
        target_time = segment_threshold_samples
        while target_time < len(wav):
            closest_point = min(sorted_potential_splits, key=lambda p: abs(p - target_time))
            final_split_points_s.add(closest_point)
            target_time += segment_threshold_samples
        final_ordered_splits = sorted(list(final_split_points_s))

        max_segment_threshold_samples = max_segment_threshold_s * WAV_SAMPLE_RATE
        new_split_points = [0.0]

        for i in range(1, len(final_ordered_splits)):
            start = final_ordered_splits[i - 1]
            end = final_ordered_splits[i]
            segment_length = end - start

            if segment_length <= max_segment_threshold_samples:
                new_split_points.append(end)
            else:
                num_subsegments = int(np.ceil(segment_length / max_segment_threshold_samples))
                subsegment_length = segment_length / num_subsegments

                for j in range(1, num_subsegments):
                    split_point = start + j * subsegment_length
                    new_split_points.append(split_point)

                new_split_points.append(end)

        segmented_wavs = []
        for i in range(len(new_split_points) - 1):
            start_sample = int(new_split_points[i])
            end_sample = int(new_split_points[i + 1])
            segmented_wavs.append((start_sample, end_sample, wav[start_sample:end_sample]))
        return segmented_wavs

    except Exception as e:
        # Fallback: fixed size chunking
        segmented_wavs = []
        total_samples = len(wav)
        max_chunk_size_samples = max_segment_threshold_s * WAV_SAMPLE_RATE

        for start_sample in range(0, total_samples, max_chunk_size_samples):
            end_sample = min(start_sample + max_chunk_size_samples, total_samples)
            segment = wav[start_sample:end_sample]
            if len(segment) > 0:
                segmented_wavs.append((start_sample, end_sample, segment))

        return segmented_wavs

def save_audio_file(wav: np.ndarray, file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    sf.write(file_path, wav, WAV_SAMPLE_RATE)

def transcribe_audio_qwen(audio_path, api_key, num_threads=4, status_callback=None):
    """
    Transcribe audio using Qwen3-ASR-Flash with parallel processing.
    Returns list of segments: [{'start': float, 'end': float, 'text': str}]
    """
    get_client(api_key)
    
    if status_callback:
        status_callback(phase="initializing", segments_count=0)

    # Load Audio
    wav = load_audio(audio_path)
    duration = len(wav) / WAV_SAMPLE_RATE
    
    # VAD / Segmentation
    if duration >= 180: # 3 mins
        if status_callback:
            status_callback(phase="segmenting", segments_count=0)
        worker_vad_model = load_silero_vad(onnx=True)
        wav_list = process_vad(wav, worker_vad_model)
    else:
        wav_list = [(0, len(wav), wav)]

    # Prepare chunks
    tmp_dir = os.path.join(os.path.expanduser("~"), ".qwen3-asr-cache")
    os.makedirs(tmp_dir, exist_ok=True)
    
    wav_name = os.path.basename(audio_path)
    wav_dir_name = os.path.splitext(wav_name)[0]
    save_dir = os.path.join(tmp_dir, wav_dir_name)
    
    wav_path_list = []
    for idx, (_, _, wav_data) in enumerate(wav_list):
        wav_path = os.path.join(save_dir, f"{wav_name}_{idx}.wav")
        save_audio_file(wav_data, wav_path)
        wav_path_list.append(wav_path)

    # Parallel Transcription
    qwen_asr = QwenASR(model="qwen3-asr-flash")
    results = [] # (idx, text)
    
    if status_callback:
        status_callback(phase="transcribing", segments_count=0)

    completed_count = 0
    total_chunks = len(wav_path_list)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_dict = {
            executor.submit(qwen_asr.asr, wav_path): idx
            for idx, wav_path in enumerate(wav_path_list)
        }
        
        for future in concurrent.futures.as_completed(future_dict):
            idx = future_dict[future]
            try:
                _, recog_text = future.result()
                results.append((idx, recog_text))
            except Exception as e:
                print(f"Chunk {idx} failed: {e}")
                results.append((idx, "")) # Append empty string to keep order if needed, or just ignore? 
                # Better to have empty string to avoid shifting indices if we map back to time
            
            completed_count += 1
            if status_callback:
                status_callback(phase="transcribing", segments_count=completed_count) # Note: this is chunks count, not final segments

    # Cleanup
    try:
        shutil.rmtree(save_dir)
    except:
        pass

    # Assemble Result
    results.sort(key=lambda x: x[0])
    
    # Convert to standard segment format
    # Qwen ASR returns a single text block for the chunk.
    # We map it to the time range of the chunk.
    final_segments = []
    for idx, text in results:
        if not text.strip():
            continue
        start_sample = wav_list[idx][0]
        end_sample = wav_list[idx][1]
        start_time = start_sample / WAV_SAMPLE_RATE
        end_time = end_sample / WAV_SAMPLE_RATE
        
        final_segments.append({
            "start": start_time,
            "end": end_time,
            "text": text.strip()
        })

    if status_callback:
        status_callback(phase="transcribed", segments_count=len(final_segments))

    return final_segments

# --- LLM Components ---

def optimize_transcript_qwen(text, api_key):
    """
    Use Qwen to fix typos, punctuation, and paragraphing.
    """
    get_client(api_key)
    
    system_prompt = (
        "You are a professional editor. Your task is to optimize the following spoken text into "
        "clear, written text. \n"
        "Rules:\n"
        "1. Correct typos and homophone errors based on context.\n"
        "2. Remove filler words (e.g., 'um', 'ah', 'like', 'you know', '这个', '那个', '呃').\n"
        "3. Fix punctuation and sentence structure.\n"
        "4. Keep the original meaning and tone.\n"
        "5. Organize the text into logical paragraphs.\n"
        "6. Output ONLY the optimized text, no explanations."
    )
    
    # Qwen-plus is a good balance.
    response = Generation.call(
        model=Generation.Models.qwen_plus,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': text}
        ],
        result_format='message'
    )
    
    if response.status_code == HTTPStatus.OK:
        return response.output.choices[0]['message']['content']
    else:
        raise Exception(f"Qwen API Error: {response.code} - {response.message}")

def summarize_transcript_qwen(text, api_key):
    """
    Generate a structured summary of the transcript using Qwen.
    """
    get_client(api_key)
    
    system_prompt = (
        "You are an expert summarizer. Create a comprehensive summary of the provided text.\n"
        "Structure:\n"
        "1. **One-sentence Summary**: The core message.\n"
        "2. **Key Points**: Bullet points of main ideas.\n"
        "3. **Detailed Breakdown**: Logical sections with headers.\n"
        "4. **Conclusion**: Final thoughts or takeaways.\n"
        "Output in Markdown format."
    )
    
    response = Generation.call(
        model=Generation.Models.qwen_plus,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': text}
        ],
        result_format='message'
    )
    
    if response.status_code == HTTPStatus.OK:
        return response.output.choices[0]['message']['content']
    else:
        raise Exception(f"Qwen API Error: {response.code} - {response.message}")
