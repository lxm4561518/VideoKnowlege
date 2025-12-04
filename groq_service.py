import os
from groq import Groq

def get_client(api_key):
    if not api_key:
        raise ValueError("Groq API Key is required")
    return Groq(api_key=api_key)

def transcribe_audio_groq(audio_path, api_key, prompt=None):
    """
    Transcribe audio using Groq Whisper API.
    Returns a list of segments similar to faster-whisper format.
    """
    client = get_client(api_key)
    
    with open(audio_path, "rb") as file:
        # Groq Whisper API currently returns a single text or verbose_json
        # verbose_json includes segments
        transcription = client.audio.transcriptions.create(
            file=(os.path.basename(audio_path), file.read()),
            model="whisper-large-v3",
            prompt=prompt,
            response_format="verbose_json",
            language="zh" # Default to Chinese for Bilibili, or make it configurable if needed
        )
    
    # Convert Groq response to our standard segment format
    # Groq verbose_json has 'segments' list
    segments = []
    for seg in transcription.segments:
        segments.append({
            "start": seg['start'],
            "end": seg['end'],
            "text": seg['text']
        })
        
    return segments

def optimize_transcript_groq(text, api_key):
    """
    Use LLM to fix typos, punctuation, and paragraphing.
    """
    client = get_client(api_key)
    
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
    
    completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ],
        temperature=0.5,
        max_tokens=8192,
        top_p=1,
        stream=False,
        stop=None,
    )
    
    return completion.choices[0].message.content

def summarize_transcript_groq(text, api_key):
    """
    Generate a structured summary of the transcript.
    """
    client = get_client(api_key)
    
    system_prompt = (
        "You are an expert summarizer. Create a comprehensive summary of the provided text.\n"
        "Structure:\n"
        "1. **One-sentence Summary**: The core message.\n"
        "2. **Key Points**: Bullet points of main ideas.\n"
        "3. **Detailed Breakdown**: Logical sections with headers.\n"
        "4. **Conclusion**: Final thoughts or takeaways.\n"
        "Output in Markdown format."
    )
    
    completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ],
        temperature=0.5,
        max_tokens=8192,
        top_p=1,
        stream=False,
        stop=None,
    )
    
    return completion.choices[0].message.content
