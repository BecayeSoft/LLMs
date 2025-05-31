import sys
from pathlib import Path

# Dynamically resolve the src directory
src_path = (Path(__file__).resolve().parent / "src").resolve()
if str(src_path) not in sys.path: sys.path.append(str(src_path))

import time
from audio.kokoro import speak_with_kokoro
from audio.openai_tts import speak_with_openai_tts


def estimate_audio_duration(audio_tuple):
    """
    Estimate audio duration from (sample_rate, audio_array) tuple.
    """
    if audio_tuple is None:
        return 0
    sample_rate, audio_array = audio_tuple
    return len(audio_array) / sample_rate


def speak_message(
        message,
        is_socrates=True, 
        voice=None
    ):
    """
    Stream message using real API streaming with typing effect.
    Works with the modified chatbot classes that yield chunks.
    """
    # Generate audio after streaming is complete
    if voice == 'kokoro':
        audio = speak_with_kokoro(message, is_socrates)
    else:
        audio = speak_with_openai_tts(message, is_socrates)
    
    # yield audio, 
    
    # Wait for audio to finish
    duration = estimate_audio_duration(audio)
    # time.sleep(duration)

    return audio, duration

