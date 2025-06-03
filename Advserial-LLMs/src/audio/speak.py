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
    Estimate the duration of an audio clip from a (sample_rate, audio_array) tuple.

    Args:
        audio_tuple (tuple or None): A tuple containing (sample_rate: int, audio_array: np.ndarray),
            or None if no audio is present.

    Returns:
        float: The estimated duration of the audio in seconds. Returns 0 if audio_tuple is None.
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
    Generate speech audio for a message using the selected TTS backend.

    Args:
        message (str): The message to convert to speech.
        is_socrates (bool, optional): If True, use Socrates voice/persona. Defaults to True.
        voice (str or None, optional): The TTS backend to use ('kokoro' or None for OpenAI TTS). Defaults to None.

    Returns:
        tuple: (audio, duration)
            audio: (sample_rate, np.ndarray) tuple representing the audio.
            duration: float, estimated duration of the audio in seconds.
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

