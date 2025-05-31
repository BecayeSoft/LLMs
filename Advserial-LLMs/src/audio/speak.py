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


def stream_message(
        history, 
        chatbot_instance, 
        user_message, 
        is_socrates=True, 
        delay=0.01, 
        voice=None
    ):
    """
    Stream message using real API streaming with typing effect.
    Works with the modified chatbot classes that yield chunks.
    """
    # Add initial empty message to history
    if is_socrates:
        history.append({"role": "assistant", "content": ""})
    else:
        history.append({"role": "user", "content": ""})

    current_message = ""
    
    # Stream from API
    try:
        for chunk in chatbot_instance.stream(user_message):
            if chunk:
                # Add delay for typing effect (optional, since API already has natural delays)
                if delay > 0:
                    time.sleep(delay)
                
                current_message += chunk
                history[-1]["content"] = current_message
                yield history, None
    except Exception as e:
        # Handle streaming errors gracefully
        error_msg = f"[Streaming error: {str(e)}]"
        current_message += error_msg
        history[-1]["content"] = current_message
        yield history, None

    # Generate audio after streaming is complete
    if voice == 'kokoro':
        audio = speak_with_kokoro(current_message, is_socrates)
    else:
        audio = speak_with_openai_tts(current_message, is_socrates)
    
    yield history, audio
    
    # Wait for audio to finish
    audio_duration = estimate_audio_duration(audio)
    time.sleep(audio_duration)

    return current_message


def stream_first_message(
        history, 
        message, 
        is_socrates=True, 
        delay=0.03, 
        voice=None
    ):
    """
    Simple typing effect for pre-written messages (like the opening message).
    """
    if is_socrates:
        history.append({"role": "assistant", "content": ""})
    else:
        history.append({"role": "user", "content": ""})

    current_message = ""
    for char in message:
        current_message += char
        history[-1]["content"] = current_message
        yield history, None
        time.sleep(delay)
    
    # Generate audio after typing is complete
    if voice == 'kokoro':
        audio = speak_with_kokoro(current_message, is_socrates)
    else:
        audio = speak_with_openai_tts(current_message, is_socrates)

    yield history, audio
    
    # Wait for audio to finish
    audio_duration = estimate_audio_duration(audio)
    time.sleep(audio_duration)

