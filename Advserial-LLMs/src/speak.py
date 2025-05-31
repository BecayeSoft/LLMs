import sys
sys.path.append("..")

import time
import re
import numpy as np
from kokoro import KPipeline


pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')


def clean_message(message):
    """
    Clean up AI-generated text for TTS:
    - Replace double newlines with a single newline.
    - Remove markdown or formatting characters (like **bold**, *italic*, etc.).
    - Strip markdown/code artifacts (e.g., ```code blocks```, >>> prompts).
    - Remove excess whitespace and trim.
    """
    if not isinstance(message, str):
        return ""

    # Replace triple backticks and markdown-style code blocks
    message = re.sub(r"```.*?```", "", message, flags=re.DOTALL)

    # Remove markdown formatting characters
    message = re.sub(r"\*\*|\*", "", message)        # bold/italic
    message = re.sub(r"_", "", message)              # underscore emphasis
    message = re.sub(r"#+", "", message)             # markdown headers like ###

    # Remove '>>>', often used in prompt examples
    message = message.replace(">>>", "")

    # Replace double newlines with three dots
    message = re.sub(r"\n{2,}", "... ", message)

    # Replace single newlines with a space
    message = re.sub(r"\n", " ", message)

    # Remove stray symbols like •, →, etc.
    message = re.sub(r"[•→]", "", message)

    # Replace multiple spaces with a single space
    message = re.sub(r" {2,}", " ", message)

    # Remove leading/trailing whitespace
    return message.strip()


def estimate_audio_duration(audio_tuple):
    """
    Estimate audio duration from (sample_rate, audio_array) tuple.
    """
    if audio_tuple is None:
        return 0
    sample_rate, audio_array = audio_tuple
    return len(audio_array) / sample_rate


# def speak_with_voice(text, is_socrates=True):
#     """
#     Generate speech with different voices for each character.
#     Socrates gets a masculine voice while Eris gets a feminine voice. 

#     Voices can be found at: https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md.
#     """
#     voice = "am_adam" if is_socrates else "af_heart"
#     generator = pipeline(text, voice=voice)
#     for i, (_, _, audio_tensor) in enumerate(generator):
#         audio_np = audio_tensor.numpy()
#         return (24000, audio_np)

def speak_with_voice(text, is_socrates=True):
    """
    Generate speech with different voices for each character.
    Socrates gets a masculine voice while Eris gets a feminine voice. 

    Voices can be found at: https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md.
    """
    voice = "am_adam" if is_socrates else "af_heart"
    generator = pipeline(text, voice=voice)
    
    # Collect all audio chunks
    # Because kokoro may generate several chunks for a single text.
    audio_chunks = []
    for i, (_, _, audio_tensor) in enumerate(generator):
        audio_np = audio_tensor.numpy()
        audio_chunks.append(audio_np)
    
    # Concatenate all chunks into a single audio array
    if audio_chunks:
        complete_audio = np.concatenate(audio_chunks)
        return (24000, complete_audio)
    else:
        # Return empty audio if no chunks were generated
        return (24000, np.array([]))


def stream_message(history, chatbot_instance, user_message, is_socrates=True, delay=0.01):
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
    # if current_message.strip():  # Only generate audio if there's actual content
    current_message = clean_message(current_message)
    audio = speak_with_voice(current_message, is_socrates)
    yield history, audio
    
    # Wait for audio to finish
    audio_duration = estimate_audio_duration(audio)
    time.sleep(audio_duration)

    return current_message


def stream_first_message(history, message, is_socrates=True, delay=0.03):
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
    current_message = clean_message(current_message)
    audio = speak_with_voice(message, is_socrates)
    yield history, audio
    
    # Wait for audio to finish
    audio_duration = estimate_audio_duration(audio)
    time.sleep(audio_duration)
