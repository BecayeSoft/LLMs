import sys
from pathlib import Path

# Dynamically resolve the src directory
src_path = (Path(__file__).resolve().parent / "src").resolve()
if str(src_path) not in sys.path: sys.path.append(str(src_path))

import time
# from audio.kokoro import speak_with_kokoro
# from audio.openai_tts import speak_with_openai_tts
from audio.test_speak import speak_message


# def init_conversation(
#         history, 
#         message, 
#         is_socrates=True, 
#         delay=0.03
#     ):
#     """
#     Simple typing effect for pre-written messages (like the opening message).
#     """
#     role = "assistant" if is_socrates else "user"
#     history.append({"role": role, "content": ""})

#     current_message = ""
#     for char in message:
#         current_message += char
#         history[-1]["content"] = current_message
#         print(f"Streaming first message chunk: {current_message}")
#         yield history
#         time.sleep(delay)
    
#     yield history



# def stream_message(
#         history, 
#         chatbot_instance, 
#         user_message, 
#         is_socrates=True, 
#         # delay=0.01
#     ):
#     """
#     Stream message using real API streaming with typing effect.
#     Works with the modified chatbot classes that yield chunks.
#     """
#     # Add initial empty message to history
#     role = "assistant" if is_socrates else "user"
#     history.append({"role": role, "content": ""})

#     reply = ""
    
#     # Stream from API
#     try:
#         for chunk in chatbot_instance.stream(user_message):
#             # # Add delay for typing effect
#             # time.sleep(delay)   
#             reply += chunk
#             history[-1]["content"] = reply
#             print(f"Streaming chunk: {chunk}")
#             yield history
#     except Exception as e:
#         # Handle streaming errors gracefully
#         error_msg = f"[Streaming error: {str(e)}]"
#         reply += error_msg
#         history[-1]["content"] = reply
    
#     yield history

#     return reply



def stream_with_typing(
        history, 
        message, 
        is_socrates=True, 
        delay=0.03
    ):
    """
    Streams a message to the history with optional typing effect.
    - `message_source`: Either a string (for pre-written message) or an object with `.stream(message)` method.
    - `simulated`: If True, treat message_source as a plain string (simulate typing).
    """
    role = "assistant" if is_socrates else "user"
    history.append({"role": role, "content": ""})

    content = ""
    # Simulate character-by-character typing
    for char in message:
        content += char
        history[-1]["content"] = content
        yield history
        time.sleep(delay)
    
    # print(f"test_stream -> history: {history}")

    yield history
    return content



def stream_with_typing_and_audio(
        history, 
        message,
        is_socrates: bool = True,
        delay: float = 0.01,
        voice: str | None = None
    ):
    """
    Yield (updated_chat_history, audio_or_None) at every character.
    On the final yield the second element is a (sample_rate, np.array) tuple
    so a gr.Audio component can autoplay it.
    """
    role = "assistant" if is_socrates else "user"
    history.append({"role": role, "content": ""})

    content = ""
    for char in message:
        content += char
        history[-1]["content"] = content
        yield history, None           # update textbox only
        time.sleep(delay)             # micro-pause for typing illusion

    print(f"stream_with_typing_and_audio -> history: {message}")

    # ----- finished typing → make speech -----
    audio, duration = speak_message(message, is_socrates=is_socrates, voice=voice)

    yield history, audio              # final frame includes the wav

    # Halt until the audio finishes
    time.sleep(duration)


