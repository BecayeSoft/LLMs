import sys
from pathlib import Path

# Dynamically resolve the src directory
src_path = (Path(__file__).resolve().parent / "src").resolve()
if str(src_path) not in sys.path: sys.path.append(str(src_path))

import time
from audio.speak import speak_message



def stream_with_typing(
        history: list[dict],
        message: str,
        is_socrates: bool = True,
        delay: float = 0.03
    ):
    """
    Streams a message to the chat history with an optional typing effect.

    Args:
        history (list[dict]): The chat history, a list of message dicts with 'role' and 'content'.
        message (str): The message to stream character by character.
        is_socrates (bool, optional): If True, sets the role to 'assistant', else 'user'. Defaults to True.
        delay (float, optional): Delay in seconds between each character to simulate typing. Defaults to 0.03.

    Yields:
        list[dict]: The updated chat history after each character is added.

    Returns:
        str: The full message content after streaming is complete.
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
    
    yield history
    return content



def stream_with_typing_and_audio(
        history: list[dict],
        message: str,
        is_socrates: bool = True,
        delay: float = 0.01,
        voice: str | None = None
    ):
    """
    Streams a message to the chat history with a typing effect and generates audio at the end.

    Args:
        history (list[dict]): The chat history, a list of message dicts with 'role' and 'content'.
        message (str): The message to stream character by character.
        is_socrates (bool, optional): If True, sets the role to 'assistant', else 'user'. Defaults to True.
        delay (float, optional): Delay in seconds between each character to simulate typing. Defaults to 0.01.
        voice (str | None, optional): The voice to use for audio synthesis. Defaults to None.

    Yields:
        tuple[list[dict], None | tuple]:
            - At each character: (updated chat history, None)
            - At the end: (updated chat history, (sample_rate, np.array)) for audio playback
    """
    role = "assistant" if is_socrates else "user"
    history.append({"role": role, "content": ""})

    content = ""
    for char in message:
        content += char
        history[-1]["content"] = content
        yield history, None           # update textbox only
        time.sleep(delay)

    # ----- finished typing → make speech -----
    audio, duration = speak_message(content, is_socrates=is_socrates, voice=voice)

    yield history, audio              # yield chat history and audio

    # Halt until the audio finishes
    time.sleep(duration + 0.5)
