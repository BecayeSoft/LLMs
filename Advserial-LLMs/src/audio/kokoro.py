import re
import numpy as np
from kokoro import KPipeline


pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')


def clean_message(message):
    """
    Clean up AI-generated text for the Kokoro TTS model.

    This function performs the following operations:
        - Replaces double newlines with three dots.
        - Replaces single newlines with a space.
        - Removes markdown or formatting characters (e.g., **bold**, *italic*, _underscore_).
        - Strips markdown/code artifacts (e.g., code blocks, >>> prompts).
        - Removes stray symbols (e.g., •, →).
        - Collapses multiple spaces and trims whitespace.

    Args:
        message (str): The message to clean.

    Returns:
        str: The cleaned message suitable for TTS input.
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



def speak_with_kokoro(text, is_socrates=True):
    """
    Generate speech audio using the Kokoro TTS model for either Socrates or Eris persona.

    Args:
        text (str): The text to convert to speech.
        is_socrates (bool, optional): If True, use Socrates' (masculine) voice; if False, use Eris' (feminine) voice. Defaults to True.

    Returns:
        tuple: (sample_rate, audio_array)
            sample_rate (int): The sample rate of the generated audio (always 24000).
            audio_array (np.ndarray): The audio waveform as a numpy array (float32, mono).
    """
    voice = "am_adam" if is_socrates else "af_heart"

    # Clean the text for Kokoro
    text = clean_message(text)

    # Generate the audio
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

