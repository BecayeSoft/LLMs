import re
import numpy as np
from kokoro import KPipeline


pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')


def clean_message(message):
    """
    Clean up AI-generated text for Kokoro TTS model:
    - Replace double newlines with three dots.
    - Replace new lines with a space.
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



def speak_with_kokoro(text, is_socrates=True):
    """
    Generate speech with different voices for each character.
    Socrates gets a masculine voice while Eris gets a feminine voice. 

    Voices can be found at: https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md.
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

