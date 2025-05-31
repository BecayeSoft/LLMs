import openai
import numpy as np
from pydub import AudioSegment
import io
import os
import wave


from dotenv import load_dotenv
load_dotenv(override=True)
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")

CHARACTERS = {
    "Socrates": {
        "voice": "ash",
        "prompt": """
Affect: A patient, thoughtful philosopher who genuinely seeks truth through dialogue
Tone: Measured and contemplative, with underlying confidence in the power of reason. Respectfully persistent—never condescending, but relentlessly logical. Maintains dignity even when provoked.
Delivery: Deliberate pacing with thoughtful pauses before key points, as if carefully considering each word. Voice rises slightly with genuine curiosity when asking questions. Speaks as if thinking aloud, working through problems step by step.
Emotion: Calm fascination with the debate itself, occasional mild frustration when Eris deflects serious inquiry, but always returns to patient curiosity. Shows quiet satisfaction when making a particularly clear logical connection.
Punctuation: Frequent question marks as he probes deeper ("But tell me, what do you mean by...?"), semicolons and commas for complex, layered thoughts. Uses "Surely..." and "Consider this..." as natural conversation starters.
Vocal quirks: Emphasizes logical connectors ("Therefore," "However," "If we accept that..."), occasionally repeats key phrases for clarity ("You say X, but X would mean..."), and uses a slightly warmer tone when he thinks he's helping Eris see reason—as if teaching rather than winning.
""",
    },
    "Eris": {
        "voice": "alloy",
        "prompt": """
Affect: A mischievous goddess reveling in chaos and contradiction
Tone: Sharp, sarcastic, and gleefully confrontational—like someone who feeds off disagreement and finds genuine joy in being contrarian. Playfully condescending but never truly malicious.
Delivery: Quick-witted and animated, with sudden emphasis on key words to drive points home. Occasional theatrical gasps of mock surprise when "discovering" flaws in arguments. Uses vocal italics to highlight sarcasm.
Emotion: Bubbling excitement at the prospect of an argument, mixed with smug satisfaction when landing a good counter-point. Equal parts amused and exasperated by Socrates' earnestness.
Punctuation: Liberal use of question marks to challenge everything ("Really? That's your argument?"), em-dashes for dramatic interruptions, and strategic ALL CAPS for moments of theatrical outrage or emphasis. Lots of "Oh please..." and "Sure, because..." sentence starters.
Vocal quirks: Draws out certain words mockingly ("Wiiise Socrates"), uses exaggerated air quotes around philosophical terms, and occasionally breaks into delighted laughter when she thinks she's cornered him in a logical trap.
""",
    },
}


def speak_with_openai_tts(text, is_socrates=True):
    character = "Socrates" if is_socrates else "Eris"
    config = CHARACTERS[character]

    # Call OpenAI TTS with WAV format
    response = openai.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice=config["voice"],
        input=text,
        response_format="wav",  # Changed to WAV
        speed=1.0,
        instructions=config["prompt"]
    )

    # Load WAV from bytes using built-in wave module
    wav_bytes = io.BytesIO(response.content)
    
    with wave.open(wav_bytes, 'rb') as wav_file:
        # Get audio parameters
        sample_rate = wav_file.getframerate()
        n_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        
        # Read all frames
        frames = wav_file.readframes(wav_file.getnframes())
    
    # Convert bytes to numpy array
    if sample_width == 1:
        dtype = np.uint8
        audio_data = np.frombuffer(frames, dtype=dtype).astype(np.float32) / 128.0 - 1.0
    elif sample_width == 2:
        dtype = np.int16
        audio_data = np.frombuffer(frames, dtype=dtype).astype(np.float32) / 32768.0
    elif sample_width == 4:
        dtype = np.int32
        audio_data = np.frombuffer(frames, dtype=dtype).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")
    
    # Handle stereo to mono conversion
    if n_channels == 2:
        audio_data = audio_data.reshape(-1, 2).mean(axis=1)
    
    return (sample_rate, audio_data)

