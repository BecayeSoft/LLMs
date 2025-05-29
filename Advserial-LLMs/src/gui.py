import time
import random
import sys
sys.path.append("..")

from src.chatbot import Socrates, Eris


def clean_message(message):
    """
    Replace double newlines with single newline.
    Useful because Claude like new paragraphs but we lack space.
    """
    return message.replace("\n\n", "\n").strip()


def stream_message(history, message, is_socrates=True, delay=0.01):
    """
    Generator that streams message updates character by character.

    Args:
        history (list): Current chat history.
        message (str): Message to stream.
        is_socrates (bool): True if Socrates is speaking, False if Eris.
        delay (float): Delay between character updates.
    Yields:
        Generator yielding updated chat history.
    """
    partial = ""
    for char in message:
        partial += char
        if is_socrates:
            # Socrates appears on user side
            if history and history[-1]["role"] == "user":
                history[-1]["content"] = partial
            else:
                history.append({"role": "user", "content": partial})
        else:
            # Eris appears on assistant side
            if history and history[-1]["role"] == "assistant":
                history[-1]["content"] = partial
            else:
                history.append({"role": "assistant", "content": partial})
        time.sleep(delay)

        # We use yield to create a streaming effect
        yield history


# Main conversation loop
def adversarial_chat(n_rounds, topic):
    """
    Main conversation loop for Socrates and Eris, streaming updates.
    Args:
        n_rounds (int): Number of rounds of conversation.
        topic (str): Topic for the conversation.
    Yields:
        Generator yielding chat history updates.
    """
    # Initialize Socrates and Eris chatbots
    socrates = Socrates()
    eris = Eris()
    history = []

    # Pause to simulate thinking time
    time.sleep(random.uniform(0.3, 1))

    # Start the conversation with Socrates
    soc_reply = socrates.ask(topic)
    soc_reply = clean_message(soc_reply)
    for update in stream_message(history, soc_reply, is_socrates=True):
        yield update

    # Now alternate turns between Socrates and Eris
    for i in range(n_rounds-1):
        time.sleep(random.uniform(0.6, 1))
        
        # Even rounds: Eris responds, odd rounds: Socrates responds
        if i % 2 == 0:
            eris.history = socrates.history
            eris_reply = eris.ask(soc_reply)
            eris_reply = clean_message(eris_reply)
            for update in stream_message(history, eris_reply, is_socrates=False):
                yield update
        else:
            socrates.history = eris.history
            soc_reply = socrates.ask(eris_reply)
            soc_reply = clean_message(soc_reply)
            for update in stream_message(history, soc_reply, is_socrates=True):
                yield update