import sys
sys.path.append("..")

import streamlit as st
import time
import random
from src.chatbot import Socrates, Eris 
from src.speak import stream_first_message, stream_message


st.set_page_config(page_title="⚔️ Socrates vs. Eris", layout="centered")

st.markdown("## ⚔️ Socrates vs. Eris")
st.markdown(
    "Dive into a battle of minds where **Socrates**, the calm philosopher, asks piercing questions "
    "to uncover truth — and **Eris**, the goddess of discord, responds with sarcastic wit and playful contradiction.  \n"
    "Choose a topic and watch them clash in a brief, animated debate."
)

# Inputs
col1, col2 = st.columns([1, 3])
with col1:
    rounds = st.number_input("Rounds", min_value=1, value=4, step=1)
with col2:
    topic = st.text_input("Debate Topic", placeholder="e.g. What is freedom?")

# Placeholder for chat and audio
chat_placeholder = st.container()
audio_placeholder = st.empty()

# Disable button if inputs are invalid
start_disabled = not topic.strip() or rounds < 1
start_btn = st.button("Start Debate", disabled=start_disabled)

# Streaming debate function
def run_streaming_debate(chat_placeholder, n_rounds, topic, speed=0.01):
    socrates = Socrates()
    eris = Eris()
    history = []

    eris_options = [
        f"Ready to get schooled about '{topic}'?, old man?",
        f"Today's topic is: '{topic}'. This should be easy—your logic is about as sharp as a wet napkin.",
        f"Hope you're prepared to lose spectacularly, Socrates. Let us talk about: '{topic}'",
        f"Hey Socrates. Wanna argue about: '{topic}'? It will be fun to see you cry once you have lost.",
    ]
    eris_reply = random.choice(eris_options)

    # First message from Eris
    for update, _ in stream_first_message(history, eris_reply, is_socrates=False, delay=speed):
        chat_placeholder.chat_message("user").markdown(update[-1]["content"])
        time.sleep(speed)

    for i in range(n_rounds - 1):
        if i % 2 == 0:
            socrates.history = eris.history.copy()
            stream = stream_message(history, socrates, eris_reply, is_socrates=True, delay=speed)
        else:
            eris.history = socrates.history.copy()
            stream = stream_message(history, eris, soc_reply, is_socrates=False, delay=speed)

        audio_result = None
        for updated_history, audio in stream:
            last_msg = updated_history[-1]
            role = last_msg["role"]
            content = last_msg["content"]
            chat_placeholder.chat_message("assistant" if role == "assistant" else "user").markdown(content)
            audio_result = audio  # capture last audio
            time.sleep(speed)

        if i % 2 == 0:
            soc_reply = updated_history[-1]["content"]
        else:
            eris_reply = updated_history[-1]["content"]

    return history, audio_result

# Run if button is clicked
if start_btn:
    chat_placeholder.empty()  # Clear previous messages
    history, audio = run_streaming_debate(chat_placeholder, rounds, topic)

    if audio:
        audio_placeholder.audio(audio, format="audio/wav")
