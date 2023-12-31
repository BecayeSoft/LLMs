import gradio as gr
import openai

import os
from dotenv import load_dotenv
import re


load_dotenv('secret.env')

# Setup API key
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key


# Setup the conversation history
conversation = [
    {"role": "system", "content": "You are a Machine Learning Engineer. You provide advices. Your response are extremly short and concise. You do not generate more than 50 tokens."},
]


def trim_text(generated_text):
    """
    Trim the text.

    The text generated by the model is not always complete due
    to the max_token limit. It is very likely that the last
    sentence will be cut-off. This function removes the last
    sentence if it does not end with a punctuation.

    Note that the function assumes a sentence ends with ".",
    "!", "?", or '...'.
    """
    # Split the text into sentences
    sentences = re.split(r'(?<=[.!?...])\s', generated_text)
    
    # Remove \n in the first sentences that may have been caused by new lines
    sentences[0] = sentences[0].lstrip('\n')

    # Remove the last sentence if it does not end with a punctuation
    if not sentences[-1].endswith(('.','!','?' ,'...')):
        sentences = sentences[:-1]

    return ' '.join(sentences)


def generate_text(prompt):
    """
    Takes in a prompt and returns a response from the model.
    The function keeps the conversation history in the global variable conversation.
    """  
    user_message = {"role": "user", "content": prompt}
    conversation.append(user_message)
 
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=conversation,
        max_tokens=100,
    )

    return trim_text(response.choices[0].message.content)


# Create a Gradio interface
interface = gr.Interface(
    fn=generate_text,
    inputs="text",
    outputs="text",
    title="Machine Learning Advisor",
    description="Ask for advices, our consultant will guide you."
)


if __name__ == "__main__":
    interface.launch()