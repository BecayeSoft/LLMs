# Exploring GPT-3.5 through OpenAI API

This project explores OpenAI API language models to build a Machine Learning Advisor which provides concise and helpful responses to user prompts.

## Overview

This demo app uses the GPT-3.5 Turbo model from OpenAI to create a virtual machine learning advisor. Users can input prompts, questions, or requests, and the model responds with short and informative answers. The goal is to explore the capabilities of OpenAI's language models and demonstrate how they can be used to provide AI-driven guidance.

## Features

- **Concise Advice**: The model generates brief and to-the-point responses, ensuring clarity and relevance.

- **Conversation History**: The app maintains a conversation history to provide context-aware responses.

- **Punctuation and Truncation**: A built-in function ensures that responses end with proper punctuation and handle token limits.

- **Cost-Efficient**: Only a few tokens are returned to save on API usage costs.

## Getting Started

You can either run the notebook or the `main.py` file to start the app.

1. **Clone the Repository**

   ```
   git clone https://github.com/yourusername/openai-language-model-demo.git
   ```

2. **Install Dependencies**

   ```
   pip install -r requirements.txt
   ```

3. **Setup OpenAI API Key**:

   - Create a `secret.env` file in the project directory.
   - Add your OpenAI API key to the `.env` file:

     ```
     OPENAI_API_KEY=your_api_key_here
     ```

   Replace `your_api_key_here` with your actual OpenAI API key.

4. **Run the Application**:

   - Run the application using Python:

     ```
     python main.py
     ```

   This will start the Gradio app, and you can access it in your web browser.

5. **Interact with the Model**:

   - Enter prompts or questions in the input field and receive responses from the model.
   - Experiment with different types of queries to see how the model responds.
