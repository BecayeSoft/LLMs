import os
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
from IPython.display import Markdown, display
from typing import List

load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


# ---------------- Base class for chatbots ----------------------------- #
class BaseChatbot:
    """
    Base class for chatbots, managing shared conversation state and history logic for both providers.

    Attributes:
        model (str): The model name used by the chatbot.
        system_prompt (str): The system prompt for the chatbot's behavior.
        history (List[dict]): The conversation history as a list of message dicts.
    """
    def __init__(
        self,
        model_name: str,
        system_prompt: str | None = None,
        history: List[dict] | None = None,
    ) -> None:
        """
        Initialize the chatbot with model name, system prompt, and optional history.

        Args:
            model_name (str): The name of the model to use.
            system_prompt (str | None, optional): The system prompt for the chatbot. Defaults to a helpful assistant prompt.
            history (List[dict] | None, optional): Initial conversation history. Defaults to None.
        """
        self.model = model_name
        self.system_prompt = system_prompt or "You are a helpful assistant."
        # copy the list so callers can safely share the same turns
        self.history: List[dict] = (history or []).copy()

    def _update_history(self, user_message: str, assistant_reply: str) -> None:
        """
        Add the user message and assistant reply to the conversation history.

        Args:
            user_message (str): The user's message.
            assistant_reply (str): The assistant's reply.
        """
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": assistant_reply})


    def reset(self) -> None:
        """
        Reset the conversation history to an empty list.
        """
        self.history = []


# ---------- Claude AKA Socrates -------------- #
class Socrates(BaseChatbot):
    """Similar implementation to TestEris but for Socrates/Claude."""
    
    def __init__(
        self,
        system_prompt: str,
        model_name: str = "claude-3-5-haiku-latest",
        history: List[dict] | None = None,
    ) -> None:
        """
        Initialize Socrates chatbot with model, prompt, and optional history.

        Args:
            model_name (str, optional): The model name for Claude. Defaults to "claude-3-5-haiku-latest".
            system_prompt (str): The system prompt for Socrates.
            history (List[dict] | None, optional): Initial conversation history. Defaults to None.
        """
        super().__init__(
            model_name,
            system_prompt,
            history,
        )
        if not ANTHROPIC_API_KEY:
            raise ValueError("Missing ANTHROPIC_API_KEY env var.")
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    def _build_messages(self, user_message: str) -> List[dict]:
        """
        Build the message list for the API call, appending the user message to the current history.

        Args:
            user_message (str): The user's message.
        Returns:
            List[dict]: The list of messages for the API call.
        """
        return self.history + [{"role": "user", "content": user_message}]

    def stream(self, user_message: str):
        """
        Stream the response from Claude and update the conversation history.

        Args:
            user_message (str): The user's message to send.
        Yields:
            str: The next chunk of the assistant's reply as it streams in.
        Returns:
            str: The full reply after streaming is complete.
        """
        msgs = self._build_messages(user_message)
        stream = self.client.messages.stream(
            model=self.model,
            system=self.system_prompt,
            messages=msgs,
            max_tokens=256,
            temperature=0.7,
        )
        # disp = display(Markdown(""), display_id=True)
        reply = ""
        try:
            with stream as s:
                for chunk in s.text_stream:
                    if chunk:
                        reply += chunk
                        yield chunk
                        # disp.update(Markdown(reply))
        except Exception as exc:
            error_msg = f"\n[Claude streaming error: {exc}]"
            reply += error_msg
            yield error_msg
            # disp.update(Markdown(reply))

        # Update history 
        self._update_history(user_message,  reply)

        
        return reply


# ---------------- GPT AKA Eris ----------------- # 
class Eris(BaseChatbot):
    """
    Goddess of discord backed by GPT (OpenAI).
    Recommended models: 'gpt-4.1-nano', 'gpt-4.1-mini', 'gpt-4.1', 'gpt-4o-mini'.
    """

    def __init__(
        self,
        system_prompt: str,
        model_name: str = "gpt-4.1-nano",
        history: List[dict] | None = None,
    ) -> None:
        """
        Initialize Eris chatbot with model, prompt, and optional history.

        Args:
            model_name (str, optional): The model name for GPT. Defaults to "gpt-4.1-nano".
            system_prompt (str): The system prompt for Eris.
            history (List[dict] | None, optional): Initial conversation history. Defaults to None.
        """
        super().__init__(
            model_name,
            system_prompt,
            history,
        )
        if not OPENAI_API_KEY:
            raise ValueError("Missing OPENAI_API_KEY env var.")
        self.client = OpenAI()

    def _build_messages(self, user_message: str) -> List[dict]:
        """
        Build the message list for the API call, including the system prompt, history, and user message.

        Args:
            user_message (str): The user's message.
        Returns:
            List[dict]: The list of messages for the API call.
        """
        msgs = [{"role": "system", "content": self.system_prompt}]
        msgs.extend(self.history)
        msgs.append({"role": "user", "content": user_message})
        return msgs

    def stream(self, user_message: str):
        """
        Stream the response from OpenAI GPT and update the conversation history.

        Args:
            user_message (str): The user's message to send.
        Yields:
            str: The next chunk of the assistant's reply as it streams in.
        Returns:
            str: The full reply after streaming is complete.
        """
        msgs = self._build_messages(user_message)
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=msgs,
            stream=True,
        )
        # disp = display(Markdown(""), display_id=True)
        reply = ""
        try:
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:  # Only yield non-empty deltas
                    reply += delta
                    yield delta
                    # disp.update(Markdown(reply))
        except Exception as exc:
            error_msg = f"\n[OpenAI streaming error: {exc}]"
            reply += error_msg
            yield error_msg
            # disp.update(Markdown(reply))

        # Update history 
        self._update_history(user_message, reply)
        
        return reply

