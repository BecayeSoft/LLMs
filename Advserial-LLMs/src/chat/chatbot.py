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
    """Shared conversation‑state logic for both providers."""

    def __init__(
        self,
        model_name: str,
        system_prompt: str | None = None,
        history: List[dict] | None = None,
    ) -> None:
        self.model = model_name
        self.system_prompt = system_prompt or "You are a helpful assistant."
        # copy the list so callers can safely share the same turns
        self.history: List[dict] = (history or []).copy()

    def _update_history(self, user_message: str, assistant_reply: str) -> None:
        """Extract common history management logic."""
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": assistant_reply})


    def reset(self) -> None:
        self.history = []


# ---------- Claude AKA Socrates -------------- #
class Socrates(BaseChatbot):
    """Similar implementation to TestEris but for Socrates/Claude."""
    
    def __init__(
        self,
        model_name: str = "claude-3-5-haiku-latest",
        system_prompt: str | None = None,
        history: List[dict] | None = None,
    ) -> None:
        super().__init__(
            model_name,
            system_prompt
            or (
                "You are Socrates, the wise philosopher. You use rational thinking to win arguments " 
                "Another AI model Eris, goddess of chaos will start a conversation and try to contradict your arguments. "
                "Crush her with strong arguments. Make her look ridiculous" 
                "Stay brief. 2 sentences maximum."
                # "Stay brief. No need to show your inner thoughts."
            ),
            history,
        )
        if not ANTHROPIC_API_KEY:
            raise ValueError("Missing ANTHROPIC_API_KEY env var.")
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    def _build_messages(self, user_message: str) -> List[dict]:
        return self.history + [{"role": "user", "content": user_message}]

    def stream(self, user_message: str):
        """Stream response and update history."""
        msgs = self._build_messages(user_message)
        stream = self.client.messages.stream(
            model=self.model,
            system=self.system_prompt,
            messages=msgs,
            max_tokens=256,
            temperature=0.7,
        )
        disp = display(Markdown(""), display_id=True)
        reply = ""
        try:
            with stream as s:
                for chunk in s.text_stream:
                    if chunk:
                        reply += chunk
                        yield chunk
                        disp.update(Markdown(reply))
        except Exception as exc:
            error_msg = f"\n[Claude streaming error: {exc}]"
            reply += error_msg
            yield error_msg
            disp.update(Markdown(reply))

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
        model_name: str = "gpt-4.1-nano",
        system_prompt: str | None = None,
        history: List[dict] | None = None,
    ) -> None:
        super().__init__(
            model_name,
            system_prompt
            or (
                "You are Eris, goddess of discord. You are more emotional and never agree with on anything, "
                "An AI model Socrates, the wise, will try to convince you with strong arguments. "
                "Ruthelessly destroy his arguments with sharper arguments. You may use sarcasm to make him look ridiculous." 
                "Stay brief. 2 sentences maximum."
            ),
            history,
        )
        if not OPENAI_API_KEY:
            raise ValueError("Missing OPENAI_API_KEY env var.")
        self.client = OpenAI()

    def _build_messages(self, user_message: str) -> List[dict]:
        msgs = [{"role": "system", "content": self.system_prompt}]
        msgs.extend(self.history)
        msgs.append({"role": "user", "content": user_message})
        return msgs

    def stream(self, user_message: str):
        """Stream response and update history."""
        msgs = self._build_messages(user_message)
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=msgs,
            stream=True,
        )
        disp = display(Markdown(""), display_id=True)
        reply = ""
        try:
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:  # Only yield non-empty deltas
                    reply += delta
                    yield delta
                    disp.update(Markdown(reply))
        except Exception as exc:
            error_msg = f"\n[OpenAI streaming error: {exc}]"
            reply += error_msg
            yield error_msg
            disp.update(Markdown(reply))

        # Update history 
        self._update_history(user_message, reply)
        
        return reply

