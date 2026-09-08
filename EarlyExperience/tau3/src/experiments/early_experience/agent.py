"""Same policy prompt for all baselines; hide private reflection from the user."""

import re

from tau2.agent.llm_agent import LLMAgent

from .core import policy_prompt


def strip_reflection(content):
    """Remove complete private blocks; fail closed on malformed model output."""
    if content is None:
        return None
    clean = re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()
    if "<think>" in clean or "</think>" in clean:
        raise ValueError("Unclosed or malformed private reflection")
    return clean or None


class EarlyExperienceAgent(LLMAgent):
    """Serve an IL, IWM->IL, or SR checkpoint via the native LLM provider layer."""

    @property
    def system_prompt(self):
        return policy_prompt(self.domain_policy)

    def generate_next_message(self, message, state):
        response = self._generate_next_message(message, state)
        response.content = strip_reflection(response.content)
        response.validate()
        state.messages.append(response)
        return response, state


def create_agent(tools, domain_policy, llm, llm_args=None, **kwargs):
    """Factory with native registry signature."""
    return EarlyExperienceAgent(tools, domain_policy, llm, llm_args)
