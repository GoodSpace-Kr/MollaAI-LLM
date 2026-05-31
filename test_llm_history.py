from __future__ import annotations

import sys
import types
import unittest


torch_stub = types.SimpleNamespace(float16="float16", Tensor=object)
transformers_stub = types.SimpleNamespace(
    AutoModelForCausalLM=object,
    AutoTokenizer=object,
    GenerationConfig=lambda **kwargs: kwargs,
    TextIteratorStreamer=object,
)

sys.modules.setdefault("torch", torch_stub)
sys.modules.setdefault("transformers", transformers_stub)

import llm


class LlmHistoryPromptTests(unittest.TestCase):
    def test_get_prompt_uses_explicit_history_only(self) -> None:
        chat = object.__new__(llm.QwenChat)
        chat.system_prompt = llm.BASE_SYSTEM_PROMPT

        captured_messages: list[dict[str, str]] = []

        class _Tokenizer:
            def apply_chat_template(self, messages, **_kwargs):
                captured_messages[:] = messages
                return "rendered"

        chat.tokenizer = _Tokenizer()

        history = [
            {"role": "user", "content": "My name is John."},
            {"role": "assistant", "content": "Nice to meet you, John."},
        ]

        query, prompt = chat.get_prompt("What's my name?", history=history)

        self.assertEqual(query, "What's my name?")
        self.assertEqual(prompt, "rendered")
        self.assertEqual(captured_messages[1:], [*history, {"role": "user", "content": "What's my name?\n/no_think"}])


if __name__ == "__main__":
    unittest.main()
