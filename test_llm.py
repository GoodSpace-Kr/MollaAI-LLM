from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


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


class _EmptyStreamer:
    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def __iter__(self):
        return iter(())

    def on_finalized_text(self, _text: str, stream_end: bool = False) -> None:
        return None


class LlmStreamAnswerTests(unittest.TestCase):
    def test_runtime_requirements_do_not_enable_gptq_dependencies(self) -> None:
        requirements = Path(__file__).with_name("requirements.txt").read_text().splitlines()
        active_requirements = {
            line.strip()
            for line in requirements
            if line.strip() and not line.lstrip().startswith("#")
        }

        self.assertNotIn("auto-gptq==0.7.1", active_requirements)
        self.assertNotIn("optimum==1.23.3", active_requirements)

    def test_generation_config_matches_runtime_safe_sampling_defaults(self) -> None:
        if isinstance(llm.gen_config, dict):
            self.assertEqual(llm.gen_config["max_new_tokens"], 80)
            self.assertEqual(llm.gen_config["temperature"], 0.6)
            self.assertEqual(llm.gen_config["top_p"], 0.95)
            self.assertEqual(llm.gen_config["top_k"], 20)
            return
        self.assertEqual(llm.gen_config.temperature, 0.6)
        self.assertEqual(llm.gen_config.max_new_tokens, 80)
        self.assertEqual(llm.gen_config.top_p, 0.95)
        self.assertEqual(llm.gen_config.top_k, 20)

    def test_init_uses_finetuned_tokenizer_by_default(self) -> None:
        tokenizer_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        model_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

        class _TokenizerLoader:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                tokenizer_calls.append((args, kwargs))
                return object()

        class _ModelLoader:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                model_calls.append((args, kwargs))
                return object()

        with patch.object(llm, "AutoTokenizer", _TokenizerLoader), patch.object(llm, "AutoModelForCausalLM", _ModelLoader):
            llm.QwenChat()

        self.assertEqual(tokenizer_calls[0][0][0], llm.MODEL_NAME)
        self.assertEqual(tokenizer_calls[0][1], {"extra_special_tokens": {}})
        self.assertEqual(model_calls[0][0][0], llm.MODEL_NAME)
        self.assertEqual(model_calls[0][1]["torch_dtype"], "auto")

    def test_generation_config_enables_invalid_logits_guards(self) -> None:
        if isinstance(llm.gen_config, dict):
            self.assertTrue(llm.gen_config["remove_invalid_values"])
            self.assertTrue(llm.gen_config["renormalize_logits"])
            return
        self.assertTrue(llm.gen_config.remove_invalid_values)
        self.assertTrue(llm.gen_config.renormalize_logits)

    def test_tokenize_prompt_moves_inputs_to_embedding_device_for_auto_device_map(self) -> None:
        chat = object.__new__(llm.QwenChat)

        class _Tensor:
            def __init__(self) -> None:
                self.calls: list[object] = []

            def to(self, device):
                self.calls.append(device)
                return f"moved:{device}"

        class _Tokenizer:
            def __call__(self, _prompt, return_tensors="pt"):
                return {"input_ids": tensor}

        class _Embeddings:
            class _Weight:
                device = "cuda:0"

            weight = _Weight()

        class _Model:
            device = "cuda:0"
            hf_device_map = {"model.embed_tokens": 0}

            def get_input_embeddings(self):
                return _Embeddings()

        tensor = _Tensor()
        chat.tokenizer = _Tokenizer()
        chat.model = _Model()

        inputs = chat._tokenize_prompt("hello")

        self.assertEqual(inputs["input_ids"], "moved:cuda:0")
        self.assertEqual(tensor.calls, ["cuda:0"])

    def test_get_prompt_retries_without_enable_thinking(self) -> None:
        chat = object.__new__(llm.QwenChat)
        captured_messages: list[dict[str, str]] = []

        class _TokenizerWithoutEnableThinking:
            def apply_chat_template(self, messages, **kwargs):
                captured_messages[:] = messages
                if "enable_thinking" in kwargs:
                    raise TypeError("unexpected keyword argument 'enable_thinking'")
                return "prompt-without-enable-thinking"

        chat.tokenizer = _TokenizerWithoutEnableThinking()

        full_prompt, prompt = chat.get_prompt("hello")

        self.assertEqual(full_prompt, "hello")
        self.assertEqual(prompt, "prompt-without-enable-thinking")
        self.assertEqual(captured_messages[1]["content"], "hello")
        self.assertNotIn("<question>", captured_messages[1]["content"])
        self.assertIn("friendly spoken English conversation partner", captured_messages[0]["content"])

    def test_returns_fallback_when_generate_thread_fails(self) -> None:
        chat = object.__new__(llm.QwenChat)
        chat.tokenizer = object()

        class _BrokenModel:
            device = None

            def generate(self, **_kwargs) -> None:
                raise RuntimeError("generation failed")

        chat.model = _BrokenModel()
        chat.get_prompt = lambda query: ("prompt", "prompt")
        chat._tokenize_prompt = lambda prompt: {"input_ids": object()}

        with patch.object(llm, "TextIteratorStreamer", _EmptyStreamer):
            chunks = list(chat.stream_answer("hello", request_id="test"))

        self.assertEqual(chunks, ["오류가 발생했습니다."])

    def test_stream_filter_removes_thinking_blocks_across_chunks(self) -> None:
        chat = object.__new__(llm.QwenChat)
        state = {"in_think": False}

        chunks = [
            chat._filter_stream_chunk("<think>\nI should not be spoken.", state),
            chat._filter_stream_chunk(" Still hidden.</think>\n안녕하세요!", state),
            chat._filter_stream_chunk(" 무엇을 도와드릴까요?", state),
        ]

        self.assertEqual("".join(chunks), "\n안녕하세요! 무엇을 도와드릴까요?")

    def test_strip_unspeakable_symbols_removes_emoji(self) -> None:
        chat = object.__new__(llm.QwenChat)

        self.assertEqual(
            chat._strip_unspeakable_symbols("😊 Stephen Curry 🏀⚽ is great."),
            " Stephen Curry  is great.",
        )


if __name__ == "__main__":
    unittest.main()
