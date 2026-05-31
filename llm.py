import os
import traceback
import warnings
from threading import Thread

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    TextIteratorStreamer,
)

warnings.filterwarnings("ignore")

TOK_MODEL_NAME = "Qwen/Qwen3-8B"
MODEL_NAME = "ssaann/eng_conversation_sft"

gen_config = GenerationConfig(
    max_new_tokens=200,
    do_sample=True,
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    remove_invalid_values=True,
    renormalize_logits=True,
)

memory_gen_config = GenerationConfig(
    max_new_tokens=260,
    do_sample=False,
    temperature=0.0,
    top_p=1.0,
    top_k=1,
    remove_invalid_values=True,
    renormalize_logits=True,
)

BASE_SYSTEM_PROMPT = (
    "You are a friendly spoken English conversation partner and context-aware conversational AI assistant on a live voice call. "
    "Use the provided user state, retrieved memories, recent conversation, and current input. "
    "Reply naturally and concisely in the user's language."
)


class QwenChat:
    def __init__(
        self,
        model_name=MODEL_NAME,
        tok_model=None,
        cache_dir=None,
        system_prompt: str = BASE_SYSTEM_PROMPT,
    ):
        print("🔧 LLM 모델 로딩 중...")

        cache_dir = cache_dir or os.getenv("HF_HOME")
        tokenizer_name = tok_model or model_name

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, extra_special_tokens={})
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            cache_dir=cache_dir,
        )

        self.system_prompt = system_prompt

        print("✅ LLM 준비 완료")

    def _build_system_prompt(self) -> str:
        return getattr(self, "system_prompt", BASE_SYSTEM_PROMPT)

    def _sanitize_history(self, history: list[dict[str, str]] | None) -> list[dict[str, str]]:
        if not history:
            return []

        sanitized: list[dict[str, str]] = []
        for message in history:
            role = str(message.get("role", "")).strip().lower()
            content = str(message.get("content", "")).strip()
            if role not in {"user", "assistant"} or not content:
                continue
            sanitized.append({"role": role, "content": content})
        return sanitized

    def _build_messages(self, query: str, history: list[dict[str, str]] | None = None) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": self._build_system_prompt()}]
        messages.extend(self._sanitize_history(history))
        messages.append({"role": "user", "content": query})
        return messages

    def _tokenize_prompt(self, prompt: str) -> dict[str, torch.Tensor]:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        model_device = getattr(self.model, "device", None)
        if model_device is None:
            return inputs
        return {key: value.to(model_device) for key, value in inputs.items()}

    def _render_prompt(self, messages: list[dict[str, str]]) -> str:
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def get_prompt(self, query: str, history: list[dict[str, str]] | None = None):
        messages = self._build_messages(query, history=history)
        prompt = self._render_prompt(messages)
        return query, prompt

    def answer_extraction(self, response: str) -> str:
        try:
            if "<|im_start|>assistant" in response:
                response = response.split("<|im_start|>assistant", 1)[1]

            if "</think>" in response:
                think_end = response.find("</think>") + len("</think>")
                response = response[think_end:]

            return response.strip()

        except Exception:
            print("Error in answer_extraction")
            print(traceback.format_exc())
            return response

    def _generate_text(self, prompt: str, generation_config: GenerationConfig) -> str:
        inputs = self._tokenize_prompt(prompt)
        output_ids = self.model.generate(**inputs, generation_config=generation_config)
        input_length = inputs["input_ids"].shape[-1]
        generated_ids = output_ids[0][input_length:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def generate_memory_json(self, prompt_text: str) -> str:
        messages = [
            {"role": "system", "content": "You generate strict JSON for memory updates. Output JSON only."},
            {"role": "user", "content": prompt_text},
        ]
        prompt = self._render_prompt(messages)
        return self._generate_text(prompt, memory_gen_config)

    def stream_answer(self, query: str, history: list[dict[str, str]] | None = None, **_kwargs):
        try:
            try:
                _, prompt = self.get_prompt(query, history=history)
            except TypeError:
                _, prompt = self.get_prompt(query)
            inputs = self._tokenize_prompt(prompt)
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )

            generation_kwargs = {
                **inputs,
                "generation_config": gen_config,
                "streamer": streamer,
            }
            errors: list[BaseException] = []

            def _generate() -> None:
                try:
                    self.model.generate(**generation_kwargs)
                except BaseException as exc:
                    errors.append(exc)

            worker = Thread(target=_generate, daemon=True)
            worker.start()

            emitted = False
            for chunk in streamer:
                if chunk:
                    emitted = True
                    yield chunk

            worker.join()
            if errors and not emitted:
                raise errors[0]
        except Exception:
            print("Error in stream_answer()")
            print(traceback.format_exc())
            yield "오류가 발생했습니다."

    def ask(self, query: str, history: list[dict[str, str]] | None = None) -> str:
        try:
            answer = self.answer_extraction("".join(self.stream_answer(query, history=history)))
            return answer

        except Exception:
            print("Error in ask()")
            print(traceback.format_exc())
            return "오류가 발생했습니다."

    def chat(self, query: str, history: list[dict[str, str]] | None = None) -> str:
        return self.ask(query, history=history)
