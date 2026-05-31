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
    temperature=0.7,
    top_p=0.9,
    top_k=50,
)

BASE_SYSTEM_PROMPT = (
    "You are an English conversation teacher on a live phone call. "
    "Reply briefly in plain spoken English, only as the teacher, "
    "and never generate the student's side of the conversation."
)


class QwenChat:
    def __init__(
        self,
        model_name=MODEL_NAME,
        tok_model=TOK_MODEL_NAME,
        cache_dir=None,
        system_prompt: str = BASE_SYSTEM_PROMPT,
    ):
        print("🔧 LLM 모델 로딩 중...")

        cache_dir = cache_dir or os.getenv("HF_HOME")

        self.tokenizer = AutoTokenizer.from_pretrained(tok_model)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir=cache_dir,
        )

        self.system_prompt = system_prompt

        print("✅ LLM 준비 완료")

    def _build_system_prompt(self) -> str:
        return self.system_prompt

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
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
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

    def stream_answer(self, query: str, history: list[dict[str, str]] | None = None):
        try:
            _, prompt = self.get_prompt(query, history=history)
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

            worker = Thread(
                target=self.model.generate,
                kwargs=generation_kwargs,
                daemon=True,
            )
            worker.start()

            for chunk in streamer:
                if chunk:
                    yield chunk

            worker.join()
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
