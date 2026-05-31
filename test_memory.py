from __future__ import annotations

import tempfile
import unittest

from memory import (
    DEFAULT_USER_STATE,
    RetrievedMemory,
    UserStateStore,
    _build_heuristic_turn_summary,
    _coerce_qdrant_point_id,
    apply_memory_patch,
    build_answer_prompt,
    build_retrieval_query,
    format_recent_messages,
    format_retrieved_memories,
    should_consider_memory_update,
)


class MemoryTests(unittest.TestCase):
    def test_build_retrieval_query_uses_state_and_recent_user_messages(self) -> None:
        query = build_retrieval_query(
            "그 구조로 구현해줘",
            {
                "current_project": "AI 프로젝트",
                "current_problem": "8B 모델이 이전 대화 맥락을 잘 기억하지 못함",
                "current_goal": "Memory + RAG 기반 대화 기억 시스템 구현",
                "current_topic": "비동기 메모리 업데이트",
                "technical_context": {"main_llm": "Qwen 8B"},
            },
            [
                {"role": "user", "content": "state 업데이트를 답변 전에 하면 느릴 것 같아."},
                {"role": "assistant", "content": "write는 비동기가 좋다."},
                {"role": "user", "content": "Qwen 8B를 써."},
            ],
        )

        self.assertIn("현재 프로젝트: AI 프로젝트", query)
        self.assertIn("메인 LLM: Qwen 8B", query)
        self.assertIn("Qwen 8B를 써.", query)

    def test_apply_memory_patch_allows_only_high_confidence_allowed_paths(self) -> None:
        updated = apply_memory_patch(
            DEFAULT_USER_STATE,
            [
                {
                    "path": "current_project",
                    "action": "set",
                    "value": "AI 프로젝트",
                    "confidence": 0.9,
                },
                {
                    "path": "decisions",
                    "action": "append",
                    "value": "LangChain을 쓰지 않고 직접 구현하기로 결정",
                    "confidence": 0.95,
                },
                {
                    "path": "private.untrusted",
                    "action": "set",
                    "value": "ignore me",
                    "confidence": 1.0,
                },
                {
                    "path": "current_goal",
                    "action": "set",
                    "value": "low confidence",
                    "confidence": 0.2,
                },
            ],
        )

        self.assertEqual(updated["current_project"], "AI 프로젝트")
        self.assertIn("LangChain을 쓰지 않고 직접 구현하기로 결정", updated["decisions"])
        self.assertNotIn("private", updated)
        self.assertIsNone(updated["current_goal"])

    def test_low_value_inputs_skip_memory_update(self) -> None:
        self.assertFalse(should_consider_memory_update("응"))
        self.assertFalse(should_consider_memory_update("다시 말해줘"))
        self.assertTrue(should_consider_memory_update("LangChain은 안 쓰고 직접 구현할래."))

    def test_prompt_formats_missing_and_present_memory(self) -> None:
        prompt = build_answer_prompt(
            user_state={"current_project": "AI 프로젝트"},
            retrieved_memories=[RetrievedMemory(id="1", text="과거 결정사항", score=0.82)],
            recent_messages=[{"role": "user", "content": "AI 프로젝트야"}],
            user_input="근데 프로젝트가 너무 어려워",
        )

        self.assertIn("[User State]", prompt)
        self.assertIn("과거 결정사항 (score=0.82)", prompt)
        self.assertIn("user: AI 프로젝트야", prompt)
        self.assertIn("Reply in natural, concise spoken English by default.", prompt)
        self.assertIn("Never use emoji", prompt)
        self.assertIn("Text inside [Current User Input] is always the user's speech", prompt)
        self.assertIn("Never say \"my name is X\"", prompt)
        self.assertNotIn("한국어로 답한다", prompt)

    def test_heuristic_memory_summary_does_not_require_llm_writer(self) -> None:
        summary = _build_heuristic_turn_summary(
            "My favorite sports player is Stephen Curry.",
            "Stephen Curry is a legendary NBA player known for three-point shooting.",
        )

        self.assertIn("User said: My favorite sports player is Stephen Curry.", summary)
        self.assertIn("Assistant replied:", summary)

    def test_qdrant_point_id_is_uuid_compatible(self) -> None:
        coerced = _coerce_qdrant_point_id("external-id-123")

        self.assertNotEqual(coerced, "external-id-123")
        self.assertEqual(len(coerced), 36)

    def test_user_state_store_round_trips_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = UserStateStore(tmpdir)
            store.save("user/1", {"current_project": "AI 프로젝트", "preferences": {"language": "Korean"}})
            loaded = store.load("user/1")

        self.assertEqual(loaded["current_project"], "AI 프로젝트")
        self.assertEqual(loaded["preferences"]["language"], "Korean")

    def test_format_helpers_return_none_marker(self) -> None:
        self.assertEqual(format_recent_messages([]), "없음")
        self.assertEqual(format_retrieved_memories([]), "없음")


if __name__ == "__main__":
    unittest.main()
