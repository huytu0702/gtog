"""LLM-as-Judge metrics for evaluation."""

from dataclasses import dataclass
from typing import Any
import logging
import json

from graphrag.language_model.protocol.base import ChatModel

logger = logging.getLogger(__name__)


# Prompts as module constants
CORRECTNESS_PROMPT = """Given the question, ground truth answer, and predicted answer,
judge if the prediction is correct.

Question: {question}
Ground Truth: {ground_truth}
Predicted: {predicted}

Score 1 if the predicted answer conveys the same meaning as ground truth
(exact wording not required). Score 0 if incorrect or missing key information.

Return JSON: {{"score": 0 or 1, "reason": "brief explanation"}}"""

FAITHFULNESS_PROMPT = """Given the context retrieved and the answer generated,
judge if the answer is supported by the context.

Context: {context}
Answer: {answer}

Score 1 if all claims in the answer can be traced to the context.
Score 0 if the answer contains unsupported claims (hallucination).

Return JSON: {{"score": 0 or 1, "reason": "brief explanation"}}"""

RELEVANCE_PROMPT = """Given the question and the context retrieved,
judge if the context is relevant to answering the question.

Question: {question}
Context: {context}

Score 1 if the context contains information useful for answering the question.
Score 0 if the context is unrelated or unhelpful.

Return JSON: {{"score": 0 or 1, "reason": "brief explanation"}}"""

COMPLETENESS_PROMPT = """Given the question and the answer generated,
judge if the answer fully addresses the question.

Question: {question}
Answer: {answer}

Score 1 if the answer addresses all aspects of the question.
Score 0 if the answer is partial or misses key aspects.

Return JSON: {{"score": 0 or 1, "reason": "brief explanation"}}"""


@dataclass
class JudgeResult:
    """Result from a single LLM judge evaluation."""
    score: int  # 0 or 1
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"score": self.score, "reason": self.reason}


@dataclass
class MetricScores:
    """All metric scores for a single query-response pair."""
    correctness: JudgeResult
    faithfulness: JudgeResult
    relevance: JudgeResult
    completeness: JudgeResult

    def to_dict(self) -> dict[str, int]:
        """Return just scores for summary aggregation."""
        return {
            "correctness": self.correctness.score,
            "faithfulness": self.faithfulness.score,
            "relevance": self.relevance.score,
            "completeness": self.completeness.score,
        }

    def to_detailed_dict(self) -> dict[str, dict[str, Any]]:
        """Return full scores with reasons."""
        return {
            "correctness": self.correctness.to_dict(),
            "faithfulness": self.faithfulness.to_dict(),
            "relevance": self.relevance.to_dict(),
            "completeness": self.completeness.to_dict(),
        }


class LLMJudge:
    """LLM-as-Judge for evaluating search responses."""

    def __init__(
        self,
        model: ChatModel,
        temperature: float = 0.0,
    ):
        self.model = model
        self.temperature = temperature

    async def _call_judge(self, prompt: str) -> JudgeResult:
        """Call LLM and parse JSON response."""
        response = ""
        async for chunk in self.model.achat_stream(
            prompt=prompt,
            history=[],
            model_parameters={"temperature": self.temperature},
        ):
            response += chunk

        try:
            # Try to parse JSON from response
            # Handle potential markdown code blocks
            response = response.strip()
            if response.startswith("```"):
                # Extract content between code blocks
                lines = response.split("\n")
                json_lines = []
                in_block = False
                for line in lines:
                    if line.startswith("```") and not in_block:
                        in_block = True
                        continue
                    if line.startswith("```") and in_block:
                        break
                    if in_block:
                        json_lines.append(line)
                response = "\n".join(json_lines)

            result = json.loads(response)
            return JudgeResult(
                score=int(result.get("score", 0)),
                reason=result.get("reason", "No reason provided"),
            )
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse judge response: {response[:100]}... Error: {e}")
            # Try to extract score from text
            if "1" in response and "0" not in response:
                return JudgeResult(score=1, reason=f"Inferred from: {response[:100]}")
            return JudgeResult(score=0, reason=f"Parse error: {response[:100]}")

    async def judge_correctness(
        self,
        question: str,
        ground_truth: str,
        predicted: str,
    ) -> JudgeResult:
        """Judge if predicted answer matches ground truth."""
        prompt = CORRECTNESS_PROMPT.format(
            question=question,
            ground_truth=ground_truth,
            predicted=predicted,
        )
        return await self._call_judge(prompt)

    async def judge_faithfulness(
        self,
        context: str,
        answer: str,
    ) -> JudgeResult:
        """Judge if answer is supported by context."""
        prompt = FAITHFULNESS_PROMPT.format(
            context=context,
            answer=answer,
        )
        return await self._call_judge(prompt)

    async def judge_relevance(
        self,
        question: str,
        context: str,
    ) -> JudgeResult:
        """Judge if context is relevant to question."""
        prompt = RELEVANCE_PROMPT.format(
            question=question,
            context=context,
        )
        return await self._call_judge(prompt)

    async def judge_completeness(
        self,
        question: str,
        answer: str,
    ) -> JudgeResult:
        """Judge if answer fully addresses question."""
        prompt = COMPLETENESS_PROMPT.format(
            question=question,
            answer=answer,
        )
        return await self._call_judge(prompt)

    async def evaluate_all(
        self,
        question: str,
        ground_truth: str,
        predicted: str,
        context: str,
    ) -> MetricScores:
        """Evaluate all metrics for a query-response pair."""
        correctness = await self.judge_correctness(question, ground_truth, predicted)
        faithfulness = await self.judge_faithfulness(context, predicted)
        relevance = await self.judge_relevance(question, context)
        completeness = await self.judge_completeness(question, predicted)

        return MetricScores(
            correctness=correctness,
            faithfulness=faithfulness,
            relevance=relevance,
            completeness=completeness,
        )
