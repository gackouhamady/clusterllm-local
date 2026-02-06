from __future__ import annotations

from typing import Literal

from clusterllm.llm_client.abstract import LLMClient


class OpenAIClient(LLMClient):
    """
    OpenAI-based LLM judge client.

    This client reproduces the behavior described in the CLUSTERLLM paper,
    using GPT-style models as an oracle for triplet and pairwise judgments.
    """

    def triplet_judge(
        self,
        instruction: str,
        anchor: str,
        c1: str,
        c2: str,
    ) -> Literal[1, 2]:
        """
        Decide which candidate sentence is closer to the anchor.

        Args:
            instruction: Task instruction (perspective).
            anchor: Anchor sentence.
            c1: First candidate sentence.
            c2: Second candidate sentence.

        Returns:
            1 if c1 is closer to anchor, 2 if c2 is closer.
        """
        raise NotImplementedError("triplet_judge is not implemented yet")

    def pairwise_judge(
        self,
        instruction: str,
        s1: str,
        s2: str,
    ) -> bool:
        """
        Decide whether two sentences belong to the same cluster.

        Args:
            instruction: Task instruction (granularity).
            s1: First sentence.
            s2: Second sentence.

        Returns:
            True if sentences belong to the same cluster, False otherwise.
        """
        raise NotImplementedError("pairwise_judge is not implemented yet")
