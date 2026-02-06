from abc import ABC, abstractmethod


class LLMClient(ABC):

    @abstractmethod
    def triplet_judge(self, instruction: str, anchor: str, c1: str, c2: str) -> int:
        """
        Returns:
            1 if c1 is closer to anchor
            2 if c2 is closer to anchor
        """
        pass

    @abstractmethod
    def pairwise_judge(self, instruction: str, s1: str, s2: str) -> bool:
        """
        Returns:
            True if same cluster
            False otherwise
        """
        pass
