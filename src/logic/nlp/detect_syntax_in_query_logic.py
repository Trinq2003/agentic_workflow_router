from logic.base_classes import BaseLogic
import torch
import re
from typing import List
import logging

logger = logging.getLogger(__name__)


class DetectSyntaxInQueryLogic(BaseLogic):
    """
    Logic class for detecting command syntax patterns in queries.

    Detects specific command patterns:
    - /remind or \remind
    - \tasks or /tasks
    - \task or /task
    - /meeting or \meeting

    Uses simple whitespace-based detection to avoid file path confusion.

    Returns a 1x4 tensor indicating which commands are present:
    [remind_detected, task_detected, tasks_detected, meeting_detected]
    """

    def __init__(self):
        super().__init__()

        # Command patterns to detect (treating \ and / as equivalent)
        self.commands = ['remind', 'task', 'tasks', 'meeting']

        logger.info("Initialized DetectSyntaxInQueryLogic")

    def _detect_commands(self, query: str) -> List[float]:
        """
        Detect command patterns by checking if they're separated by whitespace.

        Args:
            query: Input query string

        Returns:
            List[float]: List of 4 values [remind, task, tasks, meeting] where 1.0 = detected, 0.0 = not detected
        """
        if not query or not isinstance(query, str):
            return [0.0, 0.0, 0.0, 0.0]

        # Add whitespace padding to handle beginning/end cases
        padded_query = f"  {query}  "

        # Initialize results for each command
        results = [0.0] * len(self.commands)

        # Check each command pattern
        for i, command in enumerate(self.commands):
            # Check both forward slash and backslash versions
            for separator in ['/', '\\']:
                pattern = f"{separator}{command}"

                # Look for the pattern between whitespaces
                if f" {pattern} " in padded_query:
                    results[i] = 1.0
                    logging.debug(f"Command pattern '{pattern}' found in query: {query}")
                    break  # Found this command, no need to check other separators

        if any(r > 0 for r in results):
            logging.debug(f"Command patterns found in query: {query} -> {results}")
        else:
            logging.debug(f"No command patterns found in query: {query}")

        return results

    def forward(self, query: str) -> torch.Tensor:
        """
        Detect which command syntax patterns are present in the query.

        Args:
            query: Input query string

        Returns:
            torch.Tensor: 1x4 tensor [remind, task, tasks, meeting] where 1.0 = detected, 0.0 = not detected
        """
        if not query or not isinstance(query, str):
            return torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)

        # Get command detection results
        command_results = self._detect_commands(query)

        # Convert to tensor
        result = torch.tensor(command_results, dtype=torch.float32)

        logger.debug(f"Query: '{query}' -> Command detection results: {command_results}")
        return result