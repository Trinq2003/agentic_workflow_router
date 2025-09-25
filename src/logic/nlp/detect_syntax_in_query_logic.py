from logic.base_classes import BaseLogic
import numpy as np
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

    Returns a 2D tensor indicating if any command syntax is detected:
    [[0,0,1,0,0]] if any command detected, [[0,0,0,0,0]] otherwise
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

                # Look for the pattern with proper command context (case-insensitive)
                # Commands should appear at start of query or after command-indicating words
                padded_lower = padded_query.lower()
                
                # Find all occurrences of the pattern
                import re
                # Pattern: start of string/query OR after command words, followed by our pattern, then space/end
                command_indicators = r'(^|\s|please\s|can\s|could\s|would\s|let\s|help\s|run\s|execute\s)'
                regex_pattern = rf'{command_indicators}({re.escape(separator)}{re.escape(command)})(\s|$)'
                
                if re.search(regex_pattern, padded_lower, re.IGNORECASE):
                    # Additional check: make sure it's not in a file path context
                    # Avoid matching if preceded by words like "folder", "directory", "file", "path"
                    path_context_pattern = rf'(folder|directory|file|path|contains|in|the)\s+{re.escape(separator)}{re.escape(command)}'
                    if not re.search(path_context_pattern, padded_lower, re.IGNORECASE):
                        results[i] = 1.0
                        break  # Found this command, no need to check other separators

        return results

    def _detect_reminder_verbs(self, query: str) -> bool:
        """
        Detect reminder-related verbs in Vietnamese and English.
        
        Args:
            query: Input query string
            
        Returns:
            bool: True if reminder verbs are detected, False otherwise
        """
        if not query or not isinstance(query, str):
            return False
            
        query_lower = query.lower()
        
        # Vietnamese reminder verbs and related words
        vietnamese_reminder_words = [
            # Direct reminder verbs
            "nhắc", "nhắc nhở", "nhở", "nhắc lại",
            
            # Reminder-related actions
            "ghi nhớ", "ghi chú", "lưu ý", "chú ý",
            "đặt lịch", "đặt nhắc", "báo thức",
            "thông báo", "cảnh báo", "nhắc việc",
            
            # Memory-related verbs
            "nhớ", "quên", "làm nhớ", "giúp nhớ",
            "erinnern", "memorize", "remember",
            
            # Task/appointment reminders
            "hẹn", "hẹn giờ", "đặt hẹn", "lịch hẹn",
            "cuộc hẹn", "buổi hẹn", "meeting reminder",
            
            # Schedule-related
            "lịch trình", "thời gian biểu", "kế hoạch",
            "sắp xếp", "bố trí", "an toàn thời gian"
        ]
        
        # English reminder verbs and related words
        english_reminder_words = [
            # Direct reminder verbs
            "remind", "remember", "recall", "memorize",
            
            # Reminder actions
            "notify", "alert", "warn", "prompt",
            "ping", "buzz", "signal", "flag",
            
            # Schedule/appointment related
            "schedule", "book", "set", "arrange",
            "plan", "organize", "calendar", "agenda",
            
            # Task reminders
            "todo", "task", "deadline", "due",
            "follow up", "check in", "touch base"
        ]
        
        # Combine all reminder words
        all_reminder_words = vietnamese_reminder_words + english_reminder_words
        
        # Check for exact matches and word boundaries
        import re
        for word in all_reminder_words:
            # For multi-word phrases, use simple substring matching
            if len(word.split()) > 1:
                if word in query_lower:
                    logger.debug(f"[DetectSyntaxInQueryLogic] Found reminder phrase: '{word}' in query")
                    return True
            else:
                # For single words, use word boundary matching
                pattern = r'\b' + re.escape(word) + r'\b'
                if re.search(pattern, query_lower):
                    logger.debug(f"[DetectSyntaxInQueryLogic] Found reminder word: '{word}' in query")
                    return True
        
        return False

    def forward(self, query: str):
        """
        Detect command syntax patterns and return tailored vectors for 8 workers.

        Workers mapping:
        0: FUNCTION_CALLER_AGENT
        1: DOCS_SEARCHER_AGENT  
        2: NETMIND_INFOR_AGENT
        3: REGION_IDENTIFIER_AGENT
        4: WEBS_SEARCHER_AGENT
        5: TIME_IDENTIFIER_AGENT
        6: EMPLOYEE_INFOR_AGENT
        7: REMINDER_AGENT

        Args:
            query: Input query string

        Returns:
            Vector: Probability distribution over 8 workers based on detected command syntax
        """
        logger.debug(f"[DetectSyntaxInQueryLogic] Processing query: '{query}'")

        if not query or not isinstance(query, str):
            logger.debug(f"[DetectSyntaxInQueryLogic] Invalid input, returning zero vector")
            return np.array([[0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)

        # Get command detection results
        command_results = self._detect_commands(query)
        logger.debug(f"[DetectSyntaxInQueryLogic] Command detection results: {command_results}")
        
        # Get reminder verb detection results
        reminder_verbs_detected = self._detect_reminder_verbs(query)
        logger.debug(f"[DetectSyntaxInQueryLogic] Reminder verbs detected: {reminder_verbs_detected}")

        # Create contribution mapping for this logic
        command_names = ['remind', 'task', 'tasks', 'meeting']
        detected_commands = [cmd for cmd, detected in zip(command_names, command_results) if detected > 0]
        if reminder_verbs_detected:
            detected_commands.append("reminder_verbs")
        logger.debug(f"[DetectSyntaxInQueryLogic] Detected features: {detected_commands}")

        # Initialize result vector for 8 workers
        result = np.array([[0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)

        # Check if any syntax or reminder features detected
        has_syntax_features = any(r > 0 for r in command_results) or reminder_verbs_detected
        
        if has_syntax_features:
            # Command syntax or reminder verbs detected - distribute based on detected features
            remind_detected = command_results[0] > 0  # remind command
            task_detected = command_results[1] > 0    # task command
            tasks_detected = command_results[2] > 0   # tasks command
            meeting_detected = command_results[3] > 0 # meeting command

            # Assign weights based on detected features
            if remind_detected:
                result[0][7] += 1  # REMINDER_AGENT (primary for reminders)
            
            if task_detected or tasks_detected:
                result[0][7] += 1  # REMINDER_AGENT (task management)
            
            if meeting_detected:
                result[0][7] += 1  # REMINDER_AGENT (meeting reminders)
            
            # Add weight for reminder verbs detected in natural language
            if reminder_verbs_detected:
                result[0][7] += 1  # REMINDER_AGENT (reminder verbs)

        else:
            # No syntax features detected - return zero vector
            logger.debug(f"[DetectSyntaxInQueryLogic] No syntax features detected -> zero vector")
            return np.array([[0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)

        # Normalize to ensure sum = 1
        if np.sum(result) > 0:
            result = result / np.sum(result)
        
        logger.debug(f"[DetectSyntaxInQueryLogic] Final contribution - Query: '{query}' -> Commands: {detected_commands} -> Vector: {result.tolist()}")
        return result