"""This module contains an implementation of a stack that fits an online
container for video clips."""

import threading
from typing import Any, List


class Stack:
    """Create a stack object with a given maximum size."""

    def __init__(self, max_size: int) -> None:
        self._stack = []
        self._max_size = max_size
        self._lock = threading.Lock()

    def put(self, item: Any) -> None:
        """Put an item into the stack."""
        with self._lock:
            self._stack.append(item)
            if len(self._stack) > self._max_size:
                del self._stack[0]

    def get(self, size: int = -1) -> List[Any]:
        """Get an item from the stack."""
        if size == -1:
            size = self._max_size
        return self._stack[-size:]

    def __len__(self) -> int:
        return len(self._stack)

    def full(self) -> bool:
        return len(self._stack) == self._max_size
