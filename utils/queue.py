"""This module contains an implementation of a queue that fits an online
container for video clips."""
import threading
from typing import Any, List


class Queue:
    """Create a queue object with a given maximum size."""

    def __init__(self, max_size: int) -> None:
        self._queue = []
        self._max_size = max_size
        self._lock = threading.Lock()

    def put(self, item: Any) -> None:
        """Put an item into the queue."""
        with self._lock:
            self._queue.append(item)
            if len(self._queue) > self._max_size:
                del self._queue[0]

    def get(self, size: int = -1) -> List[Any]:
        """Get an item from the queue."""
        if size == -1:
            size = self._max_size
        return self._queue[-size:]

    def __len__(self) -> int:
        return len(self._queue)

    def full(self) -> bool:
        return len(self._queue) == self._max_size
