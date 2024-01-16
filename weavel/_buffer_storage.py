from __future__ import annotations

from threading import Condition, Lock
from typing import Dict, List

from weavel.types import WeavelRequest


class BufferStorage:
    def __init__(self, max_buffer_size: int):
        self.max_buffer_size = max_buffer_size
        self.buffer: List[WeavelRequest] = []
        self.buffer_lock = Lock()
        self.not_full_cv = Condition(self.buffer_lock)
        self.not_empty_cv = Condition(self.buffer_lock)

    @property
    def buffer_size(self) -> int:
        return len(self.buffer)

    def push(self, event: Dict) -> None:
        with self.buffer_lock:
            while self.buffer_size >= self.max_buffer_size:
                self.not_empty_cv.notify()  # immediately wake up possibly sleeping consumer
                self.not_full_cv.wait()
            self.buffer.append(event)
            if self.buffer_size >= self.max_buffer_size:
                self.not_empty_cv.notify()

    def pull(self, batch_size: int) -> List[WeavelRequest]:
        while not self.buffer_size:
            self.not_empty_cv.wait()
        pull_size = min(batch_size, self.buffer_size)
        out = self.buffer[:pull_size]
        self.buffer = self.buffer[pull_size:]
        self.not_full_cv.notify()
        return out

    def pull_all(self) -> List[WeavelRequest]:
        out = self.buffer[:]
        self.buffer = []
        self.not_full_cv.notify()
        return out
