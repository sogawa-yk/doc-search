from queue import Queue, Empty
from typing import Any, Optional
from langchain.callbacks.base import BaseCallbackHandler
import logging
import asyncio


class ThreadedGenerator:
    def __init__(self):
        self.queue = Queue()
        self._is_closed = False
        self.retry_count = 0
        self.max_retries = 10

    def __iter__(self):
        return self

    def __next__(self):
        if self._is_closed and self.queue.empty():
            raise StopIteration
        
        try:
            item = self.queue.get(timeout=1.0)  # タイムアウトを設定
            if isinstance(item, Exception):
                raise item
            self.retry_count = 0  # 成功したらリトライカウントをリセット
            return item
        except Empty:
            self.retry_count += 1
            if self.retry_count >= self.max_retries or self._is_closed:
                raise StopIteration
            return None  # 空の場合はNoneを返す（再帰呼び出しを避ける）

    def send(self, value):
        if not self._is_closed and value:
            logging.debug(f"Sending value to queue: {value}")
            self.queue.put(value)

    def close(self):
        logging.debug("Closing generator")
        self._is_closed = True


class ChainStreamHandler(BaseCallbackHandler):
    def __init__(self, gen: ThreadedGenerator):
        super().__init__()
        self.gen = gen

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """新しいトークンを受け取ったときの処理"""
        logging.debug(f"Received new token: {token}")
        if token:
            self.gen.send(token)

    async def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """LLMの処理が終了したときの処理"""
        logging.debug("LLM ended")
        self.gen.close()

    async def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        """エラーが発生したときの処理"""
        logging.error(f"LLM error: {error}")
        self.gen.send(str(error))
        self.gen.close()