from queue import Queue, Empty
from typing import Any, Dict
from langchain.callbacks.base import BaseCallbackHandler
import logging
import json
import time
import uuid

class ThreadedGenerator:
    def __init__(self):
        self.queue = Queue()
        self._is_closed = False
        self.retry_count = 0
        self.max_retries = 10
        self.response_id = str(uuid.uuid4())
        self.created = int(time.time())

    def __iter__(self):
        return self

    def __next__(self):
        if self._is_closed and self.queue.empty():
            raise StopIteration
        
        try:
            item = self.queue.get(timeout=1.0)
            if isinstance(item, Exception):
                raise item
            self.retry_count = 0
            return item
        except Empty:
            self.retry_count += 1
            if self.retry_count >= self.max_retries or self._is_closed:
                raise StopIteration
            return None

    def send(self, value: str):
        """値をキューに送信"""
        if not self._is_closed and value:
            logging.debug(f"Sending value to queue: {value}")
            self.queue.put(value)

    def close(self):
        """ジェネレータを閉じる"""
        logging.debug("Closing generator")
        self._is_closed = True

class ChainStreamHandler(BaseCallbackHandler):
    def __init__(self, gen: ThreadedGenerator):
        super().__init__()
        self.gen = gen
        self.first_token = True

    def create_chunk_response(self, content: Dict[str, Any] = None, finish_reason: str = None) -> str:
        """OpenAI互換のチャンクレスポンスを生成"""
        response = {
            "id": self.gen.response_id,
            "object": "chat.completion.chunk",
            "created": self.gen.created,
            "model": "oci-llm",  # OCIのモデル名
            "choices": [{
                "index": 0,
                "delta": content or {},
                "finish_reason": finish_reason
            }]
        }
        return f"data: {json.dumps(response)}\n\n"

    async def on_llm_start(self, *args, **kwargs):
        """LLM開始時に初期レスポンスを送信"""
        initial_response = self.create_chunk_response({"role": "assistant"})
        self.gen.send(initial_response)

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """新しいトークンを受け取ったときの処理"""
        if token:
            chunk = self.create_chunk_response({"content": token})
            self.gen.send(chunk)

    async def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """LLMの処理が終了したときの処理"""
        # 最終チャンクを送信
        final_response = self.create_chunk_response(finish_reason="stop")
        self.gen.send(final_response)
        # 終了シグナルを送信
        self.gen.send("data: [DONE]\n\n")
        self.gen.close()

    async def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        """エラーが発生したときの処理"""
        error_response = {
            "error": {
                "message": str(error),
                "type": "server_error",
                "code": "internal_error"
            }
        }
        self.gen.send(f"data: {json.dumps(error_response)}\n\n")
        self.gen.close()
