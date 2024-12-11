import threading
import os
from typing import List

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI
from langchain.memory import ChatMessageHistory
from langchain.prompts import ChatPromptTemplate

from callbacks.streaming import ThreadedGenerator, ChainStreamHandler

import oci
import logging

logging.getLogger('oci').setLevel(logging.DEBUG)

class SummaryConversationChat:
    def __init__(self, history):
        self.message_history = ChatMessageHistory()
        self.set_memory(history)
        
        # チャットモデルの初期化
        self.llm = ChatOCIGenAI(
            model_id=os.getenv("LLM_MODEL_ID"),
            service_endpoint=os.getenv("OCI_SERVICE_ENDPOINT"),
            compartment_id=os.getenv("OCI_COMPARTMENT_ID"),
            auth_profile=os.getenv("OCI_AUTH_PROFILE"),
            model_kwargs={
                "temperature": float(os.getenv("LLM_TEMPERATURE", "0.7")),
                "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "500"))
            }
        )

        # プロンプトテンプレートの設定
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "以下は人間とAIの会話です。適切に応答してください。"),
            ("human", "{input}")
        ])

        # Runnableの設定
        self.chain = RunnableWithMessageHistory(
            self.prompt | self.llm,
            lambda session_id: self.message_history,
            input_messages_key="input",
            history_messages_key="history"
        )

    def set_memory(self, history):
        for message in history:
            if message.role == 'assistant':
                self.message_history.add_message(AIMessage(content=message.content))
            else:
                self.message_history.add_message(HumanMessage(content=message.content))

    def generator(self, user_message):
        g = ThreadedGenerator()
        threading.Thread(target=self.llm_thread, args=(g, user_message)).start()
        return g

    def llm_thread(self, g, user_message):
        try:
            # ストリーミング用のコールバックを設定
            self.llm.callbacks = [ChainStreamHandler(g)]
            
            # 会話の実行
            response = self.chain.invoke(
                {"input": user_message},
                config={"configurable": {"session_id": "default"}}
            )
            
            # メッセージ履歴に追加
            self.message_history.add_message(HumanMessage(content=user_message))
            self.message_history.add_message(AIMessage(content=str(response)))
            
        finally:
            g.close()