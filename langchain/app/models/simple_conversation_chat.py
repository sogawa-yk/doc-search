import threading
import os
from typing import List

from langchain import ConversationChain
from langchain.callbacks.manager import CallbackManager
from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)

from app.callbacks.streaming import ThreadedGenerator, ChainStreamHandler

class SimpleConversationChat:
    def __init__(self, history, system_prompt: str = None):
        self.memory = ConversationBufferMemory(return_messages=True)
        self.system_prompt = system_prompt
        self.set_memory(history)

    def set_memory(self, history):
        """過去の会話履歴をメモリにセット"""
        for message in history:
            if message.role == 'assistant':
                self.memory.chat_memory.add_ai_message(message.content)
            elif message.role == 'user':
                self.memory.chat_memory.add_user_message(message.content)

    def create_prompt(self):
        """システムプロンプトを含むプロンプトテンプレートを作成"""
        messages = []
        if self.system_prompt:
            messages.append(SystemMessagePromptTemplate.from_template(self.system_prompt))
        messages.extend([
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ])
        return ChatPromptTemplate.from_messages(messages)

    def generator(self, user_message):
        """チャット応答のジェネレータを作成"""
        g = ThreadedGenerator()
        threading.Thread(target=self.llm_thread, args=(g, user_message)).start()
        return g

    def llm_thread(self, g, user_message):
        """LLMとの対話を処理するスレッド"""
        try:
            llm = ChatOCIGenAI(
                model_id=os.getenv("LLM_MODEL_ID"),
                service_endpoint=os.getenv("OCI_SERVICE_ENDPOINT"),
                compartment_id=os.getenv("OCI_COMPARTMENT_ID"),
                auth_profile=os.getenv("OCI_AUTH_PROFILE"),
                callback_manager=CallbackManager([ChainStreamHandler(g)]),
                model_kwargs={
                    "temperature": float(os.getenv("LLM_TEMPERATURE", "0.7")),
                    "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "500")),
                    "stream": True
                }
            )

            prompt = self.create_prompt()
            conv = ConversationChain(
                llm=llm,
                memory=self.memory,
                prompt=prompt,
                verbose=True
            )

            conv.predict(input=user_message)
        except Exception as e:
            g.send(str(e))
        finally:
            g.close()
