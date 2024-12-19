import asyncio
from typing import List
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI
from langchain.memory import ChatMessageHistory
from langchain.prompts import ChatPromptTemplate
from callbacks.streaming import ThreadedGenerator, ChainStreamHandler
import os
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
        logging.info('Chat Initialization Complited')

    def set_memory(self, history):
        for message in history:
            if message.role == 'assistant':
                self.message_history.add_message(AIMessage(content=message.content))
            else:
                self.message_history.add_message(HumanMessage(content=message.content))

    def generator(self, user_message):
        g = ThreadedGenerator()
        handler = ChainStreamHandler(g)
        
        async def process_response():
            try:
                logging.info(f"Sending message to LLM: {user_message}")
                
                # コールバックハンドラーを設定して非同期で実行
                response = await self.chain.ainvoke(
                    {"input": user_message},
                    config={
                        "configurable": {"session_id": "default"},
                        "callbacks": [handler]
                    }
                )
                logging.info(f"Response received: {response}")
                
                # メッセージ履歴に追加
                self.message_history.add_message(HumanMessage(content=user_message))
                self.message_history.add_message(AIMessage(content=str(response)))
                
                # 最初のトークンを送信
                g.send(str(response))
                
            except Exception as e:
                logging.error(f"Error in process_response: {e}", exc_info=True)
                g.send(str(e))
            finally:
                g.close()

        # 非同期タスクを開始
        asyncio.create_task(process_response())
        return g