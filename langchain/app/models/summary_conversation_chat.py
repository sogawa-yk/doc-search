import threading

from langchain import OpenAI, ConversationChain
from langchain.callbacks import CallbackManager
from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI
from langchain.memory import ConversationSummaryMemory

from app.callbacks.streaming import ThreadedGenerator, ChainStreamHandler


class SummaryConversationChat:
    def __init__(self, history):
        self.memory = ConversationSummaryMemory(llm=OpenAI(temperature=0))
        self.set_memory(history)

    def set_memory(self, history):
        for message in history:
            if message.role == 'assistant':
                self.memory.chat_memory.add_ai_message(message.content)
            else:
                self.memory.chat_memory.add_user_message(message.content)

    def generator(self, user_message):
        g = ThreadedGenerator()
        threading.Thread(target=self.llm_thread, args=(g, user_message)).start()
        return g

    def llm_thread(self, g, user_message):
        try:
            llm = ChatOCIGenAI(
                verbose=True,
                streaming=True,
                callback_manager=CallbackManager([ChainStreamHandler(g)]),
                temperature=0.7,
            )
            conv = ConversationChain(
                llm=llm,
                memory=self.memory,
            )
            conv.predict(input=user_message)
        finally:
            g.close()