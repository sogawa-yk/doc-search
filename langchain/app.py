import threading
import queue
import json
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import oci
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain.text_splitter import CharacterTextSplitter
import oracledb
from langchain_community.vectorstores import oraclevs
from langchain_community.vectorstores.oraclevs import OracleVS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.embeddings import OCIGenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.manager import CallbackManager

import os
from dotenv import load_dotenv

# .envファイルを読み込む
load_dotenv()

app = FastAPI(
    title="Document QA API",
    description="ドキュメントに対する質問応答API"
)

# リクエストのモデル
class Question(BaseModel):
    question: str

class CompletionRequest(BaseModel):
    model: str
    messages: list

# グローバル変数として保持
chain = None

# 環境変数の取得と検証
def get_required_env(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise ValueError(f"Required environment variable {key} is not set")
    return value

class ThreadedGenerator:
    def __init__(self):
        self.queue = queue.Queue()

    def __iter__(self):
        return self

    def __next__(self):
        item = self.queue.get()
        if item is StopIteration:
            raise item
        return item

    def send(self, data):
        self.queue.put(data)

    def close(self):
        self.queue.put(StopIteration)

class ChainStreamHandler:
    def __init__(self, gen):
        self.gen = gen

    def send_streaming_token(self, token: str):
        self.gen.send(token)

class SimpleConversationChat:
    def __init__(self, history):
        self.memory = ConversationBufferMemory(memory_key="chat_memory", return_messages=True)
        self.set_memory(history)

    def set_memory(self, history):
        for message in history:
            if message['role'] == 'assistant':
                self.memory.chat_memory.add_ai_message(message['content'])
            else:
                self.memory.chat_memory.add_user_message(message['content'])

    def generator(self, user_message):
        g = ThreadedGenerator()
        threading.Thread(target=self.llm_thread, args=(g, user_message)).start()
        return g

    def llm_thread(self, g, user_message):
        try:
            llm = ChatOCIGenAI(
                model_id=get_required_env("LLM_MODEL_ID"),
                service_endpoint=get_required_env("OCI_SERVICE_ENDPOINT"),
                compartment_id=get_required_env("OCI_COMPARTMENT_ID"),
                model_kwargs={
                    "temperature": 0.7
                },
                streaming=True,
                callback_manager=CallbackManager([ChainStreamHandler(g)]),
                auth_profile=get_required_env("OCI_AUTH_PROFILE")
            )
            conv = ConversationChain(
                llm=llm,
                memory=self.memory,
            )
            conv.predict(input=user_message)
        finally:
            g.close()



def chat_generator(prompt, model, history):
    if model == "simple-conversation-chat":
        chat = SimpleConversationChat(history)
        return chat.generator(prompt)
    elif model == "summary-conversation-chat":
        g = ThreadedGenerator()
        threading.Thread(target=lambda: chain.invoke(prompt)).start()
        return g
    else:
        raise ValueError(f"Unknown model: {model}")
    g = ThreadedGenerator()
    threading.Thread(target=llm_thread, args=(g, prompt, model, history)).start()
    return g

async def streaming_response(json_data, chat_generator):
    json_without_choices = json_data.copy()
    json_without_choices["choices"] = [{
        "text": '',
        "index": 0,
        "logprobs": None,
        "finish_reason": 'length',
        "delta": {"content": ''}
    }]

    yield f"data: {json.dumps(json_without_choices)}\n\n"  # NOTE: EventSource

    text = ""
    for chunk in chat_generator:
        text += chunk
        if "choices" not in json_data or len(json_data["choices"]) == 0:
            json_data["choices"].append({
                "text": '',
                "index": 0,
                "logprobs": None,
                "finish_reason": 'length',
                "delta": {"content": 'Initializing...'}
            })
        json_data["choices"][0]["delta"] = {"content": chunk}
        yield f"data: {json.dumps(json_data)}\n\n"  # NOTE: EventSource

    json_data["choices"][0]["text"] = text
    yield f"data: {json.dumps(json_data)}\n\n"  # NOTE: EventSource
    yield f"data: [DONE]\n\n"  # NOTE: EventSource

@app.get("/question-stream")
async def stream(question: str, model: str):
    json_data = {
        "id": "chatcmpl-123",
        "object": "chat.completion.chunk",
        "created": 1234567890,
        "model": model,
        "choices": []
    }
    return StreamingResponse(
        streaming_response(json_data, chat_generator(question, model, [])),
        media_type='text/event-stream'
    )

@app.get("/v1/models")
async def models():
    return {
        "data": [
            {
                "id": "simple-conversation-chat",
                "object": "model",
                "owned_by": "organization-owner"
            },
            {
                "id": "summary-conversation-chat",
                "object": "model",
                "owned_by": "organization-owner"
            },
        ],
        "object": "list",
    }

@app.post("/v1/chat/completions")
async def chat_completions(completion_request: CompletionRequest):
    try:
        history_messages = completion_request.messages[:-1]
        user_message = completion_request.messages[-1]["content"]
        model = completion_request.model

        if model == "simple-conversation-chat" or model == "summary-conversation-chat":
            chat_gen = chat_generator(user_message, model, history_messages)
            json_data = {
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 1234567890,
                "model": model,
                "choices": []
            }
            return StreamingResponse(
                streaming_response(json_data, chat_gen),
                media_type='text/event-stream'
            )
        else:
            raise ValueError(f"Unknown model: {model}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    global chain

    # 既存のセットアップコード
    loader = UnstructuredHTMLLoader("html_data/container-instance.html")
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    docs = text_splitter.split_documents(documents)

    # データベース接続情報を環境変数から取得
    connection = oracledb.connect(
        user=get_required_env("DB_USERNAME"),
        password=get_required_env("DB_PASSWORD"),
        dsn=get_required_env("DB_DSN"),
        config_dir=get_required_env("WALLET_DIR"),
        wallet_location=get_required_env("WALLET_DIR"),
        wallet_password=get_required_env("WALLET_PASSWORD")
    )

    # OCIの設定を環境変数から取得
    embeddings = OCIGenAIEmbeddings(
        model_id=get_required_env("EMBEDDING_MODEL_ID"),
        service_endpoint=get_required_env("OCI_SERVICE_ENDPOINT"),
        compartment_id=get_required_env("OCI_COMPARTMENT_ID"),
        auth_profile=get_required_env("OCI_AUTH_PROFILE")
    )

    vector_store_dot = OracleVS.from_documents(
        docs,
        embeddings,
        client=connection,
        table_name=get_required_env("VECTOR_TABLE_NAME"),
        distance_strategy=DistanceStrategy.DOT_PRODUCT,
    )

    template = """contextに従って回答してください:
    {context}

    質問: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOCIGenAI(
        model_id=get_required_env("LLM_MODEL_ID"),
        service_endpoint=get_required_env("OCI_SERVICE_ENDPOINT"),
        compartment_id=get_required_env("OCI_COMPARTMENT_ID"),
        model_kwargs={
            "temperature": float(os.getenv("LLM_TEMPERATURE", "0.7")),
            "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "500"))
        },
        auth_profile=get_required_env("OCI_AUTH_PROFILE")
    )

    retriever = vector_store_dot.as_retriever()

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
