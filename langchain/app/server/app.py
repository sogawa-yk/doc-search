import json
import os
import time
import uuid
from typing import List
from dotenv import load_dotenv

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import logging

from pydantic import BaseModel

from models.simple_conversation_chat import SimpleConversationChat
from models.summary_conversation_chat import SummaryConversationChat

from langchain_community.embeddings import OCIGenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores.oraclevs import OracleVS
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

import oracledb

logging.basicConfig(level=logging.INFO)
load_dotenv('/app/.env')

app = FastAPI(
    title="LangChain API",
)


class Message(BaseModel):
    role: str
    content: str


class CompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: int
    temperature: float
    stream: bool


async def streaming_response(json_data, chat_generator):
    json_without_choices = json_data.copy()
    json_without_choices["choices"] = [{
        "text": '',
        "index": 0,
        "logprobs": None,
        "finish_reason": 'length',
        "delta": {"content": ''}
    }]

    # logging.info(f"Sending initial JSON: {json_without_choices}")
    yield f"data: {json.dumps(json_without_choices)}\n\n"  # NOTE: EventSource

    text = ""
    for chunk in chat_generator:
        text += chunk
        json_data["choices"][0]["delta"] = {"content": chunk}
        # logging.info(f"Sending chunk: {json.dumps(json_data)}")
        yield f"data: {json.dumps(json_data)}\n\n"  # NOTE: EventSource

    json_data["choices"][0]["text"] = text
    logging.info(f"reply: {text}")
    yield f"data: {json.dumps(json_data)}\n\n"  # NOTE: EventSource
    yield f"data: [DONE]\n\n"  # NOTE: EventSource


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
    logging.info(f"Received request: {completion_request}")

    history_messages = completion_request.messages[:-1]
    user_message = completion_request.messages[-1]

    if completion_request.model == "simple-conversation-chat":
        chat = SimpleConversationChat(history_messages)
    elif completion_request.model == "summary-conversation-chat":
        chat = SummaryConversationChat(history_messages)
    else:
        raise ValueError(f"Unknown model: {completion_request.model}")

    json_data = {
        "id": str(uuid.uuid4()),
        "object": "text_completion",
        "created": int(time.time()),
        "model": completion_request.model,
        "choices": [
            {
                "text": "",
                "index": 0,
                "logprobs": None,
                "finish_reason": "length"
            }
        ]
    }

    if completion_request.stream:
        return StreamingResponse(
            streaming_response(json_data, chat.generator(user_message.content)),
            media_type="text/event-stream"
        )
    else:
        return json_data
    
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
        user=os.getenv("DB_USERNAME"),
        password=os.getenv("DB_PASSWORD"),
        dsn=os.getenv("DB_DSN"),
        config_dir=os.getenv("WALLET_DIR"),
        wallet_location=os.getenv("WALLET_DIR"),
        wallet_password=os.getenv("WALLET_PASSWORD")
    )

    # OCIの設定を環境変数から取得
    embeddings = OCIGenAIEmbeddings(
        model_id=os.getenv("EMBEDDING_MODEL_ID"),
        service_endpoint=os.getenv("OCI_SERVICE_ENDPOINT"),
        compartment_id=os.getenv("OCI_COMPARTMENT_ID"),
        auth_profile=os.getenv("OCI_AUTH_PROFILE")
    )

    vector_store_dot = OracleVS.from_documents(
        docs,
        embeddings,
        client=connection,
        table_name=os.getenv("VECTOR_TABLE_NAME"),
        distance_strategy=DistanceStrategy.DOT_PRODUCT,
    )

    template = """contextに従って回答してください:
    {context}

    質問: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOCIGenAI(
        model_id=os.getenv("LLM_MODEL_ID"),
        service_endpoint=os.getenv("OCI_SERVICE_ENDPOINT"),
        compartment_id=os.getenv("OCI_COMPARTMENT_ID"),
        model_kwargs={
            "temperature": float(os.getenv("LLM_TEMPERATURE", "0.7")),
            "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "500"))
        },
        auth_profile=os.getenv("OCI_AUTH_PROFILE")
    )

    retriever = vector_store_dot.as_retriever()

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )