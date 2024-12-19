import json
import os
import time
import uuid
from typing import List
from dotenv import load_dotenv

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
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

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

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
    try:
        for chunk in chat_generator:
            chunk_response = {
                "id": json_data["id"],
                "object": "chat.completion.chunk",
                "created": json_data["created"],
                "model": json_data["model"],
                "choices": [{
                    "index": 0,
                    "delta": {
                        "content": chunk
                    },
                    "finish_reason": None
                }]
            }
            logging.debug(f"Sending chunk: {chunk_response}")
            yield f"data: {json.dumps(chunk_response)}\n\n"
    except Exception as e:
        logging.error(f"Error in streaming_response: {e}", exc_info=True)
        raise


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
    # ログレベル確認
    logger.debug(f"Current logger level: {logger.getEffectiveLevel()}")

    logger.info(f"Received chat completion request with model: {completion_request.model}")
    logger.debug(f"Full request details: {completion_request.dict()}")
    
    try:
        history_messages = completion_request.messages[:-1]
        user_message = completion_request.messages[-1]
        logger.debug(f"History messages: {history_messages}")
        logger.debug(f"Current user message: {user_message}")

        if completion_request.model == "simple-conversation-chat":
            logger.info("Initializing SimpleConversationChat")
            chat = SimpleConversationChat(history_messages)
        elif completion_request.model == "summary-conversation-chat":
            logger.info("Initializing SummaryConversationChat")
            chat = SummaryConversationChat(history_messages)
        else:
            logger.error(f"Invalid model requested: {completion_request.model}")
            raise ValueError(f"Unknown model: {completion_request.model}")

        json_data = {
            "id": str(uuid.uuid4()),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": completion_request.model,
        }
        logger.debug(f"Created response metadata: {json_data}")

        # ジェネレータのテスト
        logger.info("Creating response generator")
        generator = chat.generator(user_message.content)
        logger.debug("Testing first response from generator")
        try:
            first_chunk = next(generator)
            if first_chunk is not None:
                logger.info(f"Successfully received first chunk: {first_chunk[:100]}...")
            else:
                logger.error("Received empty first chunk from OCI Generative AI")
                raise HTTPException(
                    status_code=500,
                    detail="Failed to get response from language model"
                )
        except StopIteration:
            logger.error("Generator returned no content")
            raise ValueError("No response from model")
        except Exception as e:
            logger.error(f"Unexpected error getting first chunk: {str(e)}", exc_info=True)
            raise

        if completion_request.stream:
            logger.info("Initiating streaming response")
            async def stream_with_logging():
                try:
                    # 初期レスポンスのロギング
                    initial_response = {
                        "id": json_data["id"],
                        "object": "chat.completion.chunk",
                        "created": json_data["created"],
                        "model": json_data["model"],
                        "choices": [{
                            "index": 0,
                            "delta": {"role": "assistant"},
                            "finish_reason": None
                        }]
                    }
                    logger.debug(f"Sending initial streaming response: {initial_response}")
                    yield f"data: {json.dumps(initial_response)}\n\n"

                    # 以降の各チャンクの詳細なロギング
                    chunk_count = 0
                    for chunk in [first_chunk] + list(generator):
                        chunk_count += 1
                        logger.debug(f"Processing chunk {chunk_count}: {chunk[:50]}...")
                        chunk_response = {
                            "id": json_data["id"],
                            "object": "chat.completion.chunk",
                            "created": json_data["created"],
                            "model": json_data["model"],
                            "choices": [{
                                "index": 0,
                                "delta": {"content": chunk},
                                "finish_reason": None
                            }]
                        }
                        yield f"data: {json.dumps(chunk_response)}\n\n"

                    logger.info(f"Streaming completed. Total chunks sent: {chunk_count}")
                    # 終了レスポンスのロギング
                    final_response = {
                        "id": json_data["id"],
                        "object": "chat.completion.chunk",
                        "created": json_data["created"],
                        "model": json_data["model"],
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop"
                        }]
                    }
                    logger.debug("Sending final streaming response")
                    yield f"data: {json.dumps(final_response)}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    logger.error(f"Error during streaming: {str(e)}", exc_info=True)
                    raise

            return StreamingResponse(
                stream_with_logging(),
                media_type="text/event-stream"
            )
        else:
            logger.info("Processing non-streaming response")
            response_text = first_chunk + "".join(list(generator))
            json_data["choices"] = [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }]
            logger.debug(f"Sending non-streaming response: {json_data}")
            return json_data

    except ValueError as e:
        logger.error(f"ValueError occurred: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": str(e),
                    "type": "server_error",
                    "code": "model_error"
                }
            }
        )
    except Exception as e:
        logger.error(f"Unexpected error occurred: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": "An unexpected error occurred",
                    "type": "server_error",
                    "code": "internal_error"
                }
            }
        )
    
@app.on_event("startup")
async def startup_event():
    logger.debug(f"Starting application with log level: {logger.getEffectiveLevel()}")
    logger.debug("Root logger level: %s", logging.getLogger().getEffectiveLevel())
    logger.info("Starting application initialization")
    try:
        global chain
        
        logger.info("Loading HTML data")
        loader = UnstructuredHTMLLoader("html_data/container-instance.html")
        documents = loader.load()
        logger.debug(f"Loaded {len(documents)} documents")

        logger.info("Splitting text")
        text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=10)
        docs = text_splitter.split_documents(documents)
        logger.debug(f"Created {len(docs)} text chunks")

        logger.info("Initializing database connection")
        connection = oracledb.connect(
            user=os.getenv("DB_USERNAME"),
            password=os.getenv("DB_PASSWORD"),
            dsn=os.getenv("DB_DSN"),
            config_dir=os.getenv("WALLET_DIR"),
            wallet_location=os.getenv("WALLET_DIR"),
            wallet_password=os.getenv("WALLET_PASSWORD")
        )
        logger.info("Database connection established")

        logger.info("Initializing embeddings")
        embeddings = OCIGenAIEmbeddings(
            model_id=os.getenv("EMBEDDING_MODEL_ID"),
            service_endpoint=os.getenv("OCI_SERVICE_ENDPOINT"),
            compartment_id=os.getenv("OCI_COMPARTMENT_ID"),
            auth_profile=os.getenv("OCI_AUTH_PROFILE")
        )

        logger.info("Creating vector store")
        vector_store_dot = OracleVS.from_documents(
            docs,
            embeddings,
            client=connection,
            table_name=os.getenv("VECTOR_TABLE_NAME"),
            distance_strategy=DistanceStrategy.DOT_PRODUCT,
        )
        logger.info("Vector store created successfully")

        logger.info("Initializing LLM and chain")
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
        logger.info("Application initialization completed successfully")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}", exc_info=True)
        raise

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url}")
    response = await call_next(request)
    return response

@app.post("/api/chat")
async def chat(request: Request):
    logger.info("Received chat request")
    try:
        # リクエストボディの取得
        body = await request.json()
        logger.debug(f"Request body: {body}")

        # OpenAI API���式のリクエストを内部形式に変換
        completion_request = CompletionRequest(
            model=body.get("model", "simple-conversation-chat"),
            messages=[Message(**msg) for msg in body.get("messages", [])],
            max_tokens=body.get("max_tokens", 500),
            temperature=body.get("temperature", 0.7),
            stream=body.get("stream", False)
        )

        # 既存のchat_completionsエンドポイントを再利用
        return await chat_completions(completion_request)

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": "An error occurred during chat processing",
                    "type": "server_error",
                    "code": "internal_error",
                    "details": str(e)
                }
            }
        )

@app.get("/health")
async def health_check():
    return {"status": "healthy"}