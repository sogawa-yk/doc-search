from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
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

# グローバル変数として保持
chain = None

# 環境変数の取得と検証
def get_required_env(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise ValueError(f"Required environment variable {key} is not set")
    return value

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

@app.post("/ask")
async def ask_question(question: Question):
    try:
        answer = chain.invoke(question.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

