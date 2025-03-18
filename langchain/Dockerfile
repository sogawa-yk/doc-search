FROM python:3.9-slim

# 作業ディレクトリの設定
WORKDIR /app

# 必要なシステムパッケージのインストール
RUN apt-get update && apt-get install -y \
    libaio1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 環境変数の設定
ENV LD_LIBRARY_PATH=/opt/oracle
ENV PYTHONUNBUFFERED=1

# 必要なPythonパッケージをインストール
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# インスタンスウォレットをコピー
COPY wallet /app/wallet

# スクリプトをコピー
COPY app /app/app
COPY html_data /app/html_data/
COPY .env .

# ポートの設定
EXPOSE 8000

# PYTHONPATHを設定
ENV PYTHONPATH=/app

# FastAPIアプリケーションを起動
CMD ["uvicorn", "app.server.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
