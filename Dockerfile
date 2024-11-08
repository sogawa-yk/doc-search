FROM python:3.9-slim

# 作業ディレクトリの設定
WORKDIR /app

# 必要なシステムパッケージのインストール
RUN apt-get update && apt-get install -y \
    libaio1 \
    && rm -rf /var/lib/apt/lists/*

# 環境変数の設定
ENV LD_LIBRARY_PATH=/opt/oracle

# 必要なPythonパッケージをインストール
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# インスタンスウォレットをコピー
COPY wallet /app/wallet

ENV PYTHONUNBUFFERED=1

# スクリプトをコピー
COPY app.py app.py

# 実行
CMD ["python", "app.py"]
