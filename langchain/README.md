# langchainサーバ
## 準備
Oracle DatabaseをOCI上にプロビジョニングし、langchainディレクトリは以下にwalletという名前のディレクトリを作成。このディレクトリ内に作成したOracle Databaseのウォレットを配置する。

langchainディレクトリにhtml_dataというディレクトリを作成し、RAGに使用したいhtmlをそのディレクトリに配置する。
## 起動
```bash
docker run -v ~/.oci:/root/.oci -p 8000:8000 langchain
```