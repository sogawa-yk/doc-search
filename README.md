# doc-search
コンテナを起動する前に、以下の環境変数を設定してください:

- `DB_USERNAME`: データベースに接続するためのユーザー名を指定します。
- `DB_PASSWORD`: データベースに接続するためのパスワードを指定します。
- `DB_DSN`: データベースのDSN（Data Source Name）を指定します。これはデータベースの接続情報を含む文字列です。
- `TNS_ADMIN`: OracleデータベースのTNS（Transparent Network Substrate）設定ファイルのディレクトリパスを指定します。

これらの環境変数を設定した後、コンテナを起動してください。