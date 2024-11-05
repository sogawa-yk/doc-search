import oracledb
import os
import gradio as gr
import logging


def execute_query(query_text):
    # Oracle Databaseに接続するためのユーザー情報
    username = os.getenv("DB_USERNAME")
    password = os.getenv("DB_PASSWORD")
    dsn = os.getenv("DB_DSN")
    
    connection = None
    cursor = None
    
    try:
        # データベース接続を確立
        connection = oracledb.connect(user=username, password=password, dsn=dsn, config_dir='/app/wallet', wallet_location='/app/wallet', wallet_password='We1comeOU2023#')
        cursor = connection.cursor()
        
        cursor.callproc('DBMS_CLOUD_AI.SET_PROFILE', ['OCIGENAI_ORACLE'])
        # クエリ実行
        ai_query = f"SELECT AI narrate '{query_text}'"
        cursor.execute(ai_query)
        
        # 結果を取得
        result = cursor.fetchone()
        if result:
            return f"AIの応答: {result[0]}"
        else:
            return "結果が見つかりませんでした。"
    
    except oracledb.DatabaseError as e:
        return f"データベースエラー: {e}"
    
    finally:
        # リソースを解放
        if cursor is not None:
            cursor.close()
        if connection is not None:
            connection.close()

def main():
    logging.basicConfig(level=logging.DEBUG)
    logging.info('アプリケーション開始')
    
    os.environ['ORACLE_HOME'] = "/app"
    logging.info('ウォレットの設定完了')
    
    def chat_interface(message, history):
        logging.debug(f'リクエスト受信: {message}')
        if not message:
            return history
            
        response = execute_query(message)
        logging.debug(f'応答生成: {response}')
        
        # レスポンスを分割
        parts = response.split('Sources:')
        ai_response = parts[0].replace('AIの応答:', '').strip()
        sources = parts[1].strip() if len(parts) > 1 else ""
        
        # Markdownフォーマットで返答
        formatted_response = f"{ai_response}\n\n---\n**参考資料:**\n{sources}"
        history.append((message, formatted_response))
        return history
    
    with gr.Blocks(title="OCI 嘘つきチャットボット") as demo:
        gr.Markdown("# OCI 噓つきチャットボット")
        gr.Markdown("Oracle Databaseを利用したAIチャットボットです。OCIドキュメントから返答します")
        
        chatbot = gr.Chatbot()
        msg = gr.Textbox(
            label="メッセージを入力",
            placeholder="ここに質問を入力してください...",
            lines=2
        )
        
        with gr.Row():
            submit = gr.Button("送信")
            clear = gr.Button("クリア")
        
        with gr.Row():
            gr.Examples(
                examples=[
                    "OKEのノードアップグレード方法は？",
                    "OCIのデータベースサービスについて教えて",
                    "Autonomous Databaseとは何ですか"
                ],
                inputs=msg
            )
        
        submit.click(
            chat_interface,
            inputs=[msg, chatbot],
            outputs=chatbot
        )
        clear.click(lambda: None, None, chatbot, queue=False)
        msg.submit(
            chat_interface,
            inputs=[msg, chatbot],
            outputs=chatbot
        )
    
    logging.info('サーバー起動開始')
    demo.launch(
        server_name="0.0.0.0",
        server_port=80,
        show_api=False,
        show_error=True,
        quiet=False
    )

if __name__ == "__main__":
    main()
