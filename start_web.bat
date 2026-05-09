@echo off
cd /d %~dp0

echo Starting Milvus with Docker...
docker compose up -d

echo.
echo Starting web server...
start "" cmd /k python main.py --backend milvus --milvus-uri http://127.0.0.1:19530 serve --host 127.0.0.1 --port 8765

timeout /t 3 /nobreak >nul
start "" http://127.0.0.1:8765/

echo.
echo Web UI: http://127.0.0.1:8765/
