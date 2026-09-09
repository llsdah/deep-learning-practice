# Media Agent Backend

현재 단계는 **AI 기능을 전혀 구현하지 않은 실행 가능한 최소 뼈대**다.

실행 결과:

```text
GET /
→ hello
```

## 현재 실제 설치 대상

- Python 3.14.7
- FastAPI 0.141.1
- Uvicorn 0.52.4

OpenAI SDK, DB, Vector DB, Kafka, Redis, Agent Framework는 아직 설치하지 않는다.

## 구조

```text
media-agent-hello/
├── app/
│   ├── __init__.py
│   └── main.py
├── .python-version
├── .gitignore
├── pyproject.toml
├── README.md
├── ARCHITECTURE.md
└── AGENTS.md
```

## 실행

### Windows PowerShell

```powershell
py -3.14 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
uvicorn app.main:app --reload --port 8000
```

브라우저:

```text
http://127.0.0.1:8000/
```

결과:

```text
hello
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

## 지금 하지 않는 것

- LLM 호출
- Tool
- Agent Loop
- RAG
- Database
- Kafka
- Redis
- Docker
- LangChain
- LangGraph
- OpenAI Agents SDK

각 단계는 직접 코드를 작성하면서 하나씩 추가한다.

자세한 향후 설계는 `ARCHITECTURE.md`를 참고한다.
Codex 작업 규칙은 `AGENTS.md`를 참고한다.
