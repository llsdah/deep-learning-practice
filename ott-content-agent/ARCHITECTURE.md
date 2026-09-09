# Media Agent Backend Architecture

## 1. 프로젝트 목표

이 프로젝트의 목적은 Python으로 AI Agent 기반 Backend System을 직접 구현하면서 아래 개념을 코드 수준에서 이해하는 것이다.

```text
LLM
Tool Calling
Agent Loop
RAG
Backend Service
Database
Async Event
Production Guardrail
```

모델을 직접 학습하는 ML 프로젝트가 아니다.

목표 시스템은 기존 LLM을 판단 엔진으로 사용하고,
Backend가 제공하는 Tool과 검색 시스템을 조합하는 Agent Application이다.

---

# 2. Phase 0 - 현재 구조

현재 실제 구현은 이것뿐이다.

```text
Browser / Client
       |
       | HTTP GET /
       v
    Uvicorn
       |
       | ASGI
       v
    FastAPI
       |
       v
    hello()
       |
       v
     "hello"
```

현재 코드:

```text
app/main.py
```

외에는 실행 로직이 없다.

---

# 3. 최종 목표 아키텍처

향후 최종 구조는 다음 방향으로 확장한다.

```text
                        Client
                           |
                           | HTTP
                           v
                     FastAPI API
                           |
                           v
                    Agent Service
                           |
              +------------+------------+
              |                         |
              v                         v
         LLM Gateway                Agent State
              |
              v
          LLM Model
      OpenAI / other LLM
              |
        function_call
              |
              v
        Tool Dispatcher
              |
     +--------+---------+----------+-------------+
     |                  |          |             |
     v                  v          v             v
 Search Tool        User Tool   Watchlist    Notification
     |                  |          |             |
     v                  +----------+-------------+
 RAG Service                       |
     |                             v
     |                         PostgreSQL
     |
     +--> Embedding
     |
     +--> Vector Search
             |
             v
        Vector Store
```

Agent가 직접 DB에 접근하게 하지 않는다.

```text
LLM
 ↓
Tool Definition
 ↓
Tool Dispatcher
 ↓
Application Service
 ↓
Repository / External API
```

라는 Backend 계층을 유지한다.

---

# 4. 핵심 개념별 역할

## LLM

LLM은 판단 엔진이다.

예:

```text
GPT 계열
Claude 계열
Gemini 계열
Llama 계열
```

Backend는 LLM에게 사용자의 자연어와 사용 가능한 Tool Schema를 전달한다.

LLM은 직접 Python 함수를 실행하지 않는다.

LLM은 다음과 같은 결정을 반환한다.

```text
search_contents를 호출해야 한다.
```

실제 함수 실행은 Backend가 담당한다.

---

## Tool

Tool은 LLM에게 호출 가능하도록 공개된 Backend Function이다.

개념:

```text
Python Function
     +
Tool Schema
     =
LLM이 선택할 수 있는 Tool
```

Java/Spring으로 비교하면:

```text
@Controller
   ↓
@Service
   ↓
@Repository
```

기존 구조에서 LLM이 Controller 대신 특정 Application Function의 호출 여부를 판단하는 것에 가깝다.

단, 모든 Service Method를 LLM에 공개하지 않는다.

---

## Agent

Agent는 별도의 AI 모델이 아니다.

```text
Agent
=
LLM
+ Tool
+ Tool Dispatcher
+ Execution Loop
+ State
+ Stop Condition
+ Guardrail
```

핵심 실행 구조:

```text
사용자 요청
   |
   v
LLM 호출
   |
   +-- 일반 답변 ------> 종료
   |
   +-- Tool Call
          |
          v
      Tool 실행
          |
          v
      실행 결과
          |
          v
      다시 LLM
          |
          +-- Tool Call -> 반복
          |
          +-- 최종 답변 -> 종료
```

Python에서는 초기 단계에 이 Loop를 직접 구현한다.

LangGraph나 Agents SDK에 맡기지 않는다.

---

## RAG

RAG는 Agent 자체가 아니다.

```text
RAG
=
Retrieval
+
LLM Context
+
Generation
```

초기 검색:

```text
keyword
→ SQL LIKE / Full Text Search
```

향후 검색:

```text
사용자 문장
→ Embedding
→ Vector Search
→ 관련 문서
→ LLM Context
→ 답변
```

RAG는 Agent의 Tool로 사용할 수도 있고,
LLM 호출 전에 Backend Pipeline으로 실행할 수도 있다.

프로젝트에서는 학습 효과를 위해 먼저 아래처럼 구성한다.

```text
Agent
  |
  +--> search_contents Tool
            |
            v
         RAG Service
```

---

# 5. 단계별 구현 계획

## Phase 0 - Hello Backend

현재 단계.

```text
GET /
→ hello
```

설치:

```text
FastAPI
Uvicorn
```

AI 관련 dependency 없음.

---

## Phase 1 - API Contract

추가 예정:

```text
Pydantic Request Model
Pydantic Response Model
POST /agent
```

아직 LLM 없음.

목표:

```text
Client
→ FastAPI
→ Request Validation
→ Response
```

---

## Phase 2 - LLM Integration

추가 예정:

```text
openai Python SDK
```

흐름:

```text
Client
→ FastAPI
→ LLM Service
→ OpenAI Responses API
→ Client
```

아직 Tool 없음.

---

## Phase 3 - Tool Calling

평범한 Python 함수 작성:

```text
search_contents()
get_content()
```

이후 Tool Schema를 LLM에 전달한다.

```text
LLM
→ function_call
→ Backend
```

---

## Phase 4 - Agent Loop

직접 구현:

```text
LLM
→ Tool Call
→ Python Function
→ Tool Output
→ LLM
→ Tool Call or Final Response
```

여기서 처음으로 Agent System이 된다.

---

## Phase 5 - Database

메모리 데이터를 PostgreSQL로 변경한다.

예정 계층:

```text
Tool
 ↓
Service
 ↓
Repository
 ↓
PostgreSQL
```

Tool이 SQL을 직접 작성하지 않는다.

---

## Phase 6 - Vector Search

콘텐츠 데이터에 Embedding을 추가한다.

```text
Content
→ Embedding
→ Vector
→ Vector Store
```

사용자 query도 Embedding한다.

```text
Query
→ Embedding
→ Similarity Search
```

---

## Phase 7 - RAG

검색 결과를 LLM Context로 전달한다.

```text
Question
 ↓
Retriever
 ↓
Relevant Contents
 ↓
Prompt / Context
 ↓
LLM
 ↓
Answer
```

---

## Phase 8 - Agent + RAG + Write Tool

최종 핵심 학습 단계.

예:

```text
"야구 콘텐츠를 찾아서 하나 찜해줘"
```

흐름:

```text
LLM
 ↓
search_contents Tool
 ↓
RAG
 ↓
검색 결과
 ↓
LLM
 ↓
add_watchlist Tool
 ↓
Database
 ↓
LLM
 ↓
최종 답변
```

---

## Phase 9 - Production Control

여기서부터 Backend Engineering이 중요하다.

추가:

```text
Timeout
Retry
Max Agent Steps
Idempotency
Authentication
Authorization
Rate Limit
Audit Log
Structured Logging
Tracing
Token / Cost Metrics
Tool Allowlist
```

특히 WRITE Tool은 LLM의 parameter를 그대로 신뢰하지 않는다.

예:

```text
잘못된 방식

LLM
→ add_watchlist(user_id=999, content_id=1)
```

개선:

```text
Authenticated Request Context
        |
        +---- user_id
        |
LLM ----+---- content_id
        |
        v
Backend Tool
```

사용자 식별자는 인증 계층에서 주입한다.

---

## Phase 10 - AI Metadata Pipeline

최종 확장.

```text
Content 등록
   |
   v
Kafka
   |
   v
Metadata Worker
   |
   +--> LLM Summary
   |
   +--> Keyword Extraction
   |
   +--> Category Classification
   |
   +--> Embedding
   |
   v
DB / Vector Store
```

여기서는 Agent와 별개의 비동기 AI Pipeline을 경험한다.

---

# 6. 버전 기준

기준일:

```text
2026-08-23
```

## 현재 프로젝트에서 실제 사용하는 버전

| Component | Version | 사용 여부 |
|---|---:|---|
| CPython | 3.14.7 | 현재 |
| FastAPI | 0.141.1 | 현재 |
| Uvicorn | 0.52.4 | 현재 |
| Pydantic | FastAPI dependency | 간접 |
| OpenAI Python SDK | 미설치 | Phase 2 |
| PostgreSQL | 미설치 | Phase 5 |
| Vector Store | 미선정 | Phase 6 |
| Kafka | 미설치 | Phase 10 |
| Redis | 미설치 | 필요 시 |
| LangChain | 사용 안 함 | - |
| LangGraph | 사용 안 함 | - |
| OpenAI Agents SDK | 초기에는 사용 안 함 | - |

## Phase 2 도입 시 기준으로 검토할 버전

2026-08-23 시점의 OpenAI Python SDK 최신 확인 버전:

```text
openai 2.54.0
```

하지만 실제 Phase 2를 시작하는 날 다시 최신 stable을 확인한 뒤 pin한다.

이 프로젝트에서는 아직 의존성에 넣지 않는다.

---

# 7. Python 패키지 관리 형식

## 현재 선택: pyproject.toml

현재 프로젝트는 다음 방식을 사용한다.

```text
pyproject.toml
```

예:

```toml
[project]
dependencies = [
    "fastapi==0.141.1",
    "uvicorn[standard]==0.52.4",
]
```

이 프로젝트에서는 이것을 표준으로 사용한다.

---

## requirements.txt

Python 프로젝트에서 매우 흔하게 볼 수 있는 형식이다.

```text
fastapi==0.141.1
uvicorn[standard]==0.52.4
```

설치:

```bash
pip install -r requirements.txt
```

여전히 사용할 수 있고 배포/운영에서도 많이 존재한다.

그러나 이 학습 프로젝트에서는 dependency metadata를 `pyproject.toml` 한 곳에 두기 위해 별도 생성하지 않는다.

---

## setup.py

과거 Python Package에서 흔히 사용했던 방식.

예:

```python
from setuptools import setup

setup(
    name="media-agent-backend",
    install_requires=[
        "fastapi",
        "uvicorn",
    ],
)
```

새 프로젝트에서는 dependency 선언의 중심을 `pyproject.toml`로 두는 방향을 사용한다.

따라서 이 프로젝트에 `setup.py`는 만들지 않는다.

---

## setup.cfg

`setup.py`의 설정 일부를 선언형으로 분리하던 방식도 존재한다.

예:

```ini
[metadata]
name = media-agent-backend

[options]
install_requires =
    fastapi
    uvicorn
```

이 프로젝트에서는 사용하지 않는다.

---

# 8. ASGI와 구형 WSGI 개념

FastAPI는 ASGI Application이다.

현재:

```text
Uvicorn
  |
 ASGI
  |
FastAPI
```

전통적인 Python Web Application에서는 WSGI 구조도 많이 사용했다.

```text
Gunicorn
  |
 WSGI
  |
Flask / Django
```

단순하게 비교하면:

```text
WSGI
- 전통적인 synchronous request/response 중심

ASGI
- async 지원
- WebSocket
- long-lived connection
- network I/O가 많은 application에 적합
```

LLM API, Streaming, SSE 등 network I/O가 많은 AI Backend에서는
ASGI 기반 FastAPI 구조가 자연스럽다.

---

# 9. Java / Spring과 대응

| Python/FastAPI | Java/Spring |
|---|---|
| Python | Java |
| FastAPI | Spring Boot Web |
| Uvicorn | Embedded Tomcat/Netty와 비슷한 서버 역할 |
| Pydantic Model | Request/Response DTO |
| decorator `@app.get` | `@GetMapping` |
| dependency injection | Spring DI와 목적은 유사하나 방식은 다름 |
| async/await | CompletableFuture/Reactor와 일부 개념 비교 가능 |
| Tool Function | AI에 제한적으로 공개한 Service Function |
| Agent Service | Orchestration Service |
| Repository | Repository/DAO |

완전히 동일한 구현은 아니며 개념 비교용이다.

---

# 10. 초기 디렉터리를 일부러 크게 만들지 않는 이유

처음부터 다음 구조를 만들지 않는다.

```text
controller/
service/
repository/
domain/
agent/
rag/
tool/
llm/
vector/
```

아직 아무 코드도 없는데 디렉터리부터 만드는 것은 과설계다.

현재:

```text
app/
└── main.py
```

로 시작한다.

기능이 실제 생기는 시점에 분리한다.

예상 성장 과정:

```text
Phase 0

app/
└── main.py
```

↓

```text
Phase 2

app/
├── main.py
├── api/
└── llm/
```

↓

```text
Phase 4

app/
├── api/
├── agent/
├── llm/
└── tools/
```

↓

```text
Phase 7+

app/
├── api/
├── agent/
├── llm/
├── tools/
├── rag/
├── domain/
└── infrastructure/
```

즉 디렉터리는 기능이 생긴 뒤 분리한다.

---

# 11. 설계 원칙

이 프로젝트에서 유지할 핵심 원칙:

```text
1. Framework보다 원리를 먼저 이해한다.
2. Agent Loop는 최초에 직접 구현한다.
3. LLM과 Business Logic을 분리한다.
4. Tool은 Allowlist 방식으로 노출한다.
5. LLM이 직접 DB/SQL을 제어하지 않는다.
6. 사용자 권한 정보는 LLM에게 결정시키지 않는다.
7. RAG와 Agent를 동일 개념으로 취급하지 않는다.
8. side-effect Tool에는 idempotency를 고려한다.
9. Agent에는 반드시 종료 조건을 둔다.
10. AI 기능도 일반 Backend처럼 테스트/모니터링한다.
```
