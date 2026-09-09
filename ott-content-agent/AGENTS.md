# Codex 작업 지침

이 프로젝트는 AI Agent Backend를 학습하기 위한 프로젝트다.

## 가장 중요한 규칙

- 사용자가 요청한 단계만 구현한다.
- 다음 단계 코드를 미리 작성하지 않는다.
- 한 번에 가능한 한 하나의 개념만 추가한다.
- 사용자가 직접 코드를 따라 작성할 수 있도록 변경 이유를 설명한다.
- Python 코드는 Java/Spring Backend 개념과 비교하여 설명한다.
- 현재 프로젝트 구조를 불필요하게 복잡하게 만들지 않는다.

## Agent Framework 사용 금지

학습 초기에는 아래 프레임워크를 사용하지 않는다.

- LangChain
- LangGraph
- OpenAI Agents SDK

Agent Loop와 Tool Dispatcher는 직접 Python으로 구현한다.

## 단계별 목표

Phase 0:
- FastAPI 실행
- `GET /` → `hello`

Phase 1:
- Request/Response DTO
- Pydantic 이해

Phase 2:
- OpenAI SDK 추가
- LLM 단일 호출

Phase 3:
- Python Tool 함수
- Function Calling

Phase 4:
- Agent Loop 직접 구현

Phase 5:
- DB Tool

Phase 6:
- Embedding / Vector Search

Phase 7:
- RAG

Phase 8:
- Agent + RAG + Write Tool

Phase 9:
- Timeout / Retry / Idempotency / Authorization / Observability

Phase 10:
- Kafka 기반 AI Metadata Pipeline

## 코드 생성 정책

사용자가 예를 들어

> 다음은 Tool을 구현하자

라고 요청하기 전에는 Tool 관련 파일을 만들지 않는다.

사용자가

> RAG를 붙이자

라고 요청하기 전에는 Vector DB, Embedding 관련 의존성을 추가하지 않는다.

즉, 항상 현재 학습 단계의 최소 코드만 유지한다.
