"""
OTT Content Agent Backend의 FastAPI 진입점.
 
현재 학습 단계: Phase 4 agent Loop 구현

기능 명세:
1. Get 서버 실행상태 확인
2. Post agent 요청으로 사용자의 자여어 문장 전달받는다. .
3. Pydantic 통해 데이터 검증 
4. 로컬 컨텐츠 목록에서 제목 장르 줄거리 검색
5. OpenAI Function Calling 으로 search_contents Tool을 선택
6. LLM 판단과 Tool 실행을 반복하는 Agent Loop를 직접 구현
7. OpenAI 미사용 상태에서는 고정된 문장 반환
8. 한줄의 주석 처리 여부로 OpenAI 사용 여부 전화



현재 단계에서 구현하지 않는 기능:
- 사용자 정보 조회 
- 콘텐스 DB 검색
- 찜 및 알림 등록
- Function Calling
- Tool Dispatcher 
- Agent Loop
- RAG

"""

# FastAPI는 Python에서 HTTP API 서버를 만드는 웹 프레임워크다.
# Java/Spring의 @SpringBootApplication, @RestController 등의 기능을 제공한다.
from fastapi import FastAPI

# 문자열 응답이 JSON 형식이 아닌 text/plain 형식으로 전송되도록 사용한다.
# Java/Spring에서 ResponseEntity<String> 또는 produces = "text/plain"을
# 지정하는 것과 비슷한 역할을 한다.
from fastapi.responses import PlainTextResponse

from app.schemas import AgentRequest, AgentResponse
from app.agent import create_agent_response
from app.database import initialize_database


# FastAPI 애플리케이션 객체를 생성한다.
#
# Java/Spring의 SpringApplication.run(...)으로 만들어지는
# 애플리케이션 컨텍스트와 비슷한 출발점이다.
#
# title:
#   Swagger 문서에 표시되는 API 서버 이름이다.
#
# version:
#   현재 백엔드 애플리케이션의 버전 정보다.
app = FastAPI(
    title="Media Agent Backend",
    version="0.1.0",
)

initialize_database()


@app.get(
    "/",
    response_class=PlainTextResponse,
)
async def hello() -> str:
    """
    서버 실행 여부를 확인하는 기본 API.

    HTTP 요청:
        GET /

    HTTP 응답:
        상태 코드: 200 OK
        Content-Type: text/plain
        본문: hello

    현재는 별도의 요청 데이터나 비즈니스 로직 없이
    고정된 문자열만 반환한다.
    """

    # Python 함수의 반환값이 HTTP 응답 본문이 된다.
    return "hello"




@app.post(
    "/agent",
    response_model=AgentResponse,
)
# 동기로 변경 
def agent(request: AgentRequest) -> AgentResponse:
    """
사용자 요청을 분석하고 필요한 경우 콘텐츠 검색 Tool을 한 번 실행한다.

    처리 순서:
    1. Pydantic이 사용자 요청을 검증한다.
    2. 사용자 문장과 search_contents Tool Schema를 LLM에 전달한다.
    3. LLM이 Tool 사용 여부를 판단한다.
    4. Tool 요청이 없으면 LLM의 일반 답변을 바로 반환한다.
    5. Tool 요청이 있으면 arguments JSON을 Python dict로 변환한다.
    6. 백엔드가 실제 search_contents Python 함수를 실행한다.
    7. Tool 실행 결과를 JSON 문자열로 변환해 LLM에 전달한다.
    8. LLM이 검색 결과를 사용자용 자연어 답변으로 정리한다.
    9. 최종 답변을 AgentResponse로 반환한다.

    Phase 3 제한:
    - Tool은 search_contents 하나만 제공한다.
    - 한 요청에서 Tool을 최대 한 번만 실행한다.
    - 병렬 Tool 호출은 허용하지 않는다.
    - 반복 Agent Loop는 아직 구현하지 않는다.
    - 사용자 정보 조회 및 쓰기 기능은 없다.
    """

    answer = create_agent_response(
        message=request.message, 
        user_id=request.user_id
        )

    # token 이 없는 경우 현재
     
    return AgentResponse(
        answer=answer
    )


