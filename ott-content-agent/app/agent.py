# Json 문자열 변환을 위해 사용 
import json

from openai import OpenAI
from app.tools import search_contents, get_user_profile

# Open API 요청을 담당 클라이너드 
# 기능명세:
# - OPENAI_API_KEY 변수에 API키를 읽는다. 
# - Http 요청마다 클라인트 새로 생성 없이 재사용
# openai_client = OpenAI()
MODEL = "gpt-5.6-luna"

MAX_TOOL_CALLS = 3

# 쉼표를 넣지 않는다. 쉼표를 넣으면 tuple이 된다. 안넣으면 문자열이다. 
SYSTEM_INSTRUCTIONS = (
    "너는 OTT 콘텐츠 검색과 추천을 돕는 AI 도우미다. "
    "사용자의 취향 정보가 필요하면 get_user_profile 도구를 사용하라. "
    "콘텐츠 검색이 필요하면 search_contents 도구를 사용하라. "
    "get_user_profile의 사용자 ID는 백엔드가 결정한다. "
    "사용자에게 사용자 ID를 Tool 인자로 요구하지 마라. "
    "콘텐츠는 search_contents가 반환한 결과 안에서만 추천하라. "
    "검색 결과에 없는 콘텐츠를 만들어내지 마라. "
    "현재는 찜 등록과 알림 등록 기능이 없다. "
    "실제로 실행하지 않은 행동을 실행했다고 말하지 마라. "
    "사용자에게 한국어로 간결하게 답하라."
)

# 중요 : 
# 이 dict 가 자동으로 실행하는것은 아님
SEARCH_CONTENTS_TOOL = {
    "type": "function",
    "name": "search_contents",
    "description": (
        "OTT 콘텐츠의 제목, 장르, 줄거리에서 관련 콘텐츠를 검색한다. "
        "사용자가 콘텐츠 검색이나 추천을 요청했을 때 사용한다."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "콘텐츠 검색에 사용할 간결한 검색어. "
                    "예: SF, 우주, 스릴러, 가족 애니메이션"
                ),
            },
        },
        "required": ["query"], # 필수 
        "additionalProperties": False, # false로 설정하면 정의되지 않은 속성은 허용하지 않는다.
    },
    "strict": True, # 모델이 Tool 인자를 Json Schema에 맞게 정확히 전달하도록 강제한다. 
}


GET_USER_PROFILE_TOOL = {
    "type": "function",
    "name": "get_user_profile",
    "description": (
        "현재 사용자의 이름과 선호 장르를 데이터베이스에서 조회한다. "
        "사용자가 자신의 취향에 맞는 추천을 요청했을 때 사용한다."
    ),
    "parameters": {
        "type": "object",

        # user_id는 LLM이 전달하지 않는다.
        #
        # 현재 HTTP 요청의 user_id를 백엔드가 직접 사용하므로
        # Tool arguments는 빈 객체 {}다.
        "properties": {},
        "required": [],
        "additionalProperties": False,
    },
    "strict": True,
}



# Agent가 선택할 수 있는 Tool Schema 목록이다.
#
# Java의 List<ToolDefinition>과 비슷하다.
AGENT_TOOLS = [
    SEARCH_CONTENTS_TOOL,
    GET_USER_PROFILE_TOOL,
]

def create_fixed_answer(_message: str, _user_id: int) -> str:
    """
    OpenAI 를 사용하지 않는 고정응답 메소드
    """
    return "요청을 정상적으로 받았습니다."


def create_agent_response(message: str, user_id: int) -> str:
    """
    Agent 응답 생성 방식을 선택하는 메소드
    """

    answer_method = create_fixed_answer
    # OpenAI API 키가 설정되어 있으면 Agent Loop를 실행하도록 변경
    # answer_method = run_agent_loop

    return answer_method(message, user_id)



def execute_tool(
        tool_name: str, arguments: dict, user_id: int
) -> object:
    """
    Agent가 선택한 Tool을 실행하는 메소드

    기능 명세:
    1. tool_name에 따라 적절한 Tool 함수를 호출
    2. arguments는 Tool이 요구하는 인자 dict
    3. user_id는 현재 HTTP 요청의 사용자 ID
    4. Tool 실행 결과를 반환
    """

    if tool_name == "search_contents":
        return search_contents(
            query=arguments["query"]
        )

    if tool_name == "get_user_profile":
        return get_user_profile(user_id)

    return {
        "error": f"알 수 없는 Tool 이름: {tool_name}"
    }



def run_agent_loop(message: str, user_id: int) -> str:
    """
    OpenAI와 Tool을 이용하여 응답을 생성하는 Agent Loop를 실행한다. 
    """
    openai_client = OpenAI()

    response = openai_client.responses.create(
        model=MODEL,
        instructions=SYSTEM_INSTRUCTIONS,
        input=message,
        tools=AGENT_TOOLS,
        tool_choice="auto",
        # 한응답에서 여러 Tool 호출을 허용하지 않음
        parallel_tool_calls=False,
    )

    tool_call_count = 0

    while True:
        tool_call = next(
            (
                output_item
                for output_item in response.output
                if output_item.type == "function_call"
            ),
            None,
        )

        if tool_call is None:
            if response.output_text:
                return response.output_text

            return "답변을 생성하지 못했습니다."

        # 무한반복 방지를 위한 실행 횟수 검사
        if tool_call_count >= MAX_TOOL_CALLS:
            return "호출 횟수 초과로 답변을 생성하지 못했습니다."

        # 허용된 Tool외 요청을 하지 않음
        if tool_call.name not in [tool["name"] for tool in AGENT_TOOLS]:
            return "현재는 해당 기능만 지원합니다."

        # tool argument 는 Json 문자열로 전달
        arguments = json.loads(tool_call.arguments)

        # LLM query이용한 python 함수 실행 
        tool_result = execute_tool(
            tool_name=tool_call.name,
            arguments=arguments,
            user_id=user_id
        )

        tool_call_count += 1

        # LLM에게 tool 실행 결과를 전달해 최종 답변 생성 
        response = openai_client.responses.create(
            model=MODEL,
            previous_response_id=response.id,
            # previous_response_id를 지정하면 이전 응답의 컨텍스트를 유지하면서 새로운 요청을 처리할 수 있다.
            instructions=SYSTEM_INSTRUCTIONS,

            tools=AGENT_TOOLS,
            tool_choice="auto",
            parallel_tool_calls=False,

            input=[
                {
                    "type": "function_call_output",
                    # 실행 결과가 어떤 tool 요청인지 식별 위한 것 
                    "call_id": tool_call.call_id,
                    "output": json.dumps(
                        tool_result,
                        ensure_ascii=False
                    ),
                }
            ],
        )


