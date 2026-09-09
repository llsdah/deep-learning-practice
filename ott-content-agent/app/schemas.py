from pydantic import BaseModel, Field

"""
    AgentRequest 모델은 FastAPI에서 요청 본문을 검증하고 문서화하는 데 사용되는 Pydantic 모델이다.
"""
class AgentRequest(BaseModel):
    """
    Post / agent 요청 본문의 데이터 주고. 

    기능 명세 : 
    - message는 사용자가 agent에 전달하는 자연어 문장이다. 
    - message는 반드시 요청 JSON에 포함되어야한다. 
    - message는 최소 1자 이상이여야 한다. 
    - 지나치게 큰 요청을 방지하기 위해 최대 1000자로 제한한다.

    예)
    {
        "user_id": 1,
        "message": "안녕, agent야. 오늘 날씨 어때?"
    } 
    """

    user_id: int = Field(
        gt=0,
        description="사용자 ID",
        examples=[1],
    )

    message: str = Field (
        min_length=1,
        max_length=1000,
        description="OTT Agent에게 전달한 사용자의 자연서 요청",
        examples=["주말에 볼만한 SF 영화 추천해줘"],
    )

class AgentResponse(BaseModel):
    """
    Post / agent 응답 본문 데이터 주고. 
    
    기능명세:
    - answer는 agent가 사용자에게 반환하는 문자열이다. 
    - LLM이 없으므로 고정된 안내 문장 반환 
    - 이후 Phase 2에게 이필드에 LLM 답변을 담는다. 

    응답 예:
    {
        "answer": "요청을 정상적으로 받았습니다. "
    }
    """

    answer: str = Field(
        description="Agent가 사용자에게 반환하는 답변",
        examples=["요청을 정상적으로 받았습니다."]
    )