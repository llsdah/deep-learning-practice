
from app.database import (
    get_user_profile as load_get_user_profile,
)
# 임시 인메모리 목록
# list + dict 구조 map<String, object> 와 비슷 
CONTENT_CATALOG = [
    {
        "content_id": 1,
        "title": "인터스텔라",
        "genres": ["SF", "드라마"],
        "summary": "우주 탐사를 통해 인류가 살아갈 새로운 행성을 찾는 이야기",
    },
    {
        "content_id": 2,
        "title": "컨택트",
        "genres": ["SF", "미스터리"],
        "summary": "외계 생명체와 소통하기 위해 언어학자가 미지의 언어를 해석하는 이야기",
    },
    {
        "content_id": 3,
        "title": "블레이드 러너 2049",
        "genres": ["SF", "스릴러"],
        "summary": "인간과 복제 인간의 경계를 추적하는 미래 사회 이야기",
    },
    {
        "content_id": 4,
        "title": "기생충",
        "genres": ["드라마", "스릴러"],
        "summary": "서로 다른 두 가족의 만남을 통해 계층 문제를 다룬 이야기",
    },
    {
        "content_id": 5,
        "title": "코코",
        "genres": ["애니메이션", "가족"],
        "summary": "음악을 좋아하는 소년이 죽은 자들의 세상에서 가족의 비밀을 찾는 이야기",
    },
]

#LLM에게 search_content 함수의 존재와 사용방법을 알려줄 tool search 
def search_contents(query: str) -> list[dict]:
    """
    콘텐츠 제목, 장르, 줄거리에서 검색어와 관련 항목을 조회 
    
    기능 명세 :
    1. query 앞뒤 불요 공백 제거 
    2. 대소문자 차이 업이 검색 -> 소문자 변환
    3. 여러 검색 단어 입력시 공백 기준 분리 
    4. 검색 단어 중 하나라도 포함된 콘텐츠 결과 추가 
    5. 검색 결과 최대 5개 반환
    6. 일치하는 콘텐츠가 없다면 빈리스트 반환

    매게변수 : 
    query: str
        LLM이 콘텐츠 검색에 필요하다고 판단한 검색어. 
        예) "SF 영화", "드라마", "인류", "우주 탐사"

    반환값 : 
        검색된 콘텐츠 dict 의 list
        
    """

    #
    normalized_query = query.strip().lower()

    # ["가족","애니메이션"] 형태로 분리 
    keywords = normalized_query.split()

    # 결과 저장 용 
    result = []

    for content in CONTENT_CATALOG:
        # 제목 장르 줄거리를 하나의 문자열로 합친다. 
        searchable_text = " ".join(
            [
                content["title"].lower(),
                " ".join(content["genres"]),
                content["summary"],
            ]
        ).lower()

        # 검색단어 중 하나라도 콘텐츠 문자열에 들어 있는지 검색 

        has_matching_keyword = any(
            keyword in searchable_text for keyword in keywords
        )

        if has_matching_keyword:
            result.append(content)


    return result[:5]



def get_user_profile(user_id: int) -> dict | None:
    return load_get_user_profile(user_id)
