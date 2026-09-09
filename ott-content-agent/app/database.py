
# SQL lite3 관계형 DB  
import sqlite3
from pathlib import Path

# 
DATABASE_FILE = (
    Path(__file__).resolve().parent.parent / "ott_content_agent.db"
)

def get_database_connection() -> sqlite3.Connection:
    """
    SQLite 데이터베이스에 연결하고 연결 객체를 반환한다.
    """
    connection = sqlite3.connect(DATABASE_FILE)

    connection.row_factory = sqlite3.Row  # 결과를 dict 형태로 반환하도록 설정

    return connection


def initialize_database() -> None:
    """
    사용자 프로필 테이블과 하습용 데이터 초기화 

    기능 명세 : 
    1. SQLite 데이터베이스에 연결
    2. user_profiles 테이블이 존재하지 않으면 생성
    3. 테스트용 사용자 데이터 저장
    4. 같은 사용자 ID 존재시 중복저장 안함
    5. with 블록이 끝나면 transaction이 자동 커밋되거나 롤백

    """

    connection = get_database_connection()

    try:

        # with 사용시 정상 종료 수행되면 commit 
        with connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    preferred_genres TEXT NOT NULL
                )
                """
            )

            # insert or merge 중복시 추가하지 않음
            connection.executemany(
                """
                INSERT OR IGNORE INTO user_profiles (
                    user_id, 
                    name, 
                    preferred_genres
                )
                VALUES (?, ?, ?)
                """,
                [
                    (1, "Alice", "SF, 드라마"),
                    (2, "Bob", "스릴러, 미스터리"),
                    (3, "Charlie", "애니메이션, 가족"),
                ]
            )
    finally:
        connection.close()




def get_user_profile(user_id: int) -> dict | None:
    """
    사용자 ID로 사용자 프로필을 조회한다.

    기능 명세:
    1. SQLite 데이터베이스에 연결
    2. user_profiles 테이블에서 user_id로 조회
    3. 조회 결과가 없으면 None 반환
    4. 조회 결과가 있으면 dict 형태로 반환

    매개변수:
        user_id: int
            조회할 사용자 ID

    반환값:
        dict | None
            사용자 프로필이 존재하면 dict, 없으면 None
    """

    connection = get_database_connection()

    try:

        user_row = connection.execute(
            """
            SELECT 
                user_id,
                name,
                preferred_genres
            FROM user_profiles
            WHERE user_id = ?
            """,
            (user_id,),
        ).fetchone()
    finally:
        connection.close()

    if user_row is None:
        return None

    return {
        "user_id": user_row["user_id"],
        "name": user_row["name"],
        # 장르 구분자 ","
        "preferred_genres": [
            genre.strip() for genre in user_row["preferred_genres"].split(",")
        ]
    }


initialize_database()