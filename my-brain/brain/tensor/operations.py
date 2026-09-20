# Phase 10: 내적, 전치, 행렬 곱
# 수학적 이론에 기반: dot=sum_i a_i*b_i; C_ij=sum_k A_ik*B_kj; transpose(A)_ji=A_ij.
# 유지할 수학: 표준 행렬 곱을 구현하는 한 이 원소별 정의와 shape 조건은 유지해야 한다.
# 변경 가능한 구현/설계: 세 반복문 순서, 저장 배치, 병렬화는 변경 가능하다. 부동소수점 결과는 합산 순서로 달라질 수 있다.
# AI Developer Experiment Points / 연구·실험: 화두 코드: matmul의 i/j/k loop. 복잡도·수치오차를 공부하고 손계산/gradient/시간 비교. 새로운 고속 알고리즘과 단순 튜닝을 구분한다.
# 추천 교재/논문: Deisenroth, Faisal, Ong, Mathematics for Machine Learning (2020), https://mml-book.github.io/ — 2장 Linear Algebra, 3장 Analytic Geometry.
# 주의: 수학적 정의가 고정되어도 소스코드가 영구 불변이라는 뜻은 아니다.
# 상세 질문 답변·검증: PHASE_01_14_REVIEW.md

# 여러 가중합을 행렬 연산으로 표현 내적, 전치, 행렬 곱

# 공통 실행·검증 방법 은 어떤 테스트 코드의 샘플로 이름이 지어서 만드는 것으로 하자 

from brain.autograd.value import Value
from brain.tensor.tensor import Tensor


def dot(a,b):
    if len(a.shape) != 1 or len(b.shape) != 1:
        raise ValueError("dot은 1차원 Tensor 두개 사용합니다.")

    if a.shape != b.shape:
        raise ValueError("백터 길이가 같아야합니다.")

    result = Value(0.0)
    for i in range(a.shape[0]):
        result = result +a[i] *b[i]

    return result


def transpose(matrix):
    if len(matrix.shape) != 2:
        raise ValueError("transpose는 2차원 Tensor를 받습니다.")

    rows, cols = matrix.shape

    # B[j,i] = A[i,j], 숫자를 복사하지 않고 동일한 Value를 재배치한다.
    values = [
        matrix[i, j]
        for j in range(cols)
        for i in range(rows)
    ]

    return Tensor(values, (cols, rows))

def matmul(a, b):
    if len(a.shape) != 2 or len(b.shape) != 2:
        raise ValueError("matmul은 2차원 Tensor 두개를 받습니다.")

    rows, inner = a.shape
    other_inner, cols = b.shape

    if inner != other_inner:
        raise ValueError("A의 열 수와 B의 행 수가 같아야합니다.")

    values = []
    # A:(rows,inner), B:(inner,cols) -> C:(rows,cols)
    #
    # C[i,j] = Σ_k A[i,k] * B[k,j]
    # 각 곱셈과 덧셈을 Value로 수행하여 autograd를 재사용한다.

    for i in range(rows):
        for j in range(cols):
            total = Value(0.0)

            for k in range(inner):
                total = total + a[i, k] * b[k, j]

            values.append(total)

    return Tensor(values, (rows, cols))