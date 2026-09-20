# Phase 9: 다차원 배열
# 수학적 이론에 기반: 원소 수 prod(shape), 행 우선 인덱스 offset=i*cols+j, 원소별 연산.
# 유지할 수학: 같은 shape와 인덱스 규약에서는 주소 대응을 보존해야 한다.
# 변경 가능한 구현/설계: 평탄 list, Value 원소, shape tuple, reshape의 객체 공유는 구현 선택이다. 기하학적 tensor 전체 이론을 구현한 것은 아니다.
# AI Developer Experiment Points / 연구·실험: 화두 코드: data, __getitem__, reshape. stride/view·broadcasting·Tensor 단위 역전파는 확장 과제; 성능 최적화가 모두 새 연구는 아니다.
# 추천 교재/논문: Deisenroth, Faisal, Ong, Mathematics for Machine Learning (2020), https://mml-book.github.io/ — 2장(벡터/행렬 기반). Python 객체 공유는 PYTHON_FOR_JAVA.md 참조.
# 주의: 수학적 정의가 고정되어도 소스코드가 영구 불변이라는 뜻은 아니다.
# 상세 질문 답변·검증: PHASE_01_14_REVIEW.md

# 스칼라 여러개를 백터, 행렬, 배치처럼 다루는 것 

import math

from brain.autograd.value import Value


class Tensor:

    def __init__(self, data, shape):
        self.shape = tuple(shape)

        if any( 
            not isinstance(size, int) or size <= 0
            for size in self.shape
        ):
            raise ValueError("각 차원의 크기는 양의 정수여야 합니다.")

        # 기존 Value 그대로 보존해야 미분 연결 유지 가능
        self.data = [
            value if isinstance(value, Value) else Value(value)
            for value in data
        ]

        # 스칼라 shape()의 원소갯수도 math.prod(()) = 1 
        if len(self.data) != math.prod(self.shape):
            raise ValueError("원소 갯수와 shape가 일치하지 않는다.")

    def __repr__(self):
        values = [value.data for value in self.data]
        return f"Tensor(shape={self.shape}, data={values})"

    def __getitem__(self, indices):
        if not isinstance(indices, tuple):
            indices = (indices,)

        if len(indices) != len(self.shape):
            raise IndexError("모든 축의 인덱스를 지정해야합니다.")

        # 행 우선 저장의 인덱스 계산. shape(r,c) 이면 offset = i*c +j
        offset = 0
        for index, size in zip(indices, self.shape):
            if not isinstance(index, int):
                raise TypeError("인덱스는 정수여야 합니다.")

            if index < 0:
                index += size

            if not 0 <= index < size:
                raise IndexError("인덱스가 범위를 벗어났습니다.")

            offset = offset * size + index

        return self.data[offset]

    def reshape(self, shape):
        # 원소의 순서 Value 객체 유지, shape 만 변경 
        # 목록은 새로 만들지만 내부 Value 객체는 공유한다. 원소의 값, 순서, 미분 연결은 그대로 유지 
        return Tensor(self.data, shape)

    def _check_shape(self, other):
        if not isinstance(other, Tensor):
            raise TypeError("다른 피연산자도 Tensor여야합니다.")

        if self.shape != other.shape:
            raise ValueError("원소별 연산은 같은 shape가 필요합니다.")

    def __add__(self, other):
        self._check_shape(other)

        return Tensor(
            [a + b for a,b in zip(self.data, other.data)],
            self.shape,
        )

    def __mul__(self, other):
        self._check_shape(other)

        return Tensor(
            [a * b for a,b in zip(self.data, other.data)],
            self.shape,
        )

    def sum(self):
         # L = Σ_i x_i
        # 각 입력의 국소 미분 dL/dx_i = 1.
        # 반환값은 스칼라 Value이므로 backward()를 호출할 수 있다
        total = Value(0.0)
        for value in self.data:
            total = total + value
        return total
