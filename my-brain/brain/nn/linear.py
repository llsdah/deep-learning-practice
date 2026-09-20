# Phase 11: 배치 affine 변환
# 수학적 이론에 기반: Y=XW+b; dL/dW=X^T G, dL/dX=GW^T, dL/db=sum_batch G.
# 유지할 수학: 행 벡터 입력 규약을 유지하면 이 수학적 관계를 보존해야 한다.
# 변경 가능한 구현/설계: input/output 크기, bias, bound, seed는 설계 선택이다. 함수 이름 Linear여도 bias가 있으면 affine이다.
# AI Developer Experiment Points / 연구·실험: 화두 코드: 초기화 bound, self.bias, self.weight.shape. 분산 전파를 공부하고 학습/검증 손실과 파라미터 수 비교.
# 추천 교재/논문: Deisenroth, Faisal, Ong, Mathematics for Machine Learning (2020), https://mml-book.github.io/ — 2장, 5장.
# 초기화 배경: Glorot & Bengio (2010), https://proceedings.mlr.press/v9/glorot10a.html
# 주의: 수학적 정의가 고정되어도 소스코드가 영구 불변이라는 뜻은 아니다.
# 상세 질문 답변·검증: PHASE_01_14_REVIEW.md

import math
import random


from brain.autograd.value import Value
from brain.tensor.tensor import Tensor
from brain.tensor.operations import matmul


class Linear:
    def __init__(self, input_size, output_size, bias=True, seed=None):
        if any( 
            type(size) is not int or size <= 0
            for size in (input_size, output_size)
        ):
            raise ValueError("입력, 출력 크기는 양의 정수여야 합니다.")

        rng = random.Random(seed)
        bound = 1.0 / math.sqrt(input_size)

        self.weight = Tensor(
            [
                rng.uniform(-bound, bound)
                for _ in range(input_size * output_size)
            ],
            (input_size, output_size)
        )
        self.bias = (
            Tensor([0.0] * output_size, (output_size,))
            if bias else None
        )

    def forward(self, inputs):
        if len(inputs.shape) != 2:
            raise ValueError("입력 shape는 (batch, input_size)여야 합니다. ")

        output = matmul(inputs, self.weight)

        if self.bias is None:
            return output

        batch, width = output.shape

        # 따라서 dL/db_j는 각 생플의 미분 기여를 합한 값이다. 
        return Tensor(
            [
                output[i, j] + self.bias[j]
                for i in range(batch)
                for j in range(width)
            ],
            output.shape,
        )

    def __call__(self, inputs):
        return self.forward(inputs)

    def parameters(self):
        parameters = list(self.weight.data)
        if self.bias is not None:
            parameters.extend(self.bias.data)
        return parameters
