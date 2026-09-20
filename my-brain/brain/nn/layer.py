# Phase 7: 여러 neuron의 병렬 구성
# 수학적 이론에 기반: 벡터 함수 h_j=phi(sum_i x_i*W_ij+b_j), 파라미터 수 I*O+O.
# 유지할 수학: 정한 구조의 출력과 파라미터 개수는 정해지지만 구조 자체는 선택이다.
# 변경 가능한 구현/설계: output_size, nonlinear, 파라미터 공유 유무를 바꿀 수 있다.
# AI Developer Experiment Points / 연구·실험: 화두 코드: self.neurons 생성. 폭과 표현력을 공부하고 seed별 검증 손실/계산량 비교.
# 추천 교재/논문: Goodfellow, Bengio, Courville, Deep Learning (2016), https://www.deeplearningbook.org/ — 6장. Deisenroth, Faisal, Ong, Mathematics for Machine Learning (2020), https://mml-book.github.io/ — 2장.
# 주의: 수학적 정의가 고정되어도 소스코드가 영구 불변이라는 뜻은 아니다.
# 상세 질문 답변·검증: PHASE_01_14_REVIEW.md

# neuron을 병렬로 구성 


import random

from brain.nn.neuron import Neuron


class Layer:
    def __init__(self, input_size, output_size, nonlinear=True, rng=None):
        if not isinstance(output_size, int) or output_size <= 0:
            raise ValueError("출력크기는 양의 정수여야 합니다.")

        rng = rng if rng is not None else random.Random()

        # 같은 입력으로 다른 m개의 특징을 계산하며, 각 Neuron은 독립적인 가중치 객체를 가진다. 
        self.neurons = [
            Neuron(input_size, nonlinear=nonlinear, rng=rng)
            for _ in range(output_size)
        ]

    def forward(self, inputs):
        # h_j = φ(Σ_i x_i * W_ij + b_j)
        # 하나의 축력이여도 리스트 유지 
        return [neuron(inputs) for neuron in self.neurons]

    def __call__(self, inputs):
        return self.forward(inputs)

    def parameters(self):
        return [
            parameter
            for neuron in self.neurons
            for parameter in neuron.parameters()
        ]
