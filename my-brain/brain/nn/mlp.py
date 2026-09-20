# Phase 8: 다층 함수 합성
# 수학적 이론에 기반: 함수 합성 및 연쇄법칙. 비선형성이 없으면 affine 층들의 합성도 affine이다.
# 유지할 수학: 선택한 구조의 순전파·미분은 정해진다. 깊이와 폭의 최적값은 정리가 정하지 않는다.
# 변경 가능한 구현/설계: sizes, hidden tanh, 마지막 affine 출력, seed는 설계 선택이다.
# AI Developer Experiment Points / 연구·실험: 화두 코드: self.layers와 nonlinear 조건. 깊이/폭/활성화 ablation, 여러 seed의 MSE·일반화 비교.
# 추천 교재/논문: Goodfellow, Bengio, Courville, Deep Learning (2016), https://www.deeplearningbook.org/ — 6장, 8장.
# 역사: 역전파의 핵심은 연쇄법칙이며, 현재 MLP의 모든 설계를 특정 논문이 고정한 것은 아니다.
# 주의: 수학적 정의가 고정되어도 소스코드가 영구 불변이라는 뜻은 아니다.
# 상세 질문 답변·검증: PHASE_01_14_REVIEW.md

# 여러 layer의 합성으로 복잡한 비선형 관계 학습
# Layer -> 함수합성 -> 손실 -> 기존 backward() -> SGD


"""
깊이, 폭, 비선형 유무를 하나씩 변경한다. 여러 seed에서 MSE 성공률을 비교한다. 
합성함수, 과적합, gradient 소실 -> 함수 근사 이론 
Rumelhart, Hinton & Williams, 1986

"""

import random

from brain.nn.layer import Layer


class MLP:
    def __init__(self, sizes, seed=None):
        sizes = tuple(sizes)

        if len(sizes) < 2:
            raise ValueError("입력과 출력 크기를 지정해야합니다.")

        if any(not isinstance(n, int) or n <= 0 for n in sizes):
            raise ValueError("모든 크기는 양의 정수여야합니다.")

        # 하나의 난수 흐름을 고융해 재현성을 확보하며, neuron마다 같은 seed로 다시 시작하지 않는다. 
        rng = random.Random(seed)

        self.layers = [
            Layer(sizes[i], sizes[i+1],nonlinear=(i < len(sizes)-2), rng=rng)
            for i in range(len(sizes) - 1)
        ]

    def forward(self, inputs):
        # h_(l+1) = f_l(h_l)
        # hidden layer는 tanh, 마지막 layer는 affine 출력이다.
        output = inputs
        for layer in self.layers:
            output = layer(output)

        return output

    def __call__(self, inputs):
        return self.forward(inputs)

    def parameters(self):
        return [
            parameter
            for layer in self.layers
            for parameter in layer.parameters()
        ]