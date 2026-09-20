# Phase 6: 하나의 neuron
# 수학적 이론에 기반: affine 가중합 z=sum(w_i*x_i)+b 및 tanh 미분 1-tanh(z)**2.
# 유지할 수학: 이 함수를 선택한 뒤의 계산식과 도함수는 수학적으로 정해진다.
# 변경 가능한 구현/설계: tanh 선택, bias, 초기화 bound=1/sqrt(input_size)는 설계 선택이다. 초기화는 Xavier 공식 그 자체가 아니다.
# AI Developer Experiment Points / 연구·실험: 화두 코드: bound, nonlinear, tanh(). 초기 분산/포화를 공부하고 gradient 크기와 검증 손실 비교.
# 추천 교재/논문: Goodfellow, Bengio, Courville, Deep Learning (2016), https://www.deeplearningbook.org/ — 6장.
# 논문: Glorot & Bengio (2010), https://proceedings.mlr.press/v9/glorot10a.html
# 문제/아이디어/연결: 깊은 신경망의 포화·분산 문제 → 초기화와 활성화의 신호 전파 분석 → bound 실험.
# 주의: 수학적 정의가 고정되어도 소스코드가 영구 불변이라는 뜻은 아니다.
# 상세 질문 답변·검증: PHASE_01_14_REVIEW.md

# 입력들의 가중합과 비선형 변환 
# Glorot & Bengio, 2010
# 내적, 연쇄법칙, 분산, 포화 -> 신호전파 공부

import math
import random

from brain.autograd.value import Value


def tanh(value):
  # a = tanh(z)
  # da/dz = 1 - a^2
  output = math.tanh(value.data)
  result = Value(output, (value,))

  def backward():
    # dL/dz = dL/da * da/dz
    value.grad += (1.0 - output * output) * result.grad

  result._backward = backward
  return result

class Neuron:
  def __init__(self, input_size, nonlinear=True, rng=None):
    if not isinstance(input_size, int) or input_size <= 0:
      raise ValueError("입력 크기는 양의 정수여야 합니다.")

    rng = rng if rng is not None else random.Random()

    # 입력 수가 커질 때 가중할 규모가 지나치게 커지는 것을 줄이기 위해 초기 범위를 1/sqrt(n) 로 조절
    bound = 1.0 / math.sqrt(input_size)
    self.weights = [
      Value(rng.uniform(-bound, bound))
      for _ in range(input_size)
    ]
    self.bias = Value(0.0)
    self.nonlinear = nonlinear

  def forward(self, inputs):
    if len(inputs) != len(self.weights):
      raise ValueError("입력 길이와 가중치 개수가 다릅니다.")

    # z = sum(w_i * x_i) + b
    # 각 연산이 Value이므로 기준 autograd에 연결
    output = self.bias

    for weight, value in zip(self.weights, inputs):
      output = output + weight * value

    return tanh(output) if self.nonlinear else output

  def __call__(self, inputs):
    return self.forward(inputs)

  def parameters(self):
    return self.weights + [self.bias]
