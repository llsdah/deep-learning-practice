# Phase 5: 기울기 기반 최적화
# 수학적 이론에 기반: Taylor 1차 근사와 최급강하. theta_new = theta - lr * grad.
# 유지할 수학: 이는 선택한 SGD 알고리즘의 정의이지 모든 최적화 방법의 유일한 정답이 아니다.
# 변경 가능한 구현/설계: lr, batch 크기, schedule, momentum, Adam 선택은 변경 가능하다. 이 파일은 기본 업데이트만 제공한다.
# AI Developer Experiment Points / 연구·실험: 화두 코드: step()의 업데이트와 self.lr. 곡률/확률적 gradient를 공부하고 동일 데이터·seed에서 손실/시간 비교.
# 추천 교재/논문: Boyd & Vandenberghe, Convex Optimization, 9장, https://web.stanford.edu/~boyd/cvxbook/
# Goodfellow, Bengio, Courville, Deep Learning (2016), https://www.deeplearningbook.org/ — 8장.
# 질문 답: PHASE_01_14_REVIEW.md의 Q1/Q2 참조. eta가 충분히 작다는 가정 없이 수렴을 보장하지 않는다.
# 주의: 수학적 정의가 고정되어도 소스코드가 영구 불변이라는 뜻은 아니다.
# 상세 질문 답변·검증: PHASE_01_14_REVIEW.md

# 기울기로 파라미터 업데이트 
# backward() = 기울기 계산, 실제 학습은 기술기를 가중치로 바꿀대 발생 
# 같은 업데이크 규칙도, 전체 데이터 손실을 사용하면 batch gradient descent, 무작위 샘플이나 미니배치의 손실을 쓰면 SGD 방식

"""
질문
1. 손실학습률 공식이 어떻게 발생했는지?
2. 학습률이 크면 이근사가 맞지 않아 발산한다? 왜 그런공식을 채택 했는지 수렵하게 하면되지 않는지 


답변 요약
1. 손실은 오차를 재는 목적함수의 선택이다. 음의 기울기 업데이트는 Taylor 1차 근사로 유도한다.
2. L=(w-3)^2에서 오차는 e_next=(1-2*lr)*e이므로 0<lr<1에서 수렴한다.
   lr=1은 진동, lr>1은 발산한다. 자세한 유도와 line search 대안은 PHASE_01_14_REVIEW.md 참고.

초점
학습률을 바꾸고 손실 감소, 진동, 발산을 비교한다. 
기울기, 학습률, 곡률 및 확률적 최적화 수렵을 공부하고 - 배경인 최급강하법 확인 
"""

import math


class SGD:
    def __init__(self, parameters, lr=0.01):
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError("학습률은 양의 유한한 값이어야 합니다.")

        self.parameters = list(dict.fromkeys(parameters))
        self.lr = lr


    def zero_grad(self):
        for parameter in self.parameters:
            parameter.grad = 0.0


    def step(self):
        for parameter in self.parameters:
            # θ_new = θ - η * dL/dθ
            # 미분 결과로 현재 값을 갱신한다.
            # 업데이트 자체는 계산 그래프에 연결하지 않는다.
            # 업데이트 후 새로운 순전파로 다시 그래프를 만든다. 
            parameter.data -= self.lr * parameter.grad

