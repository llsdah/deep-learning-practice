# Phase 14: Cross Entropy
# 수학적 이론에 기반: categorical 최대우도와 음의 로그우도 L=-mean(log p_target); logits 미분은 (p-one_hot)/N.
# 유지할 수학: 이 손실과 평균 규약을 선택하면 위 식을 유지한다. 목적함수의 선택 자체는 고정되지 않는다.
# 변경 가능한 구현/설계: mean/sum, ignore_index, class weighting, label smoothing 등은 변경 가능한 학습 목표/정책이다.
# AI Developer Experiment Points / 연구·실험: 화두 코드: valid 구성, total/len(valid). 우도·정보이론을 공부하고 정확도/NLL 및 불균형 클래스별 성능 비교.
# 추천 교재/논문: Goodfellow, Bengio, Courville, Deep Learning (2016), https://www.deeplearningbook.org/ — 3장, 5장, 6장.
# log_softmax는 log(0) 문제를 줄이지만 모든 float 범위의 계산을 보장하지 않는다.
# 주의: 수학적 정의가 고정되어도 소스코드가 영구 불변이라는 뜻은 아니다.
# 상세 질문 답변·검증: PHASE_01_14_REVIEW.md

from brain.autograd.value import Value
from brain.nn.activation import log_softmax


# 정답에 높은 확률을 부여하도록 모델 학습 
# SGD - 정답 점수 높으로 오답점수를 낮추는 방향으로
# 최대우도 → 음의 로그우도 → 안정적인 log-softmax → 스칼라 손실.
# inear는 affine 사상, Cross Entropy는 최대우도와 정보이론에 연결됩니다. 활성화 포화와 초기화가 학습에 미치는 영향은 Glorot & Bengio, 2010을 참고할 수 있습니다. 
def cross_entropy_from_logits(logits, targets, ignore_index=None):
    if len(logits.shape) != 2:
        raise ValueError("logits shape는 (batch, classes)여야 합니다.")

    batch, classes = logits.shape
    targets = list(targets)

    if len(targets) != batch:
        raise ValueError("샘플 수와 정답 개수가 다릅니다.")

    valid = []

    for row, target in enumerate(targets):
        if type(target) is not int:
            raise TypeError("정답은 정수 클래스 ID여야 합니다.")

        if ignore_index is not None and target == ignore_index:
            continue

        if not 0 <= target < classes:
            raise ValueError("정답 클래스 ID가 범위를 벗어났습니다.")

        valid.append((row, target))

    if not valid:
        raise ValueError("손실을 계산할 유효 샘플이 없습니다.")

    log_probabilities = log_softmax(logits)

    # L = -(1/N) Σ_b log p[b, target_b]
    # 확률을 먼저 계산한 뒤 log를 취하지 않아 underflow를 피한다.
    total = Value(0.0)
    for row, target in valid:
        total = total - log_probabilities[row, target]

    return total / len(valid)