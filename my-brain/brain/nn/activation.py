# Phase 12-13: 활성화와 확률 정규화
# 수학적 이론에 기반: 합성함수 미분, exp/log, sigmoid, tanh, softmax의 공통 이동 불변성.
# 유지할 수학: 같은 함수를 선택하면 도함수도 정해진다. ReLU의 0은 미분 불가능하며 기울기 0은 규약이다.
# 변경 가능한 구현/설계: relu/sigmoid/tanh 선택, temperature, 마지막 축 정규화, 안정화 계산 경로는 설계/구현 선택이다.
# AI Developer Experiment Points / 연구·실험: 화두 코드: 활성화 선택, log_softmax의 정규화 축과 안정화. 포화·엔트로피를 공부하고 중심차분/큰 logits/검증 손실 비교.
# 추천 교재/논문: Goodfellow, Bengio, Courville, Deep Learning (2016), https://www.deeplearningbook.org/ — 4장 Numerical Computation, 6장 Deep Feedforward Networks.
# 현재 유한 logits·작은 그래프가 대상이며 -inf mask는 지원하지 않는다.
# 주의: 수학적 정의가 고정되어도 소스코드가 영구 불변이라는 뜻은 아니다.
# 상세 질문 답변·검증: PHASE_01_14_REVIEW.md

# 여러 linear 를 합쳐 단순 affine 변환으로 축약되지 않도록 비선형성을 넣는다. 

import math

from brain.autograd.value import Value
from brain.tensor.tensor import Tensor


def _unary(value, function, derivative):
    # y = f(x), dL/dx = dL/dy * f'(x)
    # 순전파 시점의 값으로 국소 미분을 보존한다.
    x = value.data
    y = function(x)
    slope = derivative(x, y)
    result = Value(y, (value,))

    def backward():
        value.grad += slope * result.grad

    result._backward = backward
    return result


def _exp(value):
    # d(exp(x))/dx = exp(x)
    return _unary(value, math.exp, lambda x, y: y)


def _log(value):
    if value.data <= 0:
        raise ValueError("log의 입력은 양수여야 합니다.")

    # d(log(x))/dx = 1/x
    return _unary(value, math.log, lambda x, y: 1.0 / x)


# 각원소를 독립적으로 미분
def relu(tensor):
    return Tensor(
        [
            _unary(
                value,
                lambda x: max(0.0, x),
                lambda x, y: 1.0 if x > 0 else 0.0,
            )
            for value in tensor.data
        ],
        tensor.shape,
    )


def _sigmoid_number(x):
    # exp에는 0 이하의 지수만 넣어 overflow를 피한다.
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))

    exponential = math.exp(x)
    return exponential / (1.0 + exponential)


def sigmoid(tensor):
    return Tensor(
        [
            _unary(
                value,
                _sigmoid_number,
                lambda x, y: y * (1.0 - y),
            )
            for value in tensor.data
        ],
        tensor.shape,
    )


def tanh(tensor):
    return Tensor(
        [
            _unary(
                value,
                math.tanh,
                lambda x, y: 1.0 - y * y,
            )
            for value in tensor.data
        ],
        tensor.shape,
    )


# 클래스 점수 또는 attension 점수를 확률로 변환
def log_softmax(tensor):
    if not tensor.shape:
        raise ValueError("정규화할 마지막 축이 필요합니다.")

    if any(not math.isfinite(v.data) for v in tensor.data):
        raise ValueError("현재 구현은 유한한 logits만 지원합니다.")

    width = tensor.shape[-1]
    output = []

    for start in range(0, len(tensor.data), width):
        row = tensor.data[start:start + width]
        maximum = max(value.data for value in row)

        # log p_i = (z_i-m) - log Σ_j exp(z_j-m)
        #
        # 공통 상수 이동에 불변이므로 m은 숫자로 사용한다.
        # 미분할 때도 올바른 log-softmax gradient가 나온다.
        shifted = [value - maximum for value in row]
        total = sum((_exp(value) for value in shifted), Value(0.0))
        log_total = _log(total)

        output.extend(value - log_total for value in shifted)

    return Tensor(output, tensor.shape)


def softmax(tensor):
    # 확률 p = exp(log p).
    # log p <= 0이므로 큰 양의 지수에 의한 overflow를 피한다.
    log_probabilities = log_softmax(tensor)
    return Tensor(
        [_exp(value) for value in log_probabilities.data],
        tensor.shape,
    )