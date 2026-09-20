# my-brain — Implementation Roadmap

## 프로젝트 목표

`my-brain`은 AI 모델의 내부 수학과 학습 과정을 Python Standard Library로 직접 구현하는 프로젝트다.

최종 목표는 모델을 실행하는 데 그치지 않고 다음을 설명하고 실험할 수 있는 것이다.

- 연산으로 계산 그래프가 만들어지는 과정
- 연쇄법칙으로 파라미터의 기울기를 계산하는 과정
- 기울기로 파라미터를 학습시키는 과정
- 벡터와 행렬이 표현을 변환하는 방식
- Attention이 문맥을 반영하는 방식
- 언어 모델과 추천 모델의 학습 목표 차이
- 구조·손실 함수·최적화 방법 변경에 따른 효과

전체 학습 경로는 다음과 같다.

```text
수학 기초
→ Scalar / Computational Graph
→ Derivative / Autograd / SGD
→ Neuron / Layer / MLP
→ Tensor / Matrix / Linear
→ Activation / Softmax / Cross Entropy
→ Embedding
→ Attention / Transformer
→ Tokenizer / Language Model
→ Training / Generation
→ Recommendation
→ OTT Dataset
→ Natural Language Recommendation
```

## 구현 원칙

### 사용 도구

초기 구현은 `math`, `random`, `typing`, `unittest`, `csv`, `json` 등 Python 표준 라이브러리로 진행한다.

NumPy, PyTorch, TensorFlow, JAX, SciPy, SymPy, Keras, Transformers는 초기 구현에 사용하지 않는다.

라이브러리 비교는 해당 기능의 직접 구현과 검증이 끝난 뒤 별도 단계에서 수행한다.

### 진행 단위

각 단계는 다음 순서로 진행한다.

```text
개념
→ 수학적 정의
→ 유도
→ 손으로 계산하는 예제
→ 작은 구현
→ 테스트
→ AI에서의 역할
→ 변경 실험
```

로드맵에 등장하는 클래스와 함수는 구현 예정 인터페이스다. 한 번에 모두 구현하지 않는다.

새 코드 파일을 설명할 때는 다음 항목을 포함한다.

1. Why
2. AI Usage
3. Mathematical Intuition
4. Mathematical Definition
5. Derivation
6. Relationship / Mathematical Relation
7. Code
8. Execution Example
9. Mathematical Test
10. AI Developer Experiment Points
11. What To Study Next
12. Research Connection

### 수학 표기와 shape 규칙

행렬 구현에서는 **입력 샘플을 행으로 배치**한다.

$$
Y=XW+b
$$

| 기호 | 의미 | Shape |
|---|---|---|
| $X$ | 입력 배치 | $(B,d_{\text{in}})$ |
| $W$ | 학습할 가중치 | $(d_{\text{in}},d_{\text{out}})$ |
| $b$ | 출력 차원별 편향 | $(d_{\text{out}},)$ |
| $Y$ | 출력 배치 | $(B,d_{\text{out}})$ |

열벡터 표기의 $y=Wx+b$와 혼용하지 않는다.

Transformer에서는 다음 기호를 사용한다.

- $B$: 배치 크기
- $T$: 토큰 수
- $D$: 모델 표현 차원
- $H$: Attention head 수
- $V$: 어휘 크기

### Scalar에서 Tensor로 확장하는 방법

첫 Tensor는 `Value` 객체를 담는 구조로 구현한다.

```text
Value: 숫자 하나와 미분 연결
    ↓
Tensor: 여러 Value와 shape
    ↓
Matrix Operations: Value 연산을 조합
    ↓
기존 backward() 재사용
```

이렇게 하면 Tensor를 도입하면서 자동미분을 새로 만드는 부담을 줄일 수 있다. Tensor 단위 backward와 메모리 최적화는 이후 별도 과제로 다룬다.

### 공통 테스트 기준

| 기준 | 확인할 내용 |
|---|---|
| Mathematical Correctness | 정의·유도·손계산과 일치하는가 |
| Numerical Correctness | 허용 오차 내에서 정확하고 안정적인가 |
| Edge Cases | 잘못된 shape, 경계값, 빈 입력을 명확히 처리하는가 |
| Gradient Correctness | 해석적 미분 및 수치 미분과 일치하는가 |
| Learning Behavior | 학습 가능한 작은 문제에서 실제로 개선되는가 |

미분 대상이 아닌 tokenizer나 데이터 분할에는 gradient test를 적용하지 않는다. 대신 연결된 모델의 통합 테스트로 학습 영향을 확인한다.

수치 미분은 중심차분을 사용한다.

$$
g_{\text{numeric}}
=
\frac{f(x+h)-f(x-h)}{2h}
$$

비교는 절대오차와 상대오차를 함께 사용한다.

$$
|g_{\text{auto}}-g_{\text{numeric}}|
\le
\text{atol}
+
\text{rtol}\max(|g_{\text{auto}}|,|g_{\text{numeric}}|)
$$

미분 불가능한 점은 별도 테스트로 분리한다. $h$를 지나치게 작게 하면 부동소수점 오차가 커질 수 있으므로 여러 크기로 확인한다.

### 공통 연구 실험 규칙

각 실험에는 다음을 기록한다.

- 가정과 변경할 요소
- 변경 이유와 예상 효과
- 필요한 수학
- 고정한 조건과 random seed
- 평가 지표
- 계산 시간과 메모리 부담
- 실패 사례
- 특정 요소를 제거하는 ablation 결과

학습 데이터와 검증 데이터를 구분하고, 최종 테스트 데이터는 모델 선택에 사용하지 않는다.

---

## Phase 0 — Mathematical Foundation

1. **Why**  
   코드가 표현하는 함수와 변화율을 이해하기 위한 공통 언어를 마련한다.

2. **AI Usage**  
   순전파는 함수 계산, 역전파는 미분, 학습은 최적화, 출력 해석은 확률에 연결된다.

3. **Mathematics / Mathematical Relation**  
   함수 → 합성함수 → 미분 → 편미분 → 연쇄법칙을 먼저 공부한다. 벡터·행렬·확률은 이후 단계에서 다시 확장한다. 출발 예제는 $L(w)=(wx-y)^2$이다.

4. **Implementation**  
   손계산 예제와 수학 노트를 설계한다. 아직 모델 코드는 구현하지 않는다.

5. **Prerequisite**  
   사칙연산, 지수, 간단한 방정식, Python 변수와 함수.

6. **Next**  
   실수 하나를 저장하는 `Value`가 어떤 수학적 대상을 표현하는지 이해한다.

7. **Testing**  
   $x=2,y=3,w=1$이면 $L=1$, $dL/dw=-4$임을 유도한다. $w=1.01$에서 손실이 $0.9604$로 감소하는지 확인한다.

8. **Experiment / AI Developer Experiment Points**  
   $w$의 이동량을 바꾸어 선형 근사의 유효 범위를 살핀다. Taylor 근사를 공부하고, 예측한 손실 변화와 실제 변화를 비교한다.

9. **What To Study Next**  
   **Must Know:** 함수, 기울기, 합성함수. **Good To Know:** Taylor 전개, 벡터. **Research Level:** 수치해석과 최적화 이론.

10. **Research / Historical Reference**  
    고전 미적분학의 미분과 연쇄법칙이 기반이다. 특정 논문을 재현하는 단계가 아니라 이후 알고리즘의 수학적 토대를 만드는 단계다.

---

## Phase 1 — Scalar Value

1. **Why**  
   숫자와 그 숫자의 학습 관련 정보를 함께 관리할 최소 단위를 만든다.

2. **AI Usage**  
   가중치 하나, 편향 하나, 중간 계산 결과, 최종 손실을 표현한다.

3. **Mathematics / Mathematical Relation**  
   $v\in\mathbb R$. `data`는 현재 값이며, 이후 `grad`는 선택한 출력 $L$에 대한 $\partial L/\partial v$를 담는다. 값과 기울기는 다른 정보다.

4. **Implementation**  
   `Value(data)`와 `data`, `grad`부터 시작한다. 덧셈과 곱셈은 각각 별도 작은 단계로 추가한다.

5. **Prerequisite**  
   Phase 0의 실수와 함수 개념.

6. **Next**  
   연산 결과가 어떤 입력으로 만들어졌는지 기록하여 계산 그래프로 확장한다.

7. **Testing**  
   정수·실수·음수·0의 저장과 변환, 초기 `grad=0`, 객체별 상태 독립성을 확인한다. 학습 테스트는 Phase 5에서 연결한다.

8. **Experiment / AI Developer Experiment Points**  
   입력값의 크기를 바꾸어 부동소수점 표현 한계를 관찰한다. 유효숫자를 공부하고 큰 수에 작은 수를 더했을 때의 오차를 비교한다.

9. **What To Study Next**  
   **Must Know:** 실수와 변수. **Good To Know:** 부동소수점. **Research Level:** 수치적 조건수.

10. **Research / Historical Reference**  
    프로그램의 연산에 미분 정보를 결합하는 접근은 자동미분과 연결된다. 배경은 [Baydin et al., 자동미분 개관](https://jmlr.org/papers/v18/17-468.html)을 참고한다.

---

## Phase 2 — Computational Graph

1. **Why**  
   최종 출력에 이르는 연산 의존관계를 보존해야 기울기를 전달할 수 있다.

2. **AI Usage**  
   순전파에서 그래프를 만들고 역전파에서 역순으로 탐색한다.

3. **Mathematics / Mathematical Relation**  
   $a=xy,\;L=a+x$를 방향 비순환 그래프로 표현한다. 하나의 입력이 여러 경로에 기여할 수 있다.

4. **Implementation**  
   `Value`에 부모 노드와 연산 정보를 추가한다. `trace_graph()`, `topological_sort()`를 설계한다. 그래프 출력은 표준 라이브러리 기반 텍스트로 시작한다.

5. **Prerequisite**  
   Phase 1의 덧셈·곱셈 결과 객체.

6. **Next**  
   그래프의 각 연결에 국소 미분을 대응시킨다.

7. **Testing**  
   공유 노드가 있는 그래프, `x*x`, 독립적인 두 그래프를 확인한다. 노드 방문 중복 제거와 동일 피연산자의 미분 기여 누적을 구분한다.

8. **Experiment / AI Developer Experiment Points**  
   재귀 탐색과 명시적 스택을 비교한다. 위상정렬과 계산 복잡도를 공부하고 긴 연산 체인의 실행 시간·재귀 한계를 측정한다.

9. **What To Study Next**  
   **Must Know:** 그래프, 의존관계. **Good To Know:** DFS, 위상정렬. **Research Level:** 그래프 변환과 계산 스케줄링.

10. **Research / Historical Reference**  
    자동미분은 계산을 기본 연산으로 분해하고 미분 규칙을 조합한다. 이 단계의 그래프는 그 실행 순서를 보존한다. [자동미분 개관](https://jmlr.org/papers/v18/17-468.html)

---

## Phase 3 — Manual Derivative

1. **Why**  
   자동미분이 계산해야 하는 정답을 먼저 손으로 이해한다.

2. **AI Usage**  
   연산별 backward 규칙을 설계하고 디버깅하는 기준이 된다.

3. **Mathematics / Mathematical Relation**  
   $u=wx-y,\;L=u^2$이면
   $$
   \frac{\partial L}{\partial w}
   =
   \frac{\partial L}{\partial u}
   \frac{\partial u}{\partial w}
   =
   2ux=2(wx-y)x.
   $$
   여러 경로의 기여는 더한다.

4. **Implementation**  
   덧셈·곱셈·거듭제곱의 국소 미분을 유도하고 작은 예제에서 직접 기울기를 계산한다.

5. **Prerequisite**  
   Phase 2의 계산 그래프.

6. **Next**  
   수동으로 수행한 기울기 전달을 `backward()`로 자동화한다.

7. **Testing**  
   $L=x^2+x$에서 $dL/dx=2x+1$을 확인한다. $x=3$이면 기울기는 7이다. 손계산과 중심차분을 비교한다.

8. **Experiment / AI Developer Experiment Points**  
   중심차분 간격 $h$를 바꾼다. 절단오차와 반올림오차를 공부하고 해석적 미분과의 오차 곡선을 비교한다.

9. **What To Study Next**  
   **Must Know:** 편미분, 연쇄법칙. **Good To Know:** 전미분, Jacobian. **Research Level:** 방향미분과 미분 가능한 프로그램.

10. **Research / Historical Reference**  
    핵심은 고전 미적분학의 연쇄법칙이다. 신경망 역전파도 이 법칙을 연산 의존관계에 맞춰 반복 적용한다.

---

## Phase 4 — Automatic Differentiation

1. **Why**  
   파라미터마다 미분식을 직접 작성하는 일을 자동화한다.

2. **AI Usage**  
   하나의 손실에서 모든 학습 파라미터의 기울기를 구한다.

3. **Mathematics / Mathematical Relation**  
   노드 $v$의 출력 방향 이웃을 $u$라 하면
   $$
   \frac{\partial L}{\partial v}
   =
   \sum_u
   \frac{\partial L}{\partial u}
   \frac{\partial u}{\partial v}.
   $$
   출력의 초기 기울기는 $\partial L/\partial L=1$이다.

4. **Implementation**  
   연산별 backward 동작, 역위상순회, 기울기 누적, `backward()`를 구현한다. 이후 뺄셈·부호 반전·상수 지수 거듭제곱·나눗셈을 확장한다.

5. **Prerequisite**  
   Phase 3의 국소 미분과 경로별 기여 합산.

6. **Next**  
   계산한 기울기를 SGD의 업데이트에 사용한다.

7. **Testing**  
   공유 노드, `x*x`, 상수 연산, 깊은 그래프를 검사한다. 0으로 나누기와 거듭제곱 정의역을 확인한다. 반복 backward와 gradient 초기화의 동작을 명시한다.

8. **Experiment / AI Developer Experiment Points**  
   잘못된 대입과 올바른 누적을 비교하는 실패 예제를 만든다. 다변수 연쇄법칙을 공부하고 공유 노드가 있는 식에서 gradient check로 차이를 검증한다.

9. **What To Study Next**  
   **Must Know:** 역방향 자동미분. **Good To Know:** Vector-Jacobian Product. **Research Level:** 고차 미분, checkpointing.

10. **Research / Historical Reference**  
    자동미분은 수치 차분과 달리 기본 연산의 미분을 조합한다. 우리 구현에서는 연산별 미분 규칙과 역순 순회가 대응한다. [자동미분 개관](https://jmlr.org/papers/v18/17-468.html)

---

## Phase 5 — Gradient Descent / SGD

1. **Why**  
   기울기를 이용해 손실을 줄이는 방향으로 파라미터를 바꾼다.

2. **AI Usage**  
   모든 학습 가능한 가중치와 편향의 업데이트.

3. **Mathematics / Mathematical Relation**  
   $$
   \theta'=\theta-\eta\nabla L(\theta).
   $$
   1차 근사에서
   $$
   L(\theta-\eta g)\approx L(\theta)-\eta\|g\|^2
   $$
   이므로 충분히 작은 이동에서는 감소를 기대할 수 있다. 큰 학습률에는 이 근사가 보장되지 않는다.

4. **Implementation**  
   `SGD`, `step()`, `zero_grad()`, 작은 선형회귀 학습 예제를 설계한다. 전체 배치에서 시작해 샘플·미니배치 업데이트로 확장한다.

5. **Prerequisite**  
   Phase 4의 정확한 기울기.

6. **Next**  
   단순 스칼라 함수 대신 neuron의 파라미터를 학습한다.

7. **Testing**  
   $L=(w-3)^2$에서 $w=0,\eta=0.1$이면 다음 $w=0.6$이다. 손실 감소와 반복 학습 수렴을 확인한다.

8. **Experiment / AI Developer Experiment Points**  
   학습률과 배치 크기를 변경한다. 곡률과 gradient 분산을 공부하고 수렴 속도·진동·발산·최종 손실을 비교한다.

9. **What To Study Next**  
   **Must Know:** gradient descent, 학습률. **Good To Know:** 확률적 추정, convexity. **Research Level:** 확률적 최적화의 수렴 분석.

10. **Research / Historical Reference**  
    Gradient descent는 고전적인 최급강하법에, SGD는 확률적 근사에 연결된다. 여기서는 음의 기울기 방향 이동을 직접 구현한다.

---

## Phase 6 — Neuron

1. **Why**  
   여러 입력을 학습 가능한 하나의 출력으로 결합한다.

2. **AI Usage**  
   신경망의 기본 계산 단위.

3. **Mathematics / Mathematical Relation**  
   $$
   z=\sum_i w_ix_i+b,\qquad a=\phi(z).
   $$
   먼저 affine 계산을 확인하고, 작은 비선형 함수 하나를 추가한다.

4. **Implementation**  
   `Neuron`, `forward()`, `parameters()`를 구현한다. 비선형성은 직접 미분 가능한 `tanh` 한 종류로 시작하고 Phase 12에서 일반화한다.

5. **Prerequisite**  
   Value 연산과 SGD.

6. **Next**  
   여러 neuron을 병렬로 구성하여 Layer를 만든다.

7. **Testing**  
   $x=(1,2),w=(3,4),b=-1$이면 $z=10$이다. 가중치와 입력의 미분을 확인하고 작은 회귀 문제를 학습한다.

8. **Experiment / AI Developer Experiment Points**  
   편향 유무를 바꾼다. affine 변환을 공부하고 원점을 지나지 않는 목표 함수를 학습할 때의 오차를 비교한다.

9. **What To Study Next**  
   **Must Know:** 내적, affine 함수. **Good To Know:** 결정 경계. **Research Level:** 표현력과 inductive bias.

10. **Research / Historical Reference**  
    고전 perceptron의 가중합 아이디어와 연결된다. 이 프로젝트는 미분 가능한 활성화와 gradient 기반 학습을 사용한다.

---

## Phase 7 — Layer

1. **Why**  
   같은 입력에서 여러 특징을 동시에 계산한다.

2. **AI Usage**  
   신경망의 hidden representation 생성.

3. **Mathematics / Mathematical Relation**  
   $$
   h_j=\phi\left(\sum_i x_iW_{ij}+b_j\right).
   $$
   출력 neuron 수가 표현 차원이 된다.

4. **Implementation**  
   `Layer`, neuron 목록, `forward()`, `parameters()`를 구현한다.

5. **Prerequisite**  
   Phase 6의 Neuron.

6. **Next**  
   Layer를 순차 연결하여 MLP를 만든다.

7. **Testing**  
   출력 길이, 입력 차원 오류, 파라미터 개수 $nm+m$, neuron 간 파라미터 독립성을 검사한다.

8. **Experiment / AI Developer Experiment Points**  
   layer 폭을 바꾼다. 선형독립성과 표현 차원을 공부하고 학습·검증 손실, 파라미터 수, 실행 시간을 비교한다.

9. **What To Study Next**  
   **Must Know:** 벡터 함수. **Good To Know:** 기저와 rank. **Research Level:** 표현 차원과 일반화.

10. **Research / Historical Reference**  
    다층 신경망의 층별 표현 구성에 해당한다. 여러 가중합을 묶는 것이 이후 행렬 곱 구현으로 이어진다.

---

## Phase 8 — MLP

1. **Why**  
   비선형 함수를 합성하여 단일 neuron보다 복잡한 관계를 학습한다.

2. **AI Usage**  
   분류·회귀 모델과 Transformer FFN의 기반.

3. **Mathematics / Mathematical Relation**  
   $$
   f(x)=f_L(\cdots f_2(f_1(x))).
   $$
   비선형성이 없으면 여러 affine layer의 합성도 하나의 affine 변환으로 축약된다.

4. **Implementation**  
   `MLP`, layer 연결, 파라미터 수집을 구현한다. 초기 손실은 직접 작성한 MSE를 사용한다.

5. **Prerequisite**  
   Phase 7의 Layer와 Phase 5의 SGD.

6. **Next**  
   스칼라 반복 구조를 Tensor와 행렬 연산으로 표현한다.

7. **Testing**  
   XOR 같은 작은 문제를 학습하고 단일 선형 모델과 비교한다. 작은 데이터 과적합, 전체 파라미터 gradient check를 수행한다.

8. **Experiment / AI Developer Experiment Points**  
   깊이·폭·비선형성 유무를 변경한다. 함수 합성과 gradient 전파를 공부하고 여러 seed에서 성공률과 학습 곡선을 비교한다.

9. **What To Study Next**  
   **Must Know:** 비선형 함수 합성. **Good To Know:** 과적합, bias–variance. **Research Level:** 함수 근사 이론.

10. **Research / Historical Reference**  
    다층 신경망의 표현 학습과 역전파에 연결된다. 우리 구현에서는 layer 합성과 기존 autograd의 결합으로 실현한다.

---

## Phase 9 — Tensor

1. **Why**  
   여러 스칼라를 shape가 있는 하나의 데이터 구조로 다룬다.

2. **AI Usage**  
   배치, 특징 벡터, 토큰 시퀀스, 가중치 표현.

3. **Mathematics / Mathematical Relation**  
   계산용 Tensor는 다차원 배열로 다룬다. Shape가 $(d_1,\ldots,d_k)$이면 원소 수는 $\prod_i d_i$다.

4. **Implementation**  
   `Tensor`, `shape`, 인덱싱, `reshape()`, 원소별 연산, `sum()`을 구현한다. 내부 원소는 기존 `Value`를 사용한다.

5. **Prerequisite**  
   Scalar autograd와 MLP의 반복 계산 경험.

6. **Next**  
   Tensor 위에 내적·전치·행렬 곱을 구현한다.

7. **Testing**  
   비정형 중첩 리스트, 잘못된 reshape, 원소 수 보존, 합산 미분을 확인한다. reshape가 미분 연결을 유지하는지 검사한다.

8. **Experiment / AI Developer Experiment Points**  
   중첩 리스트와 평탄 저장 구조를 비교한다. 인덱스 사상을 공부하고 동일 결과·gradient를 유지하면서 접근 비용을 측정한다.

9. **What To Study Next**  
   **Must Know:** shape, 축, 인덱스. **Good To Know:** stride, 메모리 배치. **Research Level:** Tensor contraction.

10. **Research / Historical Reference**  
    다차원 배열과 다중 인덱스 표기에 기반한다. 기하학적 tensor의 일반 정의와 구현상의 배열 구조는 구별한다.

---

## Phase 10 — Matrix Operations

1. **Why**  
   여러 가중합을 일반적인 연산으로 표현한다.

2. **AI Usage**  
   Linear layer, Attention, embedding projection.

3. **Mathematics / Mathematical Relation**  
   $$
   C_{ij}=\sum_{k=1}^{n}A_{ik}B_{kj},
   \quad
   (m,n)(n,p)\rightarrow(m,p).
   $$

4. **Implementation**  
   `dot()`, `transpose()`, `matmul()`을 Python 반복문으로 구현한다. 2차원에서 시작하여 배치 연산을 확장한다.

5. **Prerequisite**  
   Phase 9의 Tensor.

6. **Next**  
   행렬 곱과 편향을 묶어 Linear layer를 만든다.

7. **Testing**  
   $$
   \begin{bmatrix}1&2\\3&4\end{bmatrix}
   \begin{bmatrix}5&6\\7&8\end{bmatrix}
   =
   \begin{bmatrix}19&22\\43&50\end{bmatrix}.
   $$
   항등행렬·직사각형 행렬·차원 오류·입력 양쪽의 미분을 검증한다.

8. **Experiment / AI Developer Experiment Points**  
   반복문 순서를 변경한다. 합의 결합 순서와 계산 복잡도를 공부하고 수치 오차·실행 시간을 비교한다.

9. **What To Study Next**  
   **Must Know:** 내적과 행렬 곱. **Good To Know:** 선형사상, rank. **Research Level:** 행렬 분해와 고속 곱셈.

10. **Research / Historical Reference**  
    고전 선형대수의 행렬 곱에 해당한다. 우리 코드의 세 중첩 반복문이 원소별 정의를 그대로 계산한다.

---

## Phase 11 — Linear Layer

1. **Why**  
   입력 표현을 학습 가능한 새로운 좌표로 변환한다.

2. **AI Usage**  
   hidden layer, Q/K/V 생성, 어휘 출력 점수 계산.

3. **Mathematics / Mathematical Relation**  
   $$
   Y=XW+b.
   $$
   $G=\partial L/\partial Y$이면
   $$
   \frac{\partial L}{\partial X}=GW^\top,\quad
   \frac{\partial L}{\partial W}=X^\top G,\quad
   \frac{\partial L}{\partial b}=\sum_{\text{batch}}G.
   $$

4. **Implementation**  
   `Linear`, 가중치 초기화, bias broadcasting, 파라미터 조회를 구현한다.

5. **Prerequisite**  
   Phase 10의 행렬 연산.

6. **Next**  
   Linear 뒤에 적용할 활성화 함수를 정리한다.

7. **Testing**  
   기존 Layer의 affine 출력과 일치하는지 검사한다. batch별 bias gradient 합산과 모든 shape를 확인한다.

8. **Experiment / AI Developer Experiment Points**  
   초기화 분산과 bias 사용 여부를 바꾼다. 분산 전파를 공부하고 activation·gradient 분포와 학습 안정성을 비교한다.

9. **What To Study Next**  
   **Must Know:** affine 변환, 행렬 미분. **Good To Know:** 초기화와 분산. **Research Level:** 특이값과 신호 전파.

10. **Research / Historical Reference**  
    선형대수의 affine 사상이 기반이다. bias가 있으므로 엄밀하게는 순수 선형사상과 구별한다.

---

## Phase 12 — Activation Functions

1. **Why**  
   선형 변환의 합성만으로 표현할 수 없는 함수를 학습한다.

2. **AI Usage**  
   MLP와 Transformer FFN.

3. **Mathematics / Mathematical Relation**  
   $$
   \operatorname{ReLU}(x)=\max(0,x),\qquad
   \sigma(x)=\frac1{1+e^{-x}}.
   $$
   Sigmoid와 tanh의 미분, 포화 구간, ReLU의 0에서의 미분 처리 규칙을 유도한다.

4. **Implementation**  
   `relu()`, `sigmoid()`, `tanh()`를 구현한다. `exp()` 등 필요한 기본 연산의 backward도 직접 정의한다.

5. **Prerequisite**  
   Linear와 스칼라 미분 규칙.

6. **Next**  
   개별 원소 변환에서 벡터 전체를 확률로 바꾸는 Softmax로 확장한다.

7. **Testing**  
   큰 양수·음수에서 overflow를 검사한다. 매끄러운 지점의 gradient check와 ReLU 0의 별도 규칙 테스트를 수행한다.

8. **Experiment / AI Developer Experiment Points**  
   동일한 MLP에서 활성화만 교체한다. 도함수와 포화를 공부하고 gradient 크기·죽은 ReLU 비율·검증 손실을 비교한다.

9. **What To Study Next**  
   **Must Know:** 지수함수와 도함수. **Good To Know:** 포화와 비매끄러운 함수. **Research Level:** 활성화의 신호 전파 분석.

10. **Research / Historical Reference**  
    Sigmoid는 logistic 함수, tanh는 쌍곡함수와 연결된다. 각 함수의 역사와 신경망에서의 채택은 구분하여 다룬다.

---

## Phase 13 — Softmax

1. **Why**  
   여러 점수를 합이 1인 양의 확률로 변환한다.

2. **AI Usage**  
   다음 토큰 분포와 Attention 가중치.

3. **Mathematics / Mathematical Relation**  
   $$
   p_i=\frac{e^{z_i-m}}{\sum_j e^{z_j-m}},
   \qquad m=\max_j z_j.
   $$
   공통 상수를 빼도 분포는 변하지 않는다.
   $$
   \frac{\partial p_i}{\partial z_j}=p_i(\delta_{ij}-p_j).
   $$

4. **Implementation**  
   `softmax()`와 `logsumexp()`를 구현한다. 정규화 축을 명시한다.

5. **Prerequisite**  
   지수함수, 합산, 자동미분.

6. **Next**  
   정답 확률을 평가하는 Cross Entropy와 결합한다.

7. **Testing**  
   합이 1인지, 상수 이동 불변성이 있는지, 큰 logits에서도 유한한지 검사한다. 빈 입력과 전부 masked인 입력의 처리 규칙을 정한다.

8. **Experiment / AI Developer Experiment Points**  
   $\operatorname{softmax}(z/\tau)$의 temperature를 바꾼다. 엔트로피를 공부하고 분포의 집중도와 gradient 크기를 비교한다.

9. **What To Study Next**  
   **Must Know:** 확률 정규화. **Good To Know:** Jacobian, log-sum-exp. **Research Level:** 지수족과 calibration.

10. **Research / Historical Reference**  
    지수 정규화와 Gibbs 형태의 분포에 연결된다. 이 단계에서는 특정 모델의 관습보다 확률화와 수치 안정성을 유도한다.

---

## Phase 14 — Cross Entropy

1. **Why**  
   모델이 정답에 부여한 확률을 학습 가능한 손실로 바꾼다.

2. **AI Usage**  
   다중 클래스 분류와 다음 토큰 예측.

3. **Mathematics / Mathematical Relation**  
   $$
   L=-\sum_i y_i\log p_i.
   $$
   One-hot 정답 $t$이면
   $$
   L=-z_t+\log\sum_j e^{z_j},\qquad
   \frac{\partial L}{\partial z_i}=p_i-y_i.
   $$
   최대우도를 최대화하는 것이 음의 로그우도를 최소화하는 것과 같음을 유도한다.

4. **Implementation**  
   `cross_entropy_from_logits()`, `log()`, batch 평균, padding 제외 규칙을 구현한다.

5. **Prerequisite**  
   Phase 13의 Softmax와 log-sum-exp.

6. **Next**  
   Embedding을 포함한 모델의 파라미터를 분류 손실로 학습한다.

7. **Testing**  
   균등분포 손실이 $\log V$인지, 정답 logit 증가 시 손실이 감소하는지, gradient가 $p-y$인지 확인한다.

8. **Experiment / AI Developer Experiment Points**  
   합산과 평균 reduction을 비교한다. gradient 스케일을 공부하고 batch 크기 변화가 업데이트 크기에 미치는 영향을 확인한다.

9. **What To Study Next**  
   **Must Know:** 로그우도. **Good To Know:** 엔트로피와 KL divergence. **Research Level:** proper scoring rule.

10. **Research / Historical Reference**  
    정보이론의 cross entropy와 통계학의 최대우도 추정이 연결된다. 구현에서는 정답 logit과 log-sum-exp의 차이로 계산한다.

---

## Phase 15 — Embedding

1. **Why**  
   이산 ID를 학습 가능한 연속 벡터로 표현한다.

2. **AI Usage**  
   토큰, 사용자, 영화의 표현.

3. **Mathematics / Mathematical Relation**  
   $$
   E\in\mathbb R^{V\times D},\qquad e_t=E[t].
   $$
   One-hot 벡터와 $E$의 곱은 해당 행을 선택하는 것과 같다.

4. **Implementation**  
   `Embedding`, 행 선택, 반복 ID의 gradient 누적을 구현한다. 이 단계에서는 인공적인 정수 ID로 시험한다.

5. **Prerequisite**  
   Tensor와 미분 가능한 인덱싱.

6. **Next**  
   선택한 벡터에서 Q/K/V를 만든다. 실제 텍스트 ID는 Phase 22에서 연결한다.

7. **Testing**  
   ID 범위 오류, 반복 ID의 gradient 합산, 사용하지 않은 행의 gradient를 검사한다.

8. **Experiment / AI Developer Experiment Points**  
   embedding 차원을 바꾼다. 거리와 내적을 공부하고 검증 손실·유사도·파라미터 수를 비교한다.

9. **What To Study Next**  
   **Must Know:** one-hot과 행 선택. **Good To Know:** 벡터 공간과 유사도. **Research Level:** 표현 공간의 식별 가능성.

10. **Research / Historical Reference**  
    Bengio et al.은 단어 표현과 언어 확률을 함께 학습하여 일반화를 개선하는 접근을 제시했다. 우리 구현에서는 학습 가능한 embedding 행에 대응한다. [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/v3/bengio03a.html)

---

## Phase 16 — Query / Key / Value

1. **Why**  
   관계를 계산하는 표현과 전달할 정보를 분리한다.

2. **AI Usage**  
   Attention의 입력 변환.

3. **Mathematics / Mathematical Relation**  
   $$
   Q=XW_Q,\quad K=XW_K,\quad V=XW_V.
   $$
   단일 시퀀스에서 $X:(T,D)$, $Q,K:(T,d_k)$, $V:(T,d_v)$다. Q와 K의 마지막 차원이 같아야 내적할 수 있다.

4. **Implementation**  
   세 Linear projection을 구성한다. Q/K/V는 같은 입력으로부터 다른 가중치로 계산한다.

5. **Prerequisite**  
   Linear와 Embedding.

6. **Next**  
   $QK^\top$으로 토큰 사이 점수를 계산한다.

7. **Testing**  
   작은 고정 행렬의 수작업 결과, 출력 shape, 세 가중치의 gradient를 검사한다.

8. **Experiment / AI Developer Experiment Points**  
   Q/K 가중치를 공유하는 경우와 분리하는 경우를 비교한다. 쌍선형 점수 함수를 공부하고 표현 제약·검증 손실을 측정한다.

9. **What To Study Next**  
   **Must Know:** 학습 가능한 선형 변환. **Good To Know:** 쌍선형 형식. **Research Level:** Attention의 rank 제약.

10. **Research / Historical Reference**  
    Transformer의 학습 가능한 Q/K/V 변환에 대응한다. 여기서 projection은 일반 선형 변환이며 직교투영을 보장하지 않는다. [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)

---

## Phase 17 — Self Attention

1. **Why**  
   각 토큰이 문맥의 다른 토큰에서 필요한 정보를 모으게 한다.

2. **AI Usage**  
   Transformer의 토큰 간 정보 교환.

3. **Mathematics / Mathematical Relation**  
   $$
   A=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right),
   \qquad O=AV.
   $$
   독립적이고 평균 0·분산 1인 성분이라는 단순화된 가정에서 내적의 분산은 $d_k$다. $\sqrt{d_k}$로 나누면 분산 규모를 조절할 수 있다.

4. **Implementation**  
   `scaled_dot_product_attention()`, 행별 Softmax, causal mask를 구현한다.

5. **Prerequisite**  
   Q/K/V, 행렬 곱, Softmax.

6. **Next**  
   여러 Attention head를 병렬로 결합한다.

7. **Testing**  
   가중치 행의 합, masked 위치의 0 가중치, 미래 토큰 변경 시 과거 출력 불변성, gradient를 확인한다.

8. **Experiment / AI Developer Experiment Points**  
   scaling을 제거한다. 분산과 Softmax 포화를 공부하고 Attention entropy·gradient 크기·검증 손실을 비교한다. Causal mask 제거는 미래 정보 누출을 보여주는 진단 실험으로만 사용한다.

9. **What To Study Next**  
   **Must Know:** 내적과 가중합. **Good To Know:** 분산, 조건부 확률. **Research Level:** Attention 복잡도와 대안 구조.

10. **Research / Historical Reference**  
    Transformer의 scaled dot-product attention에 해당한다. 우리 구현의 점수 계산·정규화·가중합이 핵심 연결이다. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

---

## Phase 18 — Multi Head Attention

1. **Why**  
   서로 다른 학습된 표현 공간에서 관계를 동시에 계산한다.

2. **AI Usage**  
   Transformer의 Attention 모듈.

3. **Mathematics / Mathematical Relation**  
   $$
   \operatorname{MHA}(X)
   =
   \operatorname{Concat}(O_1,\ldots,O_H)W_O.
   $$
   첫 구현은 $D$가 $H$로 나누어지는 경우로 제한한다.

4. **Implementation**  
   `MultiHeadAttention`, head 분리·결합, 출력 projection을 구현한다.

5. **Prerequisite**  
   단일 head Self Attention.

6. **Next**  
   출력 크기를 유지하면서 정규화와 residual 구조에 연결한다.

7. **Testing**  
   $H=1$ 동등성, split/merge 복원, 출력 shape, head별 gradient를 검사한다.

8. **Experiment / AI Developer Experiment Points**  
   $D$를 고정하고 head 수를 바꾼다. head별 표현 차원과 rank를 공부하고 검증 손실·head 유사도·실행 시간을 비교한다.

9. **What To Study Next**  
   **Must Know:** reshape와 축 변환. **Good To Know:** 부분공간. **Research Level:** head 중복성과 pruning.

10. **Research / Historical Reference**  
    Transformer의 multi-head 결합에 대응한다. 여러 head가 반드시 사람이 해석 가능한 역할로 분화한다고 가정하지 않는다. [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)

---

## Phase 19 — Layer Normalization

1. **Why**  
   각 토큰의 특징 스케일을 정규화하여 학습을 안정화한다.

2. **AI Usage**  
   Transformer sublayer 주변의 정규화.

3. **Mathematics / Mathematical Relation**  
   $$
   \mu=\frac1D\sum_i x_i,\qquad
   \sigma^2=\frac1D\sum_i(x_i-\mu)^2,
   $$
   $$
   y_i=\gamma_i\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta_i.
   $$
   평균과 분산은 각 토큰의 특징 축에서 계산한다.

4. **Implementation**  
   `LayerNorm`, 학습 가능한 $\gamma,\beta$, 안정화 상수 $\epsilon$을 구현한다.

5. **Prerequisite**  
   Tensor reduction과 자동미분.

6. **Next**  
   FFN 및 Attention을 안정적으로 연결한다.

7. **Testing**  
   상수 입력에서 유한한지, batch 샘플이 서로 영향을 주지 않는지 검사한다. 정규화 분산은 $\epsilon$ 때문에 정확히 1이 아닐 수 있음을 반영한다.

8. **Experiment / AI Developer Experiment Points**  
   $\epsilon$과 affine 파라미터 유무를 바꾼다. 분산과 수치 안정성을 공부하고 작은 분산 입력의 gradient 및 학습 곡선을 비교한다.

9. **What To Study Next**  
   **Must Know:** 평균과 분산. **Good To Know:** normalization의 미분. **Research Level:** 정규화와 최적화 기하.

10. **Research / Historical Reference**  
    Ba, Kiros, Hinton은 샘플 내부의 특징을 정규화하는 방법을 제안했다. 우리 구현의 특징 축 평균·분산 계산이 대응한다. [Layer Normalization](https://arxiv.org/abs/1607.06450)

---

## Phase 20 — Feed Forward Network

1. **Why**  
   Attention으로 모은 정보를 토큰별 비선형 변환으로 가공한다.

2. **AI Usage**  
   Transformer block의 FFN.

3. **Mathematics / Mathematical Relation**  
   $$
   \operatorname{FFN}(x)=\phi(xW_1+b_1)W_2+b_2.
   $$
   $$
   D\rightarrow D_{\text{ff}}\rightarrow D.
   $$

4. **Implementation**  
   `FeedForward`를 두 Linear와 활성화로 구성한다. 첫 버전은 ReLU를 사용한다.

5. **Prerequisite**  
   Linear와 Activation.

6. **Next**  
   Attention·LayerNorm·residual과 함께 block을 구성한다.

7. **Testing**  
   출력 차원 보존, 토큰별 독립 계산, 모든 파라미터의 gradient를 확인한다.

8. **Experiment / AI Developer Experiment Points**  
   확장 차원 $D_{\text{ff}}$를 바꾼다. 비선형 표현력을 공부하고 검증 손실·파라미터 수·실행 시간을 비교한다.

9. **What To Study Next**  
   **Must Know:** MLP의 함수 합성. **Good To Know:** 병목과 확장 차원. **Research Level:** gated FFN.

10. **Research / Historical Reference**  
    Transformer의 position-wise feed-forward 구조에 대응한다. 토큰 위치마다 같은 변환을 적용한다. [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)

---

## Phase 21 — Transformer Block

1. **Why**  
   문맥 교환과 비선형 가공을 반복 가능한 단위로 묶는다.

2. **AI Usage**  
   언어 모델의 주요 구성 단위.

3. **Mathematics / Mathematical Relation**  
   첫 구현은 원 논문의 post-norm 형태를 따른다.
   $$
   U=\operatorname{LN}(X+\operatorname{MHA}(X)),
   $$
   $$
   Y=\operatorname{LN}(U+\operatorname{FFN}(U)).
   $$
   Residual 부분 $x+F(x)$의 미분에는 $I+J_F$가 나타난다.

4. **Implementation**  
   `TransformerBlock`, residual 덧셈, causal Attention을 통합한다. 위치 정보는 Phase 23에서 입력에 추가한다.

5. **Prerequisite**  
   Phase 18–20의 모듈.

6. **Next**  
   Tokenizer와 연결하여 실제 텍스트를 처리하는 언어 모델을 만든다.

7. **Testing**  
   shape 보존, 미래 정보 누출 방지, block 전체 gradient, 작은 합성 시퀀스 학습을 검사한다.

8. **Experiment / AI Developer Experiment Points**  
   residual 제거, norm 위치 변경, block 수 변경을 각각 실험한다. Jacobian 곱을 공부하고 깊이별 gradient와 학습 안정성을 비교한다.

9. **What To Study Next**  
   **Must Know:** residual과 합성함수. **Good To Know:** gradient 흐름. **Research Level:** 깊은 Transformer의 안정성.

10. **Research / Historical Reference**  
    Transformer의 Attention·FFN·residual·정규화 조합을 직접 구성한다. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

---

## Phase 22 — Tokenizer

1. **Why**  
   문자열을 모델이 처리할 수 있는 정수 시퀀스로 변환한다.

2. **AI Usage**  
   학습 텍스트 입력과 생성 토큰의 문자열 복원.

3. **Mathematics / Mathematical Relation**  
   $$
   \operatorname{encode}:\text{text}\rightarrow\mathbb N^T.
   $$
   문자열 분할 방식이 어휘 크기와 시퀀스 길이를 결정한다.

4. **Implementation**  
   문자 단위 `Tokenizer`, `encode()`, `decode()`, 어휘 저장, 특수 토큰을 구현한다. 이후 byte 기반 방식과 BPE를 확장한다.

5. **Prerequisite**  
   Embedding의 정수 ID 입력 규칙.

6. **Next**  
   Token ID를 언어 모델의 입력·정답 쌍으로 사용한다.

7. **Testing**  
   어휘 내 문자열 round trip, 한글·공백·줄바꿈, 미등록 문자 정책, 특수 토큰 충돌을 확인한다. 어휘는 학습 데이터로만 만든다.

8. **Experiment / AI Developer Experiment Points**  
   문자·byte·BPE를 비교한다. 빈도와 압축을 공부하고 어휘 크기·평균 길이·복원 가능성을 측정한다. 서로 다른 tokenizer의 token perplexity는 직접 비교하지 않는다.

9. **What To Study Next**  
   **Must Know:** Unicode와 인코딩. **Good To Know:** 빈도 기반 압축. **Research Level:** tokenization과 모델 효율.

10. **Research / Historical Reference**  
    Sennrich et al.은 희귀어 처리를 위해 subword 단위를 활용했다. 확장 구현의 merge 학습과 대응한다. [Neural Machine Translation of Rare Words with Subword Units](https://aclanthology.org/P16-1162/)

---

## Phase 23 — Language Model

1. **Why**  
   앞선 토큰을 조건으로 다음 토큰의 확률을 학습한다.

2. **AI Usage**  
   텍스트 생성과 이후 추천 설명 생성.

3. **Mathematics / Mathematical Relation**  
   $$
   P(x_1,\ldots,x_T)=\prod_t P(x_t\mid x_{<t}).
   $$
   모델은 token embedding과 position embedding을 더하고, causal block들을 통과시켜 $(B,T,V)$ logits를 만든다.

4. **Implementation**  
   `LanguageModel`, 학습 가능한 position embedding, block stack, 최종 normalization, vocabulary projection을 구현한다.

5. **Prerequisite**  
   TransformerBlock, Tokenizer, Cross Entropy.

6. **Next**  
   데이터 구성과 optimizer를 연결한 학습 루프를 만든다.

7. **Testing**  
   입력과 정답의 한 칸 이동, context 길이 제한, causal 동작, tiny corpus 과적합을 확인한다.

8. **Experiment / AI Developer Experiment Points**  
   위치 표현 유무와 context 길이를 바꾼다. 순서 정보와 조건부 확률을 공부하고 순서에 민감한 합성 과제와 validation NLL을 비교한다.

9. **What To Study Next**  
   **Must Know:** 확률의 연쇄법칙. **Good To Know:** perplexity. **Research Level:** 위치 표현과 장문 일반화.

10. **Research / Historical Reference**  
    단어 표현과 다음 단어 확률을 함께 학습하는 문제는 [Bengio et al., 2003](https://www.jmlr.org/papers/v3/bengio03a.html)에 연결된다. 여기서는 이를 causal Transformer 구조로 구현한다.

---

## Phase 24 — Training Loop

1. **Why**  
   데이터·손실·역전파·업데이트를 재현 가능한 학습 과정으로 연결한다.

2. **AI Usage**  
   언어 모델 pretraining, 제한된 instruction training, 추천 모델 학습.

3. **Mathematics / Mathematical Relation**  
   $$
   L=-\frac1N\sum_{\text{유효 토큰}}\log P_\theta(x_t\mid x_{<t}).
   $$
   유효 토큰 수 $N$에 맞춰 평균을 계산한다.

4. **Implementation**  
   batch 구성, `zero_grad → forward → loss → backward → step`, 검증, checkpoint, seed 기록을 구현한다. SGD 검증 후 Adam을 별도 작은 단계로 추가한다.

5. **Prerequisite**  
   완성된 언어 모델과 optimizer.

6. **Next**  
   학습된 파라미터를 고정하여 생성에 사용한다.

7. **Testing**  
   작은 batch 과적합, 데이터 누출 방지, 저장·복원 출력 일치, optimizer 상태를 포함한 재개를 확인한다. 검증이 파라미터를 바꾸지 않는지 검사한다.

8. **Experiment / AI Developer Experiment Points**  
   SGD와 Adam을 비교한다. 이동평균·모멘트·편향 보정을 공부하고 같은 데이터 순서에서 NLL·gradient norm·실행 시간을 기록한다.

9. **What To Study Next**  
   **Must Know:** train/validation/test. **Good To Know:** clipping, 학습률 스케줄. **Research Level:** optimizer와 일반화.

10. **Research / Historical Reference**  
    Adam은 gradient의 1차·2차 모멘트 추정으로 업데이트를 조절한다. 우리 구현에서는 모멘트 상태와 편향 보정을 직접 계산한다. [Kingma & Ba, 2014](https://arxiv.org/abs/1412.6980)

---

## Phase 25 — Text Generation

1. **Why**  
   다음 토큰 분포를 반복 사용해 시퀀스를 생성한다.

2. **AI Usage**  
   언어 모델의 추론.

3. **Mathematics / Mathematical Relation**  
   $$
   x_{t+1}\sim P_\theta(\cdot\mid x_{\le t}).
   $$
   Greedy는 최대확률 토큰을 선택하고 sampling은 확률에 따라 뽑는다.

4. **Implementation**  
   `generate()`, greedy, temperature, top-k, EOS 종료, 최대 길이 제한을 구현한다. 추론에서는 그래프 기록을 끄는 경로를 마련한다.

5. **Prerequisite**  
   학습된 모델과 Tokenizer.

6. **Next**  
   생성 기능을 독립적으로 검증한 뒤 추천 결과 설명에 연결한다.

7. **Testing**  
   고정 seed 재현성, EOS 종료, top-k 후보 제한, context 초과 처리, 파라미터 불변성을 확인한다.

8. **Experiment / AI Developer Experiment Points**  
   temperature와 k를 바꾼다. 엔트로피와 표본추출을 공부하고 반복률·다양성·과제 정확성을 함께 평가한다.

9. **What To Study Next**  
   **Must Know:** 범주형 분포 sampling. **Good To Know:** 탐색과 확률적 decoding. **Research Level:** 생성 품질 평가.

10. **Research / Historical Reference**  
    조건부 확률 모델의 반복 sampling에 기반한다. 모델 학습 목표와 decoding 정책은 별개의 선택임을 구분한다.

---

## Phase 26 — Recommendation Model

1. **Why**  
   사용자와 콘텐츠의 관계를 학습하여 후보의 순서를 정한다.

2. **AI Usage**  
   사용자별 Top-K 영화 추천.

3. **Mathematics / Mathematical Relation**  
   $$
   s(u,i)=p_u^\top q_i+b_u+b_i,
   $$
   $$
   \cos(p,q)=\frac{p^\top q}{\|p\|\|q\|}.
   $$
   명시적 평점은 MSE, 암묵적 선호 순위는 예를 들어
   $$
   L=-\log\sigma(s(u,i^+)-s(u,i^-))
   $$
   로 학습한다.

4. **Implementation**  
   내적·cosine baseline, `MatrixFactorization`, `RecommendationModel`, negative sampling, ranking loss, `recommend_top_k()`를 순서대로 구현한다.

5. **Prerequisite**  
   Embedding, 손실 함수, optimizer. 언어 모델과 독립된 추천 baseline부터 시작한다.

6. **Next**  
   합성 데이터에서 검증한 모델을 OTT 관련 데이터 구조에 연결한다.

7. **Testing**  
   손계산 점수, zero-vector cosine 정책, gradient, 양성 점수 상승, 이미 본 항목 제외를 확인한다. 작은 선호 데이터의 순위를 학습한다.

8. **Experiment / AI Developer Experiment Points**  
   embedding 차원·정규화 강도·negative 수를 바꾼다. 저랭크 근사와 pairwise ranking을 공부하고 RMSE 또는 Recall@K·NDCG@K로 목표에 맞게 평가한다.

9. **What To Study Next**  
   **Must Know:** 내적과 행렬 분해. **Good To Know:** ranking loss와 표본추출 편향. **Research Level:** exposure bias와 반사실 추천.

10. **Research / Historical Reference**  
    BPR은 암묵적 피드백에서 관측 항목을 미관측 항목보다 높게 두는 순위 학습을 다룬다. 미관측을 확정적인 비선호로 해석하지 않는다. [BPR](https://arxiv.org/abs/1205.2618)

---

## Phase 27 — OTT Recommendation Dataset

1. **Why**  
   추천 학습과 평가가 가능한 일관된 데이터 구조를 만든다.

2. **AI Usage**  
   사용자 이력, 영화 특징, 후보 필터링, 평가 정답 구성.

3. **Mathematics / Mathematical Relation**  
   사용자–콘텐츠 상호작용은 희소행렬로 표현할 수 있다. 관측되지 않은 값과 낮은 평점은 다르다. 시간 분할은 과거로 미래를 예측하는 조건을 모사한다.

4. **Implementation**  
   `load_interactions()`, ID mapping, 시간 기반 분할, 학습용 negative sampler, ranking evaluator를 구현한다. 콘텐츠·상호작용·OTT 제공 정보를 별도 구조로 관리한다.

5. **Prerequisite**  
   Phase 26의 합성 데이터 추천 모델.

6. **Next**  
   영화 메타데이터와 사용자 자연어 요구를 연결한다.

7. **Testing**  
   중복·누락 ID·시간 역전·분할 중첩을 검사한다. 통계와 mapping의 학습 범위, 평가 후보군, cold-start 처리 규칙을 확인한다.

8. **Experiment / AI Developer Experiment Points**  
   무작위 분할과 시간 분할을 비교한다. 분포 변화와 선택 편향을 공부하고 인기순 baseline 대비 Recall@K·NDCG@K·coverage를 측정한다.

9. **What To Study Next**  
   **Must Know:** 희소 데이터와 누출. **Good To Know:** 시간 평가, cold start. **Research Level:** missing-not-at-random 데이터.

10. **Research / Historical Reference**  
    MovieLens는 영화 평점 기반 학습 출발점이다. OTT 시청 로그나 현재 서비스 제공 목록으로 간주하지 않는다. 제공 정보에는 별도 출처·지역·확인 날짜가 필요하다. [GroupLens MovieLens](https://www.grouplens.org/datasets/movielens/)

---

## Phase 28 — Natural Language Recommendation

1. **Why**  
   사용자의 자연어 조건과 과거 선호를 함께 반영하고 추천 이유를 설명한다.

2. **AI Usage**  
   “생각할 거리가 있는 SF를 추천해줘” 같은 요청에 대한 추천 응답.

3. **Mathematics / Mathematical Relation**  
   질의 표현 $z_q$, 사용자 표현 $p_u$, 콘텐츠 표현 $q_i$를 사용한다.
   $$
   \tilde z_q=z_qW_q,\qquad
   s(u,q,i)=\alpha p_u^\top q_i+\beta\tilde z_q^\top q_i.
   $$
   차원이 맞는 것만으로 표현 의미가 정렬되지는 않으므로 projection과 추천 목적함수를 학습한다.

4. **Implementation**  
   질의 encoding, pooling, projection, 조건 필터, ranking, 검증된 메타데이터 기반 설명을 구현한다. 먼저 템플릿 설명을 검증하고 작은 언어 모델의 설명 생성으로 확장한다.

5. **Prerequisite**  
   언어 모델·추천 모델·콘텐츠 메타데이터·평가 데이터.

6. **Next**  
   전체 파이프라인 평가 후에만 성능 최적화와 라이브러리 비교로 확장한다.

7. **Testing**  
   catalog에 존재하는 항목만 추천하는지, 제외 조건을 지키는지, 설명이 메타데이터와 일치하는지 확인한다. 표현을 바꾼 동일 의도, 신규 사용자, 모호한 질의도 평가한다.

8. **Experiment / AI Developer Experiment Points**  
   사용자 이력·질의 표현·메타데이터를 하나씩 제거한다. 표현 정렬과 다목적 손실을 공부하고 NDCG@K·조건 위반률·설명 사실 일치율·사람 평가를 비교한다.

9. **What To Study Next**  
   **Must Know:** 표현 정렬과 ranking 평가. **Good To Know:** contrastive learning, grounding. **Research Level:** conversational recommendation과 공동학습.

10. **Research / Historical Reference**  
    언어 표현 학습과 개인화 순위 학습을 결합하는 프로젝트 설계다. [신경 언어 모델](https://www.jmlr.org/papers/v3/bengio03a.html)과 [BPR](https://arxiv.org/abs/1205.2618)을 구성 요소의 배경으로 삼으며, 특정 논문 전체를 재현한다고 보지는 않는다.

---

## 주요 구간의 가정과 실패 분석

| 구간 | 초기 가정 | 병목 | 대표 실패 | 진단 실험 |
|---|---|---|---|---|
| Scalar Autograd | 미분 가능한 스칼라 연산 | Python 객체와 그래프 | 공유 경로 기울기 누락 | 공유 노드 gradient check |
| MLP | 작은 정형 데이터 | neuron별 반복 계산 | 포화·초기화 실패 | 활성화·초기화 변경 |
| Tensor | Value 기반 원소 저장 | 객체 수와 메모리 | shape·broadcast 오류 | 스칼라 구현과 동등성 검사 |
| Transformer | 짧은 문맥과 작은 모델 | $T^2$ Attention 점수 | 미래 누출·gradient 불안정 | causal 검사·residual ablation |
| Language Model | 작은 제한된 corpus | 어휘 출력과 그래프 크기 | 암기·반복 생성 | validation NLL·중복 점검 |
| Recommendation | 관측 이력이 선호의 단서 | 전체 후보 점수 계산 | 인기 편향·cold start | 인기 baseline·사용자군별 평가 |
| 자연어 추천 | 텍스트와 콘텐츠 표현을 정렬 가능 | 학습 데이터와 평가 | 부정 조건 누락·설명 오류 | 조건별 테스트·메타데이터 대조 |

작은 표준 라이브러리 모델의 성공 기준은 범용 대화 능력이 아니다. 제한된 데이터에서 학습과 추론의 원리를 검증하고, 실패 원인을 설명할 수 있는 것을 목표로 한다.

## 프로젝트 구조 설계안

아래는 향후 파일 배치 계획이며, 아직 생성하지 않는다.

```text
my-brain/
├─ docs/
│  └─ IMPLEMENTATION_ROADMAP.md
├─ brain/
│  ├─ autograd/
│  │  ├─ value.py
│  │  └─ graph.py
│  ├─ tensor/
│  │  ├─ tensor.py
│  │  └─ operations.py
│  ├─ nn/
│  │  ├─ neuron.py
│  │  ├─ layer.py
│  │  ├─ mlp.py
│  │  ├─ linear.py
│  │  ├─ activation.py
│  │  └─ loss.py
│  ├─ optim/
│  │  ├─ sgd.py
│  │  └─ adam.py
│  ├─ transformer/
│  │  ├─ embedding.py
│  │  ├─ attention.py
│  │  ├─ multi_head_attention.py
│  │  ├─ layer_norm.py
│  │  ├─ feed_forward.py
│  │  └─ transformer_block.py
│  └─ tokenizer/
│     └─ tokenizer.py
├─ model/
│  ├─ language_model.py
│  └─ recommendation_model.py
├─ training/
│  ├─ pretrain.py
│  ├─ instruction_train.py
│  └─ recommendation_train.py
├─ data/
├─ tests/
└─ examples/
```

## 첫 구현의 진행 순서

로드맵 이후 사용자가 다음 단계를 요청하면 첫 핵심 파일은 `brain/autograd/value.py`다.

| Step | 범위 | 이해해야 할 내용 |
|---|---|---|
| 1 | `Value(data)` | 값과 기울기 저장 공간의 차이 |
| 2 | Addition | 두 값을 하나의 결과로 만드는 연산 |
| 3 | Multiplication | 입력에 따라 달라지는 변화율 |
| 4 | Computational Graph | 계산 의존관계 |
| 5 | Local Derivative | 연산별 미분 |
| 6 | Chain Rule | 경로별 전달과 누적 |
| 7 | `backward()` | 역위상순서 자동미분 |
| 8 | Gradient Check | 수치 미분과의 비교 |

이후 부호 반전·뺄셈·상수 지수 거듭제곱·나눗셈을 차례로 추가한다. 각 연산마다 정의역과 예외를 확인한다.

## 후속 성능 최적화 단계

Phase 0–28의 기준 구현과 검증을 마친 뒤 별도 단계로 진행한다.

```text
Python Standard Library 기준 구현
→ 프로파일링
→ Tensor 단위 연산과 backward
→ 벡터화
→ NumPy 결과 비교
→ PyTorch 결과·gradient 비교
→ CPU/GPU 실행 비교
→ CUDA Kernel 학습
```

비교에서는 다음 질문에 답한다.

- 라이브러리가 우리 코드의 어느 부분을 대신하는가?
- 동일한 입력과 가중치에서 출력과 gradient가 일치하는가?
- 연산 순서와 정밀도 차이가 오차에 어떤 영향을 주는가?
- 그래프 생성·메모리·행렬 연산 중 실제 병목은 어디인가?
- 속도를 얻기 위해 어떤 구현 복잡성이 추가되는가?

## 프로젝트 완료 기준

다음 내용을 수식, 작은 예제, 직접 작성한 코드로 설명할 수 있어야 한다.

- `backward()`가 연쇄법칙을 어떻게 실행하는가.
- gradient가 여러 경로에서 왜 누적되는가.
- 행렬 곱이 여러 가중합을 어떻게 표현하는가.
- embedding이 학습되는 과정은 무엇인가.
- Softmax와 Cross Entropy를 왜 함께 사용하는가.
- Q/K/V와 $\sqrt{d_k}$ scaling이 어떤 역할을 하는가.
- residual과 normalization을 제거하면 무엇이 달라지는가.
- optimizer 변경이 학습에 어떤 영향을 주는가.
- 언어 생성 손실과 추천 ranking 손실은 무엇이 다른가.
- 데이터 누출 없이 변경 효과를 어떻게 평가하는가.
