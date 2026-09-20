# Phase 1-4: 스칼라와 역방향 자동미분
# 수학적 이론에 기반: 연쇄법칙, 합/곱/상수 거듭제곱의 미분. dL/dx = sum_u (dL/du)(du/dx).
# 유지할 수학: 같은 함수와 정의역에서는 이 미분 관계를 임의로 변경할 수 없다.
# 변경 가능한 구현/설계: children의 set, 재귀 위상정렬, 매 backward의 초기화는 구현 정책이다. 0**0=1도 여기서의 규약이다.
# AI Developer Experiment Points / 연구·실험: 화두 코드: backward()의 메모리/재계산, 반복형 순회, gradient 누적 정책. 변경 시 공유 노드·중심차분 검사.
# 추천 교재/논문: Deisenroth, Faisal, Ong, Mathematics for Machine Learning (2020), https://mml-book.github.io/ — 5장 Vector Calculus.
# 논문: Baydin et al. (2018), https://jmlr.org/papers/v18/17-468.html
# 문제/아이디어/연결: 프로그램 미분을 기본 연산의 미분 조합으로 계산 → _backward와 backward().
# 주의: 수학적 정의가 고정되어도 소스코드가 영구 불변이라는 뜻은 아니다.
# 상세 질문 답변·검증: PHASE_01_14_REVIEW.md

# 덧셈 곱셈 역전파 
class Value:
  def __init__(self, data, children=()):
    # 수학적 변수 x : 현재 수치의 값을 저장 
    self.data = data
    # dL/ds 누적 공간
    self.grad = 0.0 # 최종 손실이 이 가중치에 얼마나 민감한지 

    # 이 값을 만든 입력 노드들 같은 입력을 여러번 사용대호 그래프 방문은 한번 
    self.children = set(children)

    # 입력 노드는 더 전달한 연산이 없으므로, 하지 않음 
    self._backward = lambda: None

  def __repr__(self):
    return f"Value(data={self.data}, grad={self.grad})"

  def __mul__(self, other):
    other = (
        other
        if isinstance(other, Value)
        else Value(other)
    )
    # multiple
    # 미분시 순전파 시점의 입력값이 필요 
    a, b = self.data, other.data
    result = Value(a * b, (self, other))

    def backward():
      # c = a * b
      # dc / da = b
      # dc / db = a
      # self other 가 같아도 두 기여를 모두 더함 
      self.grad += b * result.grad
      other.grad += a * result.grad

    result._backward = backward
    return result

  def __rmul__(self, other):
    return self * other

  def __pow__(self, power):
    if not isinstance(power, (int,float)):
      raise TypeError("지수는 상수 int 또는 float ")

    if self.data < 0 and power % 1 != 0 :
      raise ValueError("음수 밑의 비정수 거듭제곱은 안됩니다.")

    if self.data == 0 and power < 0:
      raise ZeroDivisionError("0의 음수 거듭제곱은 저의되지 안흡니다. ")

    if self.data == 0 and 0 < power < 1:
        raise ValueError("이 지수에서는 0에서 유한한 미분이 없습니다.")

    result = Value(self.data ** power, (self,))

    # c = a^p, p상수 dc/da = p * a^(p-1), p == 0 ? 1 미분 0 

    slope = (
      0.0 if power == 0
      else power * self.data ** (power - 1)
    )

    def backward():
      self.grad += slope * result.grad

    result._backward = backward
    return result

  def __add__(self, other):

    # self.data : 왼쪽 입력 a 현재값
    # self.orther : 오른쪽 입력 b의 현재값 
    # 합 c 의 담은 새로운 Value 반환 
    # 국소 미군 덧셈의 변화율은 1 

    other = (
        other
        if isinstance(other, Value)
        else Value(other)
    )
    result = Value( self.data + other.data, (self, other))

    def backward():
      # c = a + b 
      # d/a = dl/dc * 1
      # 다른 경로의 기역도가 있으므로 대입하지 않고, 누적 
      self.grad += 1 * result.grad
      other.grad += 1 * result.grad
    result._backward = backward
    return result

  # 숫자 + Value 도 덧셈 처리 
  def __radd__(self, other):
    return self + other

  def __neg__(self):
    return self * -1.0

  def __sub__(self, other):
    return self + (-other)

  def __rsub__(self, other):
    # 숫자 - Value
    return (-self) + other

  def __truediv__(self, other):
    other = other if isinstance(other, Value) else Value(other)

    # 곱셈과 거듭제곱 연쇄법칙 미분
    return self * (other ** -1)

  def __rtruediv__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    return other / self
 
  def backward(self):
    ordered = []
    visited = set()

    def build(node):
      if node in visited:
        return

      visited.add(node)
      for child in node.children:
        build(child)

      # 입력들을 먼저 넣고 그 결과 노드를 나중에 넣음
      ordered.append(node)

    build(self)

    # 이번 호출은 이전 호출의 gradient를 이어받지 않음
    for node in ordered:
      node.grad = 0.0

    # 출력 L을 자신으로 미분하면 dL/dL = 1
    self.grad = 1.0

    # 결과 -> 입력 순서로 국소 미분과 상위 기울기를 곱함
    for node in reversed(ordered):
      node._backward()