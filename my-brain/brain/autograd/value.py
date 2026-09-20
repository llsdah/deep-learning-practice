# 덧셈 곱셈 역전파 
class Value:
  def __init__(self, data, children=()):
    # 수학적 변수 x : 현재 수치의 값을 저장 
    self.data = data
    # dL/ds 누적 공간
    self.grad = 0.0 # 최종 손실이 이 가중치에 얼마나 민감한지 
    self.children = set(children)
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
    result = Value(self.data * other.data, (self, other))

    def backward():
      # c = a * b
      # dc / da = b
      # dc / db = a
      self.grad += other.data * result.grad
      other.grad += self.data * result.grad

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
      self.grad += 1 * result.grad
      other.grad += 1 * result.grad
    result._backward = backward
    return result
 
  def backward(self):
    ordered = []
    visited = set()

    def build(node):
      if node in visited:
        return

      visited.add(node)
      for child in node.children:
        build(child)

      ordered.append(node)

    build(self)

    self.grad = 1.0

    for node in reversed(ordered):
      node._backward()