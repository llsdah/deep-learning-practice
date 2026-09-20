from brain.autograd import Value


class Neuron:
  def __init__(self, input_size):
    self.weights = []
    for _ in range(input_size):
      self.weights.append( Value(0.1) )

    self.bias = Value(0.0)

  def __call__(self, inputs):
    output = self.bias

    for weight, value in zip (
        self.weights,
          inputs
    ):
      output = (
          output + weight * value
      )

    return output
