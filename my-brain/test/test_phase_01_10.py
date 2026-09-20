"""Phase 1-10 회귀 검사: 손계산, 수치 미분, 경계 조건, 학습 동작.

수학적 이론에 기반한 기대값을 검사하지만, 테스트 입력/허용 오차는 검증 설계다.
교재: Mathematics for Machine Learning 2/5/7장, https://mml-book.github.io/
실행: python -B -m unittest discover -s test -v
"""
import contextlib
import io
import math
from pathlib import Path
import random
import runpy
import unittest

from brain.autograd.value import Value
from brain.optim.sgd import SGD
from brain.nn.neuron import Neuron, tanh
from brain.nn.layer import Layer
from brain.nn.mlp import MLP
from brain.tensor.tensor import Tensor
from brain.tensor.operations import dot, transpose, matmul


class TestPhase01To10(unittest.TestCase):
    def check_gradient(self, objective, variables):
        objective().backward()
        gradients = [v.grad for v in variables]
        for v, expected in zip(variables, gradients):
            original = v.data
            h = 1e-5
            try:
                v.data = original + h
                plus = objective().data
                v.data = original - h
                minus = objective().data
            finally:
                v.data = original
            self.assertTrue(math.isclose(
                expected, (plus - minus) / (2 * h),
                rel_tol=1e-4, abs_tol=1e-6,
            ), (expected, (plus - minus) / (2 * h)))

    def test_scalar_independence(self):
        a, b = Value(2), Value(2)
        a.grad = 9
        self.assertEqual(b.grad, 0)
        self.assertEqual(a.data, 2)

    def test_shared_graph_and_repeated_backward(self):
        x = Value(3)
        y = x * x
        loss = y + y + x
        for _ in range(2):
            loss.backward()
            self.assertEqual(loss.data, 21)
            self.assertEqual(x.grad, 13)

    def test_arithmetic_gradient(self):
        a, b = Value(1.7), Value(2.3)
        def objective():
            return (3 + a) * (b - 2) + 4 / a - a / b + a ** 3 - (2 - b)
        self.check_gradient(objective, [a, b])

    def test_power_domains(self):
        x = Value(0)
        (x ** 0).backward()
        self.assertEqual(x.grad, 0)
        (x ** 1).backward()
        self.assertEqual(x.grad, 1)
        self.assertEqual((Value(-2) ** 3).data, -8)
        with self.assertRaises(ZeroDivisionError):
            Value(1) / 0
        with self.assertRaises(ValueError):
            Value(-2) ** 0.5
        with self.assertRaises(ValueError):
            Value(0) ** 0.5

    def test_sgd_update_dedup_and_reset(self):
        w = Value(0)
        optimizer = SGD([w, w], lr=0.1)
        ((w - 3) ** 2).backward()
        optimizer.step()
        self.assertAlmostEqual(w.data, 0.6)
        optimizer.zero_grad()
        self.assertEqual(w.grad, 0)
        for lr in (0, -1, math.inf, math.nan):
            with self.assertRaises(ValueError):
                SGD([w], lr=lr)

    def test_sgd_convergence_and_divergence(self):
        results = []
        for lr in (0.1, 1.1):
            w = Value(0)
            optimizer = SGD([w], lr=lr)
            for _ in range(20):
                ((w - 3) ** 2).backward()
                optimizer.step()
            results.append(abs(w.data - 3))
        self.assertLess(results[0], 0.04)
        self.assertGreater(results[1], 3)

    def test_neuron_hand_calculation(self):
        neuron = Neuron(2, nonlinear=False, rng=random.Random(1))
        self.assertEqual(len(neuron.weights), 2)
        neuron.weights[0].data = 3
        neuron.weights[1].data = 4
        neuron.bias.data = -1
        output = neuron([1, 2])
        self.assertEqual(output.data, 10)
        output.backward()
        self.assertEqual([p.grad for p in neuron.parameters()], [1, 2, 1])
        with self.assertRaises(ValueError):
            neuron([1])

    def test_neuron_nonlinear_gradient(self):
        neuron = Neuron(2, rng=random.Random(2))
        x = [Value(0.3), Value(-0.7)]
        self.check_gradient(lambda: neuron(x), x + neuron.parameters())
        self.assertAlmostEqual(tanh(Value(0)).data, 0)

    def test_layer_and_mlp_shapes_seed(self):
        layer = Layer(3, 4, rng=random.Random(1))
        self.assertEqual(len(layer([1, 2, 3])), 4)
        self.assertEqual(len(layer.parameters()), 16)
        self.assertEqual(len(set(layer.parameters())), 16)
        first, second = MLP([2, 4, 1], seed=42), MLP([2, 4, 1], seed=42)
        self.assertEqual(len(first([0, 1])), 1)
        self.assertEqual([p.data for p in first.parameters()],
                         [p.data for p in second.parameters()])
        self.assertEqual(len(first.parameters()), 17)

    def test_mlp_gradient(self):
        model = MLP([2, 3, 1], seed=1)
        self.check_gradient(lambda: model([0.2, -0.1])[0] ** 2,
                            model.parameters())

    def test_mlp_xor_learning(self):
        model = MLP([2, 4, 1], seed=42)
        optimizer = SGD(model.parameters(), lr=0.2)
        samples = [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]
        def objective():
            return sum((model(x)[0] - y) ** 2 for x, y in samples) / 4
        before = objective().data
        for _ in range(1000):
            optimizer.zero_grad()
            objective().backward()
            optimizer.step()
        self.assertLess(objective().data, before)
        self.assertLess(objective().data, 0.01)
        self.assertEqual([int(model(x)[0].data >= 0.5) for x, _ in samples],
                         [0, 1, 1, 0])

    def test_tensor_shape_index_and_aliasing(self):
        tensor = Tensor([1, 2, 3, 4, 5, 6], (2, 3))
        self.assertEqual(tensor[1, 2].data, 6)
        self.assertEqual(tensor[-1, -1].data, 6)
        reshaped = tensor.reshape((3, 2))
        self.assertIs(reshaped[1, 0], tensor[0, 2])
        reshaped.sum().backward()
        self.assertEqual([v.grad for v in tensor.data], [1] * 6)
        self.assertEqual(Tensor([7], ())[()].data, 7)

    def test_tensor_elementwise_and_shared_gradient(self):
        a = Tensor([2, 3], (2,))
        b = Tensor([4, 5], (2,))
        result = (a * b + a).sum()
        self.assertEqual(result.data, 28)
        result.backward()
        self.assertEqual([v.grad for v in a.data], [5, 6])
        self.assertEqual([v.grad for v in b.data], [2, 3])

    def test_tensor_invalid_shapes(self):
        with self.assertRaises(ValueError):
            Tensor([1], (2,))
        with self.assertRaises(ValueError):
            Tensor([], (0,))
        tensor = Tensor([1, 2], (2,))
        with self.assertRaises(IndexError):
            tensor[2]
        with self.assertRaises(IndexError):
            tensor[0, 0]
        with self.assertRaises(ValueError):
            tensor + Tensor([1], (1,))
        with self.assertRaises(TypeError):
            tensor * 2

    def test_dot_and_transpose(self):
        a, b = Tensor([1, 2], (2,)), Tensor([3, 4], (2,))
        result = dot(a, b)
        self.assertEqual(result.data, 11)
        result.backward()
        self.assertEqual([v.grad for v in a.data], [3, 4])
        matrix = Tensor([1, 2, 3, 4, 5, 6], (2, 3))
        transposed = transpose(matrix)
        self.assertEqual(transposed.shape, (3, 2))
        self.assertEqual([v.data for v in transposed.data], [1, 4, 2, 5, 3, 6])
        self.assertIs(transposed[2, 1], matrix[1, 2])

    def test_matmul_hand_calculation_and_gradient(self):
        a = Tensor([1, 2, 3, 4], (2, 2))
        b = Tensor([5, 6, 7, 8], (2, 2))
        self.assertEqual([v.data for v in matmul(a, b).data], [19, 22, 43, 50])
        self.check_gradient(lambda: matmul(a, b).sum(), a.data + b.data)
        c = Tensor([1, 2, 3, 4, 5, 6], (2, 3))
        d = Tensor([1, 2, 3], (3, 1))
        self.assertEqual([v.data for v in matmul(c, d).data], [14, 32])
        with self.assertRaises(ValueError):
            matmul(c, a)

    def test_manual_training_example_converges(self):
        path = Path(__file__).resolve().parents[1] / 'train' / 'train_basic.py'
        with contextlib.redirect_stdout(io.StringIO()):
            namespace = runpy.run_path(str(path))
        self.assertAlmostEqual(namespace['w'], 2.0, places=5)


if __name__ == '__main__':
    unittest.main()
