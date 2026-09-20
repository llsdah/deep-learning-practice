# 검증 코드: 손계산, 중심차분, 경계값, 분류 학습. 수학 정의가 아니라 회귀 검사 사례이다.
# 테스트 설명과 전체 실행 명령: PHASE_01_14_REVIEW.md
import math
import unittest

from brain.tensor.tensor import Tensor
from brain.nn.linear import Linear
from brain.nn.activation import relu, sigmoid, tanh, softmax
from brain.nn.loss import cross_entropy_from_logits
from brain.optim.sgd import SGD


class TestPhase11To14(unittest.TestCase):
    def assert_gradient_matches(self, objective, variables):
        # 해석적 gradient는 한 번 계산해 보관한다.
        loss = objective()
        loss.backward()
        analytical = [value.grad for value in variables]

        h = 1e-5

        for value, expected in zip(variables, analytical):
            original = value.data

            try:
                # 각 perturbation마다 새 순전파가 필요하다.
                value.data = original + h
                plus = objective().data

                value.data = original - h
                minus = objective().data
            finally:
                value.data = original

            numerical = (plus - minus) / (2.0 * h)

            self.assertTrue(
                math.isclose(
                    expected,
                    numerical,
                    rel_tol=1e-4,
                    abs_tol=1e-6,
                ),
                f"autograd={expected}, numerical={numerical}",
            )

    def test_linear_hand_calculation_and_bias_gradient(self):
        layer = Linear(2, 2, seed=1)

        for parameter, value in zip(layer.weight.data, [1, 2, 3, 4]):
            parameter.data = float(value)

        layer.bias[0].data = 0.5
        layer.bias[1].data = -0.5

        inputs = Tensor([1, 2, 3, 4], (2, 2))
        output = layer(inputs)

        self.assertEqual(
            [value.data for value in output.data],
            [7.5, 9.5, 15.5, 21.5],
        )

        output.sum().backward()

        # 각 bias가 두 샘플의 출력에 한 번씩 더해진다.
        self.assertEqual([v.grad for v in layer.bias.data], [2.0, 2.0])

    def test_activation_values_and_relu_boundary(self):
        values = Tensor([-1.0, 0.0, 1.0], (3,))

        self.assertEqual(
            [v.data for v in relu(values).data],
            [0.0, 0.0, 1.0],
        )
        self.assertAlmostEqual(sigmoid(values)[1].data, 0.5)
        self.assertAlmostEqual(tanh(values)[1].data, 0.0)

        relu(values).sum().backward()

        # x=0では実装規則としてgradient=0を選択する。
        self.assertEqual([v.grad for v in values.data], [0.0, 0.0, 1.0])

        extreme = sigmoid(Tensor([-1000.0, 1000.0], (2,)))
        self.assertEqual([v.data for v in extreme.data], [0.0, 1.0])

    def test_activation_gradients(self):
        for function in (relu, sigmoid, tanh):
            with self.subTest(function=function.__name__):
                # ReLU의 미분 불가능한 점 0은 중심차분에서 제외한다.
                values = Tensor([-0.7, 0.4], (2,))
                self.assert_gradient_matches(
                    lambda: function(values).sum(),
                    values.data,
                )

    def test_softmax_rows_and_shift_invariance(self):
        inputs = Tensor([1, 2, 3, -1, 0, 1], (2, 3))
        probabilities = softmax(inputs)

        for row in range(2):
            self.assertAlmostEqual(
                sum(probabilities[row, col].data for col in range(3)),
                1.0,
            )

        shifted = Tensor([v.data + 1000 for v in inputs.data], inputs.shape)
        shifted_probabilities = softmax(shifted)

        for first, second in zip(
            probabilities.data, shifted_probabilities.data
        ):
            self.assertAlmostEqual(first.data, second.data)

        # Softmax 전체 합은 상수이므로 서로 다른 가중치의 합으로 미분을 검사한다.
        def objective():
            p = softmax(inputs)
            return p[0, 0] + 2 * p[0, 1] + 4 * p[0, 2]

        self.assert_gradient_matches(objective, inputs.data)

    def test_cross_entropy_hand_calculation(self):
        logits = Tensor([0.0, 0.0], (1, 2))
        loss = cross_entropy_from_logits(logits, [0])

        self.assertAlmostEqual(loss.data, math.log(2.0))

        loss.backward()
        self.assertAlmostEqual(logits[0, 0].grad, -0.5)
        self.assertAlmostEqual(logits[0, 1].grad, 0.5)

    def test_cross_entropy_extreme_logits(self):
        logits = Tensor([1000.0, -1000.0], (1, 2))
        loss = cross_entropy_from_logits(logits, [1])

        # 정답 확률이 반올림되어 0이 될 수 있어도 이 예제의 로그 손실은 유한하다.
        self.assertTrue(math.isfinite(loss.data))
        self.assertAlmostEqual(loss.data, 2000.0)

        loss.backward()
        self.assertAlmostEqual(logits[0, 0].grad, 1.0)
        self.assertAlmostEqual(logits[0, 1].grad, -1.0)

    def test_ignored_targets_and_errors(self):
        logits = Tensor([0.0, 0.0, 2.0, -1.0], (2, 2))
        loss = cross_entropy_from_logits(
            logits, [0, -1], ignore_index=-1
        )

        self.assertAlmostEqual(loss.data, math.log(2.0))
        loss.backward()

        self.assertEqual(logits[1, 0].grad, 0.0)
        self.assertEqual(logits[1, 1].grad, 0.0)

        with self.assertRaises(ValueError):
            cross_entropy_from_logits(logits, [-1, -1], ignore_index=-1)

        with self.assertRaises(ValueError):
            cross_entropy_from_logits(logits, [0, 2])

        with self.assertRaises(ValueError):
            Linear(3, 2)(Tensor([1, 2], (1, 2)))

    def test_combined_model_gradient(self):
        inputs = Tensor([0.2, -0.3, 0.4, 0.1], (2, 2))
        hidden = Linear(2, 3, seed=1)
        output = Linear(3, 2, seed=2)

        def objective():
            logits = output(tanh(hidden(inputs)))
            return cross_entropy_from_logits(logits, [0, 1])

        variables = (
            inputs.data
            + hidden.parameters()
            + output.parameters()
        )
        self.assert_gradient_matches(objective, variables)

    def test_learning_behavior(self):
        # 선형 분리가 가능한 두 샘플.
        inputs = Tensor([-1.0, 1.0], (2, 1))
        model = Linear(1, 2, seed=3)
        optimizer = SGD(model.parameters(), lr=0.2)

        def objective():
            return cross_entropy_from_logits(model(inputs), [0, 1])

        before = objective().data

        for _ in range(100):
            optimizer.zero_grad()
            loss = objective()
            loss.backward()
            optimizer.step()

        after = objective().data
        self.assertLess(after, before)
        self.assertLess(after, 0.1)

        logits = model(inputs)
        predictions = [
            max(range(2), key=lambda col: logits[row, col].data)
            for row in range(2)
        ]
        self.assertEqual(predictions, [0, 1])


if __name__ == "__main__":
    unittest.main()