# 우리가 학습 시킬 데이터
x_data = [1.0, 2.0, 3.0, 4.0]
y_data = [2.0, 4.0, 6.0, 8.0]

# weight  임의 값
w = 0.5

# moving rate
learning_rate = 0.01

for epoch in range(100):
  total_loss = 0.0
  total_gradient = 0.0

  for x, target in zip(x_data, y_data):
    # Forward
    prediction = x * w

    # Loss
    # (예측 - 정답)^ 2
    error = prediction - target
    loss = error * error

    # Gradient
    # loss = (x*w - target)^2
    # d(loss) / dw
    # = 2(x*w - target) * x

    gradient = 2.0 * error * x
    total_loss += loss
    total_gradient += gradient


  # 평균 gradient: 매 epoch마다 계산하고 가중치를 갱신해야 한다.
  total_gradient /= len(x_data)

  # weight update
  w = w - learning_rate * total_gradient

  if epoch % 20 == 0:
    print (
      "epoch = ",epoch,
      "loss = ",total_loss,
      "gradient = ",total_gradient,
      "w = ",w
  )

print("학습 완료")
print("w = ", w)
print( "x=10 예측 =", 10 * w)
