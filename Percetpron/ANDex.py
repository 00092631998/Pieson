# 학습 데이터: AND 게이트
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 0, 0, 1])

# 퍼셉트론 인스턴스 생성
perceptron = Perceptron(input_size=2, learning_rate=0.1, epochs=10)

# 학습
perceptron.fit(X, y)

# 학습된 가중치와 바이어스 출력
print("가중치:", perceptron.weights)
print("바이어스:", perceptron.bias)

# 결정 경계 시각화
perceptron.plot_decision_boundary(X, y)

# 학습 과정 시각화
perceptron.plot_learning_process(X, y)

# 예측 결과 확인
print("\n예측 결과:")
for x in X:
    print(f"입력: {x}, 예측 출력: {perceptron.predict(x)}")
