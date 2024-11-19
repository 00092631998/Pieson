# 학습 데이터 : XOR 게이트
X_xor = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y_xor = np.array([0, 1, 1, 0])

# 퍼셉트론 생성
perceptron_xor = Perceptron(input_size=2, learning_rate=0.1, epochs=10)

# 학습
perceptron_xor.fit(X_xor, y_xor)

# 학습된 가중치와 바이어스 출력
print("가중치 (XOR):", perceptron_xor.weights)
print("바이어스 (XOR):", perceptron_xor.bias)

# 결정 경계 시각화
perceptron_xor.plot_decision_boundary(X_xor, y_xor)

# 예측 결과 확인
print("\nXOR 게이트 예측 결과:")
for x in X_xor:
    print(f"입력: {x}, 예측 출력: {perceptron_xor.predict(x)}")
