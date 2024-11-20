import numpy as np
import matplotlib.pyplot as plt

# 활성화 함수 : 시그모이드
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 시그모이드 함수의 도함수
def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

# 손실 함수: 평균 제곱 오차 (MSE)
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2) / 2.0

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        
        # 초기 가중치 및 바이어스 설정 (고정된 값 사용)
        
        self.W1 = np.array([
            [0.2, -0.5, 0.1],
            [0.7, -1.2, 0.3],
            [-0.3, 0.8, -0.4]
        ])

        self.b1 = np.array([
            [0.1],
            [-0.2],
            [0.05]
        ])

        self.W2 = np.array([
            [1.5, -2.0, 0.5],
            [-1.0, 2.5, -0.3]
        ])

        self.b2 = np.array([
            [0.3],
            [-0.1]
        ])

        self.learning_rate = learning_rate

    # forward propagation
    def forward(self, X):
        self.Z1 = np.dot(self.W1, X) + self.b1  # (hidden_size x m)
        self.A1 = sigmoid(self.Z1)              # (hidden_size x m)
        self.Z2 = np.dot(self.W2, self.A1) + self.b2  # (output_size x m)
        self.A2 = sigmoid(self.Z2)              # (output_size x m)
        return self.A2

    # backward propagation
    def backward(self, X, Y, output):
        m = X.shape[1]

        # 출력층의 오차
        dA2 = output - Y  # (output_size x m)
        dZ2 = dA2 * sigmoid_derivative(self.Z2)  # (output_size x m)
        dW2 = (1/m) * np.dot(dZ2, self.A1.T)      # (output_size x hidden_size)
        db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)  # (output_size x 1)

        # 은닉층의 오차
        dA1 = np.dot(self.W2.T, dZ2)             # (hidden_size x m)
        dZ1 = dA1 * sigmoid_derivative(self.Z1)  # (hidden_size x m)
        dW1 = (1/m) * np.dot(dZ1, X.T)           # (hidden_size x input_size)
        db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)  # (hidden_size x 1)

        gradients = {"dW1": dW1, "db1": db1,
                     "dW2": dW2, "db2": db2}
        return gradients

    # 가중치, 바이어스 업데이트
    def update_parameters(self, gradients):
        self.W1 -= self.learning_rate * gradients['dW1']
        self.b1 -= self.learning_rate * gradients['db1']
        self.W2 -= self.learning_rate * gradients['dW2']
        self.b2 -= self.learning_rate * gradients['db2']

    def train(self, X, Y, epochs=2, print_details=True):

        grad_W1_norm = []
        grad_b1_norm = []
        grad_W2_norm = []
        grad_b2_norm = []
        loss_history = []

        for epoch in range(1, epochs + 1):

            # 순전파
            output = self.forward(X)

            # 손실 계산
            loss = mse_loss(Y, output)
            loss_history.append(loss)

            # 역전파
            gradients = self.backward(X, Y, output)

            # 기울기 저장
            grad_W1_norm.append(np.linalg.norm(gradients['dW1']))
            grad_b1_norm.append(np.linalg.norm(gradients['db1']))
            grad_W2_norm.append(np.linalg.norm(gradients['dW2']))
            grad_b2_norm.append(np.linalg.norm(gradients['db2']))

            # 가중치와 바이어스 업데이트
            self.update_parameters(gradients)

            if print_details:
                # 순전파 세부 정보 출력
                print("순전파 결과:")
                print(f"Z1:\n{self.Z1}")
                print(f"A1:\n{self.A1}")
                print(f"Z2:\n{self.Z2}")
                print(f"A2 (출력):\n{self.A2}\n")

                # 손실 출력
                print(f"손실 (Loss): {loss:.4f}\n")

                # 역전파 세부 정보 출력
                print("역전파 결과:")
                print(f"dW2 (출력층 가중치 기울기):\n{gradients['dW2']}")
                print(f"db2 (출력층 바이어스 기울기):\n{gradients['db2']}\n")

                print(f"dW1 (은닉층 가중치 기울기):\n{gradients['dW1']}")
                print(f"db1 (은닉층 바이어스 기울기):\n{gradients['db1']}\n")

                # 업데이트된 가중치와 바이어스 출력
                print("업데이트된 가중치와 바이어스:")
                print(f"W1:\n{self.W1}")
                print(f"b1:\n{self.b1}")
                print(f"W2:\n{self.W2}")
                print(f"b2:\n{self.b2}\n")
                print("------------------------------\n")

        epochs_range = range(1, epochs + 1)
        plt.figure(figsize=(12, 6))
        plt.plot(epochs_range, grad_W1_norm, label='||dW1||')
        plt.plot(epochs_range, grad_b1_norm, label='||db1||')
        plt.plot(epochs_range, grad_W2_norm, label='||dW2||')
        plt.plot(epochs_range, grad_b2_norm, label='||db2||')
        plt.xlabel('Epoch')
        plt.ylabel('Gradient')
        plt.title('Gradient')
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(12, 6))
        plt.plot(epochs_range, loss_history, label='MSE Loss', color='purple')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

    def predict(self, X):
        output = self.forward(X)
        predictions = (output > 0.5).astype(int)
        return predictions

# 학습 샘플 설정
X = np.array([[1],
              [0],
              [-1]])  # (3 x 1)

Y = np.array([[1],
              [0]])   # (2 x 1)

# 신경망 인스턴스 생성
nn = NeuralNetwork(input_size=3, hidden_size=3, output_size=2, learning_rate=0.1)

# 초기 가중치와 바이어스 출력
print("초기 가중치 W1:")
print(nn.W1)
print("\n초기 바이어스 b1:")
print(nn.b1)
print("\n초기 가중치 W2:")
print(nn.W2)
print("\n초기 바이어스 b2:")
print(nn.b2)
print("\n--- 학습 시작 ---\n")

# 신경망 학습 (2 에포크)
nn.train(X, Y, epochs=1000, print_details=False)

# 학습 후 가중치와 바이어스 출력
print("\n--- 학습 완료 ---\n")
print("최종 가중치 W1:")
print(nn.W1)
print("\n최종 바이어스 b1:")
print(nn.b1)
print("\n최종 가중치 W2:")
print(nn.W2)
print("\n최종 바이어스 b2:")
print(nn.b2)

# 예측 수행
predictions = nn.predict(X)

# 예측 결과 출력
print("\n예측 결과:")
print(predictions)
