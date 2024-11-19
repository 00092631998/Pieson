import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(25)
class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, epochs=1000):
        # 가중치 초기화 He 초기화
        self.W1 = np.random.randn(hidden_size, input_size) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = np.random.randn(output_size, hidden_size) * np.sqrt(2. / hidden_size)
        self.b2 = np.zeros((output_size, 1))
        self.learning_rate = learning_rate
        self.epochs = epochs

    # relu 함수
    def relu(self, Z):
        return np.maximum(0, Z)

    # relu 함수 미분
    def relu_derivative(self, Z):
        return Z > 0

    # sigmoid 함수
    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    # 순전파 과정
    def forward(self, X):
        Z1 = np.dot(self.W1, X) + self.b1
        A1 = self.relu(Z1)
        Z2 = np.dot(self.W2, A1) + self.b2
        A2 = self.sigmoid(Z2)

        cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
        return A2, cache

    # 손실 함수 계산
    def compute_loss(self, A2, Y):
        m = Y.shape[1]
        loss = - (1/m) * np.sum(Y * np.log(A2 + 1e-8) + (1 - Y) * np.log(1 - A2 + 1e-8))
        return loss

    # 역전파 과정
    def backward(self, X, Y, cache):
        m = X.shape[1]

        A1 = cache['A1']
        A2 = cache['A2']
        Z1 = cache['Z1']

        dZ2 = A2 - Y  
        dW2 = (1/m) * np.dot(dZ2, A1.T)  
        db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)  

        dA1 = np.dot(self.W2.T, dZ2)
        dZ1 = dA1 * self.relu_derivative(Z1)
        dW1 = (1/m) * np.dot(dZ1, X.T)
        db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

        gradients = {"dW1": dW1, "db1": db1,
                     "dW2": dW2, "db2": db2}
        return gradients

    # 가중치 업데이트
    def update_parameters(self, gradients):
        self.W1 -= self.learning_rate * gradients['dW1']
        self.b1 -= self.learning_rate * gradients['db1']
        self.W2 -= self.learning_rate * gradients['dW2']
        self.b2 -= self.learning_rate * gradients['db2']

    # 학습
    def train(self, X, Y, print_loss=False):
        for epoch in range(self.epochs):
            # 순전파
            A2, cache = self.forward(X)

            # 손실 계산
            loss = self.compute_loss(A2, Y)

            # 역전파
            gradients = self.backward(X, Y, cache)

            # 가중치와 바이어스 업데이트
            self.update_parameters(gradients)

            # 손실 출력
            if print_loss and (epoch+1) % 1000 == 0:
                print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

    def predict(self, X):
        A2, _ = self.forward(X)
        predictions = (A2 > 0.5).astype(int)
        return predictions


# 데이터 생성
X, y = make_moons(n_samples=1000, noise=0.2, random_state=20241119)

# 데이터 전처리: 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 레이블을 (1 x m) 형태로 변환
Y = y.reshape(1, -1)

# 입력 데이터를 (input_size x m) 형태로 변환
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y.T, test_size=0.2, random_state=42)
X_train = X_train.T  # (2 x m_train)
X_test = X_test.T    # (2 x m_test)
Y_train = Y_train.T  # (1 x m_train)
Y_test = Y_test.T    # (1 x m_test)

# MLP 생성
mlp = MLP(input_size=2, hidden_size=100, output_size=1, learning_rate=0.1, epochs=10000)

# 학습
mlp.train(X_train, Y_train, print_loss=True)

# 학습 데이터에 대한 예측
train_predictions = mlp.predict(X_train)
train_accuracy = np.mean(train_predictions == Y_train) * 100
print(f"Training Accuracy: {train_accuracy:.2f}%")

# 테스트 데이터에 대한 예측
test_predictions = mlp.predict(X_test)
test_accuracy = np.mean(test_predictions == Y_test) * 100
print(f"Test Accuracy: {test_accuracy:.2f}%")

def plot_decision_boundary(mlp, X, Y):
    # 데이터 포인트 시각화
    cnt_0, cnt_1 = 0, 0
    plt.figure(figsize=(10, 6))
    for i in range(Y.shape[1]):
        if Y[0, i] == 0:
            plt.scatter(X[0, i], X[1, i], color='red', marker='o', label='0' if cnt_0 == 0 else "")
            cnt_0 += 1
        elif Y[0,i] == 1:
            plt.scatter(X[0, i], X[1, i], color='blue', marker='x', label='1' if cnt_1 == 0 else "")
            cnt_1 += 1
    # 그리드 생성
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()].T  # (2 x 40000)
    Z = mlp.predict(grid)
    Z = Z.reshape(xx.shape[0],xx.shape[1])

    # 결정 경계 시각화
    plt.contourf(xx, yy, Z, alpha=0.2, cmap=plt.cm.Paired)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.title('MLP Decision Boundary')
    plt.grid(True)
    plt.show()

# 결정 경계 시각화
plot_decision_boundary(mlp, X_test, Y_test)
