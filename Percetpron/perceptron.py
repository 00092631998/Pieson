import numpy as np
import matplotlib.pyplot as plt

class Perceptron:

    # 학습률과 에포크는 각각 0.01, 100으로 지정
    def __init__(self, input_size, learning_rate=0.01, epochs=100):
        self.weights = np.zeros(input_size) # 최초 가중치 0으로 초기화
        self.bias = 0
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.history = []

    # 활성화 함수(Step function)
    def activation_function(self, z):
        return np.where(z >= 0, 1, 0)

    # 순전파
    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self.activation_function(z)

    # 역전파
    def fit(self, X, y):
        for epoch in range(self.epochs):
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.activation_function(z)
            errors = y - y_pred
            self.weights += self.learning_rate * np.dot(X.T, errors)
            self.bias += self.learning_rate * np.sum(errors)
            self.history.append((self.weights.copy(), self.bias))

            # 조기 종료
            if np.all(errors == 0):
                print(f"학습이 {epoch+1}번째 에포크에서 수렴했습니다.")
                break

    # 결정경계
    def plot_decision_boundary(self, X, y):
        plt.figure(figsize=(10, 6))
        
        # 데이터 포인트 그리기
        for idx, point in enumerate(X):
            if y[idx] == 0:
                plt.scatter(point[0], point[1], color='red', marker='o', label='0' if idx == 0 else "")
            else:
                plt.scatter(point[0], point[1], color='blue', marker='x', label='1' if idx == 3 else "")
        
        # 결정 경계 그리기
        x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
        y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
        x_values = np.linspace(x_min, x_max, 200)
        y_values = -(self.weights[0] * x_values + self.bias) / self.weights[1]
        plt.plot(x_values, y_values, color='green', label='Decision Boundary')
        
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.legend()
        plt.title('Perceptron Decision Boundary')
        plt.grid(True)
        plt.show()

    # 학습중 결정경계
    def plot_learning_process(self, X, y):
        plt.figure(figsize=(10, 6))

        # 데이터 포인트 그리기
        for idx, point in enumerate(X):
            if y[idx] == 0:
                plt.scatter(point[0], point[1], color='red', marker='o', label='0' if idx == 0 else "")
            else:
                plt.scatter(point[0], point[1], color='blue', marker='x', label='1' if idx == 3 else "")

        x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
        x_values = np.linspace(x_min, x_max, 200)

        # 각 학습 단계에서의 결정 경계 그리기
        for i in range(0, len(self.history), max(1, len(self.history)//10)):
            weights, bias = self.history[i]
            if weights[1] != 0:
                y_vals = -(weights[0] * x_values + bias) / weights[1]
                plt.plot(x_values, y_vals, color='gray', alpha=0.3)

        # 최종 결정 경계 그리기
        if self.weights[1] != 0:
            y_final = -(self.weights[0] * x_values + self.bias) / self.weights[1]
            plt.plot(x_values, y_final, color='green', label='Final Decision Boundary')

        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.legend()
        plt.title('Perceptron Learning Process')
        plt.grid(True)
        plt.show()
