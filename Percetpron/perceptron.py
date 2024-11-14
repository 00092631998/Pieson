import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, input_size, learning_rate=0.01, epochs=100):
        """
        퍼셉트론 초기화

        Parameters:
        - input_size: 입력 벡터의 크기
        - learning_rate: 학습률
        - epochs: 학습 반복 횟수
        """
        self.weights = np.zeros(input_size)
        self.bias = 0
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.history = []  # 결정 경계의 변화를 기록

    def activation_function(self, z):
        """
        활성화 함수: 계단 함수

        Parameters:
        - z: 가중합 (스칼라 또는 벡터)

        Returns:
        - 출력값 (0 또는 1)
        """
        return np.where(z >= 0, 1, 0)

    def predict(self, X):
        """
        예측 함수

        Parameters:
        - X: 입력 데이터 (2차원 배열)

        Returns:
        - 예측값 (0 또는 1)
        """
        z = np.dot(X, self.weights) + self.bias
        return self.activation_function(z)

    def fit(self, X, y):
        """
        학습 함수

        Parameters:
        - X: 입력 데이터 (2차원 배열)
        - y: 실제 출력값 (1차원 배열)
        """
        for epoch in range(self.epochs):
            # 가중합 계산
            z = np.dot(X, self.weights) + self.bias
            # 예측값 계산
            y_pred = self.activation_function(z)
            # 오차 계산
            errors = y - y_pred
            # 가중치 업데이트 (벡터화된 연산)
            self.weights += self.learning_rate * np.dot(X.T, errors)
            # 바이어스 업데이트 (벡터화된 연산)
            self.bias += self.learning_rate * np.sum(errors)
            
            # 결정 경계 기록 (가중치와 바이어스의 현재 상태)
            self.history.append((self.weights.copy(), self.bias))
            
            # 오차가 0인 경우 조기 종료
            if np.all(errors == 0):
                print(f"학습이 {epoch+1}번째 에포크에서 수렴했습니다.")
                break

    def plot_decision_boundary(self, X, y):
        """
        결정 경계를 시각화하는 함수

        Parameters:
        - X: 입력 데이터
        - y: 실제 출력값
        """
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

    def plot_learning_process(self, X, y):
        """
        학습 과정에서의 결정 경계를 시각화하는 함수

        Parameters:
        - X: 입력 데이터
        - y: 실제 출력값
        """
        plt.figure(figsize=(10, 6))

        # 데이터 포인트 그리기
        for idx, point in enumerate(X):
            if y[idx] == 0:
                plt.scatter(point[0], point[1], color='red', marker='o', label='0' if idx == 0 else "")
            else:
                plt.scatter(point[0], point[1], color='blue', marker='x', label='1' if idx == 3 else "")

        # 결정 경계를 그리기 위한 x_values 정의
        x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
        x_values = np.linspace(x_min, x_max, 200)

        # 각 학습 단계에서의 결정 경계 그리기 (일부만 시각화하여 복잡도 감소)
        for i in range(0, len(self.history), max(1, len(self.history)//10)):
            weights, bias = self.history[i]
            if weights[1] != 0:  # 분모가 0이 되지 않도록 확인
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

