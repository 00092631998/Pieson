import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# 데이터 생성
np.random.seed(20000926)
X = np.linspace(-3, 3, 100)
y = X ** 3 - 3 * X ** 2 + X + np.random.randn(100) * 3  # 실제로는 3차 함수

X = X.reshape(-1, 1)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def plot_model_train_test(degree):
    # 다항 특징 생성
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    # 모델 학습
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    # 예측
    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)
    
    # 평균 제곱 오차 계산
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    
    # 그래프 그리기
    plt.scatter(X_train, y_train, color='blue', label='train data')
    plt.scatter(X_test, y_test, color='red', label='test data')
    X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    X_line_poly = poly.transform(X_line)
    y_line_pred = model.predict(X_line_poly)
    plt.plot(X_line, y_line_pred, label=f'{degree}_model', color='green')
    plt.legend()
    plt.title(f'{degree}_model (train MSE: {mse_train:.2f}, test MSE: {mse_test:.2f})')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.show()

# 과소적합 모델 (1)
plot_model_train_test(degree=1)

# 적절한 모델 (4)
plot_model_train_test(degree=4)

# 과적합 모델 (15)
plot_model_train_test(degree=15)

# 과적합 모델 (30)
plot_model_train_test(degree=20)
