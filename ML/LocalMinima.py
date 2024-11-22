import numpy as np
import matplotlib.pyplot as plt

# 함수
def f(x):
    return x**4 - 5*x**3 + 7*x**2

# 함수의 기울기기
def grad_f(x):
    return 4*x**3 - 15*x**2 + 14*x

# 경사하강법
def gradient_descent(initial_x, learning_rate, iterations):
    x = initial_x
    x_history = [x]
    f_history = [f(x)]
    
    for i in range(iterations):
        gradient = grad_f(x)
        x = x - learning_rate * gradient
        x_history.append(x)
        f_history.append(f(x))
        print(f"Iteration {i+1}: x = {x:.6f}, f(x) = {f(x):.6f}")
    
    return x_history, f_history

# 초기값, 학습률, 반복 횟수 설정
initial_x = 3.0  
learning_rate = 0.01
iterations = 50  

# 경사하강법 실행
x_hist, f_hist = gradient_descent(initial_x, learning_rate, iterations)

# 그래프 그리기
x = np.linspace(-1, 4, 400)
y = f(x)

plt.figure(figsize=(12, 6))
plt.plot(x, y, label='f(x) = x^4 - 5x^3 + 7x^2', color='blue')

# 경사하강법 점 표시
plt.scatter(x_hist, f_hist, color='red', zorder=5, label='Gradient Descent Steps')

# 각 점에 번호 추가
for i, (xi, yi) in enumerate(zip(x_hist, f_hist)):
    plt.text(xi, yi + 0.2, f'{i}', color='red', fontsize=8, ha='center')

plt.title('Non-convex')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()
