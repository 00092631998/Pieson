import numpy as np
import matplotlib.pyplot as plt

# 활성화 함수 및 그 미분 정의
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1.0 - np.tanh(x)**2

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def softmax_derivative(x):
    s = softmax(x)
    return np.diagflat(s) - np.dot(s, s.T)

def softplus(x):
    return np.log1p(np.exp(x))

def softplus_derivative(x):
    return sigmoid(x)

def swish(x):
    return x * sigmoid(x)

def swish_derivative(x):
    s = sigmoid(x)
    return s * (1 + x * (1 - s))

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

def gelu_derivative(x):
    tanh_inner = np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))
    tanh_val = np.tanh(tanh_inner)
    return 0.5 * (1 + tanh_val) + 0.5 * x * (1 - tanh_val**2) * np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * x**2)

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def elu_derivative(x, alpha=1.0):
    return np.where(x > 0, 1, alpha * np.exp(x))

def selu(x):
    lambda_ = 1.0507009873554804934193349852946
    alpha = 1.6732632423543772848170429916717
    return lambda_ * np.where(x > 0, x, alpha * (np.exp(x) - 1))

def selu_derivative(x):
    lambda_ = 1.0507009873554804934193349852946
    alpha = 1.6732632423543772848170429916717
    return lambda_ * np.where(x > 0, 1, alpha * np.exp(x))

# x 범위 설정
x = np.linspace(-5, 5, 400)

# 활성화 함수 목록
activation_functions = {
    'Sigmoid': (sigmoid, sigmoid_derivative),
    'Tanh': (tanh, tanh_derivative),
    'ReLU': (relu, relu_derivative),
    'Leaky ReLU': (leaky_relu, leaky_relu_derivative),
    'Softplus': (softplus, softplus_derivative),
    'Swish': (swish, swish_derivative),
    'GELU': (gelu, gelu_derivative),
    'ELU': (elu, elu_derivative),
    'SELU': (selu, selu_derivative)
}

# 그래프 크기 설정
num_functions = len(activation_functions)
plt.figure(figsize=(15, num_functions * 4))

for i, (name, (func, derivative)) in enumerate(activation_functions.items(), 1):
    # 활성화 함수 값 계산
    y = func(x)
    
    # 미분 값 계산
    y_derivative = derivative(x)
    
    # 활성화 함수 그래프
    plt.subplot(num_functions, 2, 2 * i - 1)
    plt.plot(x, y, label=f'{name}')
    plt.title(f'{name} func')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True)
    plt.legend()
    
    # 미분 그래프
    plt.subplot(num_functions, 2, 2 * i)
    plt.plot(x, y_derivative, label=f'{name} derivative', color='orange')
    plt.title(f'{name} derivative')
    plt.xlabel('x')
    plt.ylabel("f'(x)")
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.show()
