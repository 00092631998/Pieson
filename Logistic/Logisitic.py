import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib.colors import ListedColormap

# Iris 데이터셋 로드
iris = load_iris()
X = iris.data[:, [2, 3]]  # 꽃잎 길이와 꽃잎 너비
y = iris.target
feature_names = iris.feature_names[2:4]
target_names = iris.target_names

# 데이터프레임 생성
df = pd.DataFrame(X, columns=feature_names)
df['species'] = y
df['species'] = df['species'].apply(lambda x: target_names[x])

# 학습용 데이터와 테스트용 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=25, stratify=y)

# 특징 스케일링
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 다중 클래스 로지스틱 회귀 모델 초기화 및 학습
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', C=1.0, max_iter=200)
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)


# 결정 경계 그리기
def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ('s', 'x', 'o')
    colors = ('red', 'blue', 'lightgreen')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')

# 결정 경계 시각화
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

plt.figure(figsize=(10,8))
plot_decision_regions(X_combined, y_combined, classifier=model)
plt.title('Multinomial Logistic Regression Decision Boundaries')
plt.xlabel('Standardized Petal Length')
plt.ylabel('Standardized Petal Width')
plt.legend(loc='upper left')
plt.show()
