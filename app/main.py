import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Irisデータセットをロード
iris = datasets.load_iris()
X = iris.data
y = iris.target

# データセットをトレーニングセットとテストセットに分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# KNN分類器を初期化
knn = KNeighborsClassifier(n_neighbors=3)

# モデルをトレーニングデータでトレーニング
knn.fit(X_train, y_train)

# テストデータを使用して予測
y_pred = knn.predict(X_test)

# モデルの精度を評価
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# サンプルデータの予測
sample_data = np.array([[5.0, 3.6, 1.4, 0.2]])
predicted_class = knn.predict(sample_data)
print(f"Predicted class: {predicted_class[0]}")
