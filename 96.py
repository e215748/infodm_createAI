import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


# MNISTデータセットをダウンロード
mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)


# 特徴量とラベルを取得
X = mnist.data
y = mnist.target


# データを訓練用とテスト用に分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# モデルの構築と訓練
model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
model.fit(X_train, y_train)


# テストデータで予測
y_pred = model.predict(X_test)


# 正解率の計算
accuracy = accuracy_score(y_test, y_pred)
print("テストデータの正解率:", accuracy)


# ランダムに選んだテストデータの表示
index = np.random.randint(0, len(X_test))


if index < len(X_test):
   digit = X_test[index].reshape(28, 28)
   label = y_pred[index]


   plt.imshow(digit, cmap='gray')
   plt.title("Predicted Label: " + str(label))
   plt.axis('off')
   plt.show()
else:
   print("指定されたインデックスは有効範囲外です。")