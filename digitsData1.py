import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
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

# 手書き画像の読み込みと前処理
i = 0
while i <= 9:
    image = Image.open(str(i) + '.PNG').convert('L')  # 手書き画像のパスを指定して読み込む
    image = image.resize((28, 28))  # 200x200にリサイズ
    image = np.array(image)  # NumPy配列に変換
    image = image.reshape(1, -1)  # 1次元の特徴量ベクトルに変換
    # モデルによる予測
    prediction = model.predict(image)
    print(prediction)
    i += 1

# 結果の表示
plt.imshow(image.reshape(28, 28), cmap='gray')
plt.title("Predicted Label: " + str(prediction[0]))
plt.axis('off')
plt.show()# data-mining
