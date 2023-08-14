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
model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# テストデータセットでの予測
y_pred = model.predict(X_test)

# 正答率の計算
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# 手書き画像の読み込みと前処理
i = 1
j = 0
while i <= 6:
    while j <= 9:
        image = Image.open('digits' + str(i) + '_' + str(j) +'.PNG').convert('L')  # 手書き画像のパスを指定して読み込む
        image = image.resize((28, 28))  # 28x28にリサイズ
        image = np.array(image)  # NumPy配列に変換
        image = image.reshape(1, -1)  # 1次元の特徴量ベクトルに変換
        #image = image / 255.0  # スケーリング (0から1の範囲に正規化)
        image = 255.0 - image

        # ラベルの作成
        label = np.array([j])

        # データを追加
        X_train = np.vstack((X_train, image))
        y_train = np.concatenate((y_train, label))
        j += 1
    i += 1


y_train = np.array(y_train, dtype=int)

# 新しい学習データでモデルを再訓練
model.fit(X_train, y_train)
