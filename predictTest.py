import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier


# MNISTデータセットをダウンロード
mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)


# 特徴量とラベルを取得
X = mnist.data
y = mnist.target


# データの前処理
X = X / 255.0  # ピクセル値を0から1の範囲にスケーリング


# モデルの構築と訓練
model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
model.fit(X, y)


# 数字を描画して予測する関数
def predict_digit():
    # 数字の描画
    fig, ax = plt.subplots()
    ax.imshow(np.zeros((28, 28)), cmap='gray', vmin=0, vmax=1)
    ax.set_title('Draw a digit (0-9)')
    ax.set_xticks([])
    ax.set_yticks([])

    def draw(event):
        if event.button == 1:
            x = int(event.xdata)
            y = int(event.ydata)
            ax.plot(x, y, marker='o', markersize=10, color='white')
            fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', draw)
    plt.show()

    # 描画された数字を予測
    img = np.array(fig.canvas.renderer._renderer)[:,:,:3]
    img = img.mean(axis=2)  # グレースケールに変換
    img = img / 255.0  # ピクセル値を0から1の範囲にスケーリング
    img = img.flatten().reshape(1, -1)

    predicted_label = model.predict(img)[0]
    print("予測されたラベル:", predicted_label)


# 数字を描画して予測を実行
predict_digit()