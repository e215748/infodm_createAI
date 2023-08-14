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

print(X[0])