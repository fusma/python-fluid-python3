# %%
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna
import tqdm

# %%


def calc_length(arr):
    res = 0
    for s in (0, 4):
        res += ((arr[s] - arr[s + 2]) ** 2 +
                (arr[s + 1] - arr[s + 3]) ** 2)
    return res


def min_length(arr):
    res = 100
    arr = arr[:8]
    # numpyに
    arr = np.array(arr)
    for s in (0, 4):
        res = min(res, (arr[s] - arr[s + 2]) ** 2 +
                  (arr[s + 1] - arr[s + 3]) ** 2)
    print(f"\r{res}", end="")
    return res


def cherry_pick(df, threshold=10):
    # data[:8]のmin_lengthが10未満のものをdrop
    df = df.apply(lambda x: min_length(x) >= threshold, axis=0)


# %%
prefix = "test1000"
for i in range(10):
    filename = f"{prefix}/{prefix}_{i}.csv"
    # csvファイルの読み込み
    if i == 0:
        df = pd.read_csv(filename, header=0)
        df = df.drop("Unnamed: 0", axis=1)
    else:
        tmp = pd.read_csv(filename, header=0)
        tmp = tmp.drop("Unnamed: 0", axis=1)
        df = pd.concat([df, tmp], axis=0)

cherry_pick(df)


# dataframeをnumpy配列に変換
data = df.values
# 深層学習の入力データと出力データに分ける
# 入力データ
len_test = 8
x = data[:, len_test:]

# x[i]をN×Nに変換
siz = 52
x = x.reshape(-1, siz, siz)

# 出力データ
y = data[:, :len_test]
# #入力データの正規化
# x = (x - x.mean()) / x.std()

# 入力データの各行をN×Nに変換


# 出力データの正規化
y = (y - y.mean()) / y.std()

# 入力データの次元数
input_dim = x.shape[1]

# 出力データの次元数
output_dim = y.shape[1]

# 入力データの数
n = x.shape[0]

# 学習データとテストデータに分ける
# 学習データ
x_train = x[:int(n*0.8)]
y_train = y[:int(n*0.8)]

# テストデータ
x_test = x[int(n*0.8):]
y_test = y[int(n*0.8):]


N = 52


# %%

# optunaを使わないCNN
def create_model():
    # モデルの定義
    base_model = tf.keras.applications.efficientnet.EfficientNetB0(
        include_top=False,
        weights=None,
        input_shape=(N, N, 1),
    )

    # 8個の出力を持つモデル
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(N, N, 1)),
        base_model,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(8),
    ])

    # モデルのコンパイル
    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"],
    )
    return model


# 学習の実行
def objective():
    model = create_model()
    history = model.fit(
        x_train,
        y_train,
        epochs=100,
        batch_size=32,
        validation_data=(x_test, y_test),
    )
    # 学習の様子をプロット
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.legend()
    plt.show()
    return history.history["val_loss"][-1]


objective()

# %%
# 1行目の値を実際に計算
print(model.predict(x_test[:1]))
# 実際の値
print(y_test[:1])
