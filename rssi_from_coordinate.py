import numpy as np
from sklearn.mixture import GaussianMixture

# 電波発信端末の座標とGMMのパラメータを設定
transmitter_1 = [500, 500]  # 端末1の平均 (中心座標)
transmitter_2 = [2500, 2500]  # 端末2の平均 (中心座標)

covariance_matrix_1 = [[50000, 0], [0, 50000]]  # 端末1の共分散行列
covariance_matrix_2 = [[50000, 0], [0, 50000]]  # 端末2の共分散行列
weights = [0.5, 0.5]  # GMMの重み

# ガウス混合モデルを初期化
gmm = GaussianMixture(n_components=2, covariance_type="full", weights_init=weights)

# ガウス混合モデルを学習
gmm.means_init = np.array([transmitter_1, transmitter_2])
gmm.covariances_init = np.array([covariance_matrix_1, covariance_matrix_2])
x = np.linspace(0, 3000, 3000)
y = np.linspace(0, 3000, 3000)
X, Y = np.meshgrid(x, y)
gmm.fit(np.column_stack((X.ravel(), Y.ravel())))

# 実際のRSSI値にスケーリング (例: -30dBm から -90dBm の範囲)
RSSI_max = -30  # dBm
RSSI_min = -90  # dBm


def get_rssi_at_coordinates(x, y):
    # 座標での電波強度を計算
    Z = -gmm.score_samples(np.array([[x, y]]))

    # Zの最大値を取得
    Z_max = np.max(Z)

    # 正規化された Z を実際のRSSI値にスケーリング
    RSSI = RSSI_max + (Z / Z_max) * (RSSI_min - RSSI_max)

    return RSSI[0]


def get_random_rssi(x, y):
    rssi_1 = get_rssi_at_coordinates(x, y)
    rssi_2 = get_rssi_at_coordinates(x, y)
    return np.random.choice([rssi_1, rssi_2], 2)


# 使用例
x_coord = 1000
y_coord = 1000
rssi_values = get_random_rssi(x_coord, y_coord)
print(f"座標 ({x_coord}, {y_coord}) でのランダムに選ばれた2つのRSSI値: {rssi_values}")
