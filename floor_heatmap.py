import numpy as np
from sklearn.mixture import GaussianMixture

# 3000x3000の座標平面を作成
x = np.linspace(0, 3000, 3000)
y = np.linspace(0, 3000, 3000)
X, Y = np.meshgrid(x, y)

# 電波発信端末の座標とGMMのパラメータを設定
transmitter_1 = [500, 2500]  # 端末3の平均 (中心座標)
transmitter_2 = [2500, 500]  # 端末4の平均 (中心座標)
covariance_matrix_1 = [[50000, 0], [0, 50000]]  # 端末3の共分散行列
covariance_matrix_2 = [[30000, 10000], [10000, 30000]]  # 端末4の共分散行列
weights = [0.5, 0.5]  # GMMの重み

# ガウス混合モデルを初期化
gmm = GaussianMixture(n_components=2, covariance_type="full", weights_init=weights)

# ガウス混合モデルを学習
gmm.means_init = np.array([transmitter_1, transmitter_2])
gmm.covariances_init = np.array([covariance_matrix_1, covariance_matrix_2])
gmm.fit(np.column_stack((X.ravel(), Y.ravel())))

# 各座標点での電波強度を計算
Z = -gmm.score_samples(np.column_stack((X.ravel(), Y.ravel())))
Z = Z.reshape(X.shape)

# Zの最大値を取得
Z_max = np.max(Z)

# 実際のRSSI値にスケーリング (例: -30dBm から -90dBm の範囲)
RSSI_max = -30  # dBm
RSSI_min = -90  # dBm

# 正規化された Z を実際のRSSI値にスケーリングし、小数点以下2桁に丸める
RSSI = np.round(RSSI_max + (Z / Z_max) * (RSSI_min - RSSI_max), 2)

# RSSI値をCSVファイルに書き込む
np.savetxt('RSSI_values.csv', RSSI, delimiter=',', fmt='%.2f')

# メッセージを表示
print("RSSI値がファイル 'RSSI_values.csv' に保存されました。")
