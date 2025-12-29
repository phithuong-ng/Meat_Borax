import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# --- 1. CẤU HÌNH ---
FILE_PATH = 'parameters (1).csv'

# --- 2. TẢI DỮ LIỆU ---
print(">>> Đang tinh chỉnh PCA...")
try:
    df = pd.read_csv(FILE_PATH)
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file '{FILE_PATH}'.")
    exit()


def get_label(filename):
    name = str(filename).lower()
    if 'zin' in name: return 0
    if '0%' in name and '10%' not in name: return 0
    return 1


df['Label'] = df['FileName'].apply(get_label)

# --- 3. TRAIN SVM "KHẮT KHE" (CHỈ ĐỂ VẼ HÌNH ĐẸP) ---
features = ['Ri', 'Re', 'p_CPE1', 'T_CPE1']
X = df[features]
y = df['Label']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =============================================================================
# THAY ĐỔI QUAN TRỌNG:
# 1. C=1000: Ép đường biên cực gắt, không cho phép lấn chiếm.
# 2. gamma=0.5: Thu hẹp vùng ảnh hưởng, làm vùng xanh co cụm lại sát điểm dữ liệu.
# =============================================================================
svm_vis = SVC(kernel='rbf', gamma=0.5, C=1000, class_weight='balanced')
svm_vis.fit(X_scaled, y)

# --- 4. VẼ PCA ---
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(12, 9))

# Vẽ nền phân vùng
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))

Z = svm_vis.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
Z = Z.reshape(xx.shape)

# Vùng Xanh (Sạch) và Vùng Đỏ (Bẩn)
plt.contourf(xx, yy, Z, alpha=0.15, cmap=plt.cm.RdYlGn_r)

# Vẽ điểm dữ liệu
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['Label'], style=df['Label'],
                markers={0: 'o', 1: 'X'},
                palette={0: 'green', 1: 'red'},
                s=200, edgecolor='k', linewidth=1.5)  # Điểm to hơn cho dễ nhìn

# Gắn tên file thông minh (Tránh đè nhau)
texts = []
for i in range(len(df)):
    label = df.iloc[i]['Label']
    fname = df.iloc[i]['FileName'].replace('.csv', '')

    x_pos = X_pca[i, 0]
    y_pos = X_pca[i, 1]

    if label == 0:
        plt.text(x_pos, y_pos - 0.2, fname, color='darkgreen', fontsize=9, ha='center', weight='bold')
    else:
        # Với mẫu bẩn, chỉ hiện tên nếu là mẫu quan trọng (2%, 0phay5) để đỡ rối
        if '2%' in fname or '0phay5' in fname:
            plt.text(x_pos, y_pos + 0.2, fname, color='darkred', fontsize=9, ha='center', weight='bold')

plt.title('Phân vùng SVM (Đã tinh chỉnh hội tụ)', fontsize=16)
plt.xlabel('PC1')
plt.ylabel('PC2')

# Chú thích
from matplotlib.lines import Line2D

legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Sạch'),
                   Line2D([0], [0], marker='X', color='w', markeredgecolor='red', markersize=10, label='Có hàn the')]
plt.legend(handles=legend_elements, loc='upper right')

plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig('PCA_High_Quality.png')
print(">>> Đã vẽ xong: 'PCA_High_Quality.png'. Bạn xem thử hình này xem đã 'khôn' hơn chưa nhé!")
plt.show()