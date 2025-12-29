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
print(">>> Đang xử lý dữ liệu PCA...")
try:
    df = pd.read_csv(FILE_PATH)
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file '{FILE_PATH}'.")
    exit()


# Dán nhãn
def get_label(filename):
    name = str(filename).lower()
    # Sạch: Chứa 'zin' hoặc '0%' (trừ 10%)
    if 'zin' in name: return 0
    if '0%' in name and '10%' not in name: return 0
    return 1  # Bẩn


df['Label'] = df['FileName'].apply(get_label)

# In kiểm tra số lượng
n_clean = len(df[df['Label'] == 0])
n_dirty = len(df[df['Label'] == 1])
print(f"Trạng thái dữ liệu: {n_clean} Sạch - {n_dirty} Bẩn -> {('CÂN BẰNG' if n_clean == n_dirty else 'CHÊNH LỆCH')}")

# --- 3. TRAIN SVM (ĐỂ VẼ NỀN) ---
features = ['Ri', 'Re', 'p_CPE1', 'T_CPE1']
X = df[features]
y = df['Label']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Cấu hình chuẩn, không thiên vị
# C=50: Đủ cứng rắn để phân loại chính xác
svm = SVC(kernel='rbf', gamma='scale', C=50, class_weight='balanced')
svm.fit(X_scaled, y)

# --- 4. VẼ PCA ---
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(12, 9))

# Vẽ nền phân vùng (Decision Boundary)
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))

Z = svm.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
Z = Z.reshape(xx.shape)

# Vùng Xanh (Sạch) và Vùng Đỏ (Bẩn)
plt.contourf(xx, yy, Z, alpha=0.15, cmap=plt.cm.RdYlGn_r)

# Vẽ điểm dữ liệu (QUAN TRỌNG: Marker đúng yêu cầu)
# 0 (Sạch): 'o' (Tròn), màu Xanh
# 1 (Bẩn): 'X' (Chéo), màu Đỏ
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['Label'], style=df['Label'],
                markers={0: 'o', 1: 'X'},
                palette={0: 'green', 1: 'red'},
                s=150, edgecolor='k', linewidth=1.5)

# Gắn tên file
for i in range(len(df)):
    label = df.iloc[i]['Label']
    fname = df.iloc[i]['FileName'].replace('.csv', '')

    # Chỉnh vị trí chữ
    if label == 0:
        plt.text(X_pca[i, 0], X_pca[i, 1] - 0.18, fname, color='darkgreen', fontsize=8, ha='center', weight='bold')
    else:
        plt.text(X_pca[i, 0], X_pca[i, 1] + 0.18, fname, color='darkred', fontsize=8, ha='center')

plt.title('Phân bố không gian PCA (Dữ liệu cân bằng 20-20)', fontsize=16)
plt.xlabel('PC1 (Thành phần chính 1)')
plt.ylabel('PC2 (Thành phần chính 2)')

# Tạo chú thích thủ công cho đúng ý
from matplotlib.lines import Line2D

custom_lines = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Sạch'),
    Line2D([0], [0], marker='X', color='w', markeredgecolor='red', markersize=10, label='Có hàn the')]
plt.legend(handles=custom_lines, title='Chú thích')

plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig('PCA_Balanced_20vs20.png')
print(">>> Đã vẽ xong biểu đồ: 'PCA_Balanced_20vs20.png'")
plt.show()