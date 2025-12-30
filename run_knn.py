import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- 1. SETUP ---
print(">>> Đang vẽ biểu đồ Trung lập (Clean Plot)...")
try:
    df = pd.read_csv('parameters (1).csv')
except:
    print("Lỗi file.")
    exit()


def get_label(filename):
    name = str(filename).lower()
    if 'zin' in name: return 0
    if '0%' in name and '10%' not in name: return 0
    return 1


df['Label'] = df['FileName'].apply(get_label)
X = df[['Ri', 'Re', 'p_CPE1', 'T_CPE1']]

# Scale & PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# --- 2. VẼ BIỂU ĐỒ SẠCH (KHÔNG VÙNG MÀU) ---
plt.figure(figsize=(10, 8))

# Vẽ điểm dữ liệu: Tròn Xanh - X Đỏ
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['Label'], style=df['Label'],
                markers={0: 'o', 1: 'X'}, palette={0: '#008000', 1: '#cc0000'},
                s=200, edgecolor='black', linewidth=1.5, alpha=0.9)

# Gắn tên file (Chọn lọc)
for i in range(len(df)):
    fname = df.iloc[i]['FileName'].replace('.csv', '')
    label = df.iloc[i]['Label']
    x, y_pos = X_pca[i, 0], X_pca[i, 1]

    # Logic: Chỉ hiện tên các mẫu quan trọng
    if label == 1:  # Bẩn -> Hiện hết màu đỏ
        plt.text(x, y_pos + 0.18, fname, color='#cc0000', fontsize=9, ha='center', weight='bold')
    elif i % 2 == 0:  # Sạch -> Hiện màu xanh
        plt.text(x, y_pos - 0.22, fname, color='#006400', fontsize=8, ha='center', weight='bold')

plt.title('Phân bố dữ liệu trên không gian PCA (20 Sạch - 20 Bẩn)', fontsize=16)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)')

# Vẽ vòng tròn ảo bao quanh cụm (Thủ thuật visual)
from matplotlib.patches import Ellipse

# Vòng tròn bao cụm Sạch (Ước lượng)
clean_center = np.mean(X_pca[df['Label'] == 0], axis=0)
clean_ellipse = Ellipse(clean_center, width=5, height=3, angle=-10,
                        edgecolor='green', facecolor='green', alpha=0.05, linestyle='--')
plt.gca().add_patch(clean_ellipse)

# Vòng tròn bao cụm Bẩn
dirty_center = np.mean(X_pca[df['Label'] == 1], axis=0)
dirty_ellipse = Ellipse(dirty_center, width=6, height=4, angle=10,
                        edgecolor='red', facecolor='red', alpha=0.05, linestyle='--')
plt.gca().add_patch(dirty_ellipse)

# Chú thích
from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#008000', markersize=12, label='Thịt Sạch (Clean)'),
    Line2D([0], [0], marker='X', color='w', markeredgecolor='#cc0000', markersize=12, label='Có Hàn the (Borax)')
]
plt.legend(handles=legend_elements, loc='upper right', fontsize=12)

plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('Clean_Scatter_Plot.png', dpi=300)
print(">>> Đã xong: 'Clean_Scatter_Plot.png'. Ảnh này bao đẹp, bao sạch!")
plt.show()