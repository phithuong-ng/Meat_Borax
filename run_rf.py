import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- 1. SETUP ---
print(">>> [RF Map] Đang vẽ bản đồ Random Forest...")
try:
    df = pd.read_csv('parameters (1).csv')
except:
    print("Lỗi: Không tìm thấy file.")
    exit()

def get_label(filename):
    name = str(filename).lower()
    if 'zin' in name: return 0
    if '0%' in name and '10%' not in name: return 0
    return 1 # Bẩn

df['Label'] = df['FileName'].apply(get_label)
X = df[['Ri', 'Re', 'p_CPE1', 'T_CPE1']]
y = df['Label']

# Scale dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 2. XỬ LÝ PCA TRƯỚC (Mẹo để vẽ hình RF 2D) ---
# Để vẽ được vùng của RF trên mặt phẳng 2D, ta cần train RF trực tiếp trên dữ liệu 2D sau khi PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Train Random Forest trên dữ liệu 2D này
rf_2d = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_2d.fit(X_pca, y)

# --- 3. VẼ VÙNG QUYẾT ĐỊNH ---
plt.figure(figsize=(10, 8))

# Tạo lưới điểm
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))

# Dự đoán
Z = rf_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Vẽ nền màu
plt.contourf(xx, yy, Z, alpha=0.15, cmap=plt.cm.RdYlGn_r)

# Vẽ điểm dữ liệu
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=df['Label'], style=df['Label'],
                markers={0: 'o', 1: 'X'}, palette={0: 'green', 1: 'red'},
                s=150, edgecolor='k', linewidth=1.5)

# Gắn tên file
for i in range(len(df)):
    fname = df.iloc[i]['FileName'].replace('.csv', '')
    color = 'darkgreen' if df.iloc[i]['Label'] == 0 else 'darkred'
    plt.text(X_pca[i,0], X_pca[i,1]-0.2, fname, color=color, fontsize=8, ha='center', weight='bold')

plt.title('Bản đồ phân vùng Random Forest (Thấy rõ dạng khối hộp)', fontsize=15)
plt.xlabel('PC1')
plt.ylabel('PC2')

# Chú thích
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Sạch'),
    Line2D([0], [0], marker='X', color='w', markeredgecolor='red', markersize=10, label='Có hàn the')
]
plt.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.savefig('RandomForest_Boundary_Map.png')
print(">>> Đã xuất ảnh: 'RandomForest_Boundary_Map.png'")
plt.show()