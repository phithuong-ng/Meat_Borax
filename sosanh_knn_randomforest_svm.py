import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ==============================================================================
# 1. TẢI DỮ LIỆU & CẤU HÌNH
# ==============================================================================
FILE_PATH = 'parameters (1).csv'
print(">>> [1/4] Đang khởi tạo hệ thống kiểm định...")

try:
    df = pd.read_csv(FILE_PATH)
except:
    print("Lỗi: Không tìm thấy file csv.")
    exit()


# Dán nhãn
def get_label(filename):
    name = str(filename).lower()
    if 'zin' in name: return 0
    if '0%' in name and '10%' not in name: return 0
    return 1  # Bẩn (Positive)


df['Label'] = df['FileName'].apply(get_label)

X = df[['Ri', 'Re', 'p_CPE1', 'T_CPE1']]
y = df['Label']

# Scale dữ liệu (Bắt buộc cho PCA và SVM/KNN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
loo = LeaveOneOut()

# ==============================================================================
# 2. THỰC NGHIỆM SO SÁNH (BENCHMARKING)
# ==============================================================================
print(">>> [2/4] Đang chạy đua giữa các mô hình (LOOCV)...")

# Định nghĩa các ứng cử viên
models = {
    'SVM (RBF)': SVC(kernel='rbf', C=50, class_weight='balanced'),
    'KNN (k=3)': KNeighborsClassifier(n_neighbors=3),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
}

results = []

for name, model in models.items():
    # Dự đoán mù (Blind Test) toàn bộ dữ liệu
    y_pred = cross_val_predict(model, X_scaled, y, cv=loo)

    # Tính điểm
    acc = accuracy_score(y, y_pred) * 100
    sens = recall_score(y, y_pred, zero_division=0) * 100  # Độ nhạy
    spec = precision_score(y, y_pred, zero_division=0) * 100  # Độ chính xác dương tính

    results.append({
        'Model': name,
        'Accuracy': acc,
        'Sensitivity': sens,
        'Precision': spec
    })

# Chuyển thành DataFrame để dễ so sánh
res_df = pd.DataFrame(results).set_index('Model')

# ==============================================================================
# 3. PHÂN TÍCH ĐẶC TRƯNG (FEATURE IMPORTANCE)
# ==============================================================================
print(">>> [3/4] Đang phân tích yếu tố ảnh hưởng...")
rf_feature = RandomForestClassifier(n_estimators=100, random_state=42)
rf_feature.fit(X, y)
importances = rf_feature.feature_importances_
feat_names = ['Ri', 'Re', 'n (p_CPE1)', 'Q (T_CPE1)']
top_idx = np.argmax(importances)
top_feat = feat_names[top_idx]

# ==============================================================================
# 4. SINH BÁO CÁO TỰ ĐỘNG (AUTO-GENERATED REPORT)
# ==============================================================================
print("\n" + "=" * 80)
print("             BÁO CÁO KẾT QUẢ NGHIÊN CỨU (FINAL ANALYSIS)")
print("=" * 80)

# Phần 1: Bảng so sánh
print("\n[I. BẢNG SO SÁNH HIỆU NĂNG CÁC MÔ HÌNH]")
print("-" * 80)
print(res_df.round(2))
print("-" * 80)

# Phần 2: Biện luận & Lựa chọn
# Tìm model có Accuracy cao nhất, nếu bằng nhau thì so Sensitivity
best_model = res_df.sort_values(by=['Accuracy', 'Sensitivity'], ascending=False).iloc[0]
best_name = best_model.name

print("\n[II. LỰA CHỌN MÔ HÌNH & BIỆN LUẬN]")
print(f"1. QUYẾT ĐỊNH CHỌN: **{best_name}**")
print(f"   - Lý do: Qua thực nghiệm so sánh, {best_name} đạt hiệu suất tổng thể tốt nhất")
print(f"     (Accuracy: {best_model['Accuracy']:.2f}%, Sensitivity: {best_model['Sensitivity']:.2f}%).")

if 'SVM' in best_name:
    print("   - Giải thích sâu: SVM phù hợp nhất với tập dữ liệu nhỏ (Small Data) nhờ nguyên lý")
    print("     Tối đa hóa lề (Max Margin). Nó tạo ra đường biên giới ổn định, ít bị nhiễu")
    print("     hơn so với KNN hay Random Forest khi số lượng mẫu hạn chế.")
elif 'KNN' in best_name:
    print("   - Giải thích sâu: Dữ liệu phân bố theo cụm rất rõ ràng, nên thuật toán dựa trên")
    print("     khoảng cách như KNN hoạt động hiệu quả và đơn giản.")
elif 'Random Forest' in best_name:
    print("   - Giải thích sâu: Random Forest vượt trội nhờ khả năng chống nhiễu (Robustness)")
    print("     và xử lý tốt các mối quan hệ phi tuyến tính phức tạp giữa các thông số đo.")

print(f"\n2. ĐÁNH GIÁ ĐỘ AN TOÀN (SENSITIVITY):")
if best_model['Sensitivity'] >= 95:
    print(f"   - Kết quả: {best_model['Sensitivity']:.2f}% -> RẤT XUẤT SẮC.")
    print("   - Ý nghĩa: Hệ thống gần như không bỏ sót mẫu thịt bẩn nào. Đây là yếu tố quan trọng")
    print("     nhất trong bài toán vệ sinh an toàn thực phẩm.")
else:
    print(f"   - Kết quả: {best_model['Sensitivity']:.2f}% -> KHÁ TỐT.")
    print("   - Ý nghĩa: Vẫn còn tỷ lệ nhỏ bỏ sót (False Negative). Cần mở rộng tập dữ liệu huấn luyện.")

# Phần 3: Phân tích Vật lý
print(f"\n[III. CƠ SỞ VẬT LÝ & ĐẶC TRƯNG TÍN HIỆU]")
print(f"1. THAM SỐ ẢNH HƯỞNG NHẤT: **{top_feat}**")
if 'Q' in top_feat:
    print("   - Cơ chế: Hàn the làm biến đổi cấu trúc màng tế bào (protein), dẫn đến sự thay đổi")
    print("     lớn về khả năng tích điện (Điện dung Q). Mô hình AI xác định đây là 'dấu vân tay'")
    print("     rõ nét nhất để phân biệt, tin cậy hơn cả sự thay đổi điện trở.")
elif 'Re' in top_feat:
    print("   - Cơ chế: Hàn the là chất điện ly mạnh, làm tăng độ dẫn điện dịch ngoại bào.")
    print("     Sự sụt giảm mạnh của Re là chỉ dấu trực tiếp của sự hiện diện ion hàn the.")

print("=" * 80)

# ==============================================================================
# 5. VẼ BIỂU ĐỒ MINH HỌA
# ==============================================================================
plt.figure(figsize=(12, 6))

# Biểu đồ so sánh Accuracy
plt.subplot(1, 2, 1)
colors = ['#2ca02c' if name == best_name else '#1f77b4' for name in res_df.index]
bars = plt.bar(res_df.index, res_df['Accuracy'], color=colors)
plt.title('So sánh Độ chính xác (Accuracy)', fontsize=12)
plt.ylabel('(%)')
plt.ylim(0, 115)
for bar in bars:
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
             f'{bar.get_height():.1f}%', ha='center', fontweight='bold')
plt.grid(axis='y', alpha=0.3)

# Biểu đồ Feature Importance
plt.subplot(1, 2, 2)
sorted_idx = np.argsort(importances)
plt.barh(np.array(feat_names)[sorted_idx], importances[sorted_idx], color='teal')
plt.title('Tầm quan trọng các thông số (Feature Importance)', fontsize=12)
plt.xlabel('Mức độ đóng góp')

plt.tight_layout()
plt.savefig('Final_Report_Chart.png')
print(f">>> Đã lưu biểu đồ báo cáo: 'Final_Report_Chart.png'")
plt.show()