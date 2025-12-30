import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score

# --- 1. CẤU HÌNH ---
FILE_PATH = 'parameters (1).csv'

# --- 2. TẢI DỮ LIỆU ---
print(">>> Đang chạy kiểm định Full Report (KNN Model)...")
try:
    df = pd.read_csv(FILE_PATH)
except:
    print(f"Lỗi: Không tìm thấy file '{FILE_PATH}'.")
    exit()

def get_label(filename):
    name = str(filename).lower()
    if 'zin' in name: return 0
    if '0%' in name and '10%' not in name: return 0
    return 1 # Bẩn

df['Label'] = df['FileName'].apply(get_label)

# --- 3. KIỂM ĐỊNH LOOCV VỚI KNN ---
features = ['Ri', 'Re', 'p_CPE1', 'T_CPE1']
X = df[features]
y = df['Label']

# Scale dữ liệu (Bắt buộc đối với KNN để tính khoảng cách chuẩn)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Cấu hình KNN
# n_neighbors=3: Chọn 3 láng giềng gần nhất (Số lẻ để tránh hòa phiếu)
knn = KNeighborsClassifier(n_neighbors=3)
loo = LeaveOneOut()

# Dự đoán mù (Blind Test)
y_pred = cross_val_predict(knn, X_scaled, y, cv=loo)

# --- 4. TÍNH TOÁN CHỈ SỐ ---
cm = confusion_matrix(y, y_pred)
tn, fp, fn, tp = cm.ravel()

accuracy = accuracy_score(y, y_pred) * 100
sensitivity = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0
specificity = (tn / (tn + fp) * 100) if (tn + fp) > 0 else 0

# --- 5. XUẤT BÁO CÁO (ĐÚNG FORMAT YÊU CẦU) ---
print("\n" + "="*60)
print("          BÁO CÁO KIỂM ĐỊNH HỆ THỐNG KNN (FINAL REPORT)")
print("="*60)
print(f"Mô hình: K-Nearest Neighbors (k=3)")
print(f"Tổng mẫu: {len(df)}")
print(f"- Sạch (Negative): {tn+fp} mẫu")
print(f"- Bẩn  (Positive): {tp+fn} mẫu")
print("-" * 60)

print(f"1. ĐỘ CHÍNH XÁC (Accuracy):      {accuracy:.2f}%")
print(f"   -> Nhận xét: {'Xuất sắc' if accuracy > 95 else 'Tốt' if accuracy > 85 else 'Cần cải thiện'}")
print("-" * 60)

print(f"2. ĐỘ NHẠY (Sensitivity):        {sensitivity:.2f}%")
print(f"   -> Ý nghĩa: Phát hiện được {tp}/{tp+fn} mẫu có hàn the.")
if sensitivity == 100:
    print("   -> Đánh giá: TUYỆT ĐỐI. Không bỏ sót bất kỳ mẫu thực phẩm bẩn nào.")
print("-" * 60)

print(f"3. ĐỘ ĐẶC HIỆU (Specificity):    {specificity:.2f}%")
print(f"   -> Ý nghĩa: Nhận diện đúng {tn}/{tn+fp} mẫu sạch.")
print("-" * 60)

print("MA TRẬN NHẦM LẪN (CONFUSION MATRIX):")
print(f"              | Máy bảo SẠCH (0)| Máy bảo BẨN (1)|")
print(f"Thực tế SẠCH  |       {tn:<2}        |       {fp:<2}       |")
print(f"Thực tế BẨN   |       {fn:<2}        |       {tp:<2}       |")
print("="*60)

# Phân tích lỗi sai
df['Predicted'] = y_pred
wrong = df[df['Label'] != df['Predicted']]

print("\n[PHÂN TÍCH CÁC TRƯỜNG HỢP SAI]:")
if len(wrong) > 0:
    print(f"{'Tên File':<20} | {'Thực tế':<10} | {'Máy đoán':<10} | {'Nguyên nhân dự đoán (KNN)'}")
    print("-" * 75)
    for idx, row in wrong.iterrows():
        thuc = "SẠCH" if row['Label']==0 else "BẨN"
        doan = "SẠCH" if row['Predicted']==0 else "BẨN"
        # Giải thích theo cơ chế KNN
        ly_do = "Có quá nhiều 'hàng xóm' là mẫu Sạch vây quanh" if row['Label']==1 else "Bị vây quanh bởi các mẫu Bẩn (Nhiễu cục bộ)"
        print(f"{row['FileName']:<20} | {thuc:<10} | {doan:<10} | {ly_do}")
    print("-" * 75)
    print("Nhận xét lỗi: KNN dựa trên khoảng cách cục bộ, các mẫu sai thường do")
    print("nằm xen kẽ trong vùng phân bố của loại kia (Outliers).")
else:
    print(">>> TUYỆT VỜI! Không có mẫu nào bị nhận diện sai.")
    print(">>> Mô hình KNN đạt độ tin cậy tuyệt đối trên tập dữ liệu này.")