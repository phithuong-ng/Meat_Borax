import joblib
import numpy as np
import pandas as pd

# --- 1. NẠP "TRÍ TUỆ" ĐÃ LƯU (LOAD MODEL) ---
print(">>> Đang khởi động não bộ AI...")
try:
    model = joblib.load('svm_model_final.pkl')
    scaler = joblib.load('scaler_final.pkl')
except FileNotFoundError:
    print("LỖI: Chưa có file model (.pkl). Hãy chạy file train trước!")
    exit()

# --- 2. GIẢ LẬP DỮ LIỆU ĐO TỪ CẢM BIẾN ---
# Giả sử bạn vừa đo được 3 mẫu giò ngoài chợ với các thông số sau:
# (Đây là input thực tế, không có tên file, chỉ có số)
samples_from_market = [
    # Mẫu A: Re rất lớn (đặc trưng giò sạch)
    {'Ri': 75.5, 'Re': 18000.0, 'p_CPE1': 0.73, 'T_CPE1': 4.5e-5},

    # Mẫu B: Re nhỏ tí (đặc trưng có hàn the)
    {'Ri': 45.2, 'Re': 500.0, 'p_CPE1': 0.65, 'T_CPE1': 6.0e-5},

    # Mẫu C: Lấp lửng (để xem AI phán thế nào)
    {'Ri': 60.0, 'Re': 8000.0, 'p_CPE1': 0.70, 'T_CPE1': 5.0e-5}
]

print("\n>>> BẮT ĐẦU KIỂM TRA MẪU LẠ (BLIND TEST)...")
print("-" * 50)
print(f"{'MẪU':<10} | {'Ri':<8} | {'Re':<10} | {'KẾT LUẬN CỦA AI'}")
print("-" * 50)

for i, sample in enumerate(samples_from_market):
    # 1. Chuyển đổi dữ liệu thành dạng bảng
    # Lưu ý: Thứ tự cột phải Y HỆT lúc train: ['Ri', 'Re', 'p_CPE1', 'T_CPE1']
    features = pd.DataFrame([sample])
    features = features[['Ri', 'Re', 'p_CPE1', 'T_CPE1']]  # Sắp xếp lại cho chắc ăn

    # 2. Chuẩn hóa dữ liệu (Bước cực quan trọng)
    # Phải dùng đúng cái scaler đã lưu lúc train để quy đổi đơn vị
    features_scaled = scaler.transform(features)

    # 3. AI phán đoán (Predict)
    prediction = model.predict(features_scaled)[0]  # Ra 0 hoặc 1
    probability = model.predict_proba(features_scaled)[0]  # Ra độ tin cậy %

    # 4. Hiển thị kết quả cho người dùng
    if prediction == 0:
        ket_luan = "✅ SẠCH (An toàn)"
        do_tin_cay = probability[0] * 100
    else:
        ket_luan = "💀 CÓ HÀN THE!"
        do_tin_cay = probability[1] * 100

    print(f"Mẫu {i + 1:<6} | {sample['Ri']:<8} | {sample['Re']:<10} | {ket_luan} ({do_tin_cay:.1f}%)")

print("-" * 50)