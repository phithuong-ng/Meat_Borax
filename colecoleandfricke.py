import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

# --- 1. CẤU HÌNH TÊN FILE VÀ TÊN HIỂN THỊ ---
# Bạn hãy đảm bảo các file này nằm cùng thư mục với file code
files = ['0%.csv', '0phay5lan5.csv', '1lan3.csv', '2%.csv', '5%.csv', '10%.csv']

# Mapping để đổi tên file loằng ngoằng thành tên ngắn gọn trên đồ thị
name_mapping = {
    '0%.csv': '0%',
    '0phay5lan5.csv': '0.5%',
    '1lan3.csv': '1%',
    '2%.csv': '2%',
    '5%.csv': '5%',
    '10%.csv': '10%'
}


# --- 2. HÀM ĐỌC DỮ LIỆU (Xử lý định dạng máy HIOKI) ---
def read_eis_file(filepath):
    try:
        # Tìm dòng bắt đầu bằng "No." để biết chỗ nào có dữ liệu
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        header_line_index = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('"No."'):
                header_line_index = i
                break

        # Đọc dữ liệu từ dòng header tìm được
        df = pd.read_csv(filepath, skiprows=header_line_index)

        # Làm sạch tên cột (bỏ dấu ngoặc kép, khoảng trắng)
        df.columns = [c.strip().replace('"', '') for c in df.columns]

        # Đổi tên cột về dạng chuẩn để dễ gọi
        col_map = {}
        for c in df.columns:
            if 'FREQ' in c:
                col_map[c] = 'f'
            elif 'Z' in c and 'ohm' in c:
                col_map[c] = 'Z_mag'
            elif 'PHASE' in c:
                col_map[c] = 'phase'

        df = df.rename(columns=col_map)

        # Chuyển đổi dữ liệu sang dạng số (float)
        for col in ['f', 'Z_mag', 'phase']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Bỏ các dòng lỗi/trống
        df = df.dropna(subset=['f', 'Z_mag', 'phase'])

        # TÍNH TOÁN Z' (Thực) và Z'' (Ảo)
        # Lưu ý: Phase của máy đo thường là độ -> đổi sang radian
        phase_rad = np.radians(df['phase'])
        df['Z_real'] = df['Z_mag'] * np.cos(phase_rad)
        df['Z_imag'] = df['Z_mag'] * np.sin(phase_rad)

        return df
    except Exception as e:
        print(f"Không đọc được file {filepath}: {e}")
        return None


# --- 3. ĐỊNH NGHĨA MÔ HÌNH TOÁN HỌC ---

# === MÔ HÌNH COLE-COLE ===
# Công thức: Z = R_inf + (R_0 - R_inf) / (1 + (j*w*tau)^alpha)
def Z_cole_cole(f, R_inf, R_0, tau, alpha):
    omega = 2 * np.pi * f
    j_omega_tau = (1j * omega * tau) ** alpha
    return R_inf + (R_0 - R_inf) / (1 + j_omega_tau)


# Hàm phụ trợ để tách phần thực/ảo cho curve_fit
def fit_cole_cole(f, R_inf, R_0, tau, alpha):
    Z = Z_cole_cole(f, R_inf, R_0, tau, alpha)
    return np.concatenate((Z.real, Z.imag))


# === MÔ HÌNH FRICKE ===
# Mạch: Re song song với (Ri nối tiếp CPE)
# Công thức: Z_CPE = 1/(Q*(jw)^a) -> Z_nhanh2 = Ri + Z_CPE
# Z_total = (Re * Z_nhanh2) / (Re + Z_nhanh2)
def Z_fricke(f, Re, Ri, Q, alpha):
    omega = 2 * np.pi * f
    j_omega_alpha = (1j * omega) ** alpha  # Lưu ý: (jw)^alpha khác với (jw*tau)^alpha

    Z_cpe = 1.0 / (Q * j_omega_alpha)
    Z_branch2 = Ri + Z_cpe

    # Tổng trở song song
    Z_total = (Re * Z_branch2) / (Re + Z_branch2)
    return Z_total


def fit_fricke(f, Re, Ri, Q, alpha):
    Z = Z_fricke(f, Re, Ri, Q, alpha)
    return np.concatenate((Z.real, Z.imag))


# --- 4. CHẠY CHÍNH (MAIN LOOP) ---

# Đọc dữ liệu vào dict
data_dict = {}
for f in files:
    df = read_eis_file(f)
    if df is not None:
        data_dict[f] = df

# Tạo khung vẽ (2 hàng, 3 cột)
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

print("Đang xử lý fitting...")

for i, f_name in enumerate(files):
    if f_name not in data_dict: continue

    # Lấy dữ liệu
    df = data_dict[f_name]
    ax = axes[i]
    display_name = name_mapping.get(f_name, f_name)  # Lấy tên hiển thị đẹp

    f_data = df['f'].values
    y_data = np.concatenate((df['Z_real'].values, df['Z_imag'].values))

    # ---------------- FITTING COLE-COLE ----------------
    # Dự đoán tham số ban đầu (Initial Guess)
    # R_inf ~ min(Z_real), R_0 ~ rất lớn, tau ~ nhỏ, alpha ~ 0.7
    p0_cole = [np.min(df['Z_real']), 1e8, 1e-4, 0.7]
    bounds_cole = ([0, 0, 0, 0], [np.inf, np.inf, 1, 1])  # alpha <= 1, tau <= 1

    try:
        popt_c, _ = curve_fit(fit_cole_cole, f_data, y_data, p0=p0_cole, bounds=bounds_cole, maxfev=10000)
        # Tạo đường mượt để vẽ
        f_smooth = np.logspace(np.log10(min(f_data)), np.log10(max(f_data)), 100)
        Z_fit_cole = Z_cole_cole(f_smooth, *popt_c)
        cole_success = True
    except:
        cole_success = False
        print(f"Fit Cole-Cole lỗi ở file {display_name}")

    # ---------------- FITTING FRICKE ----------------
    # Dự đoán: Re (ngoại bào/DC) ~ Rất lớn, Ri (nội bào) ~ min(Z_real)
    p0_fricke = [1e8, np.min(df['Z_real']), 1e-4, 0.7]
    bounds_fricke = ([0, 0, 0, 0], [np.inf, np.inf, np.inf, 1])

    try:
        popt_f, _ = curve_fit(fit_fricke, f_data, y_data, p0=p0_fricke, bounds=bounds_fricke, maxfev=10000)
        Z_fit_fricke = Z_fricke(f_smooth, *popt_f)
        fricke_success = True
    except:
        fricke_success = False
        print(f"Fit Fricke lỗi ở file {display_name}")

    # ---------------- VẼ ĐỒ THỊ ----------------
    # 1. Vẽ dữ liệu thực tế (chấm xanh mờ)
    ax.plot(df['Z_real'], -df['Z_imag'], 'bo', alpha=0.3, markersize=5, label='Data Thực tế')

    # 2. Vẽ đường Fit Cole-Cole (Nét liền Đỏ)
    if cole_success:
        ax.plot(Z_fit_cole.real, -Z_fit_cole.imag, 'r-', linewidth=2, label='Cole-Cole')

    # 3. Vẽ đường Fit Fricke (Nét đứt Xanh lá cây) - Vẽ đè lên để so sánh
    if fricke_success:
        ax.plot(Z_fit_fricke.real, -Z_fit_fricke.imag, 'g--', linewidth=2, dashes=(4, 3), label='Fricke')

    # Trang trí
    ax.set_title(f"Mẫu: {display_name}", fontsize=12, fontweight='bold')
    ax.set_xlabel("Z' (Ohm)")
    ax.set_ylabel("-Z'' (Ohm)")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.axis('equal')  # Quan trọng: để cung tròn không bị méo

    # Chỉ hiện chú thích ở hình đầu tiên cho đỡ rối
    if i == 0:
        ax.legend(loc='best')

# Tinh chỉnh khoảng cách giữa các hình cho thoáng (tránh chữ đè nhau)
plt.tight_layout(h_pad=3.0, w_pad=2.0)
plt.subplots_adjust(top=0.92, bottom=0.08)

# Lưu ảnh và hiển thị
plt.savefig('So_sanh_ColeCole_vs_Fricke.png', dpi=300)
print("Đã vẽ xong! Kiểm tra file 'So_sanh_ColeCole_vs_Fricke.png'")
plt.show()