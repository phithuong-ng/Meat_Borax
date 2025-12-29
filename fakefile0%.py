import pandas as pd
import numpy as np
import random
import os

# --- CẤU HÌNH ---
SEARCH_KEYWORD = '0%'  # Tìm file có tên chứa từ này (Ví dụ: 0phay5.csv, 0%.csv)
EXCLUDE_KEYWORD = '10%'  # Nhưng KHÔNG ĐƯỢC chứa từ này (để tránh nhầm với mẫu 10%)
NUM_COPIES = 3  # Số lượng file cần tạo
OUTPUT_PREFIX = '0%_'  # Tên file đầu ra


def find_source_file():
    """Hàm tự động tìm file gốc trong thư mục"""
    files = [f for f in os.listdir('.') if f.endswith('.csv')]
    for f in files:
        # Tìm file chứa '0%' nhưng không chứa '10%' và không phải file fake cũ
        if SEARCH_KEYWORD in f and EXCLUDE_KEYWORD not in f and 'fake' not in f and 'aug' not in f:
            return f
    return None


def generate_fake_0percent():
    # 1. Tìm file gốc
    source_file = find_source_file()
    if not source_file:
        print(
            f"LỖI: Không tìm thấy file gốc nào chứa '{SEARCH_KEYWORD}' (và không chứa '{EXCLUDE_KEYWORD}') trong thư mục hiện tại.")
        print("Hãy chắc chắn bạn để file '0%.csv' hoặc '0phay5.csv' cùng chỗ với code này.")
        return

    print(f">>> Đã tìm thấy file gốc: '{source_file}'")
    print(f">>> Bắt đầu sinh {NUM_COPIES} file giả lập mượt (Smooth Trend)...")

    # 2. Đọc header giữ nguyên định dạng máy HIOKI
    with open(source_file, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()

    header_idx = 0
    for i, line in enumerate(all_lines):
        if line.strip().startswith('"No."'):
            header_idx = i
            break
    header_lines = all_lines[:header_idx + 1]

    # 3. Đọc dữ liệu số
    try:
        df = pd.read_csv(source_file, skiprows=header_idx)
    except Exception as e:
        print(f"Lỗi đọc file CSV: {e}")
        return

    # Clean tên cột
    df.columns = [c.strip().replace('"', '') for c in df.columns]
    col_map = {}
    for c in df.columns:
        if 'FREQ' in c:
            col_map['f'] = c
        elif 'Z' in c and 'ohm' in c:
            col_map['z'] = c
        elif 'PHASE' in c:
            col_map['p'] = c

    # Lấy mảng dữ liệu
    freqs = df[col_map['f']].values.astype(float)
    zs = df[col_map['z']].values.astype(float)
    ps = df[col_map['p']].values.astype(float)

    # 4. Vòng lặp sinh file fake
    for i in range(1, NUM_COPIES + 1):
        # --- THUẬT TOÁN BIẾN THIÊN MƯỢT (SMOOTH SHIFT) ---

        # Z Shift: Lệch 3-8% (Nhẹ nhàng hơn Zin một chút vì mẫu 0% thường ổn định)
        dev_low = random.uniform(0.03, 0.05)  # Vùng tần số thấp
        dev_high = random.uniform(0.05, 0.08)  # Vùng tần số cao
        sign_z = random.choice([-1, 1])  # Tăng hoặc Giảm toàn bộ đường cong

        # Tạo đường cong hệ số lệch (Scaling Factors)
        scaling_factors = []
        for f in freqs:
            if f < 400000:
                factor = dev_low
            elif f > 600000:
                factor = dev_high
            else:
                # Nội suy tuyến tính ở giữa (400k-600k)
                ratio = (f - 400000) / 200000
                factor = dev_low + ratio * (dev_high - dev_low)
            scaling_factors.append(factor)
        scaling_factors = np.array(scaling_factors)

        # Áp dụng Z mới
        micro_noise = np.random.normal(0, 0.001, size=len(zs))  # Nhiễu vi mô 0.1%
        z_new = zs * (1 + sign_z * scaling_factors + micro_noise)

        # Phase Shift: Lệch ~5%
        dev_p = random.uniform(0.03, 0.06)
        sign_p = random.choice([-1, 1])
        micro_noise_p = np.random.normal(0, 0.002, size=len(ps))
        p_new = ps * (1 + sign_p * dev_p + micro_noise_p)

        # 5. Ghi file mới
        out_name = f"{OUTPUT_PREFIX}{i}.csv"
        with open(out_name, 'w', encoding='utf-8') as f_out:
            f_out.writelines(header_lines)
            for j in range(len(freqs)):
                # Format đúng chuẩn máy đo: "1","FREQ","Z","PHASE"
                line = (f"\"{int(df.iloc[j]['No.'])}\","
                        f"\"{df.iloc[j][col_map['f']]}\","
                        f"\"{z_new[j]:.5E}\","
                        f"\"{p_new[j]:.4f}\"\n")
                f_out.write(line)
        print(f"   -> Đã tạo: {out_name}")

    print("\n>>> HOÀN TẤT! Bạn đã có thêm 3 file dữ liệu giả lập cho mẫu 0%.")
    print(">>> Hãy chạy lại tool 'Cập nhật tham số' để thêm chúng vào hệ thống.")


# Chạy hàm
generate_fake_0percent()