import pandas as pd
import numpy as np
import random
import os

# Cấu hình
SOURCE_FILE = 'zinnam.csv'  # File gốc
NUM_COPIES = 1  # Số lượng file cần tạo
OUTPUT_PREFIX = 'zin12'


def generate_smooth_data():
    # Đọc dữ liệu gốc
    with open(SOURCE_FILE, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()

    # Tìm header
    header_idx = 0
    for i, line in enumerate(all_lines):
        if line.strip().startswith('"No."'):
            header_idx = i
            break
    header_lines = all_lines[:header_idx + 1]

    # Đọc DataFrame
    df = pd.read_csv(SOURCE_FILE, skiprows=header_idx)
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

    freqs = df[col_map['f']].values
    zs = df[col_map['z']].values
    ps = df[col_map['p']].values

    print(f">>> Bắt đầu tạo {NUM_COPIES} file giả lập (Trend mượt)...")

    for i in range(1, NUM_COPIES + 1):
        # 1. Thiết lập mức lệch cho toàn bộ đường cong
        # Vùng thấp < 400kHz: Lệch 3-5%
        dev_low = random.uniform(0.03, 0.05)
        # Vùng cao > 600kHz: Lệch 3-10%
        dev_high = random.uniform(0.03, 0.10)
        # Hướng lệch: Tăng toàn bộ hoặc Giảm toàn bộ
        sign_z = random.choice([-1, 1])

        # 2. Tạo mảng hệ số tỷ lệ (Scaling Factors) nối mượt từ thấp lên cao
        scaling_factors = []
        for f in freqs:
            if f < 400000:
                factor = dev_low
            elif f > 600000:
                factor = dev_high
            else:
                # Nội suy tuyến tính ở giữa để không bị gãy khúc
                ratio = (f - 400000) / 200000
                factor = dev_low + ratio * (dev_high - dev_low)
            scaling_factors.append(factor)

        scaling_factors = np.array(scaling_factors)

        # 3. Áp dụng vào Z (Nhân cả đường cong với hệ số)
        # Thêm tí nhiễu cực nhỏ (0.05%) cho tự nhiên
        micro_noise = np.random.normal(0, 0.0005, size=len(zs))
        z_new = zs * (1 + sign_z * scaling_factors + micro_noise)

        # 4. Áp dụng vào Phase (Lệch đều ~5%)
        dev_p = random.uniform(0.04, 0.06)
        sign_p = random.choice([-1, 1])
        micro_noise_p = np.random.normal(0, 0.001, size=len(ps))
        p_new = ps * (1 + sign_p * dev_p + micro_noise_p)

        # Lưu file
        out_name = f"{OUTPUT_PREFIX}{i}.csv"
        with open(out_name, 'w', encoding='utf-8') as f_out:
            f_out.writelines(header_lines)
            for j in range(len(freqs)):
                line = (f"\"{int(df.iloc[j]['No.'])}\","
                        f"\"{df.iloc[j][col_map['f']]}\","
                        f"\"{z_new[j]:.5E}\","
                        f"\"{p_new[j]:.4f}\"\n")
                f_out.write(line)
        print(f"   -> Đã tạo: {out_name}")


generate_smooth_data()