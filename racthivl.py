import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Cấu hình và chuẩn bị dữ liệu ---

# Danh sách các file và nhãn (lần đo) tương ứng
file_info = [
    {'name': '0phay5lan2.csv', 'label': 1},
    {'name': '0phay5lan3.csv', 'label': 2},
    {'name': '0phay5lan4.csv', 'label': 3},
    {'name': '0phay5lan5.csv', 'label': 4}
]

# Tên cột
FREQ_COL = 'FREQUENCY(Hz)'
Z_MAG_COL = 'Z[ohm]'
PHASE_COL = 'PHASE[deg]'
REAL_Z_COL = 'Z_Real[ohm]'
IMAG_Z_COL = 'Z_Imag[ohm]'
RUN_COL = 'Run'

all_data = []

print("Đang xử lý dữ liệu...")

for info in file_info:
    try:
        # Tải file CSV. Dựa vào cấu trúc file, chúng ta bỏ qua 16 dòng đầu tiên (metadata)
        df = pd.read_csv(info['name'], skiprows=16)

        # Lấy Biên độ |Z| và Pha (độ)
        Z_mag = df[Z_MAG_COL]
        Phase_rad = np.deg2rad(df[PHASE_COL]) # Chuyển đổi Pha từ độ sang radian

        # Tính toán phần Thực (Real) và phần Ảo (Imaginary) của Trở kháng Z
        # Re(Z) = |Z| * cos(Phi)
        # Im(Z) = |Z| * sin(Phi)
        df[REAL_Z_COL] = Z_mag * np.cos(Phase_rad)
        df[IMAG_Z_COL] = Z_mag * np.sin(Phase_rad)

        # Thêm cột nhãn để phân biệt lần đo
        df[RUN_COL] = info['label']

        all_data.append(df)
    except Exception as e:
        print(f"Lỗi khi xử lý file {info['name']}: {e}")
        exit()

# Gộp tất cả các DataFrame thành một
df_combined = pd.concat(all_data, ignore_index=True)
print("Xử lý dữ liệu hoàn tất.")

# --- 2. Vẽ Biểu đồ 1: Bode Plot (Biên độ trở kháng theo Tần số) ---
bode_mag_plot_filename = 'bode_plot_magnitude.png'
plt.figure(figsize=(10, 6))
for run in df_combined[RUN_COL].unique():
    subset = df_combined[df_combined[RUN_COL] == run]
    plt.plot(subset[FREQ_COL], subset[Z_MAG_COL], label=f'Lần {run}')

plt.xscale('log')
plt.xlabel('Tần số ($Hz$)')
plt.ylabel('Biên độ trở kháng $|Z|$ ($\Omega$)')
plt.title('Biểu đồ Bode: Biên độ trở kháng theo Tần số')
plt.legend(title='Lần đo')
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.savefig(bode_mag_plot_filename)
plt.close()
print(f"Đã lưu biểu đồ Bode Biên độ: {bode_mag_plot_filename}")


# --- 3. Vẽ Biểu đồ 2: Bode Plot (Pha theo Tần số) ---
bode_phase_plot_filename = 'bode_plot_phase.png'
plt.figure(figsize=(10, 6))
for run in df_combined[RUN_COL].unique():
    subset = df_combined[df_combined[RUN_COL] == run]
    plt.plot(subset[FREQ_COL], subset[PHASE_COL], label=f'Lần {run}')

plt.xscale('log')
plt.xlabel('Tần số ($Hz$)')
plt.ylabel('Pha $\\Phi$ (độ)')
plt.title('Biểu đồ Bode: Pha theo Tần số')
plt.legend(title='Lần đo')
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.savefig(bode_phase_plot_filename)
plt.close()
print(f"Đã lưu biểu đồ Bode Pha: {bode_phase_plot_filename}")


# --- 4. Vẽ Biểu đồ 3: Nyquist Plot ---
nyquist_plot_filename = 'nyquist_plot.png'
plt.figure(figsize=(8, 8))
for run in df_combined[RUN_COL].unique():
    subset = df_combined[df_combined[RUN_COL] == run]
    # Nyquist plot: Im(Z) vs Re(Z)
    plt.plot(subset[REAL_Z_COL], subset[IMAG_Z_COL], 'o-', markersize=2, label=f'Lần {run}')

plt.xlabel('Thành phần thực của trở kháng $\\mathrm{Re}(Z)$ ($\Omega$)')
plt.ylabel('Thành phần ảo của trở kháng $\\mathrm{Im}(Z)$ ($\Omega$)')
plt.title('Biểu đồ Nyquist')
plt.legend(title='Lần đo')
plt.axis('equal') # Giữ tỉ lệ trục x và y bằng nhau
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.savefig(nyquist_plot_filename)
plt.close()
print(f"Đã lưu biểu đồ Nyquist: {nyquist_plot_filename}")

print("\nHoàn thành. 3 biểu đồ đã được lưu dưới dạng file PNG.")