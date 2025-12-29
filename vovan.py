import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Tên file đã được chuyển đổi sang CSV trong môi trường
file_path = 'ket_qua_trung_binh_200_mau 0.5m.xlsx - Sheet1.csv'
df = pd.read_csv(file_path)

# ----------------------------------------------------
# PHẦN 1: VẼ BODE PLOTS (Biên độ và Pha theo Tần số)
# ----------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

# Biểu đồ 1: Biên độ trở kháng Z vs Tần số f (Log-Log)
ax1.plot(df['FREQUENCY(Hz)'], df['Z_Average(ohm)'], color='b', label='Impedance |Z|')
ax1.set_ylabel('Impedance $|Z|$ (Ohm)')
ax1.set_title('Bode Plot: Impedance Amplitude vs Frequency')
ax1.grid(True, which="both", ls="-")
ax1.set_xscale('log')
ax1.set_yscale('log')

# Biểu đồ 2: Pha phi vs Tần số f (Semi-Log X)
ax2.plot(df['FREQUENCY(Hz)'], df['Phase_Average(deg)'], color='r', label='Phase $\\phi$')
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Phase $\\phi$ (deg)')
ax2.set_title('Bode Plot: Phase vs Frequency')
ax2.grid(True, which="both", ls="-")
ax2.set_xscale('log')

plt.tight_layout()
# plt.show() # Lệnh để hiển thị biểu đồ trong môi trường lập trình

# ----------------------------------------------------
# PHẦN 2: VẼ NYQUIST PLOT (Z' vs -Z'')
# ----------------------------------------------------

# 1. Chuyển Phase từ độ sang radian
df['Phase_Average_rad'] = np.radians(df['Phase_Average(deg)'])

# 2. Tính phần Thực (Real, Z') và phần Ảo (Imaginary, Z''): Z = Z' + jZ''
# Z' = |Z| * cos(phi)
df['R_Real_Z'] = df['Z_Average(ohm)'] * np.cos(df['Phase_Average_rad'])
# Z'' = |Z| * sin(phi)
df['X_Imaginary_Z'] = df['Z_Average(ohm)'] * np.sin(df['Phase_Average_rad'])

# 3. Vẽ Nyquist Plot: Real Impedance (Z') vs -Imaginary Impedance (-Z'')
plt.figure(figsize=(8, 8))
plt.plot(df['R_Real_Z'], -df['X_Imaginary_Z'], 'o-', markersize=2, label='Nyquist Plot')
plt.xlabel('Real Impedance, $Z^{\prime}$ (Ohm)')
plt.ylabel('$-$Imaginary Impedance, $-Z^{\prime\prime}$ (Ohm)')
plt.title('Nyquist Plot')
plt.axis('equal') # Đảm bảo tỷ lệ trục chính xác
plt.grid(True)
plt.legend()

# plt.show() # Lệnh để hiển thị biểu đồ trong môi trường lập trình