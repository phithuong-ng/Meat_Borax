import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

# Khai báo tên file và nhãn (Đã thêm 0.5%)
file_names = {
    "0%": "0%.csv",
    "0.5%": "0phay5lan5.csv",  # THÊM MỚI
    "1%": "1lan3.csv",
    "2%": "2%.csv",
    "5%": "5%.csv",
    "10%": "10%.csv",
}


def load_and_preprocess_data(file_path, label):
    """
    Hàm sử dụng thư viện 'csv' gốc để đọc từng dòng và xử lý thủ công.
    Đã bổ sung tính toán Z_Real và Z_Imag_Neg cho Nyquist Plot.
    """
    data = []

    try:
        with open(file_path, 'r') as file:
            # Bỏ qua 15 dòng đầu tiên (header và metadata)
            for _ in range(15):
                next(file)

            # Sử dụng dialect='excel' để xử lý tốt các file CSV có dấu nháy kép
            reader = csv.reader(file, dialect='excel')

            for row in reader:
                # Dữ liệu cần có ít nhất 4 cột
                if len(row) >= 4:
                    try:
                        # Lấy các trường: 1 (Freq), 2 (Z), 3 (Phase)
                        freq = float(row[1].strip().replace('"', ''))
                        z_mag = float(row[2].strip().replace('"', ''))
                        phase = float(row[3].strip().replace('"', ''))
                        data.append([freq, z_mag, phase])
                    except ValueError:
                        # Bỏ qua các dòng không phải dữ liệu số hoặc dòng header cuối file
                        continue
    except FileNotFoundError:
        raise Exception(f"Lỗi: Không tìm thấy file {file_path}. Vui lòng đảm bảo file đã được tải lên.")

    if not data:
        raise Exception(f"Không tìm thấy dữ liệu hợp lệ trong file {file_path}.")

    df = pd.DataFrame(data, columns=['Frequency', 'Z_Magnitude', 'Phase_Deg'])

    # ----------------------------------------------------------------------
    # TÍNH TOÁN CÁC THÀNH PHẦN TRỞ KHÁNG
    # ----------------------------------------------------------------------
    df['Phase_Rad'] = np.deg2rad(df['Phase_Deg'])
    df['Z_Real'] = df['Z_Magnitude'] * np.cos(df['Phase_Rad'])
    df['Z_Imag_Neg'] = - (df['Z_Magnitude'] * np.sin(df['Phase_Rad']))
    df['Concentration'] = label

    return df


# Tải và kết hợp dữ liệu
all_data = []
for label, file_name in file_names.items():
    try:
        all_data.append(load_and_preprocess_data(file_name, label))
    except Exception as e:
        print(e)
        # Không thoát, tiếp tục xử lý các file khác nếu có lỗi
        continue

if not all_data:
    raise Exception("Không thể tải bất kỳ dữ liệu nào. Vui lòng kiểm tra tên file.")

df_combined = pd.concat(all_data, ignore_index=True)


# Sắp xếp thứ tự nồng độ cho chú giải
concentration_order = ['0%', '0.5%', '1%', '2%', '5%', '10%']
sorted_labels = [label for label in concentration_order if label in df_combined['Concentration'].unique()]

# ----------------------------------------------------------------------
## 1. Biểu đồ Bode: Biên độ trở kháng theo Tần số

plt.figure(figsize=(10, 6))
for label in sorted_labels:
    df_plot = df_combined[df_combined['Concentration'] == label]
    plt.loglog(df_plot['Frequency'], df_plot['Z_Magnitude'], 'o-', markersize=4, label=f'{label}')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Impedance Magnitude ($|Z|$) ($\Omega$)')
plt.title('Bode Plot: Impedance Magnitude vs Frequency')
plt.legend(title='Concentration', loc='best')
plt.grid(True, which="both", ls="--")
plt.show()

# ----------------------------------------------------------------------
## 2. Biểu đồ Bode: Pha theo Tần số

plt.figure(figsize=(10, 6))
for label in sorted_labels:
    df_plot = df_combined[df_combined['Concentration'] == label]
    plt.semilogx(df_plot['Frequency'], df_plot['Phase_Deg'], 'o-', markersize=4, label=f'{label}')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase ($\phi$) (degrees)')
plt.title('Bode Plot: Phase vs Frequency')
plt.legend(title='Concentration', loc='best')
plt.grid(True, which="both", ls="--")
plt.show()

# ----------------------------------------------------------------------
## 3. Biểu đồ Nyquist

plt.figure(figsize=(8, 8))
for label in sorted_labels:
    df_plot = df_combined[df_combined['Concentration'] == label]
    plt.plot(df_plot['Z_Real'], df_plot['Z_Imag_Neg'], 'o-', markersize=4, label=f'{label}')

plt.xlabel('Real Impedance ($Z_{\\text{re}}$) ($\Omega$)')
plt.ylabel('Negative Imaginary Impedance ($-Z_{\\text{im}}$) ($\Omega$)')
plt.title('Nyquist Plot')
plt.legend(title='Concentration', loc='best')
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True, ls="--")
plt.show()