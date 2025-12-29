import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- CẤU HÌNH ---
files = ['0%.csv', '0phay5lan5.csv', '1lan3.csv', '2%.csv', '5%.csv', '10%.csv']
name_mapping = {
    '0%.csv': '0%', '0phay5lan5.csv': '0.5%', '1lan3.csv': '1%',
    '2%.csv': '2%', '5%.csv': '5%', '10%.csv': '10%'
}


# --- HÀM ĐỌC DATA ---
def read_eis_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        header_idx = next(i for i, line in enumerate(lines) if line.strip().startswith('"No."'))
        df = pd.read_csv(filepath, skiprows=header_idx)
        df.columns = [c.strip().replace('"', '') for c in df.columns]
        col_map = {}
        for c in df.columns:
            if 'FREQ' in c:
                col_map[c] = 'f'
            elif 'Z' in c and 'ohm' in c:
                col_map[c] = 'Z_mag'
            elif 'PHASE' in c:
                col_map[c] = 'phase'
        df = df.rename(columns=col_map).dropna(subset=['f', 'Z_mag', 'phase'])
        for c in ['f', 'Z_mag', 'phase']: df[c] = pd.to_numeric(df[c], errors='coerce')
        phase_rad = np.radians(df['phase'])
        df['Z_real'] = df['Z_mag'] * np.cos(phase_rad)
        df['Z_imag'] = df['Z_mag'] * np.sin(phase_rad)
        return df
    except:
        return None


# --- MÔ HÌNH ---
def circuit_impedance(f, Rs, Rp, Q, alpha):
    omega = 2 * np.pi * f
    j_omega_alpha = (omega ** alpha) * (np.cos(np.pi * alpha / 2) + 1j * np.sin(np.pi * alpha / 2))
    return Rs + (Rp / (1 + Rp * Q * j_omega_alpha))


def func_fit(f, Rs, Rp, Q, alpha):
    Z = circuit_impedance(f, Rs, Rp, Q, alpha)
    return np.concatenate((Z.real, Z.imag))


# --- PLOTTING & FITTING ---
data_dict = {f: read_eis_file(f) for f in files if read_eis_file(f) is not None}
results = []

# Tăng kích thước figure để thoáng hơn (Chiều cao=12)
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, f_name in enumerate(files):
    if f_name not in data_dict: continue
    display_name = name_mapping.get(f_name, f_name)
    df = data_dict[f_name]
    ax = axes[i]

    f, Zr, Zi = df['f'].values, df['Z_real'].values, df['Z_imag'].values
    p0 = [np.min(Zr), np.max(Zr) * 2, 1e-4, 0.7]

    try:
        popt, _ = curve_fit(func_fit, f, np.concatenate((Zr, Zi)), p0=p0, maxfev=20000)
        Rs, Rp, Q, a = popt
        results.append({'Sample': display_name, 'Rs': Rs, 'Rp': Rp, 'Q': Q, 'alpha': a})

        f_smooth = np.logspace(np.log10(f.min()), np.log10(f.max()), 100)
        Z_fit = circuit_impedance(f_smooth, *popt)

        ax.plot(Zr, -Zi, 'bo', alpha=0.5, label='Data')
        ax.plot(Z_fit.real, -Z_fit.imag, 'r-', lw=2, label='Fit')
        ax.set_title(f"{display_name}\nRs={Rs:.1f}, a={a:.2f}")
        ax.set_xlabel("Z' (Ohm)");
        ax.set_ylabel("-Z'' (Ohm)")
        ax.grid(True);
        ax.axis('equal')
    except:
        pass

# --- CHỈNH SỬA KHOẢNG CÁCH (QUAN TRỌNG) ---
# h_pad=5.0 giúp dãn cách dòng trên và dòng dưới
plt.tight_layout(h_pad=5.0, w_pad=3.0)
plt.subplots_adjust(top=0.92, bottom=0.08)  # Chỉnh lề
plt.show()

print(pd.DataFrame(results))