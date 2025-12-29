import plotly.graph_objects as go
import numpy as np

# --- 1. Giả lập dữ liệu (khớp với hình ảnh) ---
# Tạo dữ liệu trục X (Tần số từ 50k đến 100k)
x_data = np.linspace(50000, 100000, 50)

# Tạo dữ liệu trục Y cho 3 đường (Giá trị ước lượng từ hình)
y_yellow = np.full(50, 300)   # Đường vàng (giá trị ~300)
y_pink = np.full(50, 240)     # Đường hồng (giá trị ~240)
y_black = np.full(50, 0)      # Đường đen (giá trị ~0)

# --- 2. Tạo biểu đồ ---
fig = go.Figure()

# Thêm đường màu vàng (Yellow trace)
fig.add_trace(go.Scatter(
    x=x_data,
    y=y_yellow,
    mode='lines',  # Chế độ vẽ đường liền
    name='Series 1 (Vàng)',
    line=dict(color='#d4d785', width=3) # Màu vàng nhạt, nét đậm hơn chút
))

# Thêm đường màu hồng (Pink trace)
fig.add_trace(go.Scatter(
    x=x_data,
    y=y_pink,
    mode='lines',  # Chế độ vẽ đường liền
    name='Series 2 (Hồng)',
    line=dict(color='#e377c2', width=3) # Màu hồng
))

# Thêm đường màu đen (Black trace)
fig.add_trace(go.Scatter(
    x=x_data,
    y=y_black,
    mode='lines',  # Chế độ vẽ đường liền
    name='Series 3 (Đen)',
    line=dict(color='black', width=4) # Màu đen, nét đậm nhất
))

# --- 3. Cập nhật giao diện biểu đồ ---
fig.update_layout(
    title='Bode plot (|Z|)',
    xaxis=dict(
        title='Click to enter X axis title',
        tickvals=[50000, 60000, 70000, 80000, 90000, 100000],
        ticktext=['50k', '60k', '70k', '80k', '90k', '100k']
    ),
    yaxis=dict(
        title='Click to enter Y axis title',
        range=[-10, 320] # Đặt giới hạn trục Y để giống hình
    ),
    template="simple_white", # Sử dụng nền trắng đơn giản
    showlegend=False # Ẩn chú thích (legend) cho giống hình
)

# Hiển thị biểu đồ
fig.show()

# Nếu muốn lưu thành file HTML để mở sau này:
# fig.write_html("bode_plot_connected.html")
print("Đã tạo biểu đồ thành công! Hãy kiểm tra trình duyệt hoặc file HTML.")