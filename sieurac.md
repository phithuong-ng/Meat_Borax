# BÁO CÁO PHÂN TÍCH & SO SÁNH MÔ HÌNH TRỞ KHÁNG ĐIỆN HÓA (EIS)
**Đối tượng:** Giò lụa chứa Hàn the (Borax)  
**Phương pháp:** Đo trở kháng 2 điện cực kim (Two-needle electrodes)

---

## PHẦN 1: SO SÁNH LÝ THUYẾT 3 MÔ HÌNH

Bảng dưới đây so sánh bản chất vật lý và cấu trúc của 3 mô hình phổ biến nhất được áp dụng cho dữ liệu đo.

| Tiêu chí | **1. Mạch Randles (Biến đổi)** | **2. Mô hình Fricke** (Khuyên dùng) | **3. Mô hình Cole-Cole** |
| :--- | :--- | :--- | :--- |
| **Sơ đồ mạch** | $R_s + (R_p \parallel CPE)$ | $R_e \parallel (R_i + CPE)$ | *$R_\infty + ((R_0 - R_\infty) \parallel CPE)$* |
| **Công thức tổng trở** | $Z = R_s + \frac{R_p}{1 + R_p Q (j\omega)^\alpha}$ | $Z = \frac{R_e (R_i + Z_{CPE})}{R_e + R_i + Z_{CPE}}$ | $Z = R_\infty + \frac{R_0 - R_\infty}{1 + (j\omega\tau)^\alpha}$ |
| **Tham số chính**<br>*(Đại diện độ dẫn điện)* | **$R_s$** (Điện trở dung dịch)<br>Độ dẫn điện khối của mẫu nằm giữa 2 kim. | **$R_i$** (Điện trở nội bào)<br>Đường dẫn dòng điện đi xuyên qua cấu trúc thịt/tế bào. | **$R_\infty$** (Điện trở tần số cao)<br>Giới hạn điện trở khi $f \to \infty$. |
| **Tham số phụ**<br>*(Đại diện dòng rò/cản trở)* | **$R_p$** (Điện trở phân cực)<br>Rất lớn do kim thép không gỉ trơ hóa học. | **$R_e$** (Điện trở ngoại bào)<br>Rất lớn do dòng một chiều (DC) bị chặn bởi lớp điện kép. | **$R_0$** (Điện trở tần số thấp)<br>Giới hạn điện trở tại DC ($f \to 0$). |
| **Ý nghĩa sinh học** | Thấp. Thường dùng cho ăn mòn kim loại. | **Cao.** Mô tả chính xác cấu trúc mô sinh học (thịt, tế bào). | Trung bình. Mạnh về toán học vật liệu điện môi. |

---

## PHẦN 2: KẾT QUẢ THỰC NGHIỆM TỪ BỘ DỮ LIỆU

So sánh giá trị của "Tham số chính" (Sensitive Parameter) thu được từ việc khớp (fitting) dữ liệu thực tế vào 3 mô hình.

| Mẫu (Nồng độ) | **Randles ($R_s$)**<br>($\Omega$) | **Fricke ($R_i$)**<br>($\Omega$) | **Cole-Cole ($R_\infty$)**<br>($\Omega$) | **Nhận xét & Giải thích hiện tượng** |
| :--- | :--- | :--- | :--- | :--- |
| **0% (Chuẩn)** | **50.1** | **50.1** | **51.0** | Giá trị nền (Baseline) của giò sạch. |
| **0.5%** | **70.9** | **70.9** | **72.5** | **BẤT THƯỜNG (Anomaly):** Tăng cao hơn mẫu chuẩn. Do hànthe nồng độ thấp tạo liên kết ngang làm "đanh" cấu trúc thịt, cản trở dòng điện. |
| **1%** | **42.6** | **42.6** | **44.3** | **Điểm chuyển tiếp:** Tác động dẫn điện của ion bắt đầu thắng tác động cản trở của cấu trúc. |
| **2%** | **21.0** | **21.0** | **21.9** | **DƯƠNG TÍNH RÕ:** Điện trở giảm >50% so với mẫu chuẩn. Tín hiệu dẫn điện ion cực mạnh. |
| **5%** | **14.4** | **14.4** | **15.6** | Điện trở giảm sâu, bắt đầu bão hòa. |
| **10%** | **12.3** | **12.3** | **14.0** | Bão hòa hoàn toàn (Dẫn điện như dung dịch muối đặc). |

> **Nhận định:** Cả 3 mô hình đều cho ra giá trị tham số chính tương đương nhau ($R_s \approx R_i \approx R_\infty$). Điều này khẳng định độ tin cậy của dữ liệu đo là rất cao.

---

## PHẦN 3: KẾT LUẬN & QUY LUẬT NHẬN DIỆN

Dựa trên mô hình **Fricke** (tham số $R_i$), thiết lập bảng quy chuẩn để phát hiện hàn the bằng thiết bị đo nhanh.

| Trạng thái phát hiện | Ngưỡng giá trị ($R_i$) | Kết luận hiển thị | Cơ chế vật lý |
| :--- | :--- | :--- | :--- |
| **DƯƠNG TÍNH MẠNH** | **$R_i < 30 \Omega$** | 🔴 **CẢNH BÁO ĐỎ**<br>(Nhiễm > 2%) | Mật độ ion $Na^+$ và $Borate^-$ rất cao, tạo thành kênh dẫn điện ưu thế, lấn át hoàn toàn trở kháng của mô thịt. |
| **NGHI NGỜ** | **$30 \le R_i < 45 \Omega$** | 🟡 **CẢNH BÁO VÀNG**<br>(Nhiễm ~ 1%) | Mật độ ion bắt đầu tăng cao, làm giảm điện trở xuống dưới mức nền nhưng chưa giảm sâu. |
| **KHÔNG RÕ RÀNG** | **$R_i \ge 45 \Omega$** | 🟢 **AN TOÀN / KHÔNG RÕ**<br>(< 0.5% hoặc 0%) | **Vùng mù:** Tại nồng độ thấp, hàn the làm cứng giò (tăng R) thay vì dẫn điện (giảm R). Không thể phân biệt rạch ròi với mẫu sạch bằng điện trở đơn thuần. |

---

## PHẦN 4: KHUYẾN NGHỊ CUỐI CÙNG (FINAL VERDICT)

**Nên chọn mô hình nào?**
👉 **MÔ HÌNH FRICKE ($R_e \parallel (R_i + CPE)$)**

**Lý do:**
1.  **Tính đúng đắn về Sinh học (Bio-fidelity):** Fricke là mô hình tiêu chuẩn cho mô sinh học. Tham số $R_i$ (Điện trở nội bào) mô tả trực quan dòng điện đi xuyên qua ma trận protein của giò.
2.  **Giải thích được hiện tượng lạ:** Mô hình này giúp biện luận hợp lý cho trường hợp mẫu 0.5% (tăng trở kháng) thông qua cơ chế thay đổi cấu trúc nội bào/ngoại bào, thuyết phục hơn mô hình mạch điện đơn thuần.
3.  **Tương thích phần cứng:** Tham số $R_e$ rất lớn trong mô hình phản ánh chính xác hiện tượng "chặn dòng DC" của cặp kim điện cực thép không gỉ trơ.