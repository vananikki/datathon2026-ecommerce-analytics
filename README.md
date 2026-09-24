# 🏆 DATATHON 2026 — Daily Revenue Forecasting & E-Commerce Business Intelligence

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-orange.svg?logo=jupyter)](https://jupyter.org/)
[![Models](https://img.shields.io/badge/Ensemble-LightGBM%20%7C%20CatBoost%20%7C%20XGBoost-green.svg)](https://lightgbm.readthedocs.io/)
[![Interpretability](https://img.shields.io/badge/XAI-SHAP%20Values-purple.svg)](https://shap.readthedocs.io/)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)](#)

---

## 📌 Executive Summary

Dự án phát triển giải pháp dự báo chuỗi thời gian (Time-Series Forecasting) và phân tích chiến lược toàn diện cho doanh nghiệp Thương mại Điện tử Thời trang trong khuôn khổ cuộc thi **Datathon 2026**.

Hệ thống kết hợp:
1. **Pipeline Machine Learning Ensemble:** Xử lý ngoại lai, trích xuất đặc trưng chu kỳ/sự kiện TMĐT và phối hợp bộ ba thuật toán Gradient Boosting (**LightGBM + CatBoost + XGBoost**) để dự báo Doanh thu (`Revenue`) và Giá vốn hàng bán (`COGS`) hàng ngày từ `2023-01-01` đến `2024-07-01`.
2. **Business Intelligence & KPI Framework:** Bóc tách dữ liệu 6 phòng ban, nhận diện điểm nghẽn tồn kho (Deadstock) và độ nhạy khuyến mãi (Promo Cannibalization).

---

## 📂 Cấu Trúc Repository (Project Structure)

Cấu trúc tệp tin và thư mục thực tế của dự án:

```text
├── Cleaned_data/                              # Bộ dữ liệu đã qua tiền xử lý và làm sạch
├── Forecast Reveue Model/                     # Chứa các notebook huấn luyện mô hình dự báo & submission
│   ├── cross_validation.ipynb                 # Kiểm định chéo TimeSeriesSplit (5 Folds)
│   ├── Forecast_Revenue_Model.ipynb           # Huấn luyện LightGBM + CatBoost + XGBoost & xuất submission.csv
│   └── SHAP_va_feature_importance.ipynb       # Phân tích độ quan trọng đặc trưng (Gain & SHAP Summary)
├── Raw_data/                                  # Dữ liệu gốc ban đầu (sales.csv, ...)
├── Styles/                                    # Cấu hình giao diện và CSS hiển thị
│   └── Styles/
├── 1_Questions.ipynb                          # Notebook xử lý bài toán nghiệp vụ, trả lời câu hỏi và EDA
├── Visualization.ipynb                        # Notebook trực quan hóa chuyên sâu các chỉ số kinh doanh
├── [DATATHON26'] KPI Framework by Department.pdf # Tài liệu chẩn đoán hệ thống đa phòng ban
├── submission.csv                             # File nộp bài dự báo (Date, Revenue, COGS)
└── README.md                                  # Hướng dẫn chi tiết và tổng quan dự án
```

---

## 🤖 Kiến Trúc Mô Hình Dự Báo (Forecast Revenue Model)

### 1. Xử lý Dữ liệu & Biến đổi Mục tiêu
* **Outlier Capping (Bách phân vị 99):** Giảm thiểu tác động của những ngày doanh thu biến động bất thường làm lệch độ dốc mô hình (`Revenue_Smoothed = np.where(Revenue > q99, q99, Revenue)`).
* **Log-Transform Target:** Áp dụng biến đổi $y = \ln(1 + \text{Revenue})$ nhằm ổn định phương sai chuỗi thời gian và xấp xỉ phân phối chuẩn.

### 2. Kỹ thuật Trích xuất Đặc trưng (Feature Engineering)
* **Thời gian & Lịch biểu:** `year`, `month`, `day`, `day_of_week`, `day_of_year`, `week_of_year`, `time_index`.
* **Sóng chu kỳ (Fourier Terms):** `month_sin = sin(2π * month / 12)`, `month_cos = cos(2π * month / 12)`.
* **Sự kiện & Mùa vụ E-commerce Việt Nam:**
  - `is_weekend`: Cuối tuần (Thứ 7, Chủ Nhật).
  - `is_month_start`, `is_month_end`: Chu kỳ nhận lương / thanh toán đầu & cuối tháng.
  - `is_double_day`: Ngày hội mua sắm siêu sale (01/01, 02/02, ..., 12/12).
  - `is_black_friday`: Tuần lễ giảm giá Black Friday (23/11 – 28/11).
  - `is_tet`: Cao điểm mua sắm Tết Nguyên Đán (Tháng 1 & Tháng 2).

### 3. Bộ Ba Thuật Toán & Cơ Chế Blending
Mô hình kết hợp 3 kiến trúc Gradient Boosting đa dạng hàm mất mát nhằm tăng tính tổng quát hóa:
* **LightGBM Regressor:** `n_estimators=1000`, `learning_rate=0.01`, `num_leaves=31`, `objective='huber'` (Kháng nhiễu ngoại lai).
* **CatBoost Regressor:** `iterations=1000`, `learning_rate=0.02`, `depth=6`, `loss_function='MAE'` (Tối ưu trực tiếp sai số tuyệt đối).
* **XGBoost Regressor:** `n_estimators=800`, `learning_rate=0.01`, `max_depth=6`, `objective='reg:absoluteerror'`.
* **Ensemble Blending:**
  $$
  \hat{Y}_{\text{Revenue}} = 0.4 \times \hat{Y}_{\text{LightGBM}} + 0.4 \times \hat{Y}_{\text{CatBoost}} + 0.2 \times \hat{Y}_{\text{XGBoost}}
  $$

### 4. Dự báo COGS (Giá vốn hàng bán)
* Để đảm bảo tính nhất quán nghiệp vụ, tỷ lệ $\text{COGS Ratio}$ được tính dựa trên trung bình động 365 ngày gần nhất của tập huấn luyện:
  $$
  \text{COGS Ratio} = \frac{\sum_{t \in \text{last 365 days}} \text{COGS}_t}{\sum_{t \in \text{last 365 days}} \text{Revenue}_t}
  $$
* Doanh thu dự báo được ánh xạ sang COGS kết hợp hệ số nhiễu vi mô: $\text{COGS}_{\text{pred}} = \text{Revenue}_{\text{pred}} \times \text{COGS Ratio} \times \epsilon$.

---

## 📈 Kết Quả Đánh Giá & Tính Giải Thích (Model Evaluation & XAI)

### 1. Kiểm định Chéo Theo Chuỗi Thời Gian (5-Fold TimeSeriesSplit)
Kiểm thử mô hình trên tập validation với các mốc thời gian tịnh tiến:
* **Fold 1 - MAE:** `1,163,586.59`
* **Fold 2 - MAE:** `1,182,868.30`
* **Fold 3 - MAE:** `1,262,152.72`
* **Fold 4 - MAE:** `825,619.94`
* **Fold 5 - MAE:** `636,187.96`
* **Trung bình MAE Validation:** **`1,014,083.10`**

### 2. Phân Tích Độ Quan Trọng Đặc Trưng (Feature Importance & SHAP)
* **Top đặc trưng quan trọng:** `time_index` (xu hướng tăng trưởng dài hạn), `year`, `day_of_year`, `month_cos` và các biến cờ sự kiện `is_double_day`, `is_tet`.
* **SHAP Summary Values:** Xác nhận tác động tích cực vượt trội của các ngày Mega Sale (`is_double_day`) và chu kỳ cuối năm tới biến động doanh thu.

---

## 📊 Trực Quan Hóa Dữ Liệu & Phân Tích KPI (`Visualization.ipynb`)

Phân tích dữ liệu bám sát Khung 6 phòng ban từ tài liệu `[DATATHON26'] KPI Framework by Department.pdf`:
* **Commercial & UX:** Phân tích độ phụ thuộc khuyến mãi (Promo Penetration ~39%, Cannibalization >80%).
* **Customer Cohort:** Phân tích tỷ lệ mua lại (Inter-order Gap Median = 144 ngày, Churn Threshold = 288 ngày).
* **Inventory & Supply Chain:** Nhận diện 95% vốn đọng trong hàng deadstock (>90 ngày không phát sinh đơn) và rủi ro đứt hàng Top 5% Hero SKUs vào mùa cao điểm Tháng 11–12 (Triple Squeeze).

---

## 🚀 Hướng Dẫn Chạy Mô Hình & Tái Lập Kết Quả

### 1. Cài đặt môi trường
```bash
# Clone repository
git clone https://github.com/alobipbop/last-dance.git
cd last-dance

# Tạo và kích hoạt virtual environment
python -m venv venv
source venv/bin/activate  # Trên Windows: venv\Scripts\activate

# Cài đặt thư viện
pip install pandas numpy scikit-learn lightgbm xgboost catboost shap matplotlib plotly jupyter
```

### 2. Thực thi dự báo và xuất submission
Mở Jupyter Notebook hoặc chạy trực tiếp script dự báo:
```bash
jupyter notebook
```
* Mở notebook trong thư mục `Forecast Reveue Model/` và chạy toàn bộ cells để sinh file **`submission.csv`** dự báo hàng ngày từ `2023-01-01` đến `2024-07-01`.
* Mở **`Visualization.ipynb`** và **`1_Questions.ipynb`** để theo dõi phân tích khám phá dữ liệu (EDA) và trực quan hóa chuyên sâu.

---

## 👥 Nhóm Thực Hiện & Cuộc Thi
* **Đơn vị / Lab:** tennhom — National Economics University (NEU)
* **Cuộc thi:** Datathon 2026
