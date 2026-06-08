# Lithium-Ion Battery Cycle Life Prediction Project

Dự án nghiên cứu và huấn luyện các mô hình học máy (Machine Learning) để dự báo tuổi thọ vòng đời (Cycle Life) của pin Lithium-Ion dựa trên dữ liệu chuỗi thời gian ở giai đoạn cực sớm (cửa sổ 50 và 100 chu kỳ đầu đời).

---

## 📁 Cấu trúc Thư mục Dự án

Thư mục dự án đã được sắp xếp lại gọn gàng theo cấu trúc khoa học chuẩn của một dự án Khoa học Dữ liệu (Data Science):

```text
Lithium-Ion Battery Cycle Life Time-Series/
├── data/
│   ├── raw/                  # Dữ liệu thô tải về từ Kaggle
│   │   ├── Lithium-Ion Battery Cycle Life.csv
│   │   ├── 50_Cycle_Lithium-Ion Battery Cycle Life.csv
│   │   └── 100_Cycle_Lithium-Ion Battery Cycle Life.csv
│   └── processed/            # Các tập đặc trưng (features) được trích xuất
│       ├── dataset_1_baseline_w50.csv
│       ├── dataset_2_fe_w50.csv
│       ├── dataset_3_baseline_w100.csv
│       └── dataset_4_fe_w100.csv
├── notebooks/                # Các notebook phân tích và huấn luyện chính
│   ├── EDA_Lithium_Ion_Battery_Cycle_Life.ipynb  # Phân tích dữ liệu & Trích xuất đặc trưng
│   └── modeling_pipeline.ipynb                    # Pipeline huấn luyện, tối ưu siêu tham số & Đánh giá
├── plots/                    # Các đồ thị trực quan hóa kết quả đầu ra
│   ├── severson_features_scatter.png             # Trực quan các đặc trưng Severson ban đầu
│   ├── model_comparison.png                      # So sánh hiệu năng Test RMSE giữa các mô hình
│   ├── feature_importance.png                    # Độ quan trọng đặc trưng của mô hình tốt nhất
│   └── residuals_plot.png                        # Phân tích sai số thặng dư của mô hình tốt nhất
└── README.md                 # Tài liệu hướng dẫn cấu trúc dự án này
```

---

## 🚀 Hướng dẫn Chạy các Notebook

Các Jupyter Notebook trong thư mục `notebooks/` được thiết kế để chạy trực tiếp với các đường dẫn tương đối (relative paths):

1. **Bước 1 (EDA & Feature Extraction)**:
   Chạy notebook `notebooks/EDA_Lithium_Ion_Battery_Cycle_Life.ipynb`. Notebook này sẽ phân tích các đặc trưng vật lý động học, xử lý bất thường IQR ở cấp độ pin và xuất ra 4 tập đặc trưng tương ứng tại thư mục `data/processed/`.

2. **Bước 2 (Model Training & Evaluation)**:
   Chạy notebook `notebooks/modeling_pipeline.ipynb`. Notebook này sẽ tự động đọc dữ liệu thô, thực hiện trích xuất đặc trưng động, chạy 5-Fold Cross-Validation sạch không rò rỉ thông tin (leak-free) kết hợp tối ưu siêu tham số (`RandomizedSearchCV`), kiểm định chéo trên tập Holdout Test độc lập và lưu các biểu đồ so sánh vào thư mục `plots/`.

---

## 📊 Tóm tắt Hiệu năng Mô hình Tốt nhất (Dataset 4: FE + Window 100)

* **Mô hình**: CatBoost Regressor
* **Test RMSE**: **176.06 chu kỳ** (đạt độ chính xác cực cao, giải thích hơn **71% phương sai** của tuổi thọ pin chỉ với dữ liệu 100 chu kỳ đầu đời).
* **AdaBoost Regressor**: Đạt Test MAE nhỏ nhất là **131.38 chu kỳ**.
