# Credit Risk Scoring & Explainable AI (XAI)

## Overview

Dự án Python hướng tới đánh giá rủi ro tín dụng và giải thích các dự đoán bằng Explainable AI (XAI).

Trạng thái hiện tại: chỉ khởi tạo cấu trúc project. Chưa có dữ liệu, EDA, xử lý dữ liệu, feature engineering hoặc mô hình Machine Learning.

## Business Problem

Mục tiêu dự kiến là hỗ trợ đánh giá rủi ro tín dụng và làm rõ những yếu tố ảnh hưởng đến dự đoán. Định nghĩa biến mục tiêu, phạm vi sử dụng và tiêu chí đánh giá sẽ được xác định sau khi lựa chọn dataset và làm rõ yêu cầu nghiệp vụ.

## Dataset

Chưa lựa chọn hoặc thêm dataset. Nguồn dữ liệu, mô tả các trường và điều kiện sử dụng sẽ được bổ sung ở bước tiếp theo.

- `data/raw/`: lưu dữ liệu gốc cục bộ.
- `data/processed/`: lưu dữ liệu đã xử lý cục bộ.
- Nội dung dữ liệu trong hai thư mục trên được bỏ qua bởi Git; chỉ giữ các file `.gitkeep`.
- Không đưa dataset hoặc model artifact vào Git.

## Project Structure

```text
credit-risk-scoring-xai/
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   └── processed/
│       └── .gitkeep
├── notebooks/
├── src/
│   └── __init__.py
├── api/
│   └── __init__.py
├── models/
│   └── .gitkeep
├── reports/
│   └── figures/
│       └── .gitkeep
├── tests/
│   └── __init__.py
├── .gitignore
├── requirements.txt
└── README.md
```

- `notebooks/`: dành cho notebook khi cần; hiện để trống. Git không theo dõi thư mục trống.
- `src/`: package dành cho mã nguồn xử lý và phân tích sau này.
- `api/`: package dành cho API sau này.
- `models/`: lưu model artifact cục bộ, được bỏ qua bởi Git.
- `reports/figures/`: lưu hình minh họa và biểu đồ báo cáo sau này.
- `tests/`: package dành cho các kiểm thử sau này.
- `__init__.py`: đánh dấu các thư mục tương ứng là Python package; hiện để trống.
- `.gitkeep`: giữ các thư mục cần thiết trong Git khi chưa có nội dung.
- `.gitignore`: loại trừ môi trường Python, cache, cấu hình Visual Studio cục bộ, dữ liệu và model artifact.
- `requirements.txt`: nơi khai báo thư viện khi bắt đầu sử dụng; skeleton hiện chưa cần dependency bên ngoài.

## Roadmap

- [x] Khởi tạo cấu trúc project.
- [ ] Lựa chọn dataset và xác định bài toán cụ thể.
- [ ] Khám phá dữ liệu (EDA).
- [ ] Tiền xử lý dữ liệu và feature engineering.
- [ ] Xây dựng và đánh giá mô hình.
- [ ] Giải thích dự đoán bằng XAI.
- [ ] Xây dựng API, kiểm thử và hoàn thiện báo cáo.

Các bước tiếp theo chỉ thực hiện sau khi cấu trúc project được kiểm tra và có yêu cầu triển khai.
