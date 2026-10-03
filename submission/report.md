# Lab 16 — Track 2 Report

1. Tôi sử dụng AWS tại region us-east-1, compute node loại t3.micro, source commit 47789e6.
2. Dataset Credit Card Fraud có 284,807 dòng; dữ liệu được chia train/test theo tỷ lệ 80/20, stratify theo nhãn, random_state=42.
3. Load dữ liệu mất 2.6054 giây; training mất 15.9044 giây; best iteration là 265.
4. AUC-ROC đạt 0.941576; Accuracy 0.999473; F1 0.845361; Precision 0.854167; Recall 0.836735.
5. Latency suy luận 1 dòng là 1.3981 ms; throughput 1,000 dòng là 44,038.09 dòng/giây.
6. Tại thời điểm quan sát, máy có 914 MiB RAM, CPU gần như idle sau benchmark; network interface không ghi nhận packet loss hoặc error.
7. AWS Cost Management chưa có dữ liệu chi phí vì tài khoản mới sử dụng Cost Management; cần kiểm tra lại sau khi dữ liệu được cập nhật.
8. Kết quả benchmark đã được tải về laptop; bộ bài nộp đã được commit và push lên GitHub, sau đó toàn bộ 27 tài nguyên AWS đã được destroy và Terraform state đã sạch.
