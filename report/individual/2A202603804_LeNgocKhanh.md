# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

## 2. Kết quả chính

| Phương án | F1 test | F1 SUB 40 | p95 ms/tin | USD/1k tin |
| --- | ---: | ---: | ---: | ---: |
| 1. BoW + LogReg (tự code) | 0.952 | 0.947 | 0.168 | 0 |
| 2. BoW + MLP | 0.974 | 1.000 | 0.439 | 0 |
| 3a. Encoder (MiniLM) + kNN | 0.896 | 1.000 | 50.535 | 0 |
| 3b. Encoder (MiniLM) + LogReg | 0.964 | 1.000 | 36.779 | 0 |
| 3c. Encoder (MiniLM) + MLP head | 0.975 | 1.000 | 30.867 | 0 |
| 4a. LLM nhỏ zero-shot | 0.952 | 0.952 | 2390.262 | 0 (chưa có giá) |
| 4b. LLM lớn zero-shot | 0.930 | 0.930 | 2392.400 | 0 (chưa có giá) |
| 4c. LLM nhỏ few-shot | 0.930 | 0.930 | 4230.083 | 0 (chưa có giá) |

LLM chỉ chạy trên tập SUB 40 tin, nên F1 test và F1 SUB của các hàng 4a–4c là cùng một số. Chi phí 0 của LLM nghĩa là model Groq chưa có trong bảng giá `PRICE_PER_1M`, không có nghĩa là miễn phí.

## 3. Dự đoán và thực tế

| Dự đoán | Bạn đoán | Thực tế |
| --- | --- | --- |
| `llm_models` | Model lớn F1 cao hơn model nhỏ khoảng 0.03–0.05 | Model lớn 0.930, thấp hơn model nhỏ 0.952 |
| `llm_fewshot` | Few-shot làm F1 tăng nhẹ hoặc không đổi | Few-shot 0.930, giảm so với zero-shot 0.952 |
| `tradeoff` | Encoder ít lỗi nhưng chậm hơn BoW; mình tưởng Encoder không bị lấn át | Chỉ BoW + LogReg và BoW + MLP nằm trên đường Pareto |

Vì sao lệch: model to hơn hay prompt dài hơn chưa chắc tốt hơn, và phải đo bằng tập đủ lớn thay vì đoán. Với `tradeoff`, BoW + MLP cùng 0% lỗi trên SUB nhưng nhanh hơn nhiều, nên lấn át Encoder + LogReg.

## 4. Phân tích 50 case khó

F1 test thường lên tới 0.95–1.00 nhưng trên 50 case khó chỉ còn 0.38–0.92. BoW + LogReg tụt nhiều nhất (0.947 → 0.378, giảm khoảng 0.57), BoW + MLP cũng giảm khoảng 0.55. Test thường "quá lạc quan" vì cùng phân phối với tập train, spam toàn từ khoá quen thuộc nên model chỉ cần khớp mặt chữ.

BoW khó nhất ở nhóm né từ khoá (BoW + MLP 0%), thao túng/lừa đảo (12%) và spam giọng tự nhiên (10–20%), vì BoW chỉ đếm từ rời rạc, gặp chữ bị tách ký tự hoặc spam không có từ khoá thì không nhận ra. Embedding khó nhất ở spam giọng tự nhiên (40–50%) và nói về spam (kNN 25%): nó nhìn nghĩa cả câu nhưng chỉ học ranh giới spam từ dữ liệu train. LLM khó nhất ở nhóm tưởng spam nhưng không phải (70%): đọc như người nên bị lừa bởi tin ham có giọng quảng cáo.

Hai case cụ thể với 4d (LLM nhỏ + prompt phòng thủ):

- **#2, tưởng spam nhưng không phải (thật = ham):** "FREE pizza in the office kitchen right now, come quick before it's gone". Tin ham có giọng quảng cáo nên LLM bị lừa.
- **#47, nói về spam (thật = ham):** "lol I just got a text saying I won £1000 and to call some 0906 number, people still fall for th..." lỗi này xuất phát từ việc model nhìn dấu hiệu bề mặt thay vì ý định của người viết. Cùng nhóm lỗi với “nói về spam” ở bảng nhóm case (4d đạt 75% ở nhóm này).

## 5. Thử thách cá nhân

**Đề #6 (theo SEED 521):** Sửa `build_classification_prompt` để LLM trả JSON `{"label": ..., "reason": ...}`, viết parse cho JSON, chạy lại trên 10 tin SUB, so token/chi phí với bản 1 từ. Đáng không?

**Cách làm:** code ở ô thử thách trong notebook (`build_json_prompt` và phần parse JSON).

| Bản | F1 | Accuracy | Token vào | Token ra | Chi phí | Độ trễ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 từ | 1.000 | 1.00 | ~472 | ~13 | $0 (chưa có giá) | 2455 ms/tin |
| JSON | 0.909 | 0.90 | ~512 | ~108 | $0 (chưa có giá) | 2421 ms/tin |

**Giải thích:** Trên 10 tin SUB, bản 1 từ đạt F1 = 1.000 và accuracy = 1.00, bản JSON đạt F1 = 0.909 và accuracy = 0.90, nên JSON không làm chất lượng tốt hơn. Token output tăng từ khoảng 13 lên khoảng 108 (thêm phần reason), độ trễ gần như không đổi (2455 ms so với 2421 ms mỗi tin). Chi phí hiện $0 vì model Groq chưa có trong bảng giá. Kết luận: không đáng nếu chỉ cần nhãn; JSON chỉ đáng dùng khi cần lý do để kiểm toán hoặc giải thích. Mẫu chỉ 10 tin nên chưa đủ kết luận về độ chính xác.

## 6. Phương án cho sản phẩm thật

LLM bị loại vì p95 từ khoảng 2390 đến 4230 ms, vượt xa yêu cầu 50 ms. Các phương án đạt độ trễ là BoW (p95 dưới 1 ms) và Encoder + LogReg, Encoder + MLP head (khoảng 31–37 ms). BoW + MLP rất nhanh nhưng chỉ đạt F1 0.45 trên 50 case khó, còn Encoder + MLP head đạt F1 0.975 trên test thường và 0.71 trên case khó, nên Encoder là lựa chọn đạt cả độ trễ lẫn chất lượng, đổi lại cần nhiều tài nguyên tính toán. Nếu dùng LLM cho 10 triệu tin/ngày với giá giả định 0.10/0.50 USD mỗi 1M token (100 token input, 1 token output) thì khoảng 105 USD/ngày, tức khoảng 3.150 USD/tháng.

Router: BoW + MLP hoặc Encoder xử lý tin chắc chắn (p dưới 0.3 là ham, p trên 0.7 là spam), chỉ tin lưỡng lự (p từ 0.3 đến 0.7) mới gửi LLM. Phần lớn tin rõ ràng nên LLM chỉ xử lý số ít, giữ được độ trễ và chi phí thấp mà vẫn có độ chính xác cao ở tin khó. Một cách đơn lẻ thì hoặc nhanh mà yếu ở case khó, hoặc chính xác mà chậm và đắt.

## 7. AI log và điều đã học

| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
| --- | --- | --- | --- |
| 0 | Giải thích câu 0.1 về data leakage | Giải thích khái niệm, đưa 3 gợi ý; không soạn câu trả lời | Tự viết câu trả lời, đối chiếu với số tin trùng thật (403) trong notebook |
| 1 | Gợi ý code vectorize, sigmoid, predict_proba, train_logreg, metrics | Đưa code cho 5 hàm | Tự dán, xoá raise NotImplementedError, chạy test đều pass |
| 2 | Hỏi về Linear không ReLU và opt.zero_grad() | Giải thích cơ chế, không đưa dự đoán F1 | Tự viết dự đoán F1, kiểm tra lại bằng cell thí nghiệm không ReLU |
| 2 | Gợi ý code build_mlp và vòng lặp huấn luyện 5 dòng | Đưa code nn.Sequential và zero_grad, backward, step | Sửa lỗi thụt lề và xoá raise còn sót, test pass |
| 3 | Gợi ý code cosine_similarity, top_k_similar, knn_predict | Đưa code dùng chuẩn hoá và nhân ma trận | Chạy test pass, đối chiếu kết quả bảng so sánh |
| 4 | Nhờ viết TODO 4.1–4.4 | Viết 4 hàm và thử bằng ô kiểm tra với API giả lập | Chạy ô kiểm tra pass, chạy thật với API và đối chiếu kết quả |
| 4 | Hỏi cách nhập key, cách đổi sang Gemini | Hướng dẫn các bước trên Colab | Tự tạo key, thêm vào Colab Secrets, chạy cell cấu hình |
| 4 | Hỏi lỗi 429, 400 khi gọi API và cách đổi provider | Giải thích giới hạn request, reasoning_effort, đổi sang Groq | Đổi LLM_PROVIDER sang groq, chạy lại, kiểm tra số liệu |
| 4–5 | Nhờ đối chiếu câu trả lời với output sau Run all | Sửa các con số lệch trong câu trả lời của tôi, không đổi code hay output | Đọc lại các chỗ đã sửa |

Điều hiểu ra hôm nay: mình dự đoán sai nhiều nhất việc model lớn sẽ tốt hơn model nhỏ và few-shot sẽ tăng F1. Điều này cho thấy model to hơn hay prompt dài hơn chưa chắc tốt hơn, và phải đo bằng tập đủ lớn thay vì đoán.
