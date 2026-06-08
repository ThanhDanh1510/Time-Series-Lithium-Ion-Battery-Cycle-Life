# CƠ SỞ VẬT LÝ HÓA ĐIỆN VÀ THIẾT KẾ ĐẶC TRƯNG MÔN VỊ (DOMAIN FEATURE ENGINEERING)

Tài liệu này cung cấp lý thuyết vật lý, hóa điện và nhiệt động học của pin Lithium-Ion làm cơ sở khoa học để thiết kế các đặc trưng domain phục vụ cho bài toán dự báo sớm tuổi thọ vòng đời (cycle life).

---

## 1. Cơ chế hoạt động và Sự suy thoái hóa điện

Pin Lithium-Ion hoạt động dựa trên sự xen kẽ (intercalation) và giải xen kẽ (deintercalation) của các ion Lithium ($Li^+$) giữa cực dương (cathode) và cực âm (anode) qua dung dịch chất điện phân (electrolyte). 

### 1.1. Sự hình thành và phát triển của lớp màng SEI (Solid Electrolyte Interphase)
* **Cơ chế hóa học**: Trong lần sạc đầu tiên, dung môi điện phân hữu cơ bị khử hóa hóa học trên bề mặt hoạt chất cực âm graphite (có thế thế năng hoạt động thấp hơn khoảng thế ổn định của điện phân). Phản ứng này tạo ra một lớp thụ động hóa mỏng không tan gồm các muối carbonate và alkoxide của Lithium, gọi là màng SEI.
* **Đặc tính động học**: Màng SEI là chất dẫn ion $Li^+$ nhưng là chất cách điện. Do đó, một khi màng SEI hình thành hoàn chỉnh, nó sẽ ngăn chặn sự khử điện phân tiếp theo. Tuy nhiên, SEI vẫn tăng trưởng chậm liên tục qua từng chu kỳ do sự khuếch tán của dung môi qua các lớp của SEI hoặc do hiện tượng nứt gãy cơ học lớp màng khi cực graphite co giãn thể tích (~10%) trong mỗi chu kỳ sạc/xả.
* **Hệ quả**: Sự tăng trưởng SEI tiêu tốn vĩnh viễn các ion Lithium chủ động ($Li^+$) và dung môi, dẫn đến sự suy hao dung lượng và làm tăng điện trở khuếch tán của cực âm.

### 1.2. Các chế độ suy thoái chính (Degradation Modes)
1. **Sự cạn kiệt Lithium chủ động (Loss of Active Lithium - LLI)**:
   Xảy ra do phản ứng phụ tạo SEI liên tục, hoặc do hiện tượng mạ Lithium (Lithium plating) trên bề mặt anode khi sạc nhanh (thế anode giảm xuống dưới thế $Li/Li^+$). LLI làm giảm trực tiếp lượng Lithium tự do có thể di chuyển giữa hai cực, gây suy giảm dung lượng xả thực tế ($QD$).
2. **Sự hao tổn hoạt chất điện cực (Loss of Active Material - LAM)**:
   Xảy ra do nứt gãy cơ học các hạt hoạt chất điện cực, biến dạng tinh thể vật liệu, hoặc sự hòa tan chất điện phân. LAM làm giảm diện tích bề mặt phản ứng hiệu dụng, cô lập một phần vật liệu cực dương hoặc cực âm khỏi mạng lưới dẫn điện tĩnh.

---

## 2. Ý nghĩa vật lý của các Phép đo Sớm đầu đời

Ở giai đoạn cực sớm (50-100 chu kỳ đầu), pin LFP/Graphite hoạt động ổn định và đường cong dung lượng xả ($QD$) hoàn toàn đi ngang. Việc sử dụng các phép đo vật lý động học sớm giúp mô hình dự báo sớm mà không cần đợi dung lượng sụt giảm thực tế.

### 2.1. Nội trở ban đầu ($IR$) - Dấu vết sai lệch chế tạo
Nội trở tĩnh ban đầu đo tại chu kỳ 2 ($IR_{cycle2}$) là thông số đại diện cho chất lượng chế tạo xuất xưởng (Manufacturing Quality):
* **Trở kháng ohmic**: Trở kháng của các lá thu dòng, điện cực hàn tiếp xúc cơ học, độ dẫn của dung dịch điện phân.
* **Trở kháng chuyển điện tích**: Động học phản ứng trao đổi ion tại bề mặt tiếp xúc điện cực/điện phân.
* Cell pin có nội trở ban đầu cao phản ánh lỗi lắp ráp nhẹ hoặc bất đồng nhất mật độ hoạt chất. Những pin này sẽ phát sinh nhiệt năng lớn và chịu tải dòng điện khắc nghiệt hơn, dẫn đến tăng tốc phản ứng phụ hủy hoại pin sau này.

### 2.2. Nhiệt động học phát nhiệt Joule ($I^2R$) và Độc tính Nhiệt độ
* **Tương tác nhiệt-trở**: Dòng điện sạc/xả chạy qua điện trở trong sinh ra công suất nhiệt tỏa ra bề mặt pin theo định luật Joule:
  $$P_{heat} \propto I^2 \times IR$$
* **Ảnh hưởng của Nhiệt độ bề mặt**: Nhiệt độ cục bộ trên bề mặt pin thúc đẩy tốc độ phản ứng hóa học phụ theo phương trình Arrhenius:
  $$k = A \cdot e^{-\frac{E_a}{R \cdot T}}$$
  Trong đó $E_a$ là năng lượng kích hoạt phản ứng tạo SEI, $T$ là nhiệt độ tuyệt đối. Sự gia tăng nhiệt độ nhỏ cũng đẩy nhanh tốc độ phản ứng phụ phá hủy màng SEI theo hàm mũ.
* **Sự bất đồng nhất nhiệt độ bề mặt ($T_{max} - T_{avg}$)**: Đo lường mức độ chênh lệch nhiệt bề mặt pin. Sự bất đồng nhất lớn cho thấy sự phân bố mật độ dòng điện nội bộ không đều, tạo nên các "điểm nóng cục bộ" (hotspots) kích hoạt sự thoái hóa nhanh chóng của SEI tại các vùng này.
* **Đặc trưng tương tác tương hỗ $IR \times Temp$**:
  * Tích số $IR_{cycle2} \times (Tmax_{cycle2} - Tavg_{cycle2})$ đại diện cho công suất tỏa nhiệt Joule và sự bất đồng nhất nhiệt tức thời ngay tại chu kỳ hoạt động thứ 2.
  * Tích số $IR_{mean\_w} \times Tavg_{mean\_w}$ đại diện cho tích lũy ứng suất nhiệt trung bình trong toàn bộ cửa sổ quan sát sớm.

### 2.3. Tích lũy cơ học ($QC + QD$) và Sự mỏi cơ học (Mechanical Fatigue)
Khi cực graphite hấp thụ ion Lithium trong quá trình sạc, mạng lưới graphite giãn nở khoảng 10% thể tích. Khi xả, mạng tinh thể co lại. 
* Quá trình co giãn liên tục này sinh ra **ứng suất cơ học nội bộ**, dẫn đến sự mỏi vật liệu và nứt nẻ lớp SEI.
* Tổng tích lũy dung lượng nạp và xả trong cửa sổ quan sát:
  $$\text{energy\_accum\_w} = \sum_{cycle=1}^{WINDOW} (QC_i + QD_i)$$
  Đại diện trực tiếp cho tổng tải cơ học co giãn mà cell pin đã trải qua trong giai đoạn hoạt động sớm.

---

## 3. Bản chất của việc Chuyển đổi sang Hồi quy dạng Bảng

Dữ liệu chuỗi thời gian thô của pin (thay đổi qua từng chu kỳ) là dữ liệu động học. Tuy nhiên, khi pin còn rất mới (ở chu kỳ 100), động học suy thoái dung lượng thô rất mờ nhạt. 

Bằng cách tính toán các đại lượng thống kê thống nhất trên cửa sổ $W$ (trung bình, độ lệch chuẩn, độ dốc tuyến tính, độ cong phi tuyến, hệ số biến thiên), chúng ta biến đổi bài toán chuỗi thời gian thành hồi quy dạng bảng (tabular). Phép chuyển đổi này giúp:
1. **Nén chuỗi thời gian thành các chỉ số tĩnh**: Biến đổi chuỗi 100 giá trị dung lượng phẳng lỳ thành một giá trị độ dốc duy nhất (`qd_slope`), phóng đại tín hiệu suy thoái động học sớm lên hàng trăm lần.
2. **Khai thác thuộc tính chế tạo tĩnh**: Cho phép các thuật toán cây quyết định (CatBoost, Random Forest) trực tiếp liên kết các thuộc tính chế tạo tĩnh đầu đời ($IR_{cycle2}$, $IR \times Temp$) với tuổi thọ tổng thể.
3. **Chống quá khớp**: Giảm chiều dữ liệu từ chuỗi thời gian nhiều chiều phức tạp xuống còn một vector đặc trưng duy nhất đại diện cho trạng thái cell pin, phù hợp cho kích thước mẫu pin thực tế nhỏ ($N=140$).
