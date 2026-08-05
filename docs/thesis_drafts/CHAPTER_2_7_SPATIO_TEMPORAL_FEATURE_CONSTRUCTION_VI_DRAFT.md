# Section 2.7 — Spatio-Temporal Feature Construction

**Draft language:** Vietnamese and English  
**Draft status:** Provisional methodology; final feature-artifact authority is
still pending reconciliation of the current whitelist and semantic audits.  
**Approved thesis title:** *An AI Spatio-Temporal Framework for Dynamic
Behavior Profiling and Anomaly Detection in Group-Housed Pigs*

Mục này mô tả cách xây dựng bằng chứng không gian–thời gian từ RGB và các
quan sát hình học, cùng với các mặt nạ khả dụng dùng để phân biệt giá trị đo
được với quan sát không khả dụng. Phạm vi được tách theo hạt tính toán của
evidence native phục vụ review và view cuối cùng phục vụ mô hình; vì vậy các
đại lượng phát sinh theo thời gian luôn được tính lại trong đúng view đã khai
báo.

## Vietnamese thesis draft

### 2.7.1. Phạm vi và nguyên tắc xây dựng đặc trưng

Sau khi nguồn đã hiệu chỉnh được xác lập ở Mục 2.6, pipeline thực hiện việc
xây dựng ở ba hạt tính toán tách biệt. Các đại lượng cục bộ theo frame trước
hết được suy ra từ authority frame–object có khóa nguồn; bằng chứng trong
native unit được tính trong phạm vi trọn vẹn của đơn vị đó để phục vụ review;
sau khi view của mô hình được xác định, các đặc trưng phụ thuộc quan sát được
tái tính từ đúng các slot được chọn trong view cuối cùng. Không dùng lại pair
hoặc summary đã tính ở hạt này cho hạt khác. Ở hạt native-review, khóa phạm
vi là native temporal-unit; ở hạt final-view, khóa phạm vi là view mô hình cụ
thể và mọi endpoint của pair phải nằm trong view đó. Nhánh model-facing tuân
theo quy tắc truy cập thời gian của chế độ đánh giá: cấu hình causal chỉ dùng
quan sát hiện tại và quá khứ, còn cấu hình offline chỉ dùng các quan sát thuộc
view hai chiều đã khai báo.

Các trường nhận diện nguồn, video, cá thể, nhãn hành vi, quyết định review, lý
do chọn mẫu và đường dẫn tệp được giữ để truy nguyên hoặc kiểm tra chất lượng
nhưng không được đưa vào X. Hidden cũng chỉ phục vụ kiểm soát chất lượng và
quyết định khả năng sử dụng cửa sổ; nó không phải là một đặc trưng dự đoán.

Các đặc trưng được tổ chức thành sáu nhóm bổ sung cho nhau: bằng chứng hình ảnh
RGB của actor, hình học hộp bao, động học và biến thiên hình dạng, quan hệ với
các vùng chức năng trong chuồng, ngữ cảnh xã hội giữa các cá thể và các tóm tắt
theo thời gian. Mỗi nhóm có quy tắc khả dụng riêng. Feature whitelist cố định
tên và thứ tự cột, nhóm đặc trưng, số chiều, phiên bản schema, quy tắc chuẩn
hóa, ngữ nghĩa giá trị giữ chỗ và các mặt nạ bắt buộc. Exporter dừng theo chế độ
fail-closed nếu cột bị thiếu, trùng, đổi thứ tự, không hữu hạn hoặc không được
khai báo. Khi một quan sát không đủ điều kiện hình học hoặc thiếu một quan hệ
cần thiết, giá trị giữ chỗ chỉ được dùng kèm mặt nạ hợp lệ; giá trị đó không
được hiểu là một quan sát đứng yên hoặc là không có tương tác.

### 2.7.2. Bằng chứng hình ảnh RGB của actor

Với mỗi actor, nhánh RGB chuẩn tạo chuỗi ảnh tập trung vào hộp bao của cá thể.
Crop được giữ tỉ lệ bằng phép letterbox, nhờ đó hạn chế biến dạng hình thái khi
đưa về kích thước đầu vào cố định. Chỉ số ảnh, cờ quan sát và chất lượng crop
được dùng để kiểm tra sự tương ứng giữa các frame trong một view. Các mô hình
tương tác có thể bật thêm hai nhánh ngữ cảnh tách biệt: ngữ cảnh toàn khung và
ngữ cảnh trực tiếp của partner được xác định từ hình học cùng frame; các nhánh
này không được ngầm ghép vào chuỗi actor-centred RGB. Dữ liệu độ sâu được giữ
như metadata thu nhận nhưng không được dùng trong bất kỳ cấu hình mô hình nào
được đánh giá.

### 2.7.3. Hình học hộp bao

Từ hộp bao và kích thước ảnh, hệ thống tính tâm và kích thước hộp theo tọa độ
chuẩn hóa, cùng diện tích và tỉ lệ khung hình. Các đại lượng này mô tả vị trí,
quy mô biểu kiến và biến dạng hình học của actor mà không phụ thuộc trực tiếp
vào số pixel tuyệt đối. Tính hợp lệ của hộp, biên ảnh và tính hữu hạn của các
đại lượng dẫn xuất được ghi thành cờ chất lượng. Hộp không hợp lệ không bị biến
thành một quan sát hình học hợp lệ; các slot không khả dụng được điền giá trị
trung tính và đi kèm mặt nạ để encoder bỏ qua chúng.

Với hộp (b_{i,t}=(x^{\min}_{i,t},y^{\min}_{i,t},x^{\max}_{i,t},y^{\max}_{i,t})) trên ảnh
có kích thước \(W\times H\), các đại lượng hình học cơ bản được tính như sau:

\[
\begin{aligned}
c^x_{i,t}&=\frac{x^{\min}_{i,t}+x^{\max}_{i,t}}{2W},&
c^y_{i,t}&=\frac{y^{\min}_{i,t}+y^{\max}_{i,t}}{2H},\\
w^n_{i,t}&=\frac{x^{\max}_{i,t}-x^{\min}_{i,t}}{W},&
h^n_{i,t}&=\frac{y^{\max}_{i,t}-y^{\min}_{i,t}}{H},\\
A^n_{i,t}&=w^n_{i,t}h^n_{i,t},&
r_{i,t}&=\frac{x^{\max}_{i,t}-x^{\min}_{i,t}}{y^{\max}_{i,t}-y^{\min}_{i,t}}.
\end{aligned}
\]

Tỉ lệ \(r_{i,t}\) là tỉ số chiều rộng pixel trên chiều cao pixel và chỉ được xác
định khi chiều cao dương. Sau khi dùng các tọa độ hiệu lực của box, mặt nạ hình
học được đặt là
\[
m^{\mathrm{geo}}_{i,t}=\mathbf{1}\!\left[
\begin{array}{l}
0\le x^{\min}_{i,t}<x^{\max}_{i,t}\le W,\\
0\le y^{\min}_{i,t}<y^{\max}_{i,t}\le H,\\
b_{i,t}\text{ có các tọa độ hữu hạn}
\end{array}\right].
\]
Các giá trị NaN, vô cực, ngoài biên hoặc có chiều cao bằng không bị đánh dấu
không khả dụng; nếu một bước tiền xử lý đã điều chỉnh tọa độ, việc điều chỉnh
đó vẫn được giữ trong provenance của box. Box không hợp lệ không được chuyển
thành quan sát hợp lệ bằng một hằng số làm trơn.

### 2.7.4. Động học và biến thiên hình dạng

Động học được tính từ các quan sát đã sắp xếp của cùng actor trong một phạm vi
tính toán đã khai báo. Ký hiệu \(\kappa^{(g)}_{i,t}\) là khóa phạm vi ở hạt
\(g\): tại hạt native-review, đó là native temporal-unit; tại hạt final-view,
đó là model view chính xác được chọn. Các cặp không được vượt qua khóa phạm vi
tương ứng; trong view cuối cùng, mọi boundary reset temporal-unit đã khai báo
vẫn được tôn trọng. Từ chênh lệch vị trí, kích thước và thời gian nguồn, schema động học
gồm vector vận tốc, tốc độ, các tốc độ thay đổi kích thước, thay đổi diện tích
và aspect ratio, đổi hướng, gia tốc tiếp tuyến, các thành phần gia tốc theo trục
và độ lớn gia tốc. Cặp kề và sparse pair được giữ bằng các mặt nạ riêng.

Với \(\Delta f_{i,t}=f_{i,t}-f_{i,t-1}\) và
\(\Delta\tau_{i,t}=\tau_{i,t}-\tau_{i,t-1}\), các thành phần của mặt nạ được
định nghĩa bởi:

\[
m^{\mathrm{pair}}_{i,t}=m^{\mathrm{prev}}_{i,t}
m^{\mathrm{id}}_{i,t}m^{\mathrm{scope}}_{i,t}
m^{\mathrm{geo}}_{i,t-1}m^{\mathrm{geo}}_{i,t}m^{\mathrm{time}}_{i,t},
\qquad
m^{\mathrm{scope}}_{i,t}=\mathbf{1}[\kappa^{(g)}_{i,t}=\kappa^{(g)}_{i,t-1}],
\]
trong đó \(m^{\mathrm{prev}}\) cho biết quan sát trước tồn tại,
\(m^{\mathrm{id}}\) xác nhận cùng actor trajectory, còn
\[
m^{\mathrm{time}}_{i,t}=\mathbf{1}[\Delta f_{i,t}>0,\Delta\tau_{i,t}>0,
\Delta\tau_{i,t}\text{ hữu hạn}].
\]
Nếu một cấu hình đăng ký thêm giới hạn khoảng cách thời gian, điều kiện đó
được áp dụng trong \(m^{\mathrm{time}}\); không mặc nhiên coi mọi sparse pair là
liên tục.

\[
m^{\mathrm{adj}}_{i,t}=m^{\mathrm{pair}}_{i,t}\mathbf{1}[\Delta f_{i,t}=1],
\qquad
m^{\mathrm{sparse}}_{i,t}=m^{\mathrm{pair}}_{i,t}\mathbf{1}[\Delta f_{i,t}>1].
\]

Chỉ khi \(m^{\mathrm{pair}}_{i,t}=1\) mới tính:

\[
v^x_{i,t}=\frac{c^x_{i,t}-c^x_{i,t-1}}{\Delta\tau_{i,t}},\qquad
v^y_{i,t}=\frac{c^y_{i,t}-c^y_{i,t-1}}{\Delta\tau_{i,t}},\qquad
s_{i,t}=\sqrt{(v^x_{i,t})^2+(v^y_{i,t})^2}.
\]

Với \(q\in\{w^n,h^n,A^n,r\}\), tốc độ thay đổi hình dạng được tính theo
\(\dot q_{i,t}=\frac{q_{i,t}-q_{i,t-1}}{\Delta\tau_{i,t}}\). Vận tốc được gắn
với thời điểm giữa khoảng:
\[
\bar{\tau}_{i,t}=\frac{\tau_{i,t-1}+\tau_{i,t}}{2}.
\]
Gia tốc được tính giữa hai mẫu vận tốc hợp lệ, với
\[
\Delta\tau^{a}_{i,t}=\bar{\tau}_{i,t}-\bar{\tau}_{i,t-1}
=\frac{\Delta\tau_{i,t}+\Delta\tau_{i,t-1}}{2}>0,
\qquad
m^{a}_{i,t}=m^{\mathrm{pair}}_{i,t}m^{\mathrm{pair}}_{i,t-1}
\mathbf{1}[\Delta\tau^{a}_{i,t}>0].
\]

\[
a^{\mathrm{tan}}_{i,t}=\frac{s_{i,t}-s_{i,t-1}}{\Delta\tau^{a}_{i,t}},
\qquad
a^x_{i,t}=\frac{v^x_{i,t}-v^x_{i,t-1}}{\Delta\tau^{a}_{i,t}},
\qquad
a^y_{i,t}=\frac{v^y_{i,t}-v^y_{i,t-1}}{\Delta\tau^{a}_{i,t}},
\qquad
a^{\mathrm{vec}}_{i,t}=\sqrt{(a^x_{i,t})^2+(a^y_{i,t})^2}.
\]

Các giá trị gia tốc chỉ được coi là quan sát khi \(m^{a}_{i,t}=1\). Với
\(\theta_{i,t}=\operatorname{atan2}(v^y_{i,t},v^x_{i,t})\), độ đổi hướng dùng
hiệu góc cuộn:
\[
\Delta\theta_{i,t}=\operatorname{atan2}\!\left(
\sin(\theta_{i,t}-\theta_{i,t-1}),
\cos(\theta_{i,t}-\theta_{i,t-1})\right),
\]
và chỉ hợp lệ khi hai mẫu vận tốc có tốc độ dương,
\(m^{\mathrm{dir}}_{i,t}=m^{a}_{i,t}\mathbf{1}[s_{i,t}>0]
\mathbf{1}[s_{i,t-1}>0]\). Vận tốc của một sparse pair dùng đúng khoảng thời
gian nguồn; sparse pair không tự động đóng góp vào độ dài đường đi liên tục,
thời lượng episode tiếp xúc hoặc số đếm continuity.

Các thống kê tổng hợp như độ dài đường đi, độ dịch chuyển, độ ổn định hộp và
tỉ lệ chuyển động được tính lại trong chính view thời gian sẽ đưa vào mô hình;
chúng không lấy dữ liệu ngoài phạm vi view.

Thời gian sinh học được suy ra từ timestamp hoặc đồng hồ nguồn, không từ tốc độ
đóng gói 30 fps của tệp phát lại. Quy ước này bảo đảm rằng tốc độ, khoảng cách
theo giây và thời lượng hành vi vẫn có cùng ý nghĩa khi nguồn 6 fps được đóng gói
lại ở 30 fps.

### 2.7.5. Quan hệ với vùng chức năng

Các vùng cố định của chuồng gồm feeder, drinker và toy/enrichment. Authority ROI
hiện hành có một feeder, hai drinker và một toy/enrichment region. Với mỗi lớp
vùng, hệ thống tính khoảng cách nhỏ nhất, mức chồng lấn lớn nhất, IoU lớn nhất,
quan hệ tâm nằm trong vùng và các cờ near/contact cùng trạng thái khả dụng. Mô
hình nhận đồng thời quan hệ với cả ba lớp vùng để tránh suy ra lớp vùng từ nhãn
hành vi. Các trường target_roi hoặc trường được chọn theo hành vi chỉ dùng cho
review và audit; chúng bị loại khỏi X để ngăn target leakage. Vùng không khả
dụng giữ mặt nạ riêng thay vì làm mất toàn bộ mẫu.

Sau khi quy đổi ROI về kích thước ảnh hiện tại, các đại lượng hình học được
tính trên từng hình chữ nhật trục song song. Với actor box \(b\) và một instance
\(R_{k,h}=(x^R_1,y^R_1,x^R_2,y^R_2)\), tập instance của lớp \(k\) là
\[
\mathcal R_k=\{R_{k,1},\ldots,R_{k,n_k}\}.
\]
đặt
\[
\begin{aligned}
w_{\cap}&=\max\{0,\min(x_2,x^R_2)-\max(x_1,x^R_1)\},\\
h_{\cap}&=\max\{0,\min(y_2,y^R_2)-\max(y_1,y^R_1)\},\\
I&=w_{\cap}h_{\cap}.
\end{aligned}
\]
Khoảng cách được chuẩn hóa theo đường chéo ảnh:
\[
d(b,R_{k,h})=
\frac{\sqrt{\delta_x^2+\delta_y^2}}{\sqrt{W^2+H^2}},
\quad
\delta_x=\max\{x^R_1-x_2,x_1-x^R_2,0\},
\quad
\delta_y=\max\{y^R_1-y_2,y_1-y^R_2,0\}.
\]
Với \(A_b=|b|\) và \(A_{k,h}=|R_{k,h}|\), overlap và IoU lần lượt là
\[
o(b,R_{k,h})=\frac{I}{A_b},\qquad
\operatorname{IoU}(b,R_{k,h})=\frac{I}{A_b+A_{k,h}-I}.
\]
Giá trị theo lớp được lấy bằng
\[
d_{i,t,k}=\min_{R\in\mathcal R_k}d(b_{i,t},R),\qquad
o_{i,t,k}=\max_{R\in\mathcal R_k}o(b_{i,t},R),\qquad
\operatorname{IoU}_{i,t,k}=\max_{R\in\mathcal R_k}\operatorname{IoU}(b_{i,t},R),
\]
và
\[
q^{\mathrm{inside}}_{i,t,k}=\max_{R\in\mathcal R_k}
\mathbf{1}[c(b_{i,t})\text{ nằm trong polygon của }R].
\]
Diện tích ROI dùng trong mẫu số IoU được chặn dưới ở một pixel theo phép tính
hiện tại. Cờ near và contact được xác định theo đúng ngưỡng đã đăng ký:
\[
q^{\mathrm{near}}_{i,t,k}=\mathbf{1}[d_{i,t,k}\le 0.08],
\]
\[
q^{\mathrm{contact}}_{i,t,k}=\mathbf{1}[d_{i,t,k}\le 0.02]
\lor\mathbf{1}[o_{i,t,k}>0]\lor q^{\mathrm{inside}}_{i,t,k}.
\]
Các mẫu có box không hợp lệ hoặc lớp ROI không khả dụng nhận mặt nạ khả dụng
bằng không; chúng không được diễn giải là không tương tác. Các trường ROI chọn
theo nhãn hành vi vẫn bị loại khỏi \(X\).

### 2.7.6. Ngữ cảnh xã hội và quan hệ partner

Ngữ cảnh xã hội được xây dựng từ hình học của các actor xuất hiện trong cùng
frame. Representation xã hội được đánh giá sử dụng partner hợp lệ gần nhất,
không phải một encoder top-K hay đồ thị. Hệ thống mô tả khoảng cách, IoU hoặc
mức chồng lấn, mật độ lân cận, tiếp xúc, tốc độ tiếp cận và tách xa, cùng một
chỉ báo hình học về mức độ tương tác. Các đại lượng này không đọc nhãn fight
hoặc social-nose; chúng chỉ dùng vị trí, hộp bao, thời gian và trạng thái khả
dụng của các actor. Chỉ các trường xã hội có trong whitelist của tensor cuối
cùng mới được đưa vào X; các summary khác chỉ là evidence hoặc audit.

Trong nhánh partner gần nhất, tập ứng viên hợp lệ của actor \(i\) tại frame
\(t\) được ký hiệu là \(\mathcal N_{i,t}\). Khoảng cách lựa chọn là khoảng cách
\(axis\)-normalized trong ảnh:
\[
d^{\mathrm{social}}_{ij,t}=
\sqrt{\left(\frac{\Delta x_{ij,t}}{W_t}\right)^2+
\left(\frac{\Delta y_{ij,t}}{H_t}\right)^2}.
\]
Partner được chọn bởi
\[
j^\ast_{i,t}
=
\operatorname*{arg\,min}_{j\in\mathcal N_{i,t}}^{\mathrm{lex}}
\left(d^{\mathrm{social}}_{ij,t},\kappa_j\right),
\]
trong đó \(\kappa_j\) là khóa partner canonical gắn với nguồn và phép tối ưu là
lexicographic. Near dùng \(d^{\mathrm{social}}_{ij,t}\le 0.08\); contact dùng
\(\operatorname{IoU}_{ij,t}\ge 0.01\) hoặc overlap ratio \(\ge 0.05\). Khi
\(\mathcal N_{i,t}\) rỗng, đặc trưng partner nhận giá trị trung tính và mặt nạ
khả dụng bằng không. Các biểu diễn top-\(K\) hoặc dạng đồ thị không thuộc
representation cuối cùng được đánh giá.

### 2.7.7. Tóm tắt theo thời gian và kiểm soát khả dụng

Các đặc trưng theo cửa sổ được chia thành biến dự đoán và biến eligibility/audit.
Nhóm dự đoán gồm các đại lượng hình học, động học, quan hệ ROI và quan hệ xã
hội đã được whitelist; nhóm eligibility/audit gồm độ dài view, timestamp,
maximum gap, số hàng quan sát, chất lượng box và các cờ kiểm tra lineage. Việc
tổng hợp chỉ dùng các quan sát thuộc view đã khai báo; các cặp, quan hệ xã hội
và thống kê phụ thuộc view được tái tính sau khi view được tạo. Split, review
metadata, source type, actor ID và các trường mô tả nhãn chỉ được dùng ở các
bước audit hoặc tạo target, không trở thành tín hiệu đầu vào.

Bảng feature whitelist là ranh giới cuối cùng giữa đặc trưng đã tính và đặc
trưng được phép học. Một cột chỉ được vào X khi có tên trong whitelist, có kiểu
số hợp lệ, không phải nhãn, định danh, đường dẫn hay metadata review, và vượt
qua kiểm tra tên, thứ tự, số chiều, tính đầy đủ, tính hữu hạn, chuẩn hóa và các
mặt nạ bắt buộc. Exporter dừng theo chế độ fail-closed khi một cột bị thiếu,
trùng, đổi thứ tự, không hữu hạn hoặc không được khai báo. Các mặt nạ độ dài,
frame quan sát và chất lượng được truyền như điều khiển khả dụng của sequence,
không được diễn giải như giá trị hành vi.
Với một đặc trưng \(z_{i,t}\), đặt \(\mathcal P_i^{(z)}\) là tập các vị trí
trong view tại đó đặc trưng có thể được xác định và \(m^{(z)}_{i,t}\) là mặt nạ
riêng của đặc trưng đó. Giá trị tổng hợp là
\[
\bar z_i=
\frac{\sum_{t\in\mathcal P_i^{(z)}}m^{(z)}_{i,t}z_{i,t}}
{\sum_{t\in\mathcal P_i^{(z)}}m^{(z)}_{i,t}},
\qquad
\sum_{t\in\mathcal P_i^{(z)}}m^{(z)}_{i,t}>0.
\]
và độ bao phủ là
\[
\rho_i^{(z)}=
\frac{\sum_{t\in\mathcal P_i^{(z)}}m^{(z)}_{i,t}}
{|\mathcal P_i^{(z)}|}.
\]
Các đặc trưng pair dùng \(m^{\mathrm{pair}}\), đổi hướng dùng
\(m^{\mathrm{dir}}\), gia tốc dùng \(m^a\), quan hệ ROI dùng mặt nạ ROI và
quan hệ xã hội dùng mặt nạ social tương ứng. Không dùng một mẫu số chung kiểu
\(N_{\mathrm{observed}}-2\) khi các vị trí ứng viên bị thiếu hoặc thưa. Khi
không có giá trị hợp lệ, exporter xuất placeholder theo hợp đồng và mặt nạ
bằng không; trạng thái này không được diễn giải là chuyển động bằng không. Các
mặt nạ quan sát, hình học, cặp động học, ROI và xã hội vẫn được giữ riêng.

## English academic thesis draft

This section defines the RGB-derived spatial and temporal evidence supplied to
the behavior model and the validity controls that distinguish measured values
from unavailable observations. It separates the native-review evidence grain
from the final model-view grain, so every temporal quantity is recomputed in
the exact view in which it is used.

### 2.7.1. Scope and construction principles

After the corrected source has been established in Section 2.6, construction is
performed at three explicitly separated computation grains. Frame-local
primitives are first derived from the source-qualified frame–object authority.
Native-unit evidence is then computed within each complete annotation unit for
the review procedure. Once the model view is fixed, view-dependent features are
rebuilt from the exact selected observations in that final view. No pair-derived
quantity or temporal summary is reused across grains. At the native-review
grain, the scope key is the native temporal unit; at the final-view grain, it is
the exact model view, and both endpoints of a pair must lie in that view. The
model-facing branch follows the temporal-access contract of the evaluated mode:
causal configurations use current and past observations, whereas offline
configurations use only observations contained in the declared bidirectional
view.

Source, video, actor, behavior-label, review-decision, selection-reason and
file-path fields remain available for provenance and quality control but are
excluded from model input X. Hidden is likewise used for quality control and
window eligibility; it is not a predictive feature.

The representation combines six complementary families: actor-centred RGB
appearance, bounding-box geometry, motion and shape dynamics, functional-region
relations, social context, and temporal summaries. Each family carries its own
availability information. The feature whitelist fixes the ordered column names,
feature groups, dimensionality, schema version, normalization policy,
placeholder semantics and required masks. Export therefore fails closed when a
column is missing, duplicated, reordered, non-finite or undeclared. When a
geometric or contextual observation is not valid, the exporter uses a neutral
placeholder together with an explicit validity mask; the placeholder therefore
does not denote a stationary animal or the absence of an interaction.

### 2.7.2. Actor-centred RGB appearance

For each actor, the canonical RGB branch builds a sequence of observations
centred on the actor bounding box. Letterbox resizing preserves the crop aspect
ratio when images are mapped to the fixed encoder size, reducing artificial
changes in body shape. Image indices, observation masks and crop-quality
indicators verify frame alignment within a view. Interaction models may enable
two separate context branches: full-frame context and direct-partner context,
with the partner selected from same-frame geometry rather than from a behavior
label; these branches are not implicitly concatenated with the actor-centred
RGB sequence. Depth is retained as acquisition metadata but is not used by any
evaluated model configuration.

### 2.7.3. Bounding-box geometry

The geometry branch derives normalized actor-centre coordinates and normalized
box width and height, together with box area and aspect ratio. These variables
represent apparent position, scale and shape while limiting dependence on raw
pixel dimensions. Box validity, image-boundary checks and finite-value checks are
stored as quality evidence. Invalid boxes are not converted into valid geometric
observations; unavailable slots receive neutral values and an accompanying mask
so that the sequence encoder can ignore them.

For a box \(b_{i,t}=(x^{\min}_{i,t},y^{\min}_{i,t},x^{\max}_{i,t},y^{\max}_{i,t})\) in an image
of width \(W\) and height \(H\), the normalized geometric quantities are
\[
\begin{aligned}
c^x_{i,t}&=\frac{x^{\min}_{i,t}+x^{\max}_{i,t}}{2W},&
c^y_{i,t}&=\frac{y^{\min}_{i,t}+y^{\max}_{i,t}}{2H},\\
w^n_{i,t}&=\frac{x^{\max}_{i,t}-x^{\min}_{i,t}}{W},&
h^n_{i,t}&=\frac{y^{\max}_{i,t}-y^{\min}_{i,t}}{H},\\
A^n_{i,t}&=w^n_{i,t}h^n_{i,t},&
r_{i,t}&=\frac{x^{\max}_{i,t}-x^{\min}_{i,t}}{y^{\max}_{i,t}-y^{\min}_{i,t}}.
\end{aligned}
\]
The aspect ratio is defined as pixel width divided by pixel height and only for a
positive height. After any source-side coordinate normalization, the effective
box receives the explicit validity mask
\[
m^{\mathrm{geo}}_{i,t}=\mathbf{1}\!\left[
\begin{array}{l}
0\le x^{\min}_{i,t}<x^{\max}_{i,t}\le W,\\
0\le y^{\min}_{i,t}<y^{\max}_{i,t}\le H,\\
b_{i,t}\text{ has finite coordinates}
\end{array}\right].
\]
Out-of-image, non-finite or zero-height boxes are masked as unavailable; any
prior coordinate adjustment remains available in the box provenance. No
smoothing constant is used to turn an invalid box into a valid observation.

### 2.7.4. Motion and shape dynamics

The motion branch uses ordered observations of the same actor within a declared
computation scope. Let \(\kappa^{(g)}_{i,t}\) denote the scope key at grain
\(g\): the native temporal-unit key at the native-review grain and the exact
selected model-view key at the final-view grain. Pairs may not cross the scope
key, and any declared temporal-unit reset boundary is preserved within the
final view. Using source timestamps, the schema represents velocity components, speed,
rates of change in box dimensions, area and aspect ratio, directional change,
tangential acceleration, axis-wise acceleration and acceleration magnitude.
Adjacent and sparse pairs are retained with separate masks. View-specific
summaries are recomputed inside the evaluated view rather than inherited from an
unrelated interval.

For an ordered pair in the declared view, let
\(\Delta f_{i,t}=f_{i,t}-f_{i,t-1}\) and
\(\Delta\tau_{i,t}=\tau_{i,t}-\tau_{i,t-1}\). The validity components are
\[
m^{\mathrm{pair}}_{i,t}
=m^{\mathrm{prev}}_{i,t}m^{\mathrm{id}}_{i,t}
m^{\mathrm{scope}}_{i,t}
m^{\mathrm{geo}}_{i,t-1}m^{\mathrm{geo}}_{i,t}
m^{\mathrm{time}}_{i,t},
\qquad
m^{\mathrm{scope}}_{i,t}
=\mathbf{1}[\kappa^{(g)}_{i,t}=\kappa^{(g)}_{i,t-1}],
\]
where \(m^{\mathrm{prev}}\) records the previous observation,
\(m^{\mathrm{id}}\) confirms the same actor trajectory, and
\[
m^{\mathrm{time}}_{i,t}
=\mathbf{1}[\Delta f_{i,t}>0,\ \Delta\tau_{i,t}>0,\
\Delta\tau_{i,t}\text{ finite}].
\]
If a registered configuration imposes an additional maximum-gap rule, it is
included in \(m^{\mathrm{time}}\); a sparse pair is not treated as contiguous
by default.
Adjacent and sparse evidence are retained as separate masks:
\[
m^{\mathrm{adj}}_{i,t}=m^{\mathrm{pair}}_{i,t}\mathbf{1}[\Delta f_{i,t}=1],
\qquad
m^{\mathrm{sparse}}_{i,t}=m^{\mathrm{pair}}_{i,t}\mathbf{1}[\Delta f_{i,t}>1].
\]
Only valid pairs contribute to
\[
v^x_{i,t}=\frac{c^x_{i,t}-c^x_{i,t-1}}{\Delta\tau_{i,t}},
\qquad
v^y_{i,t}=\frac{c^y_{i,t}-c^y_{i,t-1}}{\Delta\tau_{i,t}},
\qquad
s_{i,t}=\sqrt{(v^x_{i,t})^2+(v^y_{i,t})^2}.
\]
For \(q\in\{w^n,h^n,A^n,r\}\),
\[
\dot q_{i,t}=\frac{q_{i,t}-q_{i,t-1}}{\Delta\tau_{i,t}}.
\]
Each velocity sample is assigned to the interval midpoint
\[
\bar{\tau}_{i,t}=\frac{\tau_{i,t-1}+\tau_{i,t}}{2}.
\]
Acceleration uses two valid velocity samples and
\[
\Delta\tau^a_{i,t}=\bar{\tau}_{i,t}-\bar{\tau}_{i,t-1}
=\frac{\Delta\tau_{i,t}+\Delta\tau_{i,t-1}}{2}>0,
\qquad
m^a_{i,t}=m^{\mathrm{pair}}_{i,t}m^{\mathrm{pair}}_{i,t-1}
\mathbf{1}[\Delta\tau^a_{i,t}>0].
\]
\[
a^{\mathrm{tan}}_{i,t}=\frac{s_{i,t}-s_{i,t-1}}{\Delta\tau^a_{i,t}},
\quad
a^x_{i,t}=\frac{v^x_{i,t}-v^x_{i,t-1}}{\Delta\tau^a_{i,t}},
\quad
a^y_{i,t}=\frac{v^y_{i,t}-v^y_{i,t-1}}{\Delta\tau^a_{i,t}},
\quad
a^{\mathrm{vec}}_{i,t}=\sqrt{(a^x_{i,t})^2+(a^y_{i,t})^2}.
\]

Acceleration values are observations only when \(m^a_{i,t}=1\). With
\(\theta_{i,t}=\operatorname{atan2}(v^y_{i,t},v^x_{i,t})\), wrapped directional
change is
\[
\Delta\theta_{i,t}=\operatorname{atan2}\!\left(
\sin(\theta_{i,t}-\theta_{i,t-1}),
\cos(\theta_{i,t}-\theta_{i,t-1})\right),
\]
with validity mask
\[
m^{\mathrm{dir}}_{i,t}=m^a_{i,t}\mathbf{1}[s_{i,t}>0]
\mathbf{1}[s_{i,t-1}>0].
\]
Velocity for a sparse pair uses the actual source-time separation, but a sparse
pair does not automatically contribute to contiguous path length, contact
duration or continuity counts.

Biological time is derived from source timestamps or the source acquisition
clock, not from the 30-fps playback packaging. Consequently, rates expressed per
second and duration-related summaries retain the 6-fps acquisition semantics even
when the stored video is packaged at 30 fps.

### 2.7.5. Functional-region relations

The fixed pen regions are the feeder, drinker and toy/enrichment region. The
current ROI authority contains one feeder, two drinkers and one toy/enrichment
region. For each region class, the pipeline derives the minimum distance, maximum
overlap, maximum IoU, centre-inside status, near/contact indicators and region
availability. The model receives relations to all three classes so that the
target behavior cannot select its own region evidence. Target-ROI and other
behavior-routed fields remain review or audit variables and are excluded from X
to prevent target leakage. Missing region evidence is represented by an
availability mask rather than by silently dropping the sample.

After scaling each ROI instance to the current image size, the implementation
uses axis-aligned rectangle geometry. For actor box \(b\) and instance
\(R_{k,h}=(x^R_1,y^R_1,x^R_2,y^R_2)\), the instances of class \(k\) are
\[
\mathcal R_k=\{R_{k,1},\ldots,R_{k,n_k}\}.
\]
For \(b=(x_1,y_1,x_2,y_2)\),
\[
w_{\cap}=\max\{0,\min(x_2,x^R_2)-\max(x_1,x^R_1)\},\qquad
h_{\cap}=\max\{0,\min(y_2,y^R_2)-\max(y_1,y^R_1)\},
\qquad I=w_{\cap}h_{\cap}.
\]
The normalized box-to-ROI distance is
\[
d(b,R_{k,h})=
\frac{\sqrt{\delta_x^2+\delta_y^2}}{\sqrt{W^2+H^2}},
\quad
\delta_x=\max\{x^R_1-x_2,x_1-x^R_2,0\},
\quad
\delta_y=\max\{y^R_1-y_2,y_1-y^R_2,0\}.
\]
With \(A_b=|b|\) and \(A_{k,h}=|R_{k,h}|\), overlap and IoU for one instance are
\[
o(b,R_{k,h})=\frac{I}{A_b},\qquad
\operatorname{IoU}(b,R_{k,h})=\frac{I}{A_b+A_{k,h}-I}.
\]
Class-level values use
\[
d_{i,t,k}=\min_{R\in\mathcal R_k}d(b_{i,t},R),\qquad
o_{i,t,k}=\max_{R\in\mathcal R_k}o(b_{i,t},R),\qquad
\operatorname{IoU}_{i,t,k}=\max_{R\in\mathcal R_k}\operatorname{IoU}(b_{i,t},R),
\]
and
\[
q^{\mathrm{inside}}_{i,t,k}=\max_{R\in\mathcal R_k}
\mathbf{1}[c(b_{i,t})\text{ lies in the polygon of }R].
\]
The implementation floors the ROI area used in the IoU denominator at one
pixel. The registered predicates are
\[
q^{\mathrm{near}}_{i,t,k}=\mathbf{1}[d_{i,t,k}\le 0.08],
\]
\[
q^{\mathrm{contact}}_{i,t,k}=\mathbf{1}[d_{i,t,k}\le 0.02]
\lor\mathbf{1}[o_{i,t,k}>0]\lor q^{\mathrm{inside}}_{i,t,k}.
\]
Invalid boxes and unavailable ROI classes retain zero availability masks rather
than being interpreted as zero interaction. Behavior-selected target-ROI fields
remain excluded from \(X\).

### 2.7.6. Social and partner context

Social context is computed from the geometry of actors observed in the same
frame. The evaluated social representation uses the nearest valid partner; it
does not use a top-\(K\) or graph encoder. The features describe distance,
overlap or IoU, local density, contact, approach and retreat speed, and a
geometry-based interaction proxy. They do not inspect fight or social-nose
labels; they use only actor boxes, timing and availability. Only social fields
present in the final tensor whitelist enter X; other summaries remain evidence
or audit variables.

For the nearest-partner branch, let \(\mathcal N_{i,t}\) denote the valid
same-frame candidates. Selection uses the axis-normalized image distance
\[
d^{\mathrm{social}}_{ij,t}=
\sqrt{\left(\frac{\Delta x_{ij,t}}{W_t}\right)^2+
\left(\frac{\Delta y_{ij,t}}{H_t}\right)^2}.
\]
The selected partner is
\[
j^\ast_{i,t}
=
\operatorname*{arg\,min}_{j\in\mathcal N_{i,t}}^{\mathrm{lex}}
\left(d^{\mathrm{social}}_{ij,t},\kappa_j\right),
\]
where \(\kappa_j\) is the source-qualified canonical partner key and the
minimization is lexicographic. Near is defined by
\(d^{\mathrm{social}}_{ij,t}\le 0.08\); contact is defined by
\(\operatorname{IoU}_{ij,t}\ge 0.01\) or overlap ratio \(\ge 0.05\). If the
candidate set is empty, partner features receive a neutral placeholder and zero
availability. Top-\(K\) and graph representations are not part of the evaluated
final representation.

### 2.7.7. Temporal summaries and availability control

Window-level features are separated into predictive variables and
eligibility/audit variables. Predictive variables include whitelisted geometry,
motion, ROI and social evidence; eligibility/audit variables include view
length, timestamps, maximum gaps, observed-row counts, box quality and lineage
checks. Aggregation uses only observations belonging to the declared view;
view-dependent pairs, social relations and summaries are recomputed after view
construction. Split roles, review metadata, source type, actor identifiers and
label descriptors are used for audit or target construction, not as model input.

The feature whitelist defines the final boundary between computed evidence and
learned input. A column enters X only when it is explicitly whitelisted, numeric
and finite after the declared missingness handling, and free of labels,
identifiers, paths and review metadata. The whitelist also fixes column order,
group dimensions, schema version, normalization, placeholder semantics and
required masks. Export fails closed when a column is missing, duplicated,
reordered, non-finite or undeclared. Length, observation and quality masks are
passed as sequence-availability controls; they are not interpreted as behavioral
values.

For a feature \(z_{i,t}\), let \(\mathcal P_i^{(z)}\) be the candidate positions
in the view at which the feature can be defined, and let \(m^{(z)}_{i,t}\) be
its feature-specific validity mask. The view-level aggregation is
\[
\bar z_i=
\frac{\sum_{t\in\mathcal P_i^{(z)}}m^{(z)}_{i,t}z_{i,t}}
{\sum_{t\in\mathcal P_i^{(z)}}m^{(z)}_{i,t}},
\qquad
\sum_{t\in\mathcal P_i^{(z)}}m^{(z)}_{i,t}>0.
\]
Coverage is
\[
\rho_i^{(z)}=
\frac{\sum_{t\in\mathcal P_i^{(z)}}m^{(z)}_{i,t}}
{|\mathcal P_i^{(z)}|}.
\]
Pair features use \(m^{\mathrm{pair}}\), direction change uses
\(m^{\mathrm{dir}}\), acceleration uses \(m^a\), ROI relations use the ROI
availability mask and social relations use their social-availability masks. A
blanket \(N_{\mathrm{observed}}-2\) denominator is not used when candidate
positions are missing or sparse. If no valid value exists, the exporter emits
the declared neutral placeholder with zero availability; this state is not
interpreted as zero motion. Observation, geometry, motion-pair, ROI and social
masks remain distinct.

## Evidence anchors

- docs/CLASSIFICATION_V2_THESIS_BLUEPRINT_EVIDENCE_MAP.md, Sections 2.7 and
  8.2: section contract and RGB-only feature boundary.
- configs/classification_v2/feature_semantics_v2.json: feature families, spatial
  groups, forbidden fields and depth/pen-context policy.
- configs/classification_v2/trainer_contract_v1.json and
  configs/classification_v2/trainer_contract_v2.json: tabular and spatial
  whitelist contracts.
- src/pig_behavior/classification_v2/features/geometry.py and
  src/pig_behavior/classification_v2/features/spatial_semantics.py: normalized
  geometry, spatial schema and validity handling.
- src/pig_behavior/classification_v2/features/motion_schema.py and
  src/pig_behavior/classification_v2/features/spatiotemporal.py: ordered motion
  schema registry, source-time pairs and within-unit evidence construction.
- src/pig_behavior/classification_v2/features/roi.py: feeder, drinker and toy
  relations.
- src/pig_behavior/classification_v2/features/social.py: same-frame social
  geometry and deterministic partner selection.
- src/pig_behavior/classification_v2/features/sequence_windows.py and
  src/pig_behavior/classification_v2/spatial_sequence_export.py: view-local
  aggregation, masks and ordered spatial tensor groups.
- outputs/classification_v2/train_ready_windows/trainer_contract_audit.json:
  current tabular whitelist audit.
- outputs/classification_v2/train_ready_windows/feature_semantics_audit.json:
  current semantic audit, which remains non-final because the exported table
  and the newer semantic contract are not yet reconciled.

## Visual anchor

No separate figure is required at this stage. A compact feature-family table is
more informative than a software-layer diagram. The feature branch may later be
shown as part of the end-to-end framework figure, provided that the final
whitelist and tensor manifest are bound to the same snapshot.

## Open questions and claim boundary

- The final train-ready feature snapshot must bind one feature whitelist, one
  spatial schema, one mask policy and matching content hashes before Section 3
  reports model results.
- The current trainer_contract_audit is structurally valid for its declared
  tabular contract, while the current feature_semantics_audit reports a
  contract/export mismatch. This draft therefore describes the implemented
  feature protocol, not a promoted claim that the exported feature table is
  already the final scientific authority.
- The final spatial contract uses the nearest-partner social representation;
  top-\(K\) and graph variants are outside this evaluated configuration.
- Pen-boundary context is not enabled by the current trainer whitelist and
  requires a separate registered ablation before it can be described as a model
  input.

## Editorial status

The Vietnamese content is written as the working scientific meaning, and the
English text is an original academic rendering of that meaning. No numerical
performance claim is made in this section; all final counts, hashes and
comparative metrics belong to the registered experiment and results sections.
