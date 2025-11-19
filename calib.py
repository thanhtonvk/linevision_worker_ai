import cv2
import json
from utils import distored_image
import time
print("🎯 HƯỚNG DẪN SỬ DỤNG:")
print("- Click chuột trái để chọn điểm")
print("- Nhấn 'r' để reset và chọn lại")
print("- Nhấn 'q' để kết thúc")
print("- Tự động dừng khi đủ 16 điểm")
print("=" * 50)
# Địa chỉ RTSP
# http://ngocvu1.cameraddns.net:88/doc/index.html#/preview
rtsp_url = "crop_video\part_000.mp4"
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("❌ Không thể kết nối tới camera.")
else:
    print("✅ Đang phát video. Nhấn phím 'c' để chụp và dừng.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Không đọc được frame từ camera.")
            break

        cv2.imshow("Camera", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            # Lưu frame hiện tại
            cv2.imwrite("capture.jpg", frame)
            print("✅ Đã lưu ảnh thành capture.jpg")
            break

    # Dọn dẹp
    cap.release()
    cv2.destroyAllWindows()
points = []

# Callback sự kiện chuột
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 16:  # Chỉ cho phép chọn tối đa 16 điểm
            points.append({"x": x, "y": y})
            print(f"Điểm {len(points)}: ({x}, {y})")
        else:
            print("Đã đủ 16 điểm! Nhấn 'q' để kết thúc hoặc 'r' để reset.")

# Load ảnh
image = cv2.imread("capture.jpg")
image = distored_image(image)
if image is None:
    raise Exception("Không load được ảnh capture.jpg")

clone = image.copy()

# Khởi tạo cửa sổ và set callback chuột
cv2.namedWindow("Select Points")
cv2.setMouseCallback("Select Points", mouse_callback)

print("🚀 Bắt đầu chọn 16 điểm theo thứ tự...")
print("📝 Click chuột trái để chọn từng điểm")

while True:
    temp = clone.copy()
    # Vẽ các điểm và nối lại
    for i, p in enumerate(points):
        cv2.circle(temp, (p["x"], p["y"]), 5, (0, 0, 255), -1)
        if i > 0:
            cv2.line(temp, (points[i - 1]["x"], points[i - 1]["y"]), (p["x"], p["y"]), (255, 0, 0), 2)

    # Hiển thị số điểm đã chọn
    cv2.putText(temp, f"Points: {len(points)}/16", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Select Points", temp)

    key = cv2.waitKey(1)
    if key == ord('q'):  # Nhấn q để kết thúc chọn
        break
    elif key == ord('r'):  # Nhấn r để reset
        points.clear()
        print("🔄 Đã reset, chọn lại từ đầu!")
    elif len(points) >= 16:  # Dừng khi đủ 16 điểm
        print(f"✅ Đã chọn đủ {len(points)} điểm!")
        break
cv2.destroyAllWindows()

print("\n" + "=" * 50)
print("🎉 KẾT QUẢ:")
print(f"📊 Tổng số điểm đã chọn: {len(points)}")
if len(points) == 16:
    print("✅ Hoàn thành! Đã chọn đủ 16 điểm.")
else:
    print("⚠️  Chưa đủ 16 điểm.")
print("=" * 50)
print(json.dumps(points, indent=4))
