import time
import os
import subprocess

# Đường dẫn cần xoá
TRASH_FILES = os.path.expanduser("~/.local/share/Trash/files/*")
TRASH_INFO = os.path.expanduser("~/.local/share/Trash/info/*")

# Thư mục uploads của bạn (bạn có thể đổi đường dẫn)
UPLOADS_DIR = os.path.expanduser(
    "/workspace/linevision_worker_ai/uploads/*"
)  # ví dụ ~/uploads
# Nếu là thư mục trong project, dùng:
# UPLOADS_DIR = "/path/to/project/uploads/*"


def clear_trash_and_uploads():
    try:
        # Xóa file trong thư mục uploads
        subprocess.run(f"rm -rf {UPLOADS_DIR}", shell=True)
        # Xóa file trong Trash
        subprocess.run(f"rm -rf {TRASH_FILES}", shell=True)
        subprocess.run(f"rm -rf {TRASH_INFO}", shell=True)

        print("✔ Đã dọn sạch Trash và thư mục uploads!")
    except Exception as e:
        print("❌ Lỗi:", e)


def main():
    while True:
        clear_trash_and_uploads()
        print("⏳ Chờ 5 phút để dọn lại...")
        time.sleep(300)  # 300 giây = 5 phút


if __name__ == "__main__":
    main()
