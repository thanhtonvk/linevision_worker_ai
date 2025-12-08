import time
import os
import subprocess

TRASH_FILES = os.path.expanduser("~/.local/share/Trash/files/*")
TRASH_INFO = os.path.expanduser("~/.local/share/Trash/info/*")


def clear_trash():
    try:
        # Xóa file trong Trash
        subprocess.run(f"rm -rf {TRASH_FILES}", shell=True)
        subprocess.run(f"rm -rf {TRASH_INFO}", shell=True)
        print("Đã dọn sạch thùng rác!")
    except Exception as e:
        print("Lỗi:", e)


def main():
    while True:
        clear_trash()
        print("Chờ 5 phút để dọn lại...")
        time.sleep(300)  # 300 giây = 5 phút


if __name__ == "__main__":
    main()
