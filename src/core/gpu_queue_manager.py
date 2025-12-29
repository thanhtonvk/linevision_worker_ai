# =============================================================================
# GPU QUEUE MANAGER - Quản lý hàng đợi GPU để tránh OOM
# =============================================================================
# Giới hạn số request xử lý đồng thời trên GPU

import threading
import queue
import time
from dataclasses import dataclass
from typing import Callable, Any, Optional
from enum import Enum
import traceback


class TaskStatus(Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class GPUTask:
    """Một task cần xử lý trên GPU"""
    task_id: str
    func: Callable
    args: tuple
    kwargs: dict
    callback: Optional[Callable] = None
    error_callback: Optional[Callable] = None
    status: TaskStatus = TaskStatus.QUEUED
    result: Any = None
    error: Optional[str] = None
    created_at: float = None
    started_at: float = None
    completed_at: float = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()


class GPUQueueManager:
    """
    Singleton manager để quản lý hàng đợi GPU tasks.
    Chỉ cho phép 1 task chạy trên GPU tại một thời điểm.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, max_concurrent: int = 1):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, max_concurrent: int = 1):
        if self._initialized:
            return

        self.max_concurrent = max_concurrent
        self.task_queue = queue.Queue()
        self.active_tasks = 0
        self.active_lock = threading.Lock()
        self.tasks = {}  # task_id -> GPUTask
        self.tasks_lock = threading.Lock()

        # Semaphore để giới hạn concurrent tasks
        self.semaphore = threading.Semaphore(max_concurrent)

        # Start worker threads
        self.workers = []
        self.running = True
        for i in range(max_concurrent):
            worker = threading.Thread(target=self._worker, daemon=True, name=f"GPUWorker-{i}")
            worker.start()
            self.workers.append(worker)

        self._initialized = True
        print(f"[GPU Queue] Initialized with max_concurrent={max_concurrent}")

    def _worker(self):
        """Worker thread để xử lý tasks từ queue"""
        worker_name = threading.current_thread().name
        print(f"[GPU Queue] {worker_name} started and waiting for tasks...")

        while self.running:
            try:
                # Lấy task từ queue (blocking với timeout)
                try:
                    task = self.task_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                with self.active_lock:
                    self.active_tasks += 1

                try:
                    # Cập nhật status
                    task.status = TaskStatus.PROCESSING
                    task.started_at = time.time()

                    print(f"[GPU Queue] {worker_name} starting task {task.task_id}, remaining in queue: {self.task_queue.qsize()}")

                    # Chạy task
                    result = task.func(*task.args, **task.kwargs)

                    # Hoàn thành
                    task.status = TaskStatus.COMPLETED
                    task.result = result
                    task.completed_at = time.time()

                    duration = task.completed_at - task.started_at
                    print(f"[GPU Queue] Task {task.task_id} completed in {duration:.1f}s")

                    # Gọi callback nếu có
                    if task.callback:
                        try:
                            task.callback(result)
                        except Exception as cb_error:
                            print(f"[GPU Queue] Callback error for {task.task_id}: {cb_error}")

                except Exception as e:
                    # Lỗi
                    task.status = TaskStatus.FAILED
                    task.error = str(e)
                    task.completed_at = time.time()

                    print(f"[GPU Queue] Task {task.task_id} failed: {e}")
                    print(traceback.format_exc())

                    # Gọi error callback nếu có
                    if task.error_callback:
                        try:
                            task.error_callback(e)
                        except Exception as cb_error:
                            print(f"[GPU Queue] Error callback error for {task.task_id}: {cb_error}")

                finally:
                    with self.active_lock:
                        self.active_tasks -= 1

                    self.task_queue.task_done()

            except Exception as e:
                print(f"[GPU Queue] {worker_name} error: {e}")
                print(traceback.format_exc())

    def submit(
        self,
        task_id: str,
        func: Callable,
        args: tuple = (),
        kwargs: dict = None,
        callback: Callable = None,
        error_callback: Callable = None
    ) -> GPUTask:
        """
        Submit một task vào queue

        Args:
            task_id: ID duy nhất cho task
            func: Function cần chạy
            args: Arguments cho function
            kwargs: Keyword arguments cho function
            callback: Callback khi hoàn thành (nhận result)
            error_callback: Callback khi lỗi (nhận exception)

        Returns:
            GPUTask object
        """
        if kwargs is None:
            kwargs = {}

        task = GPUTask(
            task_id=task_id,
            func=func,
            args=args,
            kwargs=kwargs,
            callback=callback,
            error_callback=error_callback
        )

        with self.tasks_lock:
            self.tasks[task_id] = task

        self.task_queue.put(task)

        queue_size = self.task_queue.qsize()
        print(f"[GPU Queue] Task {task_id} queued, position: {queue_size}")

        return task

    def get_task(self, task_id: str) -> Optional[GPUTask]:
        """Lấy thông tin task theo ID"""
        with self.tasks_lock:
            return self.tasks.get(task_id)

    def get_queue_status(self) -> dict:
        """Lấy trạng thái của queue"""
        with self.active_lock:
            active = self.active_tasks

        with self.tasks_lock:
            total_tasks = len(self.tasks)
            queued = sum(1 for t in self.tasks.values() if t.status == TaskStatus.QUEUED)
            processing = sum(1 for t in self.tasks.values() if t.status == TaskStatus.PROCESSING)
            completed = sum(1 for t in self.tasks.values() if t.status == TaskStatus.COMPLETED)
            failed = sum(1 for t in self.tasks.values() if t.status == TaskStatus.FAILED)

        return {
            "max_concurrent": self.max_concurrent,
            "active_tasks": active,
            "queue_size": self.task_queue.qsize(),
            "total_tasks": total_tasks,
            "queued": queued,
            "processing": processing,
            "completed": completed,
            "failed": failed
        }

    def cleanup_old_tasks(self, max_age_hours: int = 24):
        """Xóa các tasks cũ đã hoàn thành hoặc thất bại"""
        now = time.time()
        max_age_seconds = max_age_hours * 3600

        with self.tasks_lock:
            to_remove = []
            for task_id, task in self.tasks.items():
                if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                    if task.completed_at and (now - task.completed_at) > max_age_seconds:
                        to_remove.append(task_id)

            for task_id in to_remove:
                del self.tasks[task_id]

            if to_remove:
                print(f"[GPU Queue] Cleaned up {len(to_remove)} old tasks")

    def shutdown(self):
        """Shutdown queue manager"""
        self.running = False
        for worker in self.workers:
            worker.join(timeout=5.0)
        print("[GPU Queue] Shutdown complete")


# Global instance - lazy initialization
_gpu_queue_instance = None
_gpu_queue_lock = threading.Lock()


def get_gpu_queue(max_concurrent: int = 1) -> GPUQueueManager:
    """Lấy GPU queue instance (lazy initialization)"""
    global _gpu_queue_instance
    if _gpu_queue_instance is None:
        with _gpu_queue_lock:
            if _gpu_queue_instance is None:
                _gpu_queue_instance = GPUQueueManager(max_concurrent=max_concurrent)
    return _gpu_queue_instance


# Backward compatibility - tạo property-like access
class _GPUQueueProxy:
    """Proxy để lazy load GPU queue"""
    def __getattr__(self, name):
        return getattr(get_gpu_queue(), name)

gpu_queue = _GPUQueueProxy()
