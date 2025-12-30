# =============================================================================
# GPU QUEUE MANAGER - Quản lý hàng đợi GPU để tránh OOM
# =============================================================================
# Giới hạn số request xử lý đồng thời trên GPU
# Hỗ trợ ưu tiên VAR: Khi có VAR request, các task khác tạm dừng

import threading
import queue
import time
from dataclasses import dataclass, field
from typing import Callable, Any, Optional
from enum import Enum, IntEnum
import traceback


class TaskStatus(Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"  # Tạm dừng do VAR đang chạy


class TaskPriority(IntEnum):
    """Độ ưu tiên task - số nhỏ hơn = ưu tiên cao hơn"""
    VAR = 0          # Cao nhất - VAR luôn chạy trước
    HIGH = 1         # Ưu tiên cao
    NORMAL = 2       # Bình thường
    LOW = 3          # Thấp


@dataclass(order=True)
class PrioritizedTask:
    """Wrapper cho task để sử dụng với PriorityQueue"""
    priority: int
    counter: int = field(compare=True)  # Để đảm bảo FIFO trong cùng priority
    task: Any = field(compare=False)


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
    priority: TaskPriority = TaskPriority.NORMAL
    is_var: bool = False  # Đánh dấu đây là VAR task

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()


class GPUQueueManager:
    """
    Singleton manager để quản lý hàng đợi GPU tasks.
    Chỉ cho phép 1 task chạy trên GPU tại một thời điểm.

    Hỗ trợ ưu tiên VAR:
    - Khi có VAR request, các task khác sẽ chờ VAR hoàn thành trước
    - VAR tasks được đưa vào đầu queue (priority=0)
    - Các task đang chờ sẽ không bắt đầu nếu có VAR đang chờ
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
        # Sử dụng PriorityQueue để hỗ trợ ưu tiên
        self.task_queue = queue.PriorityQueue()
        self.task_counter = 0  # Counter để đảm bảo FIFO trong cùng priority
        self.counter_lock = threading.Lock()

        self.active_tasks = 0
        self.active_lock = threading.Lock()
        self.tasks = {}  # task_id -> GPUTask
        self.tasks_lock = threading.Lock()

        # VAR priority system
        self.var_active = False  # True khi có VAR đang chạy
        self.var_waiting = 0  # Số VAR đang chờ trong queue
        self.var_lock = threading.Lock()
        self.var_event = threading.Event()  # Event để đánh thức workers khi VAR done
        self.var_event.set()  # Mặc định là set (không block)

        # Condition để pause non-VAR tasks
        self.pause_condition = threading.Condition()

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
        print(f"[GPU Queue] Initialized with max_concurrent={max_concurrent}, VAR priority enabled")

    def _should_wait_for_var(self, task: GPUTask) -> bool:
        """Kiểm tra xem task có cần chờ VAR không"""
        if task.is_var:
            return False  # VAR tasks không chờ

        with self.var_lock:
            # Nếu có VAR đang chạy hoặc đang chờ, các task khác phải chờ
            return self.var_active or self.var_waiting > 0

    def _wait_for_var_complete(self, task: GPUTask):
        """Chờ cho VAR hoàn thành trước khi bắt đầu task"""
        if not self._should_wait_for_var(task):
            return

        worker_name = threading.current_thread().name
        print(f"[GPU Queue] {worker_name} pausing task {task.task_id} - waiting for VAR to complete...")
        task.status = TaskStatus.PAUSED

        # Chờ cho đến khi không còn VAR
        while self._should_wait_for_var(task) and self.running:
            self.var_event.wait(timeout=1.0)

        if self.running:
            print(f"[GPU Queue] {worker_name} resuming task {task.task_id} - VAR completed")
            task.status = TaskStatus.QUEUED

    def _on_var_start(self, task: GPUTask):
        """Gọi khi bắt đầu xử lý VAR task"""
        if not task.is_var:
            return

        with self.var_lock:
            self.var_active = True
            if self.var_waiting > 0:
                self.var_waiting -= 1
            self.var_event.clear()  # Block non-VAR tasks

        print(f"[GPU Queue] VAR task {task.task_id} started - other tasks paused")

    def _on_var_complete(self, task: GPUTask):
        """Gọi khi VAR task hoàn thành"""
        if not task.is_var:
            return

        with self.var_lock:
            self.var_active = False
            # Nếu không còn VAR waiting, cho phép các task khác chạy
            if self.var_waiting == 0:
                self.var_event.set()  # Unblock non-VAR tasks

        print(f"[GPU Queue] VAR task {task.task_id} completed - resuming other tasks")

    def _worker(self):
        """Worker thread để xử lý tasks từ queue"""
        worker_name = threading.current_thread().name
        print(f"[GPU Queue] {worker_name} started and waiting for tasks...")

        while self.running:
            try:
                # Lấy task từ queue (blocking với timeout)
                try:
                    prioritized_task = self.task_queue.get(timeout=1.0)
                    task = prioritized_task.task
                except queue.Empty:
                    continue

                # Chờ nếu có VAR đang chạy và task này không phải VAR
                self._wait_for_var_complete(task)

                if not self.running:
                    break

                with self.active_lock:
                    self.active_tasks += 1

                try:
                    # Đánh dấu bắt đầu VAR nếu là VAR task
                    self._on_var_start(task)

                    # Cập nhật status
                    task.status = TaskStatus.PROCESSING
                    task.started_at = time.time()

                    priority_name = TaskPriority(task.priority).name if isinstance(task.priority, int) else task.priority.name
                    print(f"[GPU Queue] {worker_name} starting task {task.task_id} (priority={priority_name}), remaining in queue: {self.task_queue.qsize()}")

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
                    # Đánh dấu VAR hoàn thành
                    self._on_var_complete(task)

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
        error_callback: Callable = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        is_var: bool = False
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
            priority: Độ ưu tiên của task (VAR > HIGH > NORMAL > LOW)
            is_var: True nếu đây là VAR task (sẽ tự động set priority=VAR)

        Returns:
            GPUTask object
        """
        if kwargs is None:
            kwargs = {}

        # Nếu là VAR, luôn set priority cao nhất
        if is_var:
            priority = TaskPriority.VAR

        task = GPUTask(
            task_id=task_id,
            func=func,
            args=args,
            kwargs=kwargs,
            callback=callback,
            error_callback=error_callback,
            priority=priority,
            is_var=is_var
        )

        with self.tasks_lock:
            self.tasks[task_id] = task

        # Nếu là VAR, tăng counter var_waiting
        if is_var:
            with self.var_lock:
                self.var_waiting += 1
                self.var_event.clear()  # Block non-VAR tasks ngay lập tức
            print(f"[GPU Queue] VAR task {task_id} queued - pausing other tasks")

        # Tạo PrioritizedTask để đưa vào PriorityQueue
        with self.counter_lock:
            counter = self.task_counter
            self.task_counter += 1

        prioritized_task = PrioritizedTask(
            priority=int(priority),
            counter=counter,
            task=task
        )
        self.task_queue.put(prioritized_task)

        queue_size = self.task_queue.qsize()
        priority_name = priority.name if isinstance(priority, TaskPriority) else TaskPriority(priority).name
        print(f"[GPU Queue] Task {task_id} queued (priority={priority_name}), position: {queue_size}")

        return task

    def submit_var(
        self,
        task_id: str,
        func: Callable,
        args: tuple = (),
        kwargs: dict = None,
        callback: Callable = None,
        error_callback: Callable = None
    ) -> GPUTask:
        """
        Submit một VAR task với priority cao nhất.
        Các task khác sẽ tạm dừng cho đến khi VAR hoàn thành.

        Args:
            task_id: ID duy nhất cho task
            func: Function cần chạy
            args: Arguments cho function
            kwargs: Keyword arguments cho function
            callback: Callback khi hoàn thành
            error_callback: Callback khi lỗi

        Returns:
            GPUTask object
        """
        return self.submit(
            task_id=task_id,
            func=func,
            args=args,
            kwargs=kwargs,
            callback=callback,
            error_callback=error_callback,
            priority=TaskPriority.VAR,
            is_var=True
        )

    def get_task(self, task_id: str) -> Optional[GPUTask]:
        """Lấy thông tin task theo ID"""
        with self.tasks_lock:
            return self.tasks.get(task_id)

    def get_queue_status(self) -> dict:
        """Lấy trạng thái của queue"""
        with self.active_lock:
            active = self.active_tasks

        with self.var_lock:
            var_active = self.var_active
            var_waiting = self.var_waiting

        with self.tasks_lock:
            total_tasks = len(self.tasks)
            queued = sum(1 for t in self.tasks.values() if t.status == TaskStatus.QUEUED)
            processing = sum(1 for t in self.tasks.values() if t.status == TaskStatus.PROCESSING)
            completed = sum(1 for t in self.tasks.values() if t.status == TaskStatus.COMPLETED)
            failed = sum(1 for t in self.tasks.values() if t.status == TaskStatus.FAILED)
            paused = sum(1 for t in self.tasks.values() if t.status == TaskStatus.PAUSED)

            # Đếm theo priority
            priority_counts = {p.name: 0 for p in TaskPriority}
            for t in self.tasks.values():
                if t.status in (TaskStatus.QUEUED, TaskStatus.PROCESSING, TaskStatus.PAUSED):
                    priority_name = t.priority.name if isinstance(t.priority, TaskPriority) else TaskPriority(t.priority).name
                    priority_counts[priority_name] += 1

        return {
            "max_concurrent": self.max_concurrent,
            "active_tasks": active,
            "queue_size": self.task_queue.qsize(),
            "total_tasks": total_tasks,
            "queued": queued,
            "processing": processing,
            "completed": completed,
            "failed": failed,
            "paused": paused,
            "var_status": {
                "active": var_active,
                "waiting": var_waiting,
                "other_tasks_paused": var_active or var_waiting > 0
            },
            "priority_counts": priority_counts
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


def get_gpu_queue(max_concurrent: int = 2) -> GPUQueueManager:
    """Lấy GPU queue instance (lazy initialization)

    Args:
        max_concurrent: Số task tối đa chạy đồng thời (default: 2)
    """
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
