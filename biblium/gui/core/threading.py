# -*- coding: utf-8 -*-
"""
Threading Utilities
===================
Background task management with progress reporting and cancellation.
"""

from typing import Any, Callable, Dict, Optional
from dataclasses import dataclass
import threading
import queue
import traceback
from concurrent.futures import ThreadPoolExecutor, Future
import time


@dataclass
class TaskResult:
    """Result of a background task"""
    success: bool
    result: Any = None
    error: Optional[Exception] = None
    error_traceback: Optional[str] = None


class BackgroundTask:
    """
    A cancellable background task with progress reporting.
    
    Usage:
        def my_analysis(progress_callback, cancel_check):
            for i in range(100):
                if cancel_check():
                    return None
                # Do work...
                progress_callback(i / 100, f"Processing item {i}")
            return result
        
        task = BackgroundTask(
            func=my_analysis,
            on_complete=lambda r: print(f"Done: {r}"),
            on_progress=lambda p, m: print(f"{p*100:.0f}%: {m}"),
            on_error=lambda e: print(f"Error: {e}"),
        )
        task.start()
        
        # Later...
        task.cancel()
    """
    
    def __init__(
        self,
        func: Callable,
        on_complete: Optional[Callable[[Any], None]] = None,
        on_progress: Optional[Callable[[float, str], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        on_cancel: Optional[Callable[[], None]] = None,
        name: str = "BackgroundTask",
    ):
        """
        Initialize a background task.
        
        Parameters
        ----------
        func : Callable
            Function to run. Should accept (progress_callback, cancel_check) as arguments.
            progress_callback(progress: float, message: str)
            cancel_check() -> bool
        on_complete : Callable, optional
            Called with result when task completes successfully.
        on_progress : Callable, optional
            Called with (progress, message) during execution.
        on_error : Callable, optional
            Called with exception if task fails.
        on_cancel : Callable, optional
            Called if task is cancelled.
        name : str
            Name for the task (for logging).
        """
        self.func = func
        self.on_complete = on_complete
        self.on_progress = on_progress
        self.on_error = on_error
        self.on_cancel = on_cancel
        self.name = name
        
        self._cancelled = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._result: Optional[TaskResult] = None
        self._started = False
        self._finished = False
    
    def start(self):
        """Start the task in a background thread."""
        if self._started:
            raise RuntimeError("Task already started")
        
        self._started = True
        self._thread = threading.Thread(target=self._run, daemon=True, name=self.name)
        self._thread.start()
    
    def _run(self):
        """Run the task."""
        try:
            result = self.func(self._progress_callback, self._cancel_check)
            
            if self._cancelled.is_set():
                self._result = TaskResult(success=False)
                if self.on_cancel:
                    self.on_cancel()
            else:
                self._result = TaskResult(success=True, result=result)
                if self.on_complete:
                    self.on_complete(result)
        
        except Exception as e:
            tb = traceback.format_exc()
            self._result = TaskResult(success=False, error=e, error_traceback=tb)
            if self.on_error:
                self.on_error(e)
        
        finally:
            self._finished = True
    
    def _progress_callback(self, progress: float, message: str = ""):
        """Internal progress callback."""
        if self.on_progress and not self._cancelled.is_set():
            self.on_progress(progress, message)
    
    def _cancel_check(self) -> bool:
        """Check if cancellation was requested."""
        return self._cancelled.is_set()
    
    def cancel(self):
        """Request cancellation of the task."""
        self._cancelled.set()
    
    def is_running(self) -> bool:
        """Check if task is still running."""
        return self._started and not self._finished
    
    def is_cancelled(self) -> bool:
        """Check if task was cancelled."""
        return self._cancelled.is_set()
    
    def wait(self, timeout: float = None) -> Optional[TaskResult]:
        """Wait for task to complete."""
        if self._thread:
            self._thread.join(timeout)
        return self._result
    
    @property
    def result(self) -> Optional[TaskResult]:
        """Get the task result (None if still running)."""
        return self._result


class TaskManager:
    """
    Manages multiple background tasks.
    
    Usage:
        manager = TaskManager(max_workers=4)
        
        # Submit a task
        task_id = manager.submit(
            func=my_analysis,
            on_complete=handle_result,
            on_progress=update_progress,
        )
        
        # Cancel a task
        manager.cancel(task_id)
        
        # Cancel all tasks
        manager.cancel_all()
    """
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self._tasks: Dict[str, BackgroundTask] = {}
        self._task_counter = 0
        self._lock = threading.Lock()
    
    def submit(
        self,
        func: Callable,
        on_complete: Optional[Callable] = None,
        on_progress: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
        on_cancel: Optional[Callable] = None,
        name: str = None,
    ) -> str:
        """
        Submit a task for execution.
        
        Returns
        -------
        str
            Task ID that can be used to cancel or check status.
        """
        with self._lock:
            self._task_counter += 1
            task_id = f"task_{self._task_counter}"
        
        if name is None:
            name = task_id
        
        # Wrap callbacks to remove task on completion
        def wrapped_complete(result):
            self._remove_task(task_id)
            if on_complete:
                on_complete(result)
        
        def wrapped_error(error):
            self._remove_task(task_id)
            if on_error:
                on_error(error)
        
        def wrapped_cancel():
            self._remove_task(task_id)
            if on_cancel:
                on_cancel()
        
        task = BackgroundTask(
            func=func,
            on_complete=wrapped_complete,
            on_progress=on_progress,
            on_error=wrapped_error,
            on_cancel=wrapped_cancel,
            name=name,
        )
        
        with self._lock:
            self._tasks[task_id] = task
        
        task.start()
        return task_id
    
    def _remove_task(self, task_id: str):
        """Remove a task from the manager."""
        with self._lock:
            self._tasks.pop(task_id, None)
    
    def cancel(self, task_id: str):
        """Cancel a specific task."""
        with self._lock:
            task = self._tasks.get(task_id)
        if task:
            task.cancel()
    
    def cancel_all(self):
        """Cancel all running tasks."""
        with self._lock:
            tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
    
    def is_running(self, task_id: str) -> bool:
        """Check if a task is running."""
        with self._lock:
            task = self._tasks.get(task_id)
        return task.is_running() if task else False
    
    def get_running_tasks(self) -> list:
        """Get list of running task IDs."""
        with self._lock:
            return [tid for tid, task in self._tasks.items() if task.is_running()]
    
    def has_running_tasks(self) -> bool:
        """Check if any tasks are running."""
        return len(self.get_running_tasks()) > 0
    
    def wait_all(self, timeout: float = None):
        """Wait for all tasks to complete."""
        with self._lock:
            tasks = list(self._tasks.values())
        
        deadline = time.time() + timeout if timeout else None
        
        for task in tasks:
            if deadline:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                task.wait(remaining)
            else:
                task.wait()


def run_in_thread(func: Callable, callback: Callable = None, error_callback: Callable = None):
    """
    Simple helper to run a function in a background thread.
    
    Parameters
    ----------
    func : Callable
        Function to run (no arguments).
    callback : Callable, optional
        Called with result on success.
    error_callback : Callable, optional
        Called with exception on failure.
    """
    def wrapper():
        try:
            result = func()
            if callback:
                callback(result)
        except Exception as e:
            if error_callback:
                error_callback(e)
    
    thread = threading.Thread(target=wrapper, daemon=True)
    thread.start()
    return thread


class ThrottledCallback:
    """
    Throttle callback invocations to prevent UI flooding.
    
    Usage:
        throttled = ThrottledCallback(update_ui, min_interval=0.1)
        
        for i in range(1000):
            throttled(progress=i/1000)  # Only calls update_ui every 0.1s
    """
    
    def __init__(self, callback: Callable, min_interval: float = 0.05):
        self.callback = callback
        self.min_interval = min_interval
        self._last_call = 0
        self._pending_args = None
        self._pending_kwargs = None
    
    def __call__(self, *args, **kwargs):
        now = time.time()
        
        if now - self._last_call >= self.min_interval:
            self._last_call = now
            self.callback(*args, **kwargs)
        else:
            # Store for potential final call
            self._pending_args = args
            self._pending_kwargs = kwargs
    
    def flush(self):
        """Force call with pending arguments."""
        if self._pending_args is not None or self._pending_kwargs is not None:
            self.callback(
                *(self._pending_args or ()),
                **(self._pending_kwargs or {})
            )
            self._pending_args = None
            self._pending_kwargs = None
