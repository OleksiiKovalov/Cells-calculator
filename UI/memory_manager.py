import os
import gc
import time
import psutil
import threading
from datetime import datetime

from UI.errorhandling import app_logger


class MemoryManager:
    """
    Universal memory manager for:
    - Python garbage collection
    - NumPy / OpenCV buffers
    - PyTorch CPU/GPU memory
    - System RAM monitoring
    - Logging
    - Automatic cleaning (threshold / timer)
    """

    def __init__(self, log_file="memory.log", verbose=True):
        self.verbose = verbose
        self.log_file = log_file

    # --------------------------------------------------
    #   LOGGING
    # --------------------------------------------------
    def log(self, text):
        app_logger().info(text)
        # timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # line = f"[{timestamp}] {text}"
        # with open(self.log_file, "a", encoding="utf-8") as f:
        #     f.write(line + "\n")
        # if self.verbose:
        #     print(line)

    # --------------------------------------------------
    #   SYSTEM MEMORY
    # --------------------------------------------------
    def system_ram_used_mb(self):
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 ** 2)

    def system_ram_percent(self):
        return psutil.virtual_memory().percent

    # --------------------------------------------------
    #   BASIC CLEANUP
    # --------------------------------------------------
    def clear_python(self):
        before = self.system_ram_used_mb()
        gc.collect()
        after = self.system_ram_used_mb()
        self.log(f"[Python GC] {before:.2f} → {after:.2f} MB")

    def clear_numpy(self, locals_dict=None, globals_dict=None):
        import numpy as np

        def clear(d):
            if d:
                for k in list(d.keys()):
                    v = d.get(k)
                    if isinstance(v, np.ndarray):
                        d[k] = None
                        del v

        before = self.system_ram_used_mb()
        clear(locals_dict)
        clear(globals_dict)
        gc.collect()
        after = self.system_ram_used_mb()

        self.log(f"[NumPy] {before:.2f} → {after:.2f} MB")

    def clear_opencv(self):
        import cv2
        before = self.system_ram_used_mb()
        try:
            cv2.destroyAllWindows()
        except:
            pass
        gc.collect()
        after = self.system_ram_used_mb()
        self.log(f"[OpenCV] {before:.2f} → {after:.2f} MB")

    def clear_torch(self, aggressive=False):
        try:
            import torch
        except:
            self.log("[PyTorch] Not installed")
            return

        before = self.system_ram_used_mb()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        if aggressive:
            for obj in gc.get_objects():
                try:
                    import torch
                    if torch.is_tensor(obj) or \
                       (hasattr(obj, "data") and torch.is_tensor(obj.data)):
                        del obj
                except:
                    pass

        gc.collect()
        after = self.system_ram_used_mb()
        self.log(f"[PyTorch] {before:.2f} → {after:.2f} MB")

    # --------------------------------------------------
    #   CLEAN ALL
    # --------------------------------------------------
    def clean_all(self, locals_dict=None, globals_dict=None):
        self.log("------ CLEAN START ------")
        self.log(f"Before: {self.system_ram_used_mb():.2f} MB")

        self.clear_python()
        self.clear_numpy(locals_dict, globals_dict)
        self.clear_opencv()
        self.clear_torch(aggressive=True)

        self.log(f"After:  {self.system_ram_used_mb():.2f} MB")
        self.log("------ CLEAN END ------")

    # --------------------------------------------------
    #   AUTO CLEAN: TIMER
    # --------------------------------------------------
    def start_timer_cleanup(self, interval_sec=60, locals_dict=None, globals_dict=None):
        """
        Clean every N seconds in background thread.
        """

        def loop():
            while True:
                time.sleep(interval_sec)
                self.clean_all(locals_dict, globals_dict)

        t = threading.Thread(target=loop, daemon=True)
        t.start()
        self.log(f"[AutoClean] Started timer every {interval_sec} sec")

    # --------------------------------------------------
    #   AUTO CLEAN: THRESHOLD
    # --------------------------------------------------
    def start_threshold_cleanup(self, percent_limit=80, check_interval=5,
                                locals_dict=None, globals_dict=None):
        """
        Clean memory when system RAM exceeds threshold.
        """

        def loop():
            while True:
                time.sleep(check_interval)
                used = self.system_ram_percent()
                if used >= percent_limit:
                    self.log(f"[Threshold] RAM={used}%, cleaning triggered.")
                    self.clean_all(locals_dict, globals_dict)

        t = threading.Thread(target=loop, daemon=True)
        t.start()
        self.log(f"[AutoClean] Threshold {percent_limit}% started")



# --------------------------------------------------
#   GPU WATCHDOG 
# --------------------------------------------------

class GPUMemoryWatchdog:
    """Monitors GPU VRAM and triggers PyTorch cleanup when needed."""

    def __init__(self, limit_percent=80, interval=3):
        self.limit = limit_percent
        self.interval = interval

    def start(self):
        try:
            import torch
        except:
            print("[GPU Watchdog] PyTorch not installed")
            return

        def loop():
            while True:
                time.sleep(self.interval)
                if torch.cuda.is_available():
                    stats = torch.cuda.memory_stats()
                    usage = stats["reserved_bytes.all"] / torch.cuda.get_device_properties(0).total_memory
                    usage *= 100

                    if usage > self.limit:
                        print(f"[GPU Watchdog] GPU VRAM {usage:.1f}% > {self.limit}%, cleaning.")
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()

        threading.Thread(target=loop, daemon=True).start()
        print(f"[GPU Watchdog] Started at {self.limit}% threshold")

