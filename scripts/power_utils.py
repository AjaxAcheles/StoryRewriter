import subprocess
import logging

logger = logging.getLogger(__name__)

class WindowsCpuThrottler:
    """
    Context manager to temporarily limit Windows CPU Maximum Processor State.
    Setting this to 99% disables Turbo Boost on most Intel/AMD CPUs,
    significantly reducing heat with minimal performance loss.
    """
    def __init__(self, limit_percent=99, restore_percent=100):
        self.limit = limit_percent
        self.restore = restore_percent

    def _set_state(self, percent):
        try:
            # GUIDs: SUB_PROCESSOR = Processor Power Mgmt, PROCTHROTTLEMAX = Max State
            commands = [
                # Set for AC (Plugged In)
                f"powercfg -setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX {percent}",
                # Set for DC (Battery)
                f"powercfg -setdcvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX {percent}",
                # Apply changes
                "powercfg -setactive SCHEME_CURRENT"
            ]
            
            for cmd in commands:
                # Run silently
                subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
            return True
        except Exception as e:
            logger.warning(f"[Power] Throttling failed: {e}")
            return False

    def __enter__(self):
        logger.info(f"[Power] Throttling CPU to {self.limit}% (Disable Turbo)")
        self._set_state(self.limit)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.info(f"[Power] Restoring CPU to {self.restore}%")
        self._set_state(self.restore)