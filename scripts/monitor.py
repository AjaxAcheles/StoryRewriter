import time
import logging
import wmi
import pythoncom

logger = logging.getLogger(__name__)

class TemperatureGuard:
    """
    Monitors system temperatures via LibreHardwareMonitor.
    Pauses execution if safe limits are exceeded.
    """
    
    def __init__(self, config):
        self.enabled = config['monitoring'].get('enabled', True)
        self.max_temp = config['monitoring'].get('max_temp_celsius', 85)
        self.resume_temp = config['monitoring'].get('resume_temp_celsius', 75)
        self.check_interval = config['monitoring'].get('check_interval_seconds', 3)
        self.min_valid_temp = config['monitoring'].get('min_valid_temp_celsius', 20)
        self.cooldown_sensor_error = config['monitoring'].get('cooldown_seconds_in_case_of_sensor_error', 15)
        
        self._wmi = None
        self._use_lhm = True  # Track active sensor source
        
        if self.enabled:
            self._init_wmi()

    def _init_wmi(self):
        """Initialize connection to LibreHardwareMonitor WMI."""
        try:
            pythoncom.CoInitialize()
            
            # Attempt connection to LibreHardwareMonitor namespace
            try:
                self._wmi = wmi.WMI(namespace="root\\LibreHardwareMonitor")
                if self._get_max_temp() > 0:
                    logger.info("[Temp] Connected to LibreHardwareMonitor")
                    return
            except Exception:
                pass

            # Fallback to standard Windows WMI
            logger.warning("[Temp] LHM not found; falling back to Windows WMI")
            logger.warning("[Temp] Solution: Run LibreHardwareMonitor as Admin")
            self._wmi = wmi.WMI(namespace="root\\wmi")
            self._use_lhm = False
            
        except Exception as e:
            logger.warning(f"[Temp] Init failed: {e}. Running without protection.")
            self._wmi = None

    def _get_max_temp(self):
        """Scan thermal sensors and return the highest temperature found."""
        if not self._wmi:
            return 0.0

        max_found = 0.0
        try:
            if self._use_lhm:
                # Query LHM for Temperature sensors
                sensors = self._wmi.Sensor(SensorType="Temperature")
                for sensor in sensors:
                    if sensor.Value and sensor.Value > max_found:
                        max_found = sensor.Value
            else:
                # Query Windows MSAcpi (Deci-Kelvin to Celsius)
                thermal_zones = self._wmi.MSAcpi_ThermalZoneTemperature()
                for zone in thermal_zones:
                    current_c = (zone.CurrentTemperature - 2732) / 10.0
                    if current_c > max_found:
                        max_found = current_c
                    
        except Exception:
            # Handle cases where LHM closes during execution
            return 0.0

        return max_found

    def check_and_pause(self):
        """
        Check temps. If too hot, enter a sleep loop until cooled down.
        """
        if not self.enabled or not self._wmi:
            return

        current_temp = self._get_max_temp()
        
        # 1. Handle Sensor Read Failures
        if current_temp == 0.0 and self._use_lhm:
             logger.warning("[Temp] Read failed. Is LHM running?")
             return

        # 2. Handle Invalid/Noise Readings
        if current_temp < self.min_valid_temp: 
            logger.info(f"[Temp] Invalid read ({current_temp}°C); waiting {self.cooldown_sensor_error}s")
            time.sleep(self.cooldown_sensor_error)
            return
        
        # 3. Log Status (Safe)
        if current_temp < self.max_temp:
            logger.info(f"[Temp] Current: {current_temp:.1f}°C (Safe)")

        # 4. Handle Overheating
        if current_temp >= self.max_temp:
            logger.warning(f"[Temp] OVERHEAT: {current_temp:.1f}°C > {self.max_temp}°C. Pausing.")
            
            while current_temp > self.resume_temp:
                time.sleep(self.check_interval)
                current_temp = self._get_max_temp()
                
                # Exit loop if sensors vanish
                if current_temp == 0.0:
                    logger.warning("[Temp] Sensors lost during cooldown; resuming")
                    break

                print(f"\r    Cooling down... {current_temp:.1f}°C (Target: {self.resume_temp}°C)", end="", flush=True)
            
            print() 
            logger.info(f"[Temp] Cooled to {current_temp:.1f}°C. Resuming.")