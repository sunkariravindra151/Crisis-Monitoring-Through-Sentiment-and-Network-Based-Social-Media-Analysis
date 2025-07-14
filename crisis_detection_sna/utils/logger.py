"""
CrisisDetect+ Logging System
Centralized logging for all components
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

from utils.config import LOGGING_CONFIG, PROJECT_ROOT

class CrisisLogger:
    """Centralized logger for CrisisDetect+ system"""
    
    def __init__(self, name: str = "CrisisDetect+", log_file: Optional[str] = None):
        self.name = name
        self.logger = logging.getLogger(name)
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_logger(log_file)
    
    def _setup_logger(self, log_file: Optional[str] = None):
        """Setup logger with console and file handlers"""
        
        # Set logging level
        level = getattr(logging, LOGGING_CONFIG['level'].upper())
        self.logger.setLevel(level)
        
        # Create formatter
        formatter = logging.Formatter(LOGGING_CONFIG['format'])
        
        # Console handler with safe encoding
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            log_path = Path(log_file)
        else:
            log_path = LOGGING_CONFIG['file_path']
        
        # Ensure log directory exists
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def _clean_message(self, message: str) -> str:
        """Clean message of problematic Unicode characters"""
        try:
            # Remove or replace common emoji and Unicode characters that cause issues
            import re
            # Replace emojis and special Unicode with text equivalents
            replacements = {
                '🚀': '[INIT]',
                '✅': '[OK]',
                '❌': '[ERROR]',
                '⚠️': '[WARNING]',
                '📊': '[DATA]',
                '🔄': '[PROCESSING]',
                '🌐': '[NETWORK]',
                '👥': '[USERS]',
                '🔗': '[LINKS]',
                '📦': '[BATCH]',
                '🔍': '[ANALYSIS]',
                '📝': '[NLP]',
                '🚨': '[CRISIS]',
                '🔔': '[ALERT]',
                '😊': '[EMOTION]',
                '🎯': '[TARGET]',
                '💻': '[CPU]',
                '📈': '[PROGRESS]',
                '🏆': '[RESULT]',
                '💡': '[INSIGHT]',
                '📁': '[FILE]',
                '🛑': '[STOP]',
                '⏰': '[TIME]',
                '🌊': '[STREAM]',
                '🔧': '[CONFIG]',
                '📋': '[LIST]',
                '🎉': '[SUCCESS]',
                '💾': '[SAVE]',
                '🔴': '[HIGH]',
                '🟡': '[MEDIUM]',
                '🟢': '[LOW]',
                '🔵': '[SAFE]',
                '⚪': '[UNKNOWN]'
            }

            cleaned = message
            for emoji, replacement in replacements.items():
                cleaned = cleaned.replace(emoji, replacement)

            # Remove any remaining problematic Unicode characters
            cleaned = re.sub(r'[^\x00-\x7F]+', '[UNICODE]', cleaned)

            return cleaned
        except:
            # If cleaning fails, return a safe version
            return str(message).encode('ascii', 'ignore').decode('ascii')

    def info(self, message: str, **kwargs):
        """Log info message with safe encoding"""
        try:
            clean_msg = self._clean_message(message)
            self.logger.info(clean_msg, **kwargs)
        except Exception:
            self.logger.info(f"[LOG] {str(message)[:100]}...", **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message with safe encoding"""
        try:
            clean_msg = self._clean_message(message)
            self.logger.warning(clean_msg, **kwargs)
        except Exception:
            self.logger.warning(f"[WARNING] {str(message)[:100]}...", **kwargs)

    def error(self, message: str, **kwargs):
        """Log error message with safe encoding"""
        try:
            clean_msg = self._clean_message(message)
            self.logger.error(clean_msg, **kwargs)
        except Exception:
            self.logger.error(f"[ERROR] {str(message)[:100]}...", **kwargs)

    def debug(self, message: str, **kwargs):
        """Log debug message with safe encoding"""
        try:
            clean_msg = self._clean_message(message)
            self.logger.debug(clean_msg, **kwargs)
        except Exception:
            self.logger.debug(f"[DEBUG] {str(message)[:100]}...", **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self.logger.critical(message, **kwargs)
    
    def log_crisis_event(self, crisis_type: str, event: str, severity: float = None):
        """Log crisis-specific events with safe encoding"""
        timestamp = datetime.now().isoformat()
        if severity is not None:
            message = f"[CRISIS] [{crisis_type.upper()}] {event} | Severity: {severity:.3f} | Time: {timestamp}"
        else:
            message = f"[CRISIS] [{crisis_type.upper()}] {event} | Time: {timestamp}"

        self.info(message)
    
    def log_processing_stats(self, component: str, processed: int, total: int, duration: float):
        """Log processing statistics with safe encoding"""
        rate = processed / duration if duration > 0 else 0
        message = f"[STATS] {component} | Processed: {processed}/{total} | Duration: {duration:.2f}s | Rate: {rate:.1f}/s"
        self.info(message)
    
    def log_network_stats(self, crisis_type: str, nodes: int, edges: int, density: float):
        """Log network statistics with safe encoding"""
        message = f"[NETWORK] [{crisis_type.upper()}] | Nodes: {nodes} | Edges: {edges} | Density: {density:.3f}"
        self.info(message)
    
    def log_model_loading(self, model_name: str, status: str = "loaded"):
        """Log model loading events with safe encoding"""
        message = f"[MODEL] {status.upper()}: {model_name}"
        self.info(message)

# Global logger instances
main_logger = CrisisLogger("CrisisDetect+.Main")
nlp_logger = CrisisLogger("CrisisDetect+.NLP")
network_logger = CrisisLogger("CrisisDetect+.Network")
detection_logger = CrisisLogger("CrisisDetect+.Detection")
streaming_logger = CrisisLogger("CrisisDetect+.Streaming")

def get_logger(component: str) -> CrisisLogger:
    """Get logger for specific component"""
    return CrisisLogger(f"CrisisDetect+.{component}")

if __name__ == "__main__":
    # Test logging
    logger = get_logger("Test")
    logger.info("✅ Logging system initialized successfully!")
    logger.log_crisis_event("health", "COVID-19 spike detected", 0.75)
    logger.log_processing_stats("NLP", 1000, 5000, 45.2)
    logger.log_network_stats("technological", 150, 2500, 0.85)
