# Standard library imports
import glob
import logging
import os
import sys
import traceback  # For more detailed traceback formatting if needed
from datetime import datetime

# Third-party imports
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QMessageBox

class LogEventEmitter(QObject):
    """
    Event emitter for log file events.
    Emits signals when new lines are added to the log file.
    """
    log_line_added = pyqtSignal(str)  # Signal emitted when new log line is added
    
    def __init__(self):
        super().__init__()

# Global instance for log events
log_event_emitter = LogEventEmitter()

class EventFileHandler(logging.FileHandler):
    """
    Custom file handler that emits events when log lines are written.
    """
    def __init__(self, filename, mode='a', encoding='utf-8', delay=False):
        super().__init__(filename, mode, encoding, delay)
        
    def emit(self, record):
        """
        Emit a log record and trigger the log line event.
        
        Args:
            record: LogRecord instance
        """
        try:
            # Format the log message
            formatted_message = self.format(record)
            
            # Call parent emit to write to file
            super().emit(record)
            
            # Emit the signal with the formatted log line
            log_event_emitter.log_line_added.emit(formatted_message)
            
        except Exception as e:
            # Handle errors in logging (avoid infinite recursion)
            print(f"Error in log event emission: {e}")

def cleanup_old_logs():
    """
    Keep only the last 14 log files, delete older ones.
    """
    try:
        # Ensure logs directory exists
        logs_dir = "logs"
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
            return
        
        # Get all log files in the logs directory matching YYYYMMDD.log pattern
        log_pattern = os.path.join(logs_dir, "????????.log")
        log_files = glob.glob(log_pattern)
        
        # Sort by filename (which is the date) in descending order
        log_files.sort(reverse=True)
        
        # Keep only the first 14 files (most recent), delete the rest
        if len(log_files) > 14:
            files_to_delete = log_files[14:]
            for file_path in files_to_delete:
                try:
                    os.remove(file_path)
                    print(f"Deleted old log file: {file_path}")
                except OSError as e:
                    print(f"Could not delete log file {file_path}: {e}")
                    
    except Exception as e:
        print(f"Error during log cleanup: {e}")

# Configure your logger (do this once, typically at the start of your app)
def setup_logging():
    # Create logs directory if it doesn't exist
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    # Clean up old log files
    cleanup_old_logs()
    
    # Create a logger
    logger = logging.getLogger() # Get the root logger
    logger.setLevel(logging.INFO) # Set the minimum logging level for the logger

    # Create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO) # Log INFO and above to console

    # Create file handler with date-based filename and event emission
    current_date = datetime.now().strftime("%Y%m%d")
    log_filename = os.path.join(logs_dir, f"{current_date}.log")
    fh = EventFileHandler(log_filename, mode='a', encoding='utf-8')
    fh.setLevel(logging.INFO) # Log INFO and above to file

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s'
    )
    error_formatter = logging.Formatter( # Potentially a more detailed one for errors
        '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(funcName)s - %(message)s\n%(exc_info)s'
    )


    # Add formatter to handlers
    ch.setFormatter(formatter)
    fh.setFormatter(formatter) # Use the same formatter or a different one for the file

    # Add handlers to the logger
    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger

# --- Global Unhandled Exception Handler ---
def handle_unhandled_exception(exc_type, exc_value, exc_traceback):
    """
    Catches unhandled exceptions, logs them, and ensures they are still printed to stderr.
    """
    if issubclass(exc_type, KeyboardInterrupt):
        # Allow KeyboardInterrupt to interrupt the program as usual
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    # Log the exception with traceback
    # logger.error() or logger.critical() can be used.
    # logger.exception() is convenient as it automatically includes exception info.
    app_logger().critical("Unhandled exception caught:", exc_info=(exc_type, exc_value, exc_traceback))
    error_msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    msgbox = QMessageBox()
    msgbox.setIcon(QMessageBox.Critical)
    msgbox.setWindowTitle("Unhandled Exception")
    msgbox.setText("An error occurred:")
    msgbox.setDetailedText(error_msg)
    msgbox.exec_()
    
def app_logger():
    return app_logger_int

class LoggerWriter:
    """
    A file-like object that redirects writes to a logger instance.
    """
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.buffer = []

    def write(self, message):
        # Handle empty messages and newlines
        if message and message.strip():
            # Log the message at the specified level
            self.logger.log(self.level, message.strip())
        
        # Also write to original stdout/stderr for console visibility
        if self.level == logging.INFO:
            sys.__stdout__.write(message)
        else:
            sys.__stderr__.write(message)

    def flush(self):
        # Flush the original streams
        if self.level == logging.INFO:
            sys.__stdout__.flush()
        else:
            sys.__stderr__.flush()

def setup_console_logging():
    """
    Redirect stdout and stderr to also write to the logger.
    This ensures all print statements and error messages are logged to file.
    """
    logger = app_logger_int
    
    # Create custom writers that log to file AND display on console
    stdout_logger = LoggerWriter(logger, logging.INFO)
    stderr_logger = LoggerWriter(logger, logging.ERROR)
    
    # Redirect stdout and stderr
    sys.stdout = stdout_logger
    sys.stderr = stderr_logger
    
    logger.info("Console output redirection setup complete - all print statements will now be logged to file")

def restore_console():
    """
    Restore original stdout and stderr.
    Call this if you need to disable console logging redirection.
    """
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    app_logger_int.info("Console output redirection disabled - restored original stdout/stderr")

class TeeLogger:
    """
    A more advanced logger that can capture and log specific operations.
    Use this for selective logging of operations.
    """
    def __init__(self, logger, original_stream, log_level=logging.INFO):
        self.logger = logger
        self.original_stream = original_stream
        self.log_level = log_level
        
    def write(self, message):
        # Write to original stream (console)
        self.original_stream.write(message)
        
        # Log to file if message is not empty/whitespace
        if message and message.strip():
            self.logger.log(self.log_level, f"CONSOLE: {message.strip()}")
            
    def flush(self):
        self.original_stream.flush()

def setup_selective_console_logging():
    """
    Alternative setup that prefixes console messages in the log file.
    This makes it easier to distinguish between direct log calls and console output.
    """
    logger = app_logger_int
    
    # Store original streams
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    # Create tee loggers
    sys.stdout = TeeLogger(logger, original_stdout, logging.INFO)
    sys.stderr = TeeLogger(logger, original_stderr, logging.ERROR)
    
    logger.info("Selective console logging setup - console output will be prefixed with 'CONSOLE:' in log file")

def log_function_calls(func):
    """
    Decorator to log function calls and their results.
    Usage: @log_function_calls
    """
    def wrapper(*args, **kwargs):
        logger = app_logger_int
        func_name = func.__name__
        
        # Log function entry
        logger.info(f"FUNCTION_CALL: Entering {func_name} with args={args}, kwargs={kwargs}")
        
        try:
            result = func(*args, **kwargs)
            logger.info(f"FUNCTION_CALL: {func_name} completed successfully")
            return result
        except Exception as e:
            logger.error(f"FUNCTION_CALL: {func_name} failed with error: {str(e)}")
            raise
            
    return wrapper

def log_print(*args, **kwargs):
    """
    Enhanced print function that always logs to file.
    Use this instead of print() for important messages you want logged.
    
    Usage: log_print("This will appear in console AND log file")
    """
    # Print to console
    print(*args, **kwargs)
    
    # Also log to file
    message = ' '.join(str(arg) for arg in args)
    app_logger_int.info(f"LOG_PRINT: {message}")

def connect_to_log_events(callback):
    """
    Connect a callback function to log line events.
    
    Args:
        callback (callable): Function to call when new log line is added.
                           Should accept one string parameter (the log line).
    
    Example:
        def on_new_log_line(log_line):
            print(f"New log: {log_line}")
        
        connect_to_log_events(on_new_log_line)
    """
    log_event_emitter.log_line_added.connect(callback)

def disconnect_from_log_events(callback):
    """
    Disconnect a callback function from log line events.
    
    Args:
        callback (callable): Function to disconnect from log events.
    """
    try:
        log_event_emitter.log_line_added.disconnect(callback)
    except TypeError:
        # Signal was not connected
        pass

def get_log_event_emitter():
    """
    Get the global log event emitter instance.
    
    Returns:
        LogEventEmitter: The global log event emitter
    """
    return log_event_emitter

# Set the custom excepthook

app_logger_int = setup_logging()
sys.excepthook = handle_unhandled_exception

# Setup console output redirection
setup_console_logging()

# Get a logger for the current module (best practice)
logger = logging.getLogger(__name__) # This will inherit settings from the root logger configured above
