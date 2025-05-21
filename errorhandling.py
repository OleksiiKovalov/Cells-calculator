import logging
import sys
import traceback # For more detailed traceback formatting if needed

# Configure your logger (do this once, typically at the start of your app)
def setup_logging():
    # Create a logger
    logger = logging.getLogger() # Get the root logger
    logger.setLevel(logging.DEBUG) # Set the minimum logging level for the logger

    # Create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO) # Log INFO and above to console

    # Create file handler and set level to debug
    # Use 'a' for append mode, 'w' for overwrite mode
    fh = logging.FileHandler('app.log', mode='a', encoding='utf-8')
    fh.setLevel(logging.DEBUG) # Log DEBUG and above to file

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

def app_logger():
    return app_logger_int

# Set the custom excepthook

app_logger_int = setup_logging()
sys.excepthook = handle_unhandled_exception    

# Get a logger for the current module (best practice)
logger = logging.getLogger(__name__) # This will inherit settings from the root logger configured above
