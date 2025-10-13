import tensorflow as tf
import logging
import time

def setup_tracking():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger('device_tracker')

def track_device(func):
    logger = setup_tracking()
    
    def wrapper(*args, **kwargs):
        start_time = time.time()
        device = tf.config.list_physical_devices('GPU')
        device_type = 'GPU' if device else 'CPU'
        
        logger.info(f"Starting {func.__name__} on {device_type}")
        result = func(*args, **kwargs)
        logger.info(f"Finished {func.__name__}. Duration: {time.time() - start_time:.4f}s")
        
        return result
    return wrapper