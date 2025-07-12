import tensorflow as tf
import numpy as np

import random

def set_seed(seed: int):
    """
    Set the random seed for reproducibility.
    
    Args:
        seed (int): The seed value to set for random number generation.
    """
    # Set seeds for reproducibility
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Set TensorFlow session determinism for reproducibility
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()