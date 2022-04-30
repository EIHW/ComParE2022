import os

import config

# General:
# Adadelta optimizer configuration
learning_rate = 1.0
rho = 0.95
epsilon = 1e-07

# BNN weight regularisation:
tau = 1.0
dropout = 0.2

# Settings for mosquito event detection:
epochs = 1
batch_size = 8  # Changed from 32
lengthscale = 0.01


# Make output sub-directory for saving model
directory = os.path.join(config.model_dir, "keras")
if not os.path.isdir(directory):
    os.makedirs(directory)
    print("Created directory:", directory)
