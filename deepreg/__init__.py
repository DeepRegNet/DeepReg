"""
DeepReg package init.

We set tensorflow log level environment variable here before importing it.

- 0 = all messages are logged (default behavior)
- 1 = INFO messages are not printed
- 2 = INFO and WARNING messages are not printed
- 3 = INFO, WARNING, and ERROR messages are not printed

"""
# flake8: noqa
import os

# set tf log level if the env var is not defined
if os.getenv("TF_CPP_MIN_LOG_LEVEL") is None:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # or any {"0", "1", "2", "3"}

import deepreg.dataset
import deepreg.loss
import deepreg.model
