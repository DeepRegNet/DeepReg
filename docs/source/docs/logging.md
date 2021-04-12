# Logging

When running DeepReg, there are two types of printed messages:

1. DeepReg messages, which are entirely controlled by DeepReg code.
2. TensorFlow messages, which are automatically printed by TensorFlow during certain
   workflows.

Therefore, we use two independent environment variables to control these two types of
messages. The log level controls which types of log messages would be printed. Further
details are explained below.

## DeepReg logging

DeepReg uses the
[Python module `logging`](https://docs.python.org/3/library/logging.html) to log the
messages. The log level is controlled by the environment variable `DEEPREG_LOG_LEVEL`.
The levels are given in the table below. The default level is "2".

To adjust the logging level, there are two options. Take the training as example,

- You can first define the environment variable, then run the job.

  ```bash
  export DEEPREG_LOG_LEVEL=1
  deepreg_train --gpu "" --config_path config/unpaired_labeled_ddf.yaml --exp_name test
  ```

- Alternatively, you can define the environment variable while running the job.

  ```bash
  DEEPREG_LOG_LEVEL=1 deepreg_train --gpu "" --config_path config/unpaired_labeled_ddf.yaml --exp_name test
  ```

| DEEPREG_LOG_LEVEL | Behavior                                                                                   |
| ----------------- | ------------------------------------------------------------------------------------------ |
| "0"               | Log all messages, equivalent to `logging.DEBUG`. Same as log level "1".                    |
| "1"               | Log all messages, equivalent to `logging.DEBUG`.                                           |
| "2"               | Log all messages except DEBUG, equivalent to `logging.INFO`. (default)                     |
| "3"               | Log all messages except DEBUG and INFO, equivalent to `logging.WARNING`.                   |
| "4"               | Log all messages except DEBUG, INFO, and WARNING, equivalent to `logging.ERROR`.           |
| "5"               | Log all messages except DEBUG, INFO, WARNING, and ERROR, equivalent to `logging.CRITICAL`. |

## TensorFlow logging

With TensorFlow 2.3, its log level is controlled by the environment variable
`TF_CPP_MIN_LOG_LEVEL`. The levels are given in the table below. The default level is
"2".

To adjust the logging level, there are two options. Take the training as example,

- You can first define the environment variable, then run the job.

  ```bash
  export TF_CPP_MIN_LOG_LEVEL=1
  deepreg_train --gpu "" --config_path config/unpaired_labeled_ddf.yaml --exp_name test
  ```

- Alternatively, you can define the environment variable while running the job.

  ```bash
  TF_CPP_MIN_LOG_LEVEL=1 deepreg_train --gpu "" --config_path config/unpaired_labeled_ddf.yaml --exp_name test
  ```

| TF_CPP_MIN_LOG_LEVEL | Behavior                                            |
| -------------------- | --------------------------------------------------- |
| "0"                  | Log all messages.                                   |
| "1"                  | Log all messages except INFO.                       |
| "2"                  | Log all messages except INFO and WARNING. (default) |
| "3"                  | Log all messages except INFO, WARNING, and ERROR.   |
