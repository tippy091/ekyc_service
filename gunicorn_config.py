import multiprocessing

# The socket to bind to. This can be a host:port pair or a Unix socket path.
# For Docker, '0.0.0.0:8000' is typical to bind to all interfaces on port 8000.
# For Nginx/Apache reverse proxy, 'unix:/path/to/your/app.sock' is common.
bind = "0.0.0.0:8000"

# The number of worker processes.
# A common formula is (2 * number_of_cpu_cores) + 1.
workers = multiprocessing.cpu_count() + 1

# The type of workers to use.
# 'sync' (default): Synchronous workers (good for CPU-bound tasks).
# 'gevent', 'eventlet', 'meinheld': Asynchronous workers (good for I/O-bound tasks, requires installing the respective library).
# worker_class = "sync" # This is the default, no need to explicitly set unless changing
worker_class = "gevent"

# The maximum number of requests a worker will process before restarting.
# This helps with memory leaks. 0 means no limit.
max_requests = 1000

# The maximum jitter to add to the max_requests setting.
max_requests_jitter = 50

# Workers silent for more than this many seconds are killed and restarted.
timeout = 60

# The path to the access log file. Use '-' for stdout.
accesslog = "-"

# The path to the error log file. Use '-' for stderr.
errorlog = "-"

# The granularity of Error log outputs.
# Valid values are "debug", "info", "warning", "error", "critical".
loglevel = "info"

# The number of seconds to wait for graceful termination.
graceful_timeout = 30

# Enable or disable the use of Python's multiprocessing logging.
# If False, Gunicorn's own logging will be used.
# If True, the standard Python logging module will be used directly.
capture_output = True  # Useful for debugging or if you're using Python's logging

# Set the process name (useful for process monitoring tools)
proc_name = "py_ekyc_service"

# You can even set environment variables here if needed
# raw_env = ["MY_ENV_VAR=my_value"]
