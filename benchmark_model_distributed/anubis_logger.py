import os
import sys
import logging


LOGGER_NAME = "anubis_task_exexcutor"
KEYWORD_LOG_LEVEL = 'LOG_LEVEL'

local_rank = int(os.environ.get('LOCAL_RANK', 0))

class ContextFilter(logging.Filter):
    def filter(self, record):
        record.local_rank = local_rank
        return True

log_level = logging._nameToLevel[os.getenv(KEYWORD_LOG_LEVEL, 'INFO')]

logger = logging.getLogger(LOGGER_NAME)
logger.setLevel(log_level)
logger.addFilter(ContextFilter())

task_working_dir = os.getenv('ANUBIS_WORKING_DIR')

handler = logging.StreamHandler(sys.stdout)
if task_working_dir:
    handler = logging.FileHandler(f"{task_working_dir.rstrip('/')}/logs/AnubisTaskExecutor.log")

handler.setLevel(log_level)
if log_level == logging.INFO:
    formatter = logging.Formatter('%(asctime)s - %(local_rank)s - %(levelname)s - %(message)s')
else:
    formatter = logging.Formatter('%(asctime)s - %(local_rank)s - %(pathname)s [line:%(lineno)d] %(levelname)s - %(message)s')
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)