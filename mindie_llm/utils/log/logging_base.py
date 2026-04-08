# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""
Log Manager Module

This module provides core logging functionality, including:
- Support for multi-component logging (LLM, LLMMODELS)
- Support for selective log routing (via HandlerType)
- Support for log rotation and file permission control
- Support for error code formatting
- Support for log level control
- Support for console and file output

Basic usage example:
    >>> logger = get_logger(Component.LLM)
    >>> logger.info("This goes to all handlers")

Advanced usage example:
    >>> logger.info("This goes to specific handlers", 
    ...            extra={"handler_ids": HandlerType.TOKEN})
"""

import logging
import os
import threading
import time
from datetime import datetime, timezone, timedelta
from enum import Enum
from logging.handlers import RotatingFileHandler
from logging import setLoggerClass, StreamHandler

from ..env import ENV
from .. import file_utils
from .error_code import ErrorCode
from .utils import get_component_config, create_log_dir_and_check_permission, update_log_file_param, MAX_KEY_LENGTH


# Constants
MAX_OPEN_LOG_FILE_PERM = 0o640
MAX_CLOSE_LOG_FILE_PERM = 0o440

_SWITCH_ON = ["1", "TRUE"]
_SWITCH_OFF = ["0", "FALSE"]
_ALL_SWITCHES = _SWITCH_ON + _SWITCH_OFF

MINDIE = "mindie"
LOG = "log"
DEBUG_PATH = "debug"
EXTRA = 'extra'


class Component(Enum):
    LLM = "llm"
    LLMMODELS = "llmmodels"


class HandlerType(Enum):
    LLM = "llm"
    LLMMODELS = "llmmodels"
    TOKEN = "token"


class CustomLogger(logging.Logger):
    """Logger with error code support"""
    def log(self, level, msg, *args, **kwargs):
        if level in [logging.ERROR, logging.CRITICAL]:
            if len(args) > 0 and isinstance(args[0], ErrorCode):
                error_code = args[0]
                kwargs[EXTRA] = kwargs.get(EXTRA, {})
                kwargs[EXTRA]['error_code'] = error_code
                args = args[1:]

        kwargs['stacklevel'] = kwargs.get('stacklevel', 1) + 1
        super().log(level, msg, *args, **kwargs)


setLoggerClass(CustomLogger)


class CustomLoggerAdapter(logging.LoggerAdapter):
    """Custom LoggerAdapter that properly handles extra information"""
    
    def process(self, msg, kwargs):
        # Get existing extra
        extra = kwargs.get('extra', {})
        
        # Merge self.extra and passed extra
        if self.extra:
            if not extra:
                extra = self.extra.copy()
            else:
                extra = {**self.extra, **extra}
        
        # Update kwargs
        kwargs['extra'] = extra
        return msg, kwargs


class ErrorCodeFormatter(logging.Formatter):
    def __init__(self, component: Component, fmt=None, datefmt=None, style="%"):
        super().__init__(fmt, datefmt, style)
        self.component = component

        log_verbose = get_component_config(ENV.log_verbose, self.component.value)
        self.verbose = str(log_verbose).upper() in _SWITCH_ON or str(log_verbose).upper() not in _ALL_SWITCHES

        logging.addLevelName(logging.WARNING, 'WARN')
        self.error_fmt_verbose = logging.Formatter(
        '[%(asctime)s] [%(process)d] [%(thread)d] [%(name)s] [%(levelname)s] '
        '[%(filename)s-%(lineno)d] : [%(error_code)s] %(message)s'
        )
        self.default_fmt_verbose = logging.Formatter(
        '[%(asctime)s] [%(process)d] [%(thread)d] [%(name)s] [%(levelname)s] '
        '[%(filename)s-%(lineno)d] : %(message)s'
        )
        self.error_fmt = logging.Formatter(
        '[%(asctime)s] : [%(levelname)s] [%(error_code)s] %(message)s'
        )
        self.default_fmt = logging.Formatter(
        '[%(asctime)s] : [%(levelname)s] %(message)s'
        )

    def format(self, record):
        if self.verbose:
            if record.levelno >= logging.ERROR and hasattr(record, 'error_code'):
                formatter = self.error_fmt_verbose
            else:
                formatter = self.default_fmt_verbose   
        else:
            if record.levelno >= logging.ERROR and hasattr(record, 'error_code'):
                formatter = self.error_fmt
            else:
                formatter = self.default_fmt

        return formatter.format(record)


class SelectiveConsoleHandler(logging.StreamHandler):
    def __init__(self):
        super().__init__()

    def emit(self, record: logging.LogRecord) -> None:
        # HandlerType.TOKEN does not output to console
        active_handlers = getattr(record, 'handler_ids', None)
        if active_handlers == HandlerType.TOKEN:
            return
        
        self.setFormatter(record.formatter)
        super().emit(record)

        
class SelectiveRotatingFileHandler(RotatingFileHandler):
    """File handler that can be selectively enabled/disabled via context."""
    
    def __init__(self, handler_type: HandlerType, filename: str, max_bytes: int = 0, 
                 backup_count: int = 0, encoding: str = None, delay: bool = False):
        super().__init__(filename, "a", max_bytes, backup_count, encoding, delay)
        self.handler_type = handler_type

    def emit(self, record: logging.LogRecord) -> None:
        """Only emit if this handler is selected in the current context."""
        active_handlers = getattr(record, 'handler_ids', None)

        should_emit = False
        # If no specific handlers are selected
        if active_handlers is None or not active_handlers:
            # only use default handler
            should_emit = self.handler_type == record.default_file_handler
        else:
            # If specific handlers are selected
            should_emit = self.handler_type == active_handlers

        # Emit with corresponding formatter if should process
        if should_emit:
            self.setFormatter(record.formatter)
            super().emit(record)

    def get_rotation_filename(self, index):
        """Generate rotation filename in format mindie.01.log, mindie.02.log, etc."""
        if index == 0:
            return self.baseFilename
        # Use two-digit sequence numbers like 01, 02, 03
        return f"{self.baseFilename.rsplit('.', 1)[0]}.{index:02d}.{self.baseFilename.rsplit('.', 1)[1]}"

    def doRollover(self):
        """
        Override:
            Do a rollover and modify the permissions of old files.
        """
        if self.stream:
            self.stream.close()
            self.stream = None

        if self.backupCount > 0:
            for i in range(self.backupCount - 2, 0, -1):
                sfn = self.get_rotation_filename(i)
                dfn = self.get_rotation_filename(i + 1)

                if os.path.exists(sfn):
                    if os.path.exists(dfn):
                        os.remove(dfn)
                    os.rename(sfn, dfn)
                    file_utils.safe_chmod(dfn, MAX_CLOSE_LOG_FILE_PERM)

            dfn = self.get_rotation_filename(1)
            if os.path.exists(dfn):
                os.remove(dfn)
            self.rotate(self.baseFilename, dfn)
            file_utils.safe_chmod(dfn, MAX_CLOSE_LOG_FILE_PERM)

        if not self.delay:
            self.stream = self._open()

    def close(self):
        """
        Override:
            Close the stream and set the file permissions.
        """
        self.acquire()
        try:
            try:
                if self.stream:
                    try:
                        self.flush()
                    finally:
                        stream = self.stream
                        self.stream = None
                        if hasattr(stream, "close"):
                            stream.close()
            finally:
                StreamHandler.close(self)
        finally:
            file_utils.safe_chmod(self.baseFilename, MAX_CLOSE_LOG_FILE_PERM)
            self.release()

    def _open(self):
        """
        Override:
            Open the current base file with the (original) mode and encoding.
            Modify the permissions of current files.
            Return the resulting stream.
        """
        if os.path.exists(self.baseFilename):
            file_utils.safe_chmod(self.baseFilename, MAX_OPEN_LOG_FILE_PERM)
        ret = self._builtin_open(self.baseFilename, self.mode, encoding=self.encoding, errors=self.errors)
        file_utils.safe_chmod(self.baseFilename, MAX_OPEN_LOG_FILE_PERM)
        return ret


class LoggerManager:
    """Manages loggers with selective routing."""

    def __init__(self):
        self._handlers: dict[HandlerType, SelectiveRotatingFileHandler] = {}
        self._loggers: dict[Component, CustomLoggerAdapter] = {}
        # The first handler type is the default one to use.
        self._logger_to_handlers_map: dict[Component, list[HandlerType]] = {
            Component.LLM: [HandlerType.LLM, HandlerType.TOKEN],
            Component.LLMMODELS: [HandlerType.LLMMODELS],
        }
        self._shared_console_handler = None
        self._lock = threading.Lock()

        self.pid = os.getpid()

        milliseconds = str(time.time() * 1000)
        self.process_datetime = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d%H%M%S") + milliseconds[0:3]

    @staticmethod
    def _get_log_dir(component: Component) -> str:
        """Get log directory path"""
        # Root directory validation
        home_dir = os.path.expanduser("~")
        if len(home_dir) >= MAX_KEY_LENGTH:
            home_dir = home_dir[:MAX_KEY_LENGTH]
        if not os.path.exists(home_dir):
            raise FileNotFoundError(f"Home directory {home_dir} does not exist or access denied! "
                                "Please manually set the log storage path "
                                f"or change the home directory {home_dir} permission.")
        home_dir = file_utils.standardize_path(home_dir)
        file_utils.check_path_permission(home_dir)

        # Process log file path
        file_path = ENV.log_file_path
        file_path = get_component_config(file_path, component.value)
        if len(file_path) == 0:
            file_path = os.path.join(home_dir, MINDIE, LOG, DEBUG_PATH)
        elif not file_path.startswith("/"):
            file_path = os.path.join(home_dir, MINDIE, LOG, file_path, DEBUG_PATH)
        else:
            file_path = os.path.join(file_path, LOG, DEBUG_PATH)

        file_utils.makedir_and_change_permissions(file_path)
        file_path = file_utils.standardize_path(file_path)
        file_utils.check_path_permission(file_path)

        return file_path

    @staticmethod
    def _set_log_level(logger: CustomLogger):
        """Get log level"""
        log_level = get_component_config(ENV.log_file_level, logger.name)

        # Validate log level
        valid_levels = ["DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"]
        if log_level.upper() not in valid_levels:
            log_level = "INFO"

        # Set log level
        logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    @staticmethod
    def _get_log_rotate_config(component: Component) -> tuple[int, int]:
        """Get log rotation parameters

        Returns:
            tuple[int, int]: (maximum file size, maximum number of files)
        """
        # 1. Get and process log rotation configuration
        log_file_rotate = ENV.log_file_rotate
        log_rotate_config = get_component_config(log_file_rotate, component.value)
        max_size, max_files = update_log_file_param(log_rotate_config)

        # 2. Handle old environment variables for llmmodels component
        if component == Component.LLMMODELS and os.getenv("PYTHON_LOG_MAXSIZE") is not None:
            logging.warning(
                "Note: The old environment variable PYTHON_LOG_MAXSIZE will be deprecated on 2026/12/31.\n"
                "Please use the new environment variable MINDIE_LOG_ROTATE instead.\n"
                "Usage: export MINDIE_LOG_ROTATE=\"-fs 20 -r 10\"\n"
                "Where:\n"
                "  -fs: maximum size of each log file in MB (range: [1, 500])\n"
                "  -r : maximum number of log files per process (range: [1, 64])"
            )
            if log_rotate_config:
                logging.warning("MINDIE_LOG_ROTATE has been set. PYTHON_LOG_MAXSIZE will be ignored.")
            else:
                max_size = ENV.atb_llm_log_maxsize

        return max_size, max_files

    def get_logger(self, component: Component) -> CustomLoggerAdapter:
        """Get or create logger"""
        if component not in self._loggers:
            with self._lock:
                if component not in self._loggers:
                    self._create_logger(component)
        return self._loggers[component]

    def _create_logger(self, component: Component):
        """Create logger for a single component"""
        # Initialize console handler only when creating logger for the first time
        if self._shared_console_handler is None:
            self._shared_console_handler = SelectiveConsoleHandler()

        # Create file handlers
        self._create_file_handlers(component)

        extra_info = {
            "component": component,
            "default_file_handler": self._logger_to_handlers_map.get(component, [HandlerType.LLM])[0],
            "formatter": ErrorCodeFormatter(component)
        }
        logger_ins = logging.getLogger(component.value)

        # Clear existing handlers
        logger_ins.handlers.clear()

        # Configure logger
        self._set_log_level(logger_ins)
        self._add_file_handlers(logger_ins)
        self._add_console_handler(logger_ins)

        # Disable log propagation
        logger_ins.propagate = False

        # Store logger
        logger_ins_adapter = CustomLoggerAdapter(logger_ins, extra_info)
        self._loggers[component] = logger_ins_adapter

    def _create_file_handlers(self, component: Component) -> None:
        for handler_type in self._logger_to_handlers_map.get(component, []):
            if self._handlers.get(handler_type) is not None:
                continue

            # Get log directory
            file_dir = self._get_log_dir(component)
            
            # Check if file sharing is needed
            shared_filename = None
            if handler_type in [HandlerType.LLM, HandlerType.LLMMODELS]:
                other_type = HandlerType.LLMMODELS if handler_type == HandlerType.LLM else HandlerType.LLM
                if self._get_log_dir(other_type) == file_dir:
                    shared_filename = f"mindie-llm_{self.pid}_{self.process_datetime}.log"

            # Set filename
            if shared_filename:
                filename = shared_filename
            else:
                filename = f"mindie-llm-{handler_type.value}_{self.pid}_{self.process_datetime}.log"
                if handler_type == HandlerType.LLM:
                    filename = f"mindie-llm_{self.pid}_{self.process_datetime}.log"
                elif handler_type == HandlerType.LLMMODELS:
                    filename = f"mindie-llmmodels_{self.pid}_{self.process_datetime}.log"

            file_path = os.path.join(file_dir, filename)
            create_log_dir_and_check_permission(file_path)

            # Token handler uses special configuration
            if handler_type == HandlerType.TOKEN:
                log_file_maxsize = 1024 * 1024  # 1MB
                log_file_maxnum = 2
            else:
                # Get log rotation configuration
                log_file_maxsize, log_file_maxnum = self._get_log_rotate_config(component)

            handler = SelectiveRotatingFileHandler(
                handler_type=handler_type, 
                filename=file_path,
                max_bytes=log_file_maxsize,
                backup_count=log_file_maxnum,
                delay=True
            )

            self._handlers[handler_type] = handler

    def _add_file_handlers(self, logger: CustomLogger):
        """Configure file output"""
        log_to_file = get_component_config(ENV.log_to_file, logger.name)

        # Add file handlers
        if str(log_to_file).upper() in _SWITCH_ON or str(log_to_file).upper() not in _ALL_SWITCHES:
            # Only add handlers corresponding to current component
            handler_types = self._logger_to_handlers_map.get(Component(logger.name), [])
            for handler_type in handler_types:
                handler = self._handlers.get(handler_type)
                if handler is not None:
                    logger.addHandler(handler)

    def _add_console_handler(self, logger: CustomLogger):
        """Configure console output"""
        log_to_stdout = get_component_config(ENV.log_to_stdout, logger.name)

        # Add console handler
        if log_to_stdout.upper() in _SWITCH_ON:
            logger.addHandler(self._shared_console_handler)


# Global manager instance
_manager = LoggerManager()


def get_logger(component: Component) -> CustomLoggerAdapter:
    """Get logger"""
    return _manager.get_logger(component)
