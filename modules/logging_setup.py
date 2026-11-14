"""
Logging setup — timestamped session log files for the console mirror and the
LLM query log.

Owns the `llm_query_logger` used by the query pipeline (engineer_query.py)
so there is a single place that configures it.
"""

import logging
import os
from datetime import datetime

logger = logging.getLogger("iraceengineer")


def _session_timestamp() -> str:
    """Return a timestamp string for log file names (same format as capture sessions)."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _insert_timestamp(filename: str, ts: str) -> str:
    """Insert a timestamp before the .log extension in a filename.

    e.g. _insert_timestamp("llm_queries.log", "2024-06-08_14-30-00")
         -> "llm_queries_2024-06-08_14-30-00.log"
    """
    if filename.endswith(".log"):
        return f"{filename[:-4]}_{ts}.log"
    return f"{filename}_{ts}"


# Dedicated logger for LLM query data — handler added in setup_llm_query_log()
llm_query_logger = logging.getLogger("iraceengineer.llm_query")
llm_query_logger.propagate = False  # Don't double-print to console


def setup_llm_query_log(config: dict, session_ts: str):
    """Set up the LLM query log file handler.

    Creates a FileHandler that writes the prompt and response for every LLM
    call to a timestamped log file (e.g. logs/llm_queries_2024-06-08_14-30-00.log).
    """
    log_config = config.get("logging", {})
    log_dir = log_config.get("llm_query_log_dir", "logs")
    log_file = log_config.get("llm_query_log_file", "llm_queries.log")

    os.makedirs(log_dir, exist_ok=True)
    stamped_file = _insert_timestamp(log_file, session_ts)
    log_path = os.path.join(log_dir, stamped_file)

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s\n%(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    llm_query_logger.addHandler(handler)
    llm_query_logger.setLevel(logging.INFO)
    logger.info(f"LLM query log: {log_path}")


def setup_console_log(config: dict, session_ts: str):
    """Set up a console mirror log that captures all console output to a file.

    Adds a FileHandler to the root logger so anything printed to the console
    is also written to a timestamped log file (e.g. logs/console_2024-06-08_14-30-00.log).
    """
    log_config = config.get("logging", {})
    log_dir = log_config.get("console_log_dir", "logs")
    log_file = log_config.get("console_log_file", "console.log")

    os.makedirs(log_dir, exist_ok=True)
    stamped_file = _insert_timestamp(log_file, session_ts)
    log_path = os.path.join(log_dir, stamped_file)

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S"
        )
    )
    # No explicit level — inherits from root logger, so mirrors whatever the console shows
    logging.getLogger().addHandler(handler)
    logger.info(f"Console log: {log_path}")
