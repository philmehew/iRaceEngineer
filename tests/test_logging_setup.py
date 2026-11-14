"""Unit tests for modules/logging_setup.py — timestamped session log files."""

from modules.logging_setup import (
    _insert_timestamp,
    llm_query_logger,
    setup_llm_query_log,
)


def test_insert_timestamp_with_log_suffix():
    """_insert_timestamp inserts before the .log extension."""
    assert (
        _insert_timestamp("llm_queries.log", "2024-06-08_14-30-00")
        == "llm_queries_2024-06-08_14-30-00.log"
    )


def test_insert_timestamp_without_log_suffix():
    """_insert_timestamp appends the timestamp to non-.log filenames."""
    assert (
        _insert_timestamp("console", "2024-06-08_14-30-00")
        == "console_2024-06-08_14-30-00"
    )


def test_setup_llm_query_log_writes_file(tmp_path):
    """setup_llm_query_log creates a timestamped file that llm_query_logger lands in."""
    config = {
        "logging": {
            "llm_query_log_dir": str(tmp_path),
            "llm_query_log_file": "queries.log",
        }
    }
    setup_llm_query_log(config, "2024-06-08_14-30-00")

    # The handler writes to queries_2024-06-08_14-30-00.log
    expected_path = tmp_path / "queries_2024-06-08_14-30-00.log"
    assert expected_path.exists()

    llm_query_logger.info("TEST QUERY ENTRY")

    # Flush the handler so the record is on disk
    for handler in llm_query_logger.handlers:
        handler.flush()

    content = expected_path.read_text(encoding="utf-8")
    assert "TEST QUERY ENTRY" in content

    # Clean up the handler so it doesn't leak into other tests
    for handler in list(llm_query_logger.handlers):
        llm_query_logger.removeHandler(handler)
        handler.close()
