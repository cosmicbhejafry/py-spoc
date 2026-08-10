"""Tests for Calculator warning and exception reports."""

import warnings

from pyspoc.calculator import Calculator


def _reporting_calculator() -> Calculator:
    """Create a Calculator shell without initializing an unrelated dataset."""
    calculator = object.__new__(Calculator)
    calculator._raised_warnings = {}
    calculator._untracked_warnings = ()
    calculator._raised_errors = {}
    calculator._untracked_errors = ()
    return calculator


def test_report_warnings_prints_type_and_location(capsys) -> None:
    calculator = _reporting_calculator()
    warning = warnings.WarningMessage(
        UserWarning("careful"),
        UserWarning,
        "statistic.py",
        42,
    )
    calculator._raised_warnings = {"example-statistic": (warning,)}

    calculator._report_warnings()

    output = capsys.readouterr()
    assert "example-statistic:" in output.out
    assert "[UserWarning] careful" in output.out
    assert "Location: statistic.py:42" in output.out
    assert output.err == ""


def test_report_errors_prints_type_and_traceback_location(capsys) -> None:
    calculator = _reporting_calculator()

    try:
        raise ValueError("invalid result")
    except ValueError as error:
        captured_error = error
        calculator._raised_errors = {"example-reducer": captured_error}

    expected_line = captured_error.__traceback__.tb_lineno
    calculator._report_errors()

    output = capsys.readouterr()
    assert "example-reducer:" in output.out
    assert "[ValueError] invalid result" in output.out
    assert f"Location: {__file__}:{expected_line}" in output.out
    assert output.err == ""


def test_report_errors_handles_exception_without_traceback(capsys) -> None:
    calculator = _reporting_calculator()
    calculator._untracked_errors = (RuntimeError("not raised"),)

    calculator._report_errors()

    output = capsys.readouterr()
    assert "[RuntimeError] not raised" in output.out
    assert "Location: unavailable" in output.out
    assert output.err == ""
