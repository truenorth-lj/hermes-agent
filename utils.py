"""Shared utility functions for hermes-agent."""

import json
import logging
import os
import stat
import tempfile
from pathlib import Path
from typing import Any, Union

import yaml

logger = logging.getLogger(__name__)


TRUTHY_STRINGS = frozenset({"1", "true", "yes", "on"})


def is_truthy_value(value: Any, default: bool = False) -> bool:
    """Coerce bool-ish values using the project's shared truthy string set."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in TRUTHY_STRINGS
    return bool(value)


def env_var_enabled(name: str, default: str = "") -> bool:
    """Return True when an environment variable is set to a truthy value."""
    return is_truthy_value(os.getenv(name, default), default=False)


def _preserve_file_mode(path: Path) -> "int | None":
    """Capture the permission bits of *path* if it exists, else ``None``."""
    try:
        return stat.S_IMODE(path.stat().st_mode) if path.exists() else None
    except OSError:
        return None


def _restore_file_mode(path: Path, mode: "int | None") -> None:
    """Re-apply *mode* to *path* after an atomic replace.

    ``tempfile.mkstemp`` creates files with 0o600 (owner-only).  After
    ``os.replace`` swaps the temp file into place the target inherits
    those restrictive permissions, breaking Docker / NAS volume mounts
    that rely on broader permissions set by the user.  Calling this
    right after ``os.replace`` restores the original permissions.
    """
    if mode is None:
        return
    try:
        os.chmod(path, mode)
    except OSError:
        pass


def atomic_json_write(
    path: Union[str, Path],
    data: Any,
    *,
    indent: int = 2,
    **dump_kwargs: Any,
) -> None:
    """Write JSON data to a file atomically.

    Uses temp file + fsync + os.replace to ensure the target file is never
    left in a partially-written state. If the process crashes mid-write,
    the previous version of the file remains intact.

    Args:
        path: Target file path (will be created or overwritten).
        data: JSON-serializable data to write.
        indent: JSON indentation (default 2).
        **dump_kwargs: Additional keyword args forwarded to json.dump(), such
            as default=str for non-native types.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    original_mode = _preserve_file_mode(path)

    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.stem}_",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(
                data,
                f,
                indent=indent,
                ensure_ascii=False,
                **dump_kwargs,
            )
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
        _restore_file_mode(path, original_mode)
    except BaseException:
        # Intentionally catch BaseException so temp-file cleanup still runs for
        # KeyboardInterrupt/SystemExit before re-raising the original signal.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def atomic_yaml_write(
    path: Union[str, Path],
    data: Any,
    *,
    default_flow_style: bool = False,
    sort_keys: bool = False,
    extra_content: str | None = None,
) -> None:
    """Write YAML data to a file atomically.

    Uses temp file + fsync + os.replace to ensure the target file is never
    left in a partially-written state.  If the process crashes mid-write,
    the previous version of the file remains intact.

    Args:
        path: Target file path (will be created or overwritten).
        data: YAML-serializable data to write.
        default_flow_style: YAML flow style (default False).
        sort_keys: Whether to sort dict keys (default False).
        extra_content: Optional string to append after the YAML dump
            (e.g. commented-out sections for user reference).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    original_mode = _preserve_file_mode(path)

    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.stem}_",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=default_flow_style, sort_keys=sort_keys)
            if extra_content:
                f.write(extra_content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
        _restore_file_mode(path, original_mode)
    except BaseException:
        # Match atomic_json_write: cleanup must also happen for process-level
        # interruptions before we re-raise them.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ─── JSON Helpers ─────────────────────────────────────────────────────────────


def safe_json_loads(text: str, default: Any = None) -> Any:
    """Parse JSON, returning *default* on any parse error.

    Replaces the ``try: json.loads(x) except (JSONDecodeError, TypeError)``
    pattern duplicated across display.py, anthropic_adapter.py,
    auxiliary_client.py, and others.
    """
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return default


def repair_tool_call_json(raw: str) -> dict:
    """Parse a tool-call argument JSON string with best-effort repair.

    LLM backends like llama.cpp sometimes produce invalid JSON in tool-call
    arguments — most commonly unescaped apostrophes/single-quotes inside
    string values (e.g. ``{"summary": "The user's notes"}``).

    This function tries ``json.loads`` first, then applies lightweight
    heuristic repairs before giving up and returning an empty dict.

    Returns:
        Parsed dict on success, or ``{}`` if the JSON is unrecoverable.
    """
    if not raw or not raw.strip():
        return {}

    # Fast path: valid JSON
    try:
        result = json.loads(raw)
        return result if isinstance(result, dict) else {}
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # ── Repair pass 1: fix unescaped control characters ──────────────
    # llama.cpp sometimes emits literal control chars (tabs, newlines)
    # inside JSON string values.  json.loads(raw, strict=False) handles
    # this without any string manipulation.
    try:
        result = json.loads(raw, strict=False)
        return result if isinstance(result, dict) else {}
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # ── Repair pass 2: unescaped single quotes inside double-quoted strings ──
    # Pattern: the LLM writes  "The user's notes"  which is valid English
    # but the apostrophe sometimes appears as a stray single-quote that
    # breaks certain parser paths, or the LLM uses single-quoted strings
    # instead of double-quoted ones.
    import re

    repaired = raw

    # Replace single-quoted JSON strings with double-quoted ones.
    # This handles the case where the LLM uses Python-style single quotes
    # for the entire JSON object (e.g. {'key': 'value'}).
    if repaired.strip().startswith("{") and "'" in repaired:
        # Try replacing single quotes used as string delimiters.
        # Strategy: outside of double-quoted strings, replace ' with "
        # but only when they appear to be string delimiters.
        try:
            # Attempt: replace all single-quote delimiters with double quotes.
            # This is safe when the original has no double-quoted strings.
            if '"' not in repaired:
                candidate = repaired.replace("'", '"')
                result = json.loads(candidate)
                if isinstance(result, dict):
                    return result
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    # ── Repair pass 3: escape unescaped backslashes & control chars ──
    # Try to fix common escape issues by re-encoding problematic chars
    # within string values.
    try:
        # Walk through the string and escape characters that are invalid
        # inside JSON strings: unescaped newlines, tabs, etc.
        fixed = _escape_invalid_chars_in_json_strings(repaired)
        if fixed != repaired:
            result = json.loads(fixed)
            if isinstance(result, dict):
                return result
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    logger.warning("Could not repair tool-call JSON: %.200s", raw)
    return {}


def _escape_invalid_chars_in_json_strings(raw: str) -> str:
    """Escape unescaped control characters inside JSON string values.

    Walks the raw JSON character-by-character, tracking whether we are
    inside a double-quoted string.  Inside strings, replaces literal
    control characters (0x00-0x1F except already-escaped sequences)
    with their ``\\uXXXX`` equivalents.
    """
    out: list[str] = []
    in_string = False
    i = 0
    n = len(raw)
    while i < n:
        ch = raw[i]
        if in_string:
            if ch == '\\' and i + 1 < n:
                # Escaped char — pass through as-is
                out.append(ch)
                out.append(raw[i + 1])
                i += 2
                continue
            if ch == '"':
                in_string = False
                out.append(ch)
            elif ord(ch) < 0x20:
                # Unescaped control character inside string
                out.append(f'\\u{ord(ch):04x}')
            else:
                out.append(ch)
        else:
            if ch == '"':
                in_string = True
            out.append(ch)
        i += 1
    return ''.join(out)


# ─── Environment Variable Helpers ─────────────────────────────────────────────


def env_int(key: str, default: int = 0) -> int:
    """Read an environment variable as an integer, with fallback."""
    raw = os.getenv(key, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except (ValueError, TypeError):
        return default


def env_bool(key: str, default: bool = False) -> bool:
    """Read an environment variable as a boolean."""
    return is_truthy_value(os.getenv(key, ""), default=default)
