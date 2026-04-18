"""Tests for repair_tool_call_json — defensive JSON parsing for tool-call arguments.

Regression tests for issue #12068: llama.cpp backends sometimes produce
tool-call argument strings with unescaped apostrophes, literal newlines,
or single-quoted strings that cause json.loads to fail.
"""

from utils import repair_tool_call_json


class TestRepairToolCallJson:
    """repair_tool_call_json should recover valid dicts from malformed JSON."""

    # ── Valid JSON (fast path) ───────────────────────────────────────────

    def test_valid_json_passes_through(self):
        raw = '{"action": "save", "summary": "User preferences"}'
        assert repair_tool_call_json(raw) == {
            "action": "save",
            "summary": "User preferences",
        }

    def test_empty_string_returns_empty_dict(self):
        assert repair_tool_call_json("") == {}

    def test_whitespace_only_returns_empty_dict(self):
        assert repair_tool_call_json("   ") == {}

    def test_non_dict_json_returns_empty_dict(self):
        assert repair_tool_call_json("[1, 2, 3]") == {}
        assert repair_tool_call_json('"just a string"') == {}

    # ── Unescaped apostrophes (the #12068 bug) ───────────────────────────

    def test_apostrophe_in_double_quoted_value(self):
        """Standard apostrophe inside double-quoted string — already valid JSON."""
        raw = '{"summary": "The user\'s daily notes"}'
        result = repair_tool_call_json(raw)
        assert result["summary"] == "The user's daily notes"

    def test_literal_control_chars_in_string(self):
        """llama.cpp sometimes emits literal tab/newline inside JSON strings."""
        raw = '{"summary": "line one\\nline two"}'
        result = repair_tool_call_json(raw)
        assert "line one" in result["summary"]

    def test_literal_newline_inside_string_value(self):
        """Actual newline character (not \\n escape) inside a JSON string."""
        raw = '{"summary": "line one\nline two"}'
        result = repair_tool_call_json(raw)
        assert result["summary"] == "line one\nline two"

    def test_literal_tab_inside_string_value(self):
        """Actual tab character inside a JSON string."""
        raw = '{"summary": "col1\tcol2"}'
        result = repair_tool_call_json(raw)
        assert result["summary"] == "col1\tcol2"

    # ── Single-quoted strings ────────────────────────────────────────────

    def test_single_quoted_json_repaired(self):
        """Some backends emit Python-style single-quoted JSON."""
        raw = "{'action': 'save', 'summary': 'User preferences'}"
        result = repair_tool_call_json(raw)
        assert result == {"action": "save", "summary": "User preferences"}

    # ── Completely broken JSON ───────────────────────────────────────────

    def test_total_garbage_returns_empty_dict(self):
        assert repair_tool_call_json("this is not json at all") == {}

    def test_truncated_json_returns_empty_dict(self):
        assert repair_tool_call_json('{"action": "save", "sum') == {}

    # ── Nested values ────────────────────────────────────────────────────

    def test_nested_dict_preserved(self):
        raw = '{"action": "save", "data": {"key": "value"}}'
        result = repair_tool_call_json(raw)
        assert result == {"action": "save", "data": {"key": "value"}}

    def test_memory_save_with_apostrophe_summary(self):
        """Exact reproduction of the #12068 bug report."""
        raw = '{"action": "save", "target": "context", "content": "The user\'s project uses React and they\'re building a dashboard"}'
        result = repair_tool_call_json(raw)
        assert result["action"] == "save"
        assert result["target"] == "context"
        assert "user's project" in result["content"]
