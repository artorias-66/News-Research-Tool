"""
Tests for src/export.py — JSON, CSV, and Markdown report export.
"""

import json
import pytest

from src.export import export_to_json, export_to_csv, generate_report


class TestExportToJson:
    """Tests for JSON export."""

    def test_valid_export(self, sample_messages):
        result = export_to_json(sample_messages)
        data = json.loads(result)
        assert data["total_queries"] == 2
        assert len(data["conversations"]) == 2

    def test_empty_messages(self):
        result = export_to_json([])
        data = json.loads(result)
        assert data["total_queries"] == 0

    def test_includes_metadata(self, sample_messages):
        result = export_to_json(sample_messages, metadata={"model": "gpt-3.5"})
        data = json.loads(result)
        assert data["metadata"]["model"] == "gpt-3.5"

    def test_includes_timestamp(self, sample_messages):
        data = json.loads(export_to_json(sample_messages))
        assert "exported_at" in data


class TestExportToCsv:
    """Tests for CSV export."""

    def test_has_header(self, sample_messages):
        result = export_to_csv(sample_messages)
        lines = result.strip().split("\n")
        assert "Timestamp" in lines[0]
        assert "Question" in lines[0]
        assert "Answer" in lines[0]

    def test_correct_row_count(self, sample_messages):
        result = export_to_csv(sample_messages)
        lines = result.strip().split("\n")
        assert len(lines) == 3  # header + 2 Q&A pairs

    def test_empty_messages(self):
        result = export_to_csv([])
        lines = result.strip().split("\n")
        assert len(lines) == 1  # header only


class TestGenerateReport:
    """Tests for Markdown report generation."""

    def test_includes_header(self, sample_messages):
        report = generate_report(sample_messages)
        assert "Equity Research Report" in report

    def test_includes_qa(self, sample_messages):
        report = generate_report(sample_messages)
        assert "Apple" in report
        assert "Tesla" in report

    def test_includes_sources(self, sample_messages):
        report = generate_report(
            sample_messages,
            sources=["https://example.com/article"],
        )
        assert "https://example.com/article" in report

    def test_no_questions_message(self):
        report = generate_report([])
        assert "No questions asked" in report
