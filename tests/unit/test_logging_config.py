#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Logging configuration tests."""

from src.utils.logging_config import LoggingConfig


def test_multiple_instances_independent():
    """Ensure instances do not share mutable config."""
    lc1 = LoggingConfig()
    lc1.config["file"]["enabled"] = True
    lc1.config["modules"]["custom"] = "DEBUG"

    lc2 = LoggingConfig()

    assert lc2.config["file"]["enabled"] is False
    assert "custom" not in lc2.config["modules"]
    assert LoggingConfig.DEFAULT_CONFIG["file"]["enabled"] is False
    assert "custom" not in LoggingConfig.DEFAULT_CONFIG["modules"]

