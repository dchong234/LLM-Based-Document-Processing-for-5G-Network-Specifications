"""
Pytest configuration and shared fixtures
"""

import pytest
import os
import tempfile
import shutil
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def fixtures_dir():
    """Get the fixtures directory path."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_text():
    """Sample 3GPP-style text for testing."""
    return """3GPP TS 38.211 V17.0.0 (2022-03)

Technical Specification

5.1 Physical Layer Overview

The physical layer in 5G NR provides the radio interface between the UE and the network.
It supports multiple numerologies and flexible frame structures.

5.1.1 Numerology

Numerology is defined as the subcarrier spacing configuration parameter mu.
It determines the spacing between OFDM subcarriers.

© 2022 3GPP
Page 15
"""


@pytest.fixture
def sample_cleaned_text():
    """Expected cleaned text output."""
    return """Technical Specification

5.1 Physical Layer Overview

The physical layer in 5G NR provides the radio interface between the UE and the network.
It supports multiple numerologies and flexible frame structures.

5.1.1 Numerology

Numerology is defined as the subcarrier spacing configuration parameter mu.
It determines the spacing between OFDM subcarriers.
"""


@pytest.fixture
def sample_section():
    """Sample section data for Q&A generation."""
    return {
        "section_num": "5.1.1",
        "title": "Numerology",
        "content": "Numerology is defined as the subcarrier spacing configuration parameter mu. It determines the spacing between OFDM subcarriers.",
        "level": 2,
        "parent": "5.1"
    }


@pytest.fixture
def sample_sections():
    """Sample sections list."""
    return [
        {
            "section_num": "5.1",
            "title": "Physical Layer Overview",
            "content": "The physical layer in 5G NR provides the radio interface.",
            "level": 1,
            "parent": None
        },
        {
            "section_num": "5.1.1",
            "title": "Numerology",
            "content": "Numerology is defined as the subcarrier spacing configuration parameter mu.",
            "level": 2,
            "parent": "5.1"
        }
    ]

