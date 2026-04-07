"""
tests/conftest.py — pytest configuration and shared fixtures.
Ensures the project root is on sys.path so `app.*` imports resolve correctly.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
