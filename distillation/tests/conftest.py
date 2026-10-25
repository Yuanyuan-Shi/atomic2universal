"""Shared pytest fixtures of the distillation tests. The exact Gaussian-mixture teachers (mixture.py) are rebuilt for
every test so no test can leak state into another."""

import pytest

from distillation.tests.mixture import GMMTeacher, make_teachers


@pytest.fixture
def teachers() -> dict[int, GMMTeacher]:
    """The exact mixture teachers, keyed by task id (1, 2)."""
    return make_teachers()
