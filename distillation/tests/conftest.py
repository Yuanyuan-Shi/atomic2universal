"""Shared pytest fixtures for the Stage 3a distillation tests. The suite runs on CPU only; the exact toy
teachers are rebuilt for every test so no test can leak state into another."""

import pytest

from distillation.toy.gmm_teacher import GMMTeacher, make_teachers


@pytest.fixture
def teachers() -> dict[int, GMMTeacher]:
    """The exact GMM teachers of the toy, keyed by task id (1, 2)."""
    return make_teachers()
