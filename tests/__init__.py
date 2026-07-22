"""Test package for the ``heart_ml`` library.

Marked as a package (rather than a bare directory) so that:

* ``tests.strategies`` can be imported unambiguously by the tests in this tree
  and reused by the backend suite (``backend/tests``), and
* the shared ``strategies.py`` / ``conftest.py`` module names never collide
  with their same-named counterparts under ``backend/tests`` during pytest
  collection.
"""
