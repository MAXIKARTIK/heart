"""Meta-test: property-to-test coverage guard.

The design enumerates 19 correctness properties, and the Testing Strategy
requires each one to map to *exactly one* property-based test, tagged in the
source as ``Feature: heart-disease-prediction, Property {n}: {text}`` on the
line directly above the test function.

This module scans the two test trees (``tests`` and ``backend/tests``) and
asserts every property tag from 1 through 19 appears exactly once and sits
immediately above a ``def test_...`` function. It is a *coverage guard*, not
one of the 19 property tests: it deliberately builds the tag prefix at runtime
(so no literal, fully-numbered tag string lives in this file) and excludes
itself from the scan, so it never inflates or double-counts the property tally.
"""
from __future__ import annotations

import re
from pathlib import Path

# The full set of correctness properties documented in design.md.
EXPECTED_PROPERTY_COUNT = 19
EXPECTED_PROPERTY_NUMBERS = range(1, EXPECTED_PROPERTY_COUNT + 1)

# Built at runtime so the literal, fully-numbered tag never appears verbatim in
# this file -- that keeps the coverage guard from matching (and double-counting)
# itself when it scans the suite.
_TAG_PREFIX = "Feature: heart-disease-prediction, Property"
_ANY_TAG = re.compile(rf"{re.escape(_TAG_PREFIX)}\s+(\d+):")

# Project root is the parent of this file's ``tests`` directory.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_TEST_ROOTS = (_PROJECT_ROOT / "tests", _PROJECT_ROOT / "backend" / "tests")

# This meta-test is excluded from the scan; it is not a property test.
_SELF = Path(__file__).resolve()


def _iter_test_files():
    """Yield every ``*.py`` test file across both trees, excluding this guard."""
    for root in _TEST_ROOTS:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            if path.resolve() != _SELF:
                yield path


def _collect_tag_locations() -> dict[int, list[tuple[Path, int]]]:
    """Map each property number to every (file, line-index) where it is tagged.

    Also fails fast on any tag whose number falls outside the documented
    1..19 range (e.g. a typo'd ``Property 20``), which the per-number checks
    downstream would otherwise silently ignore.
    """
    locations: dict[int, list[tuple[Path, int]]] = {
        n: [] for n in EXPECTED_PROPERTY_NUMBERS
    }
    stray: list[str] = []

    for path in _iter_test_files():
        lines = path.read_text(encoding="utf-8").splitlines()
        for idx, line in enumerate(lines):
            match = _ANY_TAG.search(line)
            if not match:
                continue
            number = int(match.group(1))
            if number in locations:
                locations[number].append((path, idx))
            else:
                stray.append(f"Property {number} in {path.name}:{idx + 1}")

    assert not stray, (
        f"Found property tags outside the documented 1..{EXPECTED_PROPERTY_COUNT} "
        f"range: {stray}"
    )
    return locations


def test_all_nineteen_property_tags_present_exactly_once() -> None:
    """Every property 1..19 is tagged exactly once across the two test trees."""
    locations = _collect_tag_locations()

    missing = [n for n in EXPECTED_PROPERTY_NUMBERS if not locations[n]]
    duplicated = {
        n: [f"{p.relative_to(_PROJECT_ROOT)}:{i + 1}" for p, i in locs]
        for n, locs in locations.items()
        if len(locs) > 1
    }

    assert not missing, f"Property tags with no matching test: {missing}"
    assert not duplicated, f"Property tags matched more than once: {duplicated}"

    # Exactly 19 tagged tests, one per documented property.
    total = sum(len(locs) for locs in locations.values())
    assert total == EXPECTED_PROPERTY_COUNT, (
        f"Expected {EXPECTED_PROPERTY_COUNT} tagged property tests, found {total}"
    )


def test_each_property_tag_sits_above_a_test_function() -> None:
    """Each tag is immediately followed (past decorators) by a ``def test_``."""
    problems: list[str] = []

    for number, locs in _collect_tag_locations().items():
        assert len(locs) == 1, f"Property {number} is not tagged exactly once: {locs}"
        path, line_idx = locs[0]
        lines = path.read_text(encoding="utf-8").splitlines()

        # Walk forward from the tag to the next `def`/`async def`, allowing
        # decorators, blank lines, and comments in between.
        def_line = None
        for line in lines[line_idx + 1 :]:
            stripped = line.strip()
            if stripped.startswith(("def ", "async def ")):
                def_line = stripped
                break

        if def_line is None:
            problems.append(f"Property {number} ({path.name}): no function follows tag")
            continue
        func_name = def_line.split("def ", 1)[1].lstrip()
        if not func_name.startswith("test_"):
            problems.append(
                f"Property {number} ({path.name}): tag maps to non-test {def_line!r}"
            )

    assert not problems, "; ".join(problems)
