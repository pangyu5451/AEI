"""Behavioral tests for deterministic exact 3x3 FWC packing."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import itertools
import math
import random
import subprocess
import sys
from pathlib import Path

import pytest

from AGG_FWC.models.fwc_packing import PLACEMENTS_BY_SIZE, exact_pack
from AGG_FWC.models.fwc_types import Item


def test_packer_uses_a_legal_orientation_for_a_single_six_cell_item():
    layout = exact_pack([Item(index=7, color="blue", size=6, utility=1.0)])

    assert layout.selected_indices == (7,)
    assert layout.cells_used == 6
    assert layout.total_utility == pytest.approx(6.0)
    assert layout.placements[0].shape in {(2, 3), (3, 2)}
    with pytest.raises(FrozenInstanceError):
        layout.cells_used = 0


def test_precomputed_masks_have_the_expected_area_and_cover_both_six_cell_rotations():
    six_cell_shapes = {shape for _, shape in PLACEMENTS_BY_SIZE[6]}

    assert {(2, 3), (3, 2)} <= six_cell_shapes
    assert all(mask.bit_count() == size for size, placements in PLACEMENTS_BY_SIZE.items() for mask, _ in placements)
    assert all(0 < mask < (1 << 9) for placements in PLACEMENTS_BY_SIZE.values() for mask, _ in placements)


def test_packer_never_returns_overlapping_placements():
    items = (
        Item(index=0, color="blue", size=4, utility=1.0),
        Item(index=1, color="green", size=2, utility=0.8),
        Item(index=2, color="gray", size=2, utility=0.7),
    )

    layout = exact_pack(items)

    occupied = 0
    for placement in layout.placements:
        assert not occupied & placement.mask
        occupied |= placement.mask
    assert occupied.bit_count() == layout.cells_used


def test_packer_returns_an_empty_immutable_layout_for_no_candidates():
    layout = exact_pack(())

    assert layout.selected_indices == ()
    assert layout.placements == ()
    assert layout.total_utility == 0.0
    assert layout.cells_used == 0
    assert not layout.contains_large_red


def test_nine_cell_red_is_selected_and_marks_the_large_red_diagnostic():
    red = Item(index=4, color="red", size=9, utility=0.7)

    layout = exact_pack((red,))

    assert layout.selected_indices == (4,)
    assert layout.cells_used == 9
    assert layout.contains_large_red
    assert layout.placements[0].shape == (3, 3)


def test_exact_pack_rejects_nonfinite_weighted_objective():
    with pytest.raises(ValueError, match="finite"):
        exact_pack((Item(index=0, color="red", size=9, utility=1e308),))


def test_exact_pack_handles_all_512_candidates():
    candidates = tuple(
        Item(index=i, color="gray", size=1, utility=1.0 - i / 10000.0)
        for i in range(512)
    )
    layout = exact_pack(candidates)
    assert layout.cells_used == 9
    assert layout.total_utility == pytest.approx(sum(1.0 - i / 10000.0 for i in range(9)))


@pytest.mark.parametrize("size", (0, 5, 7, 8, 10))
def test_packer_rejects_illegal_or_impossible_item_shapes(size):
    with pytest.raises(ValueError, match="legal packing size"):
        exact_pack((Item(index=0, color="blue", size=size, utility=1.0),))


@pytest.mark.parametrize("utility", (math.inf, -math.inf, math.nan, -0.1))
def test_packer_rejects_nonfinite_or_negative_utility(utility):
    with pytest.raises(ValueError, match="utility"):
        exact_pack((Item(index=0, color="blue", size=1, utility=utility),))


def test_packer_rejects_finite_utility_when_its_weighted_p_is_not_finite():
    with pytest.raises(ValueError, match="weighted P"):
        exact_pack((Item(index=0, color="red", size=9, utility=sys.float_info.max),))


def test_packer_prefers_large_red_when_its_value_is_within_five_percent_of_global_maximum():
    red = Item(index=0, color="red", size=6, utility=0.95)
    alternatives = tuple(Item(index=index, color="blue", size=1, utility=1.0) for index in range(1, 10))

    layout = exact_pack((red, *alternatives))

    assert layout.total_utility == pytest.approx(8.7)
    assert layout.contains_large_red
    assert layout.selected_indices == (0, 1, 2, 3)


def test_packer_keeps_the_global_maximum_when_large_red_falls_outside_five_percent_window():
    red = Item(index=0, color="red", size=6, utility=0.90)
    alternatives = tuple(Item(index=index, color="blue", size=1, utility=1.0) for index in range(1, 10))

    layout = exact_pack((red, *alternatives))

    assert layout.total_utility == pytest.approx(9.0)
    assert not layout.contains_large_red
    assert layout.selected_indices == tuple(range(1, 10))


def test_packer_does_not_apply_red_priority_below_the_exact_five_percent_threshold():
    blue = Item(index=1, color="blue", size=6, utility=1.0 / 6.0)
    red = Item(index=0, color="red", size=6, utility=0.9499999999994999 / 6.0)

    actual = exact_pack((red, blue))
    expected = _exhaustive_reference((red, blue))

    assert actual == expected
    assert actual.total_utility == 1.0
    assert actual.selected_indices == (1,)
    assert not actual.contains_large_red


@pytest.mark.parametrize(
    "item, message",
    (
        (Item(index=0, color="orange", size=1, utility=1.0), "canonical FWC color"),
        (Item(index=-1, color="blue", size=1, utility=1.0), "non-negative integer"),
        (Item(index=True, color="blue", size=1, utility=1.0), "non-negative integer"),
        (Item(index=0, color="green", size=9, utility=1.0), "not legal for color"),
    ),
)
def test_packer_rejects_invalid_canonical_candidate_fields(item, message):
    with pytest.raises(ValueError, match=message):
        exact_pack((item,))


def test_packer_rejects_duplicate_feature_indices():
    candidates = (
        Item(index=2, color="blue", size=1, utility=1.0),
        Item(index=2, color="red", size=6, utility=1.0),
    )

    with pytest.raises(ValueError, match="feature indices must be unique"):
        exact_pack(candidates)


def test_packer_matches_independent_exhaustive_reference_on_random_small_groups():
    rng = random.Random(20260909)
    for case_index in range(30):
        count = rng.randrange(0, 9)
        items = []
        for index in range(count):
            color = "red" if rng.randrange(4) == 0 else "blue"
            items.append(
                Item(
                    index=index,
                    color=color,
                    size=rng.choice(_REFERENCE_LEGAL_SIZES[color]),
                    utility=round(rng.uniform(0.05, 1.5), 3),
                )
            )
        items = tuple(items)

        actual = exact_pack(items)
        expected = _exhaustive_reference(items)

        assert actual == expected, f"case {case_index}: {items!r}"


def test_packer_resolves_equal_scores_and_equal_placements_deterministically_by_feature_index():
    items = (
        Item(index=9, color="blue", size=4, utility=1.0),
        Item(index=3, color="blue", size=4, utility=1.0),
        Item(index=5, color="blue", size=1, utility=1.0),
    )

    layouts = tuple(exact_pack(tuple(reversed(items))) for _ in range(5))

    assert all(layout == layouts[0] for layout in layouts)
    assert layouts[0].selected_indices == (3, 5)
    assert tuple(placement.item_index for placement in layouts[0].placements) == (3, 5)


def test_packer_finishes_for_512_assign_items_candidates_within_reasonable_bound():
    script = """
import time
import torch

from AGG_FWC.models.feature_combination import assign_items
from AGG_FWC.models.fwc_packing import exact_pack

c = torch.linspace(1.0, 0.0, 512).unsqueeze(0)
s = torch.linspace(0.95, 0.55, 512).unsqueeze(0)
r = torch.linspace(0.05, 0.45, 512).unsqueeze(0)
items = assign_items(c, s, r)[0]
started = time.perf_counter()
layout = exact_pack(items)
elapsed = time.perf_counter() - started
occupied = 0
for placement in layout.placements:
    assert not occupied & placement.mask
    occupied |= placement.mask
assert occupied.bit_count() == layout.cells_used <= 9
print(elapsed)
"""
    try:
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[2],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("packing 512 assign_items candidates exceeded 10 seconds")

    assert completed.returncode == 0, completed.stderr
    # The exact solver is intentionally retained; this guards against the
    # previous unbounded 20+ second behavior while allowing CI variance.
    assert float(completed.stdout.strip()) < 6.0


def test_packer_uses_bounded_dynamic_program_for_large_nonred_frontier():
    script = """
from AGG_FWC.models.fwc_packing import exact_pack
from AGG_FWC.models.fwc_types import Item
items=(
Item(0,'gray',2,.160),Item(1,'green',2,.155),Item(2,'gray',2,.153),Item(3,'gray',2,.149),
Item(4,'blue',3,.147),Item(5,'blue',3,.140),Item(6,'gold',3,.134),Item(7,'gray',1,.183),
Item(8,'gray',1,.171),Item(9,'gray',1,.168),Item(10,'blue',1,.166),Item(11,'gray',1,.165),
Item(12,'gray',1,.165),Item(13,'gray',1,.162),Item(14,'green',1,.161),Item(15,'gray',1,.159),
Item(16,'blue',4,.135),Item(17,'blue',4,.133),Item(18,'blue',6,.127))
layout=exact_pack(items)
assert layout.cells_used <= 9
assert layout.total_utility > 0.0
print('ok')
"""
    try:
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[2],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("large non-red packing frontier exceeded 5 seconds")

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ok"


_REFERENCE_LEGAL_SIZES = {
    "gray": (1, 2),
    "green": (1, 2),
    "blue": (1, 2, 3, 4, 6),
    "purple": (1, 2, 4, 6),
    "gold": (1, 2, 3, 4, 6),
    "red": (1, 2, 4, 6, 9),
}
_REFERENCE_SHAPES = {
    1: ((1, 1),),
    2: ((1, 2), (2, 1)),
    3: ((1, 3), (3, 1)),
    4: ((2, 2),),
    6: ((2, 3), (3, 2)),
    9: ((3, 3),),
}


def _exhaustive_reference(items):
    _reference_validate_items(items)
    ordered = tuple(sorted(items, key=lambda item: item.index))
    best_total = -1.0

    def visit(position, occupied, selected, placements, total):
        nonlocal best_total
        if position == len(ordered):
            if total > best_total:
                best_total = total
            return

        visit(position + 1, occupied, selected, placements, total)
        item = ordered[position]
        for mask, shape in _reference_placements(item.size):
            if not occupied & mask:
                visit(
                    position + 1,
                    occupied | mask,
                    selected + (item,),
                    placements + ((item, mask, shape),),
                    total + item.size * item.utility,
                )

    visit(0, 0, (), (), 0.0)
    threshold = 0.95 * best_total
    eligible = _all_eligible_layouts(ordered, threshold)
    selected, placements, total = min(eligible, key=_reference_order_key)
    from AGG_FWC.models.fwc_types import Layout, Placement

    return Layout(
        selected_indices=tuple(item.index for item in selected),
        placements=tuple(Placement(item.index, mask, shape) for item, mask, shape in placements),
        total_utility=total,
        cells_used=sum(item.size for item in selected),
        contains_large_red=any(item.color == "red" and item.size >= 6 for item in selected),
    )


def _all_eligible_layouts(items, threshold):
    layouts = []

    def visit(position, occupied, selected, placements, total):
        if position == len(items):
            if total >= threshold:
                layouts.append((selected, placements, total))
            return
        visit(position + 1, occupied, selected, placements, total)
        item = items[position]
        for mask, shape in _reference_placements(item.size):
            if not occupied & mask:
                visit(
                    position + 1,
                    occupied | mask,
                    selected + (item,),
                    placements + ((item, mask, shape),),
                    total + item.size * item.utility,
                )

    visit(0, 0, (), (), 0.0)
    return layouts


def _reference_order_key(layout):
    selected, placements, _ = layout
    red_items = tuple(item for item in selected if item.color == "red")
    return (
        -int(any(item.size >= 6 for item in red_items)),
        -sum(item.size for item in red_items),
        -sum(item.size * item.utility for item in red_items),
        tuple(item.index for item in selected),
        tuple(mask for _, mask, _ in placements),
    )


def _reference_placements(size):
    placements = []
    for height, width in _REFERENCE_SHAPES[size]:
        for row, column in itertools.product(range(4 - height), range(4 - width)):
            mask = sum(1 << ((row + row_offset) * 3 + column + column_offset) for row_offset in range(height) for column_offset in range(width))
            placements.append((mask, (height, width)))
    return tuple(sorted(set(placements)))


def _reference_validate_items(items):
    indices = set()
    for item in items:
        if not isinstance(item, Item):
            raise ValueError("reference expects Item values")
        if not isinstance(item.index, int) or isinstance(item.index, bool) or item.index < 0:
            raise ValueError("reference index must be a non-negative integer")
        if not isinstance(item.color, str) or item.color not in _REFERENCE_LEGAL_SIZES:
            raise ValueError("reference color must be canonical")
        if item.size not in _REFERENCE_LEGAL_SIZES[item.color]:
            raise ValueError("reference size is not legal for color")
        if not isinstance(item.utility, (int, float)) or isinstance(item.utility, bool):
            raise ValueError("reference utility must be finite and non-negative")
        if not math.isfinite(item.utility) or item.utility < 0.0:
            raise ValueError("reference utility must be finite and non-negative")
        if item.index in indices:
            raise ValueError("reference feature indices must be unique")
        indices.add(item.index)
