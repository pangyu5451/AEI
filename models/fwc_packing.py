"""Deterministic exact 3x3 bit-mask packing for immutable FWC items."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Iterable

from AGG_FWC.models.feature_combination import LEGAL_SIZES
from AGG_FWC.models.fwc_types import Item, Layout, Placement


_GRID_WIDTH = 3
_GRID_CELLS = _GRID_WIDTH * _GRID_WIDTH
_SHAPES_BY_SIZE = {
    1: ((1, 1),),
    2: ((1, 2), (2, 1)),
    3: ((1, 3), (3, 1)),
    4: ((2, 2),),
    6: ((2, 3), (3, 2)),
    9: ((3, 3),),
}


def _rectangle_mask(row, column, height, width):
    return sum(
        1 << ((row + row_offset) * _GRID_WIDTH + column + column_offset)
        for row_offset in range(height)
        for column_offset in range(width)
    )


def _placements_for_shapes(shapes):
    placements = []
    for height, width in shapes:
        for row in range(_GRID_WIDTH - height + 1):
            for column in range(_GRID_WIDTH - width + 1):
                placements.append((_rectangle_mask(row, column, height, width), (height, width)))
    return tuple(sorted(set(placements)))


PLACEMENTS_BY_SIZE = MappingProxyType(
    {size: _placements_for_shapes(shapes) for size, shapes in _SHAPES_BY_SIZE.items()}
)


@dataclass(frozen=True)
class _SearchState:
    selected: tuple[tuple[Item, int, tuple[int, int]], ...]
    total_utility: float
    occupied_mask: int


def exact_pack(candidates: Iterable[Item]) -> Layout:
    """Return the deterministic exact packing selected by the FWC 5% red rule.

    The primary objective is the largest sum of ``item.size * item.utility``.
    A layout within 95% of that optimum is eligible for the red preference,
    which orders large red items, red occupied cells, red effective utility,
    and finally feature indices.
    """
    items = _reduce_candidates(_validated_items(candidates))
    if not items:
        return Layout((), (), 0.0, 0, False)

    dominant_red = _full_board_red_layout(items)
    if dominant_red is not None:
        return dominant_red

    # A large non-red frontier has no red-priority alternative to preserve.
    # The recursive 95%-window search is exact for small candidate groups but
    # can revisit the same 3x3 occupancy state exponentially many times when
    # the FWC palette supplies 15-20 candidates.  Occupancy-state DP retains
    # the exact maximum-value layout for this branch and is the same objective
    # used when no red item is present.
    if len(items) > 8 and not any(item.color == "red" for item in items):
        return _fast_nonred_pack(items)

    ordered = tuple(sorted(items, key=lambda item: (-_value(item), item.index)))
    suffix_values = _suffix_values(ordered)
    maximum = _maximum_utility(ordered, suffix_values)
    threshold = 0.95 * maximum
    best = _best_preferred_layout(ordered, suffix_values, threshold)
    return _layout_from_state(best)


def _fast_nonred_pack(items):
    """Solve the non-red 3x3 packing objective with 512-state DP."""

    ordered = tuple(sorted(items, key=lambda item: (-_value(item), item.index)))
    states = {0: _SearchState((), 0.0, 0)}
    for item in ordered:
        previous = states
        states = dict(previous)
        value = _value(item)
        for occupied_mask, state in previous.items():
            for placement_mask, shape in PLACEMENTS_BY_SIZE[item.size]:
                if occupied_mask & placement_mask:
                    continue
                candidate = _SearchState(
                    selected=state.selected + ((item, placement_mask, shape),),
                    total_utility=state.total_utility + value,
                    occupied_mask=occupied_mask | placement_mask,
                )
                current = states.get(candidate.occupied_mask)
                if current is None or _state_is_better(candidate, current):
                    states[candidate.occupied_mask] = candidate

    maximum = max(state.total_utility for state in states.values())
    threshold = 0.95 * maximum
    eligible = [state for state in states.values() if state.total_utility >= threshold]
    return _layout_from_state(min(eligible, key=_preference_key))


def _state_is_better(candidate, current):
    if candidate.total_utility > current.total_utility:
        return True
    if candidate.total_utility < current.total_utility:
        return False
    return _preference_key(candidate) < _preference_key(current)


def _full_board_red_layout(items):
    """Apply a lossless board-capacity dominance rule before branch search.

    A red 3x3 item whose per-cell utility is at least every candidate's
    per-cell utility reaches the global upper bound for nine cells.  It is
    therefore an exact optimum, and the red-priority rule selects it among
    all layouts in the 95 percent window.  This also collapses the common
    equal-score 512-feature case to one constant-time decision.
    """
    red_items = [item for item in items if item.color == "red" and item.size == 9]
    if not red_items:
        return None
    red_item = min(red_items, key=lambda item: (-item.utility, item.index))
    if any(item.utility > red_item.utility for item in items):
        return None
    return Layout(
        selected_indices=(red_item.index,),
        placements=(Placement(red_item.index, (1 << 9) - 1, (3, 3)),),
        total_utility=_value(red_item),
        cells_used=9,
        contains_large_red=True,
    )


def _validated_items(candidates):
    try:
        items = tuple(candidates)
    except TypeError as error:
        raise ValueError("candidates must be an iterable of Item values") from error
    indices = set()
    for item in items:
        if not isinstance(item, Item):
            raise ValueError("candidates must contain Item values")
        if not isinstance(item.index, int) or isinstance(item.index, bool) or item.index < 0:
            raise ValueError("item index must be a non-negative integer")
        if not isinstance(item.color, str) or item.color not in LEGAL_SIZES:
            raise ValueError("item color must be a canonical FWC color")
        if not isinstance(item.size, int) or isinstance(item.size, bool) or item.size not in PLACEMENTS_BY_SIZE:
            raise ValueError("item size must be a legal packing size")
        if item.size not in LEGAL_SIZES[item.color]:
            raise ValueError("item size is not legal for color")
        if item.index in indices:
            raise ValueError("candidate feature indices must be unique")
        indices.add(item.index)
        if not isinstance(item.utility, (int, float)) or isinstance(item.utility, bool):
            raise ValueError("item utility must be a finite non-negative number")
        if not math.isfinite(item.utility) or item.utility < 0.0:
            raise ValueError("item utility must be a finite non-negative number")
        if not math.isfinite(_value(item)):
            raise ValueError("weighted P must be finite")
    return items


def _reduce_candidates(items):
    """Keep a lossless candidate frontier for a nine-cell board.

    At most floor(9/size) items of one color/size can occur in a layout.
    Replacing any selected member by a higher-valued item in the same class
    preserves its legal shape and color, so lower-ranked members cannot be
    part of an optimum or an eligible red-priority layout.
    """
    buckets = {}
    for item in items:
        buckets.setdefault(item.size, []).append(item)
    kept = []
    for size, bucket in buckets.items():
        limit = _GRID_CELLS // size
        ranked = sorted(bucket, key=lambda item: (-_value(item), item.index))
        red_ranked = sorted(
            (item for item in bucket if item.color == "red"),
            key=lambda item: (-_value(item), item.index),
        )
        kept.extend(ranked[:limit])
        kept.extend(item for item in red_ranked[:limit] if item not in kept)
    return tuple(kept)


def _value(item):
    return item.size * float(item.utility)


def _suffix_values(items):
    suffix = [0.0] * (len(items) + 1)
    for position in range(len(items) - 1, -1, -1):
        suffix[position] = suffix[position + 1] + _value(items[position])
    return tuple(suffix)


def _maximum_utility(items, suffix_values):
    memo = {}

    def best_additional(position, occupied_mask):
        key = (position, occupied_mask)
        if key in memo:
            return memo[key]
        if position == len(items):
            return 0.0
        item = items[position]
        result = best_additional(position + 1, occupied_mask)
        value = _value(item)
        for mask, _ in PLACEMENTS_BY_SIZE[item.size]:
            if not occupied_mask & mask:
                result = max(result, value + best_additional(position + 1, occupied_mask | mask))
        memo[key] = result
        return result

    return best_additional(0, 0)


def _best_preferred_layout(items, suffix_values, threshold):
    best = None
    memo = {}

    def max_additional(position, occupied_mask):
        key = (position, occupied_mask)
        if key in memo:
            return memo[key]
        if position == len(items):
            return 0.0
        item = items[position]
        result = max_additional(position + 1, occupied_mask)
        value = _value(item)
        for mask, _ in PLACEMENTS_BY_SIZE[item.size]:
            if not occupied_mask & mask:
                result = max(result, value + max_additional(position + 1, occupied_mask | mask))
        memo[key] = result
        return result

    def visit(position, state):
        nonlocal best
        if state.total_utility + max_additional(position, state.occupied_mask) < threshold:
            return
        if position == len(items):
            if state.total_utility >= threshold and (
                best is None or _preference_key(state) < _preference_key(best)
            ):
                best = state
            return

        visit(position + 1, state)
        item = items[position]
        for mask, shape in PLACEMENTS_BY_SIZE[item.size]:
            if not state.occupied_mask & mask:
                visit(
                    position + 1,
                    _SearchState(
                        selected=state.selected + ((item, mask, shape),),
                        total_utility=state.total_utility + _value(item),
                        occupied_mask=state.occupied_mask | mask,
                    ),
                )

    visit(0, _SearchState((), 0.0, 0))
    return best


def _preference_key(state):
    selected = tuple(sorted(state.selected, key=lambda placement: placement[0].index))
    red = tuple(placement for placement in selected if placement[0].color == "red")
    return (
        -int(any(item.size >= 6 for item, _, _ in red)),
        -sum(item.size for item, _, _ in red),
        -sum(_value(item) for item, _, _ in red),
        tuple(item.index for item, _, _ in selected),
        tuple(mask for _, mask, _ in selected),
    )


def _layout_from_state(state):
    selected = tuple(sorted(state.selected, key=lambda placement: placement[0].index))
    red = tuple(placement for placement in selected if placement[0].color == "red")
    return Layout(
        selected_indices=tuple(item.index for item, _, _ in selected),
        placements=tuple(Placement(item.index, mask, shape) for item, mask, shape in selected),
        total_utility=sum(_value(item) for item, _, _ in selected),
        cells_used=sum(item.size for item, _, _ in selected),
        contains_large_red=any(item.size >= 6 for item, _, _ in red),
    )
