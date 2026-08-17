"""
Unit tests for matching.py -- the IOU + Hungarian assignment logic.

Run with:
    cd src/tracking && python -m pytest ../../tests/test_matching.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "tracking"))
from matching import iou_batch, linear_assignment_iou  # noqa: E402


def test_iou_identical_boxes_is_one():
    box = np.array([[0, 0, 10, 10]])
    result = iou_batch(box, box)
    assert result.shape == (1, 1)
    assert np.isclose(result[0, 0], 1.0)


def test_iou_disjoint_boxes_is_zero():
    box_a = np.array([[0, 0, 10, 10]])
    box_b = np.array([[100, 100, 110, 110]])
    result = iou_batch(box_a, box_b)
    assert np.isclose(result[0, 0], 0.0)


def test_iou_partial_overlap():
    # box_a: (0,0)-(10,10) area=100
    # box_b: (5,5)-(15,15) area=100, intersection: (5,5)-(10,10) area=25
    # union = 100+100-25=175 -> iou = 25/175
    box_a = np.array([[0, 0, 10, 10]])
    box_b = np.array([[5, 5, 15, 15]])
    result = iou_batch(box_a, box_b)
    assert np.isclose(result[0, 0], 25 / 175, atol=1e-4)


def test_linear_assignment_matches_best_overlap():
    tracks = np.array([[0, 0, 10, 10], [50, 50, 60, 60]])
    dets = np.array([[1, 1, 11, 11], [51, 51, 61, 61]])
    matches, unmatched_t, unmatched_d = linear_assignment_iou(tracks, dets, iou_threshold=0.3)

    assert len(matches) == 2
    assert len(unmatched_t) == 0
    assert len(unmatched_d) == 0
    # track 0 should match det 0, track 1 should match det 1
    assert (0, 0) in matches
    assert (1, 1) in matches


def test_linear_assignment_rejects_low_iou():
    # boxes barely touch -> IOU below threshold -> should NOT be matched
    tracks = np.array([[0, 0, 10, 10]])
    dets = np.array([[9, 9, 20, 20]])
    matches, unmatched_t, unmatched_d = linear_assignment_iou(tracks, dets, iou_threshold=0.3)

    assert len(matches) == 0
    assert unmatched_t == [0]
    assert unmatched_d == [0]


def test_empty_inputs_do_not_crash():
    matches, unmatched_t, unmatched_d = linear_assignment_iou(
        np.empty((0, 4)), np.empty((0, 4))
    )
    assert matches == []
    assert unmatched_t == []
    assert unmatched_d == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
