"""Finish coverage of ghost-derived routes, independent of game assets."""

import importlib.util
from pathlib import Path
import sys

import pytest


spec = importlib.util.spec_from_file_location(
    "route_centerline_test_module",
    Path(__file__).resolve().parents[1] / "tools/generate_route_centerline.py",
)
route = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = route
spec.loader.exec_module(route)
Point = route.Point


class FlatRoad:
    def nearest_layer(self, *args):
        return 0

    def half_width(self, position, tangent):
        return 8.0


@pytest.mark.parametrize("gap", [22.0, 5.0, 0.1, 0.0])
@pytest.mark.parametrize("finish", [40.0, 40.1])
def test_ghost_route_reaches_grounded_finish_without_mutating_input(gap, finish):
    anchors = [Point(0, 0, 0), Point(0, 0, finish)]
    legs = [[Point(0, 0, 0), Point(0, 0, finish - gap)]]
    original = [list(leg) for leg in legs]
    points = route.resample(
        legs, anchors, FlatRoad(), Point(0, route.CAR_SURFACE_OFFSET, 0), False
    )
    assert legs == original
    assert points[-1].position == Point(0, route.CAR_SURFACE_OFFSET, finish)
    assert points[-1].arc_length == pytest.approx(finish)
    assert all(b.arc_length > a.arc_length for a, b in zip(points, points[1:]))
    assert max(route.distance(a.position, b.position)
               for a, b in zip(points, points[1:])) <= route.OUTPUT_SPACING + 0.25


def test_finish_extension_preserves_offset_checkpoint_crossing():
    anchors = [Point(0, 0, 0), Point(0, 0, 20), Point(0, 0, 60)]
    crossing = Point(6, 0, 20)
    legs = [[Point(0, 0, 0), crossing], [crossing, Point(6, 0, 38)]]
    points = route.resample(
        legs, anchors, FlatRoad(), Point(0, route.CAR_SURFACE_OFFSET, 0), False
    )
    assert any(p.position == Point(6, route.CAR_SURFACE_OFFSET, 20) for p in points)
    assert points[-1].position == Point(0, route.CAR_SURFACE_OFFSET, 60)
    assert points[-1].leg_index == 1


def test_lap_route_closes_at_finish_without_collapsing_to_start():
    anchors = [Point(0, 0, 0), Point(20, 0, 20), Point(0, 0, 0)]
    legs = [
        [Point(0, 0, 0), Point(20, 0, 0), Point(20, 0, 20)],
        [Point(20, 0, 20), Point(0, 0, 20), Point(0, 0, 8)],
    ]
    points = route.resample(
        legs, anchors, FlatRoad(), Point(0, route.CAR_SURFACE_OFFSET, 0), False
    )
    assert points[-1].position == points[0].position
    assert points[-1].arc_length == pytest.approx(80)
    assert points[-1].leg_index == 1
    assert all(b.arc_length > a.arc_length for a, b in zip(points, points[1:]))


def test_finish_extension_preserves_existing_airborne_approach():
    class LandingRoad(FlatRoad):
        def nearest_layer(self, ix, iz, y, tolerance):
            return 0 if abs(y) <= tolerance else None

    anchors = [Point(0, 0, 0), Point(0, 0, 60)]
    legs = [[Point(0, 0, 0), Point(0, 6, 20), Point(0, 6, 40)]]
    points = route.resample(
        legs, anchors, LandingRoad(), Point(0, route.CAR_SURFACE_OFFSET, 0), False
    )
    # The extension must not flatten the flown portion or produce missing
    # widths when it is resampled above the road. Real airborne finish
    # placement is a separate asset-dependent validation requirement.
    assert any(p.position.y == pytest.approx(6 + route.CAR_SURFACE_OFFSET)
               for p in points if 25 < p.position.z < 35)
    assert all(p.half_width > 0 for p in points)
    assert points[-1].position == Point(0, route.CAR_SURFACE_OFFSET, 60)
