"""Numerical observations used by the planner and judge."""

from datetime import datetime, timedelta, timezone
from itertools import pairwise
import json

import numpy as np
from shapely.geometry import LineString, Point, Polygon

from .models import Limit, Metric, Query, Question, Region, RegionEvent, RegionQuery, Time, WindowQuery


EPSILON = 1e-6
EARTH_RADIUS_KM = 6371.0088


class EvidenceCalculator:
    """Run read-only queries against a single trajectory and its regions."""

    def __init__(self, question: Question) -> None:
        points = sorted(question.track.points, key=lambda point: point.time_s)
        self.times = np.array([point.time_s for point in points])
        self.coords = np.array([(point.lon, point.lat) for point in points])
        self.values = {
            metric: np.array([getattr(point, metric) for point in points], dtype=float)
            for metric in ("altitude", "speed", "heading")
        }
        self.origin = question.track.origin.astimezone(timezone.utc)
        self.regions = {region.region_id: region for region in question.regions}
        self._visits: dict[str, list[tuple[float, float]]] = {}

    def summary(self) -> dict:
        """Return the time anchor, coverage, and available metrics for planning."""
        gaps = np.diff(self.times)
        positive_gaps = gaps[gaps > 0]
        return {
            "origin_utc": self.origin.isoformat().replace("+00:00", "Z"),
            "start": self._stamp(self.times[0]), "end": self._stamp(self.times[-1]),
            "observations": len(self.times),
            "duplicate_timestamps": int(np.count_nonzero(gaps == 0)),
            "median_gap_s": float(np.median(positive_gaps)) if positive_gaps.size else None,
            "max_gap_s": float(gaps.max()) if gaps.size else None,
            "missing_values": {name: int(np.isnan(values).sum()) for name, values in self.values.items()},
        }

    def _stamp(self, seconds: float) -> dict:
        utc = self.origin + timedelta(seconds=float(seconds))
        return {"s": round(float(seconds), 6), "utc": utc.isoformat().replace("+00:00", "Z")}

    def _seconds(self, value: Time | None, default: float) -> float:
        if isinstance(value, datetime):
            return (value - self.origin).total_seconds()
        return float(default if value is None else value)

    def window(self, request: Query) -> tuple[float, float]:
        """Resolve the request to an inclusive relative time window."""
        start = self._seconds(request.start, self.times[0])
        end = self._seconds(request.end, self.times[-1])
        if isinstance(request, WindowQuery) and request.anchor is not None:
            event = self._event_time(request.anchor)
            start, end = event + start, event + end
        if start > end:
            raise ValueError("time window starts after it ends")
        return start, end

    def query_key(self, request: Query) -> str:
        """Deduplicate equivalent queries by normalized parameters."""
        start, end = self.window(request)
        values = request.model_dump(mode="json", exclude={"start", "end", "anchor"})
        values.update(start=round(start, 6), end=round(end, 6))
        values["limits"] = sorted(values["limits"], key=lambda item: json.dumps(item, sort_keys=True))
        if isinstance(request, WindowQuery):
            values["metrics"] = sorted(set(request.metrics))
        else:
            values["region_ids"] = sorted(set(request.region_ids if request.region_ids is not None else self.regions))
        return json.dumps(values, ensure_ascii=False, sort_keys=True)

    def _selection(self, start: float, end: float) -> slice:
        first = int(self.times.searchsorted(start - EPSILON))
        last = int(self.times.searchsorted(end + EPSILON, side="right"))
        return slice(first, last)

    def _measure(self, metric: Metric, selected: slice) -> dict:
        times = self.times[selected]
        values = self.values[metric][selected]
        valid = np.isfinite(values)
        observed_times, observed_values = times[valid], values[valid]
        result = {"observations": int(valid.sum()), "missing": int((~valid).sum())}
        if not observed_values.size:
            return result

        first, last = float(observed_values[0]), float(observed_values[-1])
        elapsed = float(observed_times[-1] - observed_times[0])
        result.update(
            first=first, last=last,
            first_at=self._stamp(observed_times[0]), last_at=self._stamp(observed_times[-1]),
        )
        if metric == "heading":
            changes = (np.diff(values) + 180) % 360 - 180
            result.update(turn_deg=float(np.nansum(changes)), total_turn_deg=float(np.nansum(np.abs(changes))))
        else:
            low, high = int(observed_values.argmin()), int(observed_values.argmax())
            result.update(
                min=float(observed_values[low]), min_at=self._stamp(observed_times[low]),
                max=float(observed_values[high]), max_at=self._stamp(observed_times[high]),
                change=last - first, mean_rate_per_s=(last - first) / elapsed if elapsed > 0 else None,
            )
        return result

    def _limits(self, limits: list[Limit], start: float, end: float) -> list[dict]:
        """Report observed violations and estimated duration within the window."""
        out = []
        selected = self._selection(start, end)
        left = max(0, selected.start - 1)
        right = min(len(self.times), selected.stop + 1)
        for limit in limits:
            values = self.values[limit.metric]
            low = -np.inf if limit.minimum is None else limit.minimum
            high = np.inf if limit.maximum is None else limit.maximum
            observed = values[selected]
            outside = (observed < low) | (observed > high)
            scoreable_s = violated_s = 0.0
            for i in range(left, right - 1):
                t0, t1 = self.times[i:i + 2]
                v0, v1 = values[i:i + 2]
                lo, hi = max(start, t0), min(end, t1)
                if hi <= lo or not (np.isfinite(v0) and np.isfinite(v1)):
                    continue
                duration = hi - lo
                scoreable_s += duration
                if v0 == v1:
                    violated_s += duration if v0 < low or v0 > high else 0.0
                    continue
                legal = sorted((t0 + (bound - v0) / (v1 - v0) * (t1 - t0) for bound in (low, high)))
                violated_s += duration - max(0.0, min(hi, legal[1]) - max(lo, legal[0]))
            out.append({
                **limit.model_dump(),
                "observations_outside": int(outside.sum()),
                "first_outside": self._stamp(self.times[selected][outside][0]) if outside.any() else None,
                "last_outside": self._stamp(self.times[selected][outside][-1]) if outside.any() else None,
                "estimated_outside_s": round(float(violated_s), 6),
                "interpolated_coverage_s": round(float(scoreable_s), 6),
            })
        return out

    def _window_summary(self, start: float, end: float, metrics: list[Metric]) -> dict:
        selected = self._selection(start, end)
        coords = self.coords[selected]
        timestamps = self.times[selected]
        result = {
            "start": self._stamp(start), "end": self._stamp(end), "observations": len(timestamps),
            "max_observed_gap_s": float(np.diff(timestamps).max()) if len(timestamps) > 1 else None,
            **{metric: self._measure(metric, selected) for metric in dict.fromkeys(metrics)},
        }
        if len(coords):
            result["coordinates"] = {"first": coords[0].tolist(), "last": coords[-1].tolist()}
            result["coordinates"]["path_km"] = _path_distance(coords)
            result["coordinates"]["displacement_km"] = _path_distance(coords[[0, -1]])
        if 0 < len(timestamps) <= 12:
            result["observed_points"] = [
                {"s": round(float(self.times[i]), 6), **{
                    metric: float(self.values[metric][i]) if np.isfinite(self.values[metric][i]) else None
                    for metric in dict.fromkeys(metrics)
                }}
                for i in range(selected.start, selected.stop)
            ]
        return result

    def execute(self, request: Query) -> dict:
        """Dispatch one query, with a bounded number of bins or specific regions."""
        start, end = self.window(request)
        anchor_at = start - request.start if isinstance(request, WindowQuery) and request.anchor else None
        if end < self.times[0] or start > self.times[-1]:
            return {"unavailable": "requested time is outside the recorded track"}
        start, end = max(start, float(self.times[0])), min(end, float(self.times[-1]))
        if isinstance(request, RegionQuery):
            return self._region_summary(request, start, end)

        result = self._window_summary(start, end, request.metrics)
        if anchor_at is not None:
            result["anchor_at"] = self._stamp(anchor_at)
        if request.limits:
            result["limits"] = self._limits(request.limits, start, end)
        if request.bins > 1 and end > start:
            result["bins"] = [self._window_summary(float(lo), float(hi), request.metrics)
                              for lo, hi in pairwise(np.linspace(start, end, request.bins + 1))]
        return result

    def _region_intervals(self, region_id: str) -> list[tuple[float, float]]:
        if region_id not in self.regions:
            raise ValueError("region query must use supplied region IDs")
        if region_id not in self._visits:
            self._visits[region_id] = self._crossings(self.regions[region_id])
        return self._visits[region_id]

    def _event_time(self, anchor: RegionEvent) -> float:
        visits = self._region_intervals(anchor.region_id)
        events = [lo if anchor.event == "entry" else hi for lo, hi in visits
                  if hi - lo > EPSILON and (lo > self.times[0] + EPSILON if anchor.event == "entry"
                                             else hi < self.times[-1] - EPSILON)]
        index = -1 if anchor.occurrence == "last" else anchor.occurrence - 1
        try:
            return events[index]
        except IndexError as exc:
            raise ValueError("requested region event was not observed within the recorded track") from exc

    def _region_summary(self, request: RegionQuery, start: float, end: float) -> dict:
        ids = set(request.region_ids if request.region_ids is not None else self.regions)
        if not ids or ids.difference(self.regions):
            raise ValueError("region query must use supplied region IDs")
        results, entries = {}, []
        for region_id in (key for key in self.regions if key in ids):
            intervals = self._region_intervals(region_id)
            visits, touches = [], []
            for enter, leave in intervals:
                if leave < start or enter > end:
                    continue
                if abs(leave - enter) <= EPSILON:
                    touches.append(self._stamp(enter))
                    continue
                lo, hi = max(enter, start), min(leave, end)
                entered = enter >= start - EPSILON and enter > self.times[0] + EPSILON
                departed = leave <= end + EPSILON and leave < self.times[-1] - EPSILON
                if entered:
                    entries.append((round(enter, 6), region_id))
                visit = {
                    "from": self._stamp(lo), "to": self._stamp(hi),
                    "entry": self._stamp(enter) if entered else None,
                    "exit": self._stamp(leave) if departed else None,
                    "duration_s": round(hi - lo, 6),
                    "motion": self._window_summary(lo, hi, ["altitude", "speed", "heading"]),
                }
                if visit["motion"]["observations"] >= 4:
                    # Keep ordered, local telemetry without duplicating the full window payload.
                    bins = []
                    for left, right in pairwise(np.linspace(lo, hi, 5)):
                        selected = self._selection(float(left), float(right))
                        bins.append({"start": self._stamp(left), "end": self._stamp(right),
                                     "duration_s": float(right - left),
                                     "altitude": self._measure("altitude", selected),
                                     "speed": self._measure("speed", selected)})
                    visit["motion_bins"] = bins
                if request.limits:
                    visit["limits"] = self._limits(request.limits, lo, hi)
                visits.append(visit)
            results[region_id] = {
                "visits": visits, "touches": touches,
                "inside_at_window_start": any(lo <= start <= hi for lo, hi in intervals),
                "entry_count": sum(visit["entry"] is not None for visit in visits),
                "total_duration_s": round(sum(visit["duration_s"] for visit in visits), 6),
            }
        return {
            "method": "estimated using straight segments between consecutive position observations; isolated boundary contacts are separate",
            "regions": results,
            "entry_order": [{"region_id": region_id, "at": self._stamp(at)}
                            for at, region_id in sorted(entries, key=lambda entry: entry[0])],
        }

    def _crossings(self, region: Region) -> list[tuple[float, float]]:
        anchor = region.vertices[0][0]
        polygon = Polygon([(_wrap_longitude(lon, anchor), lat) for lon, lat in region.vertices])
        if not polygon.is_valid or polygon.is_empty:
            raise ValueError(f"invalid polygon for region {region.region_id}")
        intervals = []
        if len(self.times) == 1:
            lon, lat = self.coords[0]
            if polygon.covers(Point(_wrap_longitude(lon, anchor), lat)):
                intervals.append((float(self.times[0]), float(self.times[0])))
        pxmin, pymin, pxmax, pymax = polygon.bounds
        for i, (left, right) in enumerate(pairwise(self.coords)):
            t0, t1 = self.times[i:i + 2]
            if t1 <= t0:
                continue
            if abs(left[0] - right[0]) > 180:
                right = (_wrap_longitude(right[0], left[0]), right[1])
            shift = _wrap_longitude((left[0] + right[0]) / 2, anchor) - (left[0] + right[0]) / 2
            line = LineString([(left[0] + shift, left[1]), (right[0] + shift, right[1])])
            xmin, ymin, xmax, ymax = line.bounds
            if xmax < pxmin or xmin > pxmax or ymax < pymin or ymin > pymax:
                continue
            if line.length == 0:
                if polygon.covers(Point(line.coords[0])):
                    intervals.append((float(t0), float(t1)))
                continue
            for lo, hi in _intersection_fractions(line, line.intersection(polygon)):
                intervals.append((float(t0 + lo * (t1 - t0)), float(t0 + hi * (t1 - t0))))

        merged: list[tuple[float, float]] = []
        for lo, hi in sorted(intervals):
            if merged and lo <= merged[-1][1] + EPSILON:
                merged[-1] = (merged[-1][0], max(hi, merged[-1][1]))
            else:
                merged.append((lo, hi))
        return merged


def _wrap_longitude(lon: float, center: float) -> float:
    return float(center + (lon - center + 180) % 360 - 180)


def _intersection_fractions(line: LineString, intersection):
    if intersection.is_empty:
        return
    if intersection.geom_type == "Point":
        fraction = line.project(intersection, normalized=True)
        yield fraction, fraction
    elif intersection.geom_type == "LineString":
        lo, hi = sorted(line.project(Point(intersection.coords[i]), normalized=True) for i in (0, -1))
        yield lo, hi
    else:
        for part in intersection.geoms:
            yield from _intersection_fractions(line, part)


def _path_distance(coords: np.ndarray) -> float:
    lon, lat = np.radians(coords).T
    haversine = np.sin(np.diff(lat) / 2) ** 2 + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(np.diff(lon) / 2) ** 2
    return round(float(2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(haversine, 0, 1))).sum()), 6)
