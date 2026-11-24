"""Verify cited facts against calculated evidence before accepting an answer."""

from datetime import datetime
from math import isclose, isfinite
import operator
import re

from .models import Decision, EvidenceRef


OPTION_MARK = re.compile(r"^\s*([A-D])\s*[.．、)）]\s*", re.MULTILINE)
COMPARISONS = {"lt": operator.lt, "le": operator.le, "gt": operator.gt, "ge": operator.ge}
EPSILON = 1e-6
METRIC_NAMES = {"高度": "altitude", "速度": "speed"}
EXACT_ZERO = re.compile(r"(?P<metrics>(?:高度|速度)(?:\s*[、与和及]\s*(?:高度|速度))*)"
                        r"\s*(?:均|始终|一直|全程)\s*(?:为|等于|是)\s*0(?![\d.])"
                        r"(?!\s*(?:至|到|[-–—~～]|左右))")
CLOCK_TIME = re.compile(r"(?:(\d{1,2})日)?(\d{1,2})时(\d{1,2})分(\d{1,2}(?:\.\d+)?)秒")
APPROXIMATE = re.compile(r"(?:约|近似|大致)\s*([+-]?\d+(?:\.\d+)?)(?![\d.])"
                         r"(?!\s*(?:至|到|[-–—~～])\s*\d)")


def split_options(question: str) -> tuple[str, dict[str, list[str]]]:
    """Expose stable clause indexes for the actual option text."""
    marks = list(OPTION_MARK.finditer(question))
    options = {}
    for mark, following in zip(marks, [*marks[1:], None]):
        text = question[mark.end(): following.start() if following else len(question)]
        options[mark[1]] = [part.strip() for part in re.split(r"[；;。]", text) if part.strip()]
    return (question[:marks[0].start()].strip() if marks else question), options


def option_checks(decision: Decision, options: dict[str, list[str]], evidence: list[dict], region_ids: list[str]) -> list[dict]:
    """Return precise feedback for absent, out-of-scope, or contradictory checks."""
    if not decision.checks:
        return [{"message": "Provide at least one structured evidence check for the selected option."}]

    clauses = options.get(decision.answer, [])
    items = {item["id"]: item for item in evidence}
    issues = []
    covered = set()
    checked = {}
    for check in decision.checks:
        issue = {"clause": check.clause, "check": check.model_dump(mode="json")}
        if check.clause > len(clauses):
            issues.append({**issue, "message": "The clause number is not in the selected option."})
            continue
        quote = clauses[check.clause - 1]
        issue["quote"] = quote
        mentioned = [rid for rid in region_ids if re.search(r"区域\s*" + re.escape(rid) + r"\s*(?:的)?内", quote)]
        required_region = mentioned[0] if len(mentioned) == 1 else None
        try:
            left = _numeric_field(check.left, items, required_region)
            right = (_numeric_field(check.right, items, required_region)
                     if isinstance(check.right, EvidenceRef) else check.right)
        except ValueError as exc:
            issues.append({**issue, "message": str(exc)})
            continue
        if check.op == "eq":
            tolerance = EPSILON if isinstance(check.right, EvidenceRef) else _equality_tolerance(quote, right)
            consistent = isclose(left, right, abs_tol=tolerance, rel_tol=0)
        else:
            consistent = COMPARISONS[check.op](left, right)
        if not consistent:
            issues.append({**issue, "message": "The evidence contradicts this comparison.",
                           "left_actual": left, "right_actual": right})
        else:
            covered.add(check.clause)
            checked.setdefault(check.clause, []).append((check.left, required_region))
            if isinstance(check.right, EvidenceRef):
                checked[check.clause].append((check.right, required_region))
    for number, clause in enumerate(clauses, start=1):
        if number not in covered:
            issues.append({"clause": number, "quote": clause, "message": "This option clause lacks a verified comparison."})
            continue
        for match in EXACT_ZERO.finditer(clause):
            for name in re.findall(r"高度|速度", match["metrics"]):
                metric = METRIC_NAMES[name]
                issue = {"clause": number, "quote": clause, "metric": metric}
                issues.extend(_check_exact_zero(issue, metric, checked[number], items))
        if clause.count("阶段") >= 2 and any(word in clause for word in ("先", "后", "时长", "持续时间")):
            scoped = [(ref, region) for ref, region in checked[number] if region is not None]
            if scoped and len(_local_intervals(scoped, items)) < 2:
                issues.append({"clause": number, "quote": clause,
                               "message": "Comparing in-region phases requires two distinct local subwindows, not the whole visit."})
    return issues


def _equality_tolerance(clause: str, expected: float) -> float:
    """Allow rounding only for an explicitly approximate option value."""
    for literal in APPROXIMATE.findall(clause):
        if expected and abs(float(literal)) == abs(expected):
            decimals = len(literal.partition(".")[2])
            return 0.5 * 10 ** -decimals + EPSILON
    return EPSILON


def _check_exact_zero(issue: dict, metric: str, checks: list[tuple[EvidenceRef, str | None]],
                      items: dict[str, dict]) -> list[dict]:
    """Read extrema from cited windows for a literal all-zero statement."""
    observed = []
    for ref, region in checks:
        fields = _path(ref.field)
        if len(fields) < 2 or fields[-2] != metric:
            continue
        parent = ".".join(fields[:-1])
        try:
            low = _numeric_field(ref.model_copy(update={"field": parent + ".min"}), items, region)
            high = _numeric_field(ref.model_copy(update={"field": parent + ".max"}), items, region)
        except ValueError:
            continue
        _, scope = _field_with_window(items[ref.id]["output"], fields)
        observed.append((ref.id, low, high, _matches_clock_window(issue["quote"], scope)))
    if not observed:
        return [{**issue, "message": f"This all-zero statement requires cited {metric} extrema."}]
    return [{**issue, "message": f"{eid} contradicts the literal all-zero {metric} statement.",
             "evidence_id": eid, "observed_min": low, "observed_max": high, "required_value": 0,
             "excludes_option": exact_window}
            for eid, low, high, exact_window in observed if max(abs(low), abs(high)) > EPSILON]


def _matches_clock_window(clause: str, scope: dict) -> bool:
    """Confirm both option clock times match the actual cited window before exclusion."""
    times = CLOCK_TIME.findall(clause)
    if len(times) != 2 or "start" not in scope or "end" not in scope:
        return False
    for (day, hour, minute, second), key in zip(times, ("start", "end")):
        actual = datetime.fromisoformat(scope[key]["utc"])
        if day and actual.day != int(day):
            return False
        if (actual.hour, actual.minute) != (int(hour), int(minute)):
            return False
        if not isclose(actual.second + actual.microsecond / 1e6, float(second), abs_tol=EPSILON, rel_tol=0):
            return False
    return True


def _path(field: str) -> list[str]:
    return re.sub(r"\[(\d+)\]", r".\1", field).split(".")


def _resolve_path(output: dict, field: str) -> list[str]:
    fields = _path(field)
    if len(fields) > 2 and fields[0] == "regions" and fields[2] == "limits":
        visits = output.get("regions", {}).get(fields[1], {}).get("visits", [])
        if len(visits) == 1:
            return [*fields[:2], "visits", "0", *fields[2:]]
        if len(visits) > 1:
            raise ValueError(f"{field} is ambiguous; use regions.{fields[1]}.visits.<index>.limits.")
    return fields


def _field_with_window(output: dict, fields: list[str]) -> tuple[object, dict]:
    value = window = output

    for part in fields:
        if isinstance(value, dict):
            value = value.get(part)
        elif isinstance(value, list) and part.isascii() and part.isdecimal() and len(part) <= 6:
            index = int(part)
            value = value[index] if index < len(value) else None
        else:
            value = None
        if isinstance(value, dict) and "start" in value and "end" in value:
            window = value
    return value, window


def _region_windows(items: dict[str, dict], region: str) -> list[tuple[float, float]]:
    return [(visit["from"]["s"], visit["to"]["s"]) for entry in items.values()
            for visit in entry["output"].get("regions", {}).get(region, {}).get("visits", [])]


def _local_intervals(refs: list[tuple[EvidenceRef, str]], items: dict[str, dict]) -> set[tuple[float, float]]:
    intervals = set()
    for ref, region in refs:
        output = items[ref.id]["output"]
        _, scope = _field_with_window(output, _resolve_path(output, ref.field))
        if "start" not in scope or "end" not in scope:
            continue
        start, end = scope["start"]["s"], scope["end"]["s"]
        if any(lo - EPSILON <= start < end <= hi + EPSILON and end - start < hi - lo - EPSILON
               for lo, hi in _region_windows(items, region)):
            intervals.add((start, end))
    return intervals


def _numeric_field(ref: EvidenceRef, items: dict[str, dict], required_region: str | None) -> float:
    item = items.get(ref.id)
    if item is None:
        raise ValueError(f"Unknown evidence ID {ref.id}.")
    region = ref.region_id or required_region
    if required_region is not None and region != required_region:
        raise ValueError(f"This claim requires evidence from region {required_region}.")

    fields = _resolve_path(item["output"], ref.field)
    value, window = _field_with_window(item["output"], fields)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(f"{ref.id}.{ref.field} does not resolve to a finite number.")

    if region is not None:
        if item["query"]["tool"] == "regions":
            if fields[:2] != ["regions", region]:
                raise ValueError(f"{ref.id}.{ref.field} is outside region {region}.")
        else:
            start, end = window["start"]["s"], window["end"]["s"]
            if not any(lo - EPSILON <= start <= end <= hi + EPSILON for lo, hi in _region_windows(items, region)):
                raise ValueError(f"{ref.id}.{ref.field} uses a window that is not inside region {region}.")
    return value
