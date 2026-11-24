"""Known-answer synthetic cases and scripted replies for the CLI demos."""

from .llm import ResponseModel
from .models import Question, QuestionType


def demo_question(kind: QuestionType) -> tuple[Question, str]:
    """Build a synthetic sample with a known answer for API smoke testing."""
    if kind == "segment":
        heights = [1000, 1300, 1600, 1600, 1600, 1450, 1300]
        points = [{"时间戳": 30 * i, "经度": 0.1 * i, "纬度": 0,
                   "高度": height, "速度": 80, "航向角": 90}
                  for i, height in enumerate(heights)]
        instruction = (
            "根据高度，以下哪一项符合航迹？\n"
            "A. 0–60秒爬升、60–120秒平飞、120–180秒爬升。\n"
            "B. 0–60秒爬升、60–120秒平飞、120–180秒下降。\n"
            "C. 0–60秒下降、60–120秒平飞、120–180秒下降。\n"
            "D. 0–60秒爬升、60–120秒下降、120–180秒下降。"
        )
        regions, expected = [], "B"
    else:
        points = [{"时间戳": t, "经度": lon, "纬度": 0.5, "高度": h, "速度": 30}
                  for t, lon, h in [(0, 0.5, 100), (10, 1.5, 100), (20, 0.5, 100),
                                    (30, 1.5, 100), (35, 2.0, 150), (40, 2.5, 75), (50, 3.5, 40)]]
        regions = [{"区域编号": "A", "类型": "多边形", "坐标": [[0, 0], [1, 0], [1, 1], [0, 1]]}]
        instruction = (
            "按相邻采样点之间直线匀速估算，区域 A 内的访问及运动，哪一项正确？\n"
            "A. 起点在区域外，累计停留 15 秒。\n"
            "B. 从区外进入两次，累计停留 15 秒。\n"
            "C. 从区外进入一次，累计停留 15 秒，最后一次离开后的 10 秒内高度上升。\n"
            "D. 从区外进入一次，累计停留 15 秒，最后一次离开后的 10 秒内高度保持不变。"
        )
        expected = "C"
    row = {
        "kind": kind, "样本编号": f"demo_{kind}", "文本指令": instruction, "区域": regions,
        "航迹数据": {"航迹编号": f"synthetic_{kind}", "起始时间": "2025-01-01T23:59:40Z", "点位序列": points},
    }
    return Question.model_validate(row), expected


class OfflineDemoModel:
    """Scripted responses for built-in examples, selected only by demo --offline."""

    def __init__(self, case: QuestionType) -> None:
        if case == "segment":
            plan = {"queries": [{"tool": "window", "start": start, "end": start + 60, "metrics": ["altitude"]}
                                for start in (0, 60, 120)]}
            answer = {"action": "answer", "answer": "B", "reason": "高度先升、后保持、再下降。", "checks": [
                {"clause": 1, "left": {"id": eid, "field": "altitude.change"}, "op": op, "right": 0}
                for eid, op in (("E1", "gt"), ("E2", "eq"), ("E3", "lt"))
            ]}
        else:
            plan = {"queries": [
                {"tool": "regions", "region_ids": ["A"]},
                {"tool": "window", "start": 0, "end": 10, "metrics": ["altitude"],
                 "anchor": {"region_id": "A", "event": "exit", "occurrence": "last"}},
            ]}
            answer = {"action": "answer", "answer": "C", "reason": "入区一次，累计停留 15 秒，最后离开后高度上升。", "checks": [
                {"clause": 1, "left": {"id": "E1", "field": "regions.A.entry_count"}, "op": "eq", "right": 1},
                {"clause": 1, "left": {"id": "E1", "field": "regions.A.total_duration_s"}, "op": "eq", "right": 15},
                {"clause": 1, "left": {"id": "E2", "field": "altitude.change"}, "op": "gt", "right": 0},
            ]}
        self.responses = iter((plan, answer))

    def ask(self, instructions: str, payload: dict, schema: type[ResponseModel], *, reasoning: bool = False
            ) -> tuple[ResponseModel, dict]:
        return schema.model_validate(next(self.responses)), {"model": "offline-demo"}
