"""Contest input schemas and the model-facing query protocol."""

from typing import Annotated, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, FiniteFloat, PositiveInt, model_validator

QuestionType = Literal["segment", "region"]
Answer = Literal["A", "B", "C", "D"]
Metric = Literal["altitude", "speed", "heading"]
Time = FiniteFloat | AwareDatetime


class Point(BaseModel):
    """A measured position and optional telemetry."""

    time_s: FiniteFloat = Field(alias="时间戳")
    lon: FiniteFloat = Field(alias="经度", ge=-180, le=180)
    lat: FiniteFloat = Field(alias="纬度", ge=-90, le=90)
    altitude: FiniteFloat | None = Field(None, alias="高度")
    speed: FiniteFloat | None = Field(None, alias="速度")
    heading: FiniteFloat | None = Field(None, alias="航向角")


class Track(BaseModel):
    """A trajectory with a UTC anchor and relative observation times."""

    track_id: str = Field(alias="航迹编号")
    origin: AwareDatetime = Field(alias="起始时间")
    points: list[Point] = Field(alias="点位序列", min_length=1)


class Region(BaseModel):
    """A polygon in longitude, latitude order."""

    region_id: str = Field(alias="区域编号", min_length=1)
    vertices: list[tuple[FiniteFloat, FiniteFloat]] = Field(alias="坐标", min_length=3)


class Question(BaseModel):
    """A Task 1 input record with the source file's question type."""

    kind: QuestionType
    sample_id: str = Field(alias="样本编号", min_length=1)
    instruction: str = Field(alias="文本指令", min_length=1)
    track: Track = Field(alias="航迹数据")
    regions: list[Region] = Field(default_factory=list, alias="区域")

    @model_validator(mode="after")
    def check_regions(self) -> "Question":
        """Require uniquely addressable regions in a region question."""
        ids = [region.region_id for region in self.regions]
        if self.kind == "region" and not ids:
            raise ValueError("a region question must provide a polygon")
        if len(ids) != len(set(ids)):
            raise ValueError("region IDs must be unique")
        return self


class ProtocolModel(BaseModel):
    """Reject unexpected keys in instructions coming from the model."""

    model_config = ConfigDict(extra="forbid")


class Limit(ProtocolModel):
    """An inclusive telemetry bound explicitly supplied in a question."""

    metric: Literal["altitude", "speed"]
    minimum: FiniteFloat | None = None
    maximum: FiniteFloat | None = None

    @model_validator(mode="after")
    def check_bounds(self) -> "Limit":
        """Require at least one coherent bound."""
        if self.minimum is None and self.maximum is None:
            raise ValueError("a limit needs a minimum or maximum")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("minimum must not exceed maximum")
        return self


class TimeQuery(ProtocolModel):
    """Common window and user-supplied limits for evidence tools."""

    start: Time | None = None
    end: Time | None = None
    limits: list[Limit] = Field(default_factory=list, max_length=4)


class RegionEvent(ProtocolModel):
    """An entry or exit used as the origin of a relative window."""

    region_id: str
    event: Literal["entry", "exit"]
    occurrence: PositiveInt | Literal["last"] = "last"


class WindowQuery(TimeQuery):
    """Request statistics for a window or the entire trajectory."""

    tool: Literal["window"]
    metrics: list[Metric] = Field(default_factory=lambda: ["altitude", "speed", "heading"], min_length=1, max_length=3)
    bins: int = Field(default=1, ge=1, le=12)
    anchor: RegionEvent | None = None

    @model_validator(mode="after")
    def check_anchor(self) -> "WindowQuery":
        """Require explicit numeric offsets when referring to a region event."""
        if self.anchor and not (isinstance(self.start, float) and isinstance(self.end, float)):
            raise ValueError("anchored windows need numeric start and end offsets")
        return self


class RegionQuery(TimeQuery):
    """Request visits to one or more supplied regions."""

    tool: Literal["regions"]
    region_ids: list[str] | None = None


Query = Annotated[WindowQuery | RegionQuery, Field(discriminator="tool")]


class Plan(ProtocolModel):
    """The first batch of validated evidence requests."""

    queries: list[Query] = Field(min_length=1, max_length=16)


class EvidenceRef(ProtocolModel):
    """Address a numeric field and optionally restrict it to a region."""

    id: str = Field(pattern=r"^E[1-9]\d*$")
    field: str = Field(pattern=r"^[^.]+(?:\.[^.]+)*$", max_length=160)
    region_id: str | None = None


class EvidenceCheck(ProtocolModel):
    """A statement from the selected option and a reproducible comparison."""

    clause: PositiveInt
    left: EvidenceRef
    op: Literal["eq", "lt", "le", "gt", "ge"]
    right: FiniteFloat | EvidenceRef


class Decision(ProtocolModel):
    """A final answer or a specific request for supplementary evidence."""

    action: Literal["answer", "fetch_more", "insufficient_evidence"]
    answer: Answer | None = None
    queries: list[Query] = Field(default_factory=list, max_length=8)
    reason: str = Field(default="", max_length=1200)
    checks: list[EvidenceCheck] = Field(default_factory=list, max_length=16)

    @model_validator(mode="after")
    def check_action(self) -> "Decision":
        """Enforce mutually exclusive answer and tool requests."""
        if self.action == "answer" and (self.answer is None or self.queries):
            raise ValueError("answer requires an option and no queries")
        if self.action == "fetch_more" and (self.answer is not None or not self.queries or self.checks):
            raise ValueError("fetch_more requires queries and no answer or checks")
        if self.action == "insufficient_evidence" and (self.answer is not None or self.queries or self.checks):
            raise ValueError("insufficient_evidence takes only a reason")
        return self
