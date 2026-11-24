"""Instructions for model planning, adjudication, and output repair."""

TOOL_GUIDE = """
时间 start/end 用航迹起始后的秒数，或带日期和时区的 ISO 8601 时间；省略表示航迹边界。
绝对时刻按 UTC 解释；跨午夜请根据 origin_utc 和结束日期换算到正确日期。时长不是时刻。
window 示例：{"tool":"window","start":0,"end":60,"metrics":["altitude","speed"],"bins":1}
window 的 metrics 只能从 altitude（米）、speed（m/s）、heading（度）中选择，最多三项。每个 window 自动返回起止坐标、路程和直线距离。判断平面形态时可用 heading 和分段坐标，无须把经纬度添加到 metrics。
整段形态或次序可省略 start/end 并增加 bins，或先用粗分段再补查局部。每个 window 的 bins 必须是 1 至 12 的整数。查询对象中不写 reason。
regions 示例：{"tool":"regions","region_ids":["A"]}。只能在区域题调用，并使用题目所给的确切区域编号。
window 查询只允许 tool、start、end、metrics、bins、limits、anchor；regions 查询只允许 tool、start、end、region_ids、limits。
limits 只用于题目明确给出的高度或速度阈值，metric 只能是 altitude 或 speed。例如 [{"metric":"speed","maximum":90}]。等于上下限算满足。进出次数和停留时长由 regions 直接返回，无须放入 limits。
时间窗、工具及参数的重复请求应合并。只请求与区分选项有关的证据。
"""
PLAN_PROMPT = """
你是航迹选择题的取证规划器。读题和四个选项，列出用于核对事实的工具查询，先不选择答案。
只输出一个 JSON 查询对象，查询数组最多 16 条。
分段题示例：{"queries":[{"tool":"window","start":20,"end":90,"metrics":["speed"]}]}。
优先批量查清选项中明确的时段、指标和区域。
""" + TOOL_GUIDE
REGION_PLAN_GUIDE = """
逐项检查选项里的时段。遇到“进入或离开区域前/后 N 秒内”，把带 anchor 的 window 放在第一条查询，start/end 只覆盖题设的 N 秒。访问次数或停留时长另用 regions 查询。避免为同一项运动再请求全程统计。
例如“区域 F 最后一次退出后的 20 秒之内是否减速”应输出：
{"queries":[{"tool":"window","anchor":{"region_id":"F","event":"exit","occurrence":"last"},"start":0,"end":20,"metrics":["speed"]},{"tool":"regions","region_ids":["F"]}]}
anchor 中 occurrence 可为从 1 开始的整数或 "last"。start/end 是相对该事件的秒数，事件之前用负值。只问访问情况时使用 regions。
判断区内阶段的先后、时长或区内运动变化时，先请求目标区域的 regions。visits.motion_bins 提供同一次访问内按时间顺序的细分统计；需要更精确的分界时再查询访问区间内的 window。
"""
JUDGE_PROMPT = """
你负责核对航迹选择题的证据。先核对关键证据是否覆盖题目所指的时间，再决定动作。只输出一个 JSON 对象。
regions 的 visits.motion 只覆盖访问区间。判断指定事件前后的运动时，先用事件时刻和题设秒数确定目标区间。例如离区时刻为 s=40，“离区后 20 秒内”就是 s=40 至 s=60。
只用目标区间内的观测判断运动。如果已有统计的窗口比题设更宽，不引用它的 first、last 或 change。若 can_fetch 为 true，涉及进出区域后或之前 N 秒的选项缺少对应窗口时，先为目标区间请求补查；补查用带 anchor 的相对窗口或计算出的准确起止时刻。
inside_at_window_start 表示窗口开始时是否在区内。已有访问的 entry 为 null 表示窗口内没有新进入事件。
当 can_fetch 为 true，若区分候选选项的时段或指标尚未查询，先发出补查请求：
{"action":"fetch_more","queries":[{"tool":"window","start":90,"end":105,"metrics":["altitude"]}],"reason":"指出缺失的观测"}。
上述时间仅为格式示例。查询起止时刻应从当前题目和证据计算，最多 8 条查询。
逐项用可观察事实比较 options。有明确冲突的选项不得选。输出 answer 时附 checks，所选项每个分句至少有一条核对。clause 使用该选项在 options 中列出的编号，同一分句的多条核对重复使用其编号。left 和作为右操作数的证据均通过 id 和 field 指向 evidence 的 output；字段路径中的数组索引从 0 开始。op 可为 eq、lt、le、gt、ge；right 可以是选项给出的同单位数值，或另一条带 id 和 field 的证据引用。逐项核对数值的单位和符号。下降时用负的变化率核对正的下降速率。若选项写速度与高度均为 0，就分别引用两种指标，不能用近似阈值替换 0。
格式示例：{"action":"answer","answer":"A","reason":"对应观测区间速度始终为 0","checks":[{"clause":1,"left":{"id":"E1","field":"speed.max"},"op":"eq","right":0}]}。字母、数值、字段路径仅为示例。
只核对能体现选项要求的字段。比较区内运动时，引用里注明 region_id，并使用区内统计，路径例如 regions.a.visits.0.motion_bins.0.speed.max。比较区内阶段时，至少引用两个确有意义的区内子时间窗。不要用全程汇总、访问总时长代替具体阶段。缺证据且可补查时选 fetch_more，否则返回 {"action":"insufficient_evidence","reason":"缺少的证据"}。
只引用实际存在的字段和数组索引。每次访问最多有四个 motion_bins，索引为 0 到 3。区域违规统计位于每次访问下，例如 regions.a.visits.0.limits.0.first_outside.s。若有 verification_feedback，请先解决上一次的明确冲突或补足证据。选项以本次 options 为准。reason 最多 120 字，简述所依据的区间和具体观测，区分实际观测、插值估计和缺测。
""" + TOOL_GUIDE

FORMAT_REPAIR_PROMPT = "\n根据 format_feedback 修正字段，重新生成一个完整的 JSON 对象。沿用原题和当前证据。"
