# 网信杯 B01 Task 1 智能体

用 LangGraph 处理分段题和区域题。每次运行图处理一道选择题。模型制定取证计划，Python 批量计算证据，模型根据证据作答。作答后由 Python 核对结构化的证据引用和比较结果。每题允许补查一轮，复核未通过时允许重判一次。`plan` 和 `judge` 各有一次独立的格式重试机会，反馈包含校验失败的字段。

## 安装和离线验证

在仓库根目录使用 Python 3.12 和 `uv` 安装项目。

```console
uv sync --locked --python 3.12
uv run --no-sync python -m unittest discover -s tests -v
uv run --no-sync task1-agent demo --case segment --offline
uv run --no-sync task1-agent demo --case region --offline
```

`task1-agent` 与 `python -m task1_agent` 都可以启动命令行。离线演示使用仓库内的自建数据和预设模型响应，经过实际的图、Python 取证和复核节点，无需 API 密钥。GitHub Actions 在 Windows 和 Ubuntu 上运行相同的离线验证。

## 模型配置和在线演示

参照 [`.env.example`](.env.example) 在当前终端或部署环境设置变量。程序读取环境变量。在线演示从 `SILICONFLOW_API_KEY` 读取凭证，`SILICONFLOW_MODEL` 的默认值为 `Qwen/Qwen3.5-9B`。

```console
uv run --no-sync task1-agent demo --case segment
uv run --no-sync task1-agent demo --case region
```

硅基流动作答首次研判若缺少有效时间窗观测或区内细分统计，会启用推理模式来判断是否补查。完成补查或进入复核重判时使用常规模式。模型调用的默认读取超时为 120 秒，连接超时为 10 秒，输出上限为 2,400 token，可通过模板中的对应环境变量覆盖。

## 读取赛题文件

数据文件可为 Stage 2 ZIP、解压后的 `task1` 目录或其中一个题目 JSONL。使用赛题许可的模型服务时，从 `LLM_MODEL`、`LLM_BASE_URL`、`LLM_API_KEY` 配置入口。`LLM_BASE_URL` 应包含网关要求的路径前缀，例如 `/v1`。

```console
uv run --no-sync task1-agent run --input "<数据 ZIP 或目录>" --output runs/task1_results.jsonl --trace runs/task1_trace.jsonl
```

可以重复传 `--sample-id` 只执行指定题。结果文件每行包含 `样本编号` 和 `结果`。失败样本写入 `task1_results.errors.jsonl`，批次存在失败时命令退出码为 1。`--trace` 保存计算证据、结果、模型用量及 `verification` 复核记录。格式问题记在 `model_calls[].format_issues`，规划和作答的格式重试次数分别记录在 `plan_format_retries_used` 和 `format_retries_used`。A 榜正式提交文件还需要加入 Task 2 的结果。

赛题资料、真实题结果、取证日志和实际 API 凭证留在本地。发布仓库收录源码、测试、自建演示样本和说明。运行产物可统一写到已忽略的 `runs/` 目录。

## 计算口径

- 时间窗接受相对秒或带时区的 ISO 8601 时间。输出中的时间同时给出两种表示。
- 对进出区域前后的运动，模型可以用 `anchor` 指定哪一次进入或离开。工具计算事件时刻，再查实际时间窗。
- 时序统计保留缺测数量，按有效观测点计算起止值、极值和变化率。较短的窗口也附带原始观测，便于核对窗口内的局部变化。
- 区域进出和停留时长按相邻坐标之间的直线估算，孤立的边界接触单列。起点已在区内只计停留。限值统计把端点视为满足条件，估计时长只覆盖两端都有有效观测的区间。
- 区域题的取证计划若缺少有效区域查询，程序会补上区域查询。含至少四个观测点的区域访问会提供四个区内时间窗的高度和速度统计，供模型判断变化位置。飞行阶段仍需结合证据判定。
- 答案复核按所选项的分句编号检查证据引用、数值比较和明确的区内范围。对整段高度或速度为零的表述直接核对原始极值。题设时刻和证据窗口完全匹配时，这类冲突会将选项移出重判候选。比较区内多个阶段时需要引用不同的区内子窗口。阶段语义和证据与题意的对应仍由模型判定。

## 源码结构

| 模块 | 职责 |
| --- | --- |
| `src/task1_agent/__main__.py`、`runner.py` | 解析命令、运行批处理并写入 JSONL 文件 |
| `dataset.py`、`examples.py` | 读取本地输入及构造自建演示样本 |
| `graph.py`、`prompts.py` | 定义图状态、节点、路由及模型指令 |
| `config.py`、`llm.py` | 读取模型配置，发出 HTTP 请求并校验结构化响应 |
| `models.py`、`evidence.py`、`verification.py` | 声明数据协议，计算航迹证据，复核模型的比较和引用 |

模型客户端和航迹计算器通过运行上下文注入。每道题的查询、证据和答案保存在独立图状态中。
