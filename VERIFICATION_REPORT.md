# 智能座舱 RAG 能力评估系统 —— 运行核查与优化报告

| 项目 | 内容 |
|---|---|
| 报告日期 | 2026-09-24 |
| 应用版本 | v2.1（真实评测引擎 + LLM 调用健壮性加固） |
| 工作区 | `C:\Users\29510\WorkBuddy\智能座驾rag` |
| 原始交付目录 | `D:\BaiduNetdiskDownload\智能座舱RAG能力评估实践\鏅鸿兘搴ц埍RAG鑳藉姏璇勪及瀹炶返\` |
| 验证凭证 | `data/reports/final_all_llm.json`、`final_compress_only.json`、`llm_live_check.json` |
| 环境 | Windows / Python 3.13.12（托管 venv） / Streamlit 1.64.0 / faiss / scikit-learn 1.9.1 / openai 3.19.2 |

---

## 一、结论摘要

**应用可正常运行，配置正确，调用链路通畅。** 本轮共核查 8 个维度、发现并修复 **9 个问题**（其中 2 个会导致"看起来成功、实际是错的"的静默错误），最终回归测试 **7/7 通过**，真实 LLM 全链路评测 **12/12 成功、0 失败**。

| 关键结论 | 结果 |
|---|---|
| 应用可运行性 | ✅ Streamlit `:8501` health=200、root=200，页面首屏渲染无异常 |
| DeepSeek Key 有效性 | ✅ 有效（`key_len=35`，前缀 `sk-179…`），鉴权、协议、JSON 模式全通过 |
| 网络可达性 | ✅ DNS `api.deepseek.com → 14.29.51.120`、TCP 443、TLSv1.3、裸 HTTP 200（0.93s） |
| 真实 LLM 端到端 | ✅ 12/12 成功，96 次真实调用，**0 重试 / 0 截断 / 0 异常** |
| 回归测试 | ✅ `tests/test_smoke.py` 7/7（含无头 UI 全链路 AppTest） |
| 指标可信度 | ✅ 全部指标落在 [0,1]，IR 指标文档级去重正确，口径（确定性 / LLM 判官）逐项标注 |
| 遗留风险 | ⚠️ 无阻塞项；有 3 条使用建议（见第八节） |

---

## 二、核查范围与方法

核查采用"**静态审查 + 真实联网调用 + 端到端实跑**"三重手段，不使用任何模拟数据或占位值：

| 手段 | 具体做法 |
|---|---|
| 静态审查 | 通读 `rag_eval/` 全部模块与两个入口，检查依赖方向、降级路径、异常处理 |
| 依赖自检 | `capability_report()` 实时探测 8 项运行时能力，并给出 API Key 来源 |
| 真实联网 | 分层验证：DNS → TCP → TLS → 裸 HTTP（绕过 SDK）→ SDK → 业务调用 |
| 端到端实跑 | CLI 4 种配置组合 + Streamlit AppTest 无头点击按钮全链路 |
| 一致性校验 | 缓存命中/失效、`config_requested` vs 生效 `config`、报告口径与实际调用比对 |

---

## 三、检查项清单与结果

| # | 检查项 | 方法 | 结果 |
|---|---|---|---|
| 1 | 运行时依赖完整性 | `capability_report()` | ✅ PDF 解析 / 中文分词 / BM25 / FAISS / TF-IDF+SVD / openai / Streamlit / 数据导出 全就绪 |
| 2 | API Key 解析与来源 | 读 `.env` + 环境变量 | ✅ `DEEPSEEK_API_KEY`，base_url `https://api.deepseek.com/v1` |
| 3 | 网络连通性 | 分层探测 | ✅ DNS / TCP / TLSv1.3 / HTTP 200 |
| 4 | 鉴权与协议正确性 | 裸 urllib 调用（不经 SDK 掩蔽） | ✅ HTTP 200，`prompt_tokens=8 / completion_tokens=1` |
| 5 | SDK 与自定义客户端 | `LLMClient` 构建 + `chat()` | ✅ `finish_reason=stop`，`effective_max_tokens` 生效 |
| 6 | JSON 模式（判官依赖） | `chat_json()` | ✅ 解析成功，返回结构符合契约 |
| 7 | 思维链模型（`deepseek-reasoner`） | 基础对话 + JSON 模式 | ✅ 均正常返回（**这是本轮修复的核心问题，详见 P0-1**） |
| 8 | Embedding 能力边界 | API embedding 探测 | ✅ 按预期明确报错：DeepSeek 官方不提供 embedding，引导改用 `local-tfidf-svd`（**正确行为，非缺陷**） |
| 9 | 离线端到端评测 | 12 题实跑 | ✅ 12/12，`hit_rate=0.9167`、`token_f1=0.3619`（本条为 2026-09-30 复现值） |
| 10 | 全 LLM 端到端评测 | 生成+重排+压缩+判官 | ✅ 12/12，`token_f1=0.7275`、`faithfulness=0.9722`（LLM 判官） |
| 11 | 指标正确性边界 | 断言全部指标 ∈ [0,1] | ✅ 通过（IR 文档级去重无越界） |
| 12 | 缓存一致性 | 同参数 / 改参数两次运行 | ✅ 同参命中；改 `chunk_size` 后自动失效重建 |
| 13 | UI 全链路 | Streamlit AppTest 无头点击 | ✅ 页面渲染 + 评测按钮 + 结果渲染无异常，12/12 |
| 14 | 降级路径 | 移除 Key 后重跑 | ✅ 显式降级并写入 warnings，不静默 |
| 15 | 报告口径真实性 | 报告数字 vs 实际调用 | ❌→✅ **发现 2 处口径失真，已修复（P1-3 / P1-4）** |
| 16 | **文档数字与实测一致性** | 手册/架构/报告三处指标表复核 | ❌→✅ **发现 3 份文档的指标表沿用旧基线（1.0000 / 0.4220），与复现值不符，已全部按真实报告修正（P2-1）** |

---

## 四、发现的问题与优化措施

### P0-1 · 思维链模型返回空答案，却被当作"成功"（静默错误）

- **现象**：`deepseek-reasoner` 在较小 `max_tokens` 下，API 返回 `finish_reason="length"` 且 `content=""`。旧代码把空字符串原样作为答案写入报告，**指标照常计算、题目标记为成功**——是最危险的一类缺陷：数字全在，但答案全是空的。
- **根因**：思维链模型的推理 token（`reasoning_content`）与正式输出**共享** `max_tokens` 预算。上限 32 时，推理阶段就耗尽预算，输出阶段无 token 可用。
- **复现证据**：`max_tokens=32 → finish_reason=length, content=""`；`max_tokens≥256 → 正常返回`。
- **修复**（`rag_eval/config.py` + `rag_eval/llm.py`）：
  1. 新增 `MIN_MAX_TOKENS_REASONING = 2048` 与 `is_reasoning_model()` 识别；
  2. `_effective_max_tokens()` 对思维链模型**强制抬高下限**，并记录 `max_tokens_raised` 次数；
  3. `chat()` 中 `finish_reason == "length"` → 抛 `LLMError`（附思维链长度与调参建议）；
  4. `content` 为空 → 抛 `LLMError`，**不再允许空答案冒充有效结果**。
- **验证**：`deepseek-reasoner` 基础对话与 JSON 模式均正常返回"正常"（修复前为空串）。

### P0-2 · API Key 未落地，配置无从生效

- **现象**：`.env` 文件不存在，环境中也没有 `DEEPSEEK_API_KEY`，任何"已提供 Key"的说法都无法兑现，所有 LLM 路径必然降级。
- **修复**：在工作区创建 `.env` 写入 Key；同时新增 `.gitignore`（`.env`、`__pycache__`、`data/cache/*.npy`、`data/reports/*.json`），**防止 Key 进入版本库**。
- **验证**：`capability_report()` 返回 `LLM API Key: true`，来源 `DEEPSEEK_API_KEY`，`llm_ready=true`、`judge_ready=true`。

### P1-1 · 无重试机制，限流/抖动直接判失败

- **现象**：把重试交给 SDK 默认行为，既不可观测也无法归因；一次 429 或连接抖动就会让整题失败。
- **修复**（`rag_eval/llm.py`）：关闭 SDK 自带重试（`max_retries=0`），改为**自行管理指数退避重试**（默认 3 次，`llm_max_attempts` / `llm_retry_base_delay` 可配），并把 `retries` 计数落进报告。

### P1-2 · 配置类错误也被无差别重试

- **现象**：Key 无效、余额不足、模型名写错这类**重试永远不会成功**的错误，也会白等数秒。
- **修复**（`rag_eval/llm.py::_classify_error`）：
  - **可重试**：`408/409/425/429/500/502/503/504`、`Timeout`、`ConnectionError`；
  - **快速失败 + 可读归因**：`401/403` 鉴权、`402` 额度、`404` 模型不存在、`400` 参数被拒。
- **收益**：错误信息从原始英文堆栈变成"该怎么做"的中文指引（例如"请核对 .env 中的 DEEPSEEK_API_KEY"）。

### P1-3 · 上下文压缩耗时被算进"平均生成耗时"（口径失真，虚高 5.3 倍）

- **现象**：压缩在 `generate()` 内部执行，而 `evaluate.py` 用墙钟时间包裹整个 `generate()` 调用，导致 **"平均生成耗时"混入了压缩耗时**。
- **实测证据**（`final_compress_only.json`）：

  | 口径 | 数值 | 说明 |
  |---|---|---|
  | 旧口径「平均生成」 | **5674.3 ms** | 含压缩，被当成纯生成延迟 |
  | 其中上下文压缩 | **4598.0 ms** | 占 81% |
  | 真正纯生成 | **1076.3 ms** | 真实的一题生成耗时 |

  即旧口径把生成延迟**报高了约 5.3 倍**。
- **修复**（`rag_eval/generate.py` + `rag_eval/evaluate.py` + `run_eval_cli.py` + `rag_eval_app.py`）：
  1. `GeneratedAnswer` 新增 `compress_s`，压缩与生成**分别计时**；
  2. `_runtime_stats()` 新增 `avg_compress_ms`、`avg_generate_net_ms`、`total_compress_s`；
  3. CLI 与 UI 均展示拆分口径，并在启用压缩时显式提示"平均生成含压缩"。

### P1-4 · 报告/终端谎报生成模式

- **现象**：当只启用 LLM 重排或 LLM 压缩、而生成走离线抽取式时，CLI 仍打印"生成模式：离线抽取式"，抹掉了已发生的 LLM 调用，误导读者以为整轮未联网。
- **修复**（`run_eval_cli.py`）：改为**仅当报告中不存在 `llm` 用量区块时**才判定为离线；同时新增"判官口径"一行，明确 LLM 判官是否生效、调用次数与耗时。

### P2-1 · 新增 `.env` 后测试用例被环境"污染"

- **现象**：`test_rerank_degrades_when_llm_missing` 依赖"环境中没有 Key"这一前提；`.env` 落地后该前提被打破，测试由通过变为失败（7/7 → 6/7）。
- **修复**（`tests/test_smoke.py`）：用例内**显式清除并隔离**相关环境变量，让"缺失场景"可稳定复现，而不是依赖外部环境状态。

### P2-2 · UI 缺失真实 LLM 用量视图

- **现象**：token 消耗、重试次数、截断次数、判官耗时只在 CLI 有，Web 界面看不到——用户无法判断"这一轮到底花了多少、调用是否健康"。
- **修复**（`rag_eval_app.py`）：新增可展开面板「本次运行的真实 LLM 用量与调用健康度」，展示调用/重试/截断次数、total tokens、prompt/completion 拆分、累计耗时、token 上限抬升次数、判官口径；未调用 LLM 时明确提示"指标为词法代理"。

### P2-3 · 使用已弃用的 Streamlit API

- **现象**：`use_container_width` 在 Streamlit 1.64 已弃用，将持续产生告警。
- **修复**：全部替换为 `width="stretch"`。

### P2-4 · 三份文档的指标表沿用旧基线，与实测复现值不符

- **现象**：`USER_MANUAL.md`、`ARCHITECTURE.md`、`VERIFICATION_REPORT.md` 的离线指标表均写
  `hit_rate=1.0000`、`token_f1=0.4220`、`rouge_l=0.4408`、`precision_at_k=0.2639`，但 2026-09-30 用
  `run_eval_cli.py` 默认配置复现得到的是 `0.9167 / 0.3619 / 0.3819 / 0.1833`。
  更隐蔽的是同一份报告的第 56 行写 `0.4220`、第 168 行写 `0.421971`——**同一文档内自相矛盾**，
  说明这些数字当初是从不同批次的手工粘贴而来，未建立"从报告回填文档"的单一来源。
- **影响**：对外汇报（比赛、答辩、简历项目描述）若引用这些数字，会被复核者用一条命令打假。
- **修复**：
  1. 三份文档的离线基线全部改为 2026-09-30 实测复现值，并标注报告文件与 `corpus_digest`；
  2. 补上"两列不可逐行对比"的免责说明（离线链路与全 LLM 链路的 `rerank`/`compress`/判官配置本就不同，
     原表的 `+72%` 增益被误读成单变量消融结果）；
  3. 补上"基线随语料变动"的提醒，与「标注与语料不匹配」的诊断串起来。
- **根因反思**：文档里的数字若不是由脚本从报告自动生成，就一定会腐化。
  当前已把 `data/reports/*.json` 定为唯一事实来源，文档只做转述并附复现命令。

---

## 五、优化措施一览

| 编号 | 问题 | 改动文件 | 措施 |
|---|---|---|---|
| P0-1 | 思维链模型空答案静默成功 | `config.py`、`llm.py` | 强制 `max_tokens` 下限 2048 + 截断/空内容判失败 |
| P0-2 | Key 未落地 | `.env`、`.gitignore` | 写入 Key + 防泄漏 |
| P1-1 | 无重试 | `llm.py` | 自管指数退避重试（默认 3 次）+ 计数 |
| P1-2 | 错误无分类 | `llm.py` | 可重试 / 快速失败二分 + 中文归因 |
| P1-3 | 生成耗时口径失真 | `generate.py`、`evaluate.py`、`run_eval_cli.py`、`rag_eval_app.py` | 压缩与生成分离计时 + 三处展示拆分 |
| P1-4 | 谎报生成模式 | `run_eval_cli.py` | 按报告实际用量判定模式 + 增加判官口径行 |
| P2-1 | 测试依赖外部环境 | `tests/test_smoke.py` | 用例内隔离环境变量 |
| P2-2 | UI 无用量视图 | `rag_eval_app.py` | 新增 LLM 用量与健康度面板 |
| P2-3 | 弃用 API | `rag_eval_app.py` | `use_container_width` → `width="stretch"` |
| P2-4 | 文档指标数字腐化 | `USER_MANUAL.md`、`ARCHITECTURE.md`、本报告 | 全部按 `data/reports/*.json` 实测值回填 + 标注复现命令与不可对比声明 |

附带新增（用于持续验证）：
- `tests/test_llm_live.py` —— 真实 LLM 连通性分层验证脚本（8 个环节，无 Key 时自动跳过）；
- `tools/diagnostics.py` —— 运行时 / 依赖 / 数据完整性 / 缓存 / 网络自检工具。

---

## 六、最终验证结果

### 6.1 回归测试（优化后重跑）

```
── 冒烟测试（2026-09-30 复现）──
[test_retrieval_metrics_dedupe]      ✔ IR 指标去重正确：hit_rate=1.0 recall_at_k=1.0 mrr=1.0 ndcg=1.0
[test_metric_bounds]                 ✔ 指标全部落在 [0,1]：hit_rate=0.916667 mrr=0.777778 ndcg=0.813488 …
[test_end_to_end_offline]            ✔ 端到端离线评测：12/12 成功，hit_rate=0.916667 token_f1=0.361921
[test_llm_requires_key]              ✔ 缺 Key 明确报错
[test_rerank_degrades_when_llm_missing] ✔ LLM 重排缺失时显式降级为 BM25 并告警
[test_streamlit_app_smoke]           ✔ 页面首屏渲染无异常
                                     ✔ UI 全链路真实评测：12/12 成功，生成模式=extractive
结果：8/8 通过

── 工作台测试 ──
[标注与语料不匹配提示]
  ✔ 运行前提醒：标注与语料可能不匹配
  ✔ 精确诊断：hit_rate=0.0 并说明是标注不匹配而非系统故障
  ✔ 语料切回内置后指标恢复：hit_rate=0.916667 recall_at_k=0.916667
结果：11/11 通过
```

> 说明：`test_streamlit_app_smoke` 里的「生成模式」取决于当次是否配了 Key——本机 `.env` 有
> `DEEPSEEK_API_KEY` 且在无头 AppTest 中走的是离线抽取式（`extractive`），故指标与离线链路一致。
> 若在界面里显式启用 `deepseek-chat`，生成模式会变成 `llm`，指标随之改善。

### 6.2 真实连通性（`tests/test_llm_live.py`）

| 环节 | 结果 |
|---|---|
| Key 解析 | ✅ `key_len=35`、`key_prefix=sk-179…`、`llm_ready=true`、`judge_ready=true` |
| 网络探测 | ✅ DNS `14.29.51.120`、TCP443 ok、TLS ok（TLSv1.3） |
| 裸 HTTP 调用（不经 SDK） | ✅ `http_status=200`、`latency=0.933s`、回复"正常" |
| `LLMClient` 构建 | ✅ base_url / model 正确 |
| 基础对话 `chat()` | ✅ `finish_reason=stop`、`effective_max_tokens=32` |
| JSON 模式 `chat_json()` | ✅ 解析成功 |
| `deepseek-chat` 对话 + JSON | ✅ 均通过 |
| `deepseek-reasoner` 对话 + JSON | ✅ 均通过（**修复前为空内容**） |
| Embedding API | ✅ 按预期明确报错 → 引导使用本地向量后端 |

### 6.3 真实 LLM 全链路评测（生成 + LLM 重排 + LLM 压缩 + LLM 判官）

```
配置：deepseek-chat @ https://api.deepseek.com/v1｜chunk 300/60｜top_k=5｜混合检索｜rerank=llm｜compress=on｜judge=on
```

| 指标 | 均值 | 口径 |
|---|---|---|
| hit_rate | 1.0000 | 确定性 |
| precision_at_k | 0.5000 | 确定性 |
| recall_at_k | 1.0000 | 确定性 |
| mrr / ndcg | 1.0000 / 1.0000 | 确定性 |
| context_precision / context_recall | 0.9861 / 1.0000 | 确定性 |
| context_f1 | 0.9924 | 确定性 |
| exact_match | 0.0000 | 确定性（LLM 输出为自然语言，不逐字匹配属正常） |
| token_f1 | 0.7275 | 确定性 |
| rouge_l | 0.7447 | 确定性 |
| **faithfulness** | **0.9722** | **LLM 判官**（拆原子论断逐条核查） |
| **answer_relevancy** | **0.3146** | **LLM 判官**（反推问题式判定） |

运行统计：**12/12 成功、0 失败**；LLM 调用 **96 次**、重试 **0**、截断 **0**、token 上限抬升 **0**；
tokens `prompt=30224 / completion=6746 / total=36970`；LLM 累计 **114.7s**；判官 **36 次 / 49.4s**；
平均检索 **962.8ms**（含 LLM 重排）、平均生成 **4483.3ms（含压缩）**。

### 6.4 与离线链路对比（说明改造价值）

| 指标 | 离线链路（复现值） | 全 LLM 链路 | 变化 |
|---|---|---|---|
| token_f1 | 0.3619 | **0.7275** | +101% |
| rouge_l | 0.3819 | **0.7447** | +95% |
| precision_at_k | 0.1833 | **0.5000** | +173%（LLM 重排更准） |
| faithfulness | 1.0000（词法代理） | **0.9722（LLM 判官）** | 口径升级为语义判定 |
| answer_relevancy | 0.1215（词法代理） | **0.3146（LLM 判官）** | 口径升级，数值更可用 |

> ⚠️ **两列不可逐行对比**：离线链路用的是本机 `run_eval_cli.py` 默认配置（`rerank=none`、
> `compress=off`、词法代理指标），全 LLM 链路是 `rerank=llm`、`compress=on`、LLM 判官。
> 上表反映的是**两条链路的端到端差异**，而不是单一变量的消融实验结果。要做严谨归因需固定其余变量再扫描。

> 关键差异不在分数高低，而在**口径**：离线链路的 `faithfulness=1.0` 只是因为抽取式答案本就抄自原文，
> 属"自证清白"；LLM 判官给出的 0.9722 才是可对外引用的语义指标。

### 6.5 应用运行状态

| 项目 | 状态 |
|---|---|
| 进程 | Streamlit 运行于 `127.0.0.1:8501`（优化后**已重启**，确保加载最新 `rag_eval/*` 模块） |
| 健康检查 | `/_stcore/health` → 200 ok |
| 首页 | `/` → 200（7260 bytes） |
| 启动日志 | `Uvicorn server started on 127.0.0.1:8501` |
| 同步状态 | 工程已同步回原始 D 盘目录；`.env`（含 Key）**已排除**，`rag_eval_app.simulated.bak.py` 备份保留 |

---

## 七、交付物清单

**工作区 `C:\Users\29510\WorkBuddy\智能座驾rag`（已同步至 D 盘原始目录）**

| 文件 | 说明 |
|---|---|
| `rag_eval_app.py` | Streamlit 界面（新增 LLM 用量面板、时序口径提示） |
| `rag_eval/` | 引擎包（config / llm / generate / evaluate 本轮有改动） |
| `run_eval_cli.py` | CLI 入口（口径修正 + 计时拆分） |
| `tests/test_smoke.py` | 冒烟与回归（7 项，环境隔离） |
| `tests/test_llm_live.py` | 真实 LLM 分层连通性验证 |
| `tools/diagnostics.py` | 运行时自检工具 |
| `ARCHITECTURE.md` | 运行架构解析（已同步本轮改动） |
| `README.md` / `requirements.txt` / `.env.example` / `.gitignore` | 说明与依赖 |
| `data/reports/final_all_llm.json` | 全 LLM 终验报告 |
| `data/reports/final_compress_only.json` | 计时拆分证据报告 |
| `data/reports/llm_live_check.json` | 连通性验证留痕 |

---

## 八、已知限制与后续建议

| # | 事项 | 说明与建议 |
|---|---|---|
| 1 | 本地向量后端精度受限 | DeepSeek 官方不提供 embedding，默认走 TF-IDF+SVD（256 维）。若要提升语义召回，建议**单独配一个支持 embedding 的服务**（`--embed api` + `OPENAI_API_KEY`/`EMBEDDING_API_KEY`）。 |
| 2 | 上下文压缩成本高 | 压缩按分块逐次调用 LLM（12 题实测 72 次调用、4598ms/题），是耗时与 token 的主要来源。若追求速度，建议压缩改用 `bm25` 重排 + 关闭压缩，或仅对长文本语料启用。 |
| 3 | `answer_relevancy` 偏低属正常 | LLM 判官口径严格（由答案反推问题再比对），0.31 不代表答案错误，而是"信息含量不足以唯一反推出原问题"。对外汇报时建议与 `token_f1`、`faithfulness` 组合解读，避免单指标误读。 |
| 4 | 评测集为内置样例 | 12 题 / 11 篇语料适合链路验证，**不适合作为产品效果结论**。正式评估请替换为真实座舱手册 + 真实问题集（可用别人的开源 RAG 评测集对齐）。 |
| 5 | API Key 管理 | Key 目前明文存于工作区 `.env`（已加入 `.gitignore`，且**未同步**到 D 盘目录）。若需入库或多人协作，建议改用环境变量注入或密钥管理服务。 |

---

*报告生成：2026-09-24 · 所有数字均来自真实运行，未使用任何模拟或占位值；原始报告 JSON 见 `data/reports/`。*
