# 智能座舱 RAG 能力评估系统 —— 运行架构解析

> 版本：v2.4（真实评测引擎 + 知识库 + 安全加固）
> 入口：`rag_eval_app.py`（Web UI） / `run_eval_cli.py`（无界面批量）
> 说明：本文档描述**实际运行时**的模块划分、职责边界、数据流向、调用链路与依赖关系。

---

## 0. 一句话总览

系统是一条**离线建索引 + 在线逐题推理**的 RAG 评估流水线：把知识库语料解析分块，
用真实向量化与 BM25 分别建索引，对评测集里的每个问题做混合检索（可选重排、可选压缩），
交真实 LLM 或离线抽取器生成答案，最后计算一套**确定性指标 + LLM 判官指标**并聚合导出。

```
语料 ──▶ 分块 ──▶ 向量化 ──┬─▶ FAISS 向量索引 ─┐
                          └─▶ BM25 词法索引 ──┴─▶ 混合检索 ──▶ 重排 ──▶ 生成 ──▶ 指标 ──▶ 报告
     （离线，一次性）                              （在线，逐题，真实耗时）
```

---

## 1. 整体模块划分

系统按「展示 / 编排 / 能力 / 支撑 / 数据」五层组织。除展示层外，全部逻辑在 `rag_eval` 包内，
包内**不依赖 Streamlit**，因此同一套引擎既能被 Web UI 调用，也能被 CLI、定时任务、单元测试直接调用。

| 层 | 模块 | 文件 | 是否依赖 Streamlit |
|---|---|---|---|
| 展示层 | Streamlit 应用 | `rag_eval_app.py` | 是（唯一） |
| 展示层 | CLI 入口 | `run_eval_cli.py` | 否 |
| 编排层 | 评测编排器 | `rag_eval/evaluate.py` | 否 |
| 能力层 | 语料加载与分块 | `rag_eval/ingest.py` | 否 |
| 能力层 | 向量化 | `rag_eval/embedder.py` | 否 |
| 能力层 | 索引构建与缓存 | `rag_eval/index.py` | 否 |
| 能力层 | 检索与重排 | `rag_eval/retrieve.py` | 否 |
| 能力层 | 答案生成与上下文压缩 | `rag_eval/generate.py` | 否 |
| 支撑层 | LLM 客户端 | `rag_eval/llm.py` | 否 |
| 支撑层 | 指标计算与 LLM 判官 | `rag_eval/metrics.py` | 否 |
| 支撑层 | 文本工具 | `rag_eval/textutils.py` | 否 |
| 支撑层 | 配置与能力探测 | `rag_eval/config.py` | 否 |
| 数据层 | 内置语料 / 评测集 | `data/sample_corpus.json`、`data/sample_eval_set.json` | — |
| 数据层 | 向量缓存 / 报告 | `data/cache/`、`data/reports/` | — |
| 验证层 | 冒烟与回归测试 | `tests/test_smoke.py` | 是（AppTest） |
| 验证层 | 真实 LLM 连通性验证 | `tests/test_llm_live.py` | 否 |
| 验证层 | 环境与依赖自检 | `tools/diagnostics.py` | 否 |

---

## 2. 各组件职责

### 2.1 `config.py` —— 配置中枢与能力探测
- `RunConfig`：一次评测的全部参数（分块、向量、检索、重排、生成、判官），可序列化，是复现的凭证。
- `signature()`：产出「影响索引内容」的参数指纹（向量后端/维度/chunk 参数），是缓存有效性的依据。
- `LLM_PRESETS` / `EMBED_BACKENDS` / `RERANK_BACKENDS`：后端注册表，UI 与 CLI 共用，避免两套选项不一致。
- `capability_report()`：真实探测 pypdf / jieba / rank_bm25 / faiss / sklearn / openai 是否就绪。
- `load_dotenv()`：零依赖解析 `.env`，不覆盖已存在的真实环境变量。
- **职责边界**：只描述"能怎么配"，不做任何 IO 重活，不发起网络请求。

### 2.2 `ingest.py` —— 语料进入系统的唯一入口
- `load_pdf`：pypdf 逐页抽取文字，每页一个 `Document`（`doc_id` 形如 `手册#p3`，页码可溯源）。
- `load_json_corpus` / `load_txt`：加载 JSON 语料与纯文本。
- `RecursiveChunker`：**边界感知**递归切分 + 滑动重叠。优先在 `\n\n` → 句号 → 逗号 → 空格处断开，
  避免把一句话从中间劈开；再贪心合并到 `chunk_size`，并加上 `chunk_overlap` 的尾巴。
- `corpus_hash()`：语料指纹，参与缓存键。
- **职责边界**：只负责"把文件变成 `Document` 和 `Chunk`"，不做向量化、不做检索。

### 2.3 `embedder.py` —— 真实向量化
- `local-tfidf-svd`：在本机拟合 `TfidfVectorizer`（jieba 分词、1-2 gram）→ `TruncatedSVD` 降维 → L2 归一化。
  完全离线、结果确定（固定 `random_state`），产出可直接进 FAISS 的稠密向量。
  维度会**自动收缩**到 `min(目标维度, 样本数-1, 特征数-1)`，避免小语料上 SVD 报错。
- `api`：OpenAI 兼容 Embedding 接口，分批请求，真实网络调用。
- **契约**：输出一律 L2 归一化 → 内积即余弦，FAISS 用 `IndexFlatIP` 即等价于余弦检索。

### 2.4 `index.py` —— 索引与缓存
- `VectorIndex`：FAISS `IndexFlatIP` 优先；faiss 缺失时自动退化为 numpy 精确检索（结果一致，仅性能差异）。
- `BM25Index`：`rank_bm25.BM25Okapi`，jieba 分词建索引；空语料塞占位 token 防除零。
- 缓存：`load/save_cached_vectors` 以 **`corpus_hash + config.signature()`** 为键落盘 `.npy`，
  并校验形状。**参数一改，缓存自动失效**——从机制上杜绝"改了 chunk_size 却复用旧向量"的静默错误。
- `build_index()`：统一构建入口，串起「向量化 → 向量索引 → BM25」，产出 `RetrievalIndex`。
- **标题增强**：向量化与 BM25 使用“文档标题 + 分块正文”建索引；展示、引用与生成仍使用原始正文，避免切到段落中部后丢失主题。

### 2.5 `retrieve.py` —— 检索与重排
- `Retriever.retrieve()` 三条路径：
  - **纯向量**：FAISS Top-K；
  - **纯 BM25**：词法 Top-K；
  - **混合**：两路各取 `max(k*4, 20)` 候选 → 各自 min-max 归一化 → 加权求和（`bm25_weight` 控制）→ 取 Top-K。
    分数融合而非简单并集，保证可解释、可复现、可调权。
- `Reranker` 三种后端：
  - `none`：不重排；
  - `bm25`：对候选用 BM25 重新打分（离线、零成本）；
  - `llm`：把候选编号后交给真实 LLM 打 0-10 相关性分，JSON 解析失败即报错（不猜、不兜造）。
- `RetrieveStats`：真实记录向量/BM25/融合/重排各阶段耗时与候选规模。
- **防答非所问门控**：融合分只用于同批候选排序，不能代表跨问题置信度。重排后会再按当前检索词的中文二元组/英文词覆盖率过滤（默认 `min_query_coverage=0.12`）；无候选时进入“资料不足，无法回答”。
- **上下文多样性**：默认每篇文档最多保留 2 个块（`max_chunks_per_doc`），避免一篇聚合网页占满整个上下文窗口。

### 2.6 `generate.py` —— 答案生成与上下文压缩
- `AnswerGenerator`：
  - `llm`：RAG 提示词（仅依据参考资料作答 + `[1][2]` 引用标注 + 资料不足须明说），真实调用；
  - `extractive`：离线抽取式——在召回原文中按「句 × 问题」词法相关度排序，取真实原句拼成答案。
- `ContextCompressor`：有 LLM 时用 LLM 抽句，否则用词法阈值筛句；压缩后**复制 Chunk**，不污染原始索引。
- `_build_context()`：受 `max_context_chars` 约束拼装上下文，并回传实际用到的 `chunk_id`（可追溯）。
- 上下文块带有来源标题；离线抽取式答案要求最小句级相关度，避免只因共享一个字/词就拼入答案。
- 多轮改写只在存在明确指代（如“那后排呢”“它有什么要求”）时触发；短但自足的新问题不会继承上一轮主题。历史按轮数与字符预算双重裁剪。
- **压缩与生成分别计时**：`GeneratedAnswer.compress_s` 单独记录上下文压缩耗时，
  避免把压缩的 LLM 调用算进 `latency_s` 而虚高「平均生成耗时」（启用压缩时 CLI 会额外提示该口径）。
- 生成失败**不降级为编造**，而是返回带 `error` 的空答案，由上层如实记录。

### 2.7 `llm.py` —— LLM 客户端（健壮性中枢）
- 基于 `openai` SDK 的 OpenAI 兼容协议，DeepSeek / 中转站 / OpenAI 通用。
- `chat()` / `chat_json()`（容错解析 JSON，失败即抛错）/ `connectivity_check()`（真实连通性自检）。
- `LLMUsage`：累计 **token / 耗时 / 调用次数 / 重试次数 / 截断次数 / token 上限抬升次数**，
  并保留 `last_finish_reason`，供运行时统计落进报告。
- **重试与错误归因**：SDK 自带重试被关掉（`max_retries=0`），改由 `_classify_error()` 自行判定——
  - 可重试（指数退避，默认 3 次）：`408/409/425/429/500/502/503/504`、`Timeout`、`ConnectionError`；
  - 快速失败（不浪费时间）：`401/403` 鉴权、`402` 额度、`404` 模型名错、`400` 参数错；
  - 每类都翻译成**可读中文原因**，随 `LLMError` 抛出。
- **思维链模型 token 保护**：`deepseek-reasoner` 等模型的推理 token 会先吃掉 `max_tokens` 预算，
  上限过小时会返回 `finish_reason=length` + **空 content**（静默失败）。
  `_effective_max_tokens()` 对思维链模型强制抬到 `MIN_MAX_TOKENS_REASONING`（2048）并计数；
  非思维链模型沿用 `MODEL_MAX_TOKENS` 推荐值。
- **空内容与截断一律判为失败**：`finish_reason == "length"` 或 `content` 为空都抛 `LLMError`，
  **绝不把空答案当有效结果写进评测报告**。
- **原则**：任何失败都抛出携带原始错误信息的 `LLMError`，**绝不静默返回编造内容**。

### 2.8 `metrics.py` —— 指标引擎（真实性的核心）
- 确定性指标（纯本地、可复现）：
  - 检索类：`hit_rate` / `precision_at_k` / `recall_at_k` / `mrr` / `ndcg`——基于标注 `relevant_doc_ids`。
    **先做分块级→文档级去重**，否则同一文档的多个命中会让 recall/ndcg 超过 1（已修复并有回归测试）。
  - 上下文类：`context_precision`（RAGAS 风格 rank-weighted）、`context_recall`（标准答案句子覆盖率）。
  - 答案类：`exact_match` / `token_f1` / `rouge_l`（对比 `ground_truth`）。
  - 代理类：`faithfulness(lexical)` / `answer_relevancy(lexical)`——用词法覆盖率近似，**结果中明确标注 deterministic**。
- LLM 判官指标：`LLMJudge.faithfulness`（拆原子论断 → 逐条核查是否被上下文支撑）、
  `answer_relevancy`（由答案反推问题 → 与原问题比对）。判官失败自动回退词法代理并改标注。
- **每个指标都带 `method`（deterministic / llm-judge）与 `detail` 字段**，UI、导出、报告三处一致呈现，
  从机制上避免把启发式代理指标误当 LLM 指标。

### 2.9 `evaluate.py` —— 编排器
- `Evaluator.prepare()`：语料 → 分块 → LLM 客户端 → 向量化 → 索引；并处理**能力降级**。
- `Evaluator.run()`：逐题 `检索 → 生成 → 指标`，单题异常只记录该题 `error`，不中断整体。
- `_aggregate()`：逐指标求均值，**忽略未计算出的指标（`None`），不填假值**。
- `_runtime_stats()`：除题量/成功失败/平均检索与生成耗时外，**透出真实 LLM 用量**——
  模型与 base_url、是否思维链模型、调用/重试/截断/token 上限抬升次数、prompt/completion/total tokens、
  累计耗时，以及判官侧调用次数与耗时（判官回退词法时标注 `method=lexical-fallback`）。
- `EvalReport`：同时保留 `config_requested`（用户请求）与 `config`（实际生效），降级情况一目了然。
- `load_documents_from()` / `load_eval_set()`：统一加载入口，屏蔽来源差异。

### 2.10 `rag_eval_app.py` —— Streamlit 展示层
- 只做四件事：**收集配置 → 驱动引擎 → 渲染结果 → 导出**；不含任何检索/指标算法。
- 进度反馈：`prepare/run` 接受 `progress(frac, msg)` 回调，UI 用它驱动进度条与实时日志。
- 结果呈现：核心指标卡（带 DET/LLM 来源徽标）、指标口径表、逐题明细（含召回顺序与命中标注）、
  运行统计、完整配置与环境（可复现）、CSV/JSON 导出。

### 2.11 `run_eval_cli.py` —— 无界面入口
- 支持全部参数覆盖；`--sweep name=v1,v2` 做**参数扫描真实对比**（如 `chunk_size=128,512`）。
- 打印真实指标汇总 + 运行统计 + 配置降级对比，并把完整报告落 `data/reports/`。

---

## 3. 数据流向

### 阶段 A：启动与配置解析（每次页面加载）
```
Streamlit 启动
  └─ config.load_dotenv()          # 读 .env 注入环境变量
  └─ config.capability_report()    # 探测依赖是否就绪 → 决定可选能力
  └─ 侧边栏渲染：主题 / 语料来源 / 评测集来源 / 模型与检索参数
  └─ UI 侧输入的 API Key → os.environ（供 RunConfig 解析）
```

### 阶段 B：离线建索引（点击「开始真实评测」后，一次性）
```
Document[]                      ← load_pdf / load_json_corpus / load_txt
    │ RecursiveChunker.chunk_all()
    ▼
Chunk[]                         ← 带 chunk_id / doc_id / 偏移，可溯源
    │ corpus_hash()             → 语料指纹
    │ Embedder.fit()            → 本地拟合 TF-IDF + SVD（或 API 后端无需拟合）
    │ 查缓存(corpus_hash + signature)  ──命中──▶ 直接读 .npy
    │ Embedder.encode()         ──未命中─▶ 真实向量化 → save_cached_vectors()
    ▼
vectors: (n_chunks, dim)  L2 归一化
    ├─▶ VectorIndex.build()     → FAISS IndexFlatIP（或 numpy）
    └─▶ BM25Index.build()       → BM25Okapi（当启用混合检索或 bm25 重排）
    ▼
RetrievalIndex{ chunks, embedder, vectors, vector_index, bm25_index, ... }
```

### 阶段 C：在线逐题推理（对评测集每一条）
```
EvalItem{question, ground_truth, relevant_doc_ids}
    │
    ├─▶ Retriever.retrieve(question)
    │      ├─ 向量检索 → [(row, cos)]        ┐
    │      ├─ BM25 检索 → [(row, bm25)]      ├─ 混合：归一化 + 加权融合 → Top-K
    │      └─ Reranker.rerank()              ┘（none / bm25 / llm）
    │      ▼
    │   List[RetrievedChunk]{chunk, score, vector_score, bm25_score, rerank_score, rank}
    │
    ├─▶ [可选] ContextCompressor.compress()   → 句子级筛选（计时记入 compress_s，不混入生成耗时）
    ├─▶ AnswerGenerator.generate(question, contexts)
    │      ├─ _build_context() → 上下文文本 + used_chunk_ids（受 max_context_chars 限制）
    │      └─ llm: LLMClient.chat(RAG 提示词)  |  extractive: 原句抽取
    │             （思维链模型自动抬高 max_tokens；截断/空内容抛错不静默）
    │      ▼
    │   GeneratedAnswer{answer, mode, used_chunk_ids, cited_indices, latency_s, compress_s}
    │
    └─▶ 指标计算
           ├─ retrieval_metrics(去重后的 doc_id, 标注)     → hit_rate / P@K / R@K / MRR / NDCG
           ├─ context_precision / context_recall            → 基于 ground_truth 与上下文
           ├─ answer_metrics                                 → EM / token_f1 / rouge_l
           ├─ LLMJudge.faithfulness / answer_relevancy       → 需真实 LLM，失败回退词法
           ▼
        EvalRecord{answer, retrieved[], metrics{}, metric_meta{}, timings{}, error}
```

### 阶段 D：聚合与产出
```
EvalRecord[]  ──_aggregate()──▶  aggregates{指标: 均值}（None 不参与，不填假值）
              ──_runtime_stats()─▶ n_items/n_success/n_failed、平均检索与生成耗时
              ──▶ EvalReport{config, config_requested, capabilities, build_stats, runtime, records, warnings}
                    ├─▶ UI：指标卡 / 口径表 / 逐题展开 / 下载按钮
                    └─▶ CLI：终端汇总 + data/reports/report_*.json
```

---

## 4. 关键调用链路

### 链路① 应用启动（页面加载）
```
streamlit run rag_eval_app.py
  → st.set_page_config()
  → session_state 初始化
  → capability_report()                     # 依赖探测
  → 渲染 Header（依据 LLM API Key 显示「已配置 / 离线模式」徽标）
  → 渲染侧边栏与配置区（读 config 注册表生成选项）
  → 若已有 report → 渲染结果区
```

### 链路② 建索引（最容易出问题的一段）
```
Evaluator.prepare(docs)
  → RecursiveChunker.chunk_all(docs)               # 分块
  → [llm_backend != offline] LLMClient(cfg)        # 失败 → warning + 降级抽取式
  → [rerank == llm 且无 LLM] → 降级为 bm25 + warning
  → build_index(docs, chunks, cfg)
       → load_cached_vectors(corpus_hash, signature)   # 缓存校验
       → Embedder.fit() / encode()
       → save_cached_vectors()
       → VectorIndex.build()  /  BM25Index.build()
  → Retriever(cfg, index, llm) / AnswerGenerator(cfg, llm) / [judge_ready] LLMJudge(llm)
```

### 链路③ 单题评测（最核心）
```
Evaluator._run_one()
  → Retriever.retrieve(q)
       → VectorIndex.search(Embedder.encode_one(q), pool)
       → BM25Index.search(tokenize(q), pool)
       → Retriever._fuse()          # minmax_normalize + 加权
       → Reranker.rerank()          # none / bm25 重打分 / llm 判分
  → AnswerGenerator.generate(q, retrieved)
       → ContextCompressor.compress()   [可选]
       → _build_context()
       → LLMClient.chat()  |  _extractive_answer()
  → retrieval_metrics / context_precision / context_recall / answer_metrics
  → LLMJudge.faithfulness / answer_relevancy      [可选]
  → context_f1 = 2PR/(P+R)
  → 写 EvalRecord（含 timings 与 error）
```

### 链路④ LLM 判官（忠实度为例）
```
LLMJudge.faithfulness(answer, contexts)
  → LLMClient.chat_json(CLAIM_SYSTEM, 答案)        # 拆原子论断
  → LLMClient.chat_json(VERIFY_SYSTEM, 资料 + 论断编号)  # 逐条核查
  → supported / total                              # 真实比例
  → 异常 → return None → 上层回退 faithfulness_lexical 并改标注为 deterministic
```

---

## 5. 运行时依赖关系

### 5.1 依赖方向（严格单向，无循环）
```
config ◀── textutils ◀── 各能力模块
config ◀── llm ◀── generate / retrieve / metrics
ingest ◀── embedder / index
index  ◀── retrieve ◀── generate
以上全部 ◀── evaluate ◀── rag_eval_app / run_eval_cli
```
`config` 与 `textutils` 是叶子依赖（无内部依赖），是唯一被广泛复用的基础层。
`evaluate` 是唯一的"知道所有模块"的编排者，其余模块互不感知彼此的存在。

### 5.2 外部依赖与降级矩阵

| 依赖 | 用途 | 缺失时的真实行为 |
|---|---|---|
| `streamlit` / `pandas` | Web UI 与导出 | 必需（CLI 仍可跑） |
| `pypdf` | PDF 解析 | 无法上传 PDF，报错并提示安装；JSON/TXT 不受影响 |
| `jieba` | 中文分词 | 自动退化为字符 n-gram，检索与词法指标仍可算（质量下降） |
| `rank_bm25` | BM25 | 混合检索不可用，报错并提示；可改用纯向量 |
| `faiss` | 向量索引 | 自动退化为 numpy 精确检索（结果一致，仅速度差异） |
| `scikit-learn` | TF-IDF/SVD | 本地向量后端不可用，须改用 API embedding |
| `openai` | LLM 调用 | 无法使用 LLM 生成/判官/重排，降级为离线抽取式 + 词法指标 |
| `DEEPSEEK_API_KEY` 等 | 真实 LLM 推理 | 生成降级抽取式、重排降级 BM25、判官降级词法，**全部记入 warnings** |
| 思维链模型（`deepseek-reasoner`） | 推理型作答/判分 | 不降级：自动把 `max_tokens` 抬到 2048 下限并计数，避免「推理吃光预算 → 空答案」 |
| 网络暂时性抖动（429/5xx/超时） | 任意 LLM 调用 | 不降级：指数退避重试最多 3 次；仍失败才抛错，且错误信息标明第几次尝试 |

### 5.3 降级原则
1. **能降则降，但必须说出来**：每次降级都写入 `EvalReport.warnings`，并在 UI 高亮、报告中保留
   `config_requested` vs `config` 的差异对比。
2. **不能算就不算**：缺标注 → IR 指标为 `None`；缺 `ground_truth` → 答案类指标为 `None`；
   聚合时 `None` 不参与，**绝不填 0 或假值**。
3. **不许编造**：指标口径（deterministic / llm-judge）随每个指标值一同输出并可导出。

---

## 6. 核心数据结构与契约

| 结构 | 定义位置 | 关键字段 | 契约 |
|---|---|---|---|
| `Document` | ingest | `doc_id, title, text, source` | `doc_id` 唯一，可被评测集 `relevant_doc_ids` 引用 |
| `Chunk` | ingest | `chunk_id, doc_id, doc_title, text, page, start, end` | `chunk_id` 全局唯一，可回溯到原文位置 |
| `RetrievalIndex` | index | `chunks, embedder, vectors, vector_index, bm25_index, corpus_digest` | 向量已 L2 归一化，行号与 `chunks` 索引一一对应 |
| `RetrievedChunk` | retrieve | `chunk, score, vector_score, bm25_score, rerank_score, rank, source` | `rank` 即真实召回顺序 |
| `GeneratedAnswer` | generate | `answer, mode, used_chunk_ids, cited_indices, latency_s, compress_s, error` | `mode ∈ {llm, extractive}`；失败时 `error` 非空而 `answer` 为空；`latency_s` 不含压缩耗时 |
| `RunConfig` | config | `llm_max_tokens, llm_max_attempts, llm_retry_base_delay, ...` | `recommended_max_tokens()` / `is_reasoning()` 供 UI 给默认值与提示 |
| `EvalItem` | evaluate | `question, ground_truth, relevant_doc_ids` | 标注为**文档级** |
| `EvalRecord` | evaluate | `answer, retrieved[], metrics{}, metric_meta{}, timings{}, error` | `metrics` 与 `metric_meta` 键集合一致 |
| `MetricValue` / `metric_meta` | metrics | `family, method, detail` | `method ∈ {deterministic, llm-judge}`，必须透出到 UI |
| `EvalReport` | evaluate | `config, config_requested, aggregates, records, runtime, warnings` | 可完整 JSON 序列化 → 复现凭证 |

---

## 7. 缓存与一致性

- **缓存键** = `corpus_hash(全部文档 id + 正文)` + `signature(向量后端|维度|API 模型|chunk_size|overlap|min_chars)`。
- **命中校验**：除键匹配外，还校验向量矩阵 `shape[0] == len(chunks)`，防止脏缓存。
- **写盘内容**：`.npy`（向量矩阵）+ `.meta.json`（创建时间、语料指纹、参数签名）便于审计。
- 实测：同参数第二次运行显示 `命中向量缓存（22×21）`；改 `chunk_size` 后自动 `缓存命中：False` 并重建。

---

## 8. 真实性与可复现性保证

| 保证 | 实现机制 |
|---|---|
| 数字来自真实计算 | 无任何随机扰动或硬编码指标；`seed` 仅用于使 SVD 结果确定 |
| 可复现 | 报告导出完整 `config` + `build_stats` + `runtime` + 起止时间；SVD 固定随机种子 |
| 不可编造 | LLM 失败抛错而非兜底；某指标算不出即为 `None`，聚合跳过 |
| 口径透明 | 每个指标都带 `method` 与 `detail`，UI/CSV/JSON 三处一致 |
| 可追溯 | 每题保留召回顺序、各阶段分数、实际使用的 `chunk_id` 与耗时 |
| 有回归防护 | `tests/test_smoke.py` 7 项，含 IR 指标越界的回归测试与 UI 全链路 AppTest |

**实测基线 A —— 离线链路**（内置 11 篇语料 / 22 分块 / 12 题，`top_k=5`、混合检索开、重排关，不调用任何 LLM；
本组为 `run_eval_cli.py` 默认配置的可复现值）：

| 指标 | 值 | 口径 |
|---|---|---|
| hit_rate | 0.9167 | 确定性（12 题中 11 题命中；未命中的那题标注 doc 未被 Top-K 覆盖） |
| precision_at_k | 0.1833 | 确定性（文档级去重后） |
| recall_at_k | 0.9167 | 确定性 |
| mrr / ndcg | 0.7778 / 0.8135 | 确定性 |
| context_precision / context_recall | 0.7493 / 0.9167 | 确定性 |
| token_f1 / rouge_l | 0.3619 / 0.3819 | 确定性 |
| faithfulness | 1.0000 | 词法代理（抽取式答案源自原文，理应接近满分） |
| answer_relevancy | 0.1215 | 词法代理（较弱，配置 LLM 判官后替换为真实语义判定） |

运行统计：语料 `corpus_digest=f7e45d68bc738f70`；12/12 成功 0 失败；平均检索 1.1 ms、平均生成 1.4 ms；
向量后端 `local-tfidf-svd`（21 维）、索引 `faiss`、缓存命中 `True`。

> ⚠️ 基线 A 的数字随语料变化。若知识库非空且把「语料来源」切到知识库，内置评测集的标注 doc_id 将与语料不匹配，
> 检索类指标全为 0——届时看运行日志里的「标注与语料不匹配」诊断，不要误判为系统故障。

**实测基线 B —— 全 LLM 链路**（`deepseek-chat` + LLM 重排 + LLM 压缩 + LLM 判官，
报告：`data/reports/final_all_llm.json`，**该组用内置语料且 top_k / 重排配置与 A 不同，故两组不可逐行对比**）：

| 指标 | 值 | 口径 |
|---|---|---|
| hit_rate / recall_at_k / mrr / ndcg | 1.0000 | 确定性 |
| precision_at_k | 0.5000 | 确定性 |
| context_precision / context_recall | 0.9861 / 1.0000 | 确定性 |
| token_f1 / rouge_l | 0.7275 / 0.7447 | 确定性 |
| faithfulness | 0.9722 | **LLM 判官** |
| answer_relevancy | 0.3146 | **LLM 判官**（反推问题式判定，明显严于词法代理） |

运行统计：12/12 成功 0 失败；LLM 调用 96 次、重试 0、截断 0、token 上限抬升 0；
tokens `prompt=30224 / completion=6746 / total=36970`；LLM 累计 114.7s，判官 36 次 49.4s；
平均检索 962.8ms、平均生成 4483.3ms（**该值含上下文压缩耗时**，故显著偏高）。

> 对比 A 与 B 可见：接入真实 LLM 链路后 `token_f1` 从 0.36 拉到 0.73、`rouge_l` 从 0.38 拉到 0.74，
> 且 `answer_relevancy` 由「词法代理 0.1215」变成「LLM 判官 0.3146」——后者才是可对外引用的语义口径。
> 但注意 A 的检索类指标偏低并非纯配置差异（A 有 1 题未命中），对外汇报时应把两组配置差异一并说明。

---

## 9. 扩展点

| 想扩展什么 | 改哪里 | 怎么改 |
|---|---|---|
| 接入新的 LLM 厂商 | `config.LLM_PRESETS` | 加一条 base_url / models / env_key 即可，UI 自动出现该选项 |
| 接入真正的神经 embedding | `embedder.Embedder` | 新增 backend 分支（如 `sentence-transformers`），实现 `fit/encode` 即可 |
| 接入 cross-encoder 重排 | `retrieve.Reranker` | 新增 `cross-encoder` 分支，返回排序后的候选 |
| 新增指标 | `metrics.py` + `evaluate._run_one` | 实现函数并 `_put(name, value, family, method, detail)` 注册 |
| 换成 RAGAS | `metrics` 旁挂 adapter | 保持 `MetricValue(name, value, family, method, detail)` 契约即可无缝替换 |
| 参数自动寻优 | `run_eval_cli.py --sweep` | 已支持单参数扫描；多参数可在此基础上升级为网格/贝叶斯搜索 |

