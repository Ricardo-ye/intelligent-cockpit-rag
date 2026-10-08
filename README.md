# 智能座舱 RAG 能力评估系统

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB)
![License Apache--2.0](https://img.shields.io/badge/License-Apache--2.0-green)
![Status Beta](https://img.shields.io/badge/status-beta-orange)

一个可离线运行、可追溯、可扩展的知识库问答与 RAG 评测工作台。

[用户手册](USER_MANUAL.md) · [运行架构](ARCHITECTURE.md) · [知识库文档](docs/knowledge-base/rag-engineering-security-and-evaluation.md) · [贡献指南](CONTRIBUTING.md) · [安全策略](SECURITY.md)

把「模拟演示」改为**真实评测**：真实解析语料、真实向量化、真实检索、真实生成、真实指标。
所有数字都来自本次运行的真实计算，可复现、可追溯、可导出。

> 与旧版的区别：旧版点「开始评测」后指标由内置公式 + 随机扰动生成，不加载数据、不调用模型。
> 现版彻底移除了随机模拟，改为端到端真实流水线。

**v2.3 新增**：📚 知识库页（上传/粘贴/联网导入/内置样例四种通道）+ 真正的多轮追问（检索词自动改写）
+ 知识库命令行 `run_kb_cli.py` + 完整使用手册 [`USER_MANUAL.md`](USER_MANUAL.md)。

**v2.4 加固**：实际生成上下文追踪、分口径指标、统一配置校验、原子知识库写入、SSRF/上传路径防护，以及适合 GitHub 的 CI 与社区健康文件。详见 [`CHANGELOG.md`](CHANGELOG.md)。

---

## 界面构成（工作台版）

| 区域 | 位置 | 作用 |
|---|---|---|
| 🎛 **工作台** | 左侧抽屉，点左上角 `«` 收起 / 点 `‹` 重新调出 | 自定义界面配色与外观、问答显示项、对话设置、运行时能力自检、数据位置 |
| 💬 **知识问答** | 主区第 1 页 | 基于真实索引的**多轮**对话问答，追问会自动改写成可独立检索的问题；可展开查看召回来源与各阶段分数 |
| 📚 **知识库** | 主区第 2 页 | 导入与管理资料：上传文件 / 粘贴文本 / 联网导入（URL 或关键词搜索）/ 内置样例 |
| 📊 **能力测评** | 主区第 3 页 | 评测流水线：逐题对照标准答案算全套指标，支持导出 |
| ⚙️ **配置中心** | 主区第 4 页 | 语料 / 评测集 / 模型 / 检索参数，问答与测评共用同一份配置 |
| 📜 **运行日志** | 主区第 5 页 | 每步真实进度与降级提示 |

**外观可调**：7 套预设主题（深空暗色 / 极简亮色 / 晴空蓝 / 暖纸米 / 赛博紫 / 暖阳橙 / 森林绿），
并可自由改主色、辅色、页面背景、卡片背景、圆角、字号缩放、紧凑模式。
文字色 / 边框 / 弱化色**由背景亮度自动推导**，所以任意配色都不会出现"浅底浅字看不清"。

换肤分两条路径，缺一不可：

1. **自注入 CSS（主体）**：页面、卡片、按钮、输入框、selectbox、折叠面板等一律由主题 CSS 变量驱动。
   Streamlit 1.6x 把输入类控件从 baseweb 换成了 **react-aria**，旧的 `[data-baseweb="input"]` 选择器
   已完全失效，必须按新 `data-testid` / `.react-aria-*` 适配（见 `test_native_input_selectors_present`）。
2. **原生主题（补充）**：同步写入 `.streamlit/config.toml` 保证冷启动一致，并在运行时用
   `streamlit.config.set_option` 热更新，覆盖数据表等 Canvas 渲染、CSS 够不到的控件。

> **别只依赖第 2 条**：实测 Streamlit 的运行时换肤是"半套"的——文字色变了、输入框底色仍是旧主题的
> `#161a23`。所以凡是要跟随主题的元素，都用第 1 条的 CSS 变量硬覆盖。
> 另外原生主题是**进程级**配置，多个浏览器会话同时开时以最后一次设置为准（本地单人使用无影响）。

---

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置模型（可选；不配置则自动走离线真实链路，功能完整）
cp .env.example .env      # 然后填入 DEEPSEEK_API_KEY

# 3. 启动 Web UI
streamlit run rag_eval_app.py
#    浏览器打开 http://127.0.0.1:8501

# 4. （推荐）先到 📚 知识库 导入资料，再回到 💬 知识问答 提问
```

开发者可使用 `python -m pip install -e ".[dev]"`，然后运行 `python -m ruff check .` 与 `python -m pytest -q`。

> **功能与参数的完整说明见 [`USER_MANUAL.md`](USER_MANUAL.md)**：
> 功能入口、操作步骤、每个参数的含义与调法、知识库的导入方式/支持的文件类型/更新机制、15 条常见问题。

命令行方式（无界面，适合批量与 CI）：

```bash
# 评测
python run_eval_cli.py                      # 内置语料 + 内置评测集，离线真实计算
python run_eval_cli.py --llm deepseek --model deepseek-chat --judge   # 启用真实 LLM 生成与判官
python run_eval_cli.py --pdf 手册.pdf --evalset eval.json             # 用自己的数据
python run_eval_cli.py --sweep chunk_size=128,512                     # 参数扫描真实对比
# 以相同语料、题集和参数比较模型；价格必须来自可归档的版本化表
python run_eval_cli.py --llm deepseek --compare-models deepseek-chat,deepseek-reasoner --pricing data/model_pricing.json --experiment-name model-tradeoff

# 知识库（与工作台共用同一存储目录）
python run_kb_cli.py stats
python run_kb_cli.py add-file 手册.pdf ./docs/ --tags 座舱
python run_kb_cli.py add-text --title "后排空调设定" --file notes.txt
python run_kb_cli.py search "智能座舱 语音交互 设计" --n 8 --import-top 3 --fetch
python run_kb_cli.py list / preview / remove / export / builtin / clear --yes
```

自检与回归：

```bash
python tests/test_smoke.py        # 引擎 + 评测链路
python tests/test_workbench.py    # 工作台 UI（含知识库导入→问答→索引失效闭环）
python tests/test_kb.py           # 知识库（覆盖更新/内容去重/持久化/导出）
python tests/test_chat.py         # 多轮对话（追问判定/检索词改写/历史裁剪）
python tests/test_webimport.py    # 联网导入（跳转链判定/噪声过滤/相关性过滤/doc_id 稳定）
python tests/test_llm_live.py     # 真实 LLM 分层连通性（无 Key 时自动跳过）
```

> 改了 `rag_eval/` 下的模块后必须**重启 Streamlit 进程**才能生效——
> Streamlit 只会重跑主脚本，不会重新 import 已加载的本地模块。


---

## 两种运行模式

| 模式 | 触发条件 | 生成 | 忠实度/相关性 | 是否联网 |
|---|---|---|---|---|
| **离线真实** | 未配置 API Key（默认） | 抽取式（取召回原文真实原句） | 词法代理指标（标注 deterministic） | 否 |
| **LLM 真实** | 配置 DeepSeek / OpenAI 兼容 Key | LLM 依资料作答并标注引用 | LLM 判官（拆论断核查 / 反推问题） | 是 |

两种模式共用同一条检索与指标链路；切换只影响生成与判官，**指标口径始终在结果中标注**。

---

## 知识库（语料底座）

问答与测评都从知识库取语料。四种导入通道：

| 通道 | 入口 | 说明 |
|---|---|---|
| 上传文件 | 📚 知识库 → 📁 上传文件 | 支持 `.txt` `.md` `.markdown` `.json` `.csv` `.pdf` `.docx` `.html` `.htm`，可多选 |
| 粘贴文本 | 📚 知识库 → ✍️ 粘贴文本 | 网页正文、临时结论；单篇上限 30 万字，超出截断并提示 |
| 联网导入 | 📚 知识库 → 🌐 联网导入 | ① 按 URL 直抓；② 关键词搜索后勾选导入 |
| 内置样例 | 📚 知识库 → 📦 内置样例 | 11 篇智能座舱用车知识，冷启动即用 |

**存储**：`data/kb/manifest.json`（清单）+ `data/kb/docs/<sha1>.json`（正文）。
用环境变量 `RAG_KB_DIR` 可换到别的盘或目录。

**更新机制**（三条规则）：

1. **同来源覆盖** —— doc_id 相同（网页 doc_id = `web-<域名>-<URL 的 sha1>`）→ 覆盖正文、版本 +1；
2. **内容去重** —— doc_id 不同但正文哈希相同（同一篇转载）→ 跳过入库；
3. **内容未变** —— doc_id 与正文都相同 → 跳过，不产生新版本。

新增/更新/删除任何文档都会改变**内容指纹**，检索索引与向量缓存自动失效重建，**无需手动操作**。

> 网页 doc_id 用 URL 的确定性哈希，所以同一网址无论从界面、命令行还是隔几天再导入，都会落到同一条记录并触发覆盖更新。早期版本用 Python 内置 `hash()`（按进程随机加盐），导致同一网址每次导入都变成新文档、知识库不断堆积重复——已修复并有测试守着（`test_web_doc_id_stable_and_hashable_source`）。

**联网搜索的引擎链**：`360 → 百度 → 搜狗 → Bing → DuckDuckGo`。这个顺序是实测调优的结果——
360 对中文多词查询相关性最好且结果 `<a>` 带 `data-mdurl` 给出**真实地址**（可直抓）；
百度/搜狗给的是 `link?url=` 跳转链，在部分网络下会被拦（实测 502/400），故保留为候选但排在后面并标记 `⚠ 跳转链`；
Bing 在中文多词查询下会退化成单字匹配、甚至返回完全跑题的结果，只作兜底，并有相关性过滤拦截。

---

## 数据格式

**语料** `data/sample_corpus.json`
```json
{ "documents": [ { "doc_id": "doc-seatbelt", "title": "安全带正确使用方法", "text": "……" } ] }
```

**评测集** `data/sample_eval_set.json`
```json
{ "items": [
  { "question": "安全带应该怎么正确佩戴？",
    "ground_truth": "先调整座椅……",
    "relevant_doc_ids": ["doc-seatbelt"] }
] }
```
`relevant_doc_ids` 用于计算检索类 IR 指标；`ground_truth` 用于计算答案类与上下文类指标。
CSV 也支持，列名为 `question` / `ground_truth` / `relevant_doc_ids`（分号分隔）。

---

## 目录结构

```
rag_eval_app.py          Streamlit 工作台界面（唯一依赖 streamlit 的文件）
                         工作台抽屉 + 知识问答 + 知识库 + 能力测评 + 配置中心 + 运行日志
run_eval_cli.py          评测命令行入口（支持 --sweep 参数扫描）
run_kb_cli.py            知识库命令行入口（导入/联网抓取/检索/维护，与工作台共用存储）
USER_MANUAL.md           完整使用手册（功能入口/操作步骤/参数含义/FAQ）
CONTRIBUTING.md           贡献流程、测试与代码约定
SECURITY.md               漏洞报告方式与安全边界
CHANGELOG.md              面向用户的版本变更
pyproject.toml            包元数据、测试与静态检查配置
.github/                  CI、Issue 模板与 Pull Request 模板
docs/                     文档导航与可直接入库的知识库资料
.streamlit/config.toml   由工作台自动维护：让原生控件跟随界面主题
rag_eval/                真实评测引擎（不依赖 streamlit，可独立复用）
  config.py              配置、后端注册表、能力探测、.env 加载、思维链模型 token 策略
  ingest.py              PDF/JSON/TXT/MD/CSV/DOCX/HTML 加载 + 边界感知递归分块
  embedder.py            TF-IDF+SVD 本地向量 / OpenAI 兼容 Embedding
  index.py               FAISS 向量索引 + BM25 索引 + 指纹化缓存
  retrieve.py            向量/BM25/混合检索（分数融合）+ 重排（none/bm25/llm）
  generate.py            上下文压缩 + LLM/抽取式答案生成 + 多轮检索词改写
  llm.py                 OpenAI 兼容客户端（自管重试、错误归因、空内容不放过）
  metrics.py             确定性指标 + LLM 判官指标（每个指标标注口径）
  evaluate.py            编排：建索引 → 逐题评测 → 聚合报告（含真实 token 用量）
  costs.py               版本化价格表校验与 token 成本估算（不内置会过期报价）
  kb.py                  知识库：清单 + 正文分文件存储、同源覆盖/内容去重、内容指纹
  webimport.py           联网导入：网页正文抽取 + 多引擎搜索 + 噪声与相关性过滤
  security.py            URL 公网边界与上传文件名/路径校验
  textutils.py           分词/分句/F1/ROUGE-L/归一化等
data/
  sample_corpus.json     内置真实样例语料（11 篇智能座舱用车知识）
  sample_eval_set.json   内置标注评测集（12 题，含 ground_truth 与 relevant_doc_ids）
  kb/                    知识库存储（manifest.json + docs/<sha1>.json；可用 RAG_KB_DIR 覆盖）
  cache/                 向量缓存（键 = 语料指纹 + 向量化参数签名）
  reports/               评测报告与验证留痕 JSON
  uploads/               界面上传的语料/评测集落盘位置
tests/test_smoke.py      引擎冒烟与回归
tests/test_workbench.py  工作台 UI 验证（主题/配色/文字可读性/原生控件覆盖/知识库闭环/多轮问答）
tests/test_kb.py         知识库（覆盖更新/内容去重/多类型文件/持久化/导出/截断）
tests/test_chat.py       多轮对话（追问判定/改写兜底/历史裁剪/评测口径不受影响）
tests/test_webimport.py  联网导入（跳转链/噪声/相关性过滤/doc_id 稳定/失败归因）
tests/test_llm_live.py   真实 LLM 分层连通性验证
tools/diagnostics.py     运行时 / 依赖 / 数据完整性 / 缓存 / 网络自检
ARCHITECTURE.md          运行架构解析（模块划分/职责/数据流/调用链/依赖）
VERIFICATION_REPORT.md   核查与优化报告（检查项/问题/优化措施/验证结果）
```


---

## 指标口径

| 指标 | 计算方式 | 是否需 API Key |
|---|---|---|
| hit_rate / precision_at_k / recall_at_k / mrr / ndcg | 确定性，基于标注 `relevant_doc_ids`（先做分块→文档去重） | 否 |
| context_precision | 确定性，RAGAS 风格 rank-weighted | 否 |
| context_recall | 确定性，标准答案句子覆盖率 | 否 |
| exact_match / token_f1 / rouge_l | 确定性，与 `ground_truth` 直接比对 | 否 |
| faithfulness | LLM 拆论断后逐条核查；无 Key 时退化为词法覆盖（标注 deterministic） | 可选 |
| answer_relevancy | LLM 由答案反推问题后与原问题比对；无 Key 时退化为词法 F1 | 可选 |

指标结果中每一项都带 `method`（`deterministic` / `llm-judge` / `mixed`）与 `detail` 说明。
发生单题回退时，报告通过 `metric_breakdown` 给出各口径的均值与样本数，避免把代理指标静默当成 LLM 指标。

---

## 注意

- DeepSeek 官方**不提供 embedding 接口**，因此向量化默认使用本地 TF-IDF + SVD（真实可算、完全离线）。
  若需 API embedding，请把 `OPENAI_BASE_URL` 指向支持 embedding 的兼容服务。
- 扫描件/图片型 PDF 无法直接抽取文字，需先做 OCR。
- 小语料下 SVD 维度会自动收缩（如 22 个分块 → 21 维），这是为了避免矩阵分解报错，属预期行为。

---

## 安全与隐私

- 不要把 `.env`、API Key、私有语料、上传文件或本地报告提交到公共仓库。
- 联网导入会拒绝本机、私网和其他非公网地址；生产部署仍建议用出口防火墙隔离抓取服务。
- 导入资料属于不可信数据。高风险业务应额外实施提示注入检测、引用复核和访问控制。
- 漏洞请按 [`SECURITY.md`](SECURITY.md) 私下报告，不要在公开 Issue 中披露利用细节。

## 参与贡献

提交 Issue 或 Pull Request 前请阅读 [`CONTRIBUTING.md`](CONTRIBUTING.md)。项目采用 [Apache License 2.0](LICENSE)。
