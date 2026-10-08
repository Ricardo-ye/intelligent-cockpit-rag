# 贡献指南

感谢参与智能座舱 RAG 工作台。提交变更前，请先搜索现有 Issue，较大的功能建议先开 Issue 说明使用场景和兼容性影响。

## 开发环境

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

不要提交 `.env`、API Key、真实用户数据、`data/cache/`、`data/uploads/` 或本地评测报告。

## 测试与检查

```bash
python -m ruff check .
python -m pytest -q
```

联网和真实 LLM 测试默认关闭。只有在明确需要时才设置 `RAG_LIVE_NET=1` 或配置 API Key，并确保输出已经脱敏。

## Pull Request 要求

1. 一个 PR 聚焦一个问题。
2. 修复缺陷时增加失败前可复现、修复后通过的测试。
3. 修改指标定义、报告结构或知识库格式时，说明迁移方式并更新文档。
4. 不把代理指标描述成模型判官指标；任何降级都必须在报告中可见。
5. 用户输入、网页正文和上传文件都视为不可信数据。

## 代码约定

- 引擎逻辑放在 `rag_eval/`，不要依赖 Streamlit。
- UI 只负责编排和展示，算法必须可由 CLI 与测试复用。
- 文件写入优先采用临时文件加原子替换。
- 新的外部请求必须设置超时、大小上限和明确错误归因。
