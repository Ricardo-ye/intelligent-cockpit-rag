# RAG 工程化、安全与评测知识库

更新日期：2026-09-29  
适用范围：知识库问答、RAG 评测、网页资料导入和开源项目维护。

## 1. RAG 的核心价值与系统边界

检索增强生成把参数化语言模型和外部非参数知识库组合起来。原始 RAG 论文强调，外部索引不仅有助于知识密集型任务，也让知识更新和来源追踪成为可实现的工程目标。因此，一个可审计的 RAG 系统不能只保存最终答案，还应保留问题、检索查询、召回文档、实际生成上下文、引用和配置。

工程上应把系统划分为数据接入、分块、向量化、索引、检索、重排、生成和评测。展示层只负责采集参数和呈现结果，算法层应能被 Web、CLI 和自动化测试共同调用。

## 2. 数据接入与分块

文档进入知识库前应完成格式解析、正文清洗、来源记录、内容去重和大小限制。分块需要在召回率、上下文完整性和成本之间平衡：块太小会丢失语义，块太大会降低检索区分度并消耗生成上下文。

建议每个分块至少保留：`chunk_id`、`doc_id`、标题、来源、正文、页码或字符区间。重叠块的字符位置必须表示真实来源区间，不能只根据重复前缀猜测位置。

## 3. 小语料与向量降维

TF-IDF 加 TruncatedSVD 是一种可离线运行的轻量语义表示。scikit-learn 文档规定，随机化 TruncatedSVD 的 `n_components` 不能超过特征数；同时建议固定 `random_state` 获得可复现结果。

对只有一个样本或一个特征的极小知识库，不应强行使用二维以上 SVD。更稳妥的策略是直接使用归一化 TF-IDF 稠密向量；语料规模达到要求后再启用 SVD。缓存不仅要保存语料向量，还应保存拟合后的 vectorizer 和 SVD 实例，否则查询编码仍需重新拟合，缓存收益有限。

## 4. 混合检索与指标口径

混合检索通常结合稠密向量召回和 BM25 词法召回。分数融合需要说明归一化方式和权重含义；候选集合内 min-max 归一化简单直观，但不同查询之间的分数不可直接比较。需要更稳定的排序融合时，可以评估 Reciprocal Rank Fusion。

评测至少应区分三类指标：

- 检索指标：Hit Rate、Precision@K、Recall@K、MRR 和 NDCG。
- 上下文指标：召回内容是否覆盖标准答案，以及无关内容所占比例。
- 生成指标：答案相似度、忠实度、问题相关性、引用有效性和拒答能力。

标准 Precision@K 的分母应固定为 K。若只返回少于 K 条结果，应明确是按 K 计算，还是使用另一个名称描述“已返回结果中的精确率”。同一份报告中不能把 LLM 判官值和词法代理值静默混合；发生回退时，应分别给出方法、样本数和分口径平均值。

## 5. 实际生成上下文是评测事实来源

检索结果不一定等于模型最终看到的内容。重排、上下文压缩和字符预算都会改变实际生成上下文。因此，faithfulness 和引用校验必须基于实际送入生成器的文本，而不是未经压缩的全部召回块。

建议一次生成保存两条链路：

1. 原始召回链路，用于计算检索质量。
2. 实际生成上下文链路，用于计算忠实度、验证引用和排查截断。

如果上下文超过预算，系统应截断正文并记录实际字符数，不能让第一块内容绕过预算限制。

## 6. 知识库持久化与恢复

知识库清单是正文文件的索引，写入中断可能导致整个知识库不可读。Python 文档说明，同一文件系统内成功的 `os.replace` 可以提供原子替换语义。因此常用流程是：在目标目录写临时文件、flush、fsync，然后用 `os.replace` 替换目标文件。

推荐同时保留最近一次清单备份。发现清单 JSON 损坏时应停止写入并提示恢复，而不是静默创建一个空清单。多进程或多用户部署还需要跨进程文件锁，或者迁移到具备事务能力的 SQLite/PostgreSQL。

## 7. 网页导入与 SSRF 防护

OWASP 将服务端请求伪造定义为利用应用访问其自身、内部网络或其他受限服务。允许用户输入 URL 的知识库导入功能属于典型风险点。

最低防护要求包括：

- 只允许 HTTP 和 HTTPS，不接受 `file:`、`ftp:`、`gopher:` 等协议。
- 拒绝 URL 中的用户名和密码。
- 解析全部 IPv4/IPv6 地址，拒绝回环、私网、链路本地、保留地址和云元数据地址。
- 不自动信任重定向；若业务需要跟随，每一跳都重新验证协议、主机和解析地址。
- 使用流式下载，在读取过程中执行体积上限，而不是下载完成后才检查。
- 设置连接与读取超时，并限制重定向次数。

应用层校验仍不能完全消除 DNS rebinding 和代理配置风险。公网部署时还应使用出口防火墙或独立抓取服务，从网络层禁止访问内部网段。

## 8. 上传文件与界面输出

浏览器提供的文件名属于不可信输入。落盘前应去除目录部分、Windows 设备名和非法字符，并验证解析后的最终路径仍位于上传目录中。

网页正文、标题、文档 ID 和错误日志也属于不可信输入。把它们插入 `unsafe_allow_html` 前必须进行 HTML 转义；能使用普通文本或安全 Markdown 组件时，应避免不安全 HTML。

## 9. 提示注入与引用验证

检索到的文档可能包含“忽略之前指令”等恶意文字。系统提示应明确声明参考资料是不可信数据，只能作为事实来源，不能改变系统行为。对于高风险场景，还应把抓取、检索和生成放在不同权限边界中。

解析出 `[1]`、`[2]` 并不等于引用有效。引用校验至少要确认编号存在、对应内容实际进入生成上下文，并且答案论断能够由该片段支持。

## 10. 开源仓库维护清单

GitHub 的社区健康检查建议公共仓库提供 README、许可证、贡献指南和安全策略。README 应回答项目用途、价值、快速开始、获取帮助和维护方式；较长的设计资料放入 `docs/`，并使用相对链接，方便克隆后继续工作。

建议仓库至少包含：

- `README.md`、`LICENSE`、`CONTRIBUTING.md` 和 `SECURITY.md`。
- `pyproject.toml` 与可复现的依赖管理方式。
- `.github/workflows/ci.yml`、Issue 模板和 Pull Request 模板。
- 默认离线的自动化测试；联网和真实 LLM 测试通过显式环境变量启用。
- 不提交 `.env`、API Key、用户知识库、缓存、上传文件和本地报告。

## 11. 上线前检查表

- 配置值在统一入口完成类型和范围校验。
- 极小语料、空评测集、少于 K 条召回和全部失败都有确定行为。
- 知识库写入是原子的，清单有备份，损坏不会静默清空。
- URL 每一跳都经过公网地址校验，响应按流式大小上限读取。
- 实际生成上下文、引用和指标方法均写入报告。
- 代理指标和 LLM 判官指标不会混淆。
- CI 在无 API Key、无公网条件下可以运行。
- 发布前检查 Git 历史中没有密钥和个人数据。

## 参考资料

1. Lewis 等，《Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks》，NeurIPS 2020：https://arxiv.org/abs/2005.11401
2. OWASP，Server-Side Request Forgery Prevention Cheat Sheet：https://cheatsheetseries.owasp.org/cheatsheets/Server_Side_Request_Forgery_Prevention_Cheat_Sheet.html
3. Python 文档，`os.replace`：https://docs.python.org/3/library/os.html#os.replace
4. scikit-learn 文档，`TruncatedSVD`：https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
5. GitHub 文档，About community profiles for public repositories：https://docs.github.com/en/communities/setting-up-your-project-for-healthy-contributions/about-community-profiles-for-public-repositories
6. GitHub 文档，About the repository README file：https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes
7. GitHub 文档，Adding a security policy：https://docs.github.com/en/code-security/how-tos/report-and-fix-vulnerabilities/configure-vulnerability-reporting/add-security-policy
8. pytest 文档，How to invoke pytest：https://docs.pytest.org/en/stable/how-to/usage.html
