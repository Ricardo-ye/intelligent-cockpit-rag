"""工作台 UI 的无头验证：主题系统 + 知识库导入 + 多轮问答 + 测评框架共存。

用 Streamlit 官方 AppTest 真实执行页面脚本、真实点击控件、真实跑问答链路，
不使用任何模拟数据。运行：python tests/test_workbench.py

注意：测试把知识库指向临时目录（环境变量 RAG_KB_DIR），
绝不会碰项目里的真实 `data/kb`。
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

APP_PATH = ROOT / "rag_eval_app.py"

# 必须在页面脚本被 AppTest 执行前设好：页面每次运行都会按环境变量新建知识库句柄
_KB_TMP = Path(tempfile.mkdtemp(prefix="wb_kb_"))
os.environ.setdefault("RAG_KB_DIR", str(_KB_TMP))


@contextmanager
def _fresh_kb():
    """给单个测试一个干净的知识库目录（同时隔离磁盘状态与页面内句柄）。"""
    tmp = Path(tempfile.mkdtemp(prefix="wb_kb_case_"))
    old = os.environ.get("RAG_KB_DIR")
    os.environ["RAG_KB_DIR"] = str(tmp)
    try:
        yield tmp
    finally:
        if old is None:
            os.environ.pop("RAG_KB_DIR", None)
        else:
            os.environ["RAG_KB_DIR"] = old
        shutil.rmtree(tmp, ignore_errors=True)


def _css(at) -> str:
    """把所有 markdown 注入内容拼起来，便于断言 CSS 主题是否真的生成了。"""
    return "\n".join((m.value or "") for m in at.markdown)


def _find_button(at, text: str):
    for b in at.button:
        if text in (b.label or ""):
            return b
    return None


def _click(at, text: str, timeout: int = 300):
    btn = _find_button(at, text)
    assert btn is not None, f"未找到按钮「{text}」，现有按钮：{[b.label for b in at.button]}"
    return btn.click().run(timeout=timeout)


def test_theme_preset_switches_css() -> None:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP_PATH), default_timeout=180)
    at.run()
    assert not at.exception, f"首屏渲染异常：{[e.value for e in at.exception]}"

    base_css = _css(at)
    assert "--accent:" in base_css, "应注入主题 CSS 变量"
    assert "#3b82f6" in base_css, "默认深空暗色主色应为 #3b82f6"

    at.selectbox(key="ui_preset").select("cyber").run()
    assert not at.exception, f"切换预设异常：{[e.value for e in at.exception]}"
    cyber_css = _css(at)
    assert "#a855f7" in cyber_css, "赛博紫预设应把主色写入 CSS"
    assert "#ec4899" in cyber_css, "赛博紫预设应把辅色写入 CSS"
    # 预设联动：背景/卡片色也要跟着换
    assert "#0f0a1a" in cyber_css, "赛博紫预设应同时更新页面背景"
    print("  ✔ 预设主题切换：CSS 主色/辅色/背景全部联动更新")

    # 亮色预设同样要整体联动，且文字必须自动翻成深色（否则白底白字不可读）
    at.selectbox(key="ui_preset").select("light").run()
    assert not at.exception, f"切换亮色预设异常：{[e.value for e in at.exception]}"
    light_css = _css(at)
    assert "#f5f7fb" in light_css, "极简亮色预设应更新页面背景"
    assert "#ffffff" in light_css, "极简亮色预设应更新卡片色"
    assert "#2563eb" in light_css, "极简亮色预设应更新主色"
    assert "--text:#0f172a" in light_css, "亮色背景必须自动改用深色文字"
    assert "--text:#e9eef7" not in light_css, "亮色背景下不应残留浅色文字"
    assert at.session_state["ui_mode"] == "light", "预设应把界面模式一并切到 light"
    print("  ✔ 亮色预设联动：背景/卡片/主色更新，文字自动转深色")


def test_custom_colors_and_shape() -> None:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP_PATH), default_timeout=180)
    at.run()

    at.color_picker(key="ui_accent").set_value("#ff0000").run()
    assert not at.exception, f"自定义主色异常：{[e.value for e in at.exception]}"
    css = _css(at)
    assert "#ff0000" in css, "自定义主色应写入 CSS"

    at.slider(key="ui_radius").set_value(20).run()
    at.slider(key="ui_fontscale").set_value(1.2).run()
    assert not at.exception, f"圆角/字号异常：{[e.value for e in at.exception]}"
    css = _css(at)
    assert "--radius:20px" in css, "圆角应写入 CSS 变量"
    assert "calc(16px * 1.2)" in css, "字号缩放应写入 CSS"
    print("  ✔ 自定义配色/圆角/字号：全部真实作用于 CSS")


def test_readable_text_follows_background() -> None:
    """浅色背景必须自动切深色文字，否则界面不可读。"""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP_PATH), default_timeout=180)
    at.run()

    at.color_picker(key="ui_bg").set_value("#faf5ec").run()
    assert not at.exception, f"浅色背景异常：{[e.value for e in at.exception]}"
    css = _css(at)
    assert "--text:#0f172a" in css, "浅色背景应自动使用深色文字"
    assert "--text:#e9eef7" not in css

    at.color_picker(key="ui_bg").set_value("#0d0f14").run()
    css = _css(at)
    assert "--text:#e9eef7" in css, "深色背景应自动使用浅色文字"
    print("  ✔ 文字色随背景亮度自动推导，任意配色均可读")


def test_chat_answers_with_sources() -> None:
    """真实问答：提交问题 → 检索 → 生成 → 写入历史（含召回来源）。"""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP_PATH), default_timeout=300)
    at.run()
    assert not at.exception, f"首屏渲染异常：{[e.value for e in at.exception]}"

    # 空状态应给出示例问题
    labels = [b.label or "" for b in at.button]
    assert any("安全带" in lb for lb in labels), f"空状态应提供示例问题按钮，实际按钮：{labels}"
    print("  ✔ 空状态提供示例问题入口")

    at.text_input(key="chat_input").set_value("安全带的正确佩戴方法是什么？")
    submit = None
    for b in at.button:
        if (b.label or "").strip() == "提问":
            submit = b
            break
    assert submit is not None, "未找到「提问」提交按钮"
    submit.click().run(timeout=300)
    assert not at.exception, f"问答执行异常：{[e.value for e in at.exception]}"

    chat = at.session_state.get("chat") or []
    assert len(chat) == 2, f"应产生一轮问答（user + assistant），实际 {len(chat)} 条"
    assert chat[0]["role"] == "user" and chat[1]["role"] == "assistant"

    # 关键回归：一轮问答只应渲染 2 个气泡。
    # 早期 render_turn 忽略 role，对每条消息都同时画用户气泡和助手气泡，
    # 导致 2 条消息渲染成 4 个气泡（看起来像问了两遍）——AppTest 之前没能拦住，这里补上。
    n_bubbles = len(at.chat_message)
    assert n_bubbles == 2, f"一轮问答应渲染 2 个气泡，实际 {n_bubbles} 个"

    answer = chat[1]["content"] or ""
    assert answer.strip(), f"答案不应为空：{chat[1]['meta']}"
    assert not chat[1]["meta"].get("error"), f"问答不应报错：{chat[1]['meta'].get('error')}"

    sources = chat[1]["meta"].get("sources") or []
    assert sources, "应带回真实召回来源"
    assert sources[0]["doc_title"], "召回来源应带文档标题"
    assert chat[1]["meta"].get("retrieve_ms") is not None, "应记录真实检索耗时"
    assert chat[1]["meta"].get("generate_ms") is not None, "应记录真实生成耗时"

    # 问答构建的引擎必须能被复用，不清空则不应重建
    engine = at.session_state.get("engine")
    assert engine and engine.get("sig"), "问答后应留下可复用的知识库引擎"
    print(f"  ✔ 真实问答闭环：答案 {len(answer)} 字，召回 {len(sources)} 段，"
          f"模式={chat[1]['meta']['mode']}，检索 {chat[1]['meta']['retrieve_ms']}ms / "
          f"生成 {chat[1]['meta']['generate_ms']}ms")

    # 清空会话（多轮上下文随之归零，按钮在改名为多轮版本后叫「新会话」）
    at = _click(at, "🆕 新会话", timeout=120)
    assert not at.exception, f"新会话异常：{[e.value for e in at.exception]}"
    assert (at.session_state.get("chat") or []) == [], "新会话后历史应为空"
    print("  ✔ 新会话可用（多轮上下文归零）")


def test_multi_turn_chat_keeps_context() -> None:
    """多轮对话：第二轮是追问，应带入历史并展示改写后的检索词。"""
    from streamlit.testing.v1 import AppTest
    from rag_eval import KnowledgeBase

    with _fresh_kb() as kbp:
        at = AppTest.from_file(str(APP_PATH), default_timeout=300)
        at.run()
        assert not at.exception, [e.value for e in at.exception]

        # 先把内置样例导入知识库，保证有内容可检索
        at = _click(at, "📦 导入内置样例")
        assert not at.exception, [e.value for e in at.exception]
        assert KnowledgeBase(root=kbp).stats()["n_docs"] == 11, "内置样例应导入 11 篇"

        at.radio(key="corpus_source_radio").set_value("已导入知识库").run(timeout=120)
        assert not at.exception, [e.value for e in at.exception]

        at.text_input(key="chat_input").set_value("安全带的正确佩戴方法是什么？")
        at = _click(at, "提问")
        chat = at.session_state.get("chat") or []
        assert len(chat) == 2, f"首轮应有 2 条消息，实际 {len(chat)}"
        assert chat[1]["meta"].get("history_turns") == 0, "首轮不应带入历史"
        assert chat[1]["meta"].get("n_docs") == 11, "索引应使用知识库里的 11 篇文档"

        # 第二轮：追问（短问题 + 指代）
        at.text_input(key="chat_input").set_value("那孕妇呢？")
        at = _click(at, "提问")
        assert not at.exception, [e.value for e in at.exception]
        chat = at.session_state.get("chat") or []
        assert len(chat) == 4, f"两轮应有 4 条消息，实际 {len(chat)}"
        assert chat[3]["meta"].get("history_turns") == 1, "追问应带入 1 轮上下文"
        rewritten = chat[3]["meta"].get("rewritten_query")
        assert rewritten, "追问应产生改写后的检索词（否则界面无法解释检索依据）"
        assert "安全带" in rewritten, f"改写应带上文语境，实际：{rewritten}"
        assert len(at.chat_message) == 4, f"两轮应渲染 4 个气泡，实际 {len(at.chat_message)}"
        print(f"  ✔ 多轮追问：历史 {chat[3]['meta']['history_turns']} 轮，"
              f"改写检索词「{rewritten[:26]}…」，气泡数正确")


def test_kb_import_flow_and_index_invalidation() -> None:
    """知识库导入 → 问答使用导入内容 → 新增文档使索引指纹失效并重建。"""
    from streamlit.testing.v1 import AppTest
    from rag_eval import KnowledgeBase

    with _fresh_kb() as kbp:
        at = AppTest.from_file(str(APP_PATH), default_timeout=300)
        at.run()

        # 粘贴导入一段只存在于本次测试的文本
        at.text_input(key="kb_text_title").set_value("压测文档 · 后排空调")
        at.text_area(key="kb_text_body").set_value(
            "后排空调出风口位于前排中央扶手箱后方，后排乘客可独立调节风量与温度。")
        at = _click(at, "⬆️ 导入这段文本")
        assert not at.exception, [e.value for e in at.exception]

        kb = KnowledgeBase(root=kbp)
        assert kb.stats()["n_docs"] == 1, f"应导入 1 篇，实际 {kb.stats()['n_docs']}"
        assert "后排空调" in kb.entries()[0]["title"]
        print("  ✔ 粘贴文本导入成功并落盘")

        # 切到知识库语料并提问
        at.radio(key="corpus_source_radio").set_value("已导入知识库").run(timeout=120)
        at.text_input(key="chat_input").set_value("后排空调出风口在哪里？")
        at = _click(at, "提问")
        assert not at.exception, [e.value for e in at.exception]
        chat = at.session_state.get("chat") or []
        meta = chat[-1]["meta"]
        assert meta.get("n_docs") == 1, f"问答应基于知识库的 1 篇文档，实际 {meta.get('n_docs')}"
        srcs = meta.get("sources") or []
        assert srcs and "后排空调" in srcs[0]["doc_title"], f"召回应来自导入的文档：{srcs[:1]}"
        fp1 = at.session_state["engine"].get("kb_fp")
        assert fp1, "启用知识库时，引擎签名应包含知识库指纹"
        print(f"  ✔ 问答基于导入内容：召回「{srcs[0]['doc_title']}」")

        # 再导入一篇 → 指纹变化 → 引擎签名变化 → 下次提问自动重建
        at.text_input(key="kb_text_title").set_value("压测文档 · 后备箱")
        at.text_area(key="kb_text_body").set_value(
            "后备箱支持脚踢感应开启，携带钥匙靠近车尾即可触发开启动作。")
        at = _click(at, "⬆️ 导入这段文本")
        assert KnowledgeBase(root=kbp).stats()["n_docs"] == 2

        at.text_input(key="chat_input").set_value("后备箱怎么开启？")
        at = _click(at, "提问")
        assert not at.exception, [e.value for e in at.exception]
        engine = at.session_state["engine"]
        assert engine["n_docs"] == 2, f"新增文档后索引应重建为 2 篇，实际 {engine['n_docs']}"
        assert engine["kb_fp"] != fp1, "知识库变化后指纹必须变化（索引重建的唯一依据）"
        print("  ✔ 更新机制生效：新增文档 → 指纹变化 → 索引自动重建（2 篇）")


def test_kb_ui_widgets_present() -> None:
    """知识库页的四种导入通道与文档管理入口都要在。"""
    from streamlit.testing.v1 import AppTest

    with _fresh_kb():
        at = AppTest.from_file(str(APP_PATH), default_timeout=240)
        at.run()
        assert not at.exception, [e.value for e in at.exception]

        labels = [b.label or "" for b in at.button]
        for need in ["⬆️ 导入选中文件", "⬆️ 导入这段文本", "⬇️ 抓取并导入", "🔍 搜索",
                     "📦 导入内置样例", "⬇️ 导出 JSON", "🧹 清空知识库"]:
            assert any(need in lb for lb in labels), f"知识库页缺少入口：{need}"

        assert [k for k in at.text_input if k.key == "kb_kw"], "应有联网搜索关键词输入框"
        assert any("pdf" in (m.value or "").lower() for m in at.markdown), \
            "支持的文件类型应写在界面上，用户不必猜"
        assert any("知识库还是空的" in (m.value or "") for m in at.markdown), \
            "空库时应有引导文案"
        print(f"  ✔ 知识库页入口齐全（{len(labels)} 个按钮），支持类型与空态引导都在")


def test_engine_shared_between_chat_and_eval() -> None:
    """测评跑完后，问答页应直接复用同一份索引，不再重建。"""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP_PATH), default_timeout=300)
    at.run()

    run_btn = None
    for b in at.button:
        if "开始真实评测" in (b.label or ""):
            run_btn = b
            break
    assert run_btn is not None
    run_btn.click().run(timeout=300)
    assert not at.exception, f"评测异常：{[e.value for e in at.exception]}"
    report = at.session_state.get("report")
    assert report, "应产出报告"

    engine = at.session_state.get("engine")
    assert engine and engine.get("ev") is not None, "测评后应把索引留在会话中供问答复用"
    assert engine["stats"].get("n_chunks") == report["build_stats"].get("n_chunks"), (
        "问答复用引擎的分块数应与测评报告一致"
    )
    print(f"  ✔ 测评与问答共用同一份索引：{engine['n_docs']} 篇 / "
          f"{engine['stats'].get('n_chunks')} 分块，无需二次构建")


def test_theme_config_toml_written() -> None:
    """原生主题同步：落盘 config.toml + 运行时热更新（否则切亮色后输入框仍是黑的）。"""
    from streamlit.testing.v1 import AppTest
    import streamlit.config as st_config

    at = AppTest.from_file(str(APP_PATH), default_timeout=180)
    at.run()

    cfg = ROOT / ".streamlit" / "config.toml"
    assert cfg.exists(), "应生成 .streamlit/config.toml 以驱动原生控件主题"
    text = cfg.read_text(encoding="utf-8")
    assert "[theme]" in text and "primaryColor" in text, "config.toml 应包含主题配置"

    at.color_picker(key="ui_accent").set_value("#00ff88").run()
    assert not at.exception
    text2 = cfg.read_text(encoding="utf-8")
    assert "#00ff88" in text2, "改主色后 config.toml 应同步更新（否则原生控件不跟随）"

    # 运行时热更新：切到亮色后，进程内的原生主题配置必须立刻变 light。
    # 回归背景：只写文件的话，Streamlit 只在启动时读 config，
    # 于是出现"页面变白了、输入框还是一片黑"的割裂观感（真实浏览器实测到过）。
    at.selectbox(key="ui_preset").select("light").run()
    assert not at.exception
    assert st_config.get_option("theme.base") == "light", "切亮色后原生控件应即时切到 light 主题"
    assert st_config.get_option("theme.backgroundColor") == "#f5f7fb", "原生背景色应即时同步"
    assert st_config.get_option("theme.textColor") == "#0f172a", "原生文字色应即时同步"

    at.selectbox(key="ui_preset").select("deep").run()
    assert st_config.get_option("theme.base") == "dark", "切回暗色应即时恢复"
    print("  ✔ 原生主题同步：config.toml 落盘 + 运行时热更新（无需重启）")


def test_native_input_selectors_present() -> None:
    """回归：Streamlit 1.6x 把输入类控件从 baseweb 换成 react-aria，
    只写 [data-baseweb="input"] 会失效，导致切主题后输入框仍残留旧主题底色
    （真实浏览器实测：亮色主题下 stTextInputRootElement 背景还是 #161a23 深色）。
    这里守住新 testid/class 覆盖，避免以后误删。"""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP_PATH), default_timeout=180)
    at.run()
    assert not at.exception, f"首屏渲染异常：{[e.value for e in at.exception]}"
    css = _css(at)

    must = [
        '[data-testid="stTextInputRootElement"]',
        '[data-testid="stSelectbox"] [role="combobox"]',
        ".react-aria-ComboBox",
        ".react-aria-TextField",
        '[data-testid="stFormSubmitButton"]>button',
    ]
    for sel in must:
        assert sel in css, f"原生输入控件主题覆盖缺失：{sel}"

    # 输入框底色必须跟随主题变量，而不是写死颜色
    assert "background:var(--surface2)!important" in css, "输入类控件底色应跟随主题变量"
    print("  ✔ 原生输入控件选择器覆盖到位（适配 react-aria 版控件）")


def test_corpus_eval_label_mismatch_warns() -> None:
    """回归：语料与评测集标注对不上时必须明确提示，不能给一张"全 0 但看着成功"的报告。

    背景：工作台默认语料是「已导入知识库」（库里非空时），而内置评测集的
    `relevant_doc_ids`（如 doc-seatbelt）只存在于内置样例语料里。
    用户导入知识库后直接点「开始真实评测」，hit_rate/recall_at_k/mrr/ndcg 会全为 0 ——
    真实原因是标注对不上语料，但看起来像"系统坏了"。
    """
    from streamlit.testing.v1 import AppTest

    with _fresh_kb():
        at = AppTest.from_file(str(APP_PATH), default_timeout=300)
        at.run()

        # 造一个非空知识库，并把语料切到它
        at.text_input(key="kb_text_title").set_value("压测文档 · 空调")
        at.text_area(key="kb_text_body").set_value(
            "空调可设置内循环与外循环，长时间内循环会导致车内空气变浑浊。")
        at = _click(at, "⬆️ 导入这段文本")
        at.radio(key="corpus_source_radio").set_value("已导入知识库").run(timeout=120)

        # 运行前的静态提醒
        warn_text = "\n".join((w.value or "") for w in at.warning)
        assert "标注可能对不上语料" in warn_text, (
            f"语料=知识库 + 评测集=内置评测集，应给出标注不匹配提醒；实际 warnings={warn_text[:200]}"
        )
        print("  ✔ 运行前提醒：标注与语料可能不匹配")

        # 仍然允许运行（用户可能只看答案类指标），但必须给出精确诊断
        at = _click(at, "▶  开始真实评测")
        assert not at.exception, f"评测异常：{[e.value for e in at.exception]}"
        report = at.session_state.get("report")
        assert report, "应产出报告"
        agg = report["aggregates"]
        assert agg.get("hit_rate") == 0, f"标注对不上时 hit_rate 应为 0，实际 {agg.get('hit_rate')}"
        all_warn = "\n".join((w.value or "") for w in at.warning)
        assert "标注与语料不匹配" in all_warn and "不是系统故障" in all_warn, (
            f"应给出精确诊断（含示例 doc_id），实际 warnings={all_warn[:300]}"
        )
        print(f"  ✔ 精确诊断：hit_rate={agg.get('hit_rate')} 并说明是标注不匹配而非系统故障")

        # 切回内置语料 → 标注匹配 → 检索指标恢复
        at.radio(key="corpus_source_radio").set_value("内置真实样例语料").run(timeout=120)
        at = _click(at, "▶  开始真实评测")
        assert not at.exception, f"评测异常：{[e.value for e in at.exception]}"
        agg2 = at.session_state["report"]["aggregates"]
        assert agg2.get("hit_rate", 0) > 0, f"内置语料下 hit_rate 应 > 0，实际 {agg2.get('hit_rate')}"
        print(f"  ✔ 语料切回内置后指标恢复：hit_rate={agg2.get('hit_rate')} "
              f"recall_at_k={agg2.get('recall_at_k')}")


TESTS = [
    ("主题预设联动", test_theme_preset_switches_css),
    ("自定义配色与形状", test_custom_colors_and_shape),
    ("文字色自适应", test_readable_text_follows_background),
    ("配置落盘同步", test_theme_config_toml_written),
    ("原生控件选择器覆盖", test_native_input_selectors_present),
    ("知识库页入口", test_kb_ui_widgets_present),
    ("知识库导入→问答→索引失效", test_kb_import_flow_and_index_invalidation),
    ("问答闭环（含来源）", test_chat_answers_with_sources),
    ("多轮对话上下文", test_multi_turn_chat_keeps_context),
    ("测评/问答共用索引", test_engine_shared_between_chat_and_eval),
    ("标注与语料不匹配提示", test_corpus_eval_label_mismatch_warns),
]


def main() -> int:
    print("── 工作台 UI 验证 ──")
    passed = failed = 0
    for name, fn in TESTS:
        print(f"\n[{name}]")
        try:
            fn()
            passed += 1
        except AssertionError as exc:
            failed += 1
            print(f"  ✘ 失败：{exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ✘ 异常：{type(exc).__name__}: {exc}")
    print(f"\n结果：{passed}/{passed + failed} 通过")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
