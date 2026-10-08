"""智能座舱 RAG 能力评估系统 —— 工作台版（Streamlit 前端）。

界面分五块（左侧「工作台」抽屉可随时收起/调出）：
1) 💬 知识问答：基于真实 RAG 索引的**多轮**对话问答，支持指代型追问（如「那后排呢」），
   回答只依据检索到的原文并标注 [1][2]，可展开核对召回来源与各阶段真实耗时。
2) 📚 知识库：导入 / 管理资料——上传文件、粘贴文本、联网导入（URL 或关键词搜索）、内置样例；
   支持去重、覆盖更新、删除、清空、导出。
3) 📊 能力测评：原有的真实评测流水线（真实语料 → 真实索引 → 真实检索 → 真实生成 → 真实指标）。
4) ⚙️ 配置中心：语料 / 评测集 / 模型 / 检索参数，问答与测评共用同一份配置。
5) 📜 运行日志：每步真实进度与降级提示。

与旧版的本质区别：数字全部来自真实计算，本文件只负责 UI，不含任何检索/指标算法。
运行：streamlit run rag_eval_app.py
"""

from __future__ import annotations

import json
import html
import os
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

# 保证脚本所在目录在 sys.path（便于 rag_eval 包与 data 目录被正确解析）
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from rag_eval import (  # noqa: E402
    EMBED_BACKENDS, EMBED_API_MODELS, LLM_PRESETS, RERANK_BACKENDS, SUPPORTED_EXTENSIONS,
    EvalError, Evaluator, IngestError, KnowledgeBase, LLMClient, LLMError, RunConfig,
    WebImportError, build_retrieval_query, capability_report, load_documents_from, load_eval_set, web_search,
)
from rag_eval.config import (  # noqa: E402
    MIN_MAX_TOKENS_REASONING, SAMPLE_CORPUS_PATH, SAMPLE_EVAL_PATH,
    is_reasoning_model, recommended_max_tokens,
)
from rag_eval.ingest import load_json_corpus  # noqa: E402
from rag_eval.security import resolve_upload_path  # noqa: E402

st.set_page_config(
    page_title="智能座舱 RAG 能力评估系统",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═══════════════════════════════════════════════════════════════
# 一、主题系统：预设 + 自由配色，全部由 CSS 变量驱动
# ═══════════════════════════════════════════════════════════════
THEME_PRESETS: dict[str, dict] = {
    "deep":    {"label": "🌌 深空暗色", "mode": "dark",  "bg": "#0d0f14", "surface": "#161a23", "accent": "#3b82f6", "accent2": "#06b6d4"},
    "light":   {"label": "☀️ 极简亮色", "mode": "light", "bg": "#f5f7fb", "surface": "#ffffff", "accent": "#2563eb", "accent2": "#0891b2"},
    "sky":     {"label": "🌤️ 晴空蓝",   "mode": "light", "bg": "#e8f3fd", "surface": "#ffffff", "accent": "#0369a1", "accent2": "#0ea5e9"},
    "paper":   {"label": "📜 暖纸米",   "mode": "light", "bg": "#faf5ec", "surface": "#ffffff", "accent": "#b45309", "accent2": "#0d9488"},
    "cyber":   {"label": "🔮 赛博紫",   "mode": "dark",  "bg": "#0f0a1a", "surface": "#1a1229", "accent": "#a855f7", "accent2": "#ec4899"},
    "amber":   {"label": "🔥 暖阳橙",   "mode": "dark",  "bg": "#14100c", "surface": "#1f1811", "accent": "#f59e0b", "accent2": "#ef4444"},
    "emerald": {"label": "🌿 森林绿",   "mode": "dark",  "bg": "#0b1410", "surface": "#122019", "accent": "#10b981", "accent2": "#22d3ee"},
}
PRESET_ORDER = ["deep", "light", "sky", "paper", "cyber", "amber", "emerald"]

STATE_DEFAULTS = {
    "ui_ready": False,
    "ui_preset": "deep",
    "ui_mode": "dark",
    "ui_bg": THEME_PRESETS["deep"]["bg"],
    "ui_surface": THEME_PRESETS["deep"]["surface"],
    "ui_accent": THEME_PRESETS["deep"]["accent"],
    "ui_accent2": THEME_PRESETS["deep"]["accent2"],
    "ui_radius": 10,
    "ui_fontscale": 1.0,
    "ui_show_scores": True,
    "ui_show_sources": True,
    "ui_show_timing": True,
    "ui_show_rewrite": True,      # 多轮追问时展示「改写后的检索词」，便于核对检索是否走偏
    "ui_history_turns": 3,        # 多轮对话带几轮上下文
    "ui_dense": False,
    "report": None,
    "logs": [],
    "running": False,
    "engine": None,
    "chat": [],
    "pending_q": None,
    "last_msg_meta": None,
    "_key_applied": None,
    # 知识库相关
    "kb_result": None,            # 最近一次导入结果（ImportResult.to_dict()）
    "kb_hits": None,              # 最近一次联网搜索的候选列表
    "kb_hits_query": "",
    "kb_confirm_clear": False,
}
for _k, _v in STATE_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── 颜色工具：让任何用户自选配色都能自动推出协调的辅色，保证可读性 ──
def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = (h or "#000000").lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    try:
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    except ValueError:
        return 0, 0, 0


def _rgb_to_hex(rgb) -> str:
    return "#%02x%02x%02x" % tuple(max(0, min(255, int(round(c)))) for c in rgb)


def _mix(c1: str, c2: str, ratio: float) -> str:
    """把 c1 向 c2 混合 ratio（0~1），用于自动推导边框 / 次级面 / 弱化文字。"""
    a, b = _hex_to_rgb(c1), _hex_to_rgb(c2)
    return _rgb_to_hex(tuple(x + (y - x) * ratio for x, y in zip(a, b)))


def _luminance(h: str) -> float:
    """感知亮度（0~1），用于自动决定文字用深色还是浅色。"""
    r, g, b = _hex_to_rgb(h)
    return (0.299 * r + 0.587 * g + 0.114 * b) / 255.0


def _build_theme() -> dict:
    """把工作台里的控件值组装成一份完整主题；文字/边框等由亮度自动推导，永不失可读性。"""
    p = THEME_PRESETS.get(st.session_state.ui_preset, THEME_PRESETS["deep"])
    bg = st.session_state.ui_bg or p["bg"]
    surface = st.session_state.ui_surface or p["surface"]
    accent = st.session_state.ui_accent or p["accent"]
    accent2 = st.session_state.ui_accent2 or p["accent2"]
    light_bg = _luminance(bg) > 0.55
    text = "#0f172a" if light_bg else "#e9eef7"
    return {
        "preset": st.session_state.ui_preset,
        "mode": st.session_state.ui_mode,
        "light_bg": light_bg,
        "bg": bg,
        "surface": surface,
        "surface2": _mix(surface, bg, 0.45),
        "border": _mix(surface, text, 0.16),
        "accent": accent,
        "accent2": accent2,
        "accent_soft": _mix(surface, accent, 0.22 if light_bg else 0.28),
        "text": text,
        "muted": _mix(text, bg, 0.50),
        "success": "#059669" if light_bg else "#34d399",
        "warn": "#b45309" if light_bg else "#fbbf24",
        "danger": "#dc2626" if light_bg else "#f87171",
        "radius": int(st.session_state.ui_radius),
        "fs": float(st.session_state.ui_fontscale),
        "dense": bool(st.session_state.ui_dense),
    }


def _apply_preset() -> None:
    """预设切换回调：在脚本重跑前把其它控件值一并改写，避免出现状态不一致。"""
    p = THEME_PRESETS[st.session_state.ui_preset]
    st.session_state.ui_mode = p["mode"]
    st.session_state.ui_bg = p["bg"]
    st.session_state.ui_surface = p["surface"]
    st.session_state.ui_accent = p["accent"]
    st.session_state.ui_accent2 = p["accent2"]


def _reset_theme() -> None:
    """恢复默认外观回调。"""
    st.session_state.ui_preset = "deep"
    p = THEME_PRESETS["deep"]
    st.session_state.ui_mode = p["mode"]
    st.session_state.ui_bg = p["bg"]
    st.session_state.ui_surface = p["surface"]
    st.session_state.ui_accent = p["accent"]
    st.session_state.ui_accent2 = p["accent2"]
    st.session_state.ui_radius = 10
    st.session_state.ui_fontscale = 1.0
    st.session_state.ui_dense = False


CSS_TEMPLATE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@300;400;500;700&family=JetBrains+Mono:wght@400;600&display=swap');
:root{
  --bg:__BG__; --surface:__SURFACE__; --surface2:__SURFACE2__; --border:__BORDER__;
  --accent:__ACCENT__; --accent2:__ACCENT2__; --accent-soft:__ACCENT_SOFT__;
  --text:__TEXT__; --muted:__MUTED__; --success:__SUCCESS__; --warn:__WARN__; --danger:__DANGER__;
  --radius:__RADIUS__px; --radius-sm:calc(__RADIUS__px * .55);
  --font:'Noto Sans SC',system-ui,sans-serif; --mono:'JetBrains Mono',ui-monospace,monospace;
}
html{font-size:calc(16px * __FS__);}
html,body,.stApp{font-family:var(--font);}
.stApp,[data-testid="stAppViewContainer"]{background:var(--bg);color:var(--text);}
[data-testid="stAppViewContainer"] .stMarkdown p,
[data-testid="stAppViewContainer"] .stMarkdown li,
[data-testid="stAppViewContainer"] label{color:var(--text);}
h1,h2,h3,h4,h5,h6{color:var(--text)!important;}

/* 顶栏：保留一点点高度但透明，**不能整体隐藏**，否则收起工作台后无法再展开 */
#MainMenu,footer{visibility:hidden;}
[data-testid="stHeader"]{background:transparent!important;}
[data-testid="stToolbar"]{right:8px;}
[data-testid="stSidebarCollapsedControl"]{color:var(--accent)!important;
  background:var(--surface)!important;border:1px solid var(--border)!important;
  border-radius:var(--radius-sm)!important;margin:6px;}

/* 工作台（侧边栏） */
[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border)!important;}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4,[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,[data-testid="stSidebar"] span{color:var(--text);}
[data-testid="stSidebar"] .stCaption,[data-testid="stSidebar"] small{color:var(--muted)!important;}
[data-testid="stSidebarCollapseButton"] button{color:var(--accent)!important;}

/* 自定义组件 */
.sys-header{background:linear-gradient(135deg,var(--surface) 0%,var(--surface2) 50%,var(--surface) 100%);
  border:1px solid var(--border);border-radius:var(--radius);
  padding:18px 26px;display:flex;align-items:center;gap:16px;margin-bottom:16px;position:relative;overflow:hidden;}
.sys-header::after{content:'';position:absolute;inset:0;
  background:radial-gradient(600px 60px at 8% 0%,var(--accent-soft),transparent 70%);pointer-events:none;}
.sys-header h1{font-size:1.45rem;font-weight:700;margin:0;letter-spacing:.04em;
  background:linear-gradient(90deg,var(--accent),var(--accent2));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.sys-header .sub{color:var(--muted);font-size:.76rem;}
.badge{display:inline-block;background:var(--accent-soft);border:1px solid var(--border);color:var(--accent);
  font-size:.63rem;font-family:var(--mono);padding:3px 10px;border-radius:999px;letter-spacing:.08em;}
.badge.ok{color:var(--success);border-color:var(--success);}
.badge.warn{color:var(--warn);border-color:var(--warn);}
.badge.plain{color:var(--muted);}

.section-title{font-size:.72rem;font-family:var(--mono);color:var(--accent);letter-spacing:.12em;
  text-transform:uppercase;margin-bottom:12px;display:flex;align-items:center;gap:8px;}
.section-title::before{content:'';display:inline-block;width:3px;height:14px;background:var(--accent);border-radius:2px;}

.metric-card{background:var(--surface2);border:1px solid var(--border);border-radius:var(--radius);
  padding:__CARD_PAD__;text-align:center;}
.metric-card .label{font-size:.6rem;font-family:var(--mono);color:var(--muted);text-transform:uppercase;
  letter-spacing:.1em;margin-bottom:6px;}
.metric-card .value{font-size:1.45rem;font-weight:700;font-family:var(--mono);color:var(--text);}
.metric-card .value.good{color:var(--success);}
.metric-card .value.mid{color:var(--warn);}
.metric-card .value.bad{color:var(--danger);}

.log-box{background:var(--bg);border:1px solid var(--border);border-radius:var(--radius);padding:14px;
  font-family:var(--mono);font-size:.74rem;color:var(--muted);max-height:260px;overflow-y:auto;line-height:1.8;}
.log-info{color:var(--accent);}.log-ok{color:var(--success);}
.log-warn{color:var(--warn);}.log-err{color:var(--danger);}

.retr-card{background:var(--surface2);border:1px solid var(--border);border-left:3px solid var(--accent2);
  border-radius:var(--radius-sm);padding:10px 14px;margin-bottom:8px;font-size:.79rem;color:var(--text);}
.retr-card.hit{border-left-color:var(--success);}
.retr-card .meta{color:var(--muted);font-family:var(--mono);font-size:.7rem;}

.score-pill{display:inline-block;padding:2px 10px;border-radius:999px;font-size:.74rem;font-weight:600;}
.score-good{background:var(--accent-soft);color:var(--success);border:1px solid var(--success);}
.score-mid{background:var(--accent-soft);color:var(--warn);border:1px solid var(--warn);}
.score-bad{background:var(--accent-soft);color:var(--danger);border:1px solid var(--danger);}
.method-badge{display:inline-block;padding:1px 7px;border-radius:var(--radius-sm);font-size:.6rem;
  font-family:var(--mono);letter-spacing:.05em;}
.method-det{background:var(--surface2);color:var(--muted);border:1px solid var(--border);}
.method-llm{background:var(--accent-soft);color:var(--accent);border:1px solid var(--accent);}

.bubble-user{background:var(--accent-soft);border:1px solid var(--accent);border-radius:var(--radius);
  padding:10px 14px;margin:6px 0 12px auto;max-width:82%;color:var(--text);font-size:.88rem;}
.bubble-src{background:var(--surface2);border:1px solid var(--border);border-left:3px solid var(--accent);
  border-radius:var(--radius-sm);padding:9px 13px;margin-bottom:7px;font-size:.76rem;color:var(--text);}
.bubble-src .h{font-weight:600;color:var(--accent);}
.empty-state{text-align:center;padding:34px 16px;color:var(--muted);border:1px dashed var(--border);
  border-radius:var(--radius);background:var(--surface2);}
.empty-state .big{font-size:2rem;margin-bottom:8px;}

/* 原生控件跟随主题 */
.stButton>button{background:linear-gradient(135deg,var(--accent),var(--accent2))!important;
  border:none!important;color:#fff!important;font-weight:600!important;font-size:.86rem!important;
  border-radius:var(--radius-sm)!important;padding:9px 22px!important;transition:filter .15s;}
.stButton>button:hover{filter:brightness(1.12);}
.stButton>button:disabled{opacity:.5!important;}
.stButton>button[kind="secondary"],.stButton>button[data-testid="stBaseButton-secondary"]{
  background:var(--surface2)!important;color:var(--text)!important;border:1px solid var(--border)!important;}
.stDownloadButton>button{background:var(--surface2)!important;color:var(--text)!important;
  border:1px solid var(--border)!important;border-radius:var(--radius-sm)!important;}

.stTabs [data-baseweb="tab-list"]{gap:4px;background:transparent;border-bottom:1px solid var(--border);}
.stTabs [data-baseweb="tab"]{background:transparent!important;color:var(--muted)!important;
  border-radius:var(--radius-sm) var(--radius-sm) 0 0;padding:9px 18px;font-weight:600;font-size:.86rem;}
.stTabs [aria-selected="true"]{color:var(--accent)!important;
  box-shadow:inset 0 -2px 0 0 var(--accent);background:var(--accent-soft)!important;}
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"]{background:transparent!important;}

div[data-testid="stVerticalBlockBorderWrapper"]{border:1px solid var(--border)!important;
  border-radius:var(--radius)!important;background:var(--surface)!important;}
div[data-testid="stVerticalBlockBorderWrapper"]>div{padding:__CARD_PAD__;}

/* 原生输入类控件：Streamlit 1.6x 起内部实现已从 baseweb 换成 react-aria，
   旧选择器 [data-baseweb="input"|"select"] 不再命中，于是切主题后这些控件会残留旧主题底色
   （实测亮色主题下输入框仍是 #161a23 深色）。故按新 testid/class 显式覆盖，做到与主题强一致。 */
[data-baseweb="input"]>div,[data-baseweb="select"]>div,[data-baseweb="textarea"],
[data-testid="stTextInputRootElement"],[data-testid="stTextAreaRootElement"],
[data-testid="stNumberInputContainer"],[data-testid="stSelectbox"] [role="combobox"],
.react-aria-TextField,.react-aria-ComboBox,
[data-testid="stTextInput"] input,[data-testid="stNumberInput"] input,[data-testid="stTextArea"] textarea{
  background:var(--surface2)!important;border:1px solid var(--border)!important;
  border-radius:var(--radius-sm)!important;color:var(--text)!important;}
input,textarea,[data-baseweb="select"] *{color:var(--text)!important;}
input::placeholder,textarea::placeholder{color:var(--muted)!important;opacity:1;}
[data-testid="stFormSubmitButton"]>button{background:linear-gradient(135deg,var(--accent),var(--accent2))!important;
  border:none!important;color:#fff!important;font-weight:600!important;font-size:.86rem!important;
  border-radius:var(--radius-sm)!important;padding:9px 22px!important;transition:filter .15s;}
[data-testid="stFormSubmitButton"]>button:hover{filter:brightness(1.12);}
[data-testid="stFileUploaderDropzone"]{background:var(--surface2)!important;border:1px dashed var(--border)!important;}
[data-baseweb="popover"] [role="listbox"],ul[role="listbox"],[role="listbox"]{background:var(--surface)!important;
  border:1px solid var(--border)!important;}
[role="option"]{color:var(--text)!important;}

[data-testid="stExpander"] details{background:var(--surface2);border:1px solid var(--border)!important;
  border-radius:var(--radius-sm)!important;}
/* 折叠面板的**头部**是原生 summary，不写背景的话切主题后会留一块深色底（实测 #12141c） */
[data-testid="stExpander"] summary{background:var(--surface2)!important;color:var(--text)!important;
  font-size:.84rem;border-radius:var(--radius-sm)!important;}
[data-testid="stExpander"] summary:hover{background:var(--accent-soft)!important;}
/* selectbox 里还有一个空的 role="group" 占位块，原生底色同样会残留 */
[data-testid="stSelectbox"] [role="group"]{background:var(--surface)!important;border-color:var(--border)!important;}
[data-testid="stDataFrame"],[data-testid="stTable"]{border:1px solid var(--border);border-radius:var(--radius-sm);}
[data-testid="stMetricValue"]{color:var(--text);}
[data-testid="stMetricLabel"]{color:var(--muted);}
[data-testid="stAlert"]{border-radius:var(--radius-sm);}
code,pre{background:var(--surface2)!important;color:var(--accent)!important;border-radius:4px;}
pre code{color:var(--text)!important;}
hr{border-color:var(--border)!important;}
::-webkit-scrollbar{width:9px;height:9px;}
::-webkit-scrollbar-track{background:transparent;}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:999px;}
.block-container{padding-top:1.2rem!important;max-width:1500px!important;}
</style>
"""


def build_css(t: dict) -> str:
    """把主题值直接内联进 CSS，避免变量覆盖与 !important 打架。"""
    out = CSS_TEMPLATE
    for token, value in (
        ("__BG__", t["bg"]), ("__SURFACE__", t["surface"]), ("__SURFACE2__", t["surface2"]),
        ("__BORDER__", t["border"]), ("__ACCENT__", t["accent"]), ("__ACCENT2__", t["accent2"]),
        ("__ACCENT_SOFT__", t["accent_soft"]), ("__TEXT__", t["text"]), ("__MUTED__", t["muted"]),
        ("__SUCCESS__", t["success"]), ("__WARN__", t["warn"]), ("__DANGER__", t["danger"]),
        ("__RADIUS__", str(t["radius"])), ("__FS__", str(t["fs"])),
        ("__CARD_PAD__", "10px 14px" if t["dense"] else "15px 18px"),
    ):
        out = out.replace(token, value)
    return out


def sync_native_theme(t: dict) -> dict:
    """把主题同步给 Streamlit 原生控件（输入框 / 表格 / 滑杆 / 弹层）——它们不吃自定义 CSS。

    走两条路径：
    ① **运行时热更新**：原生控件主题直接读 Streamlit 的 config，运行时 set_option 就能立刻换肤，
       否则用户切到亮色后，页面还是白的、输入框却仍是一片黑（实测过这个割裂观感）。
    ② **落盘 config.toml**：冷启动（新开进程 / 刷新）时保持一致，且内容没变就不写盘。

    注意：config 是**进程级**的，多个浏览器会话同时开时以最后一次设置为准（本项目是本地单用户，可接受）。
    """
    base = "light" if t["mode"] == "light" else "dark"
    pairs = {
        "theme.base": base,
        "theme.primaryColor": t["accent"],
        "theme.backgroundColor": t["bg"],
        "theme.secondaryBackgroundColor": t["surface"],
        "theme.textColor": t["text"],
    }

    # ① 运行时热更新
    live = False
    try:
        import streamlit.config as _st_config

        for key, val in pairs.items():
            if _st_config.get_option(key) != val:
                _st_config.set_option(key, val)
                live = True
    except Exception:  # noqa: BLE001 —— 内部 API 不可用时静默降级为「只落盘」
        live = False

    # ② 落盘，保证冷启动一致
    path = SCRIPT_DIR / ".streamlit" / "config.toml"
    content = (
        "# 由工作台自动维护：让 Streamlit 原生控件跟随界面主题\n"
        "[theme]\n"
        f'base = "{base}"\n'
        f'primaryColor = "{t["accent"]}"\n'
        f'backgroundColor = "{t["bg"]}"\n'
        f'secondaryBackgroundColor = "{t["surface"]}"\n'
        f'textColor = "{t["text"]}"\n'
        'font = "sans serif"\n'
    )
    try:
        current = path.read_text(encoding="utf-8") if path.exists() else ""
    except OSError:
        current = ""
    written = False
    if current != content:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            written = True
        except OSError:
            written = False
    return {"live": live, "file": written} 



THEME = _build_theme()
st.markdown(build_css(THEME), unsafe_allow_html=True)
_NATIVE_SYNCED = sync_native_theme(THEME)
CAPS = capability_report()

# 知识库句柄：每次脚本运行重新读取清单（保证外部改动/多次导入立刻可见），
# 单次运行内复用同一个实例，避免反复读盘。
_KB = KnowledgeBase()


def add_log(msg: str, level: str = "info") -> None:
    icon = {"info": "ℹ", "ok": "✓", "warn": "⚠", "err": "✕"}[level]
    ts = time.strftime("%H:%M:%S")
    safe = html.escape(str(msg), quote=True)
    st.session_state.logs.append(f'<span class="log-{level}">[{ts}] {icon} {safe}</span>')


def _badge(text: str, kind: str = "") -> str:
    return f'<span class="badge {kind}">{text}</span>'


# ═══════════════════════════════════════════════════════════════
# 二、页头
# ═══════════════════════════════════════════════════════════════
_engine = st.session_state.engine
_kb_badge = _badge("知识库已就绪", "ok") if _engine else _badge("知识库未构建", "warn")
_llm_badge = _badge("LLM 已配置", "ok") if CAPS.get("LLM API Key") else _badge("离线模式", "warn")

st.markdown(f"""
<div class="sys-header">
    <span style="font-size:1.7rem">🚗</span>
    <div>
        <h1>智能座舱 RAG 能力评估系统</h1>
        <div class="sub">Intelligent Cockpit RAG Workbench · 工作台 + 多轮问答 + 知识库 + 能力测评 · 真实评测引擎 v2.4</div>
    </div>
    <div style="margin-left:auto;display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
        {_badge("REAL PIPELINE", "plain")}{_kb_badge}{_llm_badge}
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# 三、工作台（左侧抽屉）
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🎛 工作台")
    st.caption("点左上角 « 可收起本抽屉；收起后点 ‹ 图标重新调出。")

    with st.expander("🎨 外观与配色", expanded=True):
        st.selectbox(
            "预设主题", PRESET_ORDER, key="ui_preset",
            format_func=lambda k: THEME_PRESETS[k]["label"],
            on_change=_apply_preset,
        )
        st.radio("明暗基准", ["dark", "light"], key="ui_mode", horizontal=True,
                 format_func=lambda m: "深色" if m == "dark" else "浅色",
                 help="决定 Streamlit 原生控件（表格、滑杆等）的底色基调")
        cc1, cc2 = st.columns(2)
        with cc1:
            st.color_picker("主色", key="ui_accent")
            st.color_picker("页面背景", key="ui_bg")
        with cc2:
            st.color_picker("辅色", key="ui_accent2")
            st.color_picker("卡片背景", key="ui_surface")
        st.slider("圆角", 0, 24, key="ui_radius")
        st.slider("字号缩放", 0.85, 1.25, key="ui_fontscale")
        st.toggle("紧凑模式（更小的卡片留白）", key="ui_dense")
        st.button("↺ 恢复默认外观", on_click=_reset_theme, width="stretch")
        st.caption("文字/边框/弱化色由背景亮度自动推导，任意配色都能保持可读。")

    with st.expander("💬 对话设置", expanded=True):
        st.slider("上下文轮数", 0, 8, key="ui_history_turns",
                  help="每次提问时，把最近几轮问答一起交给模型理解指代（如「那后排呢」）。"
                       "设 0 即为单轮问答。")
        st.toggle("显示改写后的检索词", key="ui_show_rewrite",
                  help="追问会被改写成可独立检索的问题，开启后可在回答下方看到实际用于检索的查询")
        _kb_now = _KB.stats()
        st.caption(f"当前知识库：{_kb_now['n_docs']} 篇 / {_kb_now['total_chars']} 字")

    with st.expander("💬 问答显示项", expanded=False):
        st.toggle("显示召回来源片段", key="ui_show_sources")
        st.toggle("显示相关度与各阶段分数", key="ui_show_scores")
        st.toggle("显示真实耗时", key="ui_show_timing")

    with st.expander("🧩 运行时能力自检", expanded=False):
        for k, v in CAPS.items():
            if isinstance(v, bool):
                st.markdown(f"{'✅' if v else '❌'} `{k}`")
            else:
                st.markdown(f"ℹ️ `{k}`：{v}")
        st.caption("未就绪的能力会自动降级并在日志中如实提示，不会静默造假。")

    with st.expander("📁 数据文件位置", expanded=False):
        st.code(
            f"语料：{SAMPLE_CORPUS_PATH.name}\n评测集：{SAMPLE_EVAL_PATH.name}\n工作目录：{SCRIPT_DIR}",
            language="text",
        )

    if _NATIVE_SYNCED.get("live"):
        st.caption("✓ 原生控件（输入框 / 按钮 / 滑杆）已随主题换肤；数据表若未跟随，刷新一次即可。")
    elif _NATIVE_SYNCED.get("file"):
        st.caption("ℹ️ 原生主题已写入配置，若配色未跟上，刷新页面（F5）即可。")


# ═══════════════════════════════════════════════════════════════
# 四、公共工具函数（供问答与测评共用）
# ═══════════════════════════════════════════════════════════════
def materialize_upload(uploaded) -> str | None:
    """把上传文件落到磁盘（真实流水线需要文件路径）。"""
    if uploaded is None:
        return None
    tmp_dir = SCRIPT_DIR / "data" / "uploads"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    p = resolve_upload_path(tmp_dir, uploaded.name)
    p.write_bytes(uploaded.getbuffer())
    return str(p)


def load_docs(S: dict):
    """按配置面板的选择加载知识库语料。"""
    src = S["corpus_source"]
    f = S["doc_file"]
    if src == "内置真实样例语料":
        return load_documents_from(builtin=True)
    if src == KB_SOURCE:
        docs = _KB.documents()
        if not docs:
            raise IngestError(
                "知识库为空。请先到「📚 知识库」页导入资料："
                "上传文件 / 粘贴文本 / 联网导入（网页 URL 或关键词搜索）。"
            )
        return docs
    if src == "上传 PDF":
        if f is None:
            raise IngestError("请先上传 PDF 文件。")
        return load_documents_from(pdf_path=materialize_upload(f))
    if src == "上传 JSON 语料":
        if f is None:
            raise IngestError("请先上传 JSON 语料文件。")
        return load_documents_from(corpus_json_path=materialize_upload(f))
    if f is None:
        raise IngestError("请先上传 TXT 文件。")
    return load_documents_from(txt_path=materialize_upload(f))


def load_items(S: dict):
    """按配置面板的选择加载评测集。"""
    if S["eval_source"] == "内置真实评测集":
        return load_eval_set(SAMPLE_EVAL_PATH)
    f = S["eval_file"]
    if f is None:
        raise EvalError("请先上传评测集文件。")
    return load_eval_set(materialize_upload(f))


def make_cfg(S: dict) -> RunConfig:
    return RunConfig(
        chunk_size=S["chunk_size"],
        chunk_overlap=S["chunk_overlap"],
        embed_backend=S["embed_backend"],
        embed_dim=int(S["embed_dim"]),
        embed_api_model=S["embed_api_model"],
        top_k=int(S["top_k"]),
        use_hybrid=bool(S["use_hybrid"]),
        bm25_weight=float(S["bm25_weight"]),
        rerank_backend=S["rerank_backend"],
        rerank_top_k=int(S["rerank_top_k"]),
        use_compression=bool(S["use_compression"]),
        llm_backend=S["llm_backend"],
        llm_model=S["llm_model"],
        llm_temperature=float(S["llm_temperature"]),
        llm_max_tokens=int(S["llm_max_tokens"]),
        use_llm_judge=bool(S["use_llm_judge"]),
        use_cache=bool(S["use_cache"]),
    )


KB_SOURCE = "已导入知识库"


def cfg_signature(cfg: RunConfig, kb_fp: str = "") -> str:
    """问答与测评共用引擎时的一致性判据：配置一动，索引自动重建。

    知识库指纹也纳入签名——否则导入/删除文档后，旧索引会被继续复用，
    问答检索的还是导入前的内容（这是"更新机制"能否生效的关键）。
    """
    return json.dumps(
        {**cfg.to_dict(), "_kb": kb_fp}, sort_keys=True, ensure_ascii=False
    )


def build_engine(S: dict, progress=None):
    """构建（或复用）知识库引擎。返回 (engine, cfg, rebuilt)。"""
    cfg = make_cfg(S)
    kb_fp = _KB.fingerprint() if S.get("corpus_source") == KB_SOURCE else ""
    sig = cfg_signature(cfg, kb_fp)
    eng = st.session_state.engine
    if eng and eng.get("sig") == sig:
        return eng, cfg, False
    docs = load_docs(S)
    ev = Evaluator(cfg)
    stats = ev.prepare(docs, progress=progress)
    eng = {"sig": sig, "ev": ev, "stats": stats, "n_docs": len(docs), "kb_fp": kb_fp}
    st.session_state.engine = eng
    return eng, cfg, True


# ═══════════════════════════════════════════════════════════════
# 五、四个页面（用占位容器，让「配置」先跑但视觉上排在第三页）
# ═══════════════════════════════════════════════════════════════
tab_chat, tab_kb, tab_eval, tab_cfg, tab_log = st.tabs(
    ["💬 知识问答", "📚 知识库", "📊 能力测评", "⚙️ 配置中心", "📜 运行日志"]
)
cfg_ph = tab_cfg.container()
chat_ph = tab_chat.container()
kb_ph = tab_kb.container()
eval_ph = tab_eval.container()
log_ph = tab_log.container()


# ── 配置中心 ──────────────────────────────────────────────────
def render_config_panel() -> dict:
    st.markdown('<div class="section-title">数据来源 / Data Source</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="large")

    with c1:
        with st.container(border=True):
            st.markdown("**知识库语料**")
            _kb_stats = _KB.stats()
            corpus_options = [KB_SOURCE, "内置真实样例语料", "上传 PDF", "上传 JSON 语料", "上传 TXT"]
            corpus_source = st.radio(
                "语料来源", corpus_options,
                index=0 if _kb_stats["n_docs"] else 1,
                horizontal=True, label_visibility="collapsed", key="corpus_source_radio",
            )
            if corpus_source == KB_SOURCE:
                if _kb_stats["n_docs"]:
                    st.success(
                        f"✓ 已导入知识库就绪：{_kb_stats['n_docs']} 篇 / {_kb_stats['total_chars']} 字"
                        f"（更新于 {_kb_stats['updated_at'] or '—'}）"
                    )
                    st.caption("增删文档会自动让索引失效并重建，无需手动操作。")
                else:
                    st.warning("知识库为空，请到「📚 知识库」页导入资料后回到这里。", icon="⚠️")
            doc_file = None
            if corpus_source == "上传 PDF":
                doc_file = st.file_uploader("上传知识库文档（PDF）", type=["pdf"], key="pdf_up")
            elif corpus_source == "上传 JSON 语料":
                doc_file = st.file_uploader(
                    "上传 JSON 语料", type=["json"], key="json_up",
                    help='格式：[{"doc_id":"d1","title":"标题","text":"正文"}]',
                )
            elif corpus_source == "上传 TXT":
                doc_file = st.file_uploader("上传 TXT 文本", type=["txt"], key="txt_up")

            if corpus_source == "内置真实样例语料":
                try:
                    _docs = load_json_corpus(SAMPLE_CORPUS_PATH)
                    st.success(f"✓ 内置语料就绪：{len(_docs)} 篇文档（智能座舱用车知识库）")
                    with st.expander("预览内置文档清单"):
                        st.dataframe(
                            pd.DataFrame([{"doc_id": d.doc_id, "标题": d.title, "字数": len(d.text)} for d in _docs]),
                            width="stretch", hide_index=True,
                        )
                except Exception as exc:  # noqa: BLE001
                    st.error(f"内置语料加载失败：{exc}")
            elif doc_file is not None:
                st.success(f"✓ 已接收：{doc_file.name}（{doc_file.size / 1024:.1f} KB）")

    with c2:
        with st.container(border=True):
            st.markdown("**评测集**")
            eval_source = st.radio(
                "评测集来源", ["内置真实评测集", "上传 JSON", "上传 CSV"],
                horizontal=True, label_visibility="collapsed",
            )
            eval_file = None
            if eval_source == "上传 JSON":
                eval_file = st.file_uploader(
                    "上传评测集（JSON）", type=["json"], key="eval_json",
                    help='[{"question":"...","ground_truth":"...","relevant_doc_ids":["doc-x"]}]',
                )
            elif eval_source == "上传 CSV":
                eval_file = st.file_uploader(
                    "上传评测集（CSV）", type=["csv"], key="eval_csv",
                    help="列名：question, ground_truth, relevant_doc_ids（分号分隔）",
                )

            if eval_source == "内置真实评测集":
                try:
                    _items = load_eval_set(SAMPLE_EVAL_PATH)
                    st.success(f"✓ 内置评测集就绪：{len(_items)} 条样本（含 ground_truth 与 relevant_doc_ids）")
                    with st.expander("预览内置评测集"):
                        st.dataframe(
                            pd.DataFrame([{"#": i.item_id, "问题": i.question,
                                           "相关文档": ",".join(i.relevant_doc_ids)} for i in _items]),
                            width="stretch", hide_index=True,
                        )
                except Exception as exc:  # noqa: BLE001
                    st.error(f"内置评测集加载失败：{exc}")
            elif eval_file is not None:
                st.success(f"✓ 已接收：{eval_file.name}（{eval_file.size / 1024:.1f} KB）")

    st.markdown('<div class="section-title">模型配置 / Model</div>', unsafe_allow_html=True)
    with st.container(border=True):
        mc1, mc2, mc3 = st.columns(3, gap="medium")
        with mc1:
            st.markdown("**生成模型（LLM）**")
            llm_backend = st.selectbox(
                "LLM 后端", list(LLM_PRESETS.keys()),
                format_func=lambda k: LLM_PRESETS[k]["label"],
                index=list(LLM_PRESETS.keys()).index("offline"),
                help="默认为离线抽取式，避免未明确选择时发送数据或产生 API 费用。",
            )
            preset = LLM_PRESETS[llm_backend]
            if llm_backend == "offline":
                llm_model = "extractive"
                st.info("离线抽取式：从检索结果中提取真实原句作答，无需联网。")
            else:
                llm_model = st.selectbox("模型", preset["models"], index=0)
                env_hint = preset.get("env_key") or ""
                key_input = st.text_input(
                    "API Key", type="password",
                    placeholder=f"留空则读环境变量 {env_hint} / .env",
                    help=f"Base URL：{preset['base_url']}",
                )
                if key_input.strip():
                    _new_key = key_input.strip()
                    if st.session_state.get("_key_applied") != _new_key:
                        os.environ[env_hint or "OPENAI_API_KEY"] = _new_key
                        st.session_state["_key_applied"] = _new_key
                        add_log("已接收界面输入的 API Key，凭据状态已刷新", "ok")
                        st.rerun()
                    CAPS["LLM API Key"] = True
                    CAPS["API Key 来源"] = "界面输入"
                st.caption(f"Base URL：`{preset['base_url']}`")
            llm_temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.05,
                                        disabled=(llm_backend == "offline"))
            if llm_backend == "offline":
                llm_max_tokens = 512
            else:
                llm_max_tokens = st.number_input(
                    "max_tokens（输出上限）", 64, 16384,
                    recommended_max_tokens(llm_model), step=64,
                    key=f"maxtok_{llm_backend}_{llm_model}",
                    help="思维链模型会先消耗推理 token，上限过小会导致输出被截断、内容为空",
                )
                if is_reasoning_model(llm_model):
                    st.caption(f"已识别为思维链模型：运行时自动保证不低于 {MIN_MAX_TOKENS_REASONING}。")

        with mc2:
            st.markdown("**向量模型（Embedding）**")
            embed_backend = st.selectbox(
                "向量后端", list(EMBED_BACKENDS.keys()),
                format_func=lambda k: EMBED_BACKENDS[k]["label"],
            )
            embed_dim = st.selectbox(
                "目标向量维度", EMBED_BACKENDS[embed_backend]["dims"], index=2,
                help="本地后端会按语料规模自动收缩到可用的最大维数",
            )
            if embed_backend == "api":
                embed_api_model = st.selectbox("Embedding 模型", EMBED_API_MODELS, index=0)
                st.caption("使用 `OPENAI_BASE_URL` + `OPENAI_API_KEY`（或 EMBEDDING_* 变量）")
                st.warning("DeepSeek 官方不提供 embedding 接口，此项需指向支持 embedding 的兼容服务。", icon="⚠️")
            else:
                embed_api_model = "text-embedding-3-small"

        with mc3:
            st.markdown("**重排序（Rerank）**")
            rerank_backend = st.selectbox(
                "重排后端", list(RERANK_BACKENDS.keys()),
                format_func=lambda k: RERANK_BACKENDS[k]["label"], index=0,
            )
            rerank_top_k = st.number_input("重排序后保留 Top-K", 1, 50, 3)
            if rerank_backend == "llm" and llm_backend == "offline":
                st.warning("LLM 重排需要可用的 LLM 后端，当前为 offline。", icon="⚠️")

        jc1, jc2 = st.columns([1, 3])
        with jc1:
            use_llm_judge = st.toggle(
                "启用 LLM 判官指标", value=False,
                help="Faithfulness / Answer Relevancy 由 LLM 真实判定；关闭时使用词法代理指标（会明确标注）",
            )
        with jc2:
            if use_llm_judge:
                if (llm_backend != "offline") and CAPS.get("LLM API Key"):
                    st.caption("✅ LLM 判官就绪：忠实度走「论断拆解 + 逐条核查」，相关性走「反推问题 + 相似度比对」。")
                else:
                    st.caption("⚠️ 未配置可用 LLM，判官将自动回退为词法代理指标（结果中会标注 deterministic）。")

    st.markdown('<div class="section-title">检索配置 / Retrieval</div>', unsafe_allow_html=True)
    with st.container(border=True):
        rc1, rc2, rc3 = st.columns(3, gap="medium")
        with rc1:
            st.markdown("**分块**")
            chunk_size = st.selectbox(
                "Chunk Size（字符）", [128, 256, 300, 512, 1024], index=2,
                help="分块越大上下文越完整，越小定位越精确；可对比不同取值的效果",
            )
            overlap_ratio = st.slider("Overlap 比例", 0.05, 0.5, 0.20, 0.05, format="%.0f%%")
            chunk_overlap = int(chunk_size * overlap_ratio)
            st.info(f"实际 overlap = {chunk_overlap} 字符")
        with rc2:
            st.markdown("**检索**")
            top_k = st.number_input("检索 Top-K", 1, 50, 5)
            use_hybrid = st.toggle("启用混合检索（BM25 + 向量）", value=True)
            if use_hybrid:
                bm25_weight = st.slider("BM25 权重", 0.0, 1.0, 0.4, 0.05)
                st.info(f"向量权重：{round(1 - bm25_weight, 2)}")
            else:
                bm25_weight = 0.0
                st.caption("仅使用向量检索")
        with rc3:
            st.markdown("**上下文压缩与缓存**")
            use_compression = st.toggle(
                "启用上下文压缩", value=False,
                help="压缩掉与问题无关的句子。有 LLM 时用 LLM 抽取，否则用词法筛选。",
            )
            use_cache = st.toggle("启用向量缓存（按语料指纹 + 参数校验）", value=True)
            st.caption(f"缓存目录：`{SCRIPT_DIR / 'data' / 'cache'}`")

    # 连通性自检
    cc1, cc2 = st.columns([1, 3])
    with cc1:
        do_check = st.button("🔌 测试 LLM 连通性", disabled=(llm_backend == "offline"))
    with cc2:
        if do_check:
            _S = dict(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap, embed_backend=embed_backend,
                embed_dim=embed_dim, embed_api_model=embed_api_model, top_k=top_k,
                use_hybrid=use_hybrid, bm25_weight=bm25_weight, rerank_backend=rerank_backend,
                rerank_top_k=rerank_top_k, use_compression=use_compression, llm_backend=llm_backend,
                llm_model=llm_model, llm_temperature=llm_temperature, llm_max_tokens=llm_max_tokens,
                use_llm_judge=use_llm_judge, use_cache=use_cache,
            )
            try:
                _r = LLMClient(make_cfg(_S), timeout=60).connectivity_check()
                st.success(
                    f"✅ 连通正常 · 模型 {_r['model']} · 延迟 {_r['latency_s']}s · "
                    f"生效 max_tokens={_r['effective_max_tokens']} · 回复「{_r['reply']}」"
                )
                add_log(f"LLM 连通性自检通过：{_r['model']}，{_r['latency_s']}s", "ok")
            except LLMError as exc:
                st.error(f"❌ 连通失败：{exc}")
                add_log(f"LLM 连通性自检失败：{exc}", "err")

    return {
        "corpus_source": corpus_source, "doc_file": doc_file,
        "eval_source": eval_source, "eval_file": eval_file,
        "chunk_size": chunk_size, "chunk_overlap": chunk_overlap,
        "embed_backend": embed_backend, "embed_dim": embed_dim,
        "embed_api_model": embed_api_model, "top_k": top_k,
        "use_hybrid": use_hybrid, "bm25_weight": bm25_weight,
        "rerank_backend": rerank_backend, "rerank_top_k": rerank_top_k,
        "use_compression": use_compression, "llm_backend": llm_backend,
        "llm_model": llm_model, "llm_temperature": llm_temperature,
        "llm_max_tokens": llm_max_tokens, "use_llm_judge": use_llm_judge,
        "use_cache": use_cache,
    }


with cfg_ph:
    S = render_config_panel()


# ── 知识库 ────────────────────────────────────────────────────
def _kb_metric_row() -> dict:
    """知识库统计条。"""
    stt = _KB.stats()
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("文档数", stt["n_docs"])
    m2.metric("总字数", f"{stt['total_chars']:,}")
    m3.metric("更新次数", stt["n_versions"], help="同来源再次导入且内容有变化时的累计覆盖次数")
    m4.metric("内容指纹", _KB.fingerprint()[:8] if stt["n_docs"] else "—",
              help="知识库内容指纹；它一变，问答/测评的索引缓存就会自动失效重建")
    m5.metric("最后更新", (stt["updated_at"] or "—")[5:16])
    if stt["by_type"]:
        st.caption("类型分布：" + "　".join(f"`{k}` {v} 篇" for k, v in stt["by_type"].items()))
    return stt


def _kb_show_result(res: dict | None) -> None:
    """逐条展示导入结果——成功/更新/跳过/失败都要能被看到，不能只给一个总数。"""
    if not res:
        return
    if res.get("failed"):
        st.error(f"导入结果：{res['summary']}")
    else:
        st.success(f"导入结果：{res['summary']}")
    for e in (res.get("errors") or [])[:8]:
        st.error(f"✕ {e}", icon="✕")
    msgs = res.get("messages") or []
    if msgs:
        with st.expander(f"逐条明细（{len(msgs)} 条）", expanded=bool(res.get("errors"))):
            for m in msgs[:200]:
                st.markdown(f"- {m}")


def _kb_store_upload(uploaded) -> str | None:
    p = materialize_upload(uploaded)
    return p


def render_kb_panel() -> None:
    st.markdown('<div class="section-title">知识库 / Knowledge Base</div>', unsafe_allow_html=True)
    st.caption(
        f"知识库落盘在 `{_KB.root}`，重启后依然存在；导入后**无需手动重建索引**——"
        "内容指纹一变，问答与测评会自动重建并复用同一份索引。"
    )

    _kb_metric_row()

    tb1, tb2, tb3, tb4 = st.columns([1, 1, 1, 1.2])
    with tb1:
        if st.button("🔄 重建索引", width="stretch", disabled=_KB.is_empty(),
                     help="立即丢弃内存中的索引，下次提问/评测时按当前知识库重建"):
            st.session_state.engine = None
            add_log("已丢弃旧索引，下次提问将按最新知识库重建", "warn")
            st.rerun()
    with tb2:
        if st.button("⬇️ 导出 JSON", width="stretch", disabled=_KB.is_empty(),
                     help="导出为可被评测集/CLI 直接复用的标准 JSON 语料"):
            try:
                path = _KB.export_json()
                st.session_state["kb_export_path"] = str(path)
                add_log(f"知识库已导出：{path.name}", "ok")
            except Exception as exc:  # noqa: BLE001
                st.error(f"导出失败：{exc}")
    with tb3:
        if st.button("🧹 清空知识库", width="stretch", disabled=_KB.is_empty()):
            st.session_state.kb_confirm_clear = True
    with tb4:
        if st.session_state.get("kb_export_path"):
            st.caption(f"最近导出：`{Path(st.session_state['kb_export_path']).name}`")

    if st.session_state.get("kb_confirm_clear"):
        st.warning("⚠️ 清空会删除知识库内全部文档（磁盘文件一并删除），且不可撤销。"
                   "确认后请点下方按钮。", icon="⚠️")
        cc1, cc2, _ = st.columns([1, 1, 3])
        with cc1:
            if st.button("确认清空", type="primary", width="stretch"):
                n = _KB.clear()
                st.session_state.chat = []
                st.session_state.engine = None
                st.session_state.kb_confirm_clear = False
                st.session_state.kb_result = {"summary": f"已清空 {n} 篇文档", "added": 0,
                                              "updated": 0, "skipped": 0, "failed": 0,
                                              "messages": [], "errors": []}
                add_log(f"知识库已清空（{n} 篇）", "warn")
                st.rerun()
        with cc2:
            if st.button("取消", width="stretch"):
                st.session_state.kb_confirm_clear = False
                st.rerun()

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # ── 四种导入通道 ──
    st.markdown('<div class="section-title">导入资料 / Import</div>', unsafe_allow_html=True)
    im_file, im_text, im_web, im_builtin = st.tabs(
        ["📁 上传文件", "✍️ 粘贴文本", "🌐 联网导入", "📦 内置样例"]
    )

    with im_file:
        st.caption("支持的类型：" + "　".join(f"`{e}`" for e in SUPPORTED_EXTENSIONS)
                   + "　｜　一次可多选，单篇正文上限 300,000 字（超出自动截断并留痕）")
        ups = st.file_uploader(
            "选择要导入的文件", type=[e.lstrip(".") for e in SUPPORTED_EXTENSIONS],
            accept_multiple_files=True, key="kb_uploads",
        )
        if ups:
            st.caption(f"已选择 {len(ups)} 个文件：" + "、".join(f"`{u.name}`" for u in ups[:6])
                       + ("…" if len(ups) > 6 else ""))
        if st.button("⬆️ 导入选中文件", type="primary", disabled=not ups, key="kb_do_upload"):
            agg = {"added": 0, "updated": 0, "skipped": 0, "failed": 0,
                   "messages": [], "errors": [], "docs": []}
            with st.spinner("解析并入库中…"):
                for u in ups:
                    path = _kb_store_upload(u)
                    r = _KB.add_file(path, tags=["上传"]).to_dict()
                    for k in ("added", "updated", "skipped", "failed"):
                        agg[k] += r[k]
                    agg["messages"] += r["messages"]
                    agg["errors"] += r["errors"]
                    agg["docs"] += r["docs"]
            agg["summary"] = (f"新增 {agg['added']} 篇，更新 {agg['updated']} 篇，"
                              f"去重跳过 {agg['skipped']} 篇，失败 {agg['failed']} 个文件")
            st.session_state.kb_result = agg
            add_log(f"上传导入完成：{agg['summary']}", "ok" if not agg["failed"] else "warn")
            st.rerun()

    with im_text:
        t_title = st.text_input("标题（留空则自动取正文前 24 字）", key="kb_text_title",
                                placeholder="例如：某车型用户手册 · 安全气囊章节")
        t_body = st.text_area("正文", key="kb_text_body", height=200,
                              placeholder="把资料正文粘贴到这里…")
        st.caption(f"当前 {len(t_body or '')} 字。粘贴内容同样参与去重与覆盖更新。")
        if st.button("⬆️ 导入这段文本", type="primary", key="kb_do_text"):
            r = _KB.add_text(t_title, t_body, source="手工粘贴").to_dict()
            st.session_state.kb_result = r
            add_log(f"粘贴导入：{r['summary']}", "ok" if not r["failed"] else "warn")
            st.rerun()

    with im_web:
        st.markdown("**① 直接导入网页**（每行一个 URL）")
        urls = st.text_area("网页地址", key="kb_urls", height=90,
                            placeholder="https://example.com/article-1\nhttps://example.com/article-2")
        c1, c2 = st.columns([3, 1])
        with c2:
            do_url = st.button("⬇️ 抓取并导入", type="primary", width="stretch", key="kb_do_url")
        with c1:
            st.caption("正文会自动清洗（去掉导航/脚本/广告块）；遇到 403 反爬的站点，"
                       "请改用「上传文件」或「粘贴文本」。")
        if do_url:
            lines = [u.strip() for u in (urls or "").splitlines() if u.strip()]
            if not lines:
                st.warning("请先填写至少一个 URL。", icon="⚠️")
            else:
                agg = {"added": 0, "updated": 0, "skipped": 0, "failed": 0,
                       "messages": [], "errors": [], "docs": []}
                bar = st.progress(0.0, text="抓取中…")
                for i, u in enumerate(lines, 1):
                    bar.progress(i / len(lines), text=f"抓取 {u[:60]}…")
                    r = _KB.add_url(u, tags=["联网"]).to_dict()
                    for k in ("added", "updated", "skipped", "failed"):
                        agg[k] += r[k]
                    agg["messages"] += r["messages"]
                    agg["errors"] += r["errors"]
                    agg["docs"] += r["docs"]
                bar.empty()
                agg["summary"] = (f"新增 {agg['added']} 篇，更新 {agg['updated']} 篇，"
                                  f"去重跳过 {agg['skipped']} 篇，失败 {agg['failed']} 个网址")
                st.session_state.kb_result = agg
                add_log(f"网页导入完成：{agg['summary']}", "ok" if not agg["failed"] else "warn")
                st.rerun()

        st.markdown("---")
        st.markdown("**② 关键词搜索并导入**（搜到的候选项可逐条勾选）")
        sc1, sc2, sc3 = st.columns([3, 1, 1])
        with sc1:
            kw = st.text_input("搜索关键词", key="kb_kw", placeholder="例如：智能座舱 功能 有哪些")
        with sc2:
            kw_n = st.number_input("抓取篇数", 1, 6, 3, key="kb_kw_n")
        with sc3:
            st.markdown("<div style='height:26px'></div>", unsafe_allow_html=True)
            do_search = st.button("🔍 搜索", width="stretch", key="kb_do_search")

        st.caption("快速关键词：")
        kcols = st.columns(4)
        for col, preset in zip(kcols, ["智能座舱 定义 功能", "汽车 安全带 正确佩戴",
                                       "AEB 自动紧急制动 原理", "新能源车 OTA 升级 注意事项"]):
            if col.button(preset, key=f"kw_{preset}", width="stretch"):
                st.session_state.kb_kw = preset
                st.rerun()

        if do_search and kw.strip():
            with st.spinner("搜索中…"):
                try:
                    hits = web_search(kw.strip(), n=int(kw_n) * 3)
                    st.session_state.kb_hits = [h.to_dict() for h in hits]
                    st.session_state.kb_hits_query = kw.strip()
                    add_log(f"联网搜索「{kw}」命中 {len(hits)} 条候选", "ok")
                except WebImportError as exc:
                    st.session_state.kb_hits = None
                    st.error(f"搜索失败：{exc}")
                    add_log(f"联网搜索失败：{exc}", "err")

        hits = st.session_state.get("kb_hits")
        if hits:
            n_direct = sum(1 for h in hits if h.get("is_direct"))
            st.caption(f"关键词「{st.session_state.get('kb_hits_query','')}」的候选结果"
                       f"（共 {len(hits)} 条，其中 {n_direct} 条可直连抓取）。"
                       "「可直连」为 ⚠ 的多数是搜索引擎跳转链，抓取常被站点拒绝——"
                       "若要收录它们，建议打开网页后复制正文用「粘贴文本」导入。")
            hdf = pd.DataFrame([
                {"选择": False,
                 "标题": h.get("title", "")[:70],
                 "引擎": h.get("engine", ""),
                 "可直连": "✅" if h.get("is_direct") else "⚠ 跳转链",
                 "摘要": (h.get("snippet") or "")[:90],
                 "链接": h.get("url", "")}
                for h in hits
            ])
            edited = st.data_editor(
                hdf, hide_index=True, width="stretch", key="kb_hits_editor",
                column_config={
                    "选择": st.column_config.CheckboxColumn("选择", width="small"),
                    "标题": st.column_config.TextColumn("标题", width="medium"),
                    "引擎": st.column_config.TextColumn("引擎", width="small"),
                    "可直连": st.column_config.TextColumn("可直连", width="small"),
                    "链接": st.column_config.LinkColumn("链接", width="small"),
                },
            )
            picked_idx = [i for i, v in enumerate(edited["选择"].tolist()) if v is True]
            if st.button(f"⬆️ 导入选中（{len(picked_idx)} 篇）", type="primary",
                         disabled=not picked_idx, key="kb_do_import_hits"):
                from types import SimpleNamespace

                chosen = [SimpleNamespace(**hits[i]) for i in picked_idx]
                with st.spinner("抓取正文并入库…"):
                    agg = _KB.add_search_results(chosen, n=len(chosen), tags=["联网"]).to_dict()
                st.session_state.kb_result = agg
                add_log(f"搜索导入完成：{agg['summary']}", "ok" if not agg["failed"] else "warn")
                st.rerun()

    with im_builtin:
        st.caption("内置 11 篇智能座舱用车知识（真实样例语料），用于冷启动立刻体验问答与评测。")
        c1, c2 = st.columns([1, 3])
        with c1:
            if st.button("📦 导入内置样例", type="primary", width="stretch", key="kb_do_builtin"):
                with st.spinner("导入中…"):
                    r = _KB.add_builtin().to_dict()
                st.session_state.kb_result = r
                add_log(f"内置样例导入：{r['summary']}", "ok")
                st.rerun()
        with c2:
            st.caption("导入后可在下方清单里看到 11 篇文档，来源标记为 `内置样例`。")

    result = st.session_state.pop("kb_result", None)
    if result and result.get("summary"):
        _kb_show_result(result)

    # ── 文档清单与删除 ──
    st.markdown('<div class="section-title">文档清单 / Documents</div>', unsafe_allow_html=True)
    entries = _KB.entries()
    if not entries:
        st.markdown("""
        <div class="empty-state">
            <div class="big">📚</div>
            <div style="font-weight:600;margin-bottom:4px;">知识库还是空的</div>
            <div style="font-size:.78rem;">用上面任意一种方式导入资料，然后回到「知识问答」提问，
            回答会严格基于你导入的内容。</div>
        </div>
        """, unsafe_allow_html=True)
        return

    df = pd.DataFrame(_KB.preview())
    df.insert(0, "删除", False)
    show_cols = ["删除", "标题", "类型", "字数", "版本", "来源", "更新时间"]
    edited = st.data_editor(
        df[show_cols], hide_index=True, width="stretch", key="kb_docs_editor",
        column_config={
            "删除": st.column_config.CheckboxColumn("删除", width="small",
                                                  help="勾选后点下方「删除选中文档」"),
            "标题": st.column_config.TextColumn("标题", width="large"),
            "类型": st.column_config.TextColumn("类型", width="small"),
            "字数": st.column_config.NumberColumn("字数", width="small"),
            "版本": st.column_config.NumberColumn("版本", width="small"),
        },
    )
    del_ids = [entries[i]["doc_id"] for i, v in enumerate(edited["删除"].tolist()) if v is True]
    d1, d2, d3 = st.columns([1, 1, 3])
    with d1:
        if st.button(f"🗑 删除选中（{len(del_ids)}）", disabled=not del_ids, width="stretch"):
            n = _KB.remove_many(del_ids)
            st.session_state.engine = None
            add_log(f"已删除 {n} 篇文档，索引将在下次提问时重建", "warn")
            st.rerun()
    with d2:
        if st.button("↺ 重置勾选", width="stretch"):
            st.rerun()
    with d3:
        st.caption("删除只影响知识库本身；已生成的评测报告保留在 `data/reports/`，不受影响。")


with kb_ph:
    render_kb_panel()


# ── 知识问答 ──────────────────────────────────────────────────
def _source_card(rc: dict, show_scores: bool) -> str:
    meta = f"#{rc['rank']} · {html.escape(str(rc['doc_title']), quote=True)}"
    if show_scores:
        meta += (f" · score={rc['score']} · vec={rc['vector_score']} "
                 f"· bm25={rc['bm25_score']} · rerank={rc['rerank_score']}")
    meta += f" · doc_id={html.escape(str(rc['doc_id']), quote=True)}"
    return (f'<div class="bubble-src"><div class="h">{meta}</div>'
            f'<div>{html.escape(str(rc["text"]), quote=True)}</div></div>')


def render_turn(msg: dict) -> None:
    """渲染一轮问答中的一条消息。

    必须严格按 role 分支：早期版本对每条消息都同时渲染「用户气泡 + 助手气泡」，
    结果 2 条消息渲染成了 4 个气泡（一条问答看起来像两轮）。
    """
    if msg.get("role") == "user":
        with st.chat_message("user", avatar="🧑"):
            st.markdown(msg["content"])
        return

    with st.chat_message("assistant", avatar="🚗"):
        meta = msg.get("meta") or {}
        if meta.get("error"):
            st.error(f"生成失败：{meta['error']}")
        st.markdown(msg["content"] or "_（空）_")

        if st.session_state.ui_show_timing and meta:
            bits = [f"生成模式：`{meta.get('mode', '-')}`"]
            if meta.get("history_turns"):
                bits.append(f"带入 {meta['history_turns']} 轮上下文")
            if meta.get("retrieve_ms") is not None:
                bits.append(f"检索 {meta['retrieve_ms']} ms")
            if meta.get("generate_ms") is not None:
                bits.append(f"生成 {meta['generate_ms']} ms")
            if meta.get("compress_ms"):
                bits.append(f"其中压缩 {meta['compress_ms']} ms")
            if meta.get("contexts") is not None:
                bits.append(f"上下文 {meta['contexts']} 段")
            st.caption("　|　".join(bits))

        # 多轮追问时，把「实际拿去检索的查询」摊开给用户看——
        # 否则检索跑偏了（比如追问没被正确改写）用户只会觉得"答非所问"，无从排查。
        if st.session_state.ui_show_rewrite and meta.get("rewritten_query"):
            st.caption(f"🔎 追问已改写为检索词：`{meta['rewritten_query']}`")

        srcs = meta.get("sources") or []
        if st.session_state.ui_show_sources and srcs:
            with st.expander(f"📚 召回来源（{len(srcs)} 段，按真实召回顺序）"):
                for rc in srcs:
                    st.markdown(_source_card(rc, st.session_state.ui_show_scores),
                                unsafe_allow_html=True)


def _answer(question: str, S: dict, history: Optional[list] = None) -> dict:
    """真实问答：走与测评完全相同的检索 + 生成链路。

    ``history`` 为之前若干轮问答（不含本轮问题），用于多轮追问的指代理解。
    """
    prog = st.progress(0.0, text="准备知识库…")
    status = st.empty()

    def _cb(frac: float, msg: str) -> None:
        prog.progress(min(max(frac, 0.0), 1.0), text=msg)
        status.caption(msg)

    try:
        eng, cfg, rebuilt = build_engine(S, progress=_cb)
        if rebuilt:
            add_log(f"知识库构建完成：{eng['stats'].get('n_chunks')} 个分块", "ok")
        prog.progress(1.0, text="检索并生成答案…")
        ev = eng["ev"]

        # 多轮：追问先改写成可独立检索的查询，再拿去检索
        search_q, rewritten = build_retrieval_query(
            question, history, ev.generator.llm
        )

        t0 = time.perf_counter()
        retrieved = ev.retriever.retrieve(search_q)
        t_ret = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        gen = ev.generator.generate(
            question, retrieved, history=history,
            max_history_turns=int(st.session_state.ui_history_turns),
        )
        t_gen = (time.perf_counter() - t0) * 1000

        sources = []
        for rc in retrieved:
            d = rc.to_dict()
            d["text"] = rc.chunk.text[:420] + ("…" if len(rc.chunk.text) > 420 else "")
            sources.append(d)

        return {
            "content": gen.answer or "",
            "meta": {
                "mode": gen.mode,
                "retrieve_ms": round(t_ret, 1),
                "generate_ms": round(t_gen, 1),
                "compress_ms": round(gen.compress_s * 1000, 1),
                "contexts": len(gen.used_chunk_ids),
                "sources": sources,
                "error": gen.error,
                "rewritten_query": search_q if rewritten else None,
                "history_turns": gen.history_turns,
                "n_docs": eng.get("n_docs"),
                "kb_fp": eng.get("kb_fp") or "",
            },
        }
    except (IngestError, EvalError, LLMError, WebImportError, ValueError) as exc:
        add_log(f"问答失败：{exc}", "err")
        return {"content": "", "meta": {"error": str(exc), "sources": []}}
    except Exception as exc:  # noqa: BLE001
        add_log(f"未预期错误：{type(exc).__name__}: {exc}", "err")
        return {"content": "", "meta": {"error": f"{type(exc).__name__}: {exc}", "sources": []}}
    finally:
        prog.empty()
        status.empty()


with chat_ph:
    st.markdown('<div class="section-title">知识库问答 / Ask the Knowledge Base</div>', unsafe_allow_html=True)

    _kb_st = _KB.stats()
    bar1, bar2, bar3, bar4 = st.columns([2.6, 1, 1, 1])
    with bar1:
        _e = st.session_state.engine
        if _e:
            bs = _e["stats"]
            st.caption(
                f"✅ 索引已就绪：{_e['n_docs']} 篇文档 / {bs.get('n_chunks')} 个分块　"
                f"｜向量 {bs.get('embed_backend')} ｜索引 {bs.get('vector_index_backend')}"
                f"　｜{'LLM 生成' if _e['ev'].generator.mode == 'llm' else '离线抽取式'}"
            )
        else:
            st.caption(
                f"⚙️ 索引尚未构建。知识库现有 {_kb_st['n_docs']} 篇 / {_kb_st['total_chars']} 字，"
                "发送第一个问题时自动构建（有缓存时仅需瞬间）。"
            )
    with bar2:
        if st.button("🔄 重建索引", width="stretch", disabled=not st.session_state.engine):
            st.session_state.engine = None
            add_log("已丢弃旧索引，下次提问将重建", "warn")
            st.rerun()
    with bar3:
        if st.button("🆕 新会话", width="stretch", disabled=not st.session_state.chat,
                     help="清空对话历史（不影响知识库与索引）。多轮上下文会随之归零。"):
            st.session_state.chat = []
            add_log("已开启新会话，多轮上下文归零", "info")
            st.rerun()
    with bar4:
        if st.button("🧹 清空输入", width="stretch"):
            st.session_state["chat_input"] = ""
            st.rerun()

    n_rounds = len([m for m in st.session_state.chat if m.get("role") == "user"])
    if n_rounds:
        st.caption(
            f"💬 当前会话已进行 **{n_rounds}** 轮，"
            f"提问时会自动带入最近 **{st.session_state.ui_history_turns}** 轮上下文"
            + ("（追问会被改写成可独立检索的问题）" if st.session_state.ui_show_rewrite else "")
        )

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    history_msgs = st.session_state.chat
    if not history_msgs:
        if _kb_st["n_docs"]:
            _empty_msg = (f"知识库现有 <b>{_kb_st['n_docs']}</b> 篇文档、"
                          f"{_kb_st['total_chars']:,} 字。回答只依据检索到的原文并标注 [1][2] 引用。")
        else:
            _empty_msg = ("知识库还是空的，先去 <b>📚 知识库</b> 页导入资料"
                          "（上传文件 / 粘贴文本 / 联网导入），再回来提问。")
        st.markdown(f"""
        <div class="empty-state">
            <div class="big">💬</div>
            <div style="font-weight:600;margin-bottom:4px;">向你的知识库提问（支持多轮追问）</div>
            <div style="font-size:.78rem;">{_empty_msg}</div>
        </div>
        """, unsafe_allow_html=True)
        try:
            _samples = load_eval_set(SAMPLE_EVAL_PATH)[:4]
            st.caption("试试这些问题（点一下会直接提问）：")
            bcols = st.columns(len(_samples))
            for col, it in zip(bcols, _samples):
                if col.button(it.question[:18] + ("…" if len(it.question) > 18 else ""),
                              key=f"ex_{it.item_id}", width="stretch"):
                    st.session_state.pending_q = it.question
                    st.rerun()
        except Exception:  # noqa: BLE001
            pass
    else:
        for m in history_msgs:
            render_turn(m)

    with st.form("chat_form", clear_on_submit=True, border=False):
        fc1, fc2 = st.columns([6, 1])
        with fc1:
            typed = st.text_input(
                "提问", label_visibility="collapsed", key="chat_input",
                placeholder="继续追问，例如「那后排呢？」「它支持哪些方言？」（回车或点「提问」发送）",
            )
        with fc2:
            sent = st.form_submit_button("提问", width="stretch")

    pending = st.session_state.pop("pending_q", None)
    question = (pending or typed or "").strip()
    if question and (sent or pending):
        st.session_state.chat.append({"role": "user", "content": question})
        # 交给模型的历史不含本轮问题（本轮问题单独作为 question 传入），
        # 否则改写逻辑会把"当前问题"误当成上一轮问题拼接进检索词。
        prior_history = list(st.session_state.chat[:-1])
        with st.spinner("检索并生成中…"):
            res = _answer(question, S, prior_history)
        st.session_state.chat.append({"role": "assistant", "content": res["content"], "meta": res["meta"]})
        _turns = (res["meta"] or {}).get("history_turns") or 0
        add_log(f"问答完成（带入 {_turns} 轮上下文）：{question[:28]}", "ok")
        # 输入框带 key 时 clear_on_submit 不清值，这里显式清空，避免问完还留着旧问题
        try:
            st.session_state["chat_input"] = ""
        except Exception:  # noqa: BLE001
            pass
        st.rerun()


# ── 能力测评 ──────────────────────────────────────────────────
with eval_ph:
    st.markdown('<div class="section-title">真实评测 / Real Evaluation Pipeline</div>', unsafe_allow_html=True)
    st.caption(
        "与「知识问答」共用同一套索引与生成链路；不同的是这里会逐题对照评测集标准答案，"
        "计算检索类 / 上下文类 / 答案类 / 判官类全套指标。"
    )

    eb1, eb2, eb3 = st.columns([1, 2, 1])
    with eb2:
        run_btn = st.button("▶  开始真实评测", width="stretch", disabled=st.session_state.running)
        st.caption("评测会真实加载语料、构建索引并逐题推理；离线模式下首次运行约数秒。")

    # 内置评测集的 relevant_doc_ids 是 `doc-xxx`，只对应「内置真实样例语料」。
    # 若用户把语料换成「已导入知识库」还沿用内置评测集，检索类指标必然全 0 ——
    # 这不是系统故障，是标注对不上语料。运行前先提醒，避免白跑一轮还误判成 bug。
    if S.get("corpus_source") != "内置真实样例语料" and S.get("eval_source") == "内置真实评测集":
        st.warning(
            "**标注可能对不上语料**：当前语料是「"
            f"{S.get('corpus_source')}」，而评测集用的是**内置评测集** —— "
            "它的 `relevant_doc_ids`（`doc-xxx`）只存在于「内置真实样例语料」里。"
            "直接跑的话，检索类指标（hit_rate / recall_at_k / mrr / ndcg）会全为 0。\n\n"
            "想评测你自己的知识库，需要**上传配套评测集**（`relevant_doc_ids` 填你库里的 doc_id，"
            "可在「📚 知识库」页的文档清单里查到）；"
            "只关心答案质量的话，把语料切回「内置真实样例语料」即可。",
            icon="⚠️",
        )

    if run_btn:
        st.session_state.running = True
        st.session_state.logs = []
        cfg = make_cfg(S)

        prog = st.progress(0.0, text="准备中…")
        status = st.empty()
        log_box = st.empty()

        def on_progress(frac: float, msg: str) -> None:
            prog.progress(min(max(frac, 0.0), 1.0), text=msg)
            status.caption(msg)
            add_log(msg, "ok" if frac >= 1.0 else "info")
            log_box.markdown(
                f'<div class="log-box">{"<br>".join(st.session_state.logs[-200:])}</div>',
                unsafe_allow_html=True,
            )

        try:
            on_progress(0.01, "解析语料来源…")
            docs = load_docs(S)
            on_progress(0.03, "解析评测集…")
            items = load_items(S)

            # 精确核对：评测集标注的 doc_id 在当前语料里能不能找到。
            # 一个都找不到时，IR 类指标必然全 0 —— 必须说清是「标注对不上语料」而不是系统坏了，
            # 否则用户会拿着一张全 0 的报告得出错误结论。
            _ref_ids = set()
            for _it in items:
                _ref_ids.update(_it.relevant_doc_ids or [])
            _doc_ids = {d.doc_id for d in docs}
            _miss = _ref_ids - _doc_ids
            if _ref_ids and _miss == _ref_ids:
                _msg = (f"评测集标注的 {len(_ref_ids)} 个 doc_id 在本次语料（{len(_doc_ids)} 篇）里一个都不存在，"
                        f"因此 hit_rate / recall_at_k / mrr / ndcg 会是 0 —— 这是**标注与语料不匹配**，不是系统故障。"
                        f"例如标注用的「{sorted(_ref_ids)[0]}」并不在当前语料中。"
                        f"要评测这份语料，请上传 relevant_doc_ids 与之匹配的评测集"
                        f"（doc_id 可在「📚 知识库」页文档清单里查到）。")
                add_log(_msg, "warn")
                st.warning(_msg, icon="⚠️")
            elif _miss:
                add_log(
                    f"提示：评测集有 {len(_miss)}/{len(_ref_ids)} 个标注 doc_id 不在当前语料中，"
                    f"这些题的检索指标会偏低。", "warn")

            evaluator = Evaluator(cfg)
            build_stats = evaluator.prepare(docs, progress=on_progress)
            report = evaluator.run(items, progress=on_progress)

            report_d = report.to_dict()
            report_d["build_stats"] = {**build_stats, **report.build_stats}
            st.session_state.report = report_d
            # 顺带把这份索引留在会话里，问答页可直接复用，不必重复构建
            _kb_fp = _KB.fingerprint() if S.get("corpus_source") == KB_SOURCE else ""
            st.session_state.engine = {
                "sig": cfg_signature(cfg, _kb_fp), "ev": evaluator,
                "stats": {**build_stats, **evaluator.index.build_stats},
                "n_docs": len(docs), "kb_fp": _kb_fp,
            }
            add_log(f"评测完成：{len(report.records)} 题，指标已聚合", "ok")
            status.success("✅ 真实评测完成")
        except (IngestError, EvalError, LLMError, ValueError) as exc:
            add_log(f"运行失败：{exc}", "err")
            status.error(f"运行失败：{exc}")
        except Exception as exc:  # noqa: BLE001
            add_log(f"未预期错误：{type(exc).__name__}: {exc}", "err")
            status.error(f"未预期错误：{type(exc).__name__}: {exc}")
        finally:
            st.session_state.running = False

    report = st.session_state.report
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    if not report:
        st.markdown("""
        <div class="empty-state">
            <div class="big">📊</div>
            <div style="font-weight:600;margin-bottom:4px;">暂无评测结果</div>
            <div style="font-size:.78rem;">在「配置中心」确认参数后，点上方按钮开始真实评测</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        aggs = report["aggregates"]
        meta = report["metric_meta"]
        rt = report["runtime"]
        cfg_r = report["config"]

        s1, s2, s3, s4 = st.columns(4)
        s1.markdown(f"""<div class="metric-card"><div class="label">语料分块 / Chunks</div>
        <div class="value">{report['build_stats'].get('n_chunks', '-')}</div></div>""", unsafe_allow_html=True)
        s2.markdown(f"""<div class="metric-card"><div class="label">评测题数 / Items</div>
        <div class="value">{rt.get('n_items', '-')}</div></div>""", unsafe_allow_html=True)
        s3.markdown(f"""<div class="metric-card"><div class="label">平均检索 / ms</div>
        <div class="value">{rt.get('avg_retrieve_ms', '-')}</div></div>""", unsafe_allow_html=True)
        s4.markdown(f"""<div class="metric-card"><div class="label">平均生成(纯) / ms</div>
        <div class="value">{rt.get('avg_generate_net_ms', rt.get('avg_generate_ms', '-'))}</div></div>""",
                    unsafe_allow_html=True)

        if rt.get("avg_compress_ms"):
            st.caption(
                f"计时口径：平均生成 {rt.get('avg_generate_ms')} ms 为含上下文压缩的墙钟时间，"
                f"其中压缩 {rt.get('avg_compress_ms')} ms／题，纯生成 {rt.get('avg_generate_net_ms')} ms／题。"
            )

        def _cc(v):
            if v is None:
                return ""
            return "good" if v >= 0.85 else ("mid" if v >= 0.65 else "bad")

        primary = [
            ("faithfulness", "Faithfulness"), ("answer_relevancy", "Ans Relevancy"),
            ("context_precision", "Ctx Precision"), ("context_recall", "Ctx Recall"),
            ("context_f1", "Ctx F1"), ("token_f1", "Ans Token F1"),
        ]
        cols = st.columns(len(primary))
        for col, (key, label) in zip(cols, primary):
            v = aggs.get(key)
            m = meta.get(key, {})
            method = m.get("method")
            badge = "method-llm" if method == "llm-judge" else "method-det"
            badge_text = "LLM" if method == "llm-judge" else ("MIX" if method == "mixed" else "DET")
            col.markdown(f"""<div class="metric-card">
                <div class="label">{label}</div>
                <div class="value {_cc(v)}">{'—' if v is None else f'{v:.4f}'}</div>
                <div style="margin-top:6px"><span class="method-badge {badge}">{badge_text}</span></div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        with st.expander("📐 指标口径表（每个指标的来源与含义）"):
            rows = [{"指标": k, "均值": v,
                     "计算方式": ("LLM 判官" if m.get("method") == "llm-judge"
                                  else ("混合口径" if m.get("method") == "mixed" else "确定性计算")),
                     "说明": m.get("detail", "")}
                    for k, m in meta.items()]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        if report.get("warnings"):
            for w in report["warnings"]:
                st.warning(w, icon="⚠️")

        _llm = rt.get("llm")
        _judge = rt.get("judge")
        with st.expander("📊 本次运行的真实 LLM 用量与调用健康度", expanded=bool(_llm)):
            if _llm:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("LLM 调用次数", _llm.get("calls", "-"))
                c2.metric("重试次数", _llm.get("retries", "-"))
                c3.metric("输出截断次数", _llm.get("truncated_calls", "-"))
                c4.metric("total tokens", _llm.get("total_tokens", "-"))
                st.caption(
                    f"模型 {_llm.get('model')}（思维链模型：{_llm.get('is_reasoning_model')}）"
                    f" @ {_llm.get('base_url')}｜prompt={_llm.get('prompt_tokens')} "
                    f"completion={_llm.get('completion_tokens')}｜"
                    f"LLM 累计耗时 {_llm.get('llm_latency_s')}s｜"
                    f"token 上限自动抬升 {_llm.get('max_tokens_raised')} 次"
                )
            else:
                st.info("本次未调用任何 LLM：生成走离线抽取式，忠实度/相关性为词法代理指标。")
            if _judge:
                st.caption(f"判官口径：{_judge.get('method')}｜调用 {_judge.get('judge_calls', '-')} 次"
                           f"｜耗时 {_judge.get('judge_latency_s', '-')}s")
            elif cfg_r.get("use_llm_judge"):
                st.warning("已请求 LLM 判官但不可用，忠实度/相关性已回退为词法代理指标。", icon="⚠️")

        st.markdown('<div class="section-title">指标汇总 / Aggregates</div>', unsafe_allow_html=True)
        agg_df = pd.DataFrame(
            [{"指标": k, "均值": v,
              "计算方式": ("LLM 判官" if meta.get(k, {}).get("method") == "llm-judge"
                           else ("混合口径" if meta.get(k, {}).get("method") == "mixed" else "确定性计算")),
              "类别": meta.get(k, {}).get("family", "")}
             for k, v in aggs.items()]
        )
        st.dataframe(agg_df, width="stretch", hide_index=True)

        st.markdown('<div class="section-title">逐题明细 / Per-Item Detail</div>', unsafe_allow_html=True)
        rec_rows = []
        for r in report["records"]:
            rec_rows.append({
                "#": r["index"], "问题": r["question"],
                "模型答案": (r["answer"] or "")[:120],
                "标准答案": (r["ground_truth"] or "")[:120],
                "TokenF1": r["metrics"].get("token_f1"),
                "CtxPrec": r["metrics"].get("context_precision"),
                "CtxRec": r["metrics"].get("context_recall"),
                "Faithful": r["metrics"].get("faithfulness"),
                "错误": r["error"] or "",
            })
        st.dataframe(pd.DataFrame(rec_rows), width="stretch", hide_index=True)

        for r in report["records"]:
            title = f"Q{r['index']}　{r['question']}"
            if r["error"]:
                title = "❌ " + title
            with st.expander(title):
                a, b = st.columns([1, 1])
                with a:
                    st.markdown("**模型答案**")
                    st.markdown(f"> {r['answer'] or '（空）'}")
                    st.caption(
                        f"生成模式：{r['generation_mode']}　|　耗时：检索 "
                        f"{r['timings'].get('retrieve_s', 0)}s / 生成 {r['timings'].get('generate_s', 0)}s"
                    )
                    if r["error"]:
                        st.error(r["error"])
                    st.markdown("**标准答案（ground_truth）**")
                    st.markdown(f"> {r['ground_truth']}")
                with b:
                    st.markdown("**本题指标**")
                    mrows = [{"指标": k, "值": v,
                              "计算方式": "LLM 判官" if r["metric_meta"].get(k, {}).get("method") == "llm-judge"
                              else "确定性"}
                             for k, v in r["metrics"].items()]
                    st.dataframe(pd.DataFrame(mrows), width="stretch", hide_index=True)

                st.markdown("**检索到的上下文（真实召回顺序）**")
                rel = set(r["relevant_doc_ids"])
                for rc in r["retrieved"]:
                    hit = "hit" if rc["doc_id"] in rel else ""
                    mark = "✅ 标注相关" if rc["doc_id"] in rel else "—"
                    safe_title = html.escape(str(rc["doc_title"]), quote=True)
                    safe_doc_id = html.escape(str(rc["doc_id"]), quote=True)
                    safe_text = html.escape(str(rc["text"]), quote=True)
                    st.markdown(f"""<div class="retr-card {hit}">
                        <b>#{rc['rank']} · {safe_title}</b>
                        <div class="meta">doc_id={safe_doc_id} · score={rc['score']} · vec={rc['vector_score']}
                        · bm25={rc['bm25_score']} · rerank={rc['rerank_score']} · {mark}</div>
                        <div>{safe_text}</div>
                    </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        e1, e2, e3, _ = st.columns([1, 1, 1, 3])
        ts = time.strftime("%Y%m%d_%H%M%S")
        with e1:
            st.download_button("⬇ 完整 JSON", data=json.dumps(report, ensure_ascii=False, indent=2),
                               file_name=f"rag_eval_full_{ts}.json", mime="application/json",
                               width="stretch")
        with e2:
            st.download_button("⬇ 汇总 CSV", data=agg_df.to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"rag_eval_agg_{ts}.csv", mime="text/csv", width="stretch")
        with e3:
            st.download_button("⬇ 逐题 CSV",
                               data=pd.DataFrame(rec_rows).to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"rag_eval_items_{ts}.csv", mime="text/csv", width="stretch")

        with st.expander("🔧 本次运行的完整配置与运行环境（可复现）"):
            _req = report.get("config_requested") or {}
            if _req and _req != cfg_r:
                st.warning("实际生效配置与请求配置存在差异（因能力缺失自动降级），详见下方对比。", icon="⚠️")
            st.json({
                "config_requested": _req, "config_effective": cfg_r,
                "capabilities": report["capabilities"], "build_stats": report["build_stats"],
                "runtime": rt, "started_at": report["started_at"], "finished_at": report["finished_at"],
            })


# ── 运行日志 ──────────────────────────────────────────────────
with log_ph:
    st.markdown('<div class="section-title">运行日志 / Run Log</div>', unsafe_allow_html=True)
    lc1, lc2 = st.columns([3, 1])
    with lc2:
        if st.button("🧽 清空日志", width="stretch", disabled=not st.session_state.logs):
            st.session_state.logs = []
            st.rerun()
    if not st.session_state.logs:
        st.caption("暂无日志。执行问答或测评后，这里会记录每一步的真实进度与降级提示。")
    else:
        st.markdown(
            f'<div class="log-box">{"<br>".join(st.session_state.logs)}</div>',
            unsafe_allow_html=True,
        )


st.markdown(f"""
<div style="text-align:center;padding:20px;color:{THEME['muted']};font-size:.7rem;
font-family:'JetBrains Mono',monospace;border-top:1px solid {THEME['border']};margin-top:22px;">
    智能座舱 RAG 工作台 v2.4 · 真实流水线：语料解析 → 分块 → 向量化 → FAISS+BM25 → 混合检索 → 重排 → 生成 → 指标
</div>
""", unsafe_allow_html=True)
