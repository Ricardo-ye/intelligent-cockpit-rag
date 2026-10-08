"""LLM 客户端：OpenAI 兼容协议（DeepSeek / 自建中转 / OpenAI）。

只做真实网络调用，不做任何伪造。任何失败都会抛出携带原始错误信息的异常，
由上层决定是降级还是中断，绝不静默返回编造内容。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .config import (
    MIN_MAX_TOKENS_REASONING,
    RunConfig,
    is_reasoning_model,
    recommended_max_tokens,
)


class LLMError(RuntimeError):
    """LLM 调用失败（网络、鉴权、额度、协议等），携带可读原因。"""


# ─────────────────────────────────────────────
# 模型能力适配
# ─────────────────────────────────────────────
# 思维链模型的 token 下限与预算策略统一定义在 config 中（单一事实来源），
# 这里只做引用，避免两处常量漂移。


@dataclass
class LLMUsage:
    """单次调用的真实资源消耗，用于运行时统计。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_s: float = 0.0
    calls: int = 0
    last_finish_reason: str = ""
    truncated_calls: int = 0          # finish_reason == "length" 的次数
    max_tokens_raised: int = 0        # 因模型能力而抬高 token 上限的次数
    retries: int = 0                  # 因暂时性错误而重试的次数


class LLMClient:
    """OpenAI 兼容 Chat Completions 客户端。"""

    def __init__(self, cfg: RunConfig, timeout: Optional[int] = None):
        self.cfg = cfg
        self.api_key = cfg.resolved_api_key()
        self.base_url = cfg.resolved_base_url()
        self.model = cfg.llm_model
        self.usage = LLMUsage()
        self._client = None
        # 是否思维链模型：决定 max_tokens 是否施加下限保护
        self.is_reasoning = is_reasoning_model(self.model)
        self.last_effective_max_tokens = 0

        if cfg.llm_backend == "offline":
            return
        if not self.api_key:
            raise LLMError(
                f"未检测到 API Key。请在 .env 写入 {cfg.llm_backend.upper()}_API_KEY，"
                "或在侧边栏直接填入。"
            )
        if not self.base_url:
            raise LLMError(f"后端 {cfg.llm_backend} 未配置 base_url。")

        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise LLMError(f"缺少 openai 包，无法发起真实调用：{exc}") from exc

        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=timeout or cfg.judge_request_timeout,
            max_retries=0,   # 由本类自行重试，便于统计与给出可读的错误归因
        )
        self.max_attempts = max(1, int(cfg.llm_max_attempts))
        self.retry_base_delay = float(cfg.llm_retry_base_delay)

    # ── 可用性 ──
    @property
    def available(self) -> bool:
        return self._client is not None

    # ── token 预算 ──
    def _effective_max_tokens(self, requested: Optional[int]) -> int:
        """解析本次调用的 max_tokens，并对思维链模型施加下限保护。

        对 reasoner 类模型，过小的上限会让推理吃光预算、最终 content 为空，
        因此这里把上限抬到 ``MIN_MAX_TOKENS_REASONING``，并记录被抬高的次数。
        """
        base = requested if requested is not None else (
            self.cfg.llm_max_tokens or recommended_max_tokens(self.model)
        )
        base = int(base)
        if self.is_reasoning and base < MIN_MAX_TOKENS_REASONING:
            self.usage.max_tokens_raised += 1
            return MIN_MAX_TOKENS_REASONING
        return base

    # ── 错误归因 ──
    RETRYABLE_STATUS = (408, 409, 425, 429, 500, 502, 503, 504)

    def _classify_error(self, exc: Exception) -> Tuple[bool, str]:
        """把底层异常翻译成「是否可重试 + 用户可读原因」。

        配置类错误（Key 无效 / 额度不足 / 模型不存在）快速失败，
        暂时性错误（限流 / 超时 / 连接抖动 / 5xx）才重试，避免无谓等待。
        """
        status = getattr(exc, "status_code", None)
        name = type(exc).__name__
        msg = str(exc)
        low = msg.lower()

        if status in (401, 403) or "authenticationerror" in name.lower() or "invalid api key" in low:
            return False, f"鉴权失败（HTTP {status}）：API Key 无效、已失效或无权访问该模型。请核对 .env 中的 DEEPSEEK_API_KEY。"
        if status == 402 or "insufficient balance" in low or "quota" in low or "exceeded" in low:
            return False, f"额度不足（HTTP {status}）：账户余额或配额已用尽，请充值或更换 Key。"
        if status == 404:
            return False, f"模型不存在（HTTP 404）：请检查模型名 `{self.model}` 是否正确。"
        if status == 400:
            return False, f"请求参数被拒绝（HTTP 400）：{msg[:200]}"
        if status in self.RETRYABLE_STATUS or "timeout" in name.lower() or "connection" in name.lower():
            return True, f"暂时性错误（{name}，HTTP {status}）：{msg[:160]}"
        return False, f"{name}：{msg[:200]}"

    # ── 对话 ──
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> str:
        """发起一次真实 Chat 调用并返回文本内容。

        失败一律抛 ``LLMError``；**绝不返回空串冒充成功**：
        - 暂时性错误（限流/超时/5xx）按指数退避自动重试 ``llm_max_attempts`` 次
        - ``finish_reason == "length"`` 说明输出被截断（思维链模型尤其常见）
        - 内容为空说明模型没有产出可用答案
        后两者都会被显式报错，避免把空答案当成有效结果写进评测报告。
        """
        if self._client is None:
            raise LLMError("LLM 客户端未初始化（离线模式或缺少凭据）。")

        eff_max = self._effective_max_tokens(max_tokens)
        self.last_effective_max_tokens = eff_max

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.cfg.llm_temperature if temperature is None else temperature,
            "max_tokens": eff_max,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        resp = None
        last_reason = ""
        for attempt in range(1, self.max_attempts + 1):
            t0 = time.perf_counter()
            try:
                resp = self._client.chat.completions.create(**kwargs)
                self.usage.latency_s += time.perf_counter() - t0
                break
            except Exception as exc:
                self.usage.latency_s += time.perf_counter() - t0
                retryable, reason = self._classify_error(exc)
                last_reason = reason
                if retryable and attempt < self.max_attempts:
                    self.usage.retries += 1
                    time.sleep(self.retry_base_delay * (2 ** (attempt - 1)))
                    continue
                raise LLMError(
                    f"LLM 调用失败（{self.model} @ {self.base_url}，"
                    f"第 {attempt}/{self.max_attempts} 次尝试）：{reason}"
                ) from exc

        if resp is None:
            raise LLMError(f"LLM 调用失败（{self.model} @ {self.base_url}）：{last_reason}")

        self.usage.calls += 1
        u = getattr(resp, "usage", None)
        if u:
            self.usage.prompt_tokens += getattr(u, "prompt_tokens", 0) or 0
            self.usage.completion_tokens += getattr(u, "completion_tokens", 0) or 0
            self.usage.total_tokens += getattr(u, "total_tokens", 0) or 0

        try:
            choice = resp.choices[0]
            content = (choice.message.content or "").strip()
            finish_reason = getattr(choice, "finish_reason", "") or ""
        except Exception as exc:
            raise LLMError(f"LLM 返回结构异常：{resp!r}") from exc

        self.usage.last_finish_reason = finish_reason

        if finish_reason == "length":
            self.usage.truncated_calls += 1
            reasoning_len = len(getattr(choice.message, "reasoning_content", "") or "")
            hint = (
                "该模型会先消耗推理 token，当前 max_tokens 不足"
                if reasoning_len or self.is_reasoning
                else "输出达到长度上限"
            )
            raise LLMError(
                f"输出被截断（finish_reason=length，max_tokens={eff_max}，"
                f"思维链长度={reasoning_len} 字符）。{hint}，请调大 max_tokens 后重试。"
            )

        if not content:
            raise LLMError(
                f"模型返回空内容（finish_reason={finish_reason or 'unknown'}，max_tokens={eff_max}）。"
                "已判定为失败而非有效答案，请检查提示词或调大 max_tokens。"
            )

        return content

    def chat_json(self, system: str, user: str, max_tokens: Optional[int] = None) -> Any:
        """要求模型返回 JSON，并做容错解析（真实结果不允许编造，解析失败即报错）。"""
        text = self.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.0,
            max_tokens=max_tokens,
            json_mode=True,
        )
        return _parse_json_loose(text)

    def connectivity_check(self) -> Dict[str, Any]:
        """真实连通性自检：发一次最小请求，返回真实耗时、回复与生效的 token 预算。"""
        t0 = time.perf_counter()
        text = self.chat(
            [{"role": "user", "content": "回复两个字：正常"}],
            temperature=0.0,
            max_tokens=32,
        )
        return {
            "ok": True,
            "model": self.model,
            "base_url": self.base_url,
            "is_reasoning_model": self.is_reasoning,
            "effective_max_tokens": self.last_effective_max_tokens,
            "finish_reason": self.usage.last_finish_reason,
            "latency_s": round(time.perf_counter() - t0, 3),
            "reply": text[:100],
        }


def _parse_json_loose(text: str) -> Any:
    """从模型输出中提取 JSON；失败则抛错，避免把无效内容当结果。"""
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fence:
        try:
            return json.loads(fence.group(1).strip())
        except Exception:
            pass
    for opener, closer in (("[", "]"), ("{", "}")):
        i, j = text.find(opener), text.rfind(closer)
        if i != -1 and j > i:
            try:
                return json.loads(text[i:j + 1])
            except Exception:
                continue
    raise LLMError(f"无法从模型输出中解析 JSON：{truncate_for_error(text)}")


def truncate_for_error(text: str, n: int = 300) -> str:
    return text if len(text) <= n else text[:n] + "…"


__all__ = ["LLMClient", "LLMError", "LLMUsage"]
