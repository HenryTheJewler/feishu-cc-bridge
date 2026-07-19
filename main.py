"""
Feishu ↔ AI Bridge Service
飞书消息 → DeepSeek API → 飞书回复
"""

import json
import os
import time
import logging
from collections import defaultdict

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("feishu-ai-bridge")

app = FastAPI(title="Feishu-AI-Bridge")

# ── 飞书 API 常量 ────────────────────────────────
FEISHU_BASE = "https://open.feishu.cn/open-apis"
TENANT_TOKEN_URL = f"{FEISHU_BASE}/auth/v3/tenant_access_token/internal"
SEND_MSG_URL = f"{FEISHU_BASE}/im/v1/messages"
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

# ── 运行时配置（/setup 注入或环境变量）─────────────
_runtime: dict = {}

FEISHU_WEBHOOK = os.environ.get(
    "FEISHU_WEBHOOK",
    "https://open.feishu.cn/open-apis/bot/v2/hook/786188af-8586-4ebb-9417-f8ad89943a9a",
)

for _key in ["FEISHU_APP_ID", "FEISHU_APP_SECRET", "DEEPSEEK_API_KEY"]:
    _val = os.environ.get(_key, "")
    if _val:
        _runtime[_key] = _val


def _env(key: str) -> str:
    """优先读运行时配置，其次读环境变量"""
    return _runtime.get(key) or os.environ.get(key, "")


# ── 令牌缓存 ──────────────────────────────────────
_token_cache: dict = {"token": None, "expires_at": 0}


async def get_tenant_token() -> str:
    """获取飞书 tenant_access_token（带缓存）"""
    now = time.time()
    if _token_cache["token"] and now < _token_cache["expires_at"] - 300:
        return _token_cache["token"]

    app_id = _env("FEISHU_APP_ID")
    app_secret = _env("FEISHU_APP_SECRET")
    if not app_id or not app_secret:
        raise RuntimeError("飞书 App ID/Secret 未配置，请调用 /setup")

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            TENANT_TOKEN_URL,
            json={"app_id": app_id, "app_secret": app_secret},
        )
        data = resp.json()
        if data.get("code") != 0:
            raise RuntimeError(f"获取 token 失败: {data}")

        _token_cache["token"] = data["tenant_access_token"]
        _token_cache["expires_at"] = now + data.get("expire", 7200)
        return _token_cache["token"]


# ── 系统提示词（品牌上下文）────────────────────────
SYSTEM_PROMPT = """你是「旺德兰 Wonderland」品牌主理人的 AI 助手。

## 品牌信息
- 品类：手工黄金饰品（錾刻、花丝镶嵌）
- 品牌名来源：奶奶王德兰的名字
- 定位：年轻化的手工黄金，「简陋且廉价版琳朝」，传统工艺 + 年轻审美
- 客单价：平均 27,000-28,000 RMB
- 客户：25-45岁女性，决策极快（30分钟-1小时成交）
- 当前状态：二次冷启动，月销~1件，目标月销1kg黄金（盈亏平衡线）
- 账号：小红书「北美花手第一人」（~2.4w粉）

## 内容策略
- 核心原则：每篇笔记 = 黄金 + 主理人（换个人发也成立就重写）
- 三个方向：宝二代叛逃 / 工艺里的我 / 北美视角下的中国黄金
- 节奏：5篇产品图文 + 1篇人设深度 + 2条视频/周

## 竞品
元古工坊、點石成金、屿澜细金、KINFAR砺金坊

## 主理人
宝二代（父母开金店），美国近十年（UCD/BU），FAA飞行学员，华州船员
前职业：头部博主IP孵化（策划+拍摄+销售转化）
优势：文化混血视角、职业级视频制作、真实敢说
限制：一个人扛所有事、资金不雄厚、入局晚

## 说话风格
- 直接、坦诚、不矫情
- 可以反问和挑战
- 用主理人的语气，不要 AI 腔
- 简短精炼，飞书消息不适合长篇大论"""


def build_messages(user_message: str, history: list = None) -> list:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})
    return messages


async def call_ai(user_message: str, conversation_history: list = None) -> str:
    api_key = _env("DEEPSEEK_API_KEY").strip()
    if not api_key or not api_key.startswith("sk-"):
        return "❌ DeepSeek API Key 未配置。请在飞书开发者后台调用 /setup 接口配置。"

    messages = build_messages(user_message, conversation_history)
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            DEEPSEEK_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": DEEPSEEK_MODEL,
                "messages": messages,
                "max_tokens": 1024,
                "temperature": 0.7,
            },
        )
        if resp.status_code != 200:
            logger.error(f"DeepSeek API error ({resp.status_code}): {resp.text}")
            return "抱歉，AI 暂时无法回复，稍后再试。"

        data = resp.json()
        return data["choices"][0]["message"]["content"]


async def send_feishu_message(open_id: str, text: str) -> None:
    token = await get_tenant_token()
    content = json.dumps({"text": text})
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            SEND_MSG_URL,
            params={"receive_id_type": "open_id"},
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json={
                "receive_id": open_id,
                "msg_type": "text",
                "content": content,
            },
        )
        data = resp.json()
        if data.get("code") != 0:
            logger.error(f"发送消息失败: {data}")


# ── 会话存储 ──────────────────────────────────────
conversations: dict = defaultdict(list)
MAX_HISTORY = 10
recent_logs: list = []


# ── 飞书事件回调 ─────────────────────────────────
@app.post("/feishu/event")
async def feishu_event(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"code": 0})

    header = body.get("header", {})
    event_type = (
        body.get("type", "")
        or header.get("event_type", "")
        or body.get("event", {}).get("type", "")
    )
    log_entry = {"time": time.time(), "event_type": event_type or "unknown"}
    logger.info(f"收到事件 [{event_type}]: {json.dumps(body, ensure_ascii=False)[:300]}")

    # URL 验证
    challenge = body.get("challenge", "")
    if challenge or event_type == "url_verification":
        log_entry["msg"] = f"ok: {challenge}"
        recent_logs.append(log_entry)
        recent_logs[:] = recent_logs[-20:]
        return JSONResponse({"challenge": challenge})

    # 消息事件
    if event_type == "im.message.receive_v1":
        event = body.get("event", {})
        message = event.get("message", {})
        sender = event.get("sender", {})
        sender_id = sender.get("sender_id", {})
        open_id = sender_id.get("open_id", "") or sender_id.get("user_id", "")

        if message.get("message_type") == "bot" or message.get("chat_type") == "bot":
            log_entry["msg"] = "ignored: bot"
            recent_logs.append(log_entry)
            recent_logs[:] = recent_logs[-20:]
            return JSONResponse({"code": 0})

        content_str = message.get("content", "{}")
        try:
            content = json.loads(content_str)
            text = content.get("text", "")
        except json.JSONDecodeError:
            text = content_str

        if not text or not open_id:
            log_entry["msg"] = "ignored: no content"
            recent_logs.append(log_entry)
            recent_logs[:] = recent_logs[-20:]
            return JSONResponse({"code": 0})

        log_entry["open_id"] = open_id[-8:]
        log_entry["text"] = text[:100]
        logger.info(f"用户 {open_id[-8:]}: {text}")

        try:
            history = conversations.get(open_id, [])
            reply = await call_ai(text, history)
        except Exception as e:
            log_entry["msg"] = f"AI error: {e}"
            recent_logs.append(log_entry)
            recent_logs[:] = recent_logs[-20:]
            return JSONResponse({"code": 0})

        history.append({"role": "user", "content": text})
        history.append({"role": "assistant", "content": reply})
        conversations[open_id] = history[-MAX_HISTORY:]

        try:
            await send_feishu_message(open_id, reply)
            log_entry["msg"] = f"ok: {reply[:80]}"
        except Exception as e:
            log_entry["msg"] = f"send error: {e}"

        recent_logs.append(log_entry)
        recent_logs[:] = recent_logs[-20:]
    else:
        log_entry["msg"] = f"unhandled: {event_type}"
        recent_logs.append(log_entry)
        recent_logs[:] = recent_logs[-20:]

    return JSONResponse({"code": 0})


# ── 调试端点 ─────────────────────────────────────
@app.get("/debug")
async def debug():
    return {"recent_logs": recent_logs, "conversations": len(conversations)}


@app.get("/env")
async def list_env():
    all_vars = sorted(os.environ.keys()) + sorted(_runtime.keys())
    return {"all_vars": all_vars, "total_os": len(os.environ), "total_runtime": len(_runtime)}


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": time.time()}


# ── 运行时配置 ────────────────────────────────────
@app.post("/setup")
async def setup(request: Request):
    """注入配置：FEISHU_APP_ID, FEISHU_APP_SECRET, DEEPSEEK_API_KEY"""
    body = await request.json()
    keys_set = []
    for key in ["FEISHU_APP_ID", "FEISHU_APP_SECRET", "DEEPSEEK_API_KEY"]:
        if body.get(key):
            _runtime[key] = body[key]
            keys_set.append(key)

    return {
        "ok": True,
        "keys_set": keys_set,
        "ds_key_len": len(_env("DEEPSEEK_API_KEY").strip()),
        "app_id_set": bool(_env("FEISHU_APP_ID")),
    }


# ── 提醒推送 ────────────────────────────────────
@app.get("/remind")
async def remind(msg: str = "中午了，来找Claude。今天把周计划排了。"):
    """通过 webhook 发送飞书提醒"""
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            FEISHU_WEBHOOK,
            json={"msg_type": "text", "content": {"text": msg}},
        )
        return {"ok": resp.json().get("code") == 0}


# ── 启动 ──────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
