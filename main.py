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

# ── 配置 ──────────────────────────────────────────
def _require(key: str) -> str:
    val = os.environ.get(key, "")
    if not val:
        logger.warning(f"环境变量 {key} 未设置")
    return val

FEISHU_APP_ID = _require("FEISHU_APP_ID")
FEISHU_APP_SECRET = _require("FEISHU_APP_SECRET")
DEEPSEEK_API_KEY = _require("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

# 飞书 API
FEISHU_BASE = "https://open.feishu.cn/open-apis"
TENANT_TOKEN_URL = f"{FEISHU_BASE}/auth/v3/tenant_access_token/internal"
SEND_MSG_URL = f"{FEISHU_BASE}/im/v1/messages"

# DeepSeek API (OpenAI 兼容)
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

# ── 令牌缓存 ──────────────────────────────────────
_token_cache: dict = {"token": None, "expires_at": 0}


async def get_tenant_token() -> str:
    """获取飞书 tenant_access_token（带缓存）"""
    now = time.time()
    if _token_cache["token"] and now < _token_cache["expires_at"] - 300:
        return _token_cache["token"]

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            TENANT_TOKEN_URL,
            json={"app_id": FEISHU_APP_ID, "app_secret": FEISHU_APP_SECRET},
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
    """构建 OpenAI 格式消息列表"""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})
    return messages


async def call_ai(user_message: str, conversation_history: list = None) -> str:
    """调用 DeepSeek API"""
    api_key = DEEPSEEK_API_KEY.strip()
    if not api_key or not api_key.startswith("sk-"):
        return "❌ DeepSeek API Key 未配置或格式错误。请在 Railway Variables 中设置 DEEPSEEK_API_KEY。"

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
    """通过飞书机器人回复消息"""
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


# ── 会话存储（内存，简易版）───────────────────────
conversations: dict = defaultdict(list)  # open_id → [{role, content}]
MAX_HISTORY = 10  # 保留最近 10 轮对话
recent_logs: list = []  # 最近 20 条事件日志


# ── 飞书事件回调 ─────────────────────────────────
@app.post("/feishu/event")
async def feishu_event(request: Request):
    """接收飞书事件推送"""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"code": 0})

    # 记录所有事件
    header = body.get("header", {})
    event_type = (
        body.get("type", "") or
        header.get("event_type", "") or
        body.get("event", {}).get("type", "")
    )
    log_entry = {"time": time.time(), "event_type": event_type or "unknown"}
    logger.info(f"收到事件 [{event_type}]: {json.dumps(body, ensure_ascii=False)[:300]}")

    # URL 验证（v1 格式 & v2 格式都支持）
    challenge = body.get("challenge", "")
    if challenge or event_type == "url_verification":
        log_entry["msg"] = f"url_verification: {challenge}"
        recent_logs.append(log_entry)
        recent_logs[:] = recent_logs[-20:]
        return JSONResponse({"challenge": challenge})

    # 消息事件（v1: event.type, v2: header.event_type）
    if event_type == "im.message.receive_v1":
        event = body.get("event", {})
        message = event.get("message", {})
        sender = event.get("sender", {})
        sender_id = sender.get("sender_id", {})
        open_id = sender_id.get("open_id", "") or sender_id.get("user_id", "")

        # 忽略机器人自己的消息
        if message.get("message_type") == "bot" or message.get("chat_type") == "bot":
            log_entry["msg"] = "ignored: bot message"
            recent_logs.append(log_entry)
            recent_logs[:] = recent_logs[-20:]
            return JSONResponse({"code": 0})

        # 解析消息
        content_str = message.get("content", "{}")
        try:
            content = json.loads(content_str)
            text = content.get("text", "")
        except json.JSONDecodeError:
            text = content_str

        if not text or not open_id:
            log_entry["msg"] = "ignored: no text or open_id"
            recent_logs.append(log_entry)
            recent_logs[:] = recent_logs[-20:]
            return JSONResponse({"code": 0})

        log_entry["open_id"] = open_id[-8:]
        log_entry["text"] = text[:100]
        logger.info(f"用户 {open_id[-8:]}: {text}")

        # 调 AI
        try:
            history = conversations.get(open_id, [])
            reply = await call_ai(text, history)
        except Exception as e:
            log_entry["msg"] = f"AI error: {e}"
            recent_logs.append(log_entry)
            recent_logs[:] = recent_logs[-20:]
            return JSONResponse({"code": 0})

        # 保存对话
        history.append({"role": "user", "content": text})
        history.append({"role": "assistant", "content": reply})
        conversations[open_id] = history[-MAX_HISTORY:]

        # 回复
        try:
            await send_feishu_message(open_id, reply)
            log_entry["msg"] = f"replied: {reply[:80]}"
        except Exception as e:
            log_entry["msg"] = f"send error: {e}"

        recent_logs.append(log_entry)
        recent_logs[:] = recent_logs[-20:]
    else:
        # 记录其他所有事件，用于调试
        log_entry["msg"] = f"unhandled event: {event_type}"
        recent_logs.append(log_entry)
        recent_logs[:] = recent_logs[-20:]

    return JSONResponse({"code": 0})


# ── 调试端点 ─────────────────────────────────────
@app.get("/debug")
async def debug():
    return {"recent_logs": recent_logs, "conversations": len(conversations)}


@app.get("/test-feishu")
async def test_feishu():
    """诊断飞书连接"""
    results = {}

    # 1. 检查环境变量
    results["app_id_set"] = bool(FEISHU_APP_ID)
    results["app_secret_set"] = bool(FEISHU_APP_SECRET)
    results["deepseek_key_set"] = bool(DEEPSEEK_API_KEY)

    # 2. 测试获取 tenant token
    try:
        token = await get_tenant_token()
        results["tenant_token"] = "OK"
    except Exception as e:
        results["tenant_token"] = f"FAILED: {e}"
        return results

    # 3. 测试获取 bot 信息
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{FEISHU_BASE}/bot/v3/info",
            headers={"Authorization": f"Bearer {token}"},
        )
        bot_data = resp.json()
        if bot_data.get("code") == 0:
            bot = bot_data.get("bot", {})
            results["bot_info"] = {
                "name": bot.get("app_name", "unknown"),
                "active": bot.get("activate_status", "unknown"),
            }
        else:
            results["bot_info"] = f"ERROR: {bot_data}"

    return results


@app.get("/test-ds")
async def test_deepseek():
    """诊断 DeepSeek API Key"""
    api_key = DEEPSEEK_API_KEY.strip()
    return {
        "key_length": len(api_key),
        "starts_with_sk": api_key.startswith("sk-"),
        "first_8_chars": api_key[:8] if len(api_key) >= 8 else api_key,
        "last_4_chars": api_key[-4:] if len(api_key) >= 4 else "N/A",
    }


@app.get("/env")
async def list_env():
    """列出所有环境变量名（不暴露值）"""
    all_vars = sorted(os.environ.keys())
    return {
        "all_vars": all_vars,
        "total": len(all_vars),
    }


# ── 健康检查 ─────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": time.time()}


# ── 启动 ──────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
