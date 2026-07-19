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
FEISHU_APP_ID = os.environ["FEISHU_APP_ID"]
FEISHU_APP_SECRET = os.environ["FEISHU_APP_SECRET"]
DEEPSEEK_API_KEY = os.environ["DEEPSEEK_API_KEY"]
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
    messages = build_messages(user_message, conversation_history)

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            DEEPSEEK_URL,
            headers={
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
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


# ── 飞书事件回调 ─────────────────────────────────
@app.post("/feishu/event")
async def feishu_event(request: Request):
    """接收飞书事件推送"""
    body = await request.json()
    logger.info(f"收到事件: {json.dumps(body, ensure_ascii=False)[:500]}")

    # URL 验证（首次配置时飞书会发 challenge）
    if body.get("type") == "url_verification":
        challenge = body.get("challenge", "")
        logger.info(f"URL 验证, challenge={challenge}")
        return JSONResponse({"challenge": challenge})

    # 消息事件
    if body.get("type") == "event_callback":
        event = body.get("event", {})
        event_type = event.get("type", "")

        # 收到私聊/群聊消息
        if event_type == "im.message.receive_v1":
            message = event.get("message", {})
            sender = event.get("sender", {})

            # 飞书事件可能用 open_id / user_id / union_id
            sender_id = sender.get("sender_id", {})
            open_id = sender_id.get("open_id", "") or sender_id.get("user_id", "")

            # 解析消息内容
            content_str = message.get("content", "{}")
            try:
                content = json.loads(content_str)
                text = content.get("text", "")
            except json.JSONDecodeError:
                text = content_str

            if not text or not open_id:
                return JSONResponse({"code": 0})

            # 忽略机器人自己的消息（防回环）
            if message.get("message_type") == "bot":
                return JSONResponse({"code": 0})

            logger.info(f"用户 {open_id}: {text}")

            # 构建对话历史
            history = conversations.get(open_id, [])

            # 调用 AI
            reply = await call_ai(text, history)

            # 保存对话
            history.append({"role": "user", "content": text})
            history.append({"role": "assistant", "content": reply})
            conversations[open_id] = history[-MAX_HISTORY:]

            # 回复
            await send_feishu_message(open_id, reply)
            logger.info(f"回复 {open_id}: {reply[:100]}...")

    return JSONResponse({"code": 0})


# ── 健康检查 ─────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": time.time()}


# ── 启动 ──────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
