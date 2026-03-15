"""
Meeting Copilot - 議事録・メール自動化アプリ
FastAPI バックエンド
"""

import os
import logging
from typing import Optional, List
import requests as http_requests

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from openai import OpenAI

# ロギング設定（本文はログに含めない）
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Meeting Copilot", version="1.0.0")

# OpenAI クライアント
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

MAX_CHARS = 50000


# ========== Pydantic モデル ==========

class GenerateRequest(BaseModel):
    inputType: str  # "url" | "text"
    url: Optional[str] = None
    transcriptText: Optional[str] = None
    meetingTitle: Optional[str] = None
    meetingDateTime: Optional[str] = None
    participants: Optional[str] = None
    mailPurpose: Optional[str] = "report"
    instructions: Optional[str] = None


class RegenerateRequest(BaseModel):
    tab: str  # "summary" | "minutes" | "mail" | "todos" | "decisions"
    transcriptText: str
    meetingTitle: Optional[str] = None
    meetingDateTime: Optional[str] = None
    participants: Optional[str] = None
    mailPurpose: Optional[str] = "report"
    globalInstructions: Optional[str] = None
    tabInstructions: Optional[str] = None


class GenerateResponse(BaseModel):
    summary: str
    minutes: str
    mailDraft: str
    todos: List[dict]
    decisions: List[str]
    openQuestions: List[str]


# ========== URL取得 ==========

def fetch_transcript_from_url(url: str) -> str:
    """URLから文字起こし本文を取得する（認証不要のURLのみ）"""
    try:
        resp = http_requests.get(url, timeout=15, headers={"User-Agent": "MeetingCopilot/1.0"})
        resp.raise_for_status()
        # HTMLの場合、テキスト抽出を試みる
        content_type = resp.headers.get("content-type", "")
        if "html" in content_type:
            # 簡易テキスト抽出
            import re
            text = re.sub(r'<[^>]+>', ' ', resp.text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text[:MAX_CHARS]
        else:
            return resp.text[:MAX_CHARS]
    except Exception as e:
        logger.error(f"URL fetch error: {type(e).__name__}")
        raise HTTPException(status_code=422, detail=f"URLの取得に失敗しました。本文を直接貼り付けてください。({type(e).__name__})")


# ========== AIプロンプト ==========

def build_system_prompt() -> str:
    return """あなたは会議内容の整理者・議事録作成者・ビジネスメール作成者・TODO抽出者です。
以下の方針に従ってください：
- 冗長な会話を整理し、発言内容を要点化する
- 推測しすぎない
- 不明点は「不明」と明示する
- 担当者や期限は本文に根拠がある場合のみ出力する
- 日本語で出力する"""


def build_generate_prompt(transcript: str, meta: dict, instructions: str = "") -> str:
    title = meta.get("meetingTitle") or "（会議名不明）"
    dt = meta.get("meetingDateTime") or "（日時不明）"
    participants = meta.get("participants") or "（出席者不明）"
    mail_purpose = meta.get("mailPurpose", "report")
    mail_purpose_map = {"report": "会議報告メール", "share": "議事録共有メール", "request": "依頼メール"}
    mail_label = mail_purpose_map.get(mail_purpose, "会議報告メール")

    instruction_block = f"\n\n【追加指示】\n{instructions}" if instructions else ""

    return f"""以下の会議文字起こしを分析し、指定の形式で出力してください。

【会議情報】
- 会議名: {title}
- 日時: {dt}
- 出席者: {participants}
- メール種別: {mail_label}
{instruction_block}

【文字起こし本文】
{transcript}

---
以下の形式でJSONを出力してください（コードブロック不要、純粋なJSONのみ）:

{{
  "summary": "会議全体の要約（主要トピック・決定事項・次のアクションを含む）",
  "minutes": "議事録（Markdown形式）",
  "mailDraft": "{mail_label}の文面（Markdown形式）",
  "todos": [
    {{"task": "タスク内容", "owner": "担当者（不明な場合は空文字）", "due": "期限（不明な場合は空文字）", "priority": ""}}
  ],
  "decisions": ["決定事項1", "決定事項2"],
  "openQuestions": ["未解決事項1", "未解決事項2"]
}}"""


def build_minutes_template(title: str, dt: str, participants: str) -> str:
    return f"""# 議事録

## 会議名
{title}

## 日時
{dt}

## 出席者
{participants}

## 目的
（AIが文字起こしから判断）

## 議題
（AIが文字起こしから抽出）

## 内容要約
（AIが文字起こしから生成）

## 決定事項
（AIが文字起こしから抽出）

## TODO / 宿題
（AIが文字起こしから抽出）

## 未解決事項
（AIが文字起こしから抽出）

## 次回予定
（AIが文字起こしから抽出）"""


def build_regen_prompt(tab: str, transcript: str, meta: dict, global_instructions: str = "", tab_instructions: str = "") -> str:
    tab_map = {
        "summary": "会議要約のみ",
        "minutes": "議事録（Markdown形式）のみ",
        "mail": f"メール文（{meta.get('mailPurpose','report')}種別）のみ",
        "todos": "TODO一覧のみ（JSON配列）",
        "decisions": "決定事項一覧のみ（JSON配列）",
    }
    target = tab_map.get(tab, tab)
    instructions_block = ""
    if global_instructions:
        instructions_block += f"\n【全体指示】\n{global_instructions}"
    if tab_instructions:
        instructions_block += f"\n【このタブへの追加指示】\n{tab_instructions}"

    title = meta.get("meetingTitle") or "（会議名不明）"
    dt = meta.get("meetingDateTime") or "（日時不明）"
    participants = meta.get("participants") or "（出席者不明）"

    return f"""以下の文字起こしをもとに、{target}を生成してください。

【会議情報】
- 会議名: {title}
- 日時: {dt}
- 出席者: {participants}
{instructions_block}

【文字起こし本文】
{transcript}"""


# ========== AI呼び出し ==========

def call_openai(messages: list, expect_json: bool = True) -> str:
    try:
        kwargs = {
            "model": OPENAI_MODEL,
            "messages": messages,
            "max_tokens": 4096,
            "temperature": 0.3,
        }
        if expect_json:
            kwargs["response_format"] = {"type": "json_object"}
        response = openai_client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""
    except Exception as e:
        logger.error(f"OpenAI API error: {type(e).__name__}")
        raise HTTPException(status_code=500, detail=f"AI処理に失敗しました。しばらくしてから再試行してください。({type(e).__name__})")


# ========== エンドポイント ==========

@app.post("/api/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    # 文字起こし取得
    if req.inputType == "url":
        if not req.url:
            raise HTTPException(status_code=400, detail="URLを入力してください。")
        transcript = fetch_transcript_from_url(req.url)
    else:
        if not req.transcriptText:
            raise HTTPException(status_code=400, detail="文字起こし本文を入力してください。")
        transcript = req.transcriptText

    if len(transcript) < 20:
        raise HTTPException(status_code=400, detail="文字起こしの内容が短すぎます。20文字以上入力してください。")
    if len(transcript) > MAX_CHARS:
        raise HTTPException(status_code=400, detail="入力文字数が上限を超えています。50,000文字以下にしてください。")

    meta = {
        "meetingTitle": req.meetingTitle,
        "meetingDateTime": req.meetingDateTime,
        "participants": req.participants,
        "mailPurpose": req.mailPurpose,
    }

    prompt = build_generate_prompt(transcript, meta, req.instructions or "")
    messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": prompt},
    ]

    raw = call_openai(messages, expect_json=True)

    import json
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.error("JSON parse error from OpenAI response")
        raise HTTPException(status_code=500, detail="AI出力の解析に失敗しました。再試行してください。")

    return GenerateResponse(
        summary=data.get("summary", ""),
        minutes=data.get("minutes", ""),
        mailDraft=data.get("mailDraft", ""),
        todos=data.get("todos", []),
        decisions=data.get("decisions", []),
        openQuestions=data.get("openQuestions", []),
    )


@app.post("/api/regenerate")
async def regenerate(req: RegenerateRequest):
    if len(req.transcriptText) < 20:
        raise HTTPException(status_code=400, detail="文字起こし本文が短すぎます。")
    if len(req.transcriptText) > MAX_CHARS:
        raise HTTPException(status_code=400, detail="入力文字数が上限を超えています。")

    meta = {
        "meetingTitle": req.meetingTitle,
        "meetingDateTime": req.meetingDateTime,
        "participants": req.participants,
        "mailPurpose": req.mailPurpose,
    }

    prompt = build_regen_prompt(
        req.tab, req.transcriptText, meta,
        req.globalInstructions or "", req.tabInstructions or ""
    )
    messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": prompt},
    ]

    # タブ種別によって期待フォーマットを変える
    json_tabs = {"todos", "decisions"}
    expect_json = req.tab in json_tabs
    raw = call_openai(messages, expect_json=expect_json)

    return {"tab": req.tab, "content": raw}


@app.get("/health")
async def health():
    return {"status": "ok", "model": OPENAI_MODEL}


# 静的ファイル配信
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")


@app.get("/")
async def index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))
