import json
import os
import re
import sqlite3
from datetime import date, datetime
from io import BytesIO
from typing import Any, Dict, List

import google.generativeai as genai
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

DB_PATH = "transactions.db"
CATEGORIES = ["餐飲", "交通", "網購", "休閒"]

st.set_page_config(page_title="Gemini 帳單記帳 App", page_icon="🧾", layout="wide")


def get_api_key() -> str:
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        key = ""
    return key or os.getenv("GEMINI_API_KEY", "")


def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                consume_date TEXT NOT NULL,
                amount REAL NOT NULL,
                description TEXT NOT NULL,
                category TEXT NOT NULL,
                source_file TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def load_transactions() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query("SELECT * FROM transactions ORDER BY consume_date DESC, id DESC", conn)
    if df.empty:
        return pd.DataFrame(columns=["id", "consume_date", "amount", "description", "category", "source_file", "created_at"])
    df["consume_date"] = pd.to_datetime(df["consume_date"], errors="coerce").dt.date
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
    return df


def insert_transactions(items: List[Dict[str, Any]], source_file: str) -> int:
    now = datetime.now().isoformat(timespec="seconds")
    count = 0
    with sqlite3.connect(DB_PATH) as conn:
        for item in items:
            consume_date = normalize_date(item.get("date"))
            amount = normalize_amount(item.get("amount"))
            description = str(item.get("description") or item.get("merchant") or "未命名消費").strip()
            category = normalize_category(item.get("category"), description)
            if consume_date and amount > 0:
                conn.execute(
                    """
                    INSERT INTO transactions
                    (consume_date, amount, description, category, source_file, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (consume_date, amount, description, category, source_file, now),
                )
                count += 1
        conn.commit()
    return count


def update_transaction(row_id: int, consume_date: date, amount: float, description: str, category: str) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            UPDATE transactions
            SET consume_date = ?, amount = ?, description = ?, category = ?
            WHERE id = ?
            """,
            (consume_date.isoformat(), float(amount), description.strip(), category, int(row_id)),
        )
        conn.commit()


def delete_transaction(row_id: int) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM transactions WHERE id = ?", (int(row_id),))
        conn.commit()


def delete_all_transactions() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM transactions")
        conn.commit()


def normalize_date(value: Any) -> str | None:
    if not value:
        return None
    text = str(value).strip().replace("/", "-").replace(".", "-")
    text = re.sub(r"[年月]", "-", text).replace("日", "")
    try:
        return pd.to_datetime(text, errors="raise").date().isoformat()
    except Exception:
        return None


def normalize_amount(value: Any) -> float:
    if value is None:
        return 0.0
    text = str(value)
    text = re.sub(r"[^0-9.]", "", text)
    try:
        return float(text)
    except Exception:
        return 0.0


def normalize_category(value: Any, description: str = "") -> str:
    category = str(value or "").strip()
    if category in CATEGORIES:
        return category
    text = f"{category} {description}"
    food_words = ["餐", "飲", "咖啡", "早餐", "午餐", "晚餐", "便當", "超商", "全聯", "家樂福", "麥當勞", "星巴克"]
    traffic_words = ["捷運", "高鐵", "台鐵", "公車", "計程車", "uber", "停車", "加油", "交通"]
    shop_words = ["蝦皮", "momo", "pchome", "amazon", "淘寶", "網購", "電商"]
    leisure_words = ["電影", "遊戲", "娛樂", "健身", "旅行", "旅遊", "休閒", "展覽"]
    if any(w.lower() in text.lower() for w in traffic_words):
        return "交通"
    if any(w.lower() in text.lower() for w in shop_words):
        return "網購"
    if any(w.lower() in text.lower() for w in leisure_words):
        return "休閒"
    if any(w.lower() in text.lower() for w in food_words):
        return "餐飲"
    return "休閒"


def extract_json(text: str) -> List[Dict[str, Any]]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.I).strip()
    text = re.sub(r"```$", "", text).strip()
    candidates = [text]
    array_match = re.search(r"\[.*\]", text, flags=re.S)
    if array_match:
        candidates.append(array_match.group(0))
    object_match = re.search(r"\{.*\}", text, flags=re.S)
    if object_match:
        candidates.append(object_match.group(0))
    for candidate in candidates:
        try:
            data = json.loads(candidate)
            if isinstance(data, dict) and "transactions" in data:
                data = data["transactions"]
            if isinstance(data, dict):
                data = [data]
            if isinstance(data, list):
                return [x for x in data if isinstance(x, dict)]
        except Exception:
            continue
    raise ValueError("Gemini 回傳內容不是可解析的 JSON。")


def analyze_image(uploaded_file) -> List[Dict[str, Any]]:
    api_key = get_api_key()
    if not api_key:
        st.error("尚未設定 GEMINI_API_KEY。請在 Streamlit Secrets 或環境變數加入 GEMINI_API_KEY。")
        return []

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    image = Image.open(BytesIO(uploaded_file.getvalue())).convert("RGB")
    prompt = f"""
你是一個台灣使用者的帳單/收據/信用卡明細 OCR 記帳助手。
請從圖片中擷取所有可辨識的消費交易。

規則：
1. 日期請以消費日為準，輸出 YYYY-MM-DD。
2. 金額只輸出數字，不要逗號、幣別或負號。
3. 消費內容請簡短描述，例如店名、平台或品項。
4. category 只能是以下四種之一：餐飲、交通、網購、休閒。
5. 如果同一張圖只有總金額，就輸出一筆。
6. 如果看不出年份，請優先使用今年 {date.today().year} 年。
7. 不要輸出 Markdown，不要解釋，只輸出 JSON array。

輸出格式：
[
  {{"date":"YYYY-MM-DD","amount":123,"description":"消費內容","category":"餐飲"}}
]
"""
    response = model.generate_content([prompt, image])
    raw_text = response.text or ""
    return extract_json(raw_text)


init_db()

st.title("🧾 Gemini 帳單記帳 App")
st.caption("上傳帳單、收據或信用卡截圖，自動辨識並依月份統整消費。")

with st.sidebar:
    st.header("設定")
    api_key_status = "已設定" if get_api_key() else "未設定"
    st.write(f"Gemini API Key：**{api_key_status}**")
    st.info("部署到 Streamlit Cloud 時，請在 Secrets 加入：\n\nGEMINI_API_KEY = \"你的金鑰\"")

uploaded_files = st.file_uploader(
    "上傳帳單圖片，可一次選多張",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True,
)

if uploaded_files:
    if st.button("開始辨識並新增交易", type="primary"):
        total = 0
        with st.spinner("正在用 Gemini 辨識圖片..."):
            for uploaded_file in uploaded_files:
                try:
                    items = analyze_image(uploaded_file)
                    added = insert_transactions(items, uploaded_file.name)
                    total += added
                    st.success(f"{uploaded_file.name}：新增 {added} 筆")
                except Exception as exc:
                    st.error(f"{uploaded_file.name} 辨識失敗：{exc}")
        st.toast(f"完成，新增 {total} 筆交易")
        st.rerun()

st.divider()

df = load_transactions()

if df.empty:
    st.warning("目前沒有交易資料。請先上傳帳單圖片。")
    st.stop()

df["month"] = pd.to_datetime(df["consume_date"]).dt.strftime("%Y-%m")
months = sorted(df["month"].dropna().unique(), reverse=True)
selected_month = st.selectbox("選擇月份", months)
month_df = df[df["month"] == selected_month].copy()

col1, col2, col3 = st.columns(3)
col1.metric("本月筆數", len(month_df))
col2.metric("本月總金額", f"NT$ {month_df['amount'].sum():,.0f}")
col3.metric("平均單筆", f"NT$ {month_df['amount'].mean():,.0f}" if len(month_df) else "NT$ 0")

chart_df = month_df.groupby("category", as_index=False)["amount"].sum()
chart_df["percent"] = chart_df["amount"] / chart_df["amount"].sum() * 100

left, right = st.columns([1, 1])
with left:
    st.subheader("分類百分比圓餅圖")
    fig = px.pie(chart_df, names="category", values="amount", hole=0.25)
    fig.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("分類統計")
    show_chart_df = chart_df.copy()
    show_chart_df["amount"] = show_chart_df["amount"].map(lambda x: f"NT$ {x:,.0f}")
    show_chart_df["percent"] = show_chart_df["percent"].map(lambda x: f"{x:.1f}%")
    st.dataframe(show_chart_df.rename(columns={"category": "分類", "amount": "金額", "percent": "百分比"}), use_container_width=True)

st.subheader("本月明細，可單筆編輯或刪除")

for _, row in month_df.sort_values("consume_date", ascending=False).iterrows():
    with st.expander(f"{row['consume_date']}｜{row['category']}｜NT$ {row['amount']:,.0f}｜{row['description']}"):
        with st.form(f"edit_{row['id']}"):
            c1, c2, c3, c4 = st.columns([1, 1, 2, 1])
            new_date = c1.date_input("日期", value=row["consume_date"], key=f"date_{row['id']}")
            new_amount = c2.number_input("金額", min_value=0.0, value=float(row["amount"]), step=1.0, key=f"amount_{row['id']}")
            new_description = c3.text_input("消費內容", value=row["description"], key=f"desc_{row['id']}")
            new_category = c4.selectbox("分類", CATEGORIES, index=CATEGORIES.index(row["category"]) if row["category"] in CATEGORIES else 0, key=f"cat_{row['id']}")
            save = st.form_submit_button("儲存修改")
            if save:
                update_transaction(row["id"], new_date, new_amount, new_description, new_category)
                st.success("已儲存")
                st.rerun()
        if st.button("刪除此筆", key=f"delete_{row['id']}"):
            delete_transaction(row["id"])
            st.warning("已刪除")
            st.rerun()

st.divider()
with st.expander("危險操作：一鍵刪除全部資料"):
    confirm = st.checkbox("我確定要刪除全部交易資料，且無法復原")
    if st.button("刪除全部資料", disabled=not confirm, type="secondary"):
        delete_all_transactions()
        st.error("已刪除全部資料")
        st.rerun()
