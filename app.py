import base64
import io
import json
import re
import sqlite3
from datetime import date, datetime
from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image, ImageOps, ImageEnhance
import google.generativeai as genai

DB_PATH = "database.db"
CATEGORIES = ["餐飲", "交通", "網購", "休閒"]

st.set_page_config(page_title="AI 帳單記帳助手", page_icon="🧾", layout="wide")


def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            spend_date TEXT NOT NULL,
            amount REAL NOT NULL,
            description TEXT NOT NULL,
            category TEXT NOT NULL,
            merchant TEXT,
            source_file TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def load_transactions() -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql_query("SELECT * FROM transactions ORDER BY spend_date DESC, id DESC", conn)
    conn.close()
    return df


def insert_transactions(rows: List[Dict[str, Any]], source_file: str):
    conn = get_conn()
    cur = conn.cursor()
    now = datetime.now().isoformat(timespec="seconds")
    for r in rows:
        cur.execute(
            """
            INSERT INTO transactions
            (spend_date, amount, description, category, merchant, source_file, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                r.get("spend_date") or date.today().isoformat(),
                float(r.get("amount") or 0),
                r.get("description") or "未命名消費",
                r.get("category") if r.get("category") in CATEGORIES else "休閒",
                r.get("merchant") or "",
                source_file,
                now,
            ),
        )
    conn.commit()
    conn.close()


def update_transaction(row_id: int, spend_date: str, amount: float, description: str, category: str, merchant: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE transactions
        SET spend_date=?, amount=?, description=?, category=?, merchant=?
        WHERE id=?
        """,
        (spend_date, amount, description, category, merchant, row_id),
    )
    conn.commit()
    conn.close()


def delete_transaction(row_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM transactions WHERE id=?", (row_id,))
    conn.commit()
    conn.close()


def delete_all_transactions():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM transactions")
    conn.commit()
    conn.close()


def normalize_json_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.I).strip()
    text = re.sub(r"```$", "", text).strip()
    match = re.search(r"(\[.*\]|\{.*\})", text, re.S)
    return match.group(1) if match else text


def preprocess_image(uploaded_file) -> Image.Image:
    image = Image.open(uploaded_file).convert("RGB")
    image = ImageOps.exif_transpose(image)
    max_side = 1800
    image.thumbnail((max_side, max_side))
    # 輕微提高對比，避免過度處理讓模型誤判
    image = ImageEnhance.Contrast(image).enhance(1.15)
    image = ImageEnhance.Sharpness(image).enhance(1.1)
    return image


def get_api_key() -> str:
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return ""


def analyze_bill(image: Image.Image) -> List[Dict[str, Any]]:
    api_key = get_api_key()
    if not api_key:
        st.error("尚未設定 GEMINI_API_KEY。請到 Streamlit Cloud 的 App settings → Secrets 加入 GEMINI_API_KEY。")
        return []

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    today = date.today().isoformat()
    prompt = f"""
你是台灣帳單、發票、收據、信用卡明細、外送訂單、電商訂單的資料辨識助手。
請仔細閱讀圖片，抽出所有可以確定的消費交易。

請務必遵守：
1. 日期以「實際消費日」為準，不要用帳單繳款日、列印日、結帳日，除非沒有其他日期。
2. 金額請使用實際消費金額，忽略發票號碼、統編、信用卡末四碼、電話、會員編號。
3. 若圖片有多筆交易，請輸出多筆。
4. 分類只能是：餐飲、交通、網購、休閒。
5. 若日期年份不明，請依照台灣常見收據推測為今年；今天是 {today}。
6. 不確定的欄位請合理填寫，但不要憑空新增圖片上不存在的交易。

分類規則：
- 餐飲：餐廳、咖啡、飲料、外送、超商食品、便當、早餐、夜市食物
- 交通：捷運、公車、高鐵、台鐵、計程車、Uber、停車、加油、共享機車
- 網購：蝦皮、momo、PChome、Amazon、電商訂單、線上購物
- 休閒：電影、遊戲、訂閱、娛樂、旅遊、百貨、服飾、日用品、其他不屬於前三類

只輸出 JSON array，不要輸出解釋文字。格式如下：
[
  {{
    "spend_date": "YYYY-MM-DD",
    "amount": 123,
    "description": "消費內容/品項摘要",
    "category": "餐飲",
    "merchant": "商家名稱"
  }}
]
"""
    response = model.generate_content([prompt, image], generation_config={"temperature": 0.1})
    raw = response.text or ""
    try:
        data = json.loads(normalize_json_text(raw))
        if isinstance(data, dict):
            data = data.get("transactions", [data])
        cleaned = []
        for item in data:
            if not isinstance(item, dict):
                continue
            amount = re.sub(r"[^0-9.]", "", str(item.get("amount", "0")))
            try:
                amount = float(amount)
            except Exception:
                amount = 0
            if amount <= 0:
                continue
            cat = item.get("category", "休閒")
            if cat not in CATEGORIES:
                cat = "休閒"
            spend_date = str(item.get("spend_date", date.today().isoformat()))[:10]
            cleaned.append({
                "spend_date": spend_date,
                "amount": amount,
                "description": str(item.get("description", "未命名消費")),
                "category": cat,
                "merchant": str(item.get("merchant", "")),
            })
        return cleaned
    except Exception:
        st.error("Gemini 回傳格式無法解析，下面是原始辨識結果，可手動新增。")
        st.code(raw)
        return []


init_db()
st.title("🧾 AI 帳單記帳助手｜高準確 Gemini 版")
st.caption("上傳帳單/發票/收據圖片後，自動辨識交易並累積保存。每筆資料都可以編輯或刪除。")

with st.sidebar:
    st.header("設定狀態")
    if get_api_key():
        st.success("已讀取 GEMINI_API_KEY")
    else:
        st.warning("尚未設定 GEMINI_API_KEY")
    st.divider()
    if st.button("⚠️ 一鍵刪除全部資料", type="secondary", use_container_width=True):
        delete_all_transactions()
        st.success("已刪除全部資料")
        st.rerun()

uploaded_files = st.file_uploader(
    "上傳帳單圖片，可一次多張",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True,
)

if uploaded_files:
    if st.button("開始辨識並新增交易", type="primary"):
        total_added = 0
        for f in uploaded_files:
            with st.spinner(f"正在辨識：{f.name}"):
                img = preprocess_image(f)
                with st.expander(f"預覽：{f.name}"):
                    st.image(img, use_container_width=True)
                rows = analyze_bill(img)
                if rows:
                    insert_transactions(rows, f.name)
                    total_added += len(rows)
        st.success(f"完成，已新增 {total_added} 筆交易。")
        st.rerun()

st.subheader("手動新增交易")
with st.form("manual_add", clear_on_submit=True):
    c1, c2, c3, c4 = st.columns([1, 1, 2, 1])
    d = c1.date_input("日期", value=date.today())
    amt = c2.number_input("金額", min_value=0.0, step=1.0)
    desc = c3.text_input("消費內容")
    cat = c4.selectbox("分類", CATEGORIES)
    merchant = st.text_input("商家名稱（可空白）")
    submitted = st.form_submit_button("新增")
    if submitted and amt > 0 and desc:
        insert_transactions([{"spend_date": d.isoformat(), "amount": amt, "description": desc, "category": cat, "merchant": merchant}], "manual")
        st.success("已新增")
        st.rerun()

st.divider()
df = load_transactions()

if df.empty:
    st.info("目前沒有資料。請先上傳帳單圖片或手動新增交易。")
    st.stop()

df["spend_date"] = pd.to_datetime(df["spend_date"], errors="coerce")
df = df.dropna(subset=["spend_date"])
df["month"] = df["spend_date"].dt.strftime("%Y-%m")
months = sorted(df["month"].unique(), reverse=True)
selected_month = st.selectbox("選擇月份", months)
month_df = df[df["month"] == selected_month].copy()

col_a, col_b, col_c = st.columns(3)
col_a.metric("本月總消費", f"${month_df['amount'].sum():,.0f}")
col_b.metric("交易筆數", f"{len(month_df)} 筆")
col_c.metric("平均單筆", f"${month_df['amount'].mean():,.0f}")

st.subheader(f"{selected_month} 分類百分比")
summary = month_df.groupby("category", as_index=False)["amount"].sum()
fig = px.pie(summary, values="amount", names="category", hole=0.35)
st.plotly_chart(fig, use_container_width=True)

st.subheader(f"{selected_month} 明細")
st.dataframe(
    month_df[["id", "spend_date", "amount", "description", "category", "merchant", "source_file"]]
    .sort_values("spend_date", ascending=False),
    use_container_width=True,
    hide_index=True,
)

st.subheader("編輯 / 刪除單筆資料")
ids = month_df["id"].tolist()
selected_id = st.selectbox("選擇交易 ID", ids)
row = month_df[month_df["id"] == selected_id].iloc[0]

with st.form("edit_form"):
    c1, c2, c3, c4 = st.columns([1, 1, 2, 1])
    edit_date = c1.date_input("日期", value=row["spend_date"].date(), key="edit_date")
    edit_amount = c2.number_input("金額", value=float(row["amount"]), min_value=0.0, step=1.0, key="edit_amount")
    edit_desc = c3.text_input("消費內容", value=str(row["description"]), key="edit_desc")
    edit_cat = c4.selectbox("分類", CATEGORIES, index=CATEGORIES.index(row["category"]) if row["category"] in CATEGORIES else 0, key="edit_cat")
    edit_merchant = st.text_input("商家名稱", value=str(row.get("merchant", "")), key="edit_merchant")
    save = st.form_submit_button("儲存修改")
    if save:
        update_transaction(selected_id, edit_date.isoformat(), edit_amount, edit_desc, edit_cat, edit_merchant)
        st.success("已儲存")
        st.rerun()

if st.button("刪除此筆資料", type="secondary"):
    delete_transaction(selected_id)
    st.success("已刪除")
    st.rerun()

st.download_button(
    "下載全部交易 CSV",
    data=load_transactions().to_csv(index=False).encode("utf-8-sig"),
    file_name="transactions.csv",
    mime="text/csv",
)
