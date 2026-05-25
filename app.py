import re
import sqlite3
from datetime import datetime, date
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image
import pytesseract

DB_PATH = Path("transactions.db")
CATEGORIES = ["餐飲", "交通", "網購", "休閒"]

KEYWORDS = {
    "餐飲": ["餐", "飯", "咖啡", "茶", "飲", "麥當勞", "星巴克", "便當", "早餐", "午餐", "晚餐", "restaurant", "cafe", "food", "uber eats", "ubereats", "外送"],
    "交通": ["捷運", "高鐵", "台鐵", "公車", "計程車", "uber", "taxi", "parking", "停車", "加油", "油", "交通"],
    "網購": ["蝦皮", "momo", "pchome", "amazon", "購物", "網購", "shop", "shopping", "訂單", "賣場", "電商"],
    "休閒": ["電影", "遊戲", "netflix", "spotify", "kktv", "娛樂", "休閒", "票", "展覽", "健身", "運動"],
}


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tx_date TEXT NOT NULL,
            amount REAL NOT NULL,
            description TEXT NOT NULL,
            category TEXT NOT NULL,
            raw_text TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def get_conn():
    return sqlite3.connect(DB_PATH)


def load_data():
    conn = get_conn()
    df = pd.read_sql_query("SELECT * FROM transactions ORDER BY tx_date DESC, id DESC", conn)
    conn.close()
    return df


def add_transaction(tx_date, amount, description, category, raw_text=""):
    conn = get_conn()
    conn.execute(
        "INSERT INTO transactions (tx_date, amount, description, category, raw_text, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (tx_date, float(amount), description, category, raw_text, datetime.now().isoformat(timespec="seconds")),
    )
    conn.commit()
    conn.close()


def update_transaction(row_id, tx_date, amount, description, category):
    conn = get_conn()
    conn.execute(
        "UPDATE transactions SET tx_date=?, amount=?, description=?, category=? WHERE id=?",
        (tx_date, float(amount), description, category, int(row_id)),
    )
    conn.commit()
    conn.close()


def delete_transaction(row_id):
    conn = get_conn()
    conn.execute("DELETE FROM transactions WHERE id=?", (int(row_id),))
    conn.commit()
    conn.close()


def delete_all():
    conn = get_conn()
    conn.execute("DELETE FROM transactions")
    conn.commit()
    conn.close()


def ocr_image(image: Image.Image) -> str:
    # Try mixed Chinese/English OCR first; fall back to default English if chi_tra is unavailable.
    try:
        return pytesseract.image_to_string(image, lang="chi_tra+eng")
    except Exception:
        return pytesseract.image_to_string(image)


def parse_date(text: str) -> str:
    patterns = [
        r"(20\d{2})[\/\-.年]\s*(\d{1,2})[\/\-.月]\s*(\d{1,2})",
        r"(\d{1,2})[\/\-.月]\s*(\d{1,2})[\/\-.日]",
    ]
    for p in patterns:
        m = re.search(p, text)
        if m:
            try:
                if len(m.groups()) == 3:
                    y, mo, d = map(int, m.groups())
                else:
                    y = date.today().year
                    mo, d = map(int, m.groups())
                return date(y, mo, d).isoformat()
            except ValueError:
                pass
    return date.today().isoformat()


def parse_amount(text: str) -> float:
    # Prefer amounts near total keywords.
    total_patterns = [
        r"(?:總計|合計|總額|應付|實付|金額|total|amount)\D{0,12}([0-9,]+(?:\.\d{1,2})?)",
        r"(?:NT\$|NTD|TWD|\$)\s*([0-9,]+(?:\.\d{1,2})?)",
    ]
    for p in total_patterns:
        matches = re.findall(p, text, flags=re.IGNORECASE)
        if matches:
            nums = [float(x.replace(",", "")) for x in matches if x.replace(",", "").replace(".", "", 1).isdigit()]
            if nums:
                return max(nums)

    nums = []
    for x in re.findall(r"(?<!\d)([0-9]{2,6}(?:,[0-9]{3})*(?:\.\d{1,2})?)(?!\d)", text):
        try:
            val = float(x.replace(",", ""))
            if 1 <= val <= 999999:
                nums.append(val)
        except ValueError:
            pass
    return max(nums) if nums else 0.0


def parse_description(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    bad_words = ["統一編號", "發票", "日期", "時間", "總計", "合計", "金額", "total", "amount"]
    for ln in lines[:10]:
        low = ln.lower()
        if len(ln) >= 2 and not any(w.lower() in low for w in bad_words) and not re.fullmatch(r"[0-9\s\-/:.,$]+", ln):
            return ln[:60]
    return "未命名消費"


def classify(description: str, raw_text: str) -> str:
    combined = f"{description} {raw_text}".lower()
    scores = {}
    for cat, words in KEYWORDS.items():
        scores[cat] = sum(1 for w in words if w.lower() in combined)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "休閒"


def extract_transaction(image: Image.Image):
    raw = ocr_image(image)
    tx_date = parse_date(raw)
    amount = parse_amount(raw)
    desc = parse_description(raw)
    category = classify(desc, raw)
    return tx_date, amount, desc, category, raw


st.set_page_config(page_title="無 API 帳單記帳 App", page_icon="🧾", layout="wide")
init_db()

st.title("🧾 無 API 帳單記帳 App")
st.caption("完全不需要 OpenAI / Gemini API。使用本機 Tesseract OCR，資料存放在 SQLite。")

with st.sidebar:
    st.header("新增帳單")
    uploaded = st.file_uploader("上傳帳單 / 發票圖片", type=["png", "jpg", "jpeg", "webp"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="已上傳圖片", use_container_width=True)
        if st.button("辨識並新增", type="primary"):
            tx_date, amount, desc, category, raw = extract_transaction(image)
            add_transaction(tx_date, amount, desc, category, raw)
            st.success("已新增一筆交易。若辨識不準，可在下方編輯。")
            st.rerun()

    st.divider()
    st.subheader("手動新增")
    with st.form("manual_add"):
        m_date = st.date_input("日期", value=date.today())
        m_amount = st.number_input("金額", min_value=0.0, step=1.0)
        m_desc = st.text_input("消費內容")
        m_cat = st.selectbox("分類", CATEGORIES)
        submitted = st.form_submit_button("新增")
        if submitted and m_amount > 0 and m_desc.strip():
            add_transaction(m_date.isoformat(), m_amount, m_desc.strip(), m_cat)
            st.success("已手動新增。")
            st.rerun()

    st.divider()
    if st.button("⚠️ 一鍵刪除全部資料"):
        delete_all()
        st.warning("已刪除全部資料。")
        st.rerun()


df = load_data()
if df.empty:
    st.info("目前沒有資料。請先上傳帳單圖片或手動新增一筆交易。")
    st.stop()

df["tx_date"] = pd.to_datetime(df["tx_date"])
df["month"] = df["tx_date"].dt.strftime("%Y-%m")
months = sorted(df["month"].unique(), reverse=True)
selected_month = st.selectbox("選擇月份", months)
monthly = df[df["month"] == selected_month].copy()

col1, col2, col3 = st.columns(3)
col1.metric("本月總金額", f"${monthly['amount'].sum():,.0f}")
col2.metric("交易筆數", f"{len(monthly)}")
col3.metric("平均每筆", f"${monthly['amount'].mean():,.0f}")

left, right = st.columns([1.3, 1])
with left:
    st.subheader("月份明細")
    show = monthly[["id", "tx_date", "amount", "description", "category"]].copy()
    show["tx_date"] = show["tx_date"].dt.date.astype(str)
    st.dataframe(show, use_container_width=True, hide_index=True)

with right:
    st.subheader("分類百分比圓餅圖")
    pie_df = monthly.groupby("category", as_index=False)["amount"].sum()
    fig = px.pie(pie_df, names="category", values="amount", hole=0.25)
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("編輯 / 刪除單筆資料")

for _, row in monthly.sort_values("tx_date", ascending=False).iterrows():
    with st.expander(f"#{row['id']}｜{row['tx_date'].date()}｜${row['amount']:,.0f}｜{row['description']}｜{row['category']}"):
        with st.form(f"edit_{row['id']}"):
            e_date = st.date_input("日期", value=row["tx_date"].date(), key=f"date_{row['id']}")
            e_amount = st.number_input("金額", min_value=0.0, value=float(row["amount"]), step=1.0, key=f"amount_{row['id']}")
            e_desc = st.text_input("消費內容", value=row["description"], key=f"desc_{row['id']}")
            e_cat = st.selectbox("分類", CATEGORIES, index=CATEGORIES.index(row["category"]) if row["category"] in CATEGORIES else 0, key=f"cat_{row['id']}")
            c1, c2 = st.columns(2)
            save = c1.form_submit_button("儲存修改")
            delete = c2.form_submit_button("刪除此筆")
            if save:
                update_transaction(row["id"], e_date.isoformat(), e_amount, e_desc, e_cat)
                st.success("已更新。")
                st.rerun()
            if delete:
                delete_transaction(row["id"])
                st.warning("已刪除。")
                st.rerun()

with st.expander("查看最近 OCR 原始文字"):
    latest = df.sort_values("id", ascending=False).head(1).iloc[0]
    st.text(latest.get("raw_text", ""))
