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
from google.api_core.exceptions import NotFound

DB_PATH = "database.db"
DEFAULT_CATEGORIES = ["餐飲", "交通", "網購", "休閒", "醫療", "退款", "購物", "待分類"]
PIE_EXCLUDED_CATEGORIES = ["退款"]

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
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS categories (
            name TEXT PRIMARY KEY,
            created_at TEXT NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS keyword_rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            keyword TEXT NOT NULL,
            category TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    now = datetime.now().isoformat(timespec="seconds")
    for cat in DEFAULT_CATEGORIES:
        cur.execute("INSERT OR IGNORE INTO categories (name, created_at) VALUES (?, ?)", (cat, now))
    conn.commit()
    conn.close()


def get_categories() -> List[str]:
    conn = get_conn()
    rows = conn.execute("SELECT name FROM categories ORDER BY CASE name WHEN '退款' THEN 999 WHEN '待分類' THEN 1000 ELSE 1 END, name").fetchall()
    conn.close()
    cats = [r[0] for r in rows]
    for must in ["退款", "待分類"]:
        if must not in cats:
            cats.append(must)
    return cats


def add_category(name: str) -> bool:
    name = str(name or "").strip()
    if not name:
        return False
    conn = get_conn()
    try:
        conn.execute("INSERT OR IGNORE INTO categories (name, created_at) VALUES (?, ?)", (name, datetime.now().isoformat(timespec="seconds")))
        conn.commit()
        return True
    finally:
        conn.close()


def delete_category(name: str):
    if name in ["退款", "待分類"]:
        return
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("UPDATE transactions SET category='待分類' WHERE category=?", (name,))
    cur.execute("UPDATE keyword_rules SET category='待分類' WHERE category=?", (name,))
    cur.execute("DELETE FROM categories WHERE name=?", (name,))
    conn.commit()
    conn.close()


def load_keyword_rules() -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql_query("SELECT id, keyword, category FROM keyword_rules ORDER BY id DESC", conn)
    conn.close()
    return df


def add_keyword_rule(keyword: str, category: str):
    keyword = str(keyword or "").strip()
    if not keyword:
        return False
    conn = get_conn()
    conn.execute(
        "INSERT INTO keyword_rules (keyword, category, created_at) VALUES (?, ?, ?)",
        (keyword, category, datetime.now().isoformat(timespec="seconds")),
    )
    conn.commit()
    conn.close()
    return True


def delete_keyword_rule(rule_id: int):
    conn = get_conn()
    conn.execute("DELETE FROM keyword_rules WHERE id=?", (int(rule_id),))
    conn.commit()
    conn.close()


def parse_amount(value: Any) -> float:
    """解析金額並保留負號，支援 -120、−120、(120)、退款/刷退等文字。"""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    raw = str(value).strip().replace(",", "")
    is_negative = bool(re.search(r"[-−－]|\([^)]*\)|退款|退貨|刷退|折讓|沖正|沖銷|回饋|退費|credit|refund|reversal", raw, re.I))
    m = re.search(r"\d+(?:\.\d+)?", raw)
    amount = float(m.group(0)) if m else 0.0
    return -amount if is_negative and amount > 0 else amount


def normalize_category(category: Any, amount: float | int | str = 0) -> str:
    """負數一律歸類為退款；其他非白名單分類歸待分類。"""
    try:
        amount_float = float(amount)
    except Exception:
        amount_float = 0.0
    if amount_float < 0:
        return "退款"
    cats = get_categories()
    cat = str(category or "").strip()
    aliases = {
        "吃飯": "餐飲", "飲食": "餐飲", "餐廳": "餐飲", "外送": "餐飲",
        "通勤": "交通", "醫藥": "醫療", "醫院": "醫療", "診所": "醫療", "藥局": "醫療",
        "退貨": "退款", "刷退": "退款", "退費": "退款", "折讓": "退款",
        "消費": "購物", "百貨": "購物", "其他": "待分類", "未分類": "待分類",
    }
    cat = aliases.get(cat, cat)
    return cat if cat in cats else "待分類"


def apply_keyword_rules(description: str, merchant: str, amount: float, original_category: str = "待分類") -> str:
    if amount < 0:
        return "退款"
    text = f"{description or ''} {merchant or ''}".lower()
    rules = load_keyword_rules()
    for _, r in rules.iterrows():
        keyword = str(r.get("keyword", "")).strip().lower()
        if keyword and keyword in text:
            return normalize_category(r.get("category", "待分類"), amount)
    return normalize_category(original_category, amount)


def normalize_import_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {}
    for col in df.columns:
        name = str(col).strip().lower()
        if name in ["spend_date", "date", "日期", "消費日期", "交易日期", "入帳日期"]:
            col_map[col] = "spend_date"
        elif name in ["amount", "金額", "交易金額", "消費金額", "支出", "收入"]:
            col_map[col] = "amount"
        elif name in ["description", "內容", "消費內容", "交易內容", "品項", "摘要", "備註", "說明"]:
            col_map[col] = "description"
        elif name in ["category", "分類", "類別"]:
            col_map[col] = "category"
        elif name in ["merchant", "商家", "店家", "商店", "收款方"]:
            col_map[col] = "merchant"
    return df.rename(columns=col_map)


def rows_from_imported_file(uploaded_file) -> List[Dict[str, Any]]:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        try:
            df = pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding="utf-8-sig")
    else:
        df = pd.read_excel(uploaded_file)

    df = normalize_import_columns(df)
    required = {"spend_date", "amount", "description"}
    missing = required - set(df.columns)
    if missing:
        st.error("交易明細檔至少需要日期、金額、消費內容三種欄位。可用欄位名稱例如：日期/消費日期、金額、內容/消費內容。")
        return []

    rows: List[Dict[str, Any]] = []
    for _, item in df.iterrows():
        amount = parse_amount(item.get("amount"))
        if amount == 0:
            continue
        spend_date = pd.to_datetime(item.get("spend_date"), errors="coerce")
        spend_date_str = date.today().isoformat() if pd.isna(spend_date) else spend_date.date().isoformat()
        description = str(item.get("description", "")).strip()
        if not description or description.lower() == "nan":
            description = "匯入交易"
        merchant = str(item.get("merchant", "") or "").strip()
        if merchant.lower() == "nan":
            merchant = ""
        original_cat = item.get("category", "待分類")
        category = apply_keyword_rules(description, merchant, amount, original_cat)
        rows.append({"spend_date": spend_date_str, "amount": amount, "description": description, "category": category, "merchant": merchant})
    return rows


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
        amount = float(r.get("amount") or 0)
        description = r.get("description") or "未命名消費"
        merchant = r.get("merchant") or ""
        category = apply_keyword_rules(description, merchant, amount, r.get("category", "待分類"))
        cur.execute(
            """
            INSERT INTO transactions
            (spend_date, amount, description, category, merchant, source_file, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (r.get("spend_date") or date.today().isoformat(), amount, description, category, merchant, source_file, now),
        )
    conn.commit()
    conn.close()


def update_transaction(row_id: int, spend_date: str, amount: float, description: str, category: str, merchant: str):
    conn = get_conn()
    conn.execute(
        "UPDATE transactions SET spend_date=?, amount=?, description=?, category=?, merchant=? WHERE id=?",
        (spend_date, amount, description, normalize_category(category, amount), merchant, row_id),
    )
    conn.commit()
    conn.close()


def delete_transaction(row_id: int):
    conn = get_conn()
    conn.execute("DELETE FROM transactions WHERE id=?", (row_id,))
    conn.commit()
    conn.close()


def delete_all_transactions():
    conn = get_conn()
    conn.execute("DELETE FROM transactions")
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
    image.thumbnail((1800, 1800))
    image = ImageEnhance.Contrast(image).enhance(1.15)
    image = ImageEnhance.Sharpness(image).enhance(1.1)
    return image


def get_api_key() -> str:
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return ""


def get_gemini_model():
    candidates = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash-latest", "gemini-1.5-flash"]
    available = []
    try:
        for m in genai.list_models():
            methods = getattr(m, "supported_generation_methods", []) or []
            if "generateContent" in methods:
                available.append(m.name.replace("models/", ""))
    except Exception:
        available = []
    for name in candidates:
        if not available or name in available:
            return genai.GenerativeModel(name), name
    if available:
        return genai.GenerativeModel(available[0]), available[0]
    return genai.GenerativeModel("gemini-2.0-flash"), "gemini-2.0-flash"


def analyze_bill(image: Image.Image) -> List[Dict[str, Any]]:
    api_key = get_api_key()
    if not api_key:
        st.error("尚未設定 GEMINI_API_KEY。請到 Streamlit Cloud 的 App settings → Secrets 加入 GEMINI_API_KEY。")
        return []
    genai.configure(api_key=api_key)
    model, model_name = get_gemini_model()
    today = date.today().isoformat()
    cats = "、".join(get_categories())
    rules = load_keyword_rules()
    rules_text = "\n".join([f"- 文字包含「{r.keyword}」→ 分類「{r.category}」" for _, r in rules.iterrows()]) or "無"
    prompt = f"""
你是台灣帳單、發票、收據、信用卡明細、外送訂單、電商訂單的資料辨識助手。
請仔細閱讀圖片，抽出所有可以確定的消費交易。

請務必遵守：
1. 日期以「實際消費日」為準，不要用帳單繳款日、列印日、結帳日，除非沒有其他日期。
2. 金額請使用實際消費金額，忽略發票號碼、統編、信用卡末四碼、電話、會員編號。
3. 若圖片有多筆交易，請輸出多筆。
4. 分類只能是：{cats}。
5. 金額必須保留正負號：圖片上若出現負號、退款、折讓、退貨、刷退、沖正、回饋、credit、refund、reversal，amount 請輸出負數，例如 -120；所有負數交易 category 一律輸出「退款」；一般消費才輸出正數。
6. 若日期年份不明，請依照台灣常見收據推測為今年；今天是 {today}。
7. 不確定的欄位請合理填寫，但不要憑空新增圖片上不存在的交易。

使用者自訂關鍵字分類規則，優先套用：
{rules_text}

基本分類規則：
- 餐飲：餐廳、咖啡、飲料、外送、超商食品、便當、早餐、夜市食物
- 交通：捷運、公車、高鐵、台鐵、計程車、Uber、停車、加油、共享機車
- 網購：蝦皮、momo、PChome、Amazon、電商訂單、線上購物
- 休閒：電影、遊戲、訂閱、娛樂、旅遊
- 醫療：醫院、診所、藥局、健檢、牙醫、醫療保健
- 購物：百貨、服飾、日用品、家用品、實體店一般購物
- 退款：所有負數款項、刷退、退貨、折讓、回饋、沖正
- 待分類：不屬於以上任何一類或無法判斷

只輸出 JSON array，不要輸出解釋文字。格式如下：
[
  {{"spend_date": "YYYY-MM-DD", "amount": 123, "description": "消費內容/品項摘要", "category": "餐飲", "merchant": "商家名稱"}}
]
"""
    try:
        response = model.generate_content([prompt, image], generation_config={"temperature": 0.1})
    except NotFound:
        st.error(f"Gemini 找不到目前使用的模型：{model_name}。請確認 API Key 可用，或稍後到 Google AI Studio 建立新的 API Key。")
        return []
    except Exception as e:
        st.error(f"Gemini 辨識失敗：{type(e).__name__}。請檢查 API Key、模型權限或配額。")
        return []
    raw = response.text or ""
    try:
        data = json.loads(normalize_json_text(raw))
        if isinstance(data, dict):
            data = data.get("transactions", [data])
        cleaned = []
        for item in data:
            if not isinstance(item, dict):
                continue
            amount = parse_amount(item.get("amount", "0"))
            if amount == 0:
                continue
            description = str(item.get("description", "未命名消費"))
            merchant = str(item.get("merchant", ""))
            cat = apply_keyword_rules(description, merchant, amount, item.get("category", "待分類"))
            spend_date = str(item.get("spend_date", date.today().isoformat()))[:10]
            cleaned.append({"spend_date": spend_date, "amount": amount, "description": description, "category": cat, "merchant": merchant})
        return cleaned
    except Exception:
        st.error("Gemini 回傳格式無法解析，下面是原始辨識結果，可手動新增。")
        st.code(raw)
        return []


init_db()
CATEGORIES = get_categories()

st.title("🧾 AI 帳單記帳助手｜高準確 Gemini 版")
st.caption("上傳帳單/發票/收據圖片後，自動辨識交易並累積保存。每筆資料都可以在明細表直接編輯或刪除。")

with st.sidebar:
    st.header("設定狀態")
    st.success("已讀取 GEMINI_API_KEY") if get_api_key() else st.warning("尚未設定 GEMINI_API_KEY")
    st.divider()
    st.subheader("分類設定")
    with st.form("add_category_form", clear_on_submit=True):
        new_cat = st.text_input("新增分類名稱")
        if st.form_submit_button("新增分類", use_container_width=True):
            if add_category(new_cat):
                st.success("已新增分類")
                st.rerun()
    removable = [c for c in CATEGORIES if c not in ["退款", "待分類"]]
    if removable:
        del_cat = st.selectbox("刪除分類（資料會改成待分類）", removable)
        if st.button("刪除此分類", use_container_width=True):
            delete_category(del_cat)
            st.success("已刪除分類")
            st.rerun()
    st.divider()
    st.subheader("關鍵字自動分類")
    with st.form("add_rule_form", clear_on_submit=True):
        keyword = st.text_input("關鍵字，例如：星巴克、Uber、蝦皮")
        rule_cat = st.selectbox("自動分類到", CATEGORIES, key="rule_cat")
        if st.form_submit_button("新增關鍵字規則", use_container_width=True):
            if add_keyword_rule(keyword, rule_cat):
                st.success("已新增規則")
                st.rerun()
    rules_df = load_keyword_rules()
    if not rules_df.empty:
        st.dataframe(rules_df.rename(columns={"id": "ID", "keyword": "關鍵字", "category": "分類"}), hide_index=True, use_container_width=True)
        rule_ids = rules_df["id"].tolist()
        selected_rule = st.selectbox("選擇要刪除的規則 ID", rule_ids)
        if st.button("刪除關鍵字規則", use_container_width=True):
            delete_keyword_rule(selected_rule)
            st.success("已刪除規則")
            st.rerun()
    st.divider()
    if st.button("⚠️ 一鍵刪除全部資料", type="secondary", use_container_width=True):
        delete_all_transactions()
        st.success("已刪除全部資料")
        st.rerun()

uploaded_files = st.file_uploader("上傳帳單圖片，可一次多張", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)
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

st.subheader("匯入交易明細檔")
st.caption("支援 CSV / Excel。欄位可用：日期/消費日期/交易日期、金額、內容/消費內容/摘要、分類、商家。匯入時也會套用關鍵字自動分類。")
statement_file = st.file_uploader("選擇交易明細檔", type=["csv", "xlsx"], key="statement_import")
if statement_file is not None:
    if st.button("導入交易明細檔", type="primary"):
        rows = rows_from_imported_file(statement_file)
        if rows:
            insert_transactions(rows, statement_file.name)
            st.success(f"已從 {statement_file.name} 導入 {len(rows)} 筆交易。")
            st.rerun()

st.subheader("手動新增交易")
with st.form("manual_add", clear_on_submit=True):
    c1, c2, c3, c4 = st.columns([1, 1, 2, 1])
    d = c1.date_input("日期", value=date.today())
    amt = c2.number_input("金額（退款/折讓可輸入負數）", step=1.0)
    desc = c3.text_input("消費內容")
    cat = c4.selectbox("分類", CATEGORIES)
    merchant = st.text_input("商家名稱（可空白）")
    submitted = st.form_submit_button("新增")
    if submitted and amt != 0 and desc:
        insert_transactions([{"spend_date": d.isoformat(), "amount": amt, "description": desc, "category": cat, "merchant": merchant}], "manual")
        st.success("已新增")
        st.rerun()

st.divider()
df = load_transactions()
if df.empty:
    st.info("目前沒有資料。請先上傳帳單圖片、匯入明細檔或手動新增交易。")
    st.stop()

df["spend_date"] = pd.to_datetime(df["spend_date"], errors="coerce")
df = df.dropna(subset=["spend_date"])
df["month"] = df["spend_date"].dt.strftime("%Y-%m")
months = sorted(df["month"].unique(), reverse=True)
selected_month = st.selectbox("選擇月份", months)
month_df = df[df["month"] == selected_month].copy()

st.subheader(f"{selected_month} 明細")
st.caption("可直接在表格中修改資料；表格一改，下面的統計與圓餅圖會先用畫面上的最新內容重算。按「儲存表格修改」後才會寫入資料庫。")
edit_df = month_df[["id", "spend_date", "amount", "description", "category", "merchant", "source_file"]].sort_values("spend_date", ascending=False).copy()
edit_df["spend_date"] = edit_df["spend_date"].dt.date
edit_df["刪除"] = False
edited_df = st.data_editor(
    edit_df,
    use_container_width=True,
    hide_index=True,
    disabled=["id", "source_file"],
    column_config={
        "id": st.column_config.NumberColumn("ID", help="系統流水號，不需操作"),
        "spend_date": st.column_config.DateColumn("日期", format="YYYY-MM-DD"),
        "amount": st.column_config.NumberColumn("金額", step=1.0, format="%.0f"),
        "description": st.column_config.TextColumn("消費內容"),
        "category": st.column_config.SelectboxColumn("分類", options=CATEGORIES),
        "merchant": st.column_config.TextColumn("商家"),
        "source_file": st.column_config.TextColumn("來源"),
        "刪除": st.column_config.CheckboxColumn("刪除"),
    },
    key=f"editor_{selected_month}",
)

# 使用畫面上已編輯、未刪除的表格內容即時重算統計與圖表。
live_df = edited_df.copy()
live_df = live_df[live_df["刪除"] != True].copy()
live_df["amount"] = pd.to_numeric(live_df["amount"], errors="coerce").fillna(0)
live_df["category"] = [normalize_category(cat, amt) for cat, amt in zip(live_df["category"], live_df["amount"])]

col_a, col_b, col_c = st.columns(3)
col_a.metric("本月淨額", f"${live_df['amount'].sum():,.0f}")
col_b.metric("交易筆數", f"{len(live_df)} 筆")
refund_total = live_df.loc[live_df["amount"] < 0, "amount"].sum()
col_c.metric("退款/折讓合計", f"${refund_total:,.0f}")

st.subheader(f"{selected_month} 分類百分比")
positive_df = live_df[(live_df["amount"] > 0) & (~live_df["category"].isin(PIE_EXCLUDED_CATEGORIES))].copy()
if positive_df.empty:
    st.info("本月沒有正向消費金額，無法產生消費百分比圓餅圖。")
else:
    summary = positive_df.groupby("category", as_index=False)["amount"].sum()
    fig = px.pie(summary, values="amount", names="category", hole=0.35)
    st.caption("圓餅圖會依照目前表格畫面內容即時重算；只統計正向消費，並排除退款類別。")
    st.plotly_chart(fig, use_container_width=True)

c_save, c_delete = st.columns([1, 1])
if c_save.button("儲存表格修改", type="primary", use_container_width=True):
    saved = 0
    for _, r in edited_df.iterrows():
        update_transaction(
            int(r["id"]),
            pd.to_datetime(r["spend_date"]).date().isoformat(),
            float(r["amount"]),
            str(r["description"]),
            str(r["category"]),
            str(r.get("merchant", "") or ""),
        )
        saved += 1
    st.success(f"已儲存 {saved} 筆明細修改。")
    st.rerun()

if c_delete.button("刪除已勾選明細", type="secondary", use_container_width=True):
    to_delete = edited_df.loc[edited_df["刪除"] == True, "id"].tolist()
    if not to_delete:
        st.warning("尚未勾選要刪除的明細。")
    else:
        for row_id in to_delete:
            delete_transaction(int(row_id))
        st.success(f"已刪除 {len(to_delete)} 筆明細。")
        st.rerun()

st.download_button(
    "下載全部交易 CSV",
    data=load_transactions().to_csv(index=False).encode("utf-8-sig"),
    file_name="transactions.csv",
    mime="text/csv",
)
