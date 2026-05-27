import json
import re
import sqlite3
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image, ImageOps, ImageEnhance
import google.generativeai as genai
from google.api_core.exceptions import NotFound
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
from googleapiclient.errors import HttpError
from streamlit_autorefresh import st_autorefresh
import streamlit.components.v1 as components
import io
import hashlib

DB_PATH = "database.db"
DEFAULT_CATEGORIES = ["餐飲", "交通", "網購", "休閒", "醫療", "退款", "購物", "待分類"]
PIE_EXCLUDED_CATEGORIES = ["退款"]
GDRIVE_BACKUP_FILENAME = "transactions_backup.csv"
GDRIVE_SETTINGS_FILENAME = "settings_backup.json"
BACKUP_MIN_INTERVAL_SECONDS = 60  # 保留相容性；目前採用資料變更後立即備份
GOOGLE_SCOPES = ["https://www.googleapis.com/auth/drive.file"]

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
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS app_settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
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


def update_keyword_rule(rule_id: int, keyword: str, category: str):
    keyword = str(keyword or "").strip()
    if not keyword:
        return False
    conn = get_conn()
    conn.execute(
        "UPDATE keyword_rules SET keyword=?, category=? WHERE id=?",
        (keyword, category, int(rule_id)),
    )
    conn.commit()
    conn.close()
    return True


def delete_keyword_rule(rule_id: int):
    conn = get_conn()
    conn.execute("DELETE FROM keyword_rules WHERE id=?", (int(rule_id),))
    conn.commit()
    conn.close()


def export_settings_dict() -> Dict[str, Any]:
    """匯出自訂分類與關鍵字規則。保留預設分類，匯入到新環境時也能完整還原。"""
    rules = load_keyword_rules()
    return {
        "version": 1,
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "categories": get_categories(),
        "keyword_rules": rules[["keyword", "category"]].to_dict(orient="records") if not rules.empty else [],
    }


def export_settings_json_bytes() -> bytes:
    return json.dumps(export_settings_dict(), ensure_ascii=False, indent=2).encode("utf-8")


def import_settings_dict(data: Dict[str, Any], replace_rules: bool = False) -> Dict[str, int]:
    """匯入設定檔；分類會補上，關鍵字規則可選擇覆蓋或合併。"""
    categories = data.get("categories", []) or []
    rules = data.get("keyword_rules", []) or []
    added_cats = 0
    added_rules = 0

    for cat in categories:
        before = set(get_categories())
        if add_category(str(cat)) and str(cat).strip() not in before:
            added_cats += 1

    conn = get_conn()
    try:
        if replace_rules:
            conn.execute("DELETE FROM keyword_rules")
            conn.commit()
        existing = set(
            (str(r[0]).strip().lower(), str(r[1]).strip())
            for r in conn.execute("SELECT keyword, category FROM keyword_rules").fetchall()
        )
        now = datetime.now().isoformat(timespec="seconds")
        valid_cats = set(get_categories())
        for item in rules:
            if not isinstance(item, dict):
                continue
            keyword = str(item.get("keyword", "")).strip()
            category = str(item.get("category", "待分類")).strip()
            if not keyword:
                continue
            if category not in valid_cats:
                category = "待分類"
            key = (keyword.lower(), category)
            if key in existing:
                continue
            conn.execute(
                "INSERT INTO keyword_rules (keyword, category, created_at) VALUES (?, ?, ?)",
                (keyword, category, now),
            )
            existing.add(key)
            added_rules += 1
        conn.commit()
    finally:
        conn.close()
    return {"categories": added_cats, "keyword_rules": added_rules}


def import_settings_file(uploaded_file, replace_rules: bool = False) -> bool:
    try:
        raw = uploaded_file.read()
        data = json.loads(raw.decode("utf-8-sig"))
        result = import_settings_dict(data, replace_rules=replace_rules)
        st.success(f"設定檔匯入完成：新增 {result['categories']} 個分類、{result['keyword_rules']} 筆關鍵字規則。")
        return True
    except Exception as e:
        st.error(f"設定檔匯入失敗：{type(e).__name__}。請確認是本 App 匯出的 JSON 設定檔。")
        st.exception(e)
        return False


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


def request_confirm(action_key: str, message: str):
    """顯示二次確認區塊。回傳 (confirmed, cancelled)。"""
    st.warning(message)
    c1, c2 = st.columns(2)
    confirmed = c1.button("確認刪除", type="primary", use_container_width=True, key=f"confirm_btn_{action_key}")
    cancelled = c2.button("取消", use_container_width=True, key=f"cancel_btn_{action_key}")
    return confirmed, cancelled


def render_category_keyword_settings(location: str = "main"):
    """顯示自訂分類與關鍵字規則設定。location 用來避免 Streamlit widget key 重複。"""
    cats = get_categories()

    st.subheader("分類設定")
    st.caption("可新增自己的分類；刪除分類時，原本屬於該分類的交易與規則會改成「待分類」。")
    with st.form(f"add_category_form_{location}", clear_on_submit=True):
        new_cat = st.text_input("新增分類名稱", placeholder="例如：房租、娛樂、保險、學習", key=f"new_cat_{location}")
        if st.form_submit_button("新增分類", use_container_width=True):
            if add_category(new_cat):
                st.success(f"已新增分類：{new_cat.strip()}")
                backup_after_change()
                st.rerun()
            else:
                st.warning("分類名稱不可空白，或此分類已存在。")

    cats = get_categories()
    st.write("目前分類：" + "、".join(cats))
    removable = [c for c in cats if c not in ["退款", "待分類"]]
    if removable:
        del_cat = st.selectbox("刪除分類（資料會改成待分類）", removable, key=f"del_cat_{location}")
        if st.button("刪除此分類", use_container_width=True, key=f"delete_cat_btn_{location}"):
            st.session_state[f"confirm_delete_category_{location}"] = del_cat
        pending_cat = st.session_state.get(f"confirm_delete_category_{location}")
        if pending_cat:
            ok, cancel = request_confirm(
                f"delete_category_{location}",
                f"確定要刪除分類「{pending_cat}」嗎？此分類底下的交易與關鍵字規則會改成「待分類」。",
            )
            if ok:
                delete_category(pending_cat)
                st.session_state[f"confirm_delete_category_{location}"] = None
                st.success(f"已刪除分類：{pending_cat}")
                backup_after_change()
                st.rerun()
            if cancel:
                st.session_state[f"confirm_delete_category_{location}"] = None
                st.rerun()
    else:
        st.info("目前沒有可刪除的自訂分類。")

    st.divider()
    st.subheader("關鍵字自動分類")
    st.caption("設定後，之後 OCR 或匯入交易明細時，只要商家/內容包含關鍵字，就會自動套用指定分類。負數仍會優先歸類為「退款」。")
    cats = get_categories()
    with st.form(f"add_rule_form_{location}", clear_on_submit=True):
        keyword = st.text_input("新增關鍵字", placeholder="例如：星巴克、Uber、蝦皮、Netflix", key=f"keyword_{location}")
        rule_cat = st.selectbox("自動分類到", cats, key=f"rule_cat_{location}")
        if st.form_submit_button("新增關鍵字規則", use_container_width=True):
            if add_keyword_rule(keyword, rule_cat):
                st.success(f"已新增規則：包含「{keyword.strip()}」→ {rule_cat}")
                backup_after_change()
                st.rerun()
            else:
                st.warning("關鍵字不可空白。")

    rules_df = load_keyword_rules()
    if rules_df.empty:
        st.info("目前尚未設定關鍵字規則。")
    else:
        st.caption("可直接在表格中修改關鍵字與分類；勾選刪除後，需再按確認才會真的刪除。")
        edit_rules = rules_df.rename(columns={"id": "ID", "keyword": "關鍵字", "category": "分類"}).copy()
        edit_rules["刪除"] = False
        edited_rules = st.data_editor(
            edit_rules,
            use_container_width=True,
            hide_index=True,
            disabled=["ID"],
            column_config={
                "ID": st.column_config.NumberColumn("ID"),
                "關鍵字": st.column_config.TextColumn("關鍵字"),
                "分類": st.column_config.SelectboxColumn("分類", options=cats),
                "刪除": st.column_config.CheckboxColumn("刪除"),
            },
            key=f"keyword_rules_editor_{location}",
        )
        c_save, c_delete = st.columns(2)
        if c_save.button("儲存關鍵字表格修改", type="primary", use_container_width=True, key=f"save_rules_{location}"):
            saved = 0
            for _, r in edited_rules.iterrows():
                if update_keyword_rule(int(r["ID"]), str(r["關鍵字"]), str(r["分類"])):
                    saved += 1
            st.success(f"已儲存 {saved} 筆關鍵字規則。")
            backup_after_change()
            st.rerun()

        if c_delete.button("刪除已勾選關鍵字規則", type="secondary", use_container_width=True, key=f"delete_rules_{location}"):
            ids = edited_rules.loc[edited_rules["刪除"] == True, "ID"].tolist()
            if not ids:
                st.warning("尚未勾選要刪除的關鍵字規則。")
            else:
                st.session_state[f"confirm_delete_rules_{location}"] = [int(i) for i in ids]

        pending_rules = st.session_state.get(f"confirm_delete_rules_{location}") or []
        if pending_rules:
            ok, cancel = request_confirm(
                f"delete_rules_{location}",
                f"確定要刪除 {len(pending_rules)} 筆關鍵字規則嗎？",
            )
            if ok:
                for rule_id in pending_rules:
                    delete_keyword_rule(int(rule_id))
                st.session_state[f"confirm_delete_rules_{location}"] = []
                st.success(f"已刪除 {len(pending_rules)} 筆關鍵字規則。")
                backup_after_change()
                st.rerun()
            if cancel:
                st.session_state[f"confirm_delete_rules_{location}"] = []
                st.rerun()

    st.divider()
    st.subheader("匯入 / 匯出設定檔")
    st.caption("設定檔會包含：自訂分類與關鍵字自動分類規則。交易明細請用下方的 CSV 下載/Google Drive 備份。")
    st.download_button(
        "匯出設定檔 JSON",
        data=export_settings_json_bytes(),
        file_name="settings_backup.json",
        mime="application/json",
        use_container_width=True,
        key=f"download_settings_{location}",
    )
    settings_file = st.file_uploader("匯入設定檔 JSON", type=["json"], key=f"settings_import_{location}")
    replace_rules = st.checkbox("匯入時覆蓋現有關鍵字規則", value=False, key=f"replace_rules_{location}")
    if settings_file is not None and st.button("匯入設定檔", type="primary", use_container_width=True, key=f"import_settings_btn_{location}"):
        if import_settings_file(settings_file, replace_rules=replace_rules):
            backup_after_change()
            st.rerun()


def get_secret(name: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(name, default)).strip()
    except Exception:
        return default


def get_setting(key: str) -> Optional[str]:
    conn = get_conn()
    row = conn.execute("SELECT value FROM app_settings WHERE key=?", (key,)).fetchone()
    conn.close()
    return row[0] if row else None


def set_setting(key: str, value: str):
    conn = get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO app_settings (key, value, updated_at) VALUES (?, ?, ?)",
        (key, value, datetime.now().isoformat(timespec="seconds")),
    )
    conn.commit()
    conn.close()


def delete_setting(key: str):
    conn = get_conn()
    conn.execute("DELETE FROM app_settings WHERE key=?", (key,))
    conn.commit()
    conn.close()


def get_google_redirect_uri() -> str:
    return get_secret("GOOGLE_REDIRECT_URI")


def google_oauth_ready() -> bool:
    return bool(get_secret("GOOGLE_CLIENT_ID") and get_secret("GOOGLE_CLIENT_SECRET") and get_google_redirect_uri())


def make_google_flow() -> Flow:
    client_config = {
        "web": {
            "client_id": get_secret("GOOGLE_CLIENT_ID"),
            "client_secret": get_secret("GOOGLE_CLIENT_SECRET"),
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [get_google_redirect_uri()],
        }
    }
    return Flow.from_client_config(client_config, scopes=GOOGLE_SCOPES, redirect_uri=get_google_redirect_uri())


def get_google_credentials() -> Optional[Credentials]:
    raw = get_setting("google_credentials_json")
    if not raw:
        return None
    try:
        creds = Credentials.from_authorized_user_info(json.loads(raw), scopes=GOOGLE_SCOPES)
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            set_setting("google_credentials_json", creds.to_json())
        return creds if creds and creds.valid else None
    except Exception:
        return None


def handle_google_oauth_callback():
    params = st.query_params
    code = params.get("code")
    if isinstance(code, list):
        code = code[0] if code else None
    if not code:
        return
    if not google_oauth_ready():
        st.error("尚未設定 GOOGLE_CLIENT_ID、GOOGLE_CLIENT_SECRET、GOOGLE_REDIRECT_URI。")
        return
    try:
        flow = make_google_flow()

        # google-auth-oauthlib 新版可能會自動使用 PKCE。
        # 使用者跳到 Google 授權頁後再回到 Streamlit 時，Flow 物件會重建，
        # 所以必須把當初產生的 code_verifier 存起來並在換 token 前放回去，
        # 否則會出現：InvalidGrantError: Missing code verifier。
        code_verifier = st.session_state.get("google_oauth_code_verifier") or get_setting("google_oauth_code_verifier")
        if code_verifier:
            try:
                flow.code_verifier = code_verifier
            except Exception:
                pass

        flow.fetch_token(code=code)
        set_setting("google_credentials_json", flow.credentials.to_json())
        delete_setting("google_oauth_code_verifier")
        delete_setting("google_oauth_state")
        st.session_state.pop("google_oauth_code_verifier", None)
        st.session_state.pop("google_oauth_state", None)
        st.query_params.clear()
        st.success("Google Drive 已授權成功。")
        st.rerun()
    except Exception as e:
        st.error(f"Google 授權失敗：{type(e).__name__}。請確認 OAuth redirect URI 與 Secrets 設定。")
        st.exception(e)


def build_drive_service():
    creds = get_google_credentials()
    if not creds:
        return None
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def export_transactions_csv_bytes() -> bytes:
    df_all = load_transactions()
    return df_all.to_csv(index=False).encode("utf-8-sig")


def current_transactions_hash() -> str:
    """用目前全部交易 CSV 內容產生雜湊值；資料沒變時雜湊會一樣。"""
    return hashlib.sha256(export_transactions_csv_bytes()).hexdigest()


def current_settings_hash() -> str:
    return hashlib.sha256(export_settings_json_bytes()).hexdigest()


def upload_bytes_to_google_drive(service, filename: str, data: bytes, mimetype: str, folder_id: str = "") -> None:
    media = MediaIoBaseUpload(io.BytesIO(data), mimetype=mimetype, resumable=False)
    parent_filter = f" and '{folder_id}' in parents" if folder_id else ""
    query = f"name='{filename}' and trashed=false{parent_filter}"
    result = service.files().list(
        q=query,
        spaces="drive",
        fields="files(id, name)",
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute()
    files = result.get("files", [])
    metadata = {"name": filename, "mimeType": mimetype}
    if folder_id:
        metadata["parents"] = [folder_id]
    if files:
        service.files().update(fileId=files[0]["id"], media_body=media, fields="id", supportsAllDrives=True).execute()
    else:
        service.files().create(body=metadata, media_body=media, fields="id", supportsAllDrives=True).execute()


def backup_transactions_to_google_drive(show_success: bool = True, force: bool = False) -> bool:
    service = build_drive_service()
    if service is None:
        if show_success:
            st.warning("尚未連結 Google Drive，無法備份。")
        return False
    folder_id = get_secret("GDRIVE_FOLDER_ID")
    try:
        csv_bytes = export_transactions_csv_bytes()
        settings_bytes = export_settings_json_bytes()
        transactions_hash = hashlib.sha256(csv_bytes).hexdigest()
        settings_hash = hashlib.sha256(settings_bytes).hexdigest()
        combined_hash = hashlib.sha256((transactions_hash + settings_hash).encode("utf-8")).hexdigest()
        last_hash = get_setting("last_gdrive_backup_hash")
        if (not force) and last_hash == combined_hash:
            if show_success:
                st.info("交易與設定都沒有變更，不需要重新備份。")
            return True

        upload_bytes_to_google_drive(service, GDRIVE_BACKUP_FILENAME, csv_bytes, "text/csv", folder_id)
        upload_bytes_to_google_drive(service, GDRIVE_SETTINGS_FILENAME, settings_bytes, "application/json", folder_id)

        set_setting("last_gdrive_backup_at", datetime.now().isoformat(timespec="seconds"))
        set_setting("last_gdrive_backup_hash", combined_hash)
        set_setting("last_gdrive_transactions_hash", transactions_hash)
        set_setting("last_gdrive_settings_hash", settings_hash)
        set_setting("backup_dirty", "0")
        set_setting("backup_dirty_reason", "")
        if show_success:
            st.success("Google Drive 備份成功：已同步交易 CSV 與設定檔 JSON。")
        return True
    except HttpError as e:
        if show_success:
            st.error("Google Drive 備份失敗。以下是 Google 回傳的原始錯誤：")
            try:
                st.code(e.content.decode("utf-8"), language="json")
            except Exception:
                st.exception(e)
        return False
    except Exception as e:
        if show_success:
            st.error(f"Google Drive 備份失敗：{type(e).__name__}")
            st.exception(e)
        return False


def mark_backup_dirty(reason: str = "資料已變更"):
    """標記目前有尚未備份成功的變更。備份成功後會自動清除。"""
    set_setting("backup_dirty", "1")
    set_setting("backup_dirty_reason", reason)
    set_setting("backup_dirty_at", datetime.now().isoformat(timespec="seconds"))


def backup_after_change():
    """
    安全優先備份：資料庫更新完成後才呼叫本函式。
    先標記 dirty，再立即嘗試同步 Google Drive；若備份失敗，dirty 會保留，方便之後手動補備份。
    """
    mark_backup_dirty()
    if get_google_credentials():
        with st.spinner("資料已更新，正在立即同步 Google Drive 備份…"):
            backup_transactions_to_google_drive(show_success=False, force=False)


def get_backup_dirty_status() -> bool:
    return get_setting("backup_dirty") == "1"


def maybe_auto_backup_to_google_drive():
    """延遲智慧備份：只有有未備份變更，且距離上次嘗試超過一段時間才自動上傳。"""
    if not get_google_credentials() or not get_backup_dirty_status():
        return
    if st.session_state.pop("skip_auto_backup_once", False):
        return

    now = datetime.now()
    last_attempt_raw = get_setting("last_gdrive_backup_attempt_at")
    if last_attempt_raw:
        try:
            last_attempt = datetime.fromisoformat(last_attempt_raw)
            if now - last_attempt < timedelta(seconds=BACKUP_MIN_INTERVAL_SECONDS):
                return
        except Exception:
            pass

    set_setting("last_gdrive_backup_attempt_at", now.isoformat(timespec="seconds"))
    with st.spinner("正在背景同步 Google Drive 備份…"):
        backup_transactions_to_google_drive(show_success=False, force=False)


def handle_force_backup_query():
    """瀏覽器離開頁面時，JS 會用 sendBeacon 打這個 query，若 dirty 就做最後一次備份。"""
    try:
        force_backup = st.query_params.get("force_backup")
    except Exception:
        force_backup = None
    if force_backup == "1" and get_backup_dirty_status() and get_google_credentials():
        backup_transactions_to_google_drive(show_success=False, force=False)
        try:
            st.query_params.clear()
        except Exception:
            pass


def render_beforeunload_backup_hook():
    """
    有未備份變更時，在使用者關閉/重新整理/離開頁面前送出最後備份訊號。
    注意：瀏覽器對 beforeunload 的非同步工作有限制，因此仍搭配每 1 分鐘自動檢查作為主保護。
    """
    if not (get_backup_dirty_status() and get_google_credentials()):
        return
    components.html(
        """
        <script>
        (function() {
          if (window.__billBackupBeforeUnloadInstalled) return;
          window.__billBackupBeforeUnloadInstalled = true;
          window.addEventListener('beforeunload', function() {
            try {
              const url = new URL(window.location.href);
              url.searchParams.set('force_backup', '1');
              url.searchParams.set('from_beforeunload', '1');
              if (navigator.sendBeacon) {
                navigator.sendBeacon(url.toString(), new Blob(['backup'], {type: 'text/plain'}));
              } else {
                fetch(url.toString(), {method: 'POST', body: 'backup', keepalive: true});
              }
            } catch (e) {}
          });
        })();
        </script>
        """,
        height=0,
    )


def setup_dirty_autorefresh():
    """有 dirty 時才每 1 分鐘自動 rerun，讓 Python 端有機會執行備份。"""
    if get_backup_dirty_status() and get_google_credentials():
        st_autorefresh(interval=BACKUP_MIN_INTERVAL_SECONDS * 1000, key="dirty_backup_autorefresh")


def render_google_drive_backup_panel():
    st.header("Google Drive 備份")
    if not google_oauth_ready():
        st.warning("尚未完成 Google OAuth Secrets 設定。")
        st.caption("需要設定 GOOGLE_CLIENT_ID、GOOGLE_CLIENT_SECRET、GOOGLE_REDIRECT_URI。GDRIVE_FOLDER_ID 可選，不填會備份到我的雲端硬碟根目錄。")
        return

    creds = get_google_credentials()
    if not creds:
        flow = make_google_flow()
        auth_url, state = flow.authorization_url(access_type="offline", include_granted_scopes="true", prompt="consent")

        # 儲存 OAuth state 與 PKCE code_verifier，避免 Google 回跳後換 token 失敗。
        if state:
            st.session_state["google_oauth_state"] = state
            set_setting("google_oauth_state", state)
        code_verifier = getattr(flow, "code_verifier", None)
        if code_verifier:
            st.session_state["google_oauth_code_verifier"] = code_verifier
            set_setting("google_oauth_code_verifier", code_verifier)

        st.info("尚未連結 Google Drive。第一次使用請點下方按鈕授權。")
        st.link_button("連結 Google Drive", auth_url, use_container_width=True)
        st.caption("授權後會回到本 App，之後就能備份到你的個人 Google Drive。")
        return

    st.success("Google Drive 已連結")
    last = get_setting("last_gdrive_backup_at")
    dirty = get_backup_dirty_status()
    if last:
        st.caption(f"上次備份：{last}")
    if dirty:
        dirty_at = get_setting("backup_dirty_at") or "剛剛"
        st.warning(f"有尚未備份成功的變更（{dirty_at}）。")
    else:
        st.caption("目前沒有待備份變更。")
    st.caption(f"安全優先備份：新增、編輯、刪除、匯入與設定變更完成後，會立即同步交易 CSV 與設定檔 JSON。")
    if st.button("立即備份到 Google Drive", use_container_width=True):
        backup_transactions_to_google_drive(show_success=True, force=True)
    if st.button("取消 Google Drive 連結", use_container_width=True):
        st.session_state["confirm_disconnect_google"] = True
    if st.session_state.get("confirm_disconnect_google"):
        st.warning("確定要取消 Google Drive 連結嗎？之後需重新授權才能備份。")
        c1, c2 = st.columns(2)
        if c1.button("確認取消連結", type="primary", use_container_width=True, key="confirm_disconnect_google_btn"):
            delete_setting("google_credentials_json")
            st.session_state["confirm_disconnect_google"] = False
            st.rerun()
        if c2.button("保留連結", use_container_width=True, key="cancel_disconnect_google_btn"):
            st.session_state["confirm_disconnect_google"] = False
            st.rerun()


init_db()
handle_google_oauth_callback()
handle_force_backup_query()
CATEGORIES = get_categories()

# 安全優先備份：資料變更後會立即同步；不再依賴離開頁面或定時自動備份。
# 保留手動立即備份按鈕，若先前有未備份變更可隨時補同步。

st.title("🧾 AI 帳單記帳助手｜Gemini + Google Drive 立即備份版")
st.caption("上傳帳單/發票/收據圖片後自動辨識交易；可編輯明細、設定關鍵字分類。資料變更完成後會立即同步交易 CSV 與設定檔 JSON 到個人 Google Drive，安全優先、避免漏備份。")

with st.sidebar:
    st.header("設定狀態")
    st.success("已讀取 GEMINI_API_KEY") if get_api_key() else st.warning("尚未設定 GEMINI_API_KEY")
    st.info("手機版若看不到側邊欄，請直接使用主畫面上方的「分類與關鍵字設定」。")
    st.divider()
    render_google_drive_backup_panel()
    st.divider()
    if st.button("⚠️ 一鍵刪除全部資料", type="secondary", use_container_width=True):
        st.session_state["confirm_delete_all_transactions"] = True
    if st.session_state.get("confirm_delete_all_transactions"):
        st.warning("確定要刪除全部交易資料嗎？此動作無法復原。")
        c1, c2 = st.columns(2)
        if c1.button("確認刪除全部", type="primary", use_container_width=True, key="confirm_delete_all_btn"):
            delete_all_transactions()
            backup_after_change()
            st.session_state["confirm_delete_all_transactions"] = False
            st.success("已刪除全部資料")
            st.rerun()
        if c2.button("取消", use_container_width=True, key="cancel_delete_all_btn"):
            st.session_state["confirm_delete_all_transactions"] = False
            st.rerun()

with st.expander("⚙️ 分類與關鍵字設定（點這裡新增分類 / 設定自動分類）", expanded=True):
    render_category_keyword_settings("main")

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
                    backup_after_change()
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
            backup_after_change()
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
        backup_after_change()
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
    backup_after_change()
    st.success(f"已儲存 {saved} 筆明細修改。")
    st.rerun()

if c_delete.button("刪除已勾選明細", type="secondary", use_container_width=True):
    to_delete = edited_df.loc[edited_df["刪除"] == True, "id"].tolist()
    if not to_delete:
        st.warning("尚未勾選要刪除的明細。")
    else:
        st.session_state[f"confirm_delete_transactions_{selected_month}"] = [int(i) for i in to_delete]

pending_tx = st.session_state.get(f"confirm_delete_transactions_{selected_month}") or []
if pending_tx:
    st.warning(f"確定要刪除 {len(pending_tx)} 筆明細嗎？此動作無法復原。")
    cc1, cc2 = st.columns(2)
    if cc1.button("確認刪除明細", type="primary", use_container_width=True, key=f"confirm_delete_tx_{selected_month}"):
        for row_id in pending_tx:
            delete_transaction(int(row_id))
        backup_after_change()
        st.session_state[f"confirm_delete_transactions_{selected_month}"] = []
        st.success(f"已刪除 {len(pending_tx)} 筆明細。")
        st.rerun()
    if cc2.button("取消", use_container_width=True, key=f"cancel_delete_tx_{selected_month}"):
        st.session_state[f"confirm_delete_transactions_{selected_month}"] = []
        st.rerun()

st.download_button(
    "下載全部交易 CSV",
    data=load_transactions().to_csv(index=False).encode("utf-8-sig"),
    file_name="transactions.csv",
    mime="text/csv",
)