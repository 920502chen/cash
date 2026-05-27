# 懶人記帳 PWA 外殼頁

這個做法是為了解決 Streamlit 在 iPhone「加入主畫面」時無法穩定顯示 App 名稱與 icon 的問題。

## 使用方式

1. 到 GitHub 建一個新的 repository，例如：
   lazy-ledger-pwa

2. 上傳這些檔案：
   - index.html
   - manifest.json
   - icon-180.png
   - icon-192.png
   - icon-512.png

3. 到 GitHub repo：
   Settings → Pages

4. Source 選：
   Deploy from a branch

5. Branch 選：
   main / root

6. 儲存後，GitHub Pages 會給你一個網址，例如：
   https://你的帳號.github.io/lazy-ledger-pwa/

7. 用 iPhone Safari 打開這個 GitHub Pages 網址。

8. 按分享 → 加入主畫面。

9. 主畫面會出現：
   - 名稱：懶人記帳
   - icon：你提供的 icon

10. 之後點主畫面 App，會自動開啟：
    https://mrsmartledger.streamlit.app/

注意：
不要從 Streamlit 網址加入主畫面。
要從 GitHub Pages 的 PWA 外殼網址加入主畫面。
