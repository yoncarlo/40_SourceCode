# モジュール群のインポート
import os
import re
import time
import random
import traceback
import json
import unicodedata
from typing import List, Optional, Tuple
from pathlib import Path
from urllib.request import Request, urlopen
from io import StringIO

import pandas as pd
import pyparsing as pp
from tqdm import tqdm
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from webdriver_manager.chrome import ChromeDriverManager

# ───────────────────────────────────────────────────
# ★ User-Agentリスト ★
# ============================================================
# 共通設定
# ============================================================
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:115.0) Gecko/20100101 Firefox/115.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:115.0) Gecko/20100101 Firefox/115.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Edg/115.0.0.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 OPR/85.0.4341.72",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 OPR/85.0.4341.72",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Vivaldi/5.3.2679.55",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Vivaldi/5.3.2679.55",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Brave/1.40.107",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Brave/1.40.107",
]

DEFAULT_CONFIG = {
    "urls": {
        "kb_login_url": "https://p.keibabook.co.jp/login/login",
        "kb_syutuba_base_url": "https://p.keibabook.co.jp/cyuou/syutuba",
        "kb_cyokyo_base_url": "https://p.keibabook.co.jp/cyuou/cyokyo/0/0",
    }
}

_CREDENTIALS_LOADED = False


# ============================================================
# 設定・認証
# ============================================================
def load_scraping_config(config_path: Optional[str] = None) -> dict:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if config_path is None:
        config_path = os.getenv("SCRAPING_CONFIG_PATH", str(Path(__file__).with_name("scraping_config.json")))
    path = Path(config_path)
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        for section_name, section_values in loaded.items():
            if isinstance(section_values, dict) and isinstance(cfg.get(section_name), dict):
                cfg[section_name].update(section_values)
            else:
                cfg[section_name] = section_values
    return cfg


def _load_credentials_from_file() -> None:
    global _CREDENTIALS_LOADED
    if _CREDENTIALS_LOADED:
        return

    env_path = os.getenv("SCRAPING_CREDENTIALS_PATH", "").strip()
    candidates: List[Path] = []
    if env_path:
        candidates.append(Path(env_path))
    base_dir = Path(__file__).resolve().parent
    candidates.append(base_dir / "scraping_credentials.env")
    candidates.append(base_dir / "scraping_credentials.env.example")

    for path in candidates:
        if not path.exists():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            if key and key not in os.environ:
                os.environ[key] = value
        break

    _CREDENTIALS_LOADED = True


def _get_login_credentials(site_key: str) -> tuple[str, str]:
    _load_credentials_from_file()
    env_prefix = site_key.upper()
    username = os.getenv(f"{env_prefix}_LOGIN_ID", "").strip()
    password = os.getenv(f"{env_prefix}_LOGIN_PASSWORD", "").strip()
    if not username or not password:
        raise RuntimeError(
            f"Missing login credentials. Set {env_prefix}_LOGIN_ID and {env_prefix}_LOGIN_PASSWORD."
        )
    return username, password


# ============================================================
# HTML整形ヘルパー
# ============================================================
def _normalize_label(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    return re.sub(r"\s+", "", text)


def _normalize_result_table_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_normalize_label(col) for col in df.columns]
    return df
# ───────────────────────────────────────────────────
# ============================================================
# netkeiba スクレイピング
# ============================================================
def scrape_nk_kaisai_date(from_:str, to_:str):
    """
    from_とto_をyyyy-mmの形で指定すると、間の開催日を取得する関数
    """
    nk_kaisai_date_list = []
    for date in tqdm(pd.date_range(from_,to_,freq="MS")):
        
        # 年と月を取得
        year = date.year
        month = date.month
        
        # ランダムなUser-Agentを設定
        user_agent = random.choice(USER_AGENTS)
        url = f"https://race.netkeiba.com/top/calendar.html?year={year}&month={month}"
        
        headers = {
            "User-Agent": user_agent,
            "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
        }

        request = Request(f"{url}", headers=headers)
        html    = urlopen(request).read()
        time.sleep(3) #n秒間隔で処理を回す*忘れないように！
        
        # BeautifulSoupでHTMLを解析
        soup = BeautifulSoup(html,"lxml")
        a_list = soup.find("table",class_="Calendar_Table").find_all("a")  # type: ignore

        for a in a_list:
            kaisai_date = re.findall(r"kaisai_date=(\d{8})", a["href"])[0]
            nk_kaisai_date_list.append(kaisai_date)
    return nk_kaisai_date_list

def scrape_nk_raceid_list(nk_kaisai_date_list: List[str]):
    """
    netkeibaのカレンダーページから取得した開催日を元に開催日ページへアクセスし、
    さらにレースの詳細ページにアクセスするためのレースidを抽出する。またTargetにコメントを取り込むためのレースidを生成する関数
    """
    # Chrome オプション
    ua = random.choice(USER_AGENTS)
    options = Options()
    #options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument(f"--user-agent={ua}")
    options.add_argument("--lang=ja")
    
    caps = DesiredCapabilities.CHROME.copy()
    caps["pageLoadStrategy"] = "eager"  # ページの読み込み戦略を設定

    # pageLoadStrategy を eager に
    options.set_capability("pageLoadStrategy", "eager")

    # ドライバ起動
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )
    # get() のタイムアウトを 30 秒に
    driver.set_page_load_timeout(30)


    target_raceid_list = []

    for date in tqdm(nk_kaisai_date_list, desc="レースID取得"):
        url = f"https://race.netkeiba.com/top/race_list.html?kaisai_date={date}"
        try:
            driver.get(url)
            wait = WebDriverWait(driver, 10)
            # 要素が現れたら先に進む
            wait.until(EC.presence_of_element_located((By.CLASS_NAME, "RaceList_DataItem")))
            items = driver.find_elements(By.CLASS_NAME, "RaceList_DataItem")

            for it in items:
                href = it.find_element(By.TAG_NAME, "a").get_attribute("href")
                m = re.search(r"race_id=(\d{12})", href)
                if not m:
                    continue
                nk = m.group(1)
                target_raceid_list.append(f"{date}{nk[4:]}")
        except (TimeoutException, WebDriverException) as e:
            print(f"⚠️ {date} の取得でエラー: {e.__class__.__name__}, スキップします")
            traceback.print_exc()
            # 次の日付へ
            continue
        finally:
            # 同一 IP 連打緩和のため少し待つ
            time.sleep(random.uniform(1.0, 2.5))

    driver.quit()
    return target_raceid_list

# ============================================================
# 競馬ブック スクレイピング
# ============================================================
def scrape_kb_kaisai_date(from_: str, to_: str):
    """
    ログイン済みの Selenium セッションを使って、競馬ブック（月間）カレンダーページを取得し、
    開催日（YYYYMMDD形式）のソート済みリストを返す (YYYYMMDD).
    """
    # Step 1: ブラウザ設定と認証情報の準備
    ua = random.choice(USER_AGENTS)
    options = Options()
    # options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument(f"--user-agent={ua}")
    options.add_argument("--lang=ja")
    options.set_capability("pageLoadStrategy", "eager")

    driver_path = ChromeDriverManager().install()
    days = set()
    config = load_scraping_config()
    login_url = config["urls"]["kb_login_url"]
    nittei_base_url = config["urls"].get("kb_nittei_base_url", "https://p.keibabook.co.jp/cyuou/nittei")
    username, password = _get_login_credentials("kb")

    # Step 2: ログインして月間カレンダーを巡回
    with webdriver.Chrome(service=Service(driver_path), options=options) as driver:
        wait = WebDriverWait(driver, 15)
        # login
        driver.get(login_url)
        wait.until(EC.presence_of_element_located((By.NAME, "login_id"))).send_keys(username)
        wait.until(EC.presence_of_element_located((By.NAME, "pswd"))).send_keys(password)
        wait.until(EC.element_to_be_clickable((By.XPATH, "//input[@type='submit']"))).click()
        time.sleep(1)
        # Step 3: 月ごとに開催日(YYYYMMDD)を抽出
        for d in pd.date_range(from_, to_, freq="MS"):
            year = d.year
            month = f"{d.month:02d}"
            url = f"{nittei_base_url}/{year}{month}"
            driver.get(url)
            wait.until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
            for a in driver.find_elements(By.CSS_SELECTOR, "a[href*='/cyuou/nittei/']"):
                href = a.get_attribute('href') or ''
                m = re.search(r"/cyuou/nittei/(\d{8})", href)
                if m:
                    days.add(m.group(1))
            time.sleep(0.5)
    return sorted(days)

# KB場コード → JV場コード マッピング
KB_TO_JV_PLACE = {
  "08": "01",  # 札幌
  "09": "02",  # 函館
  "06": "03",  # 福島
  "07": "04",  # 新潟
  "04": "05",  # 東京
  "05": "06",  # 中山
  "02": "07",  # 中京
  "00": "08",  # 京都
  "01": "09",  # 阪神
  "03": "10",  # 小倉
}

def _kb_to_target_raceid(day: str, kb_raceid: str) -> str:
    """
    KB 12桁(年+回次+KB場+日次+R) と day(YYYYMMDD) から、
    Target新仕様16桁 yyyymmddppkknnrr を作る。
    """
    if not (isinstance(day, str) and re.fullmatch(r"\d{8}", day)):
        raise ValueError(f"day must be YYYYMMDD (8 digits). got={day}")
    if not (isinstance(kb_raceid, str) and re.fullmatch(r"\d{12}", kb_raceid)):
        raise ValueError(f"kb_raceid must be 12 digits. got={kb_raceid}")

    kaiji   = kb_raceid[4:6]    # 回次(2桁)
    kb_place= kb_raceid[6:8]    # KB場コード(2桁)
    nichiji = kb_raceid[8:10]   # 日次(2桁)
    rr      = kb_raceid[10:12]  # R(2桁)

    jv_place = KB_TO_JV_PLACE.get(kb_place)
    if not jv_place:
        raise ValueError(f"unknown KB place code: {kb_place} (kb_raceid={kb_raceid})")

    return f"{day}{jv_place}{kaiji}{nichiji}{rr}"


def scrape_kb_raceid_list(kb_kaisai_date_list: List[str]) -> Tuple[List[str], List[str]]:
    """
    各開催日(day=YYYYMMDD)の競馬ブック日程ページを開き、
    ページ上の出現順を維持したまま kb_raceid を収集し、
    同時に Target新仕様16桁ID(yyyymmddppkknnrr) を生成して返す。

    return:
      kb_raceid_list, kb_target_raceid_list (index 1:1対応)
    """
    # Step 1: ブラウザ設定と認証情報の準備
    ua = random.choice(USER_AGENTS)
    options = Options()
    # options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument(f"--user-agent={ua}")
    options.add_argument("--lang=ja")
    options.set_capability("pageLoadStrategy", "eager")

    kb_raceids: List[str] = []
    kb_target_raceids: List[str] = []

    # 全体で重複排除（順序は維持）
    seen_global = set()

    config = load_scraping_config()
    login_url = config["urls"]["kb_login_url"]
    nittei_base_url = config["urls"].get("kb_nittei_base_url", "https://p.keibabook.co.jp/cyuou/nittei")
    username, password = _get_login_credentials("kb")

    with webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options) as driver:
        wait = WebDriverWait(driver, 15)

        # Step 2: ログイン
        driver.get(login_url)
        wait.until(EC.presence_of_element_located((By.NAME, "login_id"))).send_keys(username)
        wait.until(EC.presence_of_element_located((By.NAME, "pswd"))).send_keys(password)
        wait.until(EC.element_to_be_clickable((By.XPATH, "//input[@type='submit']"))).click()
        time.sleep(1)

        # Step 3: 開催日ごとに kb_raceid / target_raceid を収集
        for day in kb_kaisai_date_list:
            url = f"{nittei_base_url}/{day}"
            try:
                driver.get(url)
                wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))

                # main コンテンツに限定（無ければフォールバック）
                try:
                    root = WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, "#flex_container_top > div.main"))
                    )
                except TimeoutException:
                    root = driver.find_element(By.CSS_SELECTOR, "div.main")

                # 予想/確定どちらも拾えるように syutuba / seiseki から抽出
                selector = "a[href*='/cyuou/syutuba/'], a[href*='/cyuou/seiseki/']"
                anchors = root.find_elements(By.CSS_SELECTOR, selector)

                # ページ出現順のまま一旦溜める
                tmp_ids: List[str] = []
                for a in anchors:
                    href = (a.get_attribute("href") or "").split("?", 1)[0]
                    m = re.search(r"/cyuou/(?:syutuba|seiseki)/(\d{12})", href)
                    if m:
                        tmp_ids.append(m.group(1))

                # ★順序維持で重複排除（set+sorted禁止）
                seen_day = set()
                for kb_id in tmp_ids:
                    if kb_id in seen_day:
                        continue
                    seen_day.add(kb_id)

                    if kb_id in seen_global:
                        continue
                    seen_global.add(kb_id)

                    kb_raceids.append(kb_id)
                    kb_target_raceids.append(_kb_to_target_raceid(day, kb_id))

                time.sleep(1)

            except Exception as e:
                print(f"[WARN] nittei fetch failed day={day}: {e}")
                traceback.print_exc()
                continue

    return kb_raceids, kb_target_raceids

def scrape_kb_rating(kb_raceid_list: List[str], target_raceid_list: List[str]):
    """
    出馬表ページから馬番とレイティングを取得し、DataFrame形式で返す関数。
    レイティングが空欄の場合はスキップ。
    
    Parameters:
    - kb_raceid_list: 競馬ブックのレースIDのリスト
    - target_raceid_list: Target取り込み用レースIDのリスト
    - max_iterations: 最大処理レース数（デバッグ用）
    
    Returns:
    - DataFrame: 馬番付きTargetレースIDとレイティングのデータフレーム
    """

    # Step 1: 認証・URLの準備
    config = load_scraping_config()
    login_url = config["urls"]["kb_login_url"]
    syutuba_base_url = config["urls"]["kb_syutuba_base_url"]
    username, password = _get_login_credentials("kb")

    # Step 2: Selenium ブラウザを起動
    options = Options()
    # ヘッドレスモードで実行（ブラウザを表示しない）
    #options.add_argument("--headless")  
    driver_path = ChromeDriverManager().install()

    with webdriver.Chrome(service=Service(driver_path), options=options) as driver:
        # Step 3: ログインして出馬表を巡回
        wait = WebDriverWait(driver, 10)

        # ログインページにアクセス
        driver.get(login_url)
        wait.until(EC.presence_of_element_located((By.NAME, "login_id"))).send_keys(username)
        wait.until(EC.presence_of_element_located((By.NAME, "pswd"))).send_keys(password)
        wait.until(EC.element_to_be_clickable((By.XPATH, "//input[@type='submit']"))).click()
        time.sleep(2)
        
        # Step 4: 取得結果を格納するリスト
        target_horseid = []  # 競馬ブック:馬番/Target:取り込み用レースid
        kb_rating = []  # 競馬ブック:レイティング/Target:指数１

        for i, race_id in enumerate(tqdm(kb_raceid_list)):
            # デバッグ用に最大処理回数を設定（必要に応じてコメントアウト）
            #if i >= max_iterations:
            #    break

            try:
                # 出馬表ページへ
                kb_data_url = f"{syutuba_base_url}/{race_id}"
                driver.get(kb_data_url)
                time.sleep(1)

                # 出馬表ページのHTMLを待って取得
                wait.until(EC.presence_of_element_located((By.CLASS_NAME, "syutuba")))
                soup = BeautifulSoup(driver.page_source, 'html.parser')

                # 出馬表テーブルを取得
                race_card_table = soup.find('table', {'class': 'syutuba'})
                if not race_card_table:
                    print(f"Race card not found for race ID: {race_id}")
                    continue

                df = pd.read_html(StringIO(str(race_card_table)))[0]
                df = _normalize_result_table_columns(df)

                horse_col_alias = {"馬番"}
                rating_col_alias = {"レイティング", "指数", "指数1", "指数１"}

                def _find_col(columns, aliases):
                    for c in columns:
                        if c in aliases:
                            return c
                    return None

                horse_col = _find_col(df.columns, horse_col_alias)
                rating_col = _find_col(df.columns, rating_col_alias)
                if not horse_col or not rating_col:
                    print(f"[WARN] rating columns missing race_id={race_id} columns={list(df.columns)}")
                    continue

                for _, row in df.iterrows():
                    horse_number = str(row.get(horse_col, "")).strip()
                    horse_number = re.sub(r"\D", "", horse_number).zfill(2)
                    if not horse_number:
                        continue

                    rating = str(row.get(rating_col, "")).strip()
                    if not rating:
                        continue

                    target_horseid.append(target_raceid_list[i] + horse_number)
                    kb_rating.append(rating)

            except Exception as e:
                print(f"Error scraping race ID {race_id}: {e}")
                print(traceback.format_exc())
                continue
            time.sleep(1)  # 次のページにアクセスする前に待機

        # Step 5: DataFrameに整形
        rating_df = pd.DataFrame({'target_horseid': target_horseid,'レイティング': kb_rating,})

    return rating_df

# ============================================================
# 追切データのパース
# ============================================================
OIKIRI_COLS = ['乗り役','日付','コース','馬場状態','8F','7F','6F','5F(4F)','4F(3F)','3F(2F)','1F','回り位置','脚色']

# ── Utility ───────────────────────────────────
def _clean(t: str) -> str:
    return re.sub(r'\s+', ' ', (t or '')).strip()

def _page_ymd_from_text(txt: str):
    m = re.search(r'(\d{4})年\s*(\d{1,2})月\s*(\d{1,2})日', txt or '')
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None

def _norm_date(text: str, base_year: int | None) -> str:
    m = re.search(r'(\d{1,2})/(\d{1,2})', text or '')
    if not m or not base_year:
        return _clean(text)
    mm, dd = int(m.group(1)), int(m.group(2))
    return f'{base_year:04d}-{mm:02d}-{dd:02d}'

def _pick_alias(tr, tds, header_map: dict[str, int], aliases: list[str], idx_fallback: int | None = None) -> str:
    # 1) class 指定で取得（最優先）
    for cls in aliases:
        el = tr.select_one(f'td.{cls}')
        if el:
            return _clean(el.get_text())
    # 2) header_map から列位置を特定
    for alias in aliases:
        if alias in header_map:
            idx = header_map[alias]
            if 0 <= idx < len(tds):
                return _clean(tds[idx].get_text())
    # 3) 最後の保険（必要な場合のみ）
    if idx_fallback is not None and 0 <= idx_fallback < len(tds):
        return _clean(tds[idx_fallback].get_text())
    return ''

def _extract_laps_right_aligned(tds) -> list[str]:
    """セル中の 00.0 を拾い、7枠（8F..1F）に右詰めで詰める"""
    nums = []
    for td in tds:
        m = re.search(r'\d{1,2}\.\d', _clean(td.get_text()))
        if m: nums.append(m.group(0))
    out = [''] * 7
    for i, v in enumerate(reversed(nums[-7:])):
        out[-1 - i] = v
    return out

def _extract_horse_no_for_block(td_cyokyo) -> int | None:
    """同じブロックの見出しテーブル（class=cyokyo）から馬番を拾う"""
    tbl = td_cyokyo.find_parent('table', class_='cyokyo') or td_cyokyo.find_previous('table', class_='cyokyo')
    if not tbl:
        return None
    umaban = tbl.select_one('td.umaban')
    if umaban:
        val = re.sub(r"\D", "", umaban.get_text(strip=True))
        if val:
            return int(val)
    txt = tbl.get_text(' ', strip=True)
    m = re.search(r'馬\s*番.*?\b(\d{1,2})\b', txt)
    if m:
        return int(m.group(1))
    m2 = re.findall(r'\b(\d{1,2})\b', txt)
    return int(m2[0]) if m2 else None

def parse_block_rows(html: str, target_raceid: str) -> list[dict]:
    """1頭分ブロックから、time/oikiri行を抽出"""
    soup = BeautifulSoup(html, 'lxml')
    base_ymd = _page_ymd_from_text(soup.get_text(' ', strip=True))
    base_year = base_ymd[0] if base_ymd else None
    rows = []

    for td in soup.select('td.cyokyo'):
        # Step 1: 馬番を取得（取れない場合はブロックごとスキップ）
        horse_no = _extract_horse_no_for_block(td)
        if horse_no is None:
            print(f"[WARN] horse_no not found for training block: target_raceid={target_raceid}")
            continue
        target_horseid = f'{target_raceid}{horse_no:02d}'
        tbl = td.select_one('table.cyokyodata')
        if not tbl:
            continue

        # Step 2: ヘッダ行から列位置をマップ化
        header_map = {}
        ths = tbl.select('thead th')
        if ths:
            for idx, th in enumerate(ths):
                key = _normalize_label(th.get_text())
                if key:
                    header_map[key] = idx

        for tr in tbl.select('tr'):
            cls = tr.get('class') or []
            if not any(c in ('time', 'oikiri') for c in cls):
                continue

            # Step 3: 列名/クラス優先で各項目を取得
            tds = tr.select('td')
            rider  = _pick_alias(tr, tds, header_map, ['norite', '騎乗者'])
            nichi  = _pick_alias(tr, tds, header_map, ['nichi','tukihi', '日付'])
            course = _pick_alias(tr, tds, header_map, ['course','corse', 'コース'])
            baba   = _pick_alias(tr, tds, header_map, ['baba', '馬場'])
            laps   = _extract_laps_right_aligned(tds)
            mawari = _pick_alias(tr, tds, header_map, ['mawariiti', '回り位置'])
            asiiro = _pick_alias(tr, tds, header_map, ['akiiro','asiiro', '脚色'])

            row = {
                'target_horseid': target_horseid,
                '乗り役': rider,
                '日付': _norm_date(nichi, base_year),
                'コース': course,
                '馬場状態': baba,
                '8F': laps[0],
                '7F': laps[1],
                '6F': laps[2],
                '5F(4F)': laps[3],
                '4F(3F)': laps[4],
                '3F(2F)': laps[5],
                '1F': laps[6],
                '回り位置': mawari,
                '脚色': asiiro,
                'row_kind': 'oikiri' if 'oikiri' in cls else 'time',
            }
            # Step 4: 数字が1つでもあれば採用
            if any(v for v in laps):
                rows.append(row)
    return rows

# ── Web操作 ────────────────────────────────
def _kb_login(driver):
    config = load_scraping_config()
    login_url = config["urls"]["kb_login_url"]
    username, password = _get_login_credentials("kb")
    wait = WebDriverWait(driver, 15)
    driver.get(login_url)
    wait.until(EC.presence_of_element_located((By.NAME, "login_id"))).send_keys(username)
    wait.until(EC.presence_of_element_located((By.NAME, "pswd"))).send_keys(password)
    wait.until(EC.element_to_be_clickable((By.XPATH, "//input[@type='submit']"))).click()
    wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
    time.sleep(1)

def _kb_get_driver():
    ua = random.choice(USER_AGENTS)
    options = Options()
    # options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument(f"--user-agent={ua}")
    options.add_argument("--lang=ja")
    options.set_capability("pageLoadStrategy", "eager")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.set_page_load_timeout(30)
    return driver

def _open_cyokyo_page(driver, kb_raceid) -> str | None:
    """調教ページに直アクセスしてHTML取得"""
    config = load_scraping_config()
    cyokyo_base_url = config["urls"]["kb_cyokyo_base_url"]
    url = f"{cyokyo_base_url}/{kb_raceid}"
    wait = WebDriverWait(driver, 20)
    try:
        driver.get(url)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table.cyokyodata")))
        return driver.page_source
    except TimeoutException:
        print(f"[WARN] cyokyo table not found: {kb_raceid}")
        return None

# ── メインスクレイピング関数 ───────────────────────
def scrape_kb_training(kb_raceid_list: list[str], target_raceid_list: list[str]) -> pd.DataFrame:
    """
    kb_raceid と target_raceid を1:1対応させ、
    各レースの調教ページを取得し、全頭×追切本数分行を生成してDataFrameで返す関数
    """
    # Step 1: ブラウザ起動 & ログイン
    all_rows = []
    driver = _kb_get_driver()
    try:
        _kb_login(driver)
        for kb_id, target_raceid in zip(kb_raceid_list, target_raceid_list):
            try:
                # Step 2: 調教ページ取得 & 1レース分をパース
                html = _open_cyokyo_page(driver, kb_id)
                if not html:
                    print(f"[WARN] training page not found (cyokyo table missing): {kb_id}")
                    continue

                rows = parse_block_rows(html, target_raceid=target_raceid)
                if not rows:
                    print(f"[WARN] parse 0 rows: kb_id={kb_id}, target_raceid={target_raceid}")
                    continue

                # Step 3: DataFrame整形と列の揃え
                df_one = pd.DataFrame(rows)
                oikiri_cols = OIKIRI_COLS
                need = ['target_horseid'] + oikiri_cols + ['row_kind']
                for c in need:
                    if c not in df_one.columns:
                        df_one[c] = ''
                df_one = df_one[need]

                df_one['training_comment'] = df_one[oikiri_cols].apply(
                    lambda s: ' '.join([str(v).strip() for v in s if str(v).strip()]),
                    axis=1
                )
                df_one['回り位置'] = df_one['回り位置'].str.replace(r'[\[\]［］]', '', regex=True).str.strip()

                all_rows.append(df_one)
                time.sleep(1 + random.random() * 0.8)
            except Exception as e:
                print(f"[ERR] {kb_id}: {e}\n{traceback.format_exc()}")
                continue
    finally:
        try:
            driver.quit()
        except Exception:
            pass

    if not all_rows:
        cols = ['target_horseid'] + OIKIRI_COLS + ['row_kind','training_comment']
        return pd.DataFrame(columns=cols)

    df = pd.concat(all_rows, ignore_index=True)
    return df
