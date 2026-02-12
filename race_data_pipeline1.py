import os
import re
import time
import random
import traceback
import json
import unicodedata
import pyparsing as pp
import pandas as pd

from io import StringIO
from pathlib import Path
from tqdm import tqdm
from typing import List, Optional, Dict, Any
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException, WebDriverException
from webdriver_manager.chrome import ChromeDriverManager

# ───────────────────────────────────────────────────
# ★ User-Agentリスト ★
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

DEFAULT_CONFIG: Dict[str, Any] = {
    "urls": {
        "kb_login_url": "https://p.keibabook.co.jp/login/login",
        "kb_nittei_base_url": "https://p.keibabook.co.jp/cyuou/nittei",
        "kb_seiseki_base_url": "https://p.keibabook.co.jp/cyuou/seiseki",
        "nk_login_url": "https://regist.netkeiba.com/account/?pid=login&action=auth",
    },
    "output": {
        "import_base_path": "D:/Keiba/00_ImportData"
    },
}

_CREDENTIALS_LOADED = False

# 実行フロー（読み順）
# 1) scrape_nk_kaisai_date -> 2) scrape_nk_raceid_list -> 3) scrape_nk_racedata
# 4) scrape_kb_kaisai_date -> 5) scrape_kb_raceid_list -> 6) scrape_kb_racedata


# ============================================================
# 設定・認証
# ============================================================
def load_scraping_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    外部設定(JSON)を読み込む。未設定項目は DEFAULT_CONFIG を使う。
    """
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


def get_output_base_path() -> Path:
    config = load_scraping_config()
    return Path(config["output"]["import_base_path"])


def _load_credentials_from_file() -> None:
    """
    .env 形式の認証ファイルを読み込み、未設定の環境変数を補完する。
    """
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
def _normalize_label(text: Any) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    return re.sub(r"\s+", "", text)


def _normalize_result_table_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_normalize_label(col) for col in df.columns]
    return df


def _extract_kb_result_rows(race_result_table: Any, race_id: str, target_raceid: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    # Step 1: HTMLテーブルをDataFrame化し、列名の揺れを正規化
    table_df = pd.read_html(StringIO(str(race_result_table)))[0]
    table_df = _normalize_result_table_columns(table_df)

    # Step 2: 必須列を確認（不足時は WARN を出してスキップ）
    required_columns = ["馬番", "馬名", "4角位置", "前半3F"]
    missing_columns = [col for col in required_columns if col not in table_df.columns]
    if missing_columns:
        print(f"[WARN] race_id={race_id} missing columns={missing_columns}, columns={list(table_df.columns)}")
        return rows

    # Step 3: 同一行から馬単位データを同時に作る（馬間ズレ防止）
    for _, row in table_df.iterrows():
        horse_number_raw = str(row.get("馬番", "")).strip()
        horse_number_digits = re.sub(r"\D", "", horse_number_raw)
        if not horse_number_digits:
            continue

        horse_number = horse_number_digits.zfill(2)[-2:]
        horse_name = str(row.get("馬名", "")).strip() or "N/A"
        corner_position = str(row.get("4角位置", "")).strip() or "N/A"
        zen3_time = str(row.get("前半3F", "")).strip() or "N/A"

        rows.append({
            "target_horseid": f"{target_raceid}{horse_number}",
            "horse_name": horse_name,
            "spurt_position": corner_position,
            "zen3_time": zen3_time,
        })
    return rows
# ───────────────────────────────────────────────────
# ============================================================
# netkeiba スクレイピング
# ============================================================
def scrape_nk_kaisai_date(from_:str, to_:str):
    """
    from_とto_をyyyy-mmの形で指定すると、間の開催日を取得する関数
    """
    # Step 1: 月ごとに netkeiba カレンダーへアクセス
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
        # Step 2: カレンダー内の開催日リンクから YYYYMMDD を抽出
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
    # Step 1: Selenium ブラウザを準備
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

    nk_raceid_list = []
    target_raceid_list = []

    # Step 2: 開催日ごとに race_id を収集し、Target16桁IDを生成
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
                nk_raceid_list.append(nk)
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
    return nk_raceid_list, target_raceid_list

def scrape_nk_racedata(race_id_list: List[str], target_raceid_list: List[str]):

    """
        netkeibaから取得したレースidリストを参照し、各レース情報ページへアクセスする。
        馬番付きのTarget取り込み用のレースidを生成し、分析コメント・馬場コメント・馬場指数をリストとして返す関数
    """

    # Step 1: ログイン情報を読み込み
    config = load_scraping_config()
    login_url = config["urls"]["nk_login_url"]
    username, password = _get_login_credentials("nk")

    # Step 2: Selenium ブラウザを起動
    options = Options()
    #options.add_argument("--headless")  # ヘッドレスモードで実行
    driver_path = ChromeDriverManager().install()

    # Chromeドライバーの初期化
    with webdriver.Chrome(service=Service(driver_path), options=options) as driver:

        # WebDriverWaitの初期化
        wait = WebDriverWait(driver, 10)  # 最大10秒待機

        # Step 3: 馬単位/レース単位データを格納するリストを初期化
        target_horseid_list = []  # Target取り込み用のレースid
        horse_names = []  # netkeiba:馬名/Target:馬名
        after_comments = []  # netkeiba:分析コメント/Target:結果コメント
        race_comments = []  # netkeiba:馬場コメント/Target:レースコメント
        baba_condition = []  # netkeiba:馬場指数/Target::レース印

        # 取得する最大回数を設定
        #max_iterations = 5

        # 馬場指数に基づくラベル付け関数
        def get_baba_label(baba_shisuu):
            baba_value = int(baba_shisuu)
            if baba_value == 0:
                return f"OG{baba_value}"
            elif baba_value < -30:
                return f"OK{baba_value}"
            elif baba_value < -20:
                return f"OJ{baba_value}"
            elif baba_value < -10:
                return f"OI{baba_value}"
            elif baba_value < 0:
                return f"OH{baba_value}"
            elif baba_value > 50:
                return f"OP{baba_value}"
            elif baba_value > 30:
                return f"OO{baba_value}"
            elif baba_value > 20:
                return f"ON{baba_value}"
            elif baba_value > 10:
                return f"OM{baba_value}"
            else:
                return f"OL{baba_value}"
            
        # ログインページにアクセス
        driver.get(login_url)
        
        # ログイン情報を入力
        wait.until(EC.presence_of_element_located((By.NAME, "login_id"))).send_keys(username)
        wait.until(EC.presence_of_element_located((By.NAME, "pswd"))).send_keys(password)

        # ログインボタンをクリック
        wait.until(EC.element_to_be_clickable((By.XPATH, "//input[@type='image']"))).click()
        
        # Step 4: レースページを巡回して、同一行の情報を同時に蓄積
        for i, race_id in enumerate(tqdm(race_id_list)):
            
            #取得回数が最大値になったら処理を終了
            #if i >= max_iterations:
                #break
            
            data_url = f"https://db.netkeiba.com/race/{race_id}"
            
            driver.get(data_url)
            
            # ページのhtmlソースを取得
            wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))  # ページが完全にロードされるまで待機
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            
            # レース結果テーブルから馬番と備考を取得
            race_result_table = soup.find('table', {'summary': 'レース結果'})
            rows = race_result_table.find_all('tr', class_=lambda x: x != 'txt_c even') if race_result_table else []
            
            # 分析コメントの取得
            analysis_table = soup.find('table', {'summary': 'レース分析'})
            analysis_comment = analysis_table.find('td').text.strip() if analysis_table else 'N/A'

            # 馬場指数の取得とラベル付け
            baba_shisuu_table = soup.find('table', {'summary': '馬場情報'})

            # 'td' 要素が存在するか確認し、なければ 'N/A' を設定
            if baba_shisuu_table:
                baba_shisuu_td = baba_shisuu_table.find('td')
                baba_shisuu_text = baba_shisuu_td.text if baba_shisuu_td else 'N/A'
            else:
                baba_shisuu_text = 'N/A'

            # baba_shisuu_textがNoneや'N/A'でないか確認してから正規表現を適用
            if baba_shisuu_text != 'N/A' and re.search(r'-?\d+', baba_shisuu_text):
                baba_shisuu_match = re.search(r'-?\d+', baba_shisuu_text)
                baba_shisuu = baba_shisuu_match.group() if baba_shisuu_match else 'N/A'
            else:
                baba_shisuu = 'N/A'

            # 取得した馬場指数に基づいてラベルを付ける
            baba_condition.append(get_baba_label(baba_shisuu) if baba_shisuu != 'N/A' else 'N/A')

            # 馬場コメントの取得
            baba_comment_table = soup.find('table', {'summary': '馬場情報'})

            # 'td' 要素が2つ以上あるか確認し、なければ 'N/A' を設定
            if baba_comment_table:
                td_elements = baba_comment_table.find_all('td')
                baba_comment = td_elements[1].text.strip() if len(td_elements) > 1 else 'N/A'
            else:
                baba_comment = 'N/A'

            race_comments.append(baba_comment)
            
            for row in rows:
                cells = row.find_all('td')
                
                # 馬番のセルが存在するか確認
                if len(cells) >= 3:
                    horse_number = cells[2].text.strip().zfill(2)
                    
                    # Target取り込み用のレースidを生成する
                    race_id_with_horse = target_raceid_list[i] + horse_number
                    target_horseid_list.append(race_id_with_horse)
                    
                    # 馬名のセルが存在するか確認
                    if len(cells) >= 4:
                        horse_name = cells[3].text.strip()
                        horse_names.append(horse_name)
                    else:
                        horse_names.append('N/A')
                    
                    # 備考のセルが存在するか確認
                    if len(cells) >= 8:
                        
                        # 最後から4番目の列の備考欄を指定する
                        remark = cells[-4].text.strip()
                        
                        # 備考欄の中身が存在する場合のみスラッシュで結合する
                        if remark:
                            after_comments.append(remark + "/" + analysis_comment)
                        else:
                            after_comments.append(analysis_comment)  # 備考欄が空文字ならそのまま分析コメントを追加
                            
                    else:
                        print(f"Skipped row due to insufficient cells in race_id: {race_id}")
            time.sleep(2)

        horse_df = pd.DataFrame({
            "target_horseid": target_horseid_list,
            "馬名": horse_names,
            "結果コメント": after_comments,
        })
        
        horse_df["target_raceid"] = [
            comment_id[:-2] for comment_id in target_horseid_list
        ]

        race_df = pd.DataFrame({
            "target_raceid": target_raceid_list,
            "レースコメント": race_comments,
            "馬場指数": baba_condition,
        })

    nk_racedf = horse_df.merge(race_df, on="target_raceid", how="left")
    
    return nk_racedf

# --- KB場コード -> Target(JV)場コード（中央） ---
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
    nittei_base_url = config["urls"]["kb_nittei_base_url"]
    username, password = _get_login_credentials("kb")

    # Step 2: ログインして、対象月の開催日リンクを収集
    with webdriver.Chrome(service=Service(driver_path), options=options) as driver:
        wait = WebDriverWait(driver, 15)
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

def kb_to_target_raceid(day: str, kb_raceid: str) -> str:
    """
    Target新仕様(16桁)をKB情報だけで生成する:
      yyyymmdd + pp(場2) + kk(回次2) + nn(日次2) + rr(R2)
    """
    # Step 1: 入力値をバリデーション
    kb = (kb_raceid or "").strip()
    if (not kb.isdigit()) or (len(kb) != 12):
        raise ValueError(f"bad kb_raceid: {kb_raceid!r}")

    if (not day.isdigit()) or (len(day) != 8):
        raise ValueError(f"bad day(YYYYMMDD): {day!r}")

    # Step 2: KB12桁IDを分解して Target16桁IDへ変換
    kaiji   = kb[4:6]
    kb_pl   = kb[6:8]
    nichiji = kb[8:10]
    raceno  = kb[10:12]

    jv_pl = KB_TO_JV_PLACE.get(kb_pl)
    if not jv_pl:
        raise ValueError(f"unknown KB place code: {kb_pl} (kb_raceid={kb_raceid})")

    return f"{day}{jv_pl}{kaiji}{nichiji}{raceno}"


def scrape_kb_raceid_list(kb_kaisai_date_list):
    """
    ログイン済み Selenium で競馬ブックの各開催日ページを開き、
    (kb_raceid(12桁), target_raceid(16桁)) を同じ順序で作る。

    重要:
    - ページ順を維持（sorted(set(...)) は使わない）
    - 同一dayページ内の重複だけ順序維持で除外
    - 返り値は 2つのリスト（呼び出し側で受け取る必要あり）
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

    kb_raceid_list = []
    kb_target_raceid_list = []

    config = load_scraping_config()
    login_url = config["urls"]["kb_login_url"]
    nittei_base_url = config["urls"]["kb_nittei_base_url"]
    username, password = _get_login_credentials("kb")

    with webdriver.Chrome(service=Service(driver_path), options=options) as driver:
        wait = WebDriverWait(driver, 15)

        # Step 2: ログイン
        driver.get(login_url)
        wait.until(EC.presence_of_element_located((By.NAME, "login_id"))).send_keys(username)
        wait.until(EC.presence_of_element_located((By.NAME, "pswd"))).send_keys(password)
        wait.until(EC.element_to_be_clickable((By.XPATH, "//input[@type='submit']"))).click()
        time.sleep(1)

        # Step 3: 開催日ごとに kb_raceid / target_raceid を収集
        for day in tqdm(kb_kaisai_date_list, desc="KB raceid取得"):
            url = f"{nittei_base_url}/{day}"
            try:
                driver.get(url)
                wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))

                # mainコンテンツに限定（旧レイアウトも少しフォールバック）
                try:
                    root = WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, "#flex_container_top > div.main"))
                    )
                except TimeoutException:
                    root = driver.find_element(By.CSS_SELECTOR, "div.main")

                selector = "a[href*='/cyuou/seiseki/'], a[href*='/cyuou/syutuba/']"
                anchors = root.find_elements(By.CSS_SELECTOR, selector)

                # ページで出てきた順にkb_raceidを溜める
                tmp_ids = []
                for a in anchors:
                    href = (a.get_attribute("href") or "").split("?", 1)[0]
                    m = re.search(r"/cyuou/(?:seiseki|syutuba)/(\d{12})", href)
                    if m:
                        tmp_ids.append(m.group(1))

                # ★順序維持で重複排除（同一day内）
                seen = set()
                for kb_rid in tmp_ids:
                    if kb_rid in seen:
                        continue
                    seen.add(kb_rid)

                    kb_raceid_list.append(kb_rid)
                    kb_target_raceid_list.append(kb_to_target_raceid(day, kb_rid))

                time.sleep(random.uniform(0.6, 1.2))

            except Exception as e:
                print(f"[WARN] {day} の取得でエラー: {e.__class__.__name__}: {e}")
                traceback.print_exc()
                continue

    # 整合性チェック
    if len(kb_raceid_list) != len(kb_target_raceid_list):
        raise RuntimeError("kb_raceid_list と kb_target_raceid_list の件数が一致しません")

    return kb_raceid_list, kb_target_raceid_list

# ============================================================
# 文字列パース / DataFrame整形
# ============================================================
def parse_horse_status(status_str: str) -> pd.DataFrame:
    
    """
    不利情報の文字列をパースしてDataFrameを返す関数
    """
    # Step 1: 発走状況他のテキストをパースするルールを定義
    horse_no = pp.Suppress('(') + pp.Word(pp.nums) + pp.Suppress(')')
    horse_nos = pp.OneOrMore(horse_no).setResultsName('horse_nos')
    status = pp.Regex(r'.*?(?=(\(\d+\))|$)', re.DOTALL).setResultsName('発走状況他')
    entry = pp.Group(horse_nos + status)
    parser = pp.OneOrMore(entry)

    # Step 2: 文字列をパースして (馬番, ステータス) の対応を作る
    parsed = parser.parseString(status_str)

    # Step 3: DataFrame 化
    data_list = []
    for entry in parsed:
        nos = entry['horse_nos']
        status_text = entry['発走状況他'].strip()
        for no in nos:
            data_list.append({'horse_no': no, '発走状況他': status_text})

    return pd.DataFrame(data_list)

def generate_status_dataframe(target_raceid_list, kb_horse_status) -> pd.DataFrame:
    
    """
    レースIDと不利情報リストを使ってDataFrameを生成する関数
    """
    # Step 1: race単位のステータス文字列を horse単位へ展開
    all_data = pd.DataFrame()

    for race_id, state_info in zip(target_raceid_list, kb_horse_status):
        state_text = (state_info or "").strip()
        if not state_text:
            continue  # 空はスキップ
        if state_text.upper() in {"N/A", "NA", "-"}:
            continue  # 欠損値はスキップ
        try:
            parsed_data = parse_horse_status(state_text)
        except pp.ParseException:
            print(f"[WARN] status parse skipped. race_id={race_id}, text={state_text[:80]}")
            continue
        parsed_data['target_horseid'] = parsed_data['horse_no'].apply(
            lambda x: f"{race_id}{int(x):02d}"
        )
        parsed_data = parsed_data[['target_horseid', '発走状況他']]
        all_data = pd.concat([all_data, parsed_data], ignore_index=True)

    return all_data

# タイム差定義
COLUMNS = ['先頭差', 'horse_no', 'サイドポジション']
DIFF_GROUP = 0.3
DIFF_MIN   = 1.5
DIFF_MID   = 3.0
DIFF_MUCH  = 6.0

class ParsePass:
    def __init__(self):
        horse_no = pp.Word(pp.nums).setParseAction(self._horse_no_action)

        group = (pp.Suppress('(') 
        + pp.Optional(pp.delimitedList(pp.Word(pp.nums), delim='.')) 
        + pp.Suppress(')'))
        group.ignore('*')
        group.setParseAction(self._group_action)

        status_no = pp.Combine(pp.oneOf('落 止') + pp.Optional(pp.Word(pp.nums))).suppress()

        element = (group | status_no | horse_no)

        diff_min  = pp.Suppress(pp.Optional(pp.Literal('.'))).setParseAction(self._diff_min_action) + element
        diff_mid  = pp.Suppress(pp.Literal('-')).setParseAction(self._diff_mid_action) + element
        diff_much = pp.Suppress(pp.Literal('=')).setParseAction(self._diff_much_action) + element

        self._passing_order = element + pp.ZeroOrMore(diff_mid | diff_much | diff_min)

    def _horse_no_action(self, token):
        df_append = pd.DataFrame([[round(self._diff, 1), token[0], 1]], columns=COLUMNS)
        self._data = (pd.concat([self._data, df_append], ignore_index=True, axis=0)
                        .drop_duplicates().reset_index(drop=True))
        return

    def _group_action(self, token):
        for i, no in enumerate(token):
            df_append = pd.DataFrame([[round(self._diff, 1), no, 1+i]], columns=COLUMNS)
            self._data = (pd.concat([self._data, df_append], ignore_index=True, axis=0)
                            .drop_duplicates().reset_index(drop=True))
            self._diff += DIFF_GROUP
        self._diff -= DIFF_GROUP
        return

    def _diff_min_action(self, token):  self._diff += DIFF_MIN
    def _diff_mid_action(self, token):  self._diff += DIFF_MID
    def _diff_much_action(self, token): self._diff += DIFF_MUCH

    def parse(self, pass_str: str) -> pd.DataFrame:
        self._data = pd.DataFrame(columns=COLUMNS)
        self._diff = 0
        self._passing_order.parseString(pass_str)
        self._data.index = pd.RangeIndex(1, len(self._data)+1)
        self._data.index.name = '通過順位'
        return self._data

def generate_corner_dataframe(target_raceid_list, corners_list, corner_name) -> pd.DataFrame:
    """
    レースIDとコーナー通過順リストを使ってDataFrameを生成する関数
    返す列: ['target_horseid', '通過順位', '先頭差', 'サイドポジション', 'コーナー']
    """
    # Step 1: コーナー通過順文字列を horse単位に分解
    parser = ParsePass()
    all_data = pd.DataFrame()

    for race_id, corner in zip(target_raceid_list, corners_list):
        if corner and corner != 'N/A' and corner.strip() != '':
            try:
                parsed = parser.parse(corner)
                # rank を列に出す
                parsed = parsed.reset_index()  # rank 列を作る
                # race + 馬番 → target_horseid
                parsed['target_horseid'] = parsed['horse_no'].apply(
                    lambda x: f"{race_id}{int(x):02d}"
                )
                # 列整形（horse_noは監査用途で残すなら残してOK。最小なら外す）
                parsed = parsed[['target_horseid', '通過順位', '先頭差', 'サイドポジション']]
                # corner_name に応じて 'サイドポジション' をリネーム
                if corner_name == 'first_corner':
                    parsed = parsed.rename(columns={'サイドポジション': '初角サイドポジション'})
                elif corner_name == 'second_corner':
                    parsed = parsed.rename(columns={'サイドポジション': '2角サイドポジション'})
                elif corner_name == 'third_corner':
                    parsed = parsed.rename(columns={'サイドポジション': '3角サイドポジション'})
                elif corner_name == 'fourth_corner':
                    parsed = parsed.rename(columns={'サイドポジション': '4角サイドポジション'})
                parsed['コーナー'] = corner_name  # どのコーナーか識別
                all_data = pd.concat([all_data, parsed], ignore_index=True)
            except pp.ParseException as pe:
                print(f"[ParseError] race_id={race_id}, corner={corner_name}, text='{corner}' -> {pe}")
                continue

    return all_data


def scrape_kb_racedata(kb_raceid_list: List[str], target_raceid_list: List[str]):
    """
    競馬ブックのレースidリストから、Target向けデータを作成する。
    """
    # Step 1: ログイン設定の準備
    config = load_scraping_config()
    login_url = config["urls"]["kb_login_url"]
    kb_seiseki_base_url = config["urls"]["kb_seiseki_base_url"]
    username, password = _get_login_credentials("kb")

    options = Options()
    ua = random.choice(USER_AGENTS)
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument(f"--user-agent={ua}")
    options.add_argument("--lang=ja")
    options.set_capability("pageLoadStrategy", "eager")

    driver_path = ChromeDriverManager().install()

    # Step 2: ログインしてレースページを巡回
    with webdriver.Chrome(service=Service(driver_path), options=options) as driver:
        wait = WebDriverWait(driver, 10)
        driver.get(login_url)
        wait.until(EC.presence_of_element_located((By.NAME, "login_id"))).send_keys(username)
        wait.until(EC.presence_of_element_located((By.NAME, "pswd"))).send_keys(password)
        wait.until(EC.element_to_be_clickable((By.XPATH, "//input[@type='submit']"))).click()
        time.sleep(2)

        # Step 3: 収集先リストを初期化（馬単位/レース単位）
        target_kb_commentid_list = []
        horse_names_kb = []
        spurt_positions = []
        zen3_times = []
        horse_status = []
        first_corners = []
        second_corners = []
        third_corners = []
        fourth_corners = []
        interview_data = []
        next_comment_data = []

        for i, race_id in enumerate(tqdm(kb_raceid_list)):
            if i >= len(target_raceid_list):
                print(f"[WARN] race_id={race_id} skipped because target_raceid_list is shorter than kb_raceid_list")
                break

            target_raceid = target_raceid_list[i]
            first_corner = "N/A"
            second_corner = "N/A"
            third_corner = "N/A"
            fourth_corner = "N/A"
            horse_state = "N/A"

            try:
                kb_data_url = f"{kb_seiseki_base_url}/{race_id}"
                driver.get(kb_data_url)
                wait.until(EC.presence_of_element_located((By.CLASS_NAME, "seiseki")))
                soup = BeautifulSoup(driver.page_source, "html.parser")

                # Step 4-1: 成績表（馬単位）を列名ベースで取得
                race_result_table = soup.find("table", {"class": "seiseki"})
                if not race_result_table:
                    print(f"[WARN] race_result_table not found. race_id={race_id}")
                else:
                    horse_rows = _extract_kb_result_rows(race_result_table, race_id, target_raceid)
                    for horse_row in horse_rows:
                        target_kb_commentid_list.append(horse_row["target_horseid"])
                        horse_names_kb.append(horse_row["horse_name"])
                        spurt_positions.append(horse_row["spurt_position"])
                        zen3_times.append(horse_row["zen3_time"])

                # Step 4-2: コーナー通過順・発走状況他（レース単位）を取得
                etcresulttables = soup.find_all("table", {"class": "default seiseki-etc"})
                if len(etcresulttables) > 0:
                    corner_rows = etcresulttables[0].find_all("tr")
                    first_corner = corner_rows[0].find("td").text.strip() if len(corner_rows) > 0 and corner_rows[0].find("td") else "N/A"
                    second_corner = corner_rows[1].find("td").text.strip() if len(corner_rows) > 1 and corner_rows[1].find("td") else "N/A"
                    third_corner = corner_rows[2].find("td").text.strip() if len(corner_rows) > 2 and corner_rows[2].find("td") else "N/A"
                    fourth_corner = corner_rows[3].find("td").text.strip() if len(corner_rows) > 3 and corner_rows[3].find("td") else "N/A"
                else:
                    print(f"[WARN] corner table not found. race_id={race_id}")

                if len(etcresulttables) > 1:
                    race_etcresult_table = etcresulttables[1]
                    for th in race_etcresult_table.find_all("th"):
                        if _normalize_label(th.get_text()) == "発走状況他":
                            etc_td = th.find_next("td")
                            horse_state = etc_td.text.strip() if etc_td and etc_td.text.strip() else "N/A"
                            break
                else:
                    print(f"[WARN] race etc table not found. race_id={race_id}")

                # Step 4-3: インタビュー関連コメントを取得
                race_borderboxes = soup.find_all("div", class_="borderbox")
                if race_borderboxes:
                    interview_box = race_borderboxes[0]
                    interviewrows = interview_box.find_all("p", class_="honbun")
                    for interviewrow in interviewrows:
                        interview_comment = interviewrow.text.strip().replace("\u3000", " ")
                        interview_data.append([target_raceid, interview_comment])

                if len(race_borderboxes) > 1:
                    next_race_box = race_borderboxes[1]
                    next_race_rows = next_race_box.find_all("p", class_="honbun")
                    for next_race_row in next_race_rows:
                        next_race_comment = next_race_row.text.strip().replace("\u3000", " ")
                        next_comment_data.append([target_raceid, next_race_comment])

                time.sleep(1)

            except Exception as e:
                print(f"[WARN] Error scraping race ID {race_id}: {e}")
                print(traceback.format_exc())
            finally:
                # レース単位データは必ず1レース=1要素で追加して整合性を守る
                first_corners.append(first_corner)
                second_corners.append(second_corner)
                third_corners.append(third_corner)
                fourth_corners.append(fourth_corner)
                horse_status.append(horse_state)

        # Step 5: DataFrameに整形
        result_df = pd.DataFrame({
            "target_horseid": target_kb_commentid_list,
            "馬名": horse_names_kb,
            "4角位置": spurt_positions,
            "前半3F": zen3_times
        })
        interview_df = pd.DataFrame(interview_data, columns=["target_raceid", "KOL関係者コメント"])
        next_comments_df = pd.DataFrame(next_comment_data, columns=["target_raceid", "KOL次走へのメモ"])
        kb_status_df = generate_status_dataframe(target_raceid_list, horse_status)
        kb_corner_df1 = generate_corner_dataframe(target_raceid_list, first_corners, "first_corner")
        kb_corner_df2 = generate_corner_dataframe(target_raceid_list, second_corners, "second_corner")
        kb_corner_df3 = generate_corner_dataframe(target_raceid_list, third_corners, "third_corner")
        kb_corner_df4 = generate_corner_dataframe(target_raceid_list, fourth_corners, "fourth_corner")

        if not interview_df.empty:
            interview_df["馬名"] = (
                interview_df["KOL関係者コメント"]
                .astype(str)
                .str.replace("\u3000", " ")
                .str.split(r"[：:]", n=1)
                .str[0]
                .str.strip()
            )
            interview_df["馬名"] = interview_df["馬名"].str.replace(r"\s+", " ", regex=True)
        else:
            interview_df["馬名"] = pd.Series(dtype="object")

        if not next_comments_df.empty:
            next_comments_df["馬名"] = (
                next_comments_df["KOL次走へのメモ"]
                .astype(str)
                .str.replace("\u3000", " ")
                .str.split(r"[：:]", n=1)
                .str[0]
                .str.strip()
            )
            next_comments_df["馬名"] = next_comments_df["馬名"].str.replace(r"\s+", " ", regex=True)
        else:
            next_comments_df["馬名"] = pd.Series(dtype="object")

    return result_df, kb_status_df, kb_corner_df1, kb_corner_df2, kb_corner_df3, kb_corner_df4, interview_df, next_comments_df

# ============================================================
# 旧実装（比較・退避用）
# ============================================================
def _scrape_kb_racedata_legacy(kb_raceid_list: List[str],target_raceid_list: List[str]):

    """
    競馬ブックのレースidリストを参照し、各レース情報ページへアクセスする。
    """
#scrape_kb_racedata(kb_raceid_list: List[str],target_raceid_list: List[str]):
#def scrape_kb_racedata(kb_raceid_list: List[str]):
    # ログイン情報
    config = load_scraping_config()
    login_url = config["urls"]["kb_login_url"]
    kb_seiseki_base_url = config["urls"]["kb_seiseki_base_url"]
    username, password = _get_login_credentials("kb")

    # Chromeドライバーのオプション設定
    options = Options()
    ua = random.choice(USER_AGENTS)
   
    #options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument(f"--user-agent={ua}")
    options.add_argument("--lang=ja")
    
    # ドライバのパスを取得
    driver_path = ChromeDriverManager().install()
    
    caps = DesiredCapabilities.CHROME.copy()
    caps["pageLoadStrategy"] = "eager"  # ページの読み込み戦略を設定
    # pageLoadStrategy を eager に
    options.set_capability("pageLoadStrategy", "eager")
    
    # Chromeドライバーの初期化
    with webdriver.Chrome(service=Service(driver_path), options=options) as driver:
        # WebDriverWaitの初期化
        wait = WebDriverWait(driver, 10)

        # ログインページにアクセス
        driver.get(login_url)
        wait.until(EC.presence_of_element_located((By.NAME, "login_id"))).send_keys(username)
        wait.until(EC.presence_of_element_located((By.NAME, "pswd"))).send_keys(password)
        wait.until(EC.element_to_be_clickable((By.XPATH, "//input[@type='submit']"))).click()
        time.sleep(2)
        
        # 取り込み用のコメント・指数のリスト
        target_kb_commentid_list = [] #競馬ブック:馬番/Target:取り込み用レースid
        horse_names_kb = []  # 競馬ブック:馬名/Target:馬名
        spurt_positions = []  # 競馬ブック:4角位置/Target:4角位置
        zen3_times = []  # 競馬ブック:前半3F/Target:前半3F
        horse_status = [] #競馬ブック:発走状況他/Target:結果コメント
        first_corners = [] #競馬ブック:1コーナー(向正面)/Target:初角サイドポジション
        second_corners = [] #競馬ブック:2コーナー/Target:2角サイドポジション
        third_corners = [] #競馬ブック:3コーナー/Target:3角サイドポジション
        fourth_corners = [] #競馬ブック:4コーナー/Target:4角サイドポジション
        interview_comments = []  # 競馬ブック:インタビュー/Target:KOL関係者コメント
        next_comments = [] #競馬ブック:次走へのメモ/Target:KOL次走へのメモ
        interview_data = []
        next_comment_data = []

        for i, race_id in enumerate(tqdm(kb_raceid_list)):
            #if i >= max_iterations:
                #break

            # レース結果ページのURLを生成
            kb_data_url = f"{kb_seiseki_base_url}/{race_id}"
            driver.get(kb_data_url)
            
            time.sleep(1)
            
            # ページのhtmlソースを取得
            try:
                wait.until(EC.presence_of_element_located((By.CLASS_NAME, "seiseki")))  # ページが完全にロードされるまで待機
                soup = BeautifulSoup(driver.page_source, 'html.parser')
                
                # レース結果テーブルから4角位置と前半3Fを取得
                race_result_table = soup.find('table', {'class': 'seiseki'})
                if not race_result_table:
                    print(f"raceresultable not found for race ID: {race_id}")
                    continue
                
                resultrows = race_result_table.find_all('tr', class_=lambda x: x != 'txt_c even') if race_result_table else []
                
                for row in resultrows:
                    cells = row.find_all('td')
                    if len(cells) > 14:
                        corner_position = cells[13].text.strip()
                        if corner_position == '大外':
                            corner_position = '大'
                        if corner_position == '最内':
                            corner_position = '最'                            
                        
                        horse_number = cells[4].text.strip().zfill(2)
                        horse_names = cells[5].text.strip()
                        zen3_time = cells[14].text.strip()

                        target_kb_commentid_list.append(target_raceid_list[i] + horse_number)
                        horse_names_kb.append(horse_names)
                        spurt_positions.append(corner_position)
                        zen3_times.append(zen3_time)
                
                #通過順のテーブルと発送状況他のテーブルを取得する
                etcresulttables = soup.find_all('table', {'class': 'default seiseki-etc'})
                
                #通過順を取得する
                corner_podition_table = etcresulttables[0]
                corner_rows = corner_podition_table.find_all('tr')
                
                first_corner = corner_rows[0].find('td').text.strip() if len(corner_rows) > 0 and corner_rows[0].find('td') else 'N/A'
                second_corner = corner_rows[1].find('td').text.strip() if len(corner_rows) > 1 and corner_rows[1].find('td') else 'N/A'
                third_corner = corner_rows[2].find('td').text.strip() if len(corner_rows) > 2 and corner_rows[2].find('td') else 'N/A'
                fourth_corner = corner_rows[3].find('td').text.strip() if len(corner_rows) > 3 and corner_rows[3].find('td') else 'N/A'
                
                first_corners.append(first_corner)
                second_corners.append(second_corner)
                third_corners.append(third_corner)
                fourth_corners.append(fourth_corner)
                
                #発送状況他の情報を取得
                race_etcresult_table = etcresulttables[1]
                etc_th = race_etcresult_table.find('th' ,text='発走状況他')
                if etc_th:
                    etc_td = etc_th.find_next('td')
                    horse_state = etc_td.text.strip() if etc_td else 'N/A'
                    horse_status.append(horse_state)
                
                # インタビューと次走へのメモを取得　'borderbox' クラスを持つ div を全て取得
                race_borderboxes = soup.find_all('div', class_='borderbox')

                # インタビューを取得
                if race_borderboxes:
                    # インタビュー部分のコメントを取得
                    interview_box = race_borderboxes[0]
                    interviewrows = interview_box.find_all('p', class_='honbun')
                    for interviewrow in interviewrows:
                        interview_comment = interviewrow.text.strip().replace('\u3000', ' ')
                        interview_data.append([target_raceid_list[i], interview_comment])

                # 次走へのメモ部分のコメントを取得
                if len(race_borderboxes) > 1:
                    next_race_box = race_borderboxes[1]
                    next_race_rows = next_race_box.find_all('p', class_='honbun')
                    for next_race_row in next_race_rows:
                        next_race_comment = next_race_row.text.strip().replace('\u3000', ' ')
                        next_comment_data.append([target_raceid_list[i], next_race_comment])
                    
                time.sleep(1)        
                        
            except Exception as e:
                print(f"Error scraping race ID {race_id}: {str(e)}")
                print(traceback.format_exc())
                continue
            
        # データフレームに変換
        result_df = pd.DataFrame({
            'target_horseid': target_kb_commentid_list,
            '馬名': horse_names_kb,
            '4角位置': spurt_positions,
            '前半3F': zen3_times
        })
        interview_df = pd.DataFrame(interview_data, columns=['target_raceid', 'KOL関係者コメント'])
        next_comments_df = pd.DataFrame(next_comment_data, columns=['target_raceid', 'KOL次走へのメモ'])
        kb_status_df = generate_status_dataframe(target_raceid_list, horse_status)
        kb_corner_df1 = generate_corner_dataframe(target_raceid_list, first_corners, 'first_corner')
        kb_corner_df2 = generate_corner_dataframe(target_raceid_list, second_corners, 'second_corner')
        kb_corner_df3 = generate_corner_dataframe(target_raceid_list, third_corners, 'third_corner')
        kb_corner_df4 = generate_corner_dataframe(target_raceid_list, fourth_corners, 'fourth_corner')

        # インタビューのデータフレームを加工
        interview_df['馬名'] = (
            interview_df['KOL関係者コメント']
            .astype(str)
            .str.replace('\u3000', ' ')
            .str.split(r'[（(]', n=1) 
            .str[0]
            .str.strip()
        )
        interview_df['馬名'] = interview_df['馬名'].str.replace(r"\s+", " ", regex=True)

        # 次走へのメモのデータフレームを加工
        next_comments_df['馬名'] = (
            next_comments_df['KOL次走へのメモ']
            .astype(str)
            .str.replace('\u3000', ' ')
            .str.split(r'[…]', n=1)
            .str[0]
            .str.strip()
        )
        next_comments_df['馬名'] = next_comments_df['馬名'].str.replace(r"\s+", " ", regex=True)

    
    return result_df, kb_status_df,kb_corner_df1, kb_corner_df2, kb_corner_df3, kb_corner_df4, interview_df, next_comments_df
