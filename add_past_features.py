# -*- coding: utf-8 -*-
"""
Result成績データから特徴量スナップショットを生成する。
- 原本は上書きしない
- リーク防止: 当該行・同日データは集計に含めない
- 出力は用途別スナップショット（asof 付与前提）
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# ====== パス ======
PROCESSED_DIR = Path(r"G:\マイドライブ\20_HOBBY\20_KEIBA\10_Data_Source\10_Export_Data\Result_Data\Processed")
FEATURES_DIR = Path(r"G:\マイドライブ\20_HOBBY\20_KEIBA\20_Data_Features")

# ====== 列名 ======
DATE_COL = "日付S"
RACE_COL = "target_raceid"
HORSE_COL = "血統登録番号"
STYLE_COL = "決め手"
BLINKER_COL = "ブリンカー"
TRACK_COL = "馬場状態"

STYLE_LABELS = ["逃げ", "先行", "差し", "ﾏｸﾘ", "追込"]

REQUIRED_COLS = [RACE_COL, "target_horseid", HORSE_COL, DATE_COL]

FINISH_COL = "入線順位"
EARNINGS_COL = "獲得賞金"

# ====== 4章 過去n走集計対象列 ======
TARGET_COLS = [
    "距離",
    "前走距離差",
    "体重",
    "増減",
    "斤量",
    "斤量馬体重比",
    "キャリア",
    "間隔",
    "レイティング",
    "レイティング順位",
    "レイティング平均値",
    "レイティング偏差値",
    "レイティング順位分布",
    "レイティング上位差分",
    "レイティング勝馬差分",
    "レイティング偏差値上位差分",
    "レイティング偏差値勝馬差分",
    "レイティング順位分布上位差分",
    "レイティング順位分布勝馬差分",
    "平均レイティング差分",
    "ZI指数",
    "ZI指数順位",
    "ZI指数偏差値",
    "ZI指数順位分布",
    "ZI指数上位差分",
    "ZI指数勝馬差分",
    "ZI指数偏差値上位差分",
    "ZI指数偏差値勝馬差分",
    "ZI指数順位分布上位差分",
    "ZI指数順位分布勝馬差分",
    "追切指数",
    "追切指数順位",
    "追切指数偏差値",
    "追切指数順位分布",
    "追切指数上位差分",
    "追切指数勝馬差分",
    "追切指数偏差値上位差分",
    "追切指数偏差値勝馬差分",
    "追切指数順位分布上位差分",
    "追切指数順位分布勝馬差分",
    "マイニング",
    "マイニング順位",
    "マイニング偏差値",
    "マイニング順位分布",
    "マイニング上位差分",
    "マイニング勝馬差分",
    "マイニング偏差値上位差分",
    "マイニング偏差値勝馬差分",
    "マイニング順位分布上位差分",
    "マイニング順位分布勝馬差分",
    "対戦型マイニング",
    "対戦型マイニング順位",
    "対戦型マイニング偏差値",
    "対戦型マイニング順位分布",
    "対戦型マイニング上位差分",
    "対戦型マイニング勝馬差分",
    "対戦型マイニング偏差値上位差分",
    "対戦型マイニング偏差値勝馬差分",
    "対戦型マイニング順位分布上位差分",
    "対戦型マイニング順位分布勝馬差分",
    "人気",
    "単勝オッズ",
    "複勝人気",
    "複勝下限",
    "複勝上限",
    "単勝票数",
    "複勝票数",
    "単複票数比",
    "単勝シェア",
    "複勝シェア",
    "連絡みシェア",
    "ワイド絡みシェア",
    "馬単絡みシェア",
    "馬単1着軸シェア",
    "馬単2着軸シェア",
    "3連複絡みシェア",
    "初角位置",
    "初角_4角差",
    "初角_4角差上位差分",
    "初角_4角差勝馬差分",
    "4角_入線順位差",
    "4角_入線順位差上位差分",
    "4角_入線順位差勝馬差分",
    "入線順位",
    "獲得賞金",
    "-3F差",
    "-3F差上位差分",
    "-3F差勝馬差分",
    "1着差タイム",
    "タイムS上位差分",
    "タイムS勝馬差分",
    "補正走破タイム上位差分",
    "補正走破タイム勝馬差分",
    "-3Fタイム上位差分",
    "-3Fタイム勝馬差分",
    "レース強度指数",
    "レース強度指数上位差分",
    "レースレベル指数",
    "レースPCI",
    "PCI",
    "PCI3差分",
    "PCI上位差分",
    "PCI勝馬差分",
    "前半3F",
    "前半3F順位",
    "前半3F偏差値",
    "前半3F順位分布",
    "前半3F上位差分",
    "前半3F勝馬差分",
    "前半3F偏差値上位差分",
    "前半3F偏差値勝馬差分",
    "前半3F順位上位差分",
    "前半3F順位勝馬差分",
    "前半3F順位分布上位差分",
    "前半3F順位分布勝馬差分",
    "上り3F",
    "上り3F順位",
    "上り3F偏差値",
    "上り3F順位分布",
    "上り3F上位差分",
    "上り3F勝馬差分",
    "上り3F偏差値上位差分",
    "上り3F偏差値勝馬差分",
    "上り3F順位上位差分",
    "上り3F順位勝馬差分",
    "上り3F順位分布上位差分",
    "上り3F順位分布勝馬差分",
    "Ave-3F",
    "Ave-3F順位",
    "Ave-3F偏差値",
    "Ave-3F順位分布",
    "Ave-3F上位差分",
    "Ave-3F勝馬差分",
    "Ave-3F偏差値上位差分",
    "Ave-3F偏差値勝馬差分",
    "Ave-3F順位上位差分",
    "Ave-3F順位勝馬差分",
    "Ave-3F順位分布上位差分",
    "Ave-3F順位分布勝馬差分",
    "テン指数",
    "テン指数順位",
    "テン指数偏差値",
    "テン指数順位分布",
    "テン指数上位差分",
    "テン指数勝馬差分",
    "テン指数偏差値上位差分",
    "テン指数偏差値勝馬差分",
    "テン指数順位上位差分",
    "テン指数順位勝馬差分",
    "テン指数順位分布上位差分",
    "テン指数順位分布勝馬差分",
    "上り指数",
    "上り指数順位",
    "上り指数偏差値",
    "上り指数順位分布",
    "上り指数上位差分",
    "上り指数勝馬差分",
    "上り指数偏差値上位差分",
    "上り指数偏差値勝馬差分",
    "上り指数順位上位差分",
    "上り指数順位勝馬差分",
    "上り指数順位分布上位差分",
    "上り指数順位分布勝馬差分",
    "スピード指数",
    "スピード指数順位",
    "スピード指数偏差値",
    "スピード指数順位分布",
    "スピード指数上位差分",
    "スピード指数勝馬差分",
    "スピード指数偏差値上位差分",
    "スピード指数偏差値勝馬差分",
    "スピード指数順位上位差分",
    "スピード指数順位勝馬差分",
    "スピード指数順位分布上位差分",
    "スピード指数順位分布勝馬差分",
    "補正タイム",
    "補9",
    "総合指数",
    "総合指数順位",
    "総合指数偏差値",
    "総合指数順位分布",
    "総合指数上位差分",
    "総合指数勝馬差分",
    "総合指数偏差値上位差分",
    "総合指数偏差値勝馬差分",
    "総合指数順位上位差分",
    "総合指数順位勝馬差分",
    "総合指数順位分布上位差分",
    "総合指数順位分布勝馬差分",
    "Top3総合指数",
    "Top3総合指数差分",
]

# ====== キー定義 ======
BLOOD_KEYS = [
    "種牡馬",
    "種牡馬タイプ名",
    "父×母の父タイプ名",
    "父タイプ名×母の父タイプ名",
]

RELATED_KEYS = [
    "生産者",
    "馬主",
    "調教師",
    "騎手",
    "生産者×馬主",
    "生産者×調教師",
    "生産者×騎手",
    "馬主×調教師",
    "馬主×騎手",
    "調教師×騎手",
]

COURSE_KEYS = [
    "調教師×コースラベル",
    "騎手×コースラベル",
]

BLINKER_REL_KEYS = ["調教師", "騎手", "調教師×騎手"]

KEY_DIRS = {
    "種牡馬": "Sire",
    "種牡馬タイプ名": "SireLine",
    "父×母の父タイプ名": "Sire_BMS",
    "父タイプ名×母の父タイプ名": "SireLine_BMS",
    "生産者": "Breeder",
    "馬主": "Owner",
    "調教師": "Trainer",
    "騎手": "Jockey",
    "生産者×馬主": "Breeder_Owner",
    "生産者×調教師": "Breeder_Trainer",
    "生産者×騎手": "Breeder_Jockey",
    "馬主×調教師": "Owner_Trainer",
    "馬主×騎手": "Owner_Jockey",
    "調教師×騎手": "Trainer_Jockey",
    "調教師×コースラベル": "Trainer_Course",
    "騎手×コースラベル": "Jockey_Course",
}


@dataclass
class RunConfig:
    input_path: Path | None
    surface_col: str | None
    surface_turf_value: str | None
    surface_dirt_value: str | None
    course_label_col: str | None


def progress(iterable: Iterable, **kwargs) -> Iterable:
    """tqdm があれば進捗表示、無ければ素の iterable を返す。"""
    try:
        from tqdm import tqdm

        return tqdm(iterable, **kwargs)
    except Exception:
        return iterable


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def find_latest_csv(directory: Path) -> Path | None:
    files = list(directory.glob("*.csv"))
    if not files:
        return None
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]


def validate_required_cols(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"必須列が不足しています: {missing}")


def normalize_date(df: pd.DataFrame) -> pd.DataFrame:
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    bad = df[DATE_COL].isna().sum()
    if bad:
        print(f"[warn] 日付Sの変換不可行が {bad} 行あります。除外します。")
        df = df[df[DATE_COL].notna()].copy()
    return df


def coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def make_combo_column(df: pd.DataFrame, left: str, right: str, new_col: str) -> None:
    if left in df.columns and right in df.columns:
        combined = df[left].astype(str) + "×" + df[right].astype(str)
        mask = df[left].isna() | df[right].isna()
        combined = combined.where(~mask, np.nan)
        df[new_col] = combined


def _chunks(seq: list[str], size: int) -> Iterable[list[str]]:
    """列が多いので、処理を分割してメモリ負荷を抑えるためのヘルパー"""
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _normalize_date_only(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _write_snapshot(
    df: pd.DataFrame,
    out_path: Path,
    key_cols: list[str],
    feature_cols: list[str],
) -> None:
    # asof 付与前提：日付を正規化し、(キー, 日付S) で1行に正規化
    out = df[key_cols + [DATE_COL] + feature_cols].copy()
    out[DATE_COL] = _normalize_date_only(out[DATE_COL])
    out = out.dropna(subset=[DATE_COL])
    out = out.sort_values(key_cols + [DATE_COL])
    out = out.drop_duplicates(subset=key_cols + [DATE_COL], keep="last")
    ensure_dir(out_path.parent)
    out.to_csv(out_path, index=False, encoding="cp932")
    print(f"[info] saved: {out_path}")


def add_past_numeric_features(
    df: pd.DataFrame,
    window: int = 5,
    batch_size: int = 50,
    keep_order: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """
    rollingベースで過去n走特徴量を作る（prev + last{window}のmin/max/mean/std）

    リーク防止の考え方：
    - 当該行を含めない → groupby.shift(1) を使う
    - 同日データを含めない → 「馬×同日」グループ先頭行で特徴量を作り、同日全行へ配布
    """
    present_cols = [c for c in TARGET_COLS if c in df.columns]
    missing = [c for c in TARGET_COLS if c not in df.columns]
    if missing:
        print(f"[warn] 過去n走集計の対象列が不足しています（スキップ）: {missing}")

    if not present_cols:
        print("[warn] 過去n走集計の対象列が無いため、該当処理をスキップします。")
        return df, []

    df_work = df.copy()

    if keep_order:
        df_work["_orig_idx"] = np.arange(len(df_work), dtype=np.int64)

    df_work["_date_key"] = _normalize_date_only(df_work[DATE_COL])

    coerce_numeric(df_work, present_cols)

    df_work = df_work.sort_values([HORSE_COL, "_date_key", RACE_COL]).reset_index(drop=True)

    start_mask = df_work.groupby([HORSE_COL, "_date_key"]).cumcount().eq(0)
    start_idx = df_work.index[start_mask]

    base = df_work.loc[start_idx, [HORSE_COL, "_date_key"]].copy()

    added_cols: list[str] = []

    for cols in progress(
        list(_chunks(present_cols, batch_size)),
        desc="past_numeric_chunks",
    ):
        shifted = df_work.groupby(HORSE_COL)[cols].shift(1)
        roll = shifted.groupby(df_work[HORSE_COL]).rolling(window=window, min_periods=1)

        prev_df = shifted.loc[start_idx].rename(columns=lambda c: f"{c}_prev")
        min_df = (
            roll.min()
            .reset_index(level=0, drop=True)
            .loc[start_idx]
            .rename(columns=lambda c: f"{c}_min_{window}")
        )
        max_df = (
            roll.max()
            .reset_index(level=0, drop=True)
            .loc[start_idx]
            .rename(columns=lambda c: f"{c}_max_{window}")
        )
        mean_df = (
            roll.mean()
            .reset_index(level=0, drop=True)
            .loc[start_idx]
            .rename(columns=lambda c: f"{c}_mean_{window}")
        )

        # std：過去件数1のときNaNになるので、運用ルールで0.0に補正
        std_raw = roll.std(ddof=1).reset_index(level=0, drop=True)
        cnt = roll.count().reset_index(level=0, drop=True)
        std_raw = std_raw.mask(cnt == 1, 0.0)
        std_df = std_raw.loc[start_idx].rename(columns=lambda c: f"{c}_std_{window}")

        feat_chunk = pd.concat([prev_df, min_df, max_df, mean_df, std_df], axis=1)
        base = pd.concat([base, feat_chunk], axis=1)
        added_cols.extend(feat_chunk.columns.tolist())

    df_work = df_work.merge(base, on=[HORSE_COL, "_date_key"], how="left")
    df_work = df_work.drop(columns=["_date_key"])

    if keep_order:
        df_work = df_work.sort_values("_orig_idx").drop(columns=["_orig_idx"])

    df_work = df_work.copy()
    return df_work, added_cols


def add_runningstyle_features(
    df: pd.DataFrame,
    window: int = 5,
    keep_order: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """
    脚質（決め手）特徴量を作る。
    - 当該行を含めない → shift(1)
    - 同日データを含めない → 「馬×同日」先頭行で作り配布
    """
    required = [HORSE_COL, DATE_COL, RACE_COL, STYLE_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[warn] 脚質特徴量: 必須列が不足しています（スキップ）: {missing}")
        return df, []

    df_work = df.copy()

    if keep_order:
        df_work["_orig_idx"] = np.arange(len(df_work), dtype=np.int64)

    df_work["_date_key"] = _normalize_date_only(df_work[DATE_COL])
    df_work = df_work.sort_values([HORSE_COL, "_date_key", RACE_COL]).reset_index(drop=True)

    start_mask = df_work.groupby([HORSE_COL, "_date_key"]).cumcount().eq(0)
    start_idx = df_work.index[start_mask]

    rep = df_work.loc[start_idx, [HORSE_COL, "_date_key", STYLE_COL]].copy()

    # 想定外の脚質があれば warn（集計対象から除外）
    invalid = rep[STYLE_COL].dropna()
    invalid = invalid[~invalid.isin(STYLE_LABELS)]
    if not invalid.empty:
        unique_vals = sorted(invalid.unique().tolist())
        print(f"[warn] 決め手に想定外カテゴリがあります（NaN扱いで除外）: {unique_vals}")
    rep[STYLE_COL] = rep[STYLE_COL].where(rep[STYLE_COL].isin(STYLE_LABELS), np.nan)

    rep["決め手_prev"] = rep.groupby(HORSE_COL)[STYLE_COL].shift(1)

    cat = pd.Categorical(rep[STYLE_COL], categories=STYLE_LABELS)
    dummies = pd.get_dummies(cat)
    dummies.columns = STYLE_LABELS

    shifted = dummies.groupby(rep[HORSE_COL]).shift(1)
    roll = shifted.groupby(rep[HORSE_COL]).rolling(window=window, min_periods=1).sum()
    counts = roll.reset_index(level=0, drop=True)

    rename = {label: f"脚質_{label}_count_last{window}" for label in STYLE_LABELS}
    counts = counts.rename(columns=rename)

    feat = pd.concat([rep[[HORSE_COL, "_date_key", "決め手_prev"]], counts], axis=1)
    df_work = df_work.merge(feat, on=[HORSE_COL, "_date_key"], how="left")

    df_work = df_work.drop(columns=["_date_key"])

    if keep_order:
        df_work = df_work.sort_values("_orig_idx").drop(columns=["_orig_idx"])

    df_work = df_work.copy()
    added_cols = ["決め手_prev"] + list(rename.values())
    return df_work, added_cols


def _make_blinker_on(series: pd.Series) -> pd.Series:
    # NULL/空文字は未装着、それ以外は装着扱い
    return series.notna() & series.astype(str).str.strip().ne("")


def add_blinker_flags(
    df: pd.DataFrame,
    keep_order: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """
    ブリンカー装着フラグ/履歴（馬単位）を作る。
    - 当該行・同日データは過去に含めない
    """
    required = [HORSE_COL, DATE_COL, RACE_COL, BLINKER_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[warn] ブリンカー履歴: 必須列が不足しています（スキップ）: {missing}")
        return df, []

    df_work = df.copy()

    if keep_order:
        df_work["_orig_idx"] = np.arange(len(df_work), dtype=np.int64)

    df_work["_date_key"] = _normalize_date_only(df_work[DATE_COL])
    df_work = df_work.sort_values([HORSE_COL, "_date_key", RACE_COL]).reset_index(drop=True)

    blinker_on = _make_blinker_on(df_work[BLINKER_COL]).astype(int)
    df_work["_blinker_on"] = blinker_on

    # 日付単位で装着有無を集約し、過去装着を判定
    day_on = (
        df_work.groupby([HORSE_COL, "_date_key"], as_index=False)["_blinker_on"]
        .max()
        .sort_values([HORSE_COL, "_date_key"])
    )
    day_on["on_cum"] = day_on.groupby(HORSE_COL)["_blinker_on"].cumsum().shift(1)
    day_on["blinker_ever_before"] = (day_on["on_cum"] > 0).astype(int)
    day_on = day_on[[HORSE_COL, "_date_key", "blinker_ever_before"]]

    df_work = df_work.merge(day_on, on=[HORSE_COL, "_date_key"], how="left")
    df_work["blinker_ever_before"] = df_work["blinker_ever_before"].fillna(0).astype(int)

    blinker_first = (df_work["_blinker_on"] == 1) & (df_work["blinker_ever_before"] == 0)
    blinker_return = (df_work["_blinker_on"] == 1) & (df_work["blinker_ever_before"] == 1)

    flag_df = pd.DataFrame(
        {
            "ブリンカー_on": df_work["_blinker_on"].to_numpy(),
            "ブリンカー_ever_before": df_work["blinker_ever_before"].to_numpy(),
            "ブリンカー_first_time": blinker_first.astype(int).to_numpy(),
            "ブリンカー_returning": blinker_return.astype(int).to_numpy(),
        }
    )

    df_work = pd.concat([df_work, flag_df], axis=1)
    df_work = df_work.drop(columns=["_date_key", "_blinker_on", "blinker_ever_before"])

    if keep_order:
        df_work = df_work.sort_values("_orig_idx").drop(columns=["_orig_idx"])

    df_work = df_work.copy()

    added_cols = [
        "ブリンカー_on",
        "ブリンカー_ever_before",
        "ブリンカー_first_time",
        "ブリンカー_returning",
    ]
    return df_work, added_cols


def add_last5_horse_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    血統登録番号ごとの直近5走成績（勝率/連帯率/複勝率/出走回数/獲得賞金）。
    - 当該行・同日データは含めない
    """
    new_cols: list[str] = []
    if HORSE_COL not in df.columns:
        return df, new_cols
    if FINISH_COL not in df.columns or EARNINGS_COL not in df.columns:
        print("[warn] 血統登録番号: 入線順位/獲得賞金が無いため過去n走集計をスキップします。")
        return df, new_cols

    df = df.sort_values([HORSE_COL, DATE_COL, RACE_COL]).reset_index(drop=True)
    date_only = _normalize_date_only(df[DATE_COL])

    win_col = f"{HORSE_COL}_win_rate_last5"
    place_col = f"{HORSE_COL}_place_rate_last5"
    show_col = f"{HORSE_COL}_show_rate_last5"
    starts_col = f"{HORSE_COL}_starts_last5"
    earn_col = f"{HORSE_COL}_earnings_last5"
    new_cols = [win_col, place_col, show_col, starts_col, earn_col]

    win_arr = np.full(len(df), np.nan)
    place_arr = np.full(len(df), np.nan)
    show_arr = np.full(len(df), np.nan)
    starts_arr = np.full(len(df), np.nan)
    earn_arr = np.full(len(df), np.nan)

    grouped = df.groupby(HORSE_COL, sort=False)
    for _, g in progress(grouped, total=grouped.ngroups, desc="horse_last5"):
        past_finish: list[float] = []
        past_earn: list[float] = []

        for _, idxs in g.groupby(date_only, sort=True).groups.items():
            idx_list = list(idxs)

            fvals = np.array(past_finish[-5:], dtype="float64")
            evals = np.array(past_earn[-5:], dtype="float64")

            fvals = fvals[~np.isnan(fvals)]
            evals = evals[~np.isnan(evals)]

            starts = int(fvals.size)
            wins = int((fvals == 1).sum())
            top2 = int((fvals <= 2).sum())
            top3 = int((fvals <= 3).sum())
            earnings = float(np.nansum(evals)) if evals.size else np.nan

            win_arr[idx_list] = wins / starts if starts > 0 else np.nan
            place_arr[idx_list] = top2 / starts if starts > 0 else np.nan
            show_arr[idx_list] = top3 / starts if starts > 0 else np.nan
            starts_arr[idx_list] = starts
            earn_arr[idx_list] = earnings

            past_finish.extend(pd.to_numeric(df.loc[idx_list, FINISH_COL], errors="coerce").tolist())
            past_earn.extend(pd.to_numeric(df.loc[idx_list, EARNINGS_COL], errors="coerce").tolist())

    df[win_col] = win_arr
    df[place_col] = place_arr
    df[show_col] = show_arr
    df[starts_col] = starts_arr
    df[earn_col] = earn_arr

    return df, new_cols


def _date_level_agg(
    df: pd.DataFrame,
    key_col: str,
    surface_col: str | None = None,
    surface_value: str | None = None,
    track_col: str | None = None,
    track_good: bool | None = None,
) -> pd.DataFrame:
    """
    キー×日付単位の集計テーブルを作る（同日データ混入防止の土台）
    """
    cols = [key_col, DATE_COL, FINISH_COL, EARNINGS_COL]
    if surface_col is not None:
        cols.append(surface_col)
    if track_col is not None:
        cols.append(track_col)

    sub = df[cols].copy()
    sub[DATE_COL] = _normalize_date_only(sub[DATE_COL])

    # surface フィルタ
    if surface_col is not None and surface_value is not None:
        sub = sub[sub[surface_col] == surface_value]

    # 馬場状態フィルタ（良/良以外）
    if track_col is not None and track_good is not None:
        if track_good:
            sub = sub[sub[track_col] == "良"]
        else:
            sub = sub[sub[track_col].notna() & (sub[track_col] != "良")]

    sub = sub.dropna(subset=[key_col, DATE_COL])

    finish = pd.to_numeric(sub[FINISH_COL], errors="coerce")
    earnings = pd.to_numeric(sub[EARNINGS_COL], errors="coerce")

    # 入線順位が無効な行は分母から除外（FEATURES/PIPELINE の定義に合わせる）
    valid = finish.notna()
    sub = sub.loc[valid].copy()
    finish = finish.loc[valid]
    earnings = earnings.loc[valid]

    sub["races"] = 1
    sub["wins"] = (finish == 1).astype(int)
    sub["top2"] = (finish <= 2).astype(int)
    sub["top3"] = (finish <= 3).astype(int)
    sub["earnings"] = earnings

    agg = (
        sub.groupby([key_col, DATE_COL], as_index=False)[
            ["races", "wins", "top2", "top3", "earnings"]
        ]
        .sum()
        .sort_values([key_col, DATE_COL])
    )
    return agg


def _career_features(agg: pd.DataFrame, key_col: str, prefix: str) -> pd.DataFrame:
    """
    通算(career)の勝率等を作る。shift(1) で当該日を除外。
    """
    g = agg.groupby(key_col, sort=False)

    agg["races_cum"] = g["races"].cumsum().shift(1)
    agg["wins_cum"] = g["wins"].cumsum().shift(1)
    agg["top2_cum"] = g["top2"].cumsum().shift(1)
    agg["top3_cum"] = g["top3"].cumsum().shift(1)
    agg["earnings_cum"] = g["earnings"].cumsum().shift(1)

    races = agg["races_cum"].to_numpy()
    wins = agg["wins_cum"].to_numpy()
    top2 = agg["top2_cum"].to_numpy()
    top3 = agg["top3_cum"].to_numpy()
    earnings = agg["earnings_cum"].to_numpy()

    return pd.DataFrame(
        {
            key_col: agg[key_col],
            DATE_COL: agg[DATE_COL],
            f"{prefix}_win_rate_career": np.where(races > 0, wins / races, np.nan),
            f"{prefix}_place_rate_career": np.where(races > 0, top2 / races, np.nan),
            f"{prefix}_show_rate_career": np.where(races > 0, top3 / races, np.nan),
            f"{prefix}_starts_career": races,
            f"{prefix}_earnings_career": earnings,
        }
    )


def _rolling_1y_features(agg: pd.DataFrame, key_col: str, prefix: str) -> pd.DataFrame:
    """
    直近1年(365D)の勝率等。closed='left' で当該日除外。
    """
    tmp = agg.set_index(DATE_COL)

    rolling = (
        tmp.groupby(key_col)[["races", "wins", "top2", "top3", "earnings"]]
        .rolling("365D", closed="left")
        .sum()
        .reset_index()
    )

    races = rolling["races"].to_numpy()
    wins = rolling["wins"].to_numpy()
    top2 = rolling["top2"].to_numpy()
    top3 = rolling["top3"].to_numpy()
    earnings = rolling["earnings"].to_numpy()

    return pd.DataFrame(
        {
            key_col: rolling[key_col],
            DATE_COL: rolling[DATE_COL],
            f"{prefix}_win_rate_1y": np.where(races > 0, wins / races, np.nan),
            f"{prefix}_place_rate_1y": np.where(races > 0, top2 / races, np.nan),
            f"{prefix}_show_rate_1y": np.where(races > 0, top3 / races, np.nan),
            f"{prefix}_starts_1y": races,
            f"{prefix}_earnings_1y": earnings,
        }
    )


def build_perf_snapshot(
    df: pd.DataFrame,
    key_col: str,
    add_1y: bool,
    surface_col: str | None,
    surface_turf_value: str | None,
    surface_dirt_value: str | None,
    track_col: str | None,
) -> pd.DataFrame:
    """
    通常成績（血統・関係者）用のスナップショットを作る。
    - 同日除外は日付単位集計 + shift(1) で担保
    """
    agg_all = _date_level_agg(df, key_col)
    if agg_all.empty:
        return pd.DataFrame(columns=[key_col, DATE_COL])

    base = agg_all[[key_col, DATE_COL]].copy()
    feats = []

    feats.append(_career_features(agg_all, key_col, key_col))
    if add_1y:
        feats.append(_rolling_1y_features(agg_all, key_col, key_col))

    # surface 別
    if surface_col and surface_turf_value:
        turf = _date_level_agg(df, key_col, surface_col=surface_col, surface_value=surface_turf_value)
        if not turf.empty:
            feats.append(_career_features(turf, key_col, f"{key_col}_turf"))
    if surface_col and surface_dirt_value:
        dirt = _date_level_agg(df, key_col, surface_col=surface_col, surface_value=surface_dirt_value)
        if not dirt.empty:
            feats.append(_career_features(dirt, key_col, f"{key_col}_dirt"))

    # 良馬場 / 良馬場以外
    if track_col:
        good = _date_level_agg(df, key_col, track_col=track_col, track_good=True)
        other = _date_level_agg(df, key_col, track_col=track_col, track_good=False)
        if not good.empty:
            feats.append(_career_features(good, key_col, f"{key_col}_good"))
        if not other.empty:
            feats.append(_career_features(other, key_col, f"{key_col}_other"))

        # surface + 馬場状態 の組み合わせ
        if surface_col and surface_turf_value:
            turf_good = _date_level_agg(
                df, key_col,
                surface_col=surface_col, surface_value=surface_turf_value,
                track_col=track_col, track_good=True
            )
            turf_other = _date_level_agg(
                df, key_col,
                surface_col=surface_col, surface_value=surface_turf_value,
                track_col=track_col, track_good=False
            )
            if not turf_good.empty:
                feats.append(_career_features(turf_good, key_col, f"{key_col}_turf_good"))
            if not turf_other.empty:
                feats.append(_career_features(turf_other, key_col, f"{key_col}_turf_other"))

        if surface_col and surface_dirt_value:
            dirt_good = _date_level_agg(
                df, key_col,
                surface_col=surface_col, surface_value=surface_dirt_value,
                track_col=track_col, track_good=True
            )
            dirt_other = _date_level_agg(
                df, key_col,
                surface_col=surface_col, surface_value=surface_dirt_value,
                track_col=track_col, track_good=False
            )
            if not dirt_good.empty:
                feats.append(_career_features(dirt_good, key_col, f"{key_col}_dirt_good"))
            if not dirt_other.empty:
                feats.append(_career_features(dirt_other, key_col, f"{key_col}_dirt_other"))

    # 結合
    out = base
    for f in feats:
        out = out.merge(f, on=[key_col, DATE_COL], how="left")
    return out


def build_blinker_perf_snapshot(
    df: pd.DataFrame,
    key_col: str,
    prefix: str,
) -> pd.DataFrame:
    """
    ブリンカー装着時成績（career）のスナップショット。
    - フィルタ：ブリンカー_on=1
    - 当該日除外：日付単位集計 + shift(1)
    """
    if BLINKER_COL not in df.columns:
        return pd.DataFrame(columns=[key_col, DATE_COL])

    blinker_on = _make_blinker_on(df[BLINKER_COL]).astype(int)
    finish_num = pd.to_numeric(df[FINISH_COL], errors="coerce")
    valid = finish_num.notna()

    use = (blinker_on == 1) & valid
    sub = df.loc[use, [key_col, DATE_COL]].copy()
    if sub.empty:
        return pd.DataFrame(columns=[key_col, DATE_COL])

    sub[DATE_COL] = _normalize_date_only(sub[DATE_COL])
    sub["races"] = 1
    sub["wins"] = (finish_num[use] == 1).astype(int).to_numpy()
    sub["top2"] = (finish_num[use] <= 2).astype(int).to_numpy()
    sub["top3"] = (finish_num[use] <= 3).astype(int).to_numpy()
    sub["earnings"] = pd.to_numeric(df.loc[use, EARNINGS_COL], errors="coerce").to_numpy()

    day_level = (
        sub.groupby([key_col, DATE_COL], as_index=False)[
            ["races", "wins", "top2", "top3", "earnings"]
        ]
        .sum()
        .sort_values([key_col, DATE_COL])
    )

    base = day_level[[key_col, DATE_COL]].copy()
    g = day_level.groupby(key_col, sort=False)
    day_level["races_cum"] = g["races"].cumsum().shift(1)
    day_level["wins_cum"] = g["wins"].cumsum().shift(1)
    day_level["top2_cum"] = g["top2"].cumsum().shift(1)
    day_level["top3_cum"] = g["top3"].cumsum().shift(1)
    day_level["earnings_cum"] = g["earnings"].cumsum().shift(1)

    races = day_level["races_cum"].to_numpy()
    wins = day_level["wins_cum"].to_numpy()
    top2 = day_level["top2_cum"].to_numpy()
    top3 = day_level["top3_cum"].to_numpy()
    earnings = day_level["earnings_cum"].to_numpy()

    feat = pd.DataFrame(
        {
            key_col: day_level[key_col],
            DATE_COL: day_level[DATE_COL],
            f"{prefix}_win_rate_career": np.where(races > 0, wins / races, np.nan),
            f"{prefix}_place_rate_career": np.where(races > 0, top2 / races, np.nan),
            f"{prefix}_show_rate_career": np.where(races > 0, top3 / races, np.nan),
            f"{prefix}_starts_career": races,
            f"{prefix}_earnings_career": earnings,
        }
    )
    out = base.merge(feat, on=[key_col, DATE_COL], how="left")
    return out


def run(
    input_path: str | Path | None = None,
    surface_col: str | None = None,
    surface_turf_value: str | None = None,
    surface_dirt_value: str | None = None,
    course_label_col: str | None = None,
) -> None:
    cfg = RunConfig(
        input_path=Path(input_path) if input_path else None,
        surface_col=surface_col,
        surface_turf_value=surface_turf_value,
        surface_dirt_value=surface_dirt_value,
        course_label_col=course_label_col,
    )

    if cfg.input_path is None:
        latest = find_latest_csv(PROCESSED_DIR)
        if latest is None:
            raise FileNotFoundError(
                "Processed にCSVが見つかりません。input_path で手動指定してください。"
            )
        cfg.input_path = latest

    print(f"[info] input: {cfg.input_path}")
    df = pd.read_csv(cfg.input_path, encoding="cp932")

    validate_required_cols(df)
    df = normalize_date(df)

    # コースラベル結合キー（調教師/騎手×コースラベル）を作成
    if cfg.course_label_col and cfg.course_label_col in df.columns:
        if "調教師" in df.columns and "調教師×コースラベル" not in df.columns:
            make_combo_column(df, "調教師", cfg.course_label_col, "調教師×コースラベル")
        if "騎手" in df.columns and "騎手×コースラベル" not in df.columns:
            make_combo_column(df, "騎手", cfg.course_label_col, "騎手×コースラベル")

    # 調教師×騎手（ブリンカー関係者で使用）を補完
    if "調教師×騎手" not in df.columns:
        if "調教師" in df.columns and "騎手" in df.columns:
            make_combo_column(df, "調教師", "騎手", "調教師×騎手")

    # ===== Horse snapshots =====
    df_past, past_cols = add_past_numeric_features(df)
    _write_snapshot(
        df_past,
        FEATURES_DIR / "Horse" / "horse_past_numeric_snapshot.csv",
        [HORSE_COL],
        past_cols,
    )

    df_style, style_cols = add_runningstyle_features(df)
    _write_snapshot(
        df_style,
        FEATURES_DIR / "Horse" / "horse_runningstyle_snapshot.csv",
        [HORSE_COL],
        style_cols,
    )

    df_flags, flag_cols = add_blinker_flags(df)
    _write_snapshot(
        df_flags,
        FEATURES_DIR / "Horse" / "horse_blinker_flags_snapshot.csv",
        [HORSE_COL],
        flag_cols,
    )

    df_last5, last5_cols = add_last5_horse_features(df)
    _write_snapshot(
        df_last5,
        FEATURES_DIR / "Horse" / "horse_last5_perf_snapshot.csv",
        [HORSE_COL],
        last5_cols,
    )

    # ブリンカー装着時（馬）
    blinker_horse = build_blinker_perf_snapshot(
        df, key_col=HORSE_COL, prefix="ブリンカー装着時"
    )
    _write_snapshot(
        blinker_horse,
        FEATURES_DIR / "Blinker" / "Horse" / "blinker_performance_snapshot.csv",
        [HORSE_COL],
        [
            "ブリンカー装着時_win_rate_career",
            "ブリンカー装着時_place_rate_career",
            "ブリンカー装着時_show_rate_career",
            "ブリンカー装着時_starts_career",
            "ブリンカー装着時_earnings_career",
        ],
    )

    # ===== Blood / Related / Course snapshots =====
    for key in progress(BLOOD_KEYS, desc="blood_keys"):
        snap = build_perf_snapshot(
            df,
            key_col=key,
            add_1y=False,
            surface_col=cfg.surface_col,
            surface_turf_value=cfg.surface_turf_value,
            surface_dirt_value=cfg.surface_dirt_value,
            track_col=TRACK_COL if TRACK_COL in df.columns else None,
        )
        out_dir = FEATURES_DIR / KEY_DIRS[key]
        _write_snapshot(
            snap,
            out_dir / "performance_snapshot.csv",
            [key],
            [c for c in snap.columns if c not in (key, DATE_COL)],
        )

    for key in progress(RELATED_KEYS, desc="related_keys"):
        snap = build_perf_snapshot(
            df,
            key_col=key,
            add_1y=True,
            surface_col=cfg.surface_col,
            surface_turf_value=cfg.surface_turf_value,
            surface_dirt_value=cfg.surface_dirt_value,
            track_col=TRACK_COL if TRACK_COL in df.columns else None,
        )
        out_dir = FEATURES_DIR / KEY_DIRS[key]
        _write_snapshot(
            snap,
            out_dir / "performance_snapshot.csv",
            [key],
            [c for c in snap.columns if c not in (key, DATE_COL)],
        )

    for key in progress(COURSE_KEYS, desc="course_keys"):
        if key not in df.columns:
            print(f"[warn] 列 '{key}' が無いためコース適性はスキップします。")
            continue
        snap = build_perf_snapshot(
            df,
            key_col=key,
            add_1y=True,
            surface_col=None,
            surface_turf_value=None,
            surface_dirt_value=None,
            track_col=TRACK_COL if TRACK_COL in df.columns else None,
        )
        out_dir = FEATURES_DIR / KEY_DIRS[key]
        _write_snapshot(
            snap,
            out_dir / "performance_snapshot.csv",
            [key],
            [c for c in snap.columns if c not in (key, DATE_COL)],
        )

    # ===== Blinker conditional snapshots (relations) =====
    for key in progress(BLINKER_REL_KEYS, desc="blinker_rel_keys"):
        if key not in df.columns:
            print(f"[warn] 列 '{key}' が無いためブリンカー関係者はスキップします。")
            continue
        prefix = f"{key}_ブリンカー装着時"
        snap = build_blinker_perf_snapshot(df, key_col=key, prefix=prefix)
        out_dir = FEATURES_DIR / "Blinker" / KEY_DIRS.get(key, key)
        _write_snapshot(
            snap,
            out_dir / "blinker_performance_snapshot.csv",
            [key],
            [c for c in snap.columns if c not in (key, DATE_COL)],
        )

    print("[info] done")


if __name__ == "__main__":
    run()
