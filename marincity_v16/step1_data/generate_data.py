# =====================================================================
# [1차] generate_data.py – 학습용 월파 시나리오 생성
#   변수 범위 정의 → A 격자(같은 간격) → B 경계 집중(Sobol 기각 샘플링) → C 범위 가장자리 → 분할 → 검증
#   (교수님 설계안 "학습데이터" 1~8장)
#
# 실행 (프로젝트 폴더에서):
#   py step1_data/generate_data.py           생성 + 검증 리포트 + 데이터 카드
#   py step1_data/generate_data.py --check   같은 설정·시드로 다시 만들어 같은지 확인
#
# 산출물: step1_data/output/
#   scenarios.csv        학습용 3,000건 (split 열: train / val / test)
#   eval_boundary.csv    평가 전용 경계 시험 300건
#   eval_outside.csv     평가 전용 범위 밖 시험 200건
#   filter_log.csv       단계별 후보 수와 제외 사유
#   validation_report.md 커버리지·층화·최근접 거리·물리 일관성·분할·재현성
#   data_card.md         범위·가정·한계
#   coverage.png         변수별 히스토그램과 쌍별 산점도
#
# 전제 조건 (10/27 멘토링): 단면 고정 = 사면 경사·수심·조도계수 고정.
#   표본 공간은 설계안대로 (Hs, 파형경사, Rc) 3차원이고, 수심 h는 config의 고정값.
#
# 변경 이력
#   2026-10-28  최초 작성
#   2026-10-28  수심 고정, 표본 공간 (Hs, 파형경사, Rc)로 변경
#   2026-10-29  혼합 방식: A를 Sobol 층화 대신 같은 간격 격자로 (B 경계 집중·C 가장자리는 그대로)
# =====================================================================
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys

import matplotlib

matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import qmc
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import overtopping as ot  # noqa: E402

INPUTS = ["hs", "tp", "rc"]


# ---------------------------------------------------------------------
# 범위와 정규화
# ---------------------------------------------------------------------
def input_bounds(cfg: dict, ranges: dict | None = None) -> dict:
    """모델 입력 (Hs, Tp, Rc)의 이론 범위."""
    r = ranges or cfg["ranges"]
    st = cfg["structure"]
    hs_hi = min(r["hs"][1], cfg["regime"]["breaker_limit"] * st["water_depth_h"])
    return {
        "hs": (r["hs"][0], hs_hi),
        "tp": (float(ot.tp_from_steepness(r["hs"][0], r["steepness"][1])), float(ot.tp_from_steepness(hs_hi, r["steepness"][0]))),
        "rc": (r["rc"][0], r["rc"][1]),
    }


def normalize(df: pd.DataFrame, bounds: dict) -> np.ndarray:
    return np.column_stack([(df[k].values - bounds[k][0]) / (bounds[k][1] - bounds[k][0]) for k in INPUTS])


def extended_ranges(cfg: dict) -> dict:
    ext = cfg["sampling"]["eval_outside_extend"]
    out = {k: [lo - ext * (hi - lo), hi + ext * (hi - lo)] for k, (lo, hi) in cfg["ranges"].items()}
    out["hs"][0] = max(out["hs"][0], 0.1)
    out["steepness"][0] = max(out["steepness"][0], 0.002)
    return out


# ---------------------------------------------------------------------
# 후보 생성과 라벨
# ---------------------------------------------------------------------
def sobol(seed: int, n: int, dim: int = 3) -> np.ndarray:
    return qmc.Sobol(d=dim, scramble=True, seed=seed).random(n)


def make_candidates(u: np.ndarray, cfg: dict, ranges: dict, sample_type: str, tag: str) -> pd.DataFrame:
    """단위 입방체 u (n, 3) → (Hs, 파형경사, Rc) → 입력과 라벨."""
    lo = lambda k: ranges[k][0]  # noqa: E731
    span = lambda k: ranges[k][1] - ranges[k][0]  # noqa: E731
    hs = lo("hs") + u[:, 0] * span("hs")
    s = lo("steepness") + u[:, 1] * span("steepness")
    rc = lo("rc") + u[:, 2] * span("rc")
    nd = cfg["sampling"]["round_decimals"]
    df = pd.DataFrame({"hs": np.round(hs, nd), "tp": np.round(ot.tp_from_steepness(hs, s), nd), "rc": np.round(rc, nd)})
    return add_labels(df, cfg, sample_type, tag)


def add_labels(df: pd.DataFrame, cfg: dict, sample_type: str, tag: str) -> pd.DataFrame:
    df = df.copy()
    df["h_toe"] = cfg["structure"]["water_depth_h"]  # 고정 수심
    ev = ot.evaluate(df["hs"].values, df["tp"].values, df["rc"].values, cfg)
    for k in ["tm10", "l_m10", "s_m10", "xi", "gamma_f_used", "q", "grade", "building"]:
        df[k] = ev[k]
    df["s_p"] = 2 * np.pi * df["hs"] / (ot.G * df["tp"] ** 2)
    df["hs_over_h"] = df["hs"] / df["h_toe"]
    df["rc_over_hs"] = df["rc"] / df["hs"]
    df["regime"] = ot.regime(df["hs"].values, df["h_toe"].values, cfg)
    # 등급 경계는 국내 기준 고정값 (config grades)
    df["log10_q"] = np.where(df["q"] > 0, np.log10(df["q"].clip(lower=1e-12)), np.nan)
    lo, hi = cfg["formula"]["xi_valid"]
    df["xi_in_gf_range"] = ((df["xi"] >= lo) & (df["xi"] <= hi)).astype(int)
    df["sample_type"], df["batch"] = sample_type, tag

    reason = np.full(len(df), "", dtype=object)
    reason[(df["h_toe"] <= 0).values] = "수심없음"
    reason[((reason == "") & (df["hs_over_h"] > cfg["regime"]["breaker_limit"])).values] = "쇄파한계초과"
    reason[((reason == "") & (df["rc"] <= 0)).values] = "적용범위밖(Rc≤0)"
    df["exclude_reason"] = reason
    return df


def boundary_distance(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """log10 q와 가장 가까운 경계(안전/주의, 주의/위험, 건물 피해)까지의 거리."""
    lq = df["log10_q"]
    bounds = [ot.criterion(cfg, k) for k in ("safe_max", "danger_min", "building")]
    return pd.concat([(lq - np.log10(b)).abs() for b in bounds], axis=1).min(axis=1)


# ---------------------------------------------------------------------
# 중복 없이 고르기
# ---------------------------------------------------------------------
class Picker:
    """이미 고른 점들과 정규화 거리 dedupe_distance 이상 떨어진 점만 순서대로 채택."""

    def __init__(self, bounds: dict, min_dist: float):
        self.bounds, self.min_d2 = bounds, min_dist**2
        self.points = np.empty((0, 3))

    def take(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        if n <= 0 or df.empty:
            return df.iloc[0:0]
        x = normalize(df, self.bounds)
        keep = []
        for i in range(len(df)):
            if len(keep) >= n:
                break
            if len(self.points) and np.min(np.sum((self.points - x[i]) ** 2, axis=1)) < self.min_d2:
                continue
            self.points = np.vstack([self.points, x[i]])
            keep.append(i)
        return df.iloc[keep]


# ---------------------------------------------------------------------
# A. 격자: 각 변수를 최솟값부터 같은 간격으로 (혼합 방식)
# ---------------------------------------------------------------------
def make_from_inputs(hs, tp, rc, cfg: dict, sample_type: str, tag: str) -> pd.DataFrame:
    nd = cfg["sampling"]["round_decimals"]
    df = pd.DataFrame({"hs": np.round(hs, nd), "tp": np.round(tp, nd), "rc": np.round(rc, nd)})
    return add_labels(df, cfg, sample_type, tag)


def select_grid(cfg: dict, picker: Picker, log: list) -> tuple[pd.DataFrame, dict]:
    gcfg, r = cfg["sampling"]["grid"], cfg["ranges"]
    hs_hi = input_bounds(cfg)["hs"][1]
    hs_lv = np.arange(r["hs"][0], hs_hi + 1e-9, gcfg["hs_step"])
    rc_lv = np.arange(r["rc"][0], r["rc"][1] + 1e-9, gcfg["rc_step"])
    s_lo, s_hi = r["steepness"]
    H, T, R = [], [], []
    for h in hs_lv:
        # 주기는 이 파고에서 가능한 범위(파형경사 s_hi ~ s_lo)를 같은 간격으로
        for t in np.linspace(float(ot.tp_from_steepness(h, s_hi)), float(ot.tp_from_steepness(h, s_lo)), gcfg["tp_levels"]):
            for rc in rc_lv:
                H.append(h); T.append(t); R.append(rc)  # noqa: E702
    grid = make_from_inputs(np.array(H), np.array(T), np.array(R), cfg, "A", "A-격자")
    log.append(("A", "격자 생성", len(grid), f"파고 {len(hs_lv)} × 주기 {gcfg['tp_levels']} × 여유고 {len(rc_lv)}"))
    for reason, c in grid["exclude_reason"].value_counts().items():
        if reason:
            log.append(("A", "제외", int(c), reason))
    grid = grid[grid["exclude_reason"] == ""]
    a = picker.take(grid, len(grid))
    log.append(("A", "선택", len(a), ""))
    cells = {f"{rg}-{gd}": int(((a["regime"] == rg) & (a["grade"] == gd)).sum()) for rg in range(3) for gd in range(3)}
    return a.reset_index(drop=True), cells


# ---------------------------------------------------------------------
# B. 등급 경계 집중 (기각 샘플링)
# ---------------------------------------------------------------------
def select_boundary(cfg, picker, log, n, delta, tag, seed0, quota=None) -> pd.DataFrame:
    """quota: 등급별로 더 받을 수 있는 건수 (단일 등급 상한 유지). None 이면 제한 없음."""
    nb = cfg["sampling"]["candidates_per_batch"]
    quota = dict(quota) if quota else None
    out, total, k = [], 0, 0
    got = lambda: sum(len(x) for x in out)  # noqa: E731
    while got() < n and k < 60:
        c = make_candidates(sobol(seed0 + k, nb), cfg, cfg["ranges"], tag, f"{tag}{k}")
        total += len(c)
        c = c[(c["exclude_reason"] == "") & (c["q"] > 0)]
        c = c[boundary_distance(c, cfg) <= delta]
        if quota is None:
            out.append(picker.take(c, n - got()))
        else:
            for g in range(3):
                t = picker.take(c[c["grade"] == g], max(0, min(quota[g], n - got())))
                quota[g] -= len(t)
                out.append(t)
        k += 1
    b = pd.concat(out, ignore_index=True)
    log.append((tag, "후보 생성", total, ""))
    log.append((tag, f"경계 ±{delta} 채택", len(b), ""))
    return b


# ---------------------------------------------------------------------
# C. 범위 가장자리 (모서리 + 면)
# ---------------------------------------------------------------------
def select_edge(cfg, picker, log) -> tuple[pd.DataFrame, list]:
    n = cfg["sampling"]["n_edge"]
    corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], float)
    cdf = make_candidates(corners, cfg, cfg["ranges"], "C", "C-모서리")
    infeasible = [
        {"hs": r.hs, "tp": r.tp, "rc": r.rc, "사유": r.exclude_reason}
        for r in cdf.itertuples()
        if r.exclude_reason
    ]
    rows = [picker.take(cdf[cdf["exclude_reason"] == ""], len(cdf))]
    faces = [(d, v) for d in range(3) for v in (0.0, 1.0)]
    k = 0
    while sum(len(x) for x in rows) < n and k < 20:
        for d, v in faces:
            u = np.insert(sobol(cfg["seed"] + 5000 + k * 10 + d * 2 + int(v), 64, dim=2), d, v, axis=1)
            c = make_candidates(u, cfg, cfg["ranges"], "C", f"C-면{d}{int(v)}")
            need = n - sum(len(x) for x in rows)
            rows.append(picker.take(c[c["exclude_reason"] == ""], min(max(1, need // len(faces)), need)))
        k += 1
    edge = pd.concat(rows, ignore_index=True).head(n)
    log.append(("C", "선택", len(edge), ""))
    log.append(("C", "불가능 모서리", len(infeasible), ""))
    return edge, infeasible


# ---------------------------------------------------------------------
# 평가 전용: 범위 밖 시험
# ---------------------------------------------------------------------
def select_outside(cfg, picker, log) -> pd.DataFrame:
    n = cfg["sampling"]["eval_outside_n"]
    ext, r0 = extended_ranges(cfg), cfg["ranges"]
    out, k = [], 0
    while sum(len(x) for x in out) < n and k < 40:
        c = make_candidates(sobol(cfg["seed"] + 9000 + k, cfg["sampling"]["candidates_per_batch"]), cfg, ext, "OUT", f"OUT{k}")
        c = c[c["exclude_reason"] == ""]
        outside = (
            (c["hs"] < r0["hs"][0]) | (c["hs"] > r0["hs"][1])
            | (c["s_p"] < r0["steepness"][0]) | (c["s_p"] > r0["steepness"][1])
            | (c["rc"] < r0["rc"][0]) | (c["rc"] > r0["rc"][1])
        )
        out.append(picker.take(c[outside], n - sum(len(x) for x in out)))
        k += 1
    o = pd.concat(out, ignore_index=True)
    log.append(("OUT", "선택", len(o), ""))
    return o


# ---------------------------------------------------------------------
# 분할 (영역 × 등급 층화)
# ---------------------------------------------------------------------
def assign_split(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    sp = cfg["sampling"]["split"]
    strat = df["regime"].astype(str) + "-" + df["grade"].astype(str)
    strat = strat.where(strat.map(strat.value_counts()) >= 10, "기타")
    idx = np.arange(len(df))
    tr, rest = train_test_split(idx, train_size=sp["train"], stratify=strat, random_state=cfg["seed"])
    va, te = train_test_split(rest, train_size=sp["val"] / (sp["val"] + sp["test"]), stratify=strat.iloc[rest], random_state=cfg["seed"])
    split = np.empty(len(df), dtype=object)
    split[tr], split[va], split[te] = "train", "val", "test"
    df = df.copy()
    df["split"] = split
    return df


# ---------------------------------------------------------------------
# 검증
# ---------------------------------------------------------------------
def cell_table(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    t = pd.crosstab(df["regime"], df["grade"]).reindex(index=range(3), columns=range(3), fill_value=0)
    t.index, t.columns = cfg["regime"]["names"], cfg["grades"]["names"]
    return t


def monotonic_check(cfg: dict) -> dict:
    """Hs 증가 → q 증가, Rc 증가 → q 감소 가 지켜지는지 격자 점검."""
    b = input_bounds(cfg)
    hs, rc = np.linspace(*b["hs"], 60), np.linspace(*b["rc"], 60)
    res = {}
    for tp in (6.0, 10.0, 15.0):
        H, R = np.meshgrid(hs, rc)
        q = ot.mean_discharge(H, np.full_like(H, tp), R, cfg)
        res[f"Tp={tp}s"] = {
            "Hs가 커지는데 q가 줄어든 격자": int((np.diff(q, axis=1) < -1e-12).sum()),
            "Rc가 커지는데 q가 늘어난 격자": int((np.diff(q, axis=0) > 1e-12).sum()),
        }
    return res


def save_coverage(data: pd.DataFrame, path: str):
    plt.rcParams["font.family"] = ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]
    fig, ax = plt.subplots(2, 3, figsize=(13, 8))
    labels = {"hs": "Hs (m)", "tp": "Tp (s)", "rc": "Rc (m)"}
    for i, k in enumerate(INPUTS):
        ax[0, i].hist(data[k], bins=40, color="#3f6e8c")
        ax[0, i].set_title(labels[k])
    colors = np.array(["#2F8F5B", "#D9A21B", "#C63D2F"])
    for i, (a, b) in enumerate([("hs", "tp"), ("hs", "rc"), ("tp", "rc")]):
        ax[1, i].scatter(data[a], data[b], s=3, c=colors[data["grade"].values], alpha=0.6)
        ax[1, i].set_xlabel(labels[a])
        ax[1, i].set_ylabel(labels[b])
    fig.suptitle("coverage  (green = safe, yellow = caution, red = danger)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_reports(cfg, data, eval_b, eval_o, log, avail, infeasible, cfg_hash, data_hash):
    gn, rn = cfg["grades"]["names"], cfg["regime"]["names"]
    bounds = input_bounds(cfg)
    save_coverage(data, ot.path_of(cfg, "coverage_png"))

    nn = cKDTree(normalize(data, bounds)).query(normalize(data, bounds), k=2)[0][:, 1]
    cells_all, cells_a = cell_table(data, cfg), cell_table(data[data["sample_type"] == "A"], cfg)
    impossible = [f"{rn[int(k[0])]}×{gn[int(k[2])]}" for k, v in avail.items() if v == 0]
    under = [f"{r}×{g} ({cells_all.loc[r, g]}건)" for r in rn for g in gn if 0 < cells_all.loc[r, g] < cfg["sampling"]["cell_min"]]
    gratio = data["grade"].value_counts(normalize=True).reindex(range(3), fill_value=0)
    split_stats = data.groupby("split")[INPUTS].agg(["mean", "std"]).round(3)
    split_stats.columns = [f"{a} {b}" for a, b in split_stats.columns]
    split_grade = pd.crosstab(data["split"], data["grade"], normalize="index").round(3)
    split_grade.columns = gn

    rep = [
        "# [1차] 학습 데이터 검증 리포트",
        "",
        f"- 설정 파일 해시 `{cfg_hash}` · 데이터 해시 `{data_hash}` · 시드 {cfg['seed']}",
        "",
        "## 건수",
        f"- 학습용 {len(data)}건: 격자(A) {int((data.sample_type == 'A').sum())}, 경계 집중(B) {int((data.sample_type == 'B').sum())}, 범위 가장자리(C) {int((data.sample_type == 'C').sum())}",
        f"- 분할: " + ", ".join(f"{k} {v}" for k, v in data["split"].value_counts().items()),
        f"- 평가 전용: 경계 시험 {len(eval_b)}건, 범위 밖 시험 {len(eval_o)}건",
        "",
        "## 입력 범위",
        "| 변수 | 설정 범위 | 실제 데이터 범위 |",
        "|---|---|---|",
        *[f"| {k} | {bounds[k][0]:.2f} ~ {bounds[k][1]:.2f} | {data[k].min():.2f} ~ {data[k].max():.2f} |" for k in INPUTS],
        "",
        "실제 범위가 설정보다 좁으면 쇄파 한계 필터 때문입니다.",
        "",
        "## 층화: 영역 × 등급 9칸",
        "### 격자(A)",
        cells_a.to_markdown(),
        "",
        "### 전체",
        cells_all.to_markdown(),
        "",
        f"- 격자에 한 건도 없는 칸 (물리적으로 생기지 않는 조합): {', '.join(impossible) or '없음'}",
        f"- 최소 건수({cfg['sampling']['cell_min']}) 미달 칸 (전체 기준, 격자에서 모자란 칸은 B에서 보충): {', '.join(under) or '없음'}",
        "- 등급 비율: " + ", ".join(f"{gn[i]} {gratio[i]:.1%}" for i in range(3))
        + f" (목표: 각 {cfg['sampling']['grade_min_ratio']:.0%} 이상, {cfg['sampling']['grade_max_ratio']:.0%} 이하)",
        f"- 건물 피해 표시 비율: {data['building'].mean():.1%}",
        f"- 월파 없음(q = 0) 비율: {(data['q'] == 0).mean():.1%}",
        "",
        "## 최근접 이웃 거리 (정규화 입력공간)",
        f"- 최소 {nn.min():.4f}, 5% {np.percentile(nn, 5):.4f}, 중앙값 {np.median(nn):.4f}, 최대 {nn.max():.4f}",
        f"- 중복 기준({cfg['sampling']['dedupe_distance']}) 미만: {int((nn < cfg['sampling']['dedupe_distance']).sum())}건",
        "",
        "## 물리 일관성",
        "```json",
        json.dumps(monotonic_check(cfg), ensure_ascii=False, indent=1),
        "```",
        "",
        "## 분할별 분포",
        split_stats.to_markdown(),
        "",
        split_grade.to_markdown(),
        "",
        "## 불가능 모서리 (C)",
        pd.DataFrame(infeasible).to_markdown() if infeasible else "없음",
        "",
        "## 필터 로그",
        pd.DataFrame(log, columns=["단계", "항목", "건수", "사유"]).to_markdown(index=False),
        "",
        "![coverage](coverage.png)",
    ]
    with open(ot.path_of(cfg, "data_report"), "w", encoding="utf-8") as f:
        f.write("\n".join(rep))

    st = cfg["structure"]
    card = [
        "# 데이터 카드: 마린시티 호안 월파 시나리오",
        "",
        "## 무엇인가",
        "EurOtop (2018) 6장 식 6.5로 계산한 평균 월파량과 위험 등급이 붙은 가상 시나리오입니다. 실측 자료가 아닙니다.",
        "",
        "## 열",
        "| 열 | 뜻 |",
        "|---|---|",
        "| hs | 구조물 앞 파고 Hm0 (m), 모델 입력 |",
        "| tp | 첨두주기 Tp (s), 모델 입력 |",
        "| rc | 여유고 Rc = 마루 표고 − 정수위 (m), 모델 입력 |",
        "| grade | 위험 등급 0 안전 · 1 주의 · 2 위험, 모델 출력 |",
        f"| building | 건물 피해 가능 1/0 (q ≥ {ot.criterion(cfg, 'building'):g} l/s/m, 가옥 기준), 모델 출력 |",
        "| q | 평균 월파량 (l/s per m) |",
        "| regime | 0 비쇄파 · 1 전환 · 2 쇄파 (구조물 앞 Hm0/h) |",
        "| sample_type | A 격자(같은 간격) · B 경계 집중 · C 범위 가장자리 |",
        "| split | train / val / test |",
        "| 그 외 | h_toe(고정 수심), tm10, s_m10, xi, gamma_f_used 등 파라미터 |",
        "",
        "## 단면",
        f"태풍 차바(2016) 당시 제방: TTP {st['ttp_ton']:.0f} t 2층 (마루 DL+{st['armour_crest_dl']:.1f} m), 사석 코어와 콘크리트 직립벽, "
        f"방파벽 꼭대기 DL+{st['crest_parapet_dl']:.1f} m, 조도계수 γf {st['gamma_f']}, "
        f"사면 1:{st['cot_alpha']}, 구조물 앞 수심 {st['water_depth_h']} m 고정.",
        "",
        "## 범위",
        f"- 파고 {cfg['ranges']['hs'][0]}~{input_bounds(cfg)['hs'][1]:.1f} m (수심 {st['water_depth_h']} m에서 부서지지 않는 파고), "
        f"파형경사 {cfg['ranges']['steepness'][0]}~{cfg['ranges']['steepness'][1]}, "
        f"여유고 {cfg['ranges']['rc'][0]}~{cfg['ranges']['rc'][1]} m (태풍 차바 4.12 m·5.82 m와 그보다 높은 수위의 극한 조건 포함)",
        "",
        "## 등급 기준 (KDS 64 10 10 : 2020 해설 참고 표 4.3-9, 4.3-10)",
        f"- 안전: 월파량 {ot.criterion(cfg, 'safe_max'):g} l/s/m 미만 (차량 운전 가능 2×10⁻⁵, 보행 위험 없음 3×10⁻⁵ m³/m·s 중 엄격한 값)",
        f"- 주의: {ot.criterion(cfg, 'safe_max'):g} 이상 {ot.criterion(cfg, 'danger_min'):g} l/s/m 미만 (사람·차량 통행 위험)",
        f"- 위험: {ot.criterion(cfg, 'danger_min'):g} l/s/m 이상 (배후 민가·공공시설 밀집 지역 허용치 0.01 m³/m·s 초과)",
        f"- 건물 피해 가능: {ot.criterion(cfg, 'building'):g} l/s/m 이상 (가옥 위험 없음 7×10⁻⁵ m³/m·s 초과)",
        f"- {cfg['formula']['zero_q_lsm']} l/s/m 미만은 월파 없음 (참고 표 4.3-10의 가장 작은 값)",
        "",
        "## 한계",
        "- 등급 경계 근처를 일부러 많이 넣어서 위험 비율이 실제 해역의 발생 빈도와 다릅니다. 이 데이터로 만든 모델은 \"조건을 넣으면 등급을 답하는 계산 도구\"입니다.",
        "- 입사각 0°, 바람 미반영, 평균 월파량 기준.",
        "- 단면은 차바 당시 제방 기준이며, TTP 마루 폭 등 일부 치수는 그림용 가정입니다.",
        "- 사면 경사·쇄파 한계(가정)와 수심·입력 범위(결정)에 결과가 크게 좌우됩니다. 수심 고정은 전제 조건이며, 실제로는 수위가 오르면 수심도 깊어집니다.",
    ]
    with open(ot.path_of(cfg, "data_card"), "w", encoding="utf-8") as f:
        f.write("\n".join(card))


# ---------------------------------------------------------------------
def top_up_cells(cfg: dict, picker: Picker, log: list, have: pd.DataFrame) -> pd.DataFrame:
    """격자는 칸별 건수를 고를 수 없으므로, 칸당 최소 건수(cell_min)에 모자란 칸만 Sobol 후보로 채움 (B 예산에서 사용)."""
    need = {}
    for rg in range(3):
        for gd in range(3):
            n = int(((have["regime"] == rg) & (have["grade"] == gd)).sum())
            if n < cfg["sampling"]["cell_min"]:
                need[(rg, gd)] = cfg["sampling"]["cell_min"] - n
    out, k = [], 0
    while any(v > 0 for v in need.values()) and k < 20:
        cand = make_candidates(sobol(cfg["seed"] + 3000 + k, cfg["sampling"]["candidates_per_batch"]), cfg, cfg["ranges"], "B", f"B-보충{k}")
        cand = cand[cand["exclude_reason"] == ""]
        for (rg, gd), v in need.items():
            if v > 0:
                got = picker.take(cand[(cand["regime"] == rg) & (cand["grade"] == gd)], v)
                need[(rg, gd)] -= len(got)
                out.append(got)
        k += 1
    fill = pd.concat(out, ignore_index=True) if out else have.iloc[0:0]
    log.append(("B", "칸 최소 건수 보충", len(fill), ", ".join(f"{cfg['regime']['names'][r]}×{cfg['grades']['names'][g]}" for (r, g) in need)))
    return fill


def build(cfg: dict):
    sc = cfg["sampling"]
    picker = Picker(input_bounds(cfg), sc["dedupe_distance"])
    log: list = []
    a, avail = select_grid(cfg, picker, log)
    c, infeasible = select_edge(cfg, picker, log)
    fill = top_up_cells(cfg, picker, log, pd.concat([a, c]))
    n_b = sc["total"] - len(a) - len(c) - len(fill)
    cap = int(sc["grade_max_ratio"] * sc["total"])
    used = pd.concat([a, c, fill])["grade"].value_counts().to_dict()
    b = select_boundary(cfg, picker, log, n_b, sc["boundary_delta"], "B", cfg["seed"] + 1000,
                        {g: max(0, cap - used.get(g, 0)) for g in range(3)})
    data = assign_split(pd.concat([a, fill, b, c], ignore_index=True), cfg)
    data.insert(0, "id", [f"S{i:05d}" for i in range(len(data))])

    eval_b = select_boundary(cfg, picker, log, sc["eval_boundary_n"], sc["eval_boundary_delta"], "EB", cfg["seed"] + 7000)
    eval_b.insert(0, "id", [f"EB{i:04d}" for i in range(len(eval_b))])
    eval_o = select_outside(cfg, picker, log)
    eval_o.insert(0, "id", [f"EO{i:04d}" for i in range(len(eval_o))])
    return data, eval_b, eval_o, log, avail, infeasible


def file_hash(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="같은 설정·시드로 다시 만들어 기존 파일과 같은지 확인")
    args = ap.parse_args()
    cfg = ot.load_config()
    cfg_hash = file_hash(os.path.join(ot.ROOT, "config.yaml"))
    data, eval_b, eval_o, log, avail, infeasible = build(cfg)

    if args.check:
        old = pd.read_csv(ot.path_of(cfg, "scenarios"))
        same = len(old) == len(data) and np.allclose(old[INPUTS].values, data[INPUTS].values)
        print("재현성:", "같음" if same else "다름")
        return

    os.makedirs(os.path.dirname(ot.path_of(cfg, "scenarios")), exist_ok=True)
    data["config_hash"] = cfg_hash
    data.to_csv(ot.path_of(cfg, "scenarios"), index=False, encoding="utf-8-sig")
    eval_b.to_csv(ot.path_of(cfg, "eval_boundary"), index=False, encoding="utf-8-sig")
    eval_o.to_csv(ot.path_of(cfg, "eval_outside"), index=False, encoding="utf-8-sig")
    pd.DataFrame(log, columns=["단계", "항목", "건수", "사유"]).to_csv(ot.path_of(cfg, "filter_log"), index=False, encoding="utf-8-sig")
    write_reports(cfg, data, eval_b, eval_o, log, avail, infeasible, cfg_hash, file_hash(ot.path_of(cfg, "scenarios")))

    print(f"학습용 {len(data)}건, 경계 시험 {len(eval_b)}건, 범위 밖 시험 {len(eval_o)}건")
    print(cell_table(data, cfg))
    print("리포트:", ot.path_of(cfg, "data_report"))


if __name__ == "__main__":
    main()
