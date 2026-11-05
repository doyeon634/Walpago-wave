# =====================================================================
# overtopping.py – EurOtop (2018) 6장 기반 월파 계산 공통 모듈
#   1차·2차·3차 모두 이 파일의 함수를 씁니다. (벡터 입력 지원)
#
# 변경 이력
#   2026-10-28  최초 작성
#   2026-10-28  등급을 국내 기준(KDS 64 10 10 참고 표 4.3-9, 4.3-10) 고정값으로 변경, 월파 없음 기준을 l/s/m로
#   2026-10-29  넘어오는 파도 수 기준을 식 6.3 하나로 통일 (대시보드 숫자 칸 = 3D 그림)
#     - 평균 월파량: 식 6.5 / 6.6, 장주기 γf 보정 식 6.7, 천단폭 감소 식 6.8, 하한 3.3.7절
#     - 등급: KDS 64 10 10 허용 월파량 (사람·차량·가옥·배후지)
#     - 처오름 식 6.1, 넘어오는 파도 비율 식 6.3, 최대 월파량 식 6.16~6.18, 낙하 거리 식 6.20
#     - 계수 불확실성 몬테카를로 (6.3.1절 표준편차)
# =====================================================================
from __future__ import annotations

import math
import os

import numpy as np
import yaml

G = 9.81
ROOT = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------
def load_config(path: str | None = None) -> dict:
    with open(path or os.path.join(ROOT, "config.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def path_of(cfg: dict, key: str) -> str:
    """config의 paths 항목을 프로젝트 폴더 기준 절대경로로."""
    return os.path.join(ROOT, cfg["paths"][key])


# ---------------------------------------------------------------------
# 기본 물리량
# ---------------------------------------------------------------------
def nominal_diameter(cfg: dict) -> float:
    """TTP 공칭 직경 Dn = (W/ρ)^(1/3)."""
    s = cfg["structure"]
    return (s["ttp_ton"] / s["rho_c"]) ** (1 / 3)


def tp_from_steepness(hs, s):
    """파형경사 sp = 2πHs/(g Tp²) → Tp."""
    return np.sqrt(2 * np.pi * np.asarray(hs, float) / (G * np.asarray(s, float)))


def derived(hs, tp, cfg: dict) -> dict:
    """Tm-1,0, 파장 Lm-1,0, 파형경사 sm-1,0, 쇄파대 파라미터 ξm-1,0, 사용 γf (식 6.7 포함)."""
    hs, tp = np.asarray(hs, float), np.asarray(tp, float)
    st = cfg["structure"]
    tm10 = tp / cfg["formula"]["tp_over_tm10"]
    l_m10 = G * tm10**2 / (2 * np.pi)
    s_m10 = hs / l_m10
    xi = (1 / st["cot_alpha"]) / np.sqrt(s_m10)
    gf0 = st["gamma_f"]
    gf = np.where(xi > 5.0, np.minimum(st["gamma_f_max_long"], gf0 + (xi - 5.0) * (1 - gf0) / 5.0), gf0)
    return {"tm10": tm10, "l_m10": l_m10, "s_m10": s_m10, "xi": xi, "gamma_f_used": gf}


def crest_reduction(hs, cfg: dict):
    """식 6.8 천단폭 감소계수 Cr (최대 1).
    식 6.5는 천단폭 약 3Dn 까지를 이미 포함하므로, 더 넓은 천단(crest_width_m)이 있을 때만 적용."""
    hs = np.asarray(hs, float)
    gc = cfg["structure"].get("crest_width_m")
    if gc is None:
        return np.ones_like(hs)
    return np.minimum(1.0, 3.06 * np.exp(-1.5 * gc / hs))


# ---------------------------------------------------------------------
# 평균 월파량
# ---------------------------------------------------------------------
def mean_discharge(hs, tp, rc, cfg: dict, approach: str | None = None, a=None, b=None) -> np.ndarray:
    """평균 월파량 q (l/s per m).
    a, b를 주면 그 계수로 계산 (몬테카를로용, 브로드캐스트 가능)."""
    hs, rc = np.asarray(hs, float), np.asarray(rc, float)
    approach = approach or cfg["formula"]["label_approach"]
    a = cfg["formula"][approach]["a"] if a is None else a
    b = cfg["formula"][approach]["b"] if b is None else b
    d = derived(hs, tp, cfg)
    gb = cfg["structure"]["gamma_beta"]
    rc_pos = np.maximum(rc, 0.0)
    dimless = a * np.exp(-((b * rc_pos / (hs * d["gamma_f_used"] * gb)) ** 1.3)) * crest_reduction(hs, cfg)
    q = dimless * np.sqrt(G * hs**3) * 1000.0
    return np.where(q < cfg["formula"]["zero_q_lsm"], 0.0, q)


# ---------------------------------------------------------------------
# 등급 (국내 기준, 파고와 관계없는 고정값)
# ---------------------------------------------------------------------
def criterion(cfg: dict, name: str) -> float:
    """config grades의 기준 이름(safe_max 등) 또는 criteria 키 → 값 (l/s per m)."""
    g = cfg["grades"]
    key = g.get(name, name)
    return float(g["criteria"][key])


def limits(hs, cfg: dict) -> dict:
    """등급 경계 (l/s per m). hs는 모양 맞춤용."""
    shape = np.shape(np.asarray(hs, float))
    full = lambda v: np.full(shape, v, dtype=float)  # noqa: E731
    c = cfg["grades"]["criteria"]
    return {
        "safe_max": full(criterion(cfg, "safe_max")),
        "danger_min": full(criterion(cfg, "danger_min")),
        "building": full(criterion(cfg, "building")),
        "pedestrian": full(c["pedestrian"]),
        "vehicle": full(c["vehicle"]),
        "house": full(c["house"]),
        "hinterland": full(c["hinterland"]),
    }


def grade(q, hs, cfg: dict) -> np.ndarray:
    """0 = 안전, 1 = 주의, 2 = 위험."""
    q = np.asarray(q, float)
    lo, hi = criterion(cfg, "safe_max"), criterion(cfg, "danger_min")
    return np.where(q >= hi, 2, np.where(q >= lo, 1, 0)).astype(int)


def building_flag(q, cfg: dict) -> np.ndarray:
    return (np.asarray(q, float) >= criterion(cfg, "building")).astype(int)


def regime(hs, h, cfg: dict) -> np.ndarray:
    """0 = 비쇄파, 1 = 전환, 2 = 쇄파 (구조물 앞 Hm0/h)."""
    r = np.asarray(hs, float) / np.asarray(h, float)
    rg = cfg["regime"]
    return np.where(r < rg["nonbreaking_max"], 0, np.where(r < rg["breaking_min"], 1, 2)).astype(int)


def evaluate(hs, tp, rc, cfg: dict, approach: str | None = None) -> dict:
    q = mean_discharge(hs, tp, rc, cfg, approach)
    out = derived(hs, tp, cfg)
    out.update(q=q, grade=grade(q, hs, cfg), building=building_flag(q, cfg))
    return out


# ---------------------------------------------------------------------
# 경험식 불확실성 (몬테카를로)
# ---------------------------------------------------------------------
def coefficient_samples(cfg: dict, n: int, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """식 6.5 계수 a(0.09), b(1.5)를 정규분포로 뽑음 (6.3.1절 표준편차)."""
    rng = np.random.default_rng(cfg["seed"] if seed is None else seed)
    f = cfg["formula"]
    a = rng.normal(f["mean"]["a"], f["sigma"]["a"], n).clip(1e-4)
    b = rng.normal(f["mean"]["b"], f["sigma"]["b"], n).clip(0.1)
    return a, b


def formula_uncertainty(hs: float, tp: float, rc: float, cfg: dict, n: int) -> dict:
    """한 입력에 대한 경험식 등급 확률과 q의 중앙값·90% 범위."""
    a, b = coefficient_samples(cfg, n)
    q = mean_discharge(np.full(n, hs), np.full(n, tp), np.full(n, rc), cfg, a=a, b=b)
    g = grade(q, np.full(n, hs), cfg)
    return {
        "p_grade": np.bincount(g, minlength=3) / n,
        "p_building": float(np.mean(q >= criterion(cfg, "building"))),
        "q_median": float(np.median(q)),
        "q_p05": float(np.percentile(q, 5)),
        "q_p95": float(np.percentile(q, 95)),
    }


def q_band_over_rc(hs: float, tp: float, rc_grid: np.ndarray, cfg: dict, n: int) -> dict:
    """여유고에 따른 q의 중앙값과 90% 범위 (그래프용)."""
    a, b = coefficient_samples(cfg, n)
    H = np.full((len(rc_grid), n), hs)
    q = mean_discharge(H, np.full_like(H, tp), rc_grid[:, None] * np.ones((1, n)), cfg, a=a[None, :], b=b[None, :])
    return {"median": np.median(q, axis=1), "p05": np.percentile(q, 5, axis=1), "p95": np.percentile(q, 95, axis=1)}


# ---------------------------------------------------------------------
# 월파 모습 (3차 대시보드용)
# ---------------------------------------------------------------------
def runup_2pct(hs: float, tp: float, cfg: dict) -> float:
    """상위 2% 처오름 높이 Ru2% (m). 식 6.1 (평균값), 투수성 코어 상한."""
    d = derived(np.array([hs]), np.array([tp]), cfg)
    xi = float(d["xi"][0])
    gf = cfg["structure"]["gamma_f"]
    gf_s = min(1.0, gf + (xi - 1.8) * (1 - gf) / 8.2) if xi > 1.8 else gf
    r = min(1.65 * gf * xi, gf_s * (4.0 - 1.5 / math.sqrt(xi)), cfg["formula"]["runup_cap_permeable"])
    return float(r * hs)


def overtopping_events(hs: float, tp: float, q_lsm: float, rc: float, cfg: dict, hours: float = 1.0) -> dict:
    """넘어오는 파도의 수와 물의 양 (대시보드 숫자 칸과 3D 그림이 함께 쓰는 하나의 기준).

    - 넘어오는 파도 비율 Pov: 식 6.3 (처오름이 레일리 분포를 따른다는 가정, Ru2% 는 식 6.1).
      EurOtop은 식 6.3이 식 6.4 보다 넘는 파도 수를 많게 준다고 하므로 안전 쪽 기준이다.
      3D 그림의 파도별 처오름도 같은 분포를 쓰므로 두 결과가 일치한다.
    - 파도별 월파량: Weibull 분포 (식 6.16~6.18), 부피 단위 L/m.
    """
    tm = tp / cfg["formula"]["tp_over_tm"]
    tm10 = tp / cfg["formula"]["tp_over_tm10"]
    nw = hours * 3600.0 / tm
    ru2 = runup_2pct(hs, tp, cfg)
    out = {"nw": nw, "ru2": ru2, "pov": 0.0, "now": 0.0, "a": 0.0, "b": 0.85, "volume_l": 0.0, "vmax_l": 0.0}
    if q_lsm <= 0 or ru2 <= 0:
        return out
    pov = 1.0 if rc <= 0 else math.exp(-((math.sqrt(-math.log(0.02)) * rc / ru2) ** 2))
    if pov <= 0:
        return out
    now = pov * nw
    q3 = q_lsm / 1000.0
    total = q3 * hours * 3600.0
    b = 0.85 + 1500 * (q3 / (G * hs * tm10)) ** 1.3
    a = (q3 * tm / pov) / math.gamma(1 + 1 / b)
    vmax = a * math.log(now) ** (1 / b) if now > 1 else total
    out.update(pov=pov, now=now, a=a, b=b, volume_l=total * 1000.0, vmax_l=min(vmax, total) * 1000.0)
    return out


def landing_distance(frac_beyond: float, hs: float, tp: float, drop_y: float) -> float:
    """넘친 물 중 frac_beyond 비율이 이 거리보다 멀리 떨어짐 (m, 파라펫 뒷면 기준). 식 6.20, β = 0."""
    sop = 2 * np.pi * hs / (G * tp**2)
    return float(max(0.0, -0.77 * hs * math.log(frac_beyond) + 2.7 * drop_y * sop**0.15))


# ---------------------------------------------------------------------
# 모델 특성 (2차·3차 공통)
# ---------------------------------------------------------------------
def model_features(hs, tp, rc) -> "np.ndarray":
    """사용자 입력 3개(Hs, Tp, Rc)로 만든 모델 특성.
    입력 자체 3개 + 같은 입력에서 계산한 무차원 여유고 Rc/Hs, 파형경사 sp.
    (새 정보를 넣는 것이 아니라 Random Forest가 경계를 쉽게 찾도록 돕는 변환)"""
    hs, tp, rc = (np.asarray(v, float) for v in (hs, tp, rc))
    return np.column_stack([hs, tp, rc, rc / hs, 2 * np.pi * hs / (G * tp**2)])


FEATURE_NAMES = ["hs", "tp", "rc", "rc_over_hs", "s_p"]
