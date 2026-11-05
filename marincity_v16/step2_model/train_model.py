# =====================================================================
# [2차] train_model.py – Random Forest 학습·평가
#   입력: 파고 Hs, 주기 Tp, 여유고 Rc (모델 안에서는 여기서 계산한 Rc/Hs, 파형경사도 함께 사용)
#   출력: ① 위험 등급(안전/주의/위험)과 등급별 확률  ② 건물 피해 가능 여부와 확률
#
# 실행 (프로젝트 폴더에서, 1차를 먼저 실행):
#   py step2_model/train_model.py
#
# 절차
#   1. 학습 세트로 학습, 검증 세트로 하이퍼파라미터 선택 (위험 재현율 → 정확도 순)
#   2. 확률 보정 비교: 원래 확률 vs 보정 확률(isotonic) 중 검증 세트 Brier 점수가 낮은 쪽 채택
#      (3차 대시보드가 확률 %를 보여 주므로 "70%라고 하면 실제로 70% 맞는지"를 확인)
#   3. 학습+검증 세트로 최종 학습
#   4. 시험 세트, 경계 시험, 범위 밖 시험으로 평가
#
# 산출물: step2_model/output/
#   rf_model.joblib    모델 묶음 {"grade": 등급 모델, "building": 건물 피해 모델}
#   model_meta.json    입력 범위, 선택한 설정, 성능 요약 (대시보드가 읽음)
#   model_report.md    평가 리포트
#   calibration.png    확률 신뢰도 그림
#
# 변경 이력
#   2026-10-28  최초 작성
#   2026-10-28  모델 특성에 무차원 여유고 Rc/Hs, 파형경사 추가 (사용자 입력은 그대로 3개)
# =====================================================================
from __future__ import annotations

import json
import logging
import os
import sys

import joblib
import matplotlib

matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import overtopping as ot  # noqa: E402

INPUTS = ["hs", "tp", "rc"]


# ---------------------------------------------------------------------
# 모델
# ---------------------------------------------------------------------
def make_rf(cfg, depth, leaf):
    m = cfg["model"]
    return RandomForestClassifier(n_estimators=m["n_estimators"], max_depth=depth, min_samples_leaf=leaf,
                                  random_state=cfg["seed"], n_jobs=-1)


def fit(cfg, x, y, depth, leaf, calibrated):
    base = make_rf(cfg, depth, leaf)
    if calibrated:
        return CalibratedClassifierCV(base, method="isotonic", cv=5).fit(x, y)
    return base.fit(x, y)


# ---------------------------------------------------------------------
# 지표
# ---------------------------------------------------------------------
def brier(proba: np.ndarray, y: np.ndarray, n_cls: int) -> float:
    onehot = np.eye(n_cls)[y]
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def ece(proba: np.ndarray, y: np.ndarray, bins: int) -> float:
    """예측 등급의 확률(가장 큰 확률)과 실제 정답률의 평균 차이."""
    conf, pred = proba.max(axis=1), proba.argmax(axis=1)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - (pred[m] == y[m]).mean())
    return float(e)


def X(df):
    """데이터프레임 → 모델 특성 (overtopping.model_features)."""
    return ot.model_features(df["hs"], df["tp"], df["rc"])


def full_proba(model, x, n_cls):
    """학습 데이터에 없는 등급이 있어도 열 수를 맞춤."""
    p = model.predict_proba(x)
    out = np.zeros((len(x), n_cls))
    out[:, model.classes_.astype(int)] = p
    return out


def evaluate(models, df, cfg) -> dict:
    if df.empty:
        return {}
    gn, bins = cfg["grades"]["names"], cfg["model"]["calibration_bins"]
    x = X(df)
    yg, yb = df["grade"].values.astype(int), df["building"].values.astype(int)
    pg, pb = full_proba(models["grade"], x, 3), full_proba(models["building"], x, 2)
    g_pred, b_pred = pg.argmax(1), pb.argmax(1)
    rec = recall_score(yg, g_pred, labels=[0, 1, 2], average=None, zero_division=0)
    return {
        "건수": int(len(df)),
        "등급 정확도": round(accuracy_score(yg, g_pred), 4),
        "등급 macro-F1": round(f1_score(yg, g_pred, average="macro"), 4),
        "등급별 재현율": {gn[i]: round(float(rec[i]), 4) for i in range(3)},
        "위험을 안전으로 오판": int(((yg == 2) & (g_pred == 0)).sum()),
        "위험을 안전·주의로 과소판정": int(((yg == 2) & (g_pred < 2)).sum()),
        "등급 Brier 점수": round(brier(pg, yg, 3), 4),
        "등급 ECE": round(ece(pg, yg, bins), 4),
        "건물 피해 정확도": round(accuracy_score(yb, b_pred), 4),
        "건물 피해 놓침(실제 있음→없음)": int(((yb == 1) & (b_pred == 0)).sum()),
        "건물 피해 Brier 점수": round(brier(pb, yb, 2), 4),
        "혼동행렬(행=실제, 열=예측)": confusion_matrix(yg, g_pred, labels=[0, 1, 2]).tolist(),
    }


def save_calibration_plot(models, df, cfg, path):
    plt.rcParams["font.family"] = ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]
    bins = cfg["model"]["calibration_bins"]
    pg = full_proba(models["grade"], X(df), 3)
    y = df["grade"].values.astype(int)
    colors = ["#2F8F5B", "#D9A21B", "#C63D2F"]
    names = ["safe", "caution", "danger"]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], color="#888", ls="--", lw=1)
    edges = np.linspace(0, 1, bins + 1)
    for c in range(3):
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (pg[:, c] > lo) & (pg[:, c] <= hi)
            if m.sum() >= 5:
                xs.append(pg[m, c].mean())
                ys.append((y[m] == c).mean())
        ax.plot(xs, ys, marker="o", color=colors[c], label=names[c])
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed frequency")
    ax.set_title("calibration (test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------
def main():
    cfg = ot.load_config()
    data = pd.read_csv(ot.path_of(cfg, "scenarios"))
    eval_b = pd.read_csv(ot.path_of(cfg, "eval_boundary"))
    eval_o = pd.read_csv(ot.path_of(cfg, "eval_outside"))
    tr, va, te = (data[data["split"] == s] for s in ("train", "val", "test"))
    mcfg, gn = cfg["model"], cfg["grades"]["names"]

    # 1. 하이퍼파라미터 (등급 모델 기준)
    search = []
    for depth in mcfg["max_depth_grid"]:
        for leaf in mcfg["min_samples_leaf_grid"]:
            m = fit(cfg, X(tr), tr["grade"], depth, leaf, False)
            p = full_proba(m, X(va), 3)
            pred, y = p.argmax(1), va["grade"].values
            r_d = recall_score(y, pred, labels=[2], average=None, zero_division=0)[0]
            search.append({"max_depth": depth, "min_samples_leaf": leaf, "위험 재현율": round(float(r_d), 4),
                           "정확도": round(accuracy_score(y, pred), 4), "Brier": round(brier(p, y, 3), 4)})
    best = max(search, key=lambda s: (s["위험 재현율"], s["정확도"]))
    depth, leaf = best["max_depth"], best["min_samples_leaf"]

    # 2. 확률 보정 비교 (검증 세트 Brier)
    calib = {}
    for target, k in (("grade", 3), ("building", 2)):
        scores = {}
        for cal in (False, True):
            m = fit(cfg, X(tr), tr[target], depth, leaf, cal)
            scores[cal] = brier(full_proba(m, X(va), k), va[target].values.astype(int), k)
        calib[target] = {"보정 전 Brier": round(scores[False], 4), "보정 후 Brier": round(scores[True], 4),
                         "보정 사용": bool(scores[True] < scores[False])}

    # 3. 최종 학습 (학습 + 검증)
    trval = pd.concat([tr, va])
    models = {t: fit(cfg, X(trval), trval[t], depth, leaf, calib[t]["보정 사용"]) for t in ("grade", "building")}

    # 4. 평가
    report = {
        "선택한 설정": {"max_depth": depth, "min_samples_leaf": leaf, "n_estimators": mcfg["n_estimators"]},
        "하이퍼파라미터 탐색(검증 세트)": search,
        "확률 보정": calib,
        "시험 세트": evaluate(models, te, cfg),
        "경계 시험 세트": evaluate(models, eval_b, cfg),
        "범위 밖 시험 세트": evaluate(models, eval_o, cfg),
    }
    g = models["grade"]
    rfs = [g] if isinstance(g, RandomForestClassifier) else [c.estimator for c in g.calibrated_classifiers_]
    imp = np.mean([r.feature_importances_ for r in rfs], axis=0)
    report["특성 중요도"] = dict(zip(ot.FEATURE_NAMES, np.round(imp, 4).tolist()))

    cv_x, cv_y = X(trval), trval["grade"].values.astype(int)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg["seed"])
    cv_accuracy, cv_f1 = [], []
    for train_idx, val_idx in splitter.split(cv_x, cv_y):
        cv_model = fit(cfg, cv_x[train_idx], cv_y[train_idx], depth, leaf, calib["grade"]["보정 사용"])
        cv_pred = cv_model.predict(cv_x[val_idx])
        cv_accuracy.append(accuracy_score(cv_y[val_idx], cv_pred))
        cv_f1.append(f1_score(cv_y[val_idx], cv_pred, average="macro"))
    report["5겹 교차검증"] = {
        "정확도 평균": round(float(np.mean(cv_accuracy)), 4),
        "정확도 표준편차": round(float(np.std(cv_accuracy)), 4),
        "macro-F1 평균": round(float(np.mean(cv_f1)), 4),
        "macro-F1 표준편차": round(float(np.std(cv_f1)), 4),
        "겹별 정확도": [round(float(score), 4) for score in cv_accuracy],
    }

    out_dir = os.path.dirname(ot.path_of(cfg, "model"))
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(models, ot.path_of(cfg, "model"), compress=3)
    save_calibration_plot(models, te, cfg, ot.path_of(cfg, "calibration_png"))

    meta = {
        "inputs": INPUTS,
        "features": ot.FEATURE_NAMES,
        "grade_names": gn,
        "train_bounds": {k: [float(data[k].min()), float(data[k].max())] for k in INPUTS},
        "steepness_range": [float(data["s_p"].min()), float(data["s_p"].max())],
        "settings": report["선택한 설정"],
        "calibration": calib,
        "test": report["시험 세트"],
        "boundary_test": report["경계 시험 세트"],
        "outside_test": report["범위 밖 시험 세트"],
    }
    with open(ot.path_of(cfg, "model_meta"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)

    t = report["시험 세트"]
    cv = report["5겹 교차검증"]
    lines = [
        "# [2차] Random Forest 평가 리포트",
        "",
        "## 요약 (시험 세트)",
        f"- 등급 정확도 {t['등급 정확도']:.1%}, 등급 macro-F1 {t['등급 macro-F1']:.1%}, 위험을 안전으로 오판 {t['위험을 안전으로 오판']}건, 위험을 안전·주의로 과소판정 {t['위험을 안전·주의로 과소판정']}건",
        f"- 5겹 교차검증(trval): 정확도 {cv['정확도 평균']:.1%} ± {cv['정확도 표준편차']:.1%}, macro-F1 {cv['macro-F1 평균']:.1%} ± {cv['macro-F1 표준편차']:.1%}; 겹별 정확도 " + ", ".join(f"{score:.1%}" for score in cv["겹별 정확도"]),
        "- 등급별 재현율: " + ", ".join(f"{k} {v:.1%}" for k, v in t["등급별 재현율"].items()),
        f"- 확률 신뢰도: Brier {t['등급 Brier 점수']} (0에 가까울수록 좋음), ECE {t['등급 ECE']} (예측 확률과 실제 정답률의 평균 차이)",
        f"- 건물 피해 정확도 {t['건물 피해 정확도']:.1%}, 놓친 건수 {t['건물 피해 놓침(실제 있음→없음)']}건",
        "",
        "## 전체 결과",
        "```json",
        json.dumps(report, ensure_ascii=False, indent=1),
        "```",
        "",
        "## 확률 신뢰도 그림",
        "점선(대각선)에 가까울수록 확률 %가 믿을 만합니다.",
        "",
        "![calibration](calibration.png)",
        "",
        "범위 밖 시험 세트는 학습 범위를 벗어난 입력입니다. 대시보드는 이 경우 경고와 함께 경험식 결과를 기준으로 보여 줍니다.",
    ]
    with open(ot.path_of(cfg, "model_report"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines[:9]))
    print("리포트:", ot.path_of(cfg, "model_report"))


if __name__ == "__main__":
    main()
