# =====================================================================
# [3차] app.py – 마린시티 호안 월파 위험 예측 대시보드
#   파고·주기·여유고를 넣으면 2차에서 학습한 Random Forest 로 위험 등급과 등급별 확률을 보여 주고,
#   EurOtop 경사식 공식으로 월파량과 월파 모습(3D·2D)을 함께 보여 줍니다.
#   등급 기준은 국내 기준 KDS 64 10 10 : 2020 허용 월파량입니다.
#
# 실행 (step3_dashboard 폴더에서, 1차·2차를 먼저 실행):
#   py -m streamlit run app.py
#
# 화면 순서
#   [위]  입력(파고·주기·여유고) + 위험 등급 카드 | 월파 모습(3D·2D)
#         위험 등급 게이지(AI·경험식) | 월파량
#         얼마나 넘치나 (솟구침, 넘어온 물, 파도 수 …)
#   [아래] 허용 월파량 · 적용 공식 · 여유고–월파량 그래프 · 파라미터 · 전제 조건 · 모델 성능
#
# 변경 이력
#   2026-10-28  최초 작성
#   2026-10-29  (v7) 흰 배경, 월파 모습을 위험 표시 위로, 입력은 파고·주기·여유고만 위에 두고 나머지는 아래로,
#               차바 당시 제방(기립식 차수벽 없음)
#   2026-10-29  (v8) 첫 화면에 입력·그림·위험이 함께 보이게: 그림 높이 460 px, 위험 등급 카드를 슬라이더 아래로
#   2026-10-29  (v9) 3D·2D 그림 품질 개선 (부드러운 바다, 사진 같은 물보라 기둥)
#   2026-10-29  (v10) 처오름이 바다 면과 이어지게 수정 (떠 있는 물 판 제거)
#   2026-10-31  (v11) 상단에 팀 이름(월파고)과 마크, 단면에 기초사석·피복석 채움, 틈에서 솟는 물보라 제거
#   2026-10-31  (v12) 호안 꼭대기 위험 등급 색을 진하게 (3D·2D)
#   2026-11-01  (v13) 2D 실시간 입자 시뮬레이션 추가 (컴퓨터 성능에 맞춰 입자 수 최대)
#   2026-11-03  (v14) '태풍 차바(2016) 당시 조건으로' 버튼 추가
#   2026-11-03  (v15) 버튼을 슬라이더 아래 작은 '차바' 버튼으로, 2D 시뮬레이션을 시제품 방식(규칙파)으로 되돌리고 해상도·속도 개선
#   2026-11-03  (v16) 2D 시뮬레이션: 호안을 넘으려는 순간 물이 사라지던 문제 수정
# =====================================================================
from __future__ import annotations

import json
import math
import os
import sys

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import overtopping as ot  # noqa: E402
import section2d  # noqa: E402
import section_canvas  # noqa: E402
import section_sim  # noqa: E402
import threescene  # noqa: E402

GRADE_COLORS = ["#2E9E73", "#D9A21B", "#D64541"]
GRADE_MEANING = [
    "사람과 차량이 통행해도 되는 수준입니다.",
    "사람·차량 통행이 위험한 수준입니다. 보행로와 도로 통제가 필요합니다.",
    "배후 주거·상업 지역에 피해가 우려되는 수준입니다.",
]
TEXT, MUTED, PANEL, LINE = "#1F2A37", "#5B6B7B", "#F3F5F7", "#DDE3E8"
VIEW_H = 460   # 월파 모습(3D·2D) 높이 (픽셀)

st.set_page_config(page_title="마린시티 호안 월파 위험 예측", page_icon="🌊", layout="wide")
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+KR:wght@400;600&display=swap');
html, body, [class*="css"], .stMarkdown, .stSlider, .stRadio {{ font-family: 'IBM Plex Sans KR', 'Malgun Gothic', sans-serif; }}
.block-container {{ padding-top: 1.4rem; padding-bottom: 2rem; }}
div[data-testid="stSlider"] {{ margin-bottom: -0.4rem; }}
h1.t {{ font-size: 1.5rem; font-weight: 600; margin: 0; color: {TEXT}; }}
.brand {{ display: flex; align-items: center; gap: 14px; margin: 0 0 .3rem 0; }}
.brand .team {{ font-size: 1.7rem; font-weight: 800; color: #12395A; letter-spacing: -0.5px; }}
.brand .bar {{ width: 1px; height: 30px; background: {LINE}; }}
.hh {{ font-size: 1.08rem; font-weight: 600; color: {TEXT}; margin: .7rem 0 .2rem 0; }}
.hh span {{ color: {MUTED}; font-weight: 400; font-size: .85rem; }}
.card {{ border-left: 5px solid; padding: .45rem 0 .45rem 1rem; }}
.card .lbl {{ font-size: .8rem; color: {MUTED}; }}
.card .grade {{ font-size: 2.1rem; font-weight: 700; line-height: 1.15; }}
.card .mean {{ font-size: .95rem; color: {TEXT}; }}
.tile .lbl {{ font-size: .8rem; color: {MUTED}; }}
.tile .val {{ font-size: 1.5rem; font-weight: 600; color: {TEXT}; font-variant-numeric: tabular-nums; }}
.tile .val small {{ font-size: .88rem; font-weight: 400; color: {MUTED}; }}
.tiles {{ display: flex; gap: 2rem; flex-wrap: wrap; margin-top: .3rem; }}
.pcts {{ display: flex; justify-content: center; gap: 1rem; font-size: .9rem; margin-top: -.4rem; }}
.pcts b {{ font-variant-numeric: tabular-nums; }}
.form {{ background: {PANEL}; border-radius: 6px; padding: .5rem .9rem; margin: .1rem 0 .4rem 0; color: {TEXT}; }}
.premise {{ background: {PANEL}; border-radius: 6px; padding: .7rem 1rem; color: {TEXT}; font-size: .92rem; }}
.premise .row {{ display: flex; flex-wrap: wrap; gap: .3rem 1.4rem; }}
hr.sep {{ border: none; border-top: 1px solid {LINE}; margin: 1.6rem 0 .4rem 0; }}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# 불러오기
# ---------------------------------------------------------------------
@st.cache_resource
def load():
    cfg = ot.load_config()
    models = joblib.load(ot.path_of(cfg, "model"))
    with open(ot.path_of(cfg, "model_meta"), encoding="utf-8") as f:
        meta = json.load(f)
    return cfg, models, meta


try:
    cfg, models, meta = load()
except FileNotFoundError as e:
    st.error(f"필요한 파일이 없습니다: {e.filename}\n\n프로젝트 폴더에서 1차(step1_data/generate_data.py)와 2차(step2_model/train_model.py)를 먼저 실행하세요.")
    st.stop()

GN = cfg["grades"]["names"]
S = cfg["structure"]
CR = cfg["grades"]["criteria"]
H = S["water_depth_h"]
TB = meta["train_bounds"]
S_LO, S_HI = cfg["ranges"]["steepness"]


def full_proba(model, x, k):
    p = model.predict_proba(x)
    out = np.zeros((len(x), k))
    out[:, model.classes_.astype(int)] = p
    return out


def fmt_q(q: float) -> str:
    if q <= 0:
        return "0"
    return f"{q:,.0f}" if q >= 100 else f"{q:.3g}"


def tp_bounds(hs: float) -> tuple[float, float]:
    """현재 파고에서 가능한 주기 범위 (파형경사 범위로부터), 0.1 s 단위."""
    lo = float(ot.tp_from_steepness(hs, S_HI))
    hi = float(ot.tp_from_steepness(hs, S_LO))
    return math.ceil(lo * 10) / 10, math.floor(hi * 10) / 10


def show_html(html: str, height: int):
    if hasattr(st, "iframe"):          # Streamlit 최신 버전
        st.iframe(html, height=height)
    else:                              # 이전 버전
        components.html(html, height=height)


def gauge(p: np.ndarray, title: str) -> go.Figure:
    """반원 게이지: 바늘 = 안전 0 · 주의 50 · 위험 100 을 등급 확률로 가중 평균한 위치."""
    score = float(50 * p[1] + 100 * p[2])
    fig = go.Figure()
    r0, r1 = 0.62, 1.0
    for i, (lo, hi) in enumerate([(0, 100 / 3), (100 / 3, 200 / 3), (200 / 3, 100)]):
        th = np.radians(np.linspace(180 - lo * 1.8, 180 - hi * 1.8, 40))
        xs = np.r_[r1 * np.cos(th), r0 * np.cos(th[::-1])]
        ys = np.r_[r1 * np.sin(th), r0 * np.sin(th[::-1])]
        fig.add_scatter(x=xs, y=ys, fill="toself", fillcolor=GRADE_COLORS[i], line=dict(color="#FFFFFF", width=3),
                        mode="lines", hoverinfo="skip", showlegend=False)
        mid = np.radians(180 - (lo + hi) / 2 * 1.8)
        fig.add_annotation(x=1.17 * np.cos(mid), y=1.12 * np.sin(mid), text=GN[i], showarrow=False, font=dict(color=GRADE_COLORS[i], size=14))
    ang = np.radians(180 - score * 1.8)
    fig.add_scatter(x=[0, 0.9 * np.cos(ang)], y=[0, 0.9 * np.sin(ang)], mode="lines", line=dict(color=TEXT, width=5),
                    hoverinfo="skip", showlegend=False)
    fig.add_scatter(x=[0], y=[0], mode="markers", marker=dict(size=16, color=TEXT), hoverinfo="skip", showlegend=False)
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=14, color=TEXT)),
        height=210, margin=dict(l=10, r=10, t=36, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[-1.35, 1.35], visible=False, fixedrange=True),
        yaxis=dict(range=[-0.12, 1.3], visible=False, scaleanchor="x", fixedrange=True),
    )
    return fig


def pct_row(p: np.ndarray) -> str:
    return '<div class="pcts">' + "".join(
        f'<span style="color:{GRADE_COLORS[i]}">{GN[i]} <b>{p[i] * 100:.1f}%</b></span>' for i in range(3)) + "</div>"


# ---------------------------------------------------------------------
# 머리말 · 입력 · 월파 모습
# ---------------------------------------------------------------------
LOGO_A = (  # 팀 마크 (시안 A · 넘는 파도)
    '<svg width="52" height="52" viewBox="0 0 96 96" role="img" aria-label="월파고 마크">'
    '<rect x="2" y="2" width="92" height="92" rx="22" fill="#12395A"/>'
    '<rect x="64" y="50" width="10" height="34" rx="2" fill="#D9682B"/>'
    '<path d="M12 74 C 24 74, 28 44, 48 36 C 62 30, 76 34, 80 46 C 74 41, 64 42, 61 52" fill="none" stroke="#FFFFFF" '
    'stroke-width="7" stroke-linecap="round" stroke-linejoin="round"/>'
    '<path d="M12 84 C 22 80, 32 80, 42 84 C 50 87, 56 87, 62 84" fill="none" stroke="#9CC8E8" stroke-width="4" stroke-linecap="round"/>'
    '<circle cx="78" cy="26" r="3.2" fill="#FFFFFF"/><circle cx="86" cy="34" r="2.4" fill="#FFFFFF"/><circle cx="70" cy="20" r="2" fill="#FFFFFF"/>'
    "</svg>"
)
st.markdown(
    '<div class="brand">' + LOGO_A + '<span class="team">월파고</span><span class="bar"></span>'
    '<h1 class="t">마린시티 호안 월파 위험 예측</h1></div>',
    unsafe_allow_html=True,
)

hs_lo, hs_hi = math.ceil(TB["hs"][0] * 20) / 20, math.floor(TB["hs"][1] * 20) / 20
rc_lo, rc_hi = float(TB["rc"][0]), float(TB["rc"][1])
dflt = cfg["dashboard"]["default"]
st.session_state.setdefault("hs", float(np.clip(dflt["hs"], hs_lo, hs_hi)))
st.session_state.setdefault("tp", float(dflt["tp"]))
st.session_state.setdefault("rc", float(np.clip(dflt["rc"], rc_lo, rc_hi)))

CHABA = cfg["dashboard"]["chaba"]


def apply_chaba():
    """태풍 차바(2016) 당시 조건으로 슬라이더를 맞춤 (위젯이 그려지기 전에 실행되는 콜백)."""
    st.session_state.hs = float(np.clip(CHABA["hs"], hs_lo, hs_hi))
    st.session_state.tp = float(CHABA["tp"])
    st.session_state.rc = float(np.clip(CHABA["rc"], rc_lo, rc_hi))


col_in, col_view = st.columns([1, 2.7], gap="large")
with col_in:
    st.markdown('<div class="hh">입력 조건</div>', unsafe_allow_html=True)
    hs = st.slider(f"파고 Hs [m] (호안 바로 앞 파도 높이, {hs_lo:g}~{hs_hi:g} m)", hs_lo, hs_hi, step=0.05, key="hs")
    t_lo, t_hi = tp_bounds(hs)
    st.session_state.tp = float(np.clip(st.session_state.tp, t_lo, t_hi))
    tp = st.slider(f"주기 Tp [s] (파도 하나가 지나가는 시간, 이 파고에서 {t_lo:g}~{t_hi:g} s)", t_lo, t_hi, step=0.1, key="tp")
    rc = st.slider(f"여유고 Rc [m] (호안 꼭대기와 바닷물 높이의 차이, {rc_lo:g}~{rc_hi:g} m)", rc_lo, rc_hi, step=0.05, key="rc")
    st.button("차바", on_click=apply_chaba,
              help=f"태풍 차바(2016) 당시 조건: 파고 {CHABA['hs']} m · 주기 {CHABA['tp']:g}초 · 여유고 {CHABA['rc']} m")

# 계산
xf = ot.model_features([hs], [tp], [rc])
p_ai = full_proba(models["grade"], xf, 3)[0]
p_bld_ai = float(full_proba(models["building"], xf, 2)[0][1])
g_ai = int(np.argmax(p_ai))
q = float(ot.mean_discharge([hs], [tp], [rc], cfg)[0])
unc = ot.formula_uncertainty(hs, tp, rc, cfg, cfg["dashboard"]["mc_samples"])
d = ot.derived(np.array([hs]), np.array([tp]), cfg)
gf = float(d["gamma_f_used"][0])
_, sm = section2d.build_section(cfg, hs, tp, rc, False, q, g_ai)

with col_view:
    st.markdown('<div class="hh">월파 모습</div>', unsafe_allow_html=True)
    view = st.radio("보기", ["3D 모델", "2D 시뮬레이션", "2D 그림"], horizontal=True, label_visibility="collapsed")
    if view == "3D 모델":   # 보이는 그림만 만든다
        show_html(threescene.build_html(threescene.params(cfg, hs, tp, rc)), VIEW_H)
    elif view == "2D 시뮬레이션":   # 물을 입자로 나눠 직접 계산 (시제품)
        show_html(section_sim.build_html(round(hs, 2), round(tp, 1), round(rc, 2), q), VIEW_H)
    else:
        show_html(section_canvas.build_html(section_canvas.params(cfg, hs, tp, rc)), VIEW_H)

# 위험 등급 카드: 슬라이더 바로 아래 (첫 화면에서 입력·그림·위험이 함께 보이게)
with col_in:
    st.markdown(
        f'<div class="card" style="border-color:{GRADE_COLORS[g_ai]};margin-top:.6rem">'
        f'<div class="lbl">위험 등급 (AI 예측)</div>'
        f'<div class="grade" style="color:{GRADE_COLORS[g_ai]}">{GN[g_ai]}</div>'
        f'<div class="mean">{GRADE_MEANING[g_ai]}</div>'
        f'<div class="lbl" style="margin-top:.35rem">1시간에 파도 {sm["events"]["nw"]:,.0f}개 중 약 {sm["events"]["now"]:,.0f}개가 호안을 넘음 · '
        f'건물 피해 가능성 {p_bld_ai:.0%}</div>'
        "</div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------
# 위험 표시
# ---------------------------------------------------------------------
c2, c3 = st.columns([2.2, 1.3], gap="large")
with c2:
    g1, g2 = st.columns(2)
    with g1:
        st.plotly_chart(gauge(p_ai, "AI (Random Forest)"), width="stretch", config={"displayModeBar": False})
        st.markdown(pct_row(p_ai), unsafe_allow_html=True)
    with g2:
        st.plotly_chart(gauge(unc["p_grade"], "경험식 (공식 오차 반영)"), width="stretch", config={"displayModeBar": False})
        st.markdown(pct_row(unc["p_grade"]), unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="form">{sm["form"]}</div>', unsafe_allow_html=True)
    q_med = unc["q_median"]
    rng_txt = "월파 없음" if unc["q_p95"] <= 0 else f"{fmt_q(unc['q_p05'])} ~ {fmt_q(unc['q_p95'])} <small>l/s/m</small>"
    st.markdown(
        f'<div class="tile"><div class="lbl">평균 월파량 (호안 1 m당 1초에 넘어오는 물)</div>'
        f'<div class="val">{"월파 없음" if q_med <= 0 else fmt_q(q_med) + " <small>l/s/m</small>"}</div></div>'
        f'<div class="tile" style="margin-top:.6rem"><div class="lbl">월파량 90% 범위 (공식 계수 오차 반영)</div><div class="val">{rng_txt}</div></div>'
        f'<div class="tile" style="margin-top:.6rem"><div class="lbl">적용 공식</div></div>',
        unsafe_allow_html=True,
    )
    st.latex(r"\frac{q}{\sqrt{g\,H_{m0}^{3}}} = 0.09\,e^{-\left(1.5\,\frac{R_c}{H_{m0}\,\gamma_f\,\gamma_\beta}\right)^{1.3}}")

ev, d10 = sm["events"], sm["dist"].get(0.1)
st.markdown(
    '<div class="tiles">'
    f'<div class="tile"><div class="lbl">호안 위로 솟구치는 높이 (큰 파도)</div><div class="val">{max(sm["z_ru"] - sm["crest"], 0) * 1.05:.1f} <small>m</small></div></div>'
    f'<div class="tile"><div class="lbl">1시간 동안 넘어온 물 (호안 1 m당)</div><div class="val">{ev["volume_l"]:,.0f} <small>L</small></div></div>'
    f'<div class="tile"><div class="lbl">1시간 동안 호안을 넘은 파도</div><div class="val">{ev["now"]:,.0f} <small>개 / {ev["nw"]:,.0f}개 중</small></div></div>'
    f'<div class="tile"><div class="lbl">가장 큰 파도 한 번의 물 (호안 1 m당)</div><div class="val">{ev["vmax_l"]:,.0f} <small>L</small></div></div>'
    f'<div class="tile"><div class="lbl">넘친 물 대부분이 떨어지는 거리 (방파벽 뒤)</div><div class="val">{"-" if d10 is None else f"{d10:.1f}"} <small>{"" if d10 is None else "m"}</small></div></div>'
    "</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# 아래: 허용 월파량 · 계산 과정 · 그래프 · 파라미터 · 전제 조건 · 성능
# ---------------------------------------------------------------------
st.markdown('<hr class="sep">', unsafe_allow_html=True)
st.markdown('<div class="hh">허용 월파량 <span>(국내 기준 KDS 64 10 10 : 2020 해설)</span></div>', unsafe_allow_html=True)
ok = lambda lim: "넘음" if q >= lim else "이내"  # noqa: E731
st.table(pd.DataFrame(
    [
        ["사람 기준", "보행해도 위험 없음", "3×10⁻⁵", f"{CR['pedestrian']:g}", ok(CR["pedestrian"])],
        ["차량 기준", "자동차 운전 가능", "2×10⁻⁵", f"{CR['vehicle']:g}", ok(CR["vehicle"])],
        ["가옥 기준", "가옥 위험 없음", "7×10⁻⁵", f"{CR['house']:g}", ok(CR["house"])],
        ["아무것도 없을 때 기준", "배후에 민가·공공시설 밀집", "0.01", f"{CR['hinterland']:g}", ok(CR["hinterland"])],
    ],
    columns=["기준", "뜻", "허용 월파량 (m³/m·s)", "허용 월파량 (l/s/m)", f"현재 ({fmt_q(q)} l/s/m)"],
).set_index("기준"))
st.caption(f"등급: 안전 {CR['vehicle']:g} l/s/m 미만 · 주의 {CR['vehicle']:g} 이상 {CR['hinterland']:g} 미만 · 위험 {CR['hinterland']:g} 이상")

st.markdown('<div class="hh">계산 과정 <span>(현재 값을 넣은 평균 월파량)</span></div>', unsafe_allow_html=True)
a, b = cfg["formula"]["mean"]["a"], cfg["formula"]["mean"]["b"]
gb = S["gamma_beta"]
dim_raw = a * math.exp(-((b * max(rc, 0) / (hs * gf * gb)) ** 1.3))


def sci(v: float) -> str:
    if v == 0:
        return "0"
    e = int(math.floor(math.log10(abs(v))))
    return rf"{v / 10**e:.3f}\times10^{{{e}}}"


st.latex(rf"= 0.09\,\exp\!\left[-\left(1.5\times\frac{{{rc:.2f}}}{{{hs:.2f}\times{gf:.3f}\times{gb:.2f}}}\right)^{{1.3}}\right] = {sci(dim_raw)}")
st.latex(rf"q = {sci(dim_raw)}\times\sqrt{{9.81\times{hs:.2f}^{{3}}}}\times1000 = {dim_raw * math.sqrt(ot.G * hs**3) * 1000:.3g}\ \mathrm{{l/s/m}}")
if q <= 0:
    st.markdown(f"→ {cfg['formula']['zero_q_lsm']} l/s/m 미만이므로 **월파 없음**")
st.markdown(
    f"- γf (TTP가 파도 에너지를 줄이는 정도): {S['gamma_f']}"
    + (f" → 주기가 긴 파도라서 보정 후 {gf:.3f}" if gf > S["gamma_f"] + 1e-9 else "")
    + f"\n- γβ (파도가 비스듬히 올 때 줄어드는 정도): 정면 → {gb:.2f}"
)

st.markdown('<div class="hh">여유고에 따른 월파량 <span>(지금 파고·주기에서 여유고만 바꿨을 때)</span></div>', unsafe_allow_html=True)
rc_grid = np.linspace(rc_lo, rc_hi, 101)
band = ot.q_band_over_rc(hs, tp, rc_grid, cfg, 1500)
nz = lambda v: np.where(v > 0, v, np.nan)  # noqa: E731
y0, y1 = 1e-3, 1e3
fig_r = go.Figure()
for lo, hi, g in [(y0, CR["vehicle"], 0), (CR["vehicle"], CR["hinterland"], 1), (CR["hinterland"], y1, 2)]:
    fig_r.add_hrect(y0=lo, y1=hi, fillcolor=GRADE_COLORS[g], opacity=0.10, line_width=0)
    fig_r.add_annotation(x=0.005, xref="paper", y=math.log10(math.sqrt(lo * hi)), yref="y", text=GN[g], showarrow=False,
                         xanchor="left", font=dict(color=GRADE_COLORS[g], size=13), bgcolor="rgba(255,255,255,0.85)")
fig_r.add_scatter(x=rc_grid, y=nz(band["p95"]), mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False)
fig_r.add_scatter(x=rc_grid, y=nz(band["p05"]), mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(43,123,185,0.18)",
                  name="90% 범위 (공식 계수 오차)", hoverinfo="skip")
fig_r.add_scatter(x=rc_grid, y=nz(band["median"]), mode="lines", line=dict(color="#2B7BB9", width=2.5), name="중앙값",
                  hovertemplate="여유고 %{x:.2f} m<br>월파량 %{y:.3g} l/s/m<extra></extra>")
for val, name in [(CR["vehicle"], "차량 기준"), (CR["pedestrian"], "사람 기준"), (CR["house"], "가옥 기준"), (CR["hinterland"], "아무것도 없을 때 기준")]:
    fig_r.add_hline(y=val, line=dict(color="#8896A4", dash="dot", width=1))
    fig_r.add_annotation(x=0.995, xref="paper", y=math.log10(val), yref="y", text=f"{name} {val:g}", showarrow=False, xanchor="right",
                         yanchor="bottom" if name != "차량 기준" else "top", font=dict(color=MUTED, size=11), bgcolor="rgba(255,255,255,0.9)")
if q > 0:
    fig_r.add_scatter(x=[rc], y=[q], mode="markers", marker=dict(size=13, color=GRADE_COLORS[g_ai], line=dict(color="#fff", width=2)),
                      name="현재 조건", hovertemplate="현재: 여유고 %{x:.2f} m, 월파량 %{y:.3g} l/s/m<extra></extra>")
fig_r.add_vline(x=rc, line=dict(color=TEXT, dash="dot", width=1))
fig_r.update_layout(
    height=380, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(title="여유고 Rc [m]", gridcolor="#E5E9ED", color=MUTED),
    yaxis=dict(title="월파량 q [l/s/m]", type="log", range=[-3, 3], dtick=1, gridcolor="#E5E9ED", color=MUTED),
    legend=dict(orientation="h", y=1.08, font=dict(color=MUTED)),
)
st.plotly_chart(fig_r, width="stretch", config={"displayModeBar": False})

st.markdown('<div class="hh">파라미터 <span>(입력으로 계산한 값과 가능한 범위)</span></div>', unsafe_allow_html=True)
tb_lo, tb_hi = TB["tp"]
tm_lo, tm_hi = tb_lo / 1.1, tb_hi / 1.1
Lf = lambda tm: ot.G * tm**2 / (2 * math.pi)  # noqa: E731
sm_lo, sm_hi = S_LO * 1.21, S_HI * 1.21
xi = lambda s: (1 / S["cot_alpha"]) / math.sqrt(s)  # noqa: E731
rows = [
    ("무차원 여유고 Rc/Hm0 (여유고가 파고의 몇 배인지)", f"{rc / hs:.2f}", f"{rc_lo / hs_hi:.2f} ~ {rc_hi / hs_lo:.2f}"),
    ("Tm-1,0 (공식에 쓰는 주기, Tp ÷ 1.1)", f"{d['tm10'][0]:.2f} s", f"{tm_lo:.2f} ~ {tm_hi:.2f} s"),
    ("파장 Lm-1,0 (파도 하나의 길이)", f"{d['l_m10'][0]:.1f} m", f"{Lf(tm_lo):.1f} ~ {Lf(tm_hi):.1f} m"),
    ("파형경사 sm-1,0 (파도의 가파른 정도)", f"{d['s_m10'][0]:.4f}", f"{sm_lo:.4f} ~ {sm_hi:.4f}"),
    ("쇄파대 파라미터 ξm-1,0 (사면에서 파도가 부서지는 형태)", f"{d['xi'][0]:.2f}", f"{xi(sm_hi):.2f} ~ {xi(sm_lo):.2f}"),
    ("조도계수 γf (TTP가 파도 에너지를 줄이는 정도)", f"{gf:.3f}", f"{S['gamma_f']:.3f} ~ {S['gamma_f_max_long']:.3f}"),
    ("Hm0/h (수심에 비해 파도가 얼마나 큰지)", f"{hs / H:.2f}", f"{hs_lo / H:.2f} ~ {hs_hi / H:.2f}"),
    ("정수위 (방파벽 꼭대기 − 여유고)", f"DL{sm['swl']:+.2f} m", f"DL{S['crest_parapet_dl'] - rc_hi:+.2f} ~ DL{S['crest_parapet_dl'] - rc_lo:+.2f} m"),
]
st.table(pd.DataFrame(rows, columns=["파라미터", "현재 값", "범위"]).set_index("파라미터"))

st.markdown('<div class="hh">전제 조건 <span>(아래 값은 고정)</span></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="premise"><div class="row">'
    f"<span>TTP <b>{S['ttp_ton']:.0f} t 2층</b> (마루 DL+{S['armour_crest_dl']:.1f} m)</span>"
    f"<span>사면 기울기 <b>1:{S['cot_alpha']}</b></span>"
    f"<span>방파벽 꼭대기 <b>DL+{S['crest_parapet_dl']:.1f} m</b></span>"
    f"<span>수심 <b>{H} m</b></span>"
    f"<span>조도계수 <b>γf {S['gamma_f']}</b></span>"
    f"<span>파도 방향 <b>정면</b></span>"
    "</div></div>",
    unsafe_allow_html=True,
)

with st.expander("모델 성능 (학습에 쓰지 않은 시험 데이터로 공식의 정답과 비교)"):
    t, bt = meta["test"], meta["boundary_test"]
    macro_f1 = next(value for name, value in t.items() if "F1" in name)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("등급 정확도", f"{t['등급 정확도']:.1%}")
    m2.metric("위험을 안전으로 오판", f"{t['위험을 안전으로 오판']}건")
    m3.metric("경계 근처 정확도", f"{bt['등급 정확도']:.1%}")
    m4.metric("등급 macro-F1", f"{macro_f1:.3f}")
    cm = np.array(t["혼동행렬(행=실제, 열=예측)"])
    fig_cm = go.Figure(go.Heatmap(z=cm, x=[f"예측 {g}" for g in GN], y=[f"정답 {g}" for g in GN], text=cm, texttemplate="%{text}",
                                  colorscale=[[0, "#F3F5F7"], [1, "#2B7BB9"]], showscale=False))
    fig_cm.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                         yaxis=dict(autorange="reversed", color=TEXT), xaxis=dict(color=TEXT))
    st.plotly_chart(fig_cm, width="stretch", config={"displayModeBar": False})
