# =====================================================================
# [3차] section2d.py – 호안 단면 2D 애니메이션 (Plotly)
#   EurOtop 경험식으로 구한 처오름 높이·월파량·낙하 거리를 단면 그림으로 보여 줍니다.
#   물 흐름 시뮬레이션(CFD)이 아니며, 바다 물결 모양은 그림용입니다.
#
#   좌표: x = 바다(−) → 육지(+) 거리 (m, 파라펫 앞면 0), z = 표고 DL (m)
#
# 변경 이력
#   2026-10-28  최초 작성
#   2026-10-28  수심 고정(바닥 = 정수위 − 수심), 피해 구역을 국내 기준(사람·차량·가옥)으로
#   2026-10-29  넘어오는 파도 수를 식 6.3 기준으로 통일, 글자에 배경을 넣어 겹쳐도 읽히게
# =====================================================================
from __future__ import annotations

import math
import os
import sys

import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import overtopping as ot  # noqa: E402

GRADE_COLORS = ["#3FB68B", "#E6B34A", "#E5534B"]
C = {
    "sea": "#2B6CA3",
    "sea_line": "#7FC0EA",
    "runup": "#CFEAFB",
    "jet": "#9AD3F5",
    "ttp": "#9B9486",
    "ttp_dot": "#C4BDAE",
    "land": "#5B5346",
    "seabed": "#8A7A5C",
    "concrete": "#A7B0B6",
    "zone": "#6E7C8A",
    "building": "#3A4654",
    "text": "#E6EDF3",
    "muted": "#93A4B5",
}


def _poly(xs, zs, color, name="", opacity=1.0, line_color=None, hover=None):
    return go.Scatter(
        x=list(xs) + [xs[0]], y=list(zs) + [zs[0]], fill="toself", fillcolor=color, mode="lines",
        line=dict(color=line_color or color, width=1), opacity=opacity, name=name, showlegend=False,
        hoverinfo="text" if hover else "skip", text=hover,
    )


def _line(xs, zs, color, width, dash=None, hover=None):
    return go.Scatter(
        x=xs, y=zs, mode="lines", line=dict(color=color, width=width, dash=dash), showlegend=False,
        hoverinfo="text" if hover else "skip", text=hover,
    )


def overtopping_form(q: float, z_ru: float, crest: float, grade: int, d10: float, zones: dict) -> str:
    """월파 형태를 한 문장으로."""
    if q <= 0:
        if z_ru <= crest:
            return "넘치지 않음 – 파도가 사면을 타고 올라오다 호안 꼭대기 아래에서 멈춥니다."
        return "거의 넘치지 않음 – 가장 큰 파도 일부만 호안 꼭대기에 닿고, 넘어오는 양은 무시할 수준입니다."
    place = "방파벽 바로 뒤"
    if d10 + zones["parapet"] >= zones["road0"]:
        place = "도로"
    elif d10 + zones["parapet"] >= zones["walk0"]:
        place = "보행로"
    head = ["가끔 넘침", "자주 넘침", "크게 넘침"][grade]
    return f"{head} – 큰 파도가 호안을 넘고, 넘친 물의 10%는 {place}까지 날아갑니다."


def build_section(cfg: dict, hs: float, tp: float, rc: float, barrier_up: bool, q: float, grade_idx: int) -> tuple[go.Figure, dict]:
    st, sec = cfg["structure"], cfg["dashboard"]["section"]
    barrier_up = False                       # v7: 기립식 차수벽 없음 (인자는 호환용)
    crest = st["crest_parapet_dl"]
    swl = crest - rc
    cot, ac = st["cot_alpha"], st["armour_crest_dl"]
    toe = swl - st["water_depth_h"]  # 전제 조건: 수심 고정
    ground, pw = sec["ground_dl"], sec["parapet_width"]
    x_toe = -(ac - toe) * cot
    x_sea0 = x_toe - 16.0
    walk0 = sec["buffer_width"] + sec["gap_width"]
    walk1 = walk0 + sec["walkway_width"]
    road1 = walk1 + sec["road_width"]
    bld0 = road1 + sec["building_setback"]
    bld1 = bld0 + 9.0
    z_bld = ground + 11.0

    # ---- 계산 ----------------------------------------------------------
    ru = ot.runup_2pct(hs, tp, cfg)
    z_ru = swl + ru
    crit = cfg["grades"]["criteria"]
    lim_p, lim_v, lim_h = crit["pedestrian"], crit["vehicle"], crit["house"]
    ev = ot.overtopping_events(hs, tp, q, rc, cfg)
    drop_y = crest - ground
    dist = {f: ot.landing_distance(f, hs, tp, drop_y) for f in (0.5, 0.1, 0.01)} if q > 0 else {}
    walk_bad, road_bad = q >= lim_p, q >= lim_v
    bld_bad = q >= lim_h
    zones = {"parapet": pw, "walk0": walk0, "road0": walk1}
    form = overtopping_form(q, z_ru, crest, grade_idx, dist.get(0.1, 0.0), zones)

    # ---- 고정 요소 -----------------------------------------------------
    tr = []
    tr.append(_poly([x_sea0, x_toe, x_toe, x_sea0], [toe - 2, toe - 2, toe, toe], C["seabed"]))
    tr.append(_poly([x_toe, bld1 + 3, bld1 + 3, 0], [toe - 2, toe - 2, ground, ground], C["land"]))
    tr.append(_poly([x_toe, 0, 0], [toe, toe, ac], C["ttp"], hover=f"TTP 피복 ({st['ttp_ton']:.0f} t, 사면 1:{cot})"))
    # TTP 질감
    gx, gz = [], []
    for z in np.arange(toe + 0.8, ac - 0.4, 1.4):
        x_left = -(ac - z) * cot
        for x in np.arange(x_left + 0.7, -0.4, 1.6):
            gx.append(x + (0.5 if int(z) % 2 else 0))
            gz.append(z)
    tr.append(go.Scatter(x=gx, y=gz, mode="markers", marker=dict(size=7, color=C["ttp_dot"], symbol="circle-open", line=dict(width=2)),
                         hoverinfo="skip", showlegend=False))
    tr.append(_poly([0, pw, pw, 0], [ground - 3.0, ground - 3.0, st["crest_parapet_dl"], st["crest_parapet_dl"]], C["concrete"], hover="파라펫"))
    # 배후 구역
    tr.append(_line([pw, sec["buffer_width"]], [ground, ground], C["zone"], 7, hover="재해완충구역"))
    tr.append(_line([walk0, walk1], [ground, ground], GRADE_COLORS[2] if walk_bad else C["zone"], 9,
                    hover=f"보행로: {'기준 초과' if walk_bad else '기준 이내'}"))
    tr.append(_line([walk1, road1], [ground, ground], GRADE_COLORS[2] if road_bad else C["zone"], 9,
                    hover=f"도로: {'기준 초과' if road_bad else '기준 이내'}"))
    tr.append(_poly([bld0, bld1, bld1, bld0], [ground, ground, z_bld, z_bld], C["building"], hover="건물"))
    tr.append(_line([bld0, bld0], [ground, ground + 5.0], GRADE_COLORS[2] if bld_bad else C["zone"], 8,
                    hover=f"건물 1층 앞면: {'피해 가능' if bld_bad else '기준 이내'}"))
    # 마루 색상 라인 (위험 등급 색)
    gcol = GRADE_COLORS[grade_idx]
    if barrier_up:
        tr.append(_line([0.15, 0.15, 0.6], [st["crest_parapet_dl"], crest, crest], gcol, 9, hover="기립식 차수벽 (기립)"))
        tr.append(_line([0.6, pw], [st["crest_parapet_dl"], st["crest_parapet_dl"]], gcol, 9))
    else:
        tr.append(_line([0, pw], [crest + 0.12, crest + 0.12], gcol, 9, hover="파라펫 마루 (차수벽 눕힘)"))
    # 처오름 높이 표시선
    tr.append(_line([x_toe * 0.75, pw + 1.5], [z_ru, z_ru], GRADE_COLORS[2] if z_ru > crest else C["muted"], 2, dash="dash",
                    hover=f"파도가 올라온 높이 DL{z_ru:+.2f} m"))
    n_static = len(tr)

    # ---- 움직이는 요소 -------------------------------------------------
    x_swl = -(ac - swl) * cot if swl < ac else 0.0
    amp = min(hs / 2.0, 3.0)
    wl = 18.0
    xs_sea = np.linspace(x_sea0, x_swl - 0.3, 70)
    taper = np.clip((x_swl - xs_sea) / 6.0, 0, 1)
    z_top = min(z_ru, crest)
    q_vis = 0.0 if q <= 0 else float(np.clip((math.log10(q) + 2) / 4, 0.1, 1.0))  # 0.01→0.1, 100→1

    def sea(phase):
        eta = amp * np.sin(2 * np.pi * (xs_sea / wl) - phase) * taper
        xs = [x_sea0, x_toe] + [x_swl] + list(xs_sea[::-1])
        zs = [toe, toe] + [swl] + list(swl + eta[::-1])
        return go.Scatter(x=xs + [xs[0]], y=zs + [zs[0]], fill="toself", fillcolor=C["sea"], mode="lines",
                          line=dict(color=C["sea_line"], width=2), opacity=0.9, hoverinfo="skip", showlegend=False)

    def runup(frac):
        lvl = swl + frac * (z_top - swl)
        xa = -(ac - lvl) * cot if lvl < ac else 0.0
        return go.Scatter(x=[x_swl, xa, xa, x_swl, x_swl], y=[swl, lvl, lvl + 0.45, swl + 0.45, swl], fill="toself",
                          fillcolor=C["runup"], mode="lines", line=dict(color=C["runup"], width=1), opacity=0.9,
                          hoverinfo="skip", showlegend=False)

    def jet(f, width, prog):
        if q <= 0 or prog <= 0:
            return go.Scatter(x=[None], y=[None], mode="lines", showlegend=False, hoverinfo="skip")
        x_end = pw + dist[f]
        apex = crest + min(max(z_ru - crest, 0.5), 3.5)
        t = np.linspace(0, prog, 25)
        x = t * x_end
        z = (1 - t) ** 2 * crest + 2 * (1 - t) * t * (2 * apex - (crest + ground) / 2) + t**2 * ground
        return go.Scatter(x=x, y=z, mode="lines", line=dict(color=C["jet"], width=width * (0.4 + q_vis)), opacity=0.95,
                          showlegend=False, hoverinfo="text", text=f"넘친 물의 {int(f * 100)}%가 파라펫 뒤 {dist[f]:.1f} m보다 멀리 떨어짐")

    def puddle(prog):
        if q <= 0 or prog <= 0:
            return go.Scatter(x=[None], y=[None], mode="lines", showlegend=False, hoverinfo="skip")
        x1 = pw + dist[0.1] * prog
        th = 0.08 + 0.45 * q_vis
        return go.Scatter(x=[pw, x1, x1, pw, pw], y=[ground, ground, ground + th, ground + th, ground], fill="toself",
                          fillcolor=C["jet"], mode="lines", line=dict(color=C["jet"], width=0), opacity=0.55,
                          hoverinfo="skip", showlegend=False)

    jets = [(0.5, 9), (0.1, 6), (0.01, 3)]
    # 정지 화면 = 가장 높이 올라온 순간
    tr += [sea(np.pi / 2), runup(1.0)] + [jet(f, w, 1.0) for f, w in jets] + [puddle(1.0)]
    fig = go.Figure(data=tr)

    frames, n = [], 48
    over = q > 0
    for k in range(n + 1):
        ph = 2 * np.pi * 3 * k / n                       # 파도 3번
        cyc = max(0.0, math.sin(ph))                     # 처오름 오르내림
        prog = min(1.0, max(0.0, (cyc - 0.55) / 0.45)) if over else 0.0
        acc = min(1.0, k / n * 1.6) if over else 0.0     # 고인 물은 점점 늘어남
        data = [sea(ph), runup(cyc)] + [jet(f, w, prog) for f, w in jets] + [puddle(acc)]
        frames.append(go.Frame(data=data, traces=list(range(n_static, n_static + 6)), name=str(k)))
    fig.frames = frames

    # ---- 글자 ----------------------------------------------------------
    ann = [
        (x_sea0 + 7, swl + amp + 1.0, f"바다 (정수위 DL{swl:+.2f} m)", C["text"]),
        (x_toe * 0.55, (toe + ac) / 2 - 1.2, "TTP 피복", C["text"]),
        (pw / 2 + 0.2, ground - 4.4, "차수벽 기립" if barrier_up else "파라펫", C["text"]),
        (sec["buffer_width"] / 2 + 1, ground - 1.4, "재해완충구역", C["muted"]),
        ((walk0 + walk1) / 2, ground - 1.4, "보행로", C["muted"]),
        ((walk1 + road1) / 2, ground - 1.4, "도로", C["muted"]),
        ((bld0 + bld1) / 2, z_bld + 1.0, "건물", C["muted"]),
        (x_toe * 0.75, z_ru + 0.9, f"파도가 올라온 높이 DL{z_ru:+.2f} m", GRADE_COLORS[2] if z_ru > crest else C["muted"]),
    ]
    fig.update_layout(
        annotations=[dict(x=x, y=y, text=t, showarrow=False, font=dict(color=c, size=12), xanchor="left" if i in (0, 7) else "center",
                          bgcolor="rgba(15,23,34,0.75)", borderpad=2)
                     for i, (x, y, t, c) in enumerate(ann)],
        height=440,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[x_sea0, bld1 + 3], showgrid=False, zeroline=False, title="바다 ← → 육지 (m)", color=C["muted"]),
        yaxis=dict(range=[toe - 2, z_bld + 3], showgrid=False, zeroline=False, title="높이 DL (m)", color=C["muted"],
                   scaleanchor="x", scaleratio=1),
        updatemenus=[dict(type="buttons", showactive=False, x=0.0, y=1.0, xanchor="left", yanchor="top",
                          bgcolor="#172231", bordercolor="#2A3A4D", font=dict(color=C["text"]),
                          buttons=[dict(label="▶ 파도 재생", method="animate",
                                        args=[None, dict(frame=dict(duration=90, redraw=True), fromcurrent=False, transition=dict(duration=0))])])],
    )

    summary = {
        "swl": swl, "crest": crest, "ru": ru, "z_ru": z_ru, "form": form, "events": ev, "dist": dist,
        "walk_bad": walk_bad, "road_bad": road_bad, "bld_bad": bld_bad, "lim_p": lim_p, "lim_v": lim_v, "lim_h": lim_h,
        "walk0_from_rear": walk0 - pw,
    }
    return fig, summary
