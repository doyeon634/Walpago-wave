# =====================================================================
# [3차] threescene.py – 사실적인 3D 월파 장면 (three.js)
#   EurOtop 계산값(처오름, 넘어오는 파도 비율, 월파량 분포, 낙하 거리, 마루 위 흐름)을 넣어
#   하나의 HTML 문자열을 만든다. three.js는 파일 안에 들어가므로 인터넷이 필요 없다.
#
# 사용 (app.py):  html = threescene.build_html(threescene.params(cfg, hs, tp, rc))
# 단독 실행 (예시 HTML 만들기):  py threescene.py
#
# 변경 이력
#   2026-10-28  최초 작성 (교수님 요청: 실제처럼, 파도가 덮치는 모습)
#   2026-10-29  넘어오는 파도 비율과 월파량 분포를 overtopping.overtopping_events와 같은 값으로 통일,
#               평균 월파량이 0 이면 넘치지 않게
# =====================================================================
from __future__ import annotations

import json
import math
import os
import sys
from functools import lru_cache

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import overtopping as ot  # noqa: E402

GRADE_COLORS = ["#3FB68B", "#E6B34A", "#E5534B"]


def params(cfg: dict, hs: float, tp: float, rc: float) -> dict:
    """대시보드 입력과 config로 3D 장면에 넘길 값을 계산."""
    st, sec = cfg["structure"], cfg["dashboard"]["section"]
    crest = st["crest_parapet_dl"]
    swl = crest - rc
    ground, pw = sec["ground_dl"], sec["parapet_width"]

    q = float(ot.mean_discharge([hs], [tp], [rc], cfg)[0])
    g = int(ot.grade([q], [hs], cfg)[0])
    ev = ot.overtopping_events(hs, tp, q, rc, cfg)            # 대시보드 숫자 칸과 같은 기준 (식 6.3)
    ru2 = ev["ru2"]
    ex = max(swl + ru2 - crest, 0.0) if q > 0 else 0.0        # 큰 파도(상위 2%)가 마루를 넘는 높이
    lw = float(ot.derived([hs], [tp], cfg)["l_m10"][0])
    v02 = 1.45 * math.sqrt(ot.G * ex)                         # 마루를 넘는 속도 (EurOtop 5.5.3절)
    h0 = 0.2 * ex * 2 / 3                                     # 마루 위 물 두께 (식 5.59 × 2/3)
    sop = 2 * math.pi * hs / (ot.G * tp**2)
    land_c = 2.7 * (crest - ground) * sop**0.15               # 식 6.19의 높이 차 항

    walk0 = sec["buffer_width"] + sec["gap_width"]
    walk1 = walk0 + sec["walkway_width"]
    road1 = walk1 + sec["road_width"]
    bld0 = road1 + sec["building_setback"]
    v_at = lambda x: v02 * math.exp(-1.4 * x / lw)           # noqa: E731  식 5.61
    crit = cfg["grades"]["criteria"]
    return dict(
        hs=hs, tp=tp, rc=rc, crest=crest, crest_parapet=st["crest_parapet_dl"], berm=sec["berm_width"],
        swl=swl, h=st["water_depth_h"], toe=swl - st["water_depth_h"], ac=st["armour_crest_dl"], ground=ground, pw=pw,
        cot=st["cot_alpha"], buffer_w=sec["buffer_width"], gap=sec["gap_width"], walk_w=sec["walkway_width"],
        road_w=sec["road_width"], setback=sec["building_setback"],
        dn=ot.nominal_diameter(cfg), ttp_h=(st["ttp_ton"] / st["rho_c"] / 0.28) ** (1 / 3),
        Lw=lw, tm=tp / cfg["formula"]["tp_over_tm"], ru2=ru2,
        pov=ev["pov"], wa=ev["a"], wb=ev["b"], nw_h=ev["nw"], now_h=ev["now"], overtop=q > 0,
        land_c=land_c, drain=0.03,                            # drain: 땅 위 물이 빠지는 속도 (m/s, 그림용)
        q=q, grade_name=cfg["grades"]["names"][g], grade_color=GRADE_COLORS[g],
        rise2=1.05 * ex, v02=v02,
        zone=dict(h=h0, walk_v=v_at((walk0 + walk1) / 2), road_v=v_at((walk1 + road1) / 2), bld_v=v_at(bld0),
                  walk_bad=q >= crit["pedestrian"], road_bad=q >= crit["vehicle"], bld_bad=q >= crit["house"]),
    )


@lru_cache(maxsize=None)
def _read(*parts: str) -> str:
    with open(os.path.join(HERE, *parts), encoding="utf-8") as f:
        return f.read()


def build_html(p: dict) -> str:
    """장면 HTML 문자열 (three.js 포함, 약 0.7 MB)."""
    return (_read("scene_template.html")
            .replace("/*__THREE__*/", _read("vendor", "three.min.js"))
            .replace("/*__ORBIT__*/", _read("vendor", "OrbitControls.js"))
            .replace("/*__PARAMS__*/", json.dumps(p, ensure_ascii=False)))


if __name__ == "__main__":
    cfg = ot.load_config()
    for name, args in {"예시1_차바급": (5.0, 15.0, 4.12), "예시2_보통태풍": (3.0, 9.0, 4.12)}.items():
        out = os.path.join(HERE, f"{name}_3D.html")
        with open(out, "w", encoding="utf-8") as f:
            f.write(build_html(params(cfg, *args)))
        print("만듦:", out)
