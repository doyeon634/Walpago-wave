# =====================================================================
# [3차] section_sim.py – 2D 실시간 입자 시뮬레이션 (물을 입자로 나눠 직접 계산)
#   sim_template.html 에 파고·주기·여유고를 넣어 HTML 문자열을 만든다.
#   입자 크기는 실행하는 컴퓨터가 실시간으로 감당할 수 있는 가장 작은 값(입자 수 최대)을 자동으로 고른다.
#
# 사용 (app.py):  html = section_sim.build_html(hs, tp, rc, q)
#
# 변경 이력
#   2026-11-01  최초 작성 (시제품: 규칙·불규칙파, TTP 투수층, 물보라)
# =====================================================================
from __future__ import annotations

import json
import os
from functools import lru_cache

HERE = os.path.dirname(os.path.abspath(__file__))


@lru_cache(maxsize=None)
def _template() -> str:
    with open(os.path.join(HERE, "sim_template.html"), encoding="utf-8") as f:
        return f.read()


def build_html(hs: float, tp: float, rc: float, q: float | None = None) -> str:
    """q: 대시보드에서 계산한 평균 월파량 (넣으면 시뮬레이션 화면도 같은 값과 등급을 표시)."""
    params = json.dumps({"hs": round(float(hs), 2), "tp": round(float(tp), 1), "rc": round(float(rc), 2), "q": None if q is None else float(q)})
    return _template().replace("/*__SIMPARAMS__*/", f"window.SIM_PARAMS = {params};")
