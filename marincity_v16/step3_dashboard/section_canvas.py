# =====================================================================
# [3차] section_canvas.py – 단면 그림 (실사형 + 치수선), HTML 캔버스 애니메이션
#   3D 장면(threescene)과 같은 계산값을 쓰고, 여기에 넘친 물이 떨어지는 거리(식 6.20)를 더한다.
#
# 사용 (app.py):  html = section_canvas.build_html(section_canvas.params(cfg, hs, tp, rc))
# 단독 실행 (예시 HTML 만들기):  py section_canvas.py
#
# 변경 이력
#   2026-10-29  최초 작성 (실사형 단면 + 치수·표고 표시, 켜고 끌 수 있음)
# =====================================================================
from __future__ import annotations

import json
import os
import sys
from functools import lru_cache

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import overtopping as ot  # noqa: E402
import threescene  # noqa: E402


def params(cfg: dict, hs: float, tp: float, rc: float) -> dict:
    """3D 와 같은 값 + 넘친 물이 떨어지는 거리 (파라펫 뒤 기준, 50%·10%·1%)."""
    p = threescene.params(cfg, hs, tp, rc)
    drop_y = p["crest"] - p["ground"]
    p["d"] = {str(f): ot.landing_distance(f, hs, tp, drop_y) for f in (0.5, 0.1, 0.01)} if p["overtop"] else None
    return p


@lru_cache(maxsize=None)
def _template() -> str:
    with open(os.path.join(HERE, "section_template.html"), encoding="utf-8") as f:
        return f.read()


def build_html(p: dict) -> str:
    return _template().replace("/*__PARAMS__*/", json.dumps(p, ensure_ascii=False))


if __name__ == "__main__":
    cfg = ot.load_config()
    for name, args in {"예시1_차바급": (5.0, 15.0, 4.12), "예시2_보통태풍": (3.0, 9.0, 4.12)}.items():
        out = os.path.join(HERE, f"{name}_단면.html")
        with open(out, "w", encoding="utf-8") as f:
            f.write(build_html(params(cfg, *args)))
        print("만듦:", out)
