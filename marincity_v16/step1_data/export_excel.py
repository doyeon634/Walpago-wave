# =====================================================================
# [1차] export_excel.py – 학습 데이터를 엑셀로 정리 (확인용)
#
# 실행 (프로젝트 폴더에서, generate_data.py를 먼저 실행):
#   py step1_data/export_excel.py
#
# 산출물: step1_data/output/학습데이터_확인용.xlsx
#   설명       데이터가 무엇인지, 열의 뜻, 등급 기준
#   요약       건수·등급 비율·영역×등급 표 (학습데이터 시트에서 수식으로 계산)
#   격자(A)    같은 간격 격자 1,815건만, 파고 → 주기 → 여유고 순으로 정렬
#   학습데이터 3,000건 전체 (격자 → 경계 집중 → 가장자리 순)
#   평가_경계, 평가_범위밖   평가 전용 데이터
#
# 변경 이력
#   2026-10-29  최초 작성
#   2026-10-29  기준값 표시 형식 정리 (10.0 → 10)
# =====================================================================
from __future__ import annotations

import os
import sys

import pandas as pd
from openpyxl import Workbook
from openpyxl.formatting.rule import CellIsRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import overtopping as ot  # noqa: E402

FONT = "Arial"
HEAD_FILL = PatternFill("solid", start_color="1F3A4D")
GRADE_FILL = {"안전": "D5F0E3", "주의": "FBEBC2", "위험": "F8D3D0"}
THIN = Side(style="thin", color="C8CED3")

# (원래 열, 엑셀 머리글, 표시 형식, 너비)
COLUMNS = [
    ("id", "번호", "@", 9),
    ("구분", "구분", "@", 15),
    ("분할", "분할", "@", 7),
    ("hs", "파고 Hs (m)", "0.00", 11),
    ("tp", "주기 Tp (s)", "0.00", 11),
    ("rc", "여유고 Rc (m)", "0.00", 12),
    ("rc_over_hs", "무차원 여유고 Rc/Hs", "0.00", 15),
    ("s_p", "파형경사 (Tp 기준)", "0.0000", 15),
    ("q", "월파량 q (l/s/m)", "0.0000", 15),
    ("등급", "등급", "@", 7),
    ("grade", "등급 번호 (0 안전·1 주의·2 위험)", "0", 14),
    ("건물", "건물 피해 (가옥 기준 이상)", "@", 14),
    ("영역", "영역 (Hs/h)", "@", 10),
    ("tm10", "Tm-1,0 (s)", "0.00", 10),
    ("l_m10", "파장 Lm-1,0 (m)", "0.0", 12),
    ("s_m10", "파형경사 sm-1,0", "0.0000", 13),
    ("xi", "쇄파대 파라미터 ξ", "0.00", 13),
    ("gamma_f_used", "조도계수 γf", "0.000", 10),
    ("hs_over_h", "Hs/h", "0.00", 8),
    ("h_toe", "수심 h (m)", "0.0", 9),
]


def tidy(df: pd.DataFrame, cfg: dict, labels: bool = True) -> pd.DataFrame:
    gn, rn = cfg["grades"]["names"], cfg["regime"]["names"]
    df = df.copy()
    st = {"A": "A 격자", "B": "B 경계 집중", "C": "C 가장자리", "EB": "평가 경계", "OUT": "평가 범위밖"}
    df["구분"] = df["sample_type"].map(st).fillna(df["sample_type"])
    df["분할"] = df["split"].map({"train": "학습", "val": "검증", "test": "시험"}) if "split" in df else "평가"
    df["등급"] = df["grade"].map(lambda g: gn[int(g)])
    df["건물"] = df["building"].map({1: "있음", 0: "없음"})
    df["영역"] = df["regime"].map(lambda r: rn[int(r)])
    return df


def write_table(ws, df: pd.DataFrame, cols=COLUMNS, start_row=1):
    for j, (_, head, _, width) in enumerate(cols, 1):
        c = ws.cell(row=start_row, column=j, value=head)
        c.font = Font(name=FONT, bold=True, color="FFFFFF")
        c.fill = HEAD_FILL
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        ws.column_dimensions[get_column_letter(j)].width = width
    ws.row_dimensions[start_row].height = 32
    for i, row in enumerate(df[[c[0] for c in cols]].itertuples(index=False), start_row + 1):
        for j, v in enumerate(row, 1):
            c = ws.cell(row=i, column=j, value=(None if pd.isna(v) else (float(v) if isinstance(v, (float, int)) and not isinstance(v, bool) else v)))
            c.font = Font(name=FONT)
            c.number_format = cols[j - 1][2]
    last = start_row + len(df)
    ws.freeze_panes = ws.cell(row=start_row + 1, column=4)
    ws.auto_filter.ref = f"A{start_row}:{get_column_letter(len(cols))}{last}"
    gcol = get_column_letter([c[0] for c in cols].index("등급") + 1)
    for name, color in GRADE_FILL.items():
        ws.conditional_formatting.add(f"{gcol}{start_row + 1}:{gcol}{last}",
                                      CellIsRule(operator="equal", formula=[f'"{name}"'], fill=PatternFill("solid", start_color=color)))
    return last


def main():
    cfg = ot.load_config()
    data = tidy(pd.read_csv(ot.path_of(cfg, "scenarios")), cfg)
    eb = tidy(pd.read_csv(ot.path_of(cfg, "eval_boundary")), cfg)
    eo = tidy(pd.read_csv(ot.path_of(cfg, "eval_outside")), cfg)
    order = {"A": 0, "B": 1, "C": 2}
    data = data.assign(_o=data["sample_type"].map(order)).sort_values(["_o", "hs", "tp", "rc"]).drop(columns="_o")
    grid = data[data["sample_type"] == "A"]

    wb = Workbook()
    # ---------------- 설명
    ws = wb.active
    ws.title = "설명"
    crit, g = cfg["grades"]["criteria"], cfg["sampling"]["grid"]
    lines = [
        ("마린시티 호안 월파 – 1차 학습 데이터 (확인용)", True),
        ("", False),
        ("무엇인가", True),
        ("EurOtop (2018) 경사식 호안 평균 월파량 공식으로 계산한 가상 시나리오 3,000건입니다. 실측 자료가 아닙니다.", False),
        (f"전제 조건: 태풍 차바 당시 제방 (TTP 16 t 2층, 방파벽 꼭대기 DL+{cfg['structure']['crest_parapet_dl']} m), 사면 1:{cfg['structure']['cot_alpha']}, 수심 {cfg['structure']['water_depth_h']} m 고정, 조도계수 γf {cfg['structure']['gamma_f']}, 정면 입사", False),
        ("", False),
        ("어떻게 만들었나 (혼합 방식)", True),
        (f"A 격자 {int((data.sample_type == 'A').sum()):,}건: 파고 {cfg['ranges']['hs'][0]} m부터 {g['hs_step']} m 간격, 주기는 파고마다 가능한 범위를 {g['tp_levels']}단계 같은 간격, 여유고 {cfg['ranges']['rc'][0]} m부터 {g['rc_step']} m 간격", False),
        (f"B 경계 집중 {int((data.sample_type == 'B').sum()):,}건: 등급이 바뀌는 경계 근처 조건 (Sobol 표본에서 골라냄, 모자란 칸 보충 포함)", False),
        (f"C 가장자리 {int((data.sample_type == 'C').sum()):,}건: 입력 범위의 모서리와 면", False),
        ("분할: 학습 70% · 검증 15% · 시험 15% (영역 × 등급 비율을 같게 나눔)", False),
        ("", False),
        ("등급 기준 (국내 KDS 64 10 10 : 2020 해설 참고 표 4.3-9, 4.3-10)", True),
        (f"안전: 월파량 {crit['vehicle']:g} l/s/m 미만 (차량 운전 가능 기준)", False),
        (f"주의: {crit['vehicle']:g} 이상 {crit['hinterland']:g} l/s/m 미만 (사람·차량 통행 위험)", False),
        (f"위험: {crit['hinterland']:g} l/s/m 이상 (배후 민가·공공시설 밀집 지역 허용치 초과)", False),
        (f"건물 피해: {crit['house']:g} l/s/m 이상 (가옥 위험 없음 기준 초과)", False),
        (f"월파량 {cfg['formula']['zero_q_lsm']} l/s/m 미만은 0 (월파 없음)", False),
        ("", False),
        ("시트", True),
        ("요약: 건수와 비율 (학습데이터 시트에서 수식으로 계산)", False),
        ("격자(A): 같은 간격 격자만. 파고 → 주기 → 여유고 순서로 일정하게 증가", False),
        ("학습데이터: 3,000건 전체. 머리글의 필터 버튼으로 구분·분할·등급별로 볼 수 있음", False),
        ("평가_경계, 평가_범위밖: 모델 평가 전용 (학습에 쓰지 않음)", False),
        ("", False),
        ("열의 뜻", True),
        ("파고·주기·여유고: 모델 입력 3개 / 등급·건물 피해: 모델이 맞혀야 하는 정답", False),
        ("무차원 여유고 = 여유고 ÷ 파고, 파형경사 = 파도의 가파른 정도, 영역 = 수심에 비한 파고 크기 (비쇄파·전환·쇄파)", False),
        ("Tm-1,0·파장·쇄파대 파라미터·조도계수: 공식 계산에 쓴 파라미터", False),
    ]
    for i, (text, bold) in enumerate(lines, 1):
        c = ws.cell(row=i, column=1, value=text)
        c.font = Font(name=FONT, bold=bold, size=13 if i == 1 else 11)
    ws.column_dimensions["A"].width = 120

    # ---------------- 요약 (수식)
    ws = wb.create_sheet("요약")
    n = len(data)
    rng = lambda col: f"학습데이터!${col}$2:${col}${n + 1}"  # noqa: E731
    letter = {c[0]: get_column_letter(i) for i, c in enumerate(COLUMNS, 1)}
    hdr = Font(name=FONT, bold=True)
    ws["A1"] = "요약 (학습데이터 시트에서 계산)"; ws["A1"].font = Font(name=FONT, bold=True, size=13)
    ws["A3"] = "전체 건수"; ws["B3"] = f"=COUNTA({rng(letter['id'])})"
    r = 5
    ws.cell(row=r, column=1, value="구분").font = hdr; ws.cell(row=r, column=2, value="건수").font = hdr; ws.cell(row=r, column=3, value="비율").font = hdr
    for k, name in enumerate(["A 격자", "B 경계 집중", "C 가장자리"], r + 1):
        ws.cell(row=k, column=1, value=name)
        ws.cell(row=k, column=2, value=f'=COUNTIF({rng(letter["구분"])},A{k})')
        ws.cell(row=k, column=3, value=f"=B{k}/$B$3").number_format = "0.0%"
    r = 10
    ws.cell(row=r, column=1, value="등급").font = hdr; ws.cell(row=r, column=2, value="건수").font = hdr; ws.cell(row=r, column=3, value="비율").font = hdr
    for k, name in enumerate(cfg["grades"]["names"], r + 1):
        ws.cell(row=k, column=1, value=name)
        ws.cell(row=k, column=2, value=f'=COUNTIF({rng(letter["등급"])},A{k})')
        ws.cell(row=k, column=3, value=f"=B{k}/$B$3").number_format = "0.0%"
    ws.cell(row=14, column=1, value="건물 피해 있음")
    ws.cell(row=14, column=2, value=f'=COUNTIF({rng(letter["건물"])},"있음")')
    ws.cell(row=14, column=3, value="=B14/$B$3").number_format = "0.0%"
    r = 16
    ws.cell(row=r, column=1, value="영역 × 등급").font = hdr
    for j, gname in enumerate(cfg["grades"]["names"], 2):
        ws.cell(row=r, column=j, value=gname).font = hdr
    for k, rname in enumerate(cfg["regime"]["names"], r + 1):
        ws.cell(row=k, column=1, value=rname)
        for j in range(2, 5):
            col = get_column_letter(j)
            ws.cell(row=k, column=j, value=f'=COUNTIFS({rng(letter["영역"])},$A{k},{rng(letter["등급"])},{col}${r})')
    r = 21
    ws.cell(row=r, column=1, value="분할 × 등급").font = hdr
    for j, gname in enumerate(cfg["grades"]["names"], 2):
        ws.cell(row=r, column=j, value=gname).font = hdr
    ws.cell(row=r, column=5, value="합계").font = hdr
    for k, sname in enumerate(["학습", "검증", "시험"], r + 1):
        ws.cell(row=k, column=1, value=sname)
        for j in range(2, 5):
            col = get_column_letter(j)
            ws.cell(row=k, column=j, value=f'=COUNTIFS({rng(letter["분할"])},$A{k},{rng(letter["등급"])},{col}${r})')
        ws.cell(row=k, column=5, value=f"=SUM(B{k}:D{k})")
    r = 26
    for j, h in enumerate(["입력", "최솟값", "최댓값"], 1):
        ws.cell(row=r, column=j, value=h).font = hdr
    for k, (key, name) in enumerate([("hs", "파고 Hs (m)"), ("tp", "주기 Tp (s)"), ("rc", "여유고 Rc (m)")], r + 1):
        ws.cell(row=k, column=1, value=name)
        ws.cell(row=k, column=2, value=f"=MIN({rng(letter[key])})").number_format = "0.00"
        ws.cell(row=k, column=3, value=f"=MAX({rng(letter[key])})").number_format = "0.00"
    for row in ws.iter_rows(min_row=2, max_row=30, max_col=5):
        for c in row:
            if c.font is None or not c.font.bold:
                c.font = Font(name=FONT)
    for col, wdt in zip("ABCDE", (18, 10, 10, 10, 10)):
        ws.column_dimensions[col].width = wdt

    # ---------------- 데이터 시트
    ws = wb.create_sheet("격자(A)")
    write_table(ws, grid)
    ws = wb.create_sheet("학습데이터")
    write_table(ws, data)
    cols_eval = [c for c in COLUMNS if c[0] != "분할"]
    ws = wb.create_sheet("평가_경계")
    write_table(ws, eb, cols_eval)
    ws = wb.create_sheet("평가_범위밖")
    write_table(ws, eo, cols_eval)
    wb.move_sheet("학습데이터", offset=-1)  # 설명, 요약, 학습데이터, 격자(A), 평가…

    out = os.path.join(os.path.dirname(ot.path_of(cfg, "scenarios")), "학습데이터_확인용.xlsx")
    wb.save(out)
    print("만듦:", out)


if __name__ == "__main__":
    main()
