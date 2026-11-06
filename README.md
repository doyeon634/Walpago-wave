[README.md](https://github.com/user-attachments/files/34438598/README.md)
# 월파고(Walpago:wave) · 마린시티 호안 월파 위험 예측

> 파고·주기·여유고 3개만 넣으면, AI가 태풍 때 해안 호안을 넘는 파도(월파)의 위험 등급을 바로 알려 주는 대시보드

AI:ON 2026 · 팀 Walpago:wave (T-2026-0038) · 박건우, 김도연, 남선우

---

## 시연 영상 (3분)

[![시연 영상 보기](docs/images/dashboard_main.png)](youtube.com/watch?si=UGgGqyFbPfcQ5M_z&v=2mvE1hI-Nek&feature=youtu.be)

위 그림을 누르면 YouTube 시연 영상이 열립니다.

---

## 무엇을 만들었나

| 구분 | 내용 |
|---|---|
| 대상 | 부산 해운대 마린시티 호안 (태풍 차바 당시 제방: TTP 16 t 2층, 방파벽 꼭대기 DL+6.3 m) |
| 입력 | 파고 Hs, 주기 Tp, 여유고 Rc |
| 출력 | 위험 등급(안전·주의·위험), 등급별 확률, 평균 월파량, 건물 피해 가능성 |
| 핵심 AI | Random Forest 분류 모델 |
| 정답 기준 | 월파량: EurOtop(2018) 경사식 공식 / 등급: 국내 항만 및 어항 설계기준(KDS 64 10 10) 허용 월파량 |

**위험 등급 기준 (국내 허용 월파량)**

| 등급 | 월파량 (l/s/m) | 뜻 |
|---|---|---|
| 안전 | 0.02 미만 | 사람과 차량이 통행해도 되는 수준 |
| 주의 | 0.02 ~ 10 | 사람·차량 통행이 위험한 수준 |
| 위험 | 10 이상 | 배후 주거·상업 지역에 피해가 우려되는 수준 |

---

## 주요 결과

| 평가 | 결과 |
|---|---|
| 시험 데이터 450건 (학습에 쓰지 않음) | 정확도 98.9%, macro-F1 0.989 |
| 위험을 안전으로 잘못 판정 | 0건 |
| 5겹 층화 교차검증 (2,550건) | 정확도 97.7 ± 0.3%, macro-F1 0.976 ± 0.004 |
| 등급 경계 근처 300건 | 정확도 95.3% |
| 학습 범위 밖 200건 | 정확도 99.0% |

**변수 중요도**: 무차원 여유고(Rc/Hs) 63.3% > 파고 20.5% > 주기 7.6% > 여유고 6.0% > 파형경사 2.6%
→ 여유고와 파고의 비율이 위험을 가장 크게 좌우하며, EurOtop 공식의 구조와 일치합니다.

전체 결과: [`step2_model/output/model_report.md`](step2_model/output/model_report.md)

---

## 학습 데이터

EurOtop 공식으로 계산해 만든 3,000건입니다. 실측 자료가 아닙니다.

| 구성 | 건수 | 설명 |
|---|---|---|
| A. 격자 | 1,870 | 파고·주기·여유고를 같은 간격으로 나눈 모든 조합 |
| B. 등급 경계 집중 | 930 | 등급이 바뀌는 경계 근처 조건 |
| C. 범위 가장자리 | 200 | 입력 범위의 모서리와 면 |
| 분할 | 학습 2,100 / 검증 450 / 시험 450 | 등급 비율을 같게 나눔 |
| 평가 전용 | 경계 300 / 범위 밖 200 | 학습에 쓰지 않음 |

- 입력 범위: 파고 0.5~6.75 m, 여유고 2.3~6.3 m, 주기는 파고에 따라 가능한 범위(파형경사 0.01~0.06)
- 전제 조건: 수심 10.5 m, 사면 1:1.5, 조도계수 0.47, 정면 입사

| 파일 | 내용 |
|---|---|
| [`step1_data/output/학습데이터_확인용.xlsx`](step1_data/output/학습데이터_확인용.xlsx) | **먼저 보세요.** 설명·요약·전체 데이터·격자 데이터를 시트로 정리 |
| [`step1_data/output/scenarios.csv`](step1_data/output/scenarios.csv) | 학습 데이터 원본 3,000건 |
| [`step1_data/output/data_card.md`](step1_data/output/data_card.md) | 열의 뜻, 등급 기준, 한계 |
| [`step1_data/output/validation_report.md`](step1_data/output/validation_report.md) | 데이터 검증 결과 (재현성, 물리 일관성 등) |

---

## AI 활용 기록

핵심 AI는 Random Forest 분류 모델이고, 생성형 AI는 개발과 검증을 돕는 보조 도구로 썼습니다. 모든 결과는 팀원이 직접 확인했습니다.

| 단계 | 도구 | 한 일 |
|---|---|---|
| 자료 조사 | ChatGPT, Gemini | 태풍 차바 당시 조위·파랑, 마린시티 제방 단면, EurOtop·KDS 자료 조사 후 원문으로 확인 |
| 데이터·모델·대시보드 코드 | Claude | 원문을 근거로 공식·등급 기준 적용, 데이터 생성 → 학습 → 대시보드 코드 작성 |
| 코드 보완 | GitHub Copilot (VS Code) | macro-F1·5겹 교차검증 추가, 대시보드에 macro-F1 표시 |

### GitHub Copilot 활용 화면

**① 평가 지표 macro-F1 추가와 대시보드 표시**
![Copilot macro-F1](docs/images/copilot_macro_f1.png)

**② 5겹 교차검증 추가**
![Copilot 교차검증](docs/images/copilot_cv.png)

Copilot이 고친 코드는 팀이 직접 다시 실행해, 리포트와 대시보드 값이 같은지 확인한 뒤 반영했습니다.

---

## 실행 방법 (Windows)

```powershell
py -m pip install -r requirements.txt
py step1_data\generate_data.py      # 1차: 학습 데이터 3,000건 생성
py step1_data\export_excel.py       # (선택) 확인용 엑셀 만들기
py step2_model\train_model.py       # 2차: Random Forest 학습·평가
cd step3_dashboard
py -m streamlit run app.py          # 3차: 대시보드 실행 (브라우저가 열림)
```

## 폴더 구조

```
├─ config.yaml          설정값 (단면, 범위, 등급 기준)
├─ overtopping.py       EurOtop 공식 계산 (1·2·3차 공통)
├─ step1_data/          1차: 학습 데이터 생성
├─ step2_model/         2차: Random Forest 학습·평가
├─ step3_dashboard/     3차: 웹 대시보드 (3D·2D 시각화 포함)
└─ docs/images/         README 그림
```

## 한계

- 학습 데이터는 EurOtop 경험식으로 만든 가상 자료입니다. 실측·수치모델 검증은 2단계 계획입니다.
- 정면 입사(가장 위험한 경우) 기준이며, 바람의 영향은 반영하지 않았습니다.
- 3D·2D 그림은 계산 결과를 보여 주는 시각화이며, 2D 시뮬레이션은 개발 중인 시제품입니다.

## 참고 자료

- EurOtop (2018). Manual on wave overtopping of sea defences and related structures. 6장 경사식 구조물
- 해양수산부, KDS 64 10 10 : 2020 항만 및 어항 설계기준 – 설계조건, 허용 월파량(참고표 4.3-9, 4.3-10)
- 태풍 차바(2016) 부산 연안 피해 관련 논문 및 보도 자료
