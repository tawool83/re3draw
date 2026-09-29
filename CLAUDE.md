# re3draw — Claude 작업 규칙

여러 각도 사진 → 3D(가우시안 스플랫) 복원 → Flutter/웹 표시를 하는 오픈소스 라이브러리.
기획: Notion "re3draw — 사진을 3D 모델 변환" 페이지. 진행 상황과 다음 할 일: **`docs/PROGRESS.md`부터 읽는다.**

## 소통과 기록

- 사용자와의 대화, 문서(`docs/`), Notion은 **한국어**로 쓴다. 코드, 주석, docstring은 영어.
- **커밋 메시지는 한국어**로 쓴다(요약 줄과 본문 모두). 유형 접두어(`feat(worker):`, `fix:`, `docs:`)와
  마지막 `Co-Authored-By` 줄은 그대로 둔다. 이미 푸시한 커밋은 다시 쓰지 않는다.
- 커밋·푸시는 사용자가 요청할 때 한다. 푸시 전에 `git fetch`로 원격 변경이 없는지 확인한다.
- 사용자는 회사 Windows PC와 집 Mac에서 번갈아 개발한다. PC 사이에 이어져야 하는 결정·실측·보류
  계획은 Claude 메모리가 아니라 `docs/PROGRESS.md`에 남긴다.

## 구조

- `packages/worker` (Python): `mat`(매트 PDF) → `pose`(카메라 위치, COLMAP) → `segment`(SAM 2 물체 오리기)
  → `train`(gsplat). 나머지 패키지(`viewer`, `sdk-js`, `flutter`)는 아직 자리만 있다.
- 세계 좌표: 매트 중심 원점, +X 오른쪽, +Y 매트 위쪽 가장자리, +Z 위, 미터. FRONT는 −Y 쪽.
- 촬영 결과·학습 결과물은 `out/`(git 제외)에 둔다.

## 개발 환경

- 가상환경: `packages/worker/.venv` (Windows `.venv\Scripts\`, Mac `.venv/bin/`). 설치 절차는 `docs/PROGRESS.md`.
- 로컬 테스트: `packages/worker`에서 `python -m pytest` (GPU 테스트는 자동 skip).
- GPU 작업은 **Modal**(`packages/worker/modal_app.py`): `selftest`, 학습, `diagnose`. 10분 넘는 작업은
  `python -m modal run --detach ...`로 돌리고 결과는 `re3draw-captures` 볼륨에서 받는다.

## 지켜야 할 제약

- 의존성은 허용적 라이선스만(Apache-2.0, MIT, BSD). 비상업 라이선스(DUSt3R/MASt3R, BRIA RMBG 등) 금지.
  새 의존성은 `NOTICE`에 추가한다.
- 학습 이미지: gsplat 미리 빌드된 휠이 **torch 2.4 + Python 3.10 전용**이고, 설치는 `--extra-index-url`.
  SAM 2는 torch ≥ 2.5가 필요해서 **오리기와 학습은 별도 이미지**로 돌린다.
- 숫자(시간·비용·화질)는 추정과 실측을 구분해서 말하고, 실측은 `docs/PROGRESS.md`에 기록한다.
