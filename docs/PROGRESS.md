# 진행 상황 / 개발 환경 메모

> 여러 PC(회사 Windows, 집 Mac)에서 이어서 개발하기 위한 메모. 새 PC에서 시작할 때 이 문서부터 본다.
> 최종 업데이트 2026-09-30 (전체 파이프라인 한 바퀴 완주 + SPZ v4 호환 문제 발견)

## 현재 상태

| 마일스톤 | 상태 | 비고 |
| --- | --- | --- |
| M1 매트 + 카메라 위치 | ✅ 완료 | `re3draw-worker mat / pose / synth` |
| M2 학습 워커 | ✅ 합성 데이터에서 동작 | `re3draw-worker segment` (SAM 2) + `train` (gsplat). 배경 커튼 해결 |
| M2 GPU 실행 환경 | ✅ Modal 연결 완료 | `packages/worker/modal_app.py`. GPU 자체 테스트 48개 통과 (L4) |
| M2 실물 촬영 검증 | ⏳ 대기 | [m2-real-capture-test.md](m2-real-capture-test.md) |
| M2 뷰어 | ✅ 학습 결과 표시 확인 | `packages/viewer`. `npm run dev` → `splat 열기`로 **`.ply`**를 연다 (`.spz`는 아래 참고) |

### GPU를 Modal로 정한 이유

- 매월 무료 크레딧으로 M2 개발 GPU 시간을 거의 충당 (L4 기준)
- 초 단위 과금, 끝나면 자동 종료 → 파드를 안 꺼서 요금이 쌓이는 위험이 없음
- Notion 기획의 F3(서버 복원 워커)가 원래 Modal 기준 → 지금 만든 게 그대로 운영 경로
- RunPod은 직접 접속해 디버깅할 때만 예비로 사용, Vast.ai는 합성 데이터 외 사용 금지

### Modal 사용법 (`packages/worker`에서)

```bash
python -m modal run modal_app.py::selftest                          # GPU 자체 테스트 (약 1분)
python -m modal run modal_app.py --capture ../../out/synth --iters 7000
python -m modal run --detach modal_app.py::diagnose --iters 2000    # 학습 진단 (결과는 Volume에)
python -m modal volume get --force re3draw-captures _diag/2000-512 ../../out/
```

- 촬영 폴더는 `re3draw-captures` Volume에 올라가고, 결과는 로컬 `<촬영 폴더>/splat-<iters>/`로 내려온다.
- 10분이 넘는 작업은 `--detach`로 돌린다. 로컬 명령이 끊겨도 원격 작업이 계속된다.

### 전체 파이프라인 완주 (2026-09-30, L4, 합성 촬영본 44장, 1600px, 7,000회, 오리기 사용)

`synth` → `pose` → Modal(`segment` → `train`) → 로컬 다운로드 → 뷰어까지 한 번에 돌린 결과.

| 항목 | 값 |
| --- | --- |
| 오리기 (SAM 2) | 16.6초 |
| 학습 / 과금된 전체 시간 | 409.4초 / 429.6초 |
| 비용 | 약 $0.10 (약 145원) |
| PSNR 학습 / 채점 | 36.8dB / **29.1dB** |
| 가우시안 | 85,224개 (상한 30만) |
| 파일 | `.ply` 21.1MB / `.spz` 1.4MB (**15배 차이**) |
| 뷰어가 측정한 크기 | **11.8 × 11.9 × 12.0 cm** |

마지막 줄이 이 프로젝트의 핵심 주장에 대한 종단 검증이다. 합성 물체는 지름 12cm · 높이 12cm
원통인데, 인쇄 매트 → 카메라 위치 → 학습 → `.ply` → 뷰어까지 거친 뒤 **오차 2mm 이내**로 나왔다.
크기를 알 수 없는 일반 photogrammetry 경로와 갈리는 지점이다.

뷰어로 눈으로 확인한 것: 원통이 바닥 평면에 서 있고, **배경이 깨끗하다**(오리기가 실제로 동작).
무늬 없는 단색 원통이라 표면에 줄무늬 결이 남고 윗면은 위 링이 70°까지만 올라가 관측이 얕다.

### 옛 실측 (2026-09-29, 오리기 이전, 상한 100만)

| 항목 | 값 |
| --- | --- |
| 학습 / 전체 과금 시간 | 551초 / 603초 (약 10분) |
| 비용 | 약 $0.13 (약 190원) |
| 가우시안 / 파일 | 100만 개 (상한) / `.ply` 165MB, `.spz` 11MB |
| PSNR 학습 / 채점 | 21.8dB / 15.7dB (차이 6dB, 기준 3dB 초과) |

### 알려진 문제: `.spz`를 뷰어가 못 읽는다

- 워커는 Niantic 공식 인코더로 `.spz`를 쓰는데 이게 **버전 4**(매직 `NGSP`, ZSTD)다.
- Spark 2.2.0(최신)은 **gzip으로 감싼 v1~3만** 읽는다. v4를 주면 `Invalid gzip header`로 실패한다.
  Spark이 직접 쓴 `.spz`의 매직이 `1f 8b`(gzip)인 것으로 확인했다. 상류 지원 PR은 아직 미머지.
- **당장의 대응**: 뷰어에는 `.ply`를 넣는다. 뷰어가 `.spz`를 받으면 이 상황을 설명하는 오류를 낸다.
- `packages/viewer/test/spark-contract.test.ts`에 카나리아 테스트를 뒀다. Spark이 v4로 올라가면
  그 테스트가 깨지면서 알려준다.
- **M3에서 정해야 할 것**: 전송 크기가 15배 차이라 모바일에선 `.ply`를 그대로 보낼 수 없다.
  선택지 — (a) Spark v4 지원을 기다린다, (b) 워커가 Spark이 읽는 구버전 SPZ도 같이 쓴다,
  (c) Spark이 지원하는 `.sog`로 간다.

### 품질 과제

1. ~~**배경 커튼**~~ ✅ 해결 (2026-09-29): SAM 2.1로 사진마다 물체를 오려서 학습
   - 힌트: 매트 중심 1cm 위 점을 사진에 투영한 한 점. 사용자 클릭 불필요
     (박스 힌트는 매트까지 잡아 IoU 0.00~0.23, 매트 코너를 배경 힌트로 주면 더 나빠짐 → 폐기)
   - 결과 (합성, 2,000회, 512px): 오리기 IoU 평균 0.989 / 최소 0.986,
     물체 PSNR 8.4dB(상자만) → **35.5dB**(학습) / **28.5dB**(채점). 사진 20장 오리기 약 20초
   - 가정: 물체가 매트 중앙을 덮고 있어야 함 (도넛처럼 가운데가 빈 물체는 실패 가능 → 품질 검사로 걸러 상자 기준 학습)
2. ~~**결과물 크기**~~ ✅ (2026-09-29): `--cap` 기본값 100만 → **30만**. 합성 촬영본, 7,000회, 오리기 사용 시

   | 상한 | 실제 가우시안 | PSNR 학습/채점 | 학습 시간 | `.spz` |
   | --- | --- | --- | --- | --- |
   | 10만 | 4.8만 | 36.7 / 28.9dB | 5.8분 | 0.8MB |
   | 30만 | 8.4만 | 36.8 / 29.0dB | 7.1분 | 1.4MB |
   | 100만 | 17.9만 | 36.7 / 29.1dB | 8.5분 | 2.9MB |

   단색 원통이라 차이가 없음. 실물 촬영 후 재조정.
3. ~~Dockerfile / setup-gpu.sh 설치 버그~~ ✅ (2026-09-29): Ubuntu 22.04(Python 3.10) 기반 +
   `--extra-index-url` + `python3-dev`(spz 빌드) + pip 업그레이드. Modal에서 Dockerfile 그대로 빌드해 테스트 48개 통과

### 뷰어 (2026-09-29)

`packages/viewer` — Three.js + Spark(MIT). `npm run dev`로 데모가 뜨고 `splat 열기`로 로컬
`.ply` / `.spz`를 바로 확인할 수 있다. 이로써 "결과를 눈으로 볼 도구가 없다"는 제약이 풀렸다.

- **좌표계를 실측으로 확정했다.** Spark은 splat 파일을 **그대로** 읽는다(뒤집지 않는다).
  Spark 문서에 없는 내용이라 기준 도형(`test/fixtures/axes.ply`)을 만들어 디코딩 결과를 대조해
  확인했다. 따라서 Z-up(re3draw) → Y-up(Three.js) 변환은 뷰어가 직접 한다: `(x, y, z) → (x, z, -y)`.
  이 변환이면 물체의 정면(−Y)이 Three.js 기본 카메라 쪽(+Z)을 향한다.
- 실제 치수를 쓴다: 카메라 거리와 near/far를 물체 크기에서 계산하고, 바닥 격자는 1cm 눈금이다.
  (기본 near 0.1은 12cm 물체를 통째로 삼킨다.)
- 테스트 23개. 순수 계산은 node에서, 실제 렌더링은 헤드리스 Chromium(SwiftShader)에서 픽셀을
  검사한다 — 흰 캡이 위쪽 절반에, 빨강(+X)이 오른쪽 절반에 있는지. GPU 없이 돈다.

### 다음 할 일

1. 실물 촬영으로 오리기 품질 확인 (합성은 단색 원통이라 쉬운 편)
2. 7,000회 / 30,000회 비교 (실물 촬영본으로)
3. 실물 촬영 검증 ([m2-real-capture-test.md](m2-real-capture-test.md))
4. 뷰어: 모바일 터치 조작 다듬기, 큰 splat용 LOD 연결 (Spark 지원), `.glb` 메시는 워커가
   메시를 만들게 된 뒤

### 나중에 할 일 (보류)

- **촬영 가이드 앱 (기획서 F1)**: 2026-09-29 제안, 계획으로만 보류.
  - 목적: 사람이 "어느 각도를 찍었고 어디가 비었는지"를 기억하지 않아도 되게 한다.
  - 형태: 구가 아니라 **반구(돔)**. 물체가 매트 위에 있어 바닥면은 찍을 수 없다.
    링 3개(아래 10~15°, 중간 30~45°, 위 60~75°) × 방위 칸을 돔으로 보여 주고, 찍을 때마다 칸을 채운다.
  - 선택지 (결정 전):
    - **웹 앱 먼저**: 폰 브라우저에서 매트를 실시간 인식해 현재 각도를 돔에 표시. 설치 불필요,
      iPhone/Android 모두. 1~2주. 브라우저에서 ChArUco 인식이 되는지 먼저 짧게 검증해야 함.
    - **Flutter 앱 (원래 M4)**: ARCore/ARKit로 매트 없이도 각도 추적, LiDAR 경로까지 확장. 4주 이상, 스토어 배포 필요.
  - 함께 넣을 것: 매트 인식 여부 표시(인식 안 된 사진은 학습 불가), 흔들림 경고, 첫 장 FRONT 안내,
    아래 링 각도 안내(0°가 아니라 10~15°).
- **매트 대신 무늬 있는 바닥 쓰기 (기획서 P2 "마커 없는 모드"의 발전형)**: 2026-09-29 제안, 보류.
  - 아이디어: 인쇄 매트 대신 책상 상판이나 보자기를 **먼저 촬영해 등록**하고, 그 위에 물체를 놓고 찍는다.
    AR의 "이미지 타깃"과 같은 원리. 프린터 없이 누구나 할 수 있다는 게 가장 큰 장점.
  - 배경 조건: **무늬가 복잡하고 불규칙해야 함**(무늬 보자기, 신문지, 잡지). 단색 천은 특징점이 없어
    불가, 나뭇결은 반복 무늬라 불안정. 천은 구김 없이 평평하게.
  - 약점: 낮은 각도(아래 링)에서 자연 무늬 매칭이 급격히 실패 / 실제 크기를 모름 → **신용카드**
    (85.6×54mm, 규격 고정)를 함께 놓아 크기 기준으로 사용 / 특징점 매칭 엔진을 새로 만들어야 함.
  - 덤: 배경 사전 촬영본과 달라진 부분 = 물체라서 물체 오리기 힌트로도 쓸 수 있다.
  - 순서: 매트로 M2 품질 기준선을 먼저 확보한 뒤, 이 방식을 매트 대비 수치로 비교한다.

## 새 PC에서 개발 환경 만들기

Python은 **3.10 이상** (Windows PC는 3.11.5). 가상환경은 `packages/worker/.venv`에 둔다
(`.gitignore`에 포함, PC마다 새로 만든다).

### Mac (zsh)

```bash
git clone https://github.com/tawool83/re3draw.git   # 이미 있으면 git pull
cd re3draw/packages/worker

python3 --version                  # 3.10 이상인지 확인. 없으면: brew install python@3.11
python3 -m venv .venv
source .venv/bin/activate          # 새 터미널마다 다시 실행
pip install --upgrade pip
pip install -e ".[dev]"            # numpy, opencv, pillow, pytest
pip install modal                  # GPU 학습 실행용

pytest                             # 전부 통과해야 정상 (GPU 테스트는 skip)
```

Modal 연결 (PC마다 한 번):

```bash
python -m modal setup              # 브라우저가 열리면 구글 계정으로 로그인 → 승인
```

- 토큰은 `~/.modal.toml`에 저장된다. 이 파일은 저장소에 올리지 않는다.
- Apple Silicon Mac에는 CUDA가 없어서 `train`을 로컬에서 돌릴 수 없다. 학습은 Modal로 보낸다.
  `mat` / `pose` / `synth` / `pytest`는 Mac에서 그대로 된다.
- `opencv-contrib-python-headless`는 Apple Silicon용 휠이 있어 별도 빌드가 필요 없다.

### Windows (PowerShell)

`re3draw-worker` 명령은 가상환경 안에만 설치돼 있다. 가상환경을 켜지 않고 실행하면
**"'re3draw-worker' 용어가 ... 인식되지 않습니다"** 오류가 난다.

```powershell
cd C:\repository\re3draw\packages\worker
.\.venv\Scripts\Activate.ps1                 # 프롬프트 앞에 (.venv)가 붙으면 성공. 새 창마다 다시 실행
re3draw-worker mat --board a3 -o ..\..\out\mat_a3.pdf
python -m pytest
python -m modal run modal_app.py::selftest
```

- 켜지 않고 쓰려면 경로를 붙인다: `.\.venv\Scripts\re3draw-worker.exe mat --board a3 -o ..\..\out\mat_a3.pdf`
- `Activate.ps1`이 "스크립트를 실행할 수 없습니다"로 막히면 한 번만 실행:
  `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
- 처음 설치(가상환경이 없을 때)는 아래 Git Bash 절차와 같고, 경로만 `.venv\Scripts\`로 바꾼다.

### Windows (Git Bash)

```bash
cd /c/repository/re3draw/packages/worker
python -m venv .venv
.venv/Scripts/pip install -e ".[dev]"
.venv/Scripts/pip install modal
.venv/Scripts/python -m pytest
.venv/Scripts/python -m modal setup
```

- Windows에서는 `modal` 명령이 PATH에 없다는 경고가 나오지만 `python -m modal`로 쓰면 된다.
- Git Bash 경로(`/c/...`)를 Windows용 Python에 인자로 넘기면 경로를 못 찾는다.
  `$(cygpath -w <경로>)`로 바꿔서 넘긴다.

## 환경별 알려진 차이

| 항목 | Windows | Mac |
| --- | --- | --- |
| venv 실행 파일 | `.venv/Scripts/` | `.venv/bin/` |
| 로컬 학습(`train`) | 불가 (CUDA GPU 없음) | 불가 (CUDA 없음) |
| OpenCV 버전 | 5.0.0 설치됨 | 설치 시점 최신 (4.8 이상이면 동작) |
| 줄바꿈 | 커밋 시 LF→CRLF 경고 (무해) | 없음 |
