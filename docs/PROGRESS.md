# 진행 상황 / 개발 환경 메모

> 여러 PC(회사 Windows, 집 Mac)에서 이어서 개발하기 위한 메모. 새 PC에서 시작할 때 이 문서부터 본다.
> 최종 업데이트 2026-09-29 (SAM 2 물체 분리로 배경 커튼 해결)

## 현재 상태

| 마일스톤 | 상태 | 비고 |
| --- | --- | --- |
| M1 매트 + 카메라 위치 | ✅ 완료 | `re3draw-worker mat / pose / synth` |
| M2 학습 워커 | ✅ 합성 데이터에서 동작 | `re3draw-worker segment` (SAM 2) + `train` (gsplat). 배경 커튼 해결 |
| M2 GPU 실행 환경 | ✅ Modal 연결 완료 | `packages/worker/modal_app.py`. GPU 자체 테스트 48개 통과 (L4) |
| M2 실물 촬영 검증 | ⏳ 대기 | [m2-real-capture-test.md](m2-real-capture-test.md) |
| M2 뷰어 | ⏳ 대기 | `packages/viewer` |

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

### 첫 실측 (2026-09-29, L4, 합성 촬영본 44장, 1600px, 7,000회)

| 항목 | 값 |
| --- | --- |
| 학습 / 전체 과금 시간 | 551초 / 603초 (약 10분) |
| 비용 | 약 $0.13 (약 190원) |
| 가우시안 / 파일 | 100만 개 (상한) / `.ply` 165MB, `.spz` 11MB |
| PSNR 학습 / 채점 | 21.8dB / 15.7dB (차이 6dB, 기준 3dB 초과) |

### 품질 과제

1. ~~**배경 커튼**~~ ✅ 해결 (2026-09-29): SAM 2.1로 사진마다 물체를 오려서 학습
   - 힌트: 매트 중심 1cm 위 점을 사진에 투영한 한 점. 사용자 클릭 불필요
     (박스 힌트는 매트까지 잡아 IoU 0.00~0.23, 매트 코너를 배경 힌트로 주면 더 나빠짐 → 폐기)
   - 결과 (합성, 2,000회, 512px): 오리기 IoU 평균 0.989 / 최소 0.986,
     물체 PSNR 8.4dB(상자만) → **35.5dB**(학습) / **28.5dB**(채점). 사진 20장 오리기 약 20초
   - 가정: 물체가 매트 중앙을 덮고 있어야 함 (도넛처럼 가운데가 빈 물체는 실패 가능 → 품질 검사로 걸러 상자 기준 학습)
2. **결과물 크기**: 작은 물체에 가우시안 100만 개는 과함 → `--cap` 기본값을 20만~30만으로 낮추는 안 검토
3. `docker/Dockerfile`, `scripts/setup-gpu.sh`에 Modal에서 고친 설치 버그가 남아 있음
   (gsplat 휠은 Python 3.10 전용, `--index-url` 대신 `--extra-index-url` 필요)

### 다음 할 일

1. 실물 촬영으로 오리기 품질 확인 (합성은 단색 원통이라 쉬운 편)
2. `--cap` 기본값 조정, 7,000회 / 30,000회 비교
3. Dockerfile / setup-gpu.sh 설치 버그 수정
4. `docs/m2-gpu-training.md`를 Modal 중심으로 개편
5. 실물 촬영 검증 ([m2-real-capture-test.md](m2-real-capture-test.md))

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
