# 진행 상황 / 개발 환경 메모

> 여러 PC(회사 Windows, 집 Mac)에서 이어서 개발하기 위한 메모. 새 PC에서 시작할 때 이 문서부터 본다.
> 최종 업데이트 2026-09-29 (Modal 학습 첫 실측)

## 현재 상태

| 마일스톤 | 상태 | 비고 |
| --- | --- | --- |
| M1 매트 + 카메라 위치 | ✅ 완료 | `re3draw-worker mat / pose / synth` |
| M2 학습 워커 | 🟡 GPU에서 동작 확인, 품질 개선 필요 | `re3draw-worker train` (gsplat, CUDA 필요). 아래 "품질 과제" |
| M2 GPU 실행 환경 | ✅ Modal 연결 완료 | `packages/worker/modal_app.py`. GPU 자체 테스트 42개 통과 (L4) |
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

1. **배경 커튼**: 물체 상자 안에 뒤쪽 배경이 흐릿한 막으로 학습된다. PSNR 차이 6dB의 주원인.
   → 물체 분리(segmentation) 도입. 라이선스 허용 모델 검토 (SAM Apache-2.0, rembg MIT 등)
2. **결과물 크기**: 작은 물체에 가우시안 100만 개는 과함 → `--cap` 기본값을 20만~30만으로 낮추는 안 검토
3. `docker/Dockerfile`, `scripts/setup-gpu.sh`에 Modal에서 고친 설치 버그가 남아 있음
   (gsplat 휠은 Python 3.10 전용, `--index-url` 대신 `--extra-index-url` 필요)

### 다음 할 일

1. 물체 분리 방식 결정 → 배경 커튼 해결
2. `--cap` 기본값 조정, 7,000회 / 30,000회 비교
3. Dockerfile / setup-gpu.sh 설치 버그 수정
4. `docs/m2-gpu-training.md`를 Modal 중심으로 개편
5. 실물 촬영 검증 ([m2-real-capture-test.md](m2-real-capture-test.md))

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
