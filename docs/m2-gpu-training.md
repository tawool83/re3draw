# M2 GPU 학습 실행 가이드 (Modal)

> 대상: `re3draw-worker pose`로 카메라 위치가 나온 촬영본을 GPU에서 3D로 학습시켜
> `.ply` / `.spz` 스플랫을 얻는다.
> 최종 업데이트 2026-09-29 (Modal 중심으로 개편) · 관련 문서
> [PROGRESS.md](PROGRESS.md) · [m2-real-capture-test.md](m2-real-capture-test.md)

## 이게 무슨 단계인가

사진 여러 장과 각 사진의 카메라 위치를 넣으면, 공간에 수십만 개의 반투명한 타원체(가우시안)를
흩뿌려 놓고 **"이걸 그 카메라 위치에서 렌더링한 그림이 실제 사진과 같아지도록"** 파라미터를 수천~수만 번
고쳐 나간다. 끝나면 사진에 없던 각도에서도 올바르게 보이는 3D 덩어리가 남는다.

미리 학습된 AI 모델을 쓰지 않는다. 물체 하나마다 매번 새로 계산하는 **최적화**다.
(사진에서 물체를 오려내는 SAM 2만 사전 학습 모델이고, 그건 학습 전 준비 단계다.)

GPU가 필요한 이유는 이 반복이 수천~수만 번의 렌더링이고 CUDA 전용 코드로 돌기 때문이다.
**Mac과 Windows에서는 학습을 돌릴 수 없다.** `mat` / `pose` / `synth` / `pytest`는 그대로 된다.

## 왜 Modal인가

GPU를 **시간 단위로 빌려 직접 켜고 끄는 대신**, 함수를 호출하면 컨테이너가 떠서 일하고 끝나면
스스로 사라지는 방식이다. 이 프로젝트에 맞는 이유:

- **끄는 걸 잊을 수가 없다.** 작업이 끝나면 자동 종료되고 과금도 멈춘다. 파드를 빌리는 방식에서
  실제로 돈이 새는 경로는 학습이 아니라 켜두고 잊는 것인데, 그 위험 자체가 사라진다.
- **초 단위 과금**에 최소 과금 단위가 없다.
- **매월 무료 크레딧 $30** 으로 M2 개발 GPU 시간이 거의 충당된다 (L4 기준 약 37시간).
- 기획서 F3(서버 복원 워커)가 원래 Modal 기준이라, **지금 만든 게 그대로 운영 경로**가 된다.

RunPod은 GPU에 직접 접속해 디버깅해야 할 때만 예비로 쓴다. Vast.ai는 개인이 자기 그래픽카드를
빌려주는 장터라 호스트가 누구인지 보장되지 않으므로 **합성 데이터 외에는 쓰지 않는다.**

## 준비 (PC마다 한 번)

```bash
cd packages/worker
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
pip install modal
python -m modal setup              # 브라우저가 열리면 구글 계정으로 로그인 → 승인
```

토큰은 `~/.modal.toml`에 저장된다. **저장소에 올리지 않는다.**

연결 확인 — GPU에서 테스트 전체를 돌려 본다 (약 1분):

```bash
python -m modal run modal_app.py::selftest
```

여기서 통과하면 학습 경로가 실제 GPU에서 동작한다는 뜻이다. 사진을 올리기 전에 문제를 잡을 수 있다.

## 전체 흐름

```
로컬                              Modal
────                              ─────
re3draw-worker pose               (1) 촬영 폴더 업로드 → Volume "re3draw-captures"
  → images/ + sparse/0/           (2) SAM 2로 물체 오리기 → masks/   [segment_image]
                                  (3) gsplat 학습 → splat.ply/.spz   [train_image]
  ← splat-<iters>/ 자동 다운로드   (4) 결과를 Volume에 커밋
```

**이미지가 두 개인 이유**: SAM 2는 torch 2.5 이상이 필요한데 gsplat의 미리 빌드된 휠은
torch 2.4 + Python 3.10 전용이라 한 이미지에 같이 넣을 수 없다. 그래서 오리기와 학습을 나눠 돌린다.

**오리기 결과는 재사용된다.** 같은 촬영본으로 설정만 바꿔 다시 돌리면 업로드도 오리기도 다시 하지
않는다 (`masks/`가 Volume에 남아 있다).

## 학습 실행

```bash
cd packages/worker
python -m modal run modal_app.py --capture ../../out/capture1 --iters 7000
```

`--capture`는 **로컬** 폴더 경로다 (`images/`와 `sparse/0/`이 있어야 한다).
결과는 그 폴더 옆 `../../out/capture1/splat-7000/`으로 자동으로 내려온다.

**10분이 넘을 것 같으면 `--detach`를 붙인다.** 이러면 로컬 명령이 끊겨도(터미널을 닫거나 노트북을
덮어도) 원격 작업은 끝까지 계속된다.

```bash
python -m modal run --detach modal_app.py --capture ../../out/capture1 --iters 30000
```

연결을 유지한 채 기다리면 결과는 평소처럼 자동으로 내려온다. **중간에 끊었다면** 결과가 Volume에만
남으므로 직접 받는다:

```bash
python -m modal volume get --force re3draw-captures capture1/splat-30000 ../../out/capture1/
```

진행 상황은 [modal.com](https://modal.com)의 대시보드에서 볼 수 있다.

### 옵션

| 옵션 | 기본값 | 설명 |
| --- | --- | --- |
| `--iters` | 30000 | 반복 횟수. 7000으로도 쓸만하고 그만큼 빠르다 |
| `--gpu` | `L4` | `L40S`, `A100`, `H100` 등. 물체 촬영은 L4로 충분하다 |
| `--max-size` | 1600 | 학습에 쓸 사진의 긴 변. 12MP 원본을 그대로 쓰면 느리고 이득이 적다 |
| `--cap` | 0 (= 워커 기본 30만) | 가우시안 최대 개수 |
| `--board` | `a3` | 매트 규격 |
| `--no-segment` | | 물체 오리기를 건너뛰고 상자 기준으로만 학습 (품질 비교용) |
| `--no-upload` | | 이미 Volume에 올라간 촬영본을 다시 올리지 않는다 |

`--iters`/`--cap`/`--gpu`를 바꾸면 결과 폴더 이름이 달라지므로(`splat-7000-cap100k-l40s`) 여러 설정을
나란히 비교할 수 있다.

### 학습 자체의 세부 옵션

`re3draw-worker train --help`의 옵션(`--sh-degree`, `--object-width`, `--val-every`, `--no-spz` 등)은
Modal 진입점이 전부 노출하지는 않는다. 필요하면 `modal_app.py`의 `main()`에서 `args` 목록에 추가한다.

## 진단 (합성 데이터)

물체 오리기가 배경을 실제로 걷어내는지 정답이 있는 합성 촬영본으로 확인한다.
오리기 IoU, 상자만 쓴 학습 vs 오리기를 쓴 학습의 PSNR, 비교 이미지(`사진 | 상자만 | 오리기`)를 낸다.

```bash
python -m modal run --detach modal_app.py::diagnose --iters 2000
python -m modal volume get --force re3draw-captures _diag/2000-512 ../../out/
```

## 결과와 판정

`out/capture1/splat-<iters>/`:

| 파일 | 내용 |
| --- | --- |
| `splat.ply` | 모든 뷰어가 읽는 표준 형식 |
| `splat.spz` | 약 10배 작다. 폰으로 내려보낼 때 쓴다 |
| `train.json` | 아래 지표 |
| `modal.json` | 실제 GPU 이름, 과금된 벽시계 시간 |

| 항목 | 통과 기준 | 위치 |
| --- | --- | --- |
| 채점용 사진 화질 | **PSNR 25dB 이상**이면 양호, 20dB 미만이면 문제 | `train.json` → `psnr_val` |
| 학습/채점 차이 | 두 값 차이가 3dB 이내 | `psnr_train` vs `psnr_val` |
| 가우시안 수 | 5만~30만 (물체 오리기 사용 시) | `gaussians_exported` |
| 오리기 성공 | 제외된 사진이 없거나 설명 가능할 것 | 실행 출력의 `object masks: N kept, rejected ...` |

`psnr_val`이 `psnr_train`보다 많이 낮으면 사진이 부족하거나 각도가 치우친 것이다
(촬영 가이드의 중간·위 링을 확인한다).

## 비용

### 실측 (2026-09-29, L4, 합성 44장, 1600px, 7,000회)

| 항목 | 값 |
| --- | --- |
| 학습 / 과금된 전체 시간 | 551초 / 603초 |
| 비용 | **약 $0.13 (약 190원)** |

### 요율 (2026년 9월, 초 단위 과금)

| GPU | 초당 | 시간당 |
| --- | --- | --- |
| L4 | $0.000222 | 약 $0.80 |
| L40S | $0.000542 | 약 $1.95 |
| A100 40GB | $0.000583 | 약 $2.10 |
| H100 SXM | $0.001097 | 약 $3.95 |

- **무료 크레딧 월 $30** → L4 기준 약 37시간. M2 개발은 여기서 거의 다 충당된다.
- 리전을 넓게(us/eu/ap) 잡으면 **1.5배**, 좁게(us-west 등) 잡으면 **1.75배** 요율이 붙는다.
  특별한 이유가 없으면 리전을 지정하지 않는다.
- 과금은 **컨테이너가 시작해서 사라질 때까지**다. 학습 551초에 과금 603초인 차이(52초)가
  컨테이너 시작 시간이다. 짧은 실험을 여러 번 돌리면 이 고정비 비중이 커진다.

## GPU에 직접 접속해야 할 때 (예비 경로)

Modal은 컨테이너 안에 들어가 디버깅하기가 불편하다. 그럴 때만 RunPod 같은 데서 GPU를 빌리고:

```bash
git clone https://github.com/tawool83/re3draw.git && cd re3draw
bash packages/worker/scripts/setup-gpu.sh     # 설치 후 pytest까지 돌린다
re3draw-worker segment /workspace/capture1
re3draw-worker train   /workspace/capture1 -o /workspace/capture1/splat
```

`packages/worker/docker/Dockerfile`도 같은 환경을 만든다 (M3 운영용).

> ⚠️ 이 경로에서는 **끝나고 파드를 반드시 종료(terminate)** 한다. 빈 파드를 한 달 켜두면
> 약 39만 원이다. Modal을 기본으로 쓰는 이유가 이것이다.

## 알려진 제한 (M2 기준)

- **`.glb` 메시는 아직 없다.** 스플랫에서 메시를 뽑는 것은 별도의 문제이고, 학습 방식 자체를
  바꾸거나 깊이 맵을 융합하는 단계가 필요하다. 뷰어는 `.ply` / `.spz`로 먼저 붙인다.
- **결과를 눈으로 볼 도구가 아직 없다.** M2 후반의 Three.js 뷰어가 그 역할이다.
  그전까지는 `splat.ply`를 기존 스플랫 뷰어에 넣어 확인한다.
- **물체가 매트 중앙을 덮고 있어야 한다.** 오리기 힌트가 매트 중심 위의 한 점이라, 도넛처럼
  가운데가 빈 물체는 실패할 수 있다. 그런 경우는 `--no-segment`로 상자 기준 학습을 쓴다.
- **밑면은 복원되지 않는다.** 찍히지 않은 곳에는 정보가 없다.
- **실물 촬영으로는 아직 검증되지 않았다.** 위 수치는 전부 합성 촬영본(단색 원통) 기준이다.
