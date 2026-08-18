# Repository Structure

```text
ASCP-2026/
├── RL/                 # 환경, 상태, rollout, constraints, loader
├── model/              # encoder, decoder, FiLM
├── experiments/        # 공통 학습 진입점
├── evaluation/         # 평가 CLI 구현과 LP/IP master
├── baselines/          # 기존 방법 비교 진입점
├── analysis/           # constraint, 데이터 및 결과 분석
├── diagnose/           # 모델 동작 진단
├── paper_experiments/  # 로컬 전용 논문 실험 구성 (Git 제외)
├── paper_runs/         # 로컬 전용 실행 결과 (Git 제외)
├── journel/            # 로컬 전용 저널 계획, 회의, 원고 (Git 제외)
├── legacy/             # 로컬 전용 과거 구현 (Git 제외)
└── canonical code only  # 루트 compatibility wrapper 없음
```

대용량 AAAI 로그, checkpoint, 백업, 가상환경과 로컬 저널 문서는
저장소 밖 `/home/hyrn2/github/AAAI/`에 보존함.

## 작업 원칙

- 공통 모델은 `RL/`, `model/` 한 곳만 수정함.
- AAAI 재현 코드는 `paper_experiments/AAAI/`에서 찾음.
- 신규 저널 코드는 `paper_experiments/journal/`에 추가함.
- 신규 결과는 `paper_runs/journal/`에 저장함.
