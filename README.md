# ZEON

> **우리는 모델을 키우지 않는다. 우리는 모델이 *생각하는 방식*을 바꾼다.**

ZEON은 LLM이 **잠재 공간 안에 학습된 가상 머신**을 운용하도록 만드는
아키텍처 연구 프로젝트입니다. 컨텍스트를 늘리는 대신, 짧은 컨텍스트
안에서 **추론 깊이를 폭발적으로** 키우는 방향으로 갑니다.

---

## 우리가 *안* 만드는 것

- **트랜스포머 K번 굴리기.** Universal Transformer ('18). 끝.
- **어텐션 갈아끼우기.** Mamba / RWKV 류는 *긴 컨텍스트*가 의미.
  우리는 컨텍스트가 짧아도 되는 전제.
- **출력 토큰 Chain-of-Thought.** 비효율적. 컨텍스트 폭발. 모든 토큰이
  같은 compute. 우리는 사고를 *출력 밖에서* 한다.
- **더 큰 모델 + 더 많은 데이터.** Scaling-only는 우리 영역이 아님.

위 항목 중 하나라도 "근데 거기에 양념 좀 더…"가 되면 ZEON 아님.

## 우리가 *진짜* 만드는 것

추론을 **잠재 가상 머신 (Latent Virtual Machine, LVM)** 안에서 수행하는
모델. 구성 요소:

| 컴포넌트 | 역할 | 영감 |
|---|---|---|
| **Workspace Bank** | 16개 슬롯의 구조화된 작업 메모리. 슬롯마다 역할 (가설/증거/결론/스크래치…) | Baddeley 작업기억, Slot Attention |
| **Operator Library** | 16~64개 학습된 mini-expert. 추론 step마다 라우터가 sparse 선택 | MoE, Neural ISA |
| **Verifier Head** | 본체보다 작은 보조 모델. 매 step의 워크스페이스 상태 평가 → dense gradient | Process reward model |
| **Halting Critic** | RL로 학습된 중단 정책. 보상: 정답 맞춤 - step 비용 | PonderNet + REINFORCE |
| **Energy Head** | 다중 latent rollout 점수 매기고 베스트 픽 | EBM, Self-Consistency |

전부 **하나의 HF-호환 모델** 안에서 작동.
전부 **개별로 ablation** 가능하게 설계.
전부 **단일 트랜스플랜트** (기존 큰 모델에서 지식만 떼옴) 으로 시작.

---

## 한 줄 가설

기존 LLM은 *큰 사전 + 단순 추론*.
ZEON은 *작은 사전 + 거대한 추론 엔진*.

지식은 베이스 모델 (Qwen / Llama / DeepSeek) 에서 transplant 해서 얼리고,
*그 위에 추론 엔진을 새로 짠다.*

---

## 현재 상태

**Phase 0 / 7** — 토대 단계.

지금 박혀있는 `RecurrentCore` + `StepEmbedding` + `CrossStepMemory` +
PonderNet halt 는 **혁신이 아니라 Phase 0 양념**. Universal Transformer
계열 베이스라인이다. 진짜 일은 Phase 1 (Workspace) 부터.

- [x] Phase 0: HF 호환 베이스, KV cache, generate, 트랜스플랜트, 17 테스트
- [ ] Phase 1: Workspace Bank
- [ ] Phase 2: Operator Library
- [ ] Phase 3: Verifier Head
- [ ] Phase 4: Halting Critic (RL)
- [ ] Phase 5: Parallel Rollouts + Energy Head
- [ ] Phase 6: Self-Distillation in Latent Space
- [ ] Phase 7: 벤치마크 + 배포

전체 계획: [`docs/ROADMAP.md`](docs/ROADMAP.md)
기술 비전: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

---

## 빠른 시작

```bash
pip install -e ".[dev]"
pytest tests/ -q   # 17/17 green
ruff check zeon scripts tests
```

베이스 모델로부터 Phase 0 transplant:

```bash
python scripts/transplant_from_hf.py \
    --src Qwen/Qwen2.5-1.5B \
    --out checkpoints/zeon-phase0
```

---

## 1년 후 우리가 측정하려는 것

기존 베이스 모델의 vanilla 추론 대비 ZEON이:

1. **같은 평균 FLOPs** 기준 추론 벤치에서 **+20% 절대** (GSM8K, MATH, ARC-AGI, BBH)
2. **출력 토큰 5배 적게** 사용 (사고를 latent에서 하니까)
3. **어려운 문제일수록 더 많은 latent step**을 자동 할당 (HC가 똑똑함)
4. 각 컴포넌트(Workspace / Operator / Verifier / HC / Energy) **개별 ablation에서 명백한 기여**

위 4개 중 2개 이상 못 맞추면 **솔직히 인정하고 다음 가설로 간다.**
