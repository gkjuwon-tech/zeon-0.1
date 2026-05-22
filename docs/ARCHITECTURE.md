# ZEON Architecture (target, end-of-year)

> 이 문서는 **1년 후의 ZEON이 어떻게 생겼는지** 그리는 설계도.
> 현재 코드는 Phase 0 양념까지만. 이 문서는 *목적지*.

---

## 1. 핵심 가설

**기존 LLM**: 큰 사전(embedding + FFN) + 단일 forward pass의 attention 추론.
**ZEON**: 같은 크기의 사전 (transplant) + 잠재 공간에 학습된 가상 머신.

수식으로 비교:

```
기존 LLM:    y = LLM(x)
            = Decoder(Embed(x))   # single forward pass

ZEON:        h_0  = OuterEncoder(Embed(x))
            W_0   = InitWorkspace(h_0)
            for t in 1..K:
                op  = Router(W_{t-1}, h_0)              # 어떤 operator 부를지
                W_t = ApplyOperator(op, W_{t-1}, h_0)   # workspace 갱신
                v_t = Verifier(W_t, x)                  # step 평가
                if HaltCritic(W_t) > threshold: break
            y = LMHead(Project(W_K))
```

여기서 `K_max` 는 작고 (16~32), `len(x)` 도 작아도 됨 (16k~32k). 대신 위
loop 자체가 **학습된 가상 머신** — 매 step마다 모델이 *프로그램을 실행*
한다.

---

## 2. 시스템 블록 다이어그램

```
                       ┌─────────────────────────────────────┐
                       │           Input tokens (x)          │
                       └─────────────────┬───────────────────┘
                                         │
                                ┌────────▼────────┐
                                │  Token Embed    │ ← transplanted, frozen
                                └────────┬────────┘
                                         │
                          ┌──────────────▼──────────────┐
                          │   Outer Transformer Stack   │ ← transplanted, mostly frozen
                          │   (~4 layers, no halt)      │
                          └──────────────┬──────────────┘
                                         │ h_0 (B, T, D)
                                         │
                  ┌──────────────────────▼──────────────────────┐
                  │             Workspace Init                  │
                  │   slots W_0 ∈ (B, T, S, D), S=16            │
                  │   role_emb (learned per slot)               │
                  └──────────────────────┬──────────────────────┘
                                         │
                       ╔═════════════════▼═════════════════╗
                       ║   LATENT VIRTUAL MACHINE LOOP     ║
                       ║   for t in 1..K_max:              ║
                       ║                                   ║
                       ║   ┌──── ROUTER ─────────────────┐ ║
                       ║   │ W_{t-1}, h_0 → op_idx top-k │ ║
                       ║   └────────────┬────────────────┘ ║
                       ║                │                  ║
                       ║   ┌────────────▼────────────────┐ ║
                       ║   │      READ HEADS (M)         │ ║
                       ║   │ Pull from h_0 + W_{t-1}     │ ║
                       ║   └────────────┬────────────────┘ ║
                       ║                │                  ║
                       ║   ┌────────────▼────────────────┐ ║
                       ║   │   APPLY OPERATOR (sparse)   │ ║
                       ║   │ FFN_{op} + Attn_{op}        │ ║
                       ║   └────────────┬────────────────┘ ║
                       ║                │                  ║
                       ║   ┌────────────▼────────────────┐ ║
                       ║   │      WRITE HEADS (M)        │ ║
                       ║   │ Gated update of W slots     │ ║
                       ║   └────────────┬────────────────┘ ║
                       ║                │                  ║
                       ║                │ W_t              ║
                       ║                ├──→ VERIFIER ──→ score_t (training only)
                       ║                │                  ║
                       ║   ┌────────────▼────────────────┐ ║
                       ║   │      HALT CRITIC            │ ║
                       ║   │ W_t → P(halt) ──→ break?    │ ║
                       ║   └────────────┬────────────────┘ ║
                       ╚════════════════╪══════════════════╝
                                        │ W_K
                                        │
                          ┌─────────────▼─────────────┐
                          │    Workspace → Hidden     │
                          │    Project(W_K) ∈ (T, D)  │
                          └─────────────┬─────────────┘
                                        │
                          ┌─────────────▼─────────────┐
                          │       LM Head (frozen)    │ ← transplanted
                          └─────────────┬─────────────┘
                                        │
                              ┌─────────▼─────────┐
                              │ logits / sampling │
                              └───────────────────┘

Optional at inference time (Phase 5):
  Run the LVM loop K_rollout=8 times with different noise injections,
  pick the rollout with lowest Energy Head score.
```

---

## 3. 컴포넌트별 사양

### 3.1 Embedding + Outer Stack + LM Head (Phase 0, 완료)

**역할**: 세상에 대한 *지식* 보관. Transplant 해서 freeze.

**구성**:
- `embed_tokens`: HF base 모델에서 그대로 복사
- 4개 transformer block: base 모델의 *마지막* 4개 layer (가장 추상적인 표현)
- `lm_head`: base 모델에서 그대로 복사

**왜 freeze**: 추론은 새로 짜지만 *지식*은 거저 받아 쓴다. 지식 학습은
이미 base 모델이 수조 토큰으로 끝낸 일.

---

### 3.2 Workspace Bank (Phase 1)

**역할**: 사고의 *재료 보관소*. 구조화된 작업 메모리.

**상태 텐서**:
```python
slots: (B, T, S, D)
  B = batch
  T = sequence length (per-token reasoning)
  S = number of slots (16 추천)
  D = hidden dim
```

각 토큰 위치별로 *자기 workspace* 가짐. 컨텍스트 토큰 t 의 workspace 슬롯
들은 그 토큰의 "사고 재료". `T` 차원이 의외인데, 이렇게 안 하면 시퀀스 정보
손실. 학습 동안 efficiency는 시퀀스 압축 (압축된 토큰만 latent VM 통과)
또는 packed-batch 로 회피.

**Role embedding**:
- 슬롯 i 의 학습된 임베딩 `role_emb[i] ∈ R^D`
- 슬롯 i 의 초기값 = `role_emb[i]` + 입력 의존 신호
- 슬롯의 의미는 *학습으로* 정해짐 (사람이 라벨링 안 함)

**Diversity loss**: 학습 시 슬롯들 간 코사인 유사도 페널티.

---

### 3.3 Operator Library (Phase 2)

**역할**: 사고의 *도구상자*. 학습된 명령어 집합.

**Operator 1개의 구성**:
```python
class Operator(nn.Module):
    norm: RMSNorm                      # (D,)
    sub_attn: SmallAttention           # workspace 슬롯들 간 attention
    sub_ffn: SwiGLU(D, d_inter_small)  # 작은 FFN
    type_emb: Parameter(D,)            # operator의 정체성 임베딩
```

**라이브러리**:
- N = 32 (또는 64) 개의 operator
- 각각 작음 (전체 파라미터 ≈ outer stack 1개 layer 정도)
- Top-k routing (k=2 ~ 4)

**라우터**:
```python
def Router(W, h_0):
    pooled = mean(W, dim=slots)        # (B, T, D)
    logits = pooled @ all_op_type_emb  # (B, T, N)
    return top_k(softmax(logits), k=2)
```

**Sparse forward**: 호출된 operator만 실행 (메모리 절약).

---

### 3.4 Read / Write Heads (Phase 1 + Phase 2)

**Read heads (M=4)**:
- input: `(W_{t-1}, h_0, op_type)` → reads `r ∈ R^D` (per head)
- pull 정보를 workspace 슬롯들 + outer hidden에서

**Write heads (M=4)**:
- input: `(operator_output, W_{t-1})` → gated update of W slots
- slot별 sticky-ness gate 학습됨
- 형식:
  ```
  W_t[i] = (1 - gate_i) * W_{t-1}[i] + gate_i * update_i
  ```

---

### 3.5 Verifier Head (Phase 3)

**역할**: 매 step 의 workspace 가 *정답 방향으로* 가고 있는지 평가.

**구성**:
- 본체보다 작은 트랜스포머 (1/10 ~ 1/4)
- 입력:
  - 원래 질문 (x)
  - 현재 workspace (W_t)
- 출력: `score_t ∈ [0, 1]` (정답 근접도)

**학습 데이터**:
- Stage 3a: teacher 풀이 + 정답 → supervised
- Stage 3b: 본체 모델의 high-confidence 답 → self-training

**사용**:
- 학습 시: `aux_loss = -mean(log score_t)` (정답 방향이면 score 높게)
- 추론 시: optional (early-exit 보조)

---

### 3.6 Halt Critic (Phase 4)

**역할**: 매 step "지금 멈출까 계속할까" 결정. RL로 학습.

**구성**:
```python
class HaltCritic(nn.Module):
    policy: SmallNet     # W_t → logit_halt
    value: SmallNet      # W_t → V(state)
```

**Reward**:
- `r = 1` if final answer correct else `0`
- `- c * step_count` (step 비용)
- optional: `+ alpha * verifier_score` (dense bonus)

**학습**:
- Warm-up: supervised (verifier score ≥ threshold 면 halt)
- Main: REINFORCE → PPO
- KL prior baseline (PonderNet 양념을 baseline 으로 재활용)

---

### 3.7 Energy Head (Phase 5)

**역할**: 다중 rollout 중 *어떤 답이 가장 신뢰할만한지* 점수.

**구성**:
```python
class EnergyHead(nn.Module):
    encoder: SmallNet    # W_K (final workspace) → energy
    score: Linear(D, 1)
```

**학습 (contrastive)**:
- 같은 입력에 K rollout
- positive = 정답 답안의 W_K
- negative = 오답 답안의 W_K
- margin loss

**사용 (inference)**:
- K=8 rollout 병렬 실행
- min-energy rollout 의 답을 출력

---

## 4. 학습 목표 (training objective)

전체 loss (Phase 5 시점):

```
L_total = L_lm                               # next-token CE (Phase 0+)
        + λ_ponder * L_ponder                # PonderNet KL  (Phase 0+, annealing)
        + λ_distill * L_distill              # base model distill (Phase 0+)
        + λ_diversity * L_slot_diversity     # workspace diversity (Phase 1+)
        + λ_balance * L_op_load_balance      # operator load balance (Phase 2+)
        + λ_verifier * L_verifier            # per-step verifier signal (Phase 3+)
        + λ_rl * L_halt_rl                   # halt critic RL (Phase 4+)
        + λ_energy * L_energy_contrast       # energy head contrast (Phase 5+)
        + λ_compress * L_self_distill        # latent compression (Phase 6+)
```

각 λ 는 phase 따라 도입/제거 가능. 모든 컴포넌트가 *한 번에* 학습되지
않고, **점진적으로 추가**된다.

### 데이터 믹스
- Pre-distillation: base model output (FineWeb-edu 등 일반 코퍼스)
- Reasoning fine-tune: GSM8K, MATH, BBH, Open-Math-Instruct 등
- RL phase: 정답 라벨 있는 문제 위주 (정답이 binary reward의 source)

---

## 5. 인터페이스 (HF 호환 유지)

ZEON 은 **항상 `AutoModelForCausalLM.from_pretrained()` 로 로드 가능**해야 함.
이건 비협상.

```python
from transformers import AutoModelForCausalLM
import zeon  # 등록 트리거

model = AutoModelForCausalLM.from_pretrained("path/to/zeon-checkpoint")
out = model.generate(input_ids=..., max_new_tokens=128)
```

내부적으로 LVM loop이 돌든 말든, 외부 API는 동일.

이게 깨지면 학습/평가 인프라 전체가 깨짐. 매 Phase 끝나면 이거 검증:
```bash
pytest tests/test_generate_and_cache.py
pytest tests/test_transplant.py
```

---

## 6. 디렉토리 구조 (목표)

```
zeon/
├── __init__.py                  # AutoModel 등록
├── config.py                    # ZeonConfig (모든 phase의 flag 포함)
├── modeling_zeon.py             # ZeonForCausalLM (top-level)
├── outer.py                     # OuterAttention, TransformerBlock (Phase 0)
├── recurrent.py                 # RecurrentCore + ZeonBlock (Phase 0 baseline)
├── workspace.py                 # WorkspaceBank, Read/Write heads (Phase 1)
├── operators.py                 # OperatorLibrary, Router (Phase 2)
├── verifier.py                  # VerifierHead (Phase 3)
├── halting.py                   # PonderNet (Phase 0) + HaltCritic (Phase 4)
├── energy.py                    # EnergyHead, multi-rollout (Phase 5)
├── distill.py                   # Self-distillation (Phase 6)
├── transplant.py                # base model → ZEON 이식
└── losses.py                    # 통합 loss 조립

scripts/
├── transplant_from_hf.py
├── train_phase{1,2,3,4,5,6}.py  # phase별 학습 스크립트
└── eval_reasoning.py

tests/
├── test_recurrent_shape.py       # Phase 0
├── test_generate_and_cache.py    # Phase 0
├── test_transplant.py            # Phase 0
├── test_innovations.py           # Phase 0 양념
├── test_workspace.py             # Phase 1
├── test_operators.py             # Phase 2
├── test_verifier.py              # Phase 3
└── ...

docs/
├── ROADMAP.md                    # 12개월 계획
├── ARCHITECTURE.md               # 이 문서
└── PHASE_NOTES/                  # phase별 회고 (실패 케이스 포함)
    ├── phase1_workspace.md
    ├── phase2_operators.md
    └── ...
```

---

## 7. 절대 어기지 않는 원칙

1. **모든 컴포넌트는 ablation flag 를 가진다.** 끄고 켤 수 있어야 함.
2. **HF 호환은 비협상.** `AutoModelForCausalLM` 으로 로드 안 되면 phase 실패로 간주.
3. **벤치마크 없는 phase 없음.** 매 phase 끝엔 작은 reasoning benchmark 점수.
4. **회고 노트는 글로.** 머리속에 있는 건 잊는다.
5. **추론 비용은 base 모델의 5배를 넘지 않는다.** 넘으면 의미 없음 (그냥 큰 모델 쓰는 게 나음).

---

## 8. 한 줄

**ZEON 은 트랜스포머가 아니다. ZEON 은 트랜스포머를 *지식 저장소로 사용하는*
잠재 공간 가상 머신이다.**

매 코드 추가 전에 자문:
> "이게 트랜스포머에 양념 한 스푼 더 친 거 아닌가?"

답이 yes 면 그 코드 안 짠다.
답이 no 면 짠다.
