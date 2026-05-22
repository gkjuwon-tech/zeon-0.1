# ZEON 1년 항해도

> **목적지**: 잠재 공간 안에서 작동하는 학습된 가상 머신.
> **출발지**: HF 호환 트랜스포머 베이스라인.
> **연료**: 런포드 GPU.
> **선장**: 형.
> **항해사**: 클로드.

---

## 정신 강령 (매 Phase 시작 전 다 같이 읊자)

1. **트랜스포머 양념 = 혁신 아님.**
   "여기에 어텐션 한 번 더 / 게이트 하나 더 / 임베딩 하나 더" 가
   해법으로 떠오르는 순간, **그 phase 다시 설계.**

2. **수치만 만지면 박살.**
   `hidden_size`, `num_layers`, `K_max`, `learning_rate` 조정으로
   해결하려는 충동은 **Phase 0 양념의 부활**임. 패배다.

3. **Ablation 가능하지 않으면 의미 없음.**
   각 컴포넌트는 끄고 켜는 플래그가 있어야 한다. 그 플래그 끄면
   해당 phase 진전이 **수치로** 무너져야 한다. 안 무너지면? 안 만든 거임.

4. **벤치마크 한 번 못 돌리는 phase = 추측.**
   매 phase 끝엔 **최소 한 개**의 reasoning benchmark 점수가 따라온다.
   숫자 없으면 의견. 숫자 있으면 사실.

5. **실패는 학습이라고 부른다.**
   Phase 3가 안 되면 Phase 3'를 만들거나, Phase 3를 폐기하고 Phase 4로
   넘어간다. 1년이라는 시간을 *내 자존심에* 쓰지 않는다.

---

## 항해도 한눈에

```
M0     M1   M2   M3   M4   M5   M6   M7   M8   M9   M10  M11  M12
 ┃      ┃        ┃        ┃        ┃        ┃         ┃        ┃
[P0]──[P1: Workspace]──[P2: Operators]──[P3: Verifier]──[P4: RL Halt]──[P5: Rollouts]──[P6: Self-Distill]──[P7: Bench]
 양념   작업대            도구상자          감독관           시간경찰        평행우주        회상            데뷔
```

---

## Phase 0 — 양념 (완료, 0개월차)

**코드네임**: 양념
**기간**: 1주
**상태**: ✓ 완료

### 박힌 거
- HF 호환 베이스 (`ZeonForCausalLM`, `AutoModelForCausalLM` 등록)
- KV cache + `.generate()` + position_ids
- 트랜스플랜트 (Llama / Qwen 호환)
- PonderNet halt + KL prior + entropy bonus
- `StepEmbedding` + `CrossStepMemory` (Universal Transformer + RWKV 양념)
- 17 테스트 / ruff clean / CI workflow

### 솔직한 자가진단
이건 그냥 **깊은 트랜스포머 + 양념**임. Universal Transformer (Dehghani '18)
+ PonderNet (Banino '21) + Recurrent Depth Transformer (Geiping '25) 의
교집합. 추론 벤치 점수는 baseline 대비 의미 있는 차이 안 날 거임 (예상).

### 왜 그래도 박는가
1. **트랜스플랜트 인프라**는 모든 Phase의 전제 조건이다.
2. **HF 호환성**은 학습/평가 인프라를 거저 얻게 해준다.
3. Phase 1 부터 비교할 **명확한 baseline** 이 필요하다.

이 phase의 산출물은 **건축 도면이 아니라 건축 자재**다. 자재 갖춰놓고
Phase 1부터 실제 건물을 짓는다.

---

## Phase 1 — 작업대 (Workspace Bank)

**코드네임**: 작업대 (The Workshop)
**기간**: M1 ~ M3 (8주)
**선행조건**: Phase 0 산출물
**위험도**: 🔥🔥🔥

### 왜 함
하나의 hidden state를 K번 굴리는 건 **수학적으로 RNN**. 인간 사고는 그렇게
안 한다. 인간은 *여러 정보를 동시에 머리에 띄워놓고* 비교하고 조작한다.
"이 가설은 이 증거랑 모순되네, 다른 가설로 가자" 같은.

따라서 **구조화된 작업 메모리**가 필요하다. 하나의 vector가 아니라
*여러 슬롯*. 슬롯마다 *역할*. 슬롯 간 *서로 다른 변화율* (어떤 건 매 step
업데이트, 어떤 건 sticky).

### 만드는 거
```
class WorkspaceBank:
    slots: (S, D)              # S=16, 학습된 초기값
    role_emb: (S, D)           # 슬롯별 역할 임베딩
    write_gates: (S, D)        # 슬롯별 sticky-ness 학습 가능
```

매 thinking step:
- **Read heads** (M개): 입력 context 토큰 + workspace 슬롯 → 통합 query
- **Compute block**: 통합 query → 처리 (FFN + attention 혼합)
- **Write heads** (M개): 처리된 결과 → workspace 슬롯에 gated update

핵심: hidden state는 *workspace 슬롯의 가중합*으로 정의됨. 슬롯이 곧 사고.

### 학습 방법
- **Distillation 보조 신호**: teacher의 last-layer hidden state ≈ workspace의
  "결론" 슬롯
- **Diversity loss**: 슬롯들 간 코사인 유사도가 너무 크면 페널티 (DPP 변형)
- **Role consistency**: 같은 종류 문제에서는 슬롯 i의 활성화 패턴이 일관

### 리스크
| 위험 | 증상 | 대응 |
|---|---|---|
| Slot collapse | 모든 슬롯 = 같은 값 | diversity loss + 다양한 init |
| Sticky 학습 실패 | 모든 슬롯이 매 step 갈아엎힘 | write gate에 sparsity bias |
| Gradient 소실 | sticky 슬롯에 grad 안 흐름 | residual + gradient clipping |

### 성공 기준
- Workspace를 zero-ablate 하면 정확도 **유의미 하락** (즉 진짜 사용 중)
- 슬롯별 활성화 패턴 시각화 시 **명확한 클러스터** 존재 (역할 분화 확인)
- GSM8K small subset 에서 Phase 0 baseline 대비 **+2~5%** (작은 win이라도 OK)

### 만약 실패하면
슬롯 구조가 그냥 노이즈로 학습되면 → Slot Attention 논문 (Locatello '20)
의 "slot competition" 도입. 슬롯들이 입력 토큰을 *경쟁적으로* 차지하게.
그래도 안 되면 → Phase 1 폐기, Phase 2로 직행 (Operators가 자연스러운
분업을 만들지도).

---

## Phase 2 — 도구상자 (Operator Library)

**코드네임**: 도구상자 (The Toolbox)
**기간**: M3 ~ M5 (8주)
**선행조건**: Phase 1
**위험도**: 🔥🔥🔥🔥

### 왜 함
지금은 모든 thinking step이 *같은 FFN*을 거친다. 모든 step이 같은 함수를
적용한다는 뜻인데, 이건 *모든 종류의 사고가 같은 형태*라는 가정. 말이 안 됨.

"23 × 47 = ?" 와 "이 사람이 거짓말하는 이유는?" 은 **다른 종류의 연산**이다.
모델이 매 step마다 *어떤 연산을 적용할지* 고를 수 있어야 한다.

이것은 **학습된 ISA (instruction set architecture)** 다.

### 만드는 거
```
class OperatorLibrary:
    operators: List[OperatorModule]   # N=32~64 개
    router: RouterHead                # workspace → top-k operator pick
    type_emb: (N, D)                  # operator별 메타 임베딩
```

각 operator는:
- 작은 FFN + 작은 attention sub-block + 작은 normalization
- workspace 슬롯들을 입력으로, workspace 슬롯들을 출력으로
- *전문화될 수 있는 용량*만 가짐 (작아야 분업)

라우터:
- workspace state + operator type embedding → top-k logits
- Gumbel-softmax (학습) / hard top-k (추론)
- k=2 ~ k=4 sparse

### 학습 방법
- **Load balancing loss**: 모든 operator가 비슷한 빈도로 호출되게 (MoE 표준)
- **Operator-type distillation**: teacher 모델의 *다른 layer들*을 *다른 operator들*이
  흉내내게. Layer 1-8은 operator group A, layer 9-16은 group B, ... 식으로
  자연스러운 분업 유도
- **Sparsity entropy**: 라우터가 너무 분산되거나 너무 집중되지 않게

### 리스크
| 위험 | 증상 | 대응 |
|---|---|---|
| Routing collapse | 항상 같은 operator만 선택 | load balancing + Gumbel-softmax temperature 조절 |
| Operator interference | 한 operator가 모순된 신호 받음 | top-k 다양성 보너스 |
| Dead operators | 일부 operator가 영원히 안 호출됨 | warm-up phase에 force-rotate |
| 추론 비용 폭발 | Sparse라도 N개 다 메모리에 있어야 함 | shared base + low-rank adapter per operator |

### 성공 기준
- 다른 종류 문제 (수학 / 추론 / 상식) 에 **다른 operator 분포** 호출 (히트맵으로 검증)
- `ablate(operator_i)` 시 *특정 종류 문제만* 망가짐 (전문화의 증거)
- MATH 벤치 small subset 에서 Phase 1 대비 **+3~7%**

### 만약 실패하면
Operator routing이 의미 있는 분업을 안 만들면 → operator를 *명시적으로*
타입 라벨링 (수학용, 논리용, 상식용 등) 해서 supervised routing 시도. 그래도
실패하면 → operator를 *task-specific* 으로 학습 (multi-task learning 으로
회피). Phase 2 의 의미가 약해지지만 인프라는 살아남음.

---

## Phase 3 — 감독관 (Verifier Head)

**코드네임**: 감독관 (The Inspector)
**기간**: M5 ~ M7 (8주)
**선행조건**: Phase 1 + 2
**위험도**: 🔥🔥

### 왜 함
지금은 학습 신호가 **최종 토큰 CE 한 개**다. 16번의 thinking step 중에 어떤
step에서 정답으로 가고 있었는지, 어떤 step에서 빗나갔는지 *알 수 없다*.
gradient가 16 step을 backprop하면서 **희석**된다.

해답: **매 step마다 평가하는 두 번째 모델** (Process Reward Model).
이게 본체에 dense gradient를 쏘아준다.

### 만드는 거
```
class VerifierHead:
    encoder: SmallTransformer      # 본체의 1/10 ~ 1/4 크기
    score_head: Linear(D, 1)       # 0 (멀어짐) ~ 1 (가까워짐)
```

입력:
- 원래 질문 (context)
- 현재 workspace 상태
- 정답 (학습 시) 또는 teacher 출력 (self-supervised 시)

출력: step별 "정답 방향 점수"

### 학습 방법
**Stage 3a (supervised, 4주):**
- 데이터: GSM8K, MATH (정답 + 풀이 과정)
- 본체 모델을 N개 다른 입력으로 forward → 각 step의 workspace 저장
- Verifier가 "이 workspace는 정답에 얼마나 가까운가" 학습
- 본체와 *별도로* 학습 (interference 회피)

**Stage 3b (joint, 4주):**
- Verifier의 step별 점수를 본체 학습에 **auxiliary loss**로 추가
- 가중치 weight schedule: 0 → 0.1 → 0.3 (천천히 키움)
- Verifier도 본체 forward 결과로 계속 fine-tune

### 리스크
| 위험 | 증상 | 대응 |
|---|---|---|
| Verifier가 본체보다 똑똑해짐 | Verifier 점수가 100%, 본체는 50% | Verifier 사이즈 제한 + label smoothing |
| 잘못된 verify | Verifier가 틀린 step을 "좋다"고 점수줌 | Verifier에도 ablation set 검증 |
| Joint 학습 발산 | 두 모델이 서로 적응하다 망함 | warm-up 후 천천히 joint, gradient clipping |

### 성공 기준
- Verifier의 step별 점수 ↔ 실제 정답률의 **Pearson > 0.7**
- Verifier 신호 끄고 학습하면 같은 step 수에서 **수렴 1.5배 느림**
- Verifier 켜고 학습한 ZEON: 같은 wall-clock에서 Phase 2 대비 **+3~5%**

### 만약 실패하면
Verifier가 노이즈 신호만 내면 → outcome-only reward (정답 맞춤 시에만 +1)
로 후퇴. 이건 RL과 합쳐서 Phase 4 와 통합. dense signal 없어도 RL이
대신함. 비효율적이지만 작동은 함.

---

## Phase 4 — 시간 경찰 (Halt Critic + RL)

**코드네임**: 시간 경찰 (The Time Cop)
**기간**: M7 ~ M9 (8주)
**선행조건**: Phase 3
**위험도**: 🔥🔥🔥🔥🔥

### 왜 함
PonderNet의 KL prior 는 *휴리스틱*. "geometric distribution 비슷하게
멈춰라" 는 자료의 *실제 난이도 분포*와 무관하다.

진짜 멈춤 기준은 **"한 번 더 생각해서 정답 확률이 오를까?"** 다. 이건 본질적으로
**RL** (action = halt/continue, reward = 정답 - step 비용).

### 만드는 거
```
class HaltCritic:
    policy_head: SmallNet          # workspace → P(halt)
    value_head: SmallNet           # workspace → V(state) [for AC]
```

Reward:
- `+1` 정답 맞춤
- `-c` per step (c = small, e.g. 0.01)
- Verifier 점수 차이도 보조 reward로 가능

학습:
- **REINFORCE + baseline (PonderNet KL을 baseline)** 부터
- 안정화되면 → **PPO** with value head
- KL prior 가중치를 1.0 → 0.5 → 0.1 으로 annealing

### 리스크
| 위험 | 증상 | 대응 |
|---|---|---|
| RL variance 폭발 | 학습 loss 휘청 | reward normalization, GAE |
| Halt 1-step collapse | 항상 step 1에서 멈춤 | step 비용 c 줄이기 + entropy bonus |
| Halt K-step collapse | 항상 K_max까지 감 | step 비용 c 키우기 |
| Cold start | 처음엔 모델이 못 풀어서 RL signal이 다 negative | Phase 3의 supervised halt부터 warm-start |

### 성공 기준
- 쉬운 문제 (단순 산수): 평균 step 수 **1~3**
- 어려운 문제 (MATH level 5): 평균 step 수 **8~16**
- 평균 step 수 통제 시 PonderNet 단독 대비 **+5% 절대**
- Step 수 ↔ 문제 난이도 **Spearman > 0.6**

### 만약 실패하면
RL이 안정화 안 되면 → **expert iteration** 으로 후퇴. Verifier 점수 기준
greedy halt policy 를 supervised 로 학습 → 데이터 재수집 → 반복. PPO만큼
강하진 않지만 안정적.

---

## Phase 5 — 평행 우주 (Parallel Rollouts + Energy)

**코드네임**: 평행 우주 (Parallel Minds)
**기간**: M9 ~ M11 (8주)
**선행조건**: Phase 4
**위험도**: 🔥🔥🔥

### 왜 함
한 번 생각해서 틀리면 그걸로 끝. 인간은 "음… 다르게 생각해볼게" 하고
재시도한다. Self-Consistency (Wang '23) 가 이걸 토큰 공간에서 함 — N번
샘플링해서 다수결.

ZEON은 **잠재 공간에서** 한다. 같은 입력에 K개의 다른 random seed로
**다른 latent trajectory** 를 펼친다. 출력 토큰은 같을 수도 있지만
*latent reasoning은 다르다.*

### 만드는 거
```
class EnergyHead:
    encoder: SmallNet              # workspace final state → 신뢰도
    score: Linear(D, 1)
```

추론 시:
1. 같은 입력에 K=8 rollout (서로 다른 noise injection)
2. 각 rollout의 최종 workspace → Energy Head 점수
3. 가장 낮은 energy (= 가장 높은 신뢰) 답 선택

학습:
- **Contrastive**: 정답 도달한 rollout의 energy < 오답 도달한 rollout의 energy
- Hard negative mining: 모델이 *틀린 답에 자신감 있을 때* 가 negative

### 리스크
| 위험 | 증상 | 대응 |
|---|---|---|
| Rollout 다양성 0 | K개 rollout이 다 같은 답 | latent에 noise injection 강제 |
| Energy 무의미 | 정답률 ↔ energy 상관 0 | better contrastive, temperature 조절 |
| 추론 비용 K배 | K=8이면 8배 느림 | K개 rollout 병렬화 + early-stop (압도적 다수결이면 early exit) |

### 성공 기준
- K=8 rollout 시 K=1 대비 **+8~12% pass@1**
- Energy ↔ 정답률 **Pearson > 0.6** (calibrated confidence)
- 추론 시간 K=8 / K=1 < **3배** (병렬화 효과)

### 만약 실패하면
Energy Head 가 무의미하면 → **다수결** 로 후퇴 (잠재 → 토큰 → 다수결).
이건 그냥 Self-Consistency. 평범하지만 효과는 확실. 우리 차별점은
약해지지만 작동은 함.

---

## Phase 6 — 회상 (Self-Distillation in Latent Space)

**코드네임**: 회상 (Recall)
**기간**: M11 ~ M12 (4주)
**선행조건**: Phase 5
**위험도**: 🔥🔥

### 왜 함
인간은 처음엔 곱셈을 "9 + 9 + 9 + … (5번)" 으로 계산하다가, 반복 학습 후엔
"9 × 5 = 45" 를 *직관*으로 안다. **압축된 사고**.

모델도 같다. 처음엔 어려운 문제에 K=16 step 필요. 같은 패턴 수만 번 보면
K=4 step 으로 같은 결과 나와야 한다. 이게 안 되면 모델은 *암기*만 하고
*추상화*는 못 한 거.

### 만드는 거
같은 모델의 *두 모드*:
- **Teacher mode**: `K_max=16`, 시간 충분히 줌
- **Student mode**: `K_max=4`, 빠르게

학습:
- 동일 입력에 teacher / student 동시 forward
- Student 의 출력 ↔ Teacher 의 출력: KL distill
- Student 의 workspace 최종상태 ↔ Teacher 의 workspace 마지막 4 step 평균: MSE
- Student 자동으로 K를 늘릴 권한 있음 (Halt Critic이 결정) — 압축 못하는
  문제는 자동으로 K 키움

### 리스크
| 위험 | 증상 | 대응 |
|---|---|---|
| Student가 teacher 못 따라옴 | 정확도 -10% | distill 가중치 조절, K 천천히 줄임 |
| Student가 항상 K_max 까지 감 | 압축 실패 | step 비용 c 조절 |

### 성공 기준
- 쉬운 문제: student (K=4) ≈ teacher (K=16) 정확도
- 어려운 문제: student 가 자동으로 K=12+ 까지 가서 풂 (HC 작동)
- 평균 inference cost teacher 대비 **40~60%** 감소
- 정확도 손실 **-2% 이내**

### 만약 실패하면
Student 가 teacher 못 따라오면 → 그냥 teacher 만 배포 (압축 포기). 추론
비용은 비싸지만 정확도는 유지.

---

## Phase 7 — 데뷔 무대 (Benchmarks + Productionization)

**코드네임**: 데뷔 (The Stage)
**기간**: M12 (4주)
**선행조건**: Phase 1~6 중 살아남은 것들
**위험도**: 🔥

### 측정 대상
**추론 벤치**:
- GSM8K (초등 수학)
- MATH (고등 / 경시 수학)
- BBH (BIG-Bench Hard, 23개 어려운 태스크)
- ARC-AGI-1 / ARC-AGI-2 (시각 추상화)
- MMLU-Pro (대학 수준 지식 추론)
- HumanEval / MBPP (코딩)

**비교 대상**:
- 같은 base 모델의 vanilla inference
- 같은 base 모델의 CoT (긴 출력)
- 같은 base 모델의 Self-Consistency (다수결)
- 다른 reasoning 모델들 (DeepSeek-R1 류) — *parameter 수 매칭 시*

**통제 변인**:
- 같은 평균 FLOPs
- 같은 출력 토큰 수 (latent vs CoT)
- 같은 평균 wall-clock

### 산출물
- Eval 리포트 (정량 + 정성, 실패 케이스 포함)
- 검증된 모델 체크포인트 (HF Hub 업로드)
- 학습 코드 + Reproduction recipe
- 논문 ("진짜 잘 되면")
- 망신 ("진짜 안 되면")

### 목표 (회피 가능한 목표가 아니라 *측정 가능한* 목표):
1. **GSM8K**: base + Self-Consistency 대비 **+5% 절대** with *3배 적은 출력 토큰*
2. **MATH**: base + CoT 대비 **+10% 절대** with *5배 적은 출력 토큰*
3. **BBH**: base + CoT 대비 **+5% 절대 평균**, 특정 reasoning-heavy task 에서는 **+15%**
4. **ARC-AGI-1**: base + CoT 대비 **+8% 절대** (이건 우리 강점이 명확히 드러나야 할 곳)

위 4개 중 2개 이상 못 맞추면: **솔직히 인정** + 후속 연구로 패스.

---

## 매 Phase 끝나면 다음 의식

```
[ ] 새 컴포넌트 끄는 플래그 작동 확인
[ ] 끄면 점수 떨어지나 확인 (안 떨어지면 안 만든 거)
[ ] 작은 reasoning benchmark 점수 기록
[ ] 실패 케이스 5개 이상 수동 분석
[ ] 다음 Phase 전에 코드 리팩토링 (기술 부채 청산)
[ ] PR + 회고 노트 + 다음 Phase 결정
```

회고 노트는 **반드시 글로**. 머리속에 있는 건 한 달 뒤 잊는다.

---

## 변동 가능성 (현실주의)

이 항해도는 **양피지 위에 그린 지도**다. 1년이라는 시간 동안 풍랑이 분다.
가능한 시나리오:

- **Phase 1 (Workspace) 가 한 달 만에 깔끔히 됨** → 일정 1개월 여유
- **Phase 2 (Operators) 가 학습이 안 됨** → 2개월 추가, Phase 3 단축
- **Phase 4 (RL) 가 한 분기 통째로 잡아먹음** → Phase 5, 6 통합
- **중간에 OpenAI / Anthropic / DeepSeek 이 같은 거 발표** → 전략 재평가
  (포기 아님, 차별화로 재포지셔닝)

매 Phase 진입 직전에 **1주짜리 plan review** 한다. 항해도는 살아있다.

---

## 한 줄 정리

**우리는 양념을 더 치는 사람이 아니다. 우리는 잠재 공간에 작은 가상 머신을 짓는 사람이다.**

매 단계마다 자문하자:
> "이거 그냥 트랜스포머에 뭐 하나 더 붙인 거 아니야?"

답이 "응"이면 **지금 phase 다시 설계**.
답이 "아니"이면 **계속 간다**.

가자.
