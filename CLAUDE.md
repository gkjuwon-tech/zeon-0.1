# ZEON — Project Memory

> **Claude. 매 세션 시작할 때 이 문서부터 읽는다. 한 번도 빼먹지 않는다.**
> 다 읽고 본 작업 시작.

---

## 0. 한 줄 정체성

**ZEON 은 트랜스포머에 양념 치는 프로젝트가 아니다.**
**ZEON 은 트랜스포머를 *지식 저장소로 사용하는* 잠재 공간 가상 머신 (Latent VM) 을 만드는 프로젝트다.**

이거 헷갈리면 다음 작업 진입 금지. 다시 읽고 와라.

---

## 1. 절대 어기지 않는 정신 강령 (5조)

매 작업 시작 전 머릿속에 다시 적자.

1. **트랜스포머 양념 = 혁신 아님.**
   "여기에 attention 한 번 더 / gate 하나 더 / embedding 하나 더" 가
   해법으로 떠오르면 **그 작업 다시 설계.** Universal Transformer / PonderNet /
   Recurrent Depth Transformer 의 짬뽕은 이미 존재한다. 그 경계 안에 머무르면 패배.

2. **수치만 만지면 박살.**
   `hidden_size`, `num_layers`, `K_max`, `learning_rate`, `weight` 조정으로
   문제를 풀려는 충동은 **양념 모드 부활**. 구조를 바꿔야 진전.

3. **Ablation 가능하지 않으면 의미 없음.**
   각 컴포넌트는 끄고 켜는 flag 가 있어야 한다.
   끄면 점수가 무너져야 한다. 안 무너지면? **안 만든 거다.**

4. **벤치마크 없는 phase = 추측.**
   매 phase 끝엔 **최소 1개**의 reasoning benchmark 점수가 따라온다.
   숫자 없으면 의견. 숫자 있으면 사실. 의견 가지고 phase 종료 선언 금지.

5. **실패는 학습이라고 부른다.**
   Phase 3 가 안 되면 Phase 3' 만들거나, 폐기하고 Phase 4 로 간다.
   1년이라는 시간을 *자존심에* 쓰지 않는다. ROADMAP 의 "만약 실패하면" 섹션
   을 매 phase 끝에 다시 읽는다.

---

## 2. 매 작업 진입 전 셀프 체크 (3문항)

코드 한 줄 짜기 전에 자문:

1. **"이거 그냥 트랜스포머에 뭐 하나 더 붙인 거 아닌가?"**
   → 답이 yes 면 **지금 작업 다시 설계.** 코드 쓰지 마.

2. **"이거 끄면 점수가 떨어지나?"**
   → 답이 모르겠으면 **ablation flag 부터 박고 시작.**

3. **"이거 측정할 수 있나? 어떤 벤치마크의 어떤 metric 에서?"**
   → 답이 없으면 **벤치마크부터 정하고 시작.**

위 3문항 통과 못 한 작업은 시작 금지. 통과해야 코드 들어감.

---

## 3. 현재 위치 / 다음 위치

**현재**: Phase 0 (양념) 완료. baseline 코드 + 17 테스트 + CI.

**다음**: Phase 1 (Workspace Bank)
- 구조화된 작업 메모리 (16 슬롯) 박기
- Read/Write head 박기
- diversity loss, slot collapse 방지
- GSM8K small subset 에서 Phase 0 대비 +2~5% 검증

전체 7-phase 계획은 [`docs/ROADMAP.md`](docs/ROADMAP.md).
목표 시스템 설계도는 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

매 phase 끝나면:
- [ ] 새 컴포넌트 끄는 flag 작동 확인
- [ ] flag 끄면 점수 떨어지나 확인
- [ ] 작은 reasoning bench 점수 기록 (`docs/PHASE_NOTES/phase{N}_*.md`)
- [ ] 실패 케이스 5개 이상 수동 분석
- [ ] PR + 회고 노트 + 다음 phase 결정

---

## 4. 기술적 비협상 사항

매 변경에 대해 반드시 유지:

- **HF 호환성**: `AutoModelForCausalLM.from_pretrained()` 로 항상 로드 가능.
  깨지면 phase 실패로 간주.
- **테스트 그린**: `pytest tests/ -q` 항상 green. ruff 항상 clean.
  깨진 채로 commit 금지.
- **트랜스플랜트 가능**: 새 컴포넌트 추가 시 `zeon/transplant.py` 도 같이 업데이트.
  base 모델에서 옮길 수 없는 형태면 설계 다시.
- **추론 비용 ≤ base 모델 × 5배**. 넘으면 의미 없음 (그냥 큰 모델 쓰는 게 나음).
- **컨텍스트 길이 16k 가정**. 200만 컨텍스트 추구하지 않는다. 우리 가설은
  "짧은 컨텍스트 + 깊은 잠재 사고".

---

## 5. 문서 작성 의무

- 매 phase 종료 시 `docs/PHASE_NOTES/phase{N}_*.md` 회고 노트 작성.
  포함: 만든 거, 측정 점수, 실패 케이스 5개, 다음 phase 결정.
- 머리속에 있는 통찰은 한 달 뒤 잊는다. **반드시 글로.**
- 회고 노트 없이 다음 phase 진입 금지.

---

## 6. 톤 & 작업 스타일 (형이 좋아하는 거)

- 한국말 텐션 살려서 보고. 짧고 직설적.
- 이상한 모호한 추상화 ("최적화", "개선") 금지. 정량으로 말한다.
- 진행 상황은 짧게 자주. 완료 후 한 번에 X.
- 실패 / 막힘 / 의심 즉시 보고. 묻혀가지 마.

---

## 7. 문서/링크

- README: 프로젝트 외부 face. 짧고 강하게.
- `docs/ROADMAP.md`: 1년 phase 계획. 막히면 거기 "만약 실패하면" 다시 읽기.
- `docs/ARCHITECTURE.md`: 1년 후 목표 설계도. 매 phase 마칠 때 갱신.
- `docs/PHASE_NOTES/`: phase별 회고. 다음 phase 들어가기 전 반드시 작성.

---

## 8. 마지막 한 줄

> **양념 한 스푼 더 치는 사람이 되지 말자.**
> **잠재 공간에 가상 머신 짓는 사람이 되자.**

코드 짜기 전에 이 문장 한 번 더 읽자.

가자.
