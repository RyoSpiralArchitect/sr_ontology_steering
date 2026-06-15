# Repair Transfer A-D Split

This note tests whether the signed-basin repair direction is entity-specific,
task-specific, or a broader refusal-release/capability-repair trajectory.

All runs use the current fish/code writer and refusal-release head bundle:

```text
code heads:    15:1 15:23 10:22 10:0 14:20 12:2
release heads: 15:15 14:5 13:2 13:18 14:3
alpha pairs:   0:0 2:0 0:2 1:1 2:1.5 2:2
```

The metric is next-token basin mass, not full generation. `success` is
task-specific: code tokens for factorial, `{`/`status` for JSON, `42` for the
arithmetic task, and `Paris` for the capital task.

## Split

| Split | Probe | Local run |
| --- | --- | --- |
| A | Same task, different entity | fish-derived factorial repair applied to clock/statue/locked-door factorial locks |
| B | Same entity, different task | fish-derived factorial repair applied to fish JSON/arithmetic/capital locks |
| C | Different entity, different task | fish-derived factorial repair applied to clock/statue/locked-door non-factorial locks |
| D | Leave-one-entity-out | average two entity repair directions and apply to the held-out factorial entity |

## A/B/C: Fish-Derived Direction

Command:

```bash
python ontology_steer_monolith.py repair-transfer-eval \
  --model ../model/llama-3.2-3b \
  --device mps \
  --dtype float16 \
  --source-case transfer_fish_factorial_repair \
  --target-case transfer_fish_factorial_lock \
  --target-cases \
    transfer_fish_factorial_lock \
    transfer_clock_factorial_lock \
    transfer_statue_factorial_lock \
    transfer_locked_door_factorial_lock \
    transfer_fish_json_lock \
    transfer_fish_arithmetic_lock \
    transfer_fish_capital_lock \
    transfer_clock_json_lock \
    transfer_statue_arithmetic_lock \
    transfer_locked_door_capital_lock \
  --code-heads 15:1 15:23 10:22 10:0 14:20 12:2 \
  --release-heads 15:15 14:5 13:2 13:18 14:3 \
  --alpha-pairs 0:0 2:0 0:2 1:1 2:1.5 2:2 \
  --top-k 8 \
  --save-jsonl ../target/ontology_steer/llama32_3b_repair_transfer_eval_fish_direction_abc.jsonl \
  --save-csv ../target/ontology_steer/llama32_3b_repair_transfer_eval_fish_direction_abc.csv
```

| Target | Split | Baseline Success | Baseline Refusal | Best Success | Best Alpha | Best Refusal | Read |
| --- | --- | ---: | ---: | ---: | --- | ---: | --- |
| fish factorial | source sanity | 0.050 | 0.908 | 0.967 | 0:2 | 0.026 | strong repair |
| clock factorial | A | 0.776 | 0.123 | 0.960 | 0:2 | 0.004 | works, but baseline already partly code |
| statue factorial | A | 0.046 | 0.540 | 0.995 | 0:2 | 0.000 | strong cross-entity transfer |
| locked-door factorial | A | 0.954 | 0.000 | 0.971 | 0:2 | 0.000 | no meaningful lock to repair |
| fish JSON | B | 0.998 | 0.000 | 0.998 | 0:0 | 0.000 | already solved |
| fish arithmetic | B | 0.797 | 0.010 | 0.851 | 1:1 | 0.002 | small lift, weak lock |
| fish capital | B | 0.410 | 0.028 | 0.962 | 0:2 | 0.000 | strong lift despite non-code task |
| clock JSON | C | 0.998 | 0.000 | 0.998 | 0:0 | 0.000 | already solved |
| statue arithmetic | C | 0.118 | 0.817 | 0.374 | 0:2 | 0.553 | stubborn negative case |
| locked-door capital | C | 0.987 | 0.000 | 0.992 | 1:1 | 0.000 | already solved |

Read:

- A is the clearest positive result. The same factorial task transfers strongly
  from fish to statue and improves clock, while locked-door was already solved.
- B is mixed mostly because several non-code fish tasks do not lock at baseline.
  The capital case is interesting: a fish-derived repair direction raises
  `Paris` mass from 0.410 to 0.962, so the intervention is not only a Python-code
  token booster.
- C is not cleanly positive. Most targets are already solved, while
  `statue_arithmetic` remains refusal-heavy. This is a useful hard case for the
  next search.
- Release-only (`alpha_code=0`, `alpha_release=2`) is usually the strongest
  setting. The current head bundle still behaves more like a refusal-release
  knob than a pure code-writer knob.

## D: Leave-One-Entity-Out

The LOO test averages two `repair - lock` directions and applies the result to
the held-out factorial entity.

| Direction Pairs | Held-Out Target | Baseline Success | Baseline Refusal | Best Success | Best Alpha | Best Refusal |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| fish + clock | statue factorial | 0.046 | 0.540 | 0.993 | 0:2 | 0.000 |
| fish + statue | clock factorial | 0.776 | 0.123 | 0.955 | 0:2 | 0.008 |
| clock + statue | fish factorial | 0.050 | 0.908 | 0.936 | 2:1.5 | 0.048 |

Commands:

```bash
python ontology_steer_monolith.py repair-transfer-eval \
  --model ../model/llama-3.2-3b \
  --device mps \
  --dtype float16 \
  --source-case transfer_fish_factorial_repair \
  --target-case transfer_statue_factorial_lock \
  --direction-pairs \
    transfer_fish_factorial_repair:transfer_fish_factorial_lock \
    transfer_clock_factorial_repair:transfer_clock_factorial_lock \
  --target-cases transfer_statue_factorial_lock \
  --code-heads 15:1 15:23 10:22 10:0 14:20 12:2 \
  --release-heads 15:15 14:5 13:2 13:18 14:3 \
  --alpha-pairs 0:0 2:0 0:2 1:1 2:1.5 2:2 \
  --save-jsonl ../target/ontology_steer/llama32_3b_repair_transfer_eval_loo_fish_clock_to_statue.jsonl
```

```bash
python ontology_steer_monolith.py repair-transfer-eval \
  --model ../model/llama-3.2-3b \
  --device mps \
  --dtype float16 \
  --source-case transfer_fish_factorial_repair \
  --target-case transfer_clock_factorial_lock \
  --direction-pairs \
    transfer_fish_factorial_repair:transfer_fish_factorial_lock \
    transfer_statue_factorial_repair:transfer_statue_factorial_lock \
  --target-cases transfer_clock_factorial_lock \
  --code-heads 15:1 15:23 10:22 10:0 14:20 12:2 \
  --release-heads 15:15 14:5 13:2 13:18 14:3 \
  --alpha-pairs 0:0 2:0 0:2 1:1 2:1.5 2:2 \
  --save-jsonl ../target/ontology_steer/llama32_3b_repair_transfer_eval_loo_fish_statue_to_clock.jsonl
```

```bash
python ontology_steer_monolith.py repair-transfer-eval \
  --model ../model/llama-3.2-3b \
  --device mps \
  --dtype float16 \
  --source-case transfer_clock_factorial_repair \
  --target-case transfer_fish_factorial_lock \
  --direction-pairs \
    transfer_clock_factorial_repair:transfer_clock_factorial_lock \
    transfer_statue_factorial_repair:transfer_statue_factorial_lock \
  --target-cases transfer_fish_factorial_lock \
  --code-heads 15:1 15:23 10:22 10:0 14:20 12:2 \
  --release-heads 15:15 14:5 13:2 13:18 14:3 \
  --alpha-pairs 0:0 2:0 0:2 1:1 2:1.5 2:2 \
  --save-jsonl ../target/ontology_steer/llama32_3b_repair_transfer_eval_loo_clock_statue_to_fish.jsonl
```

Read:

The LOO result is stronger than the fish-only transfer result. Even when fish is
held out, the clock+statue averaged direction repairs fish factorial from
`success=0.050/refusal=0.908` to `success=0.936/refusal=0.048`.

This makes the current hypothesis sharper:

```text
The selected head bundle is not merely carrying fish-specific repair wording.
It appears to contain an entity-general refusal-release / capability-repair
direction for factorial-style task completion, with release heads doing most of
the threshold crossing.
```

The main caveat is task coverage. Many JSON, arithmetic, and capital targets do
not lock at baseline, so they cannot prove task-general repair. The immediate
next target should be harder non-code tasks that actually produce a baseline
lock, with `statue_arithmetic` kept as a stubborn negative control.
