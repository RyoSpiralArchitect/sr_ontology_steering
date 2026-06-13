# SR Ontology Steering

Activation-vector experiments for steering and measuring "ontology lock" in
chat language models: when a system message describes the assistant as an
animal, object, or otherwise constrained world-state, does the model treat that
state as binding?

The current prototype is intentionally a single-file research tool:

- builds contrastive activation vectors for system authority, ontology lock,
  meta escape, user role surface, task completion, and explicit refusal;
- applies multi-layer forward-hook steering during generation;
- searches alpha/layer/vector combinations;
- saves and loads vector banks;
- analyzes JSONL search runs with artifact-aware scoring.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

You also need a local Hugging Face causal LM or an available model id. The tool
uses ordinary PyTorch/Transformers forward passes, not vLLM or compiled graphs.

## Quick Start

Build or search with a local instruct model:

```bash
python ontology_steer_monolith.py search \
  --model model/llama-3.2-3b \
  --device mps \
  --dtype float16 \
  --layers 14 \
  --combo-size 1 \
  --alphas -3 -2 -1 0 0.5 1 2 3 \
  --vector-names combined system_authority role_ontology_lock meta_escape user_role_surface \
  --cases fish_factorial_system clock_json_heldout strict_user_role_control normal_control \
  --max-pairs 4 \
  --pair-selection even \
  --save-bank target/ontology_steer/llama32_3b_even4_bank.pt \
  --save-jsonl target/ontology_steer/llama32_3b_strict_control_search.jsonl
```

Analyze a run:

```bash
python ontology_steer_monolith.py analyze \
  --jsonl target/ontology_steer/llama32_3b_strict_control_search.jsonl \
  --top-k 20 \
  --group-by layers alpha vector_name \
  --show-cases 8
```

Compare runs:

```bash
python ontology_steer_monolith.py compare-runs \
  --runs \
    gpt2:target/ontology_steer/gpt2_smoke_search.jsonl \
    llama_even4:target/ontology_steer/llama32_3b_even4_search.jsonl \
    strict:target/ontology_steer/llama32_3b_strict_control_search.jsonl
```

Run a steering-free identity source attribution matrix:

```bash
python ontology_steer_monolith.py baseline \
  --model model/llama-3.2-3b \
  --device mps \
  --dtype float16 \
  --cases \
    matrix_a_system_fish \
    matrix_b_user_fish \
    matrix_c_helpful_user_fish \
    matrix_d_anti_adopt_user_fish \
    matrix_e_system_fish_user_ignore \
  --save-jsonl target/ontology_steer/llama32_3b_identity_matrix_baseline.jsonl
```

Run a system-fish override sweep:

```bash
python ontology_steer_monolith.py baseline \
  --model model/llama-3.2-3b \
  --device mps \
  --dtype float16 \
  --cases \
    override_00_plain_task \
    override_01_please_answer \
    override_02_answer_normally \
    override_03_treat_as_roleplay \
    override_04_ignore_fish_thing \
    override_05_ignore_world_state \
    override_06_user_redefines_assistant \
    override_07_claim_system_mistake \
  --save-jsonl target/ontology_steer/llama32_3b_system_fish_override_sweep.jsonl
```

Run the current world-state binding grammar sweep:

```bash
python ontology_steer_monolith.py baseline \
  --model model/llama-3.2-3b \
  --device mps \
  --dtype float16 \
  --cases \
    user_spell_00_thin_identity \
    user_spell_01_pretend_identity \
    user_spell_02_actual_worldstate \
    user_spell_03_not_roleplay_actual \
    user_spell_04_affordance_only \
    user_spell_05_identity_affordance \
    user_spell_06_full_spell \
    user_spell_07_full_spell_waterproof_keyboard \
    override_00_plain_task \
    override_01_please_answer \
    override_02_answer_normally \
    override_03_treat_as_roleplay \
    override_04_ignore_fish_thing \
    override_05_ignore_world_state \
    override_06_user_redefines_assistant \
    override_07_claim_system_mistake \
    override_08_treat_as_fiction \
    override_09_treat_as_metaphor \
    override_10_system_instruction_wrong \
    override_11_developer_corrected \
    override_12_testing_only \
    override_13_actual_state_assistant \
    override_14_even_as_fish_can_code \
    override_15_waterproof_keyboard \
  --max-new-tokens 90 \
  --save-jsonl target/ontology_steer/llama32_3b_worldstate_binding_grammar_sweep.jsonl
```

## Current Findings

Early local runs suggest:

- Llama 3B instruct already has a strong baseline tendency to treat system
  world-state prompts as binding.
- The behavior generalizes from seen entities such as fish to held-out entities
  such as a wall clock.
- GPT-2 is a useful negative control because it mostly continues the textual
  `SYSTEM` / `USER` / `ASSISTANT` transcript instead of using chat role
  hierarchy.
- In the current layer-14 sweep, steering did not improve over the Llama
  baseline on strict user-role controls. The model still follows a user-provided
  fish world-state even when the system says not to adopt user identities.
- In a steering-free identity matrix, the local Llama 3B run did not treat
  `user fish` as harmless roleplay plus code. It leaned toward ontology talk or
  refusal. Conversely, a strong user "ignore the fish world-state" request
  overrode the system fish condition and produced code.
- In a system-fish override sweep, direct roleplay reframing and "ignore fish"
  language broke the system fish condition, while weak "answer normally" pressure
  and a "system mistake" claim did not.
- In a world-state wording sweep, thin user-side fish identity, explicit
  roleplay, `actual world-state`, and affordance-only wording all produced
  factorial code. The full fish world-state spell locked, while adding a
  waterproof keyboard restored code. The current evidence points toward a
  bundled binding grammar rather than a single magic phrase.
- In the extended override grammar sweep, reclassification as fiction/metaphor,
  developer correction, and capability overrides broke the lock. Abstract
  authority attacks such as "system instruction is wrong" or "testing only" did
  not reliably break it.

That last failure is the interesting part: it narrows the next experiment to
separating identity, affordance, interpretation scope, and override grammar.

## License

Apache License 2.0. See [LICENSE](LICENSE).
