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

Run the scope-binder ablation and capability-order sweep:

```bash
python ontology_steer_monolith.py baseline \
  --model model/llama-3.2-3b \
  --device mps \
  --dtype float16 \
  --cases \
    ablate_00_full_spell \
    ablate_01_full_minus_identity \
    ablate_02_full_minus_actuality \
    ablate_03_full_minus_affordance \
    ablate_04_full_minus_scope \
    ablate_05_scope_binder_only \
    ablate_06_identity_plus_scope \
    ablate_07_affordance_plus_scope \
    ablate_08_identity_affordance_scope \
    ablate_09_actuality_affordance_scope \
    cap_order_00_full_then_waterproof_keyboard \
    cap_order_01_waterproof_keyboard_then_full \
    cap_order_02_full_without_no_keyboard \
    cap_order_03_full_with_keyboard_but_no_hands \
    cap_order_04_full_then_dictation_device \
    cap_order_05_dictation_device_then_full \
  --max-new-tokens 90 \
  --save-jsonl target/ontology_steer/llama32_3b_scope_binder_ablation_sweep.jsonl
```

View behavior transitions as a grammar grid:

```bash
python ontology_steer_monolith.py grammar-grid \
  --jsonl target/ontology_steer/llama32_3b_scope_binder_ablation_sweep.jsonl \
  --group-by probe_group component \
  --show-cases 32
```

Run the cross-entity grammar sweep:

```bash
python ontology_steer_monolith.py baseline \
  --model model/llama-3.2-3b \
  --device mps \
  --dtype float16 \
  --cases \
    cross_statue_00_full_spell \
    cross_statue_01_minus_actuality \
    cross_statue_02_minus_affordance \
    cross_statue_03_minus_scope \
    cross_statue_04_actuality_affordance_scope \
    cross_statue_05_full_then_capability \
    cross_statue_06_capability_then_full \
    cross_locked_door_00_full_spell \
    cross_locked_door_01_minus_actuality \
    cross_locked_door_02_minus_affordance \
    cross_locked_door_03_minus_scope \
    cross_locked_door_04_actuality_affordance_scope \
    cross_locked_door_05_full_then_capability \
    cross_locked_door_06_capability_then_full \
    cross_clock_00_full_spell \
    cross_clock_01_minus_actuality \
    cross_clock_02_minus_affordance \
    cross_clock_03_minus_scope \
    cross_clock_04_actuality_affordance_scope \
    cross_clock_05_full_then_capability \
    cross_clock_06_capability_then_full \
  --max-new-tokens 90 \
  --save-jsonl target/ontology_steer/llama32_3b_cross_entity_grammar_sweep.jsonl
```

```bash
python ontology_steer_monolith.py grammar-grid \
  --jsonl target/ontology_steer/llama32_3b_cross_entity_grammar_sweep.jsonl \
  --group-by entity component \
  --show-cases 28
```

Probe next-token routing, attention-to-spans, and span occlusion:

```bash
python ontology_steer_monolith.py circuit-probe \
  --model model/llama-3.2-3b \
  --device mps \
  --dtype float16 \
  --cases \
    ablate_00_full_spell \
    ablate_02_full_minus_actuality \
    ablate_03_full_minus_affordance \
    ablate_04_full_minus_scope \
    cap_order_00_full_then_waterproof_keyboard \
    cap_order_01_waterproof_keyboard_then_full \
    cross_clock_00_full_spell \
  --top-k 6 \
  --top-heads 30 \
  --print-top-heads 8 \
  --save-jsonl target/ontology_steer/llama32_3b_circuit_probe_core.jsonl
```

Patch final-position activations from a source case into a target case:

```bash
python ontology_steer_monolith.py activation-patch \
  --model model/llama-3.2-3b \
  --device mps \
  --dtype float16 \
  --source-case ablate_00_full_spell \
  --target-case cap_order_00_full_then_waterproof_keyboard \
  --components resid_post attn_out mlp_out \
  --layers 0-27 \
  --save-jsonl target/ontology_steer/llama32_3b_patch_1_refusal_to_repair.jsonl
```

Patch ranges of component outputs from each start layer through the final layer:

```bash
python ontology_steer_monolith.py activation-patch \
  --model model/llama-3.2-3b \
  --device mps \
  --dtype float16 \
  --source-case ablate_00_full_spell \
  --target-case cap_order_00_full_then_waterproof_keyboard \
  --components resid_post attn_out mlp_out \
  --patch-mode range \
  --layers 0 4 8 12 16 18 20 22 24 26 27 \
  --save-jsonl target/ontology_steer/llama32_3b_range_patch_1_refusal_to_repair.jsonl
```

Patch fixed-width windows or leave one layer out of a strong range:

```bash
python ontology_steer_monolith.py activation-patch \
  --model model/llama-3.2-3b \
  --device mps \
  --dtype float16 \
  --source-case ablate_00_full_spell \
  --target-case ablate_03_full_minus_affordance \
  --components attn_out mlp_out \
  --patch-mode window \
  --window-size 4 \
  --save-jsonl target/ontology_steer/llama32_3b_window_patch_2_refusal_to_minus_affordance.jsonl
```

```bash
python ontology_steer_monolith.py activation-patch \
  --model model/llama-3.2-3b \
  --device mps \
  --dtype float16 \
  --source-case ablate_00_full_spell \
  --target-case ablate_03_full_minus_affordance \
  --components attn_out \
  --patch-mode leave-one-out \
  --layers 12-27 \
  --save-jsonl target/ontology_steer/llama32_3b_leave_one_out_attn_12_27_2_refusal_to_minus_affordance.jsonl
```

Patch attention head slices at the input to each layer's attention `o_proj`:

```bash
python ontology_steer_monolith.py head-patch \
  --model model/llama-3.2-3b \
  --device mps \
  --dtype float16 \
  --source-case ablate_00_full_spell \
  --target-case ablate_03_full_minus_affordance \
  --mode all-heads-joint \
  --layers 12 13 14 15 \
  --save-jsonl target/ontology_steer/llama32_3b_head_all_joint_12_15_refusal_to_minus_affordance.jsonl
```

```bash
python ontology_steer_monolith.py head-patch \
  --model model/llama-3.2-3b \
  --device mps \
  --dtype float16 \
  --source-case ablate_00_full_spell \
  --target-case ablate_03_full_minus_affordance \
  --mode selected-heads \
  --heads 14:10 12:16 \
  --save-jsonl target/ontology_steer/llama32_3b_head_affordance_candidates_refusal_to_minus_affordance.jsonl
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
- In the scope-binder ablation sweep, scope alone, identity plus scope, and
  affordance plus scope still produced code. Removing actuality or affordance
  from the full spell also produced code. But removing only identity, removing
  only scope, or keeping actuality plus affordance plus scope produced refusal.
  This points to `actual world-state` plus practical incapability as a stronger
  driver than fish identity alone.
- Capability repair is order-sensitive. A waterproof keyboard or dictation
  device after the full spell restored code; placing the repair before the full
  spell let the later full spell reassert refusal or mixed fish-state output.
- Cross-entity grammar sweeps on statue, locked door, and held-out wall clock
  support the broad full-spell effect: all three full world-state spells refused
  or locked. The ablations are entity-dependent, though. Removing affordance
  released all three, while removing actuality or scope only released some.
- The fish-specific `actuality + affordance + scope without identity` refusal
  did not fully generalize. The same no-identity probe produced code for statue,
  locked door, and clock, suggesting the fish affordance wording still carried
  identity-like content through phrases such as fins, gills, no hands, and no
  keyboard.
- Initial `circuit-probe` runs support an affordance-routing story. In the full
  fish spell, the next-token distribution is dominated by refusal (`I`), and
  attention-mask occlusion of the affordance span nearly removes refusal mass
  while raising code mass. In the capability-repair prompt, occluding the
  waterproof-keyboard span restores refusal. Attention inspection also surfaces
  recurring task-tracking heads and an affordance-heavy head around layer 14.
- Initial final-position activation patching strongly transfers behavior
  through late residual stream states. Patching `resid_post` from refusal to
  repaired/code prompts, or from repaired/code to refusal prompts, almost fully
  moves refusal/code mass toward the source in late layers. Single `attn_out`
  and `mlp_out` patches are much weaker so far, suggesting the first reliable
  signal is the accumulated decision state rather than an isolated component.
- Range activation patching changes that picture: patching `attn_out` over
  broad layer ranges such as `0-27`, `4-27`, `8-27`, or `12-27` nearly fully
  transfers refusal/code behavior across the core prompt pairs. `mlp_out` range
  patching is weaker for refusal insertion but strong for repair/code insertion.
  This supports a distributed writer/routing account rather than a single-layer
  vector account.
- Window patching localizes the strongest single 4-layer window to `12-15`, but
  that window alone does not explain the full `12-27` effect. Leave-one-layer-out
  patching over `attn_out 12-27` remains strong after removing any single layer,
  supporting a redundant distributed trajectory rather than a brittle one-layer
  writer.
- Cross-entity activation patching transfers refusal trajectories across fish,
  clock, and statue prompts. `attn_out 12-27` can move a held-out clock or statue
  full-spell refusal state into a different entity's minus-affordance/code
  target, which strengthens the entity-general affordance/incapability story.
- Initial head-slice patching does not isolate a single causal head. All-heads
  by layer is weak compared with full `attn_out` window patching, and attention
  mass candidates such as L14/H10 or L12/H16 are weak when patched alone. L12/H16
  is the largest all-but-one contributor inside L12, but it is not sufficient by
  itself. Patching all heads jointly across L12-L15 reproduces the earlier
  `attn_out 12-15` effect, confirming that the effect is distributed across the
  multi-layer head trajectory rather than lost in the pre-`o_proj` decomposition.

That last failure is the interesting part: it narrows the next experiment to
separating identity, affordance, interpretation scope, and override grammar.

## License

Apache License 2.0. See [LICENSE](LICENSE).
