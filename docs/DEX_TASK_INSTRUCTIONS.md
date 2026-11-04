# Dex task instructions

The language side of the dex-hand VLA benchmark: one instruction list per subtask, one rule
for which episode carries which instruction, one rule for what an evaluation asks. Everything
derives from [`utils/dex_tasks.py`](../utils/dex_tasks.py); nothing else defines an
instruction.

If you are training or evaluating on these tasks, the two things to know:

1. **The rendered datasets already carry the labels.** Every dataset under
   `/mnt/ceph3/wym/dex/datasets` (`<stem>_rig_vla` for GR00T, `<stem>_rig_vla_v3.0` for
   PI0.5; `_fixcam_` for the 7 cabinet/office tasks) has one of its task's four instructions
   on each episode, 25 each per 100 episodes, identical across the two layouts. Train on them
   as they are; there is no prompt to choose.
2. **Evaluation draws the instruction per episode.** `--instruction_sampling registered` is
   the GR00T evaluator's default. Pass `canonical` only to reproduce a number measured before
   2026-09-18.

## Why this table exists

The released demonstration datasets carry no language field: `meta/tasks.parquet` maps the
env id to `task_index = 0` and nothing else. Every VLA needs a text condition anyway, so
whoever trains a policy has to supply one — and if each of us writes our own, three things
break, none of them loudly:

- **Checkpoints stop transferring.** A policy trained on `open the cabinet door` and
  evaluated on `CabinetPouringOpenCabinet-Sharpa-v1` (the env id, the usual fallback) loses accuracy and
  raises no error.
- **Numbers stop being comparable.** Two runs of the same task under different wordings are
  two different experiments.
- **Multi-task training becomes impossible.** Within a family the instruction is the *only*
  thing separating the subtasks — open the fridge from close it, aspirate from dispense.

So the instructions live here, one table, imported rather than retyped.

## The protocol

Each of the 30 subtasks carries four instructions, all describing the same success condition
in different words. `instructions[0]` is the **canonical** one: the string an exported policy
package is identified by.

| Stage | Which instruction |
| --- | --- |
| Rendering a dataset | one per **episode** — 100 episodes carry 25 of each of four |
| Training | whichever one the episode being trained on carries; no prompt is chosen |
| Evaluation | one drawn per episode from the same four |

### How an episode gets its label

`dataset_episode_instruction(env_id, episode_index)`. Each task has a fixed order of its
four instructions (a permutation derived from the task's `instruction_key` by SHA-256), and
episode *N* takes position `N mod 4`:

```
NutBoltTurnNut-Sharpa-v1
  ep 0 -> unscrew the nut counterclockwise on the bolt
  ep 1 -> twist the nut counterclockwise
  ep 2 -> turn the nut on the bolt counterclockwise
  ep 3 -> rotate the nut counterclockwise on the bolt
  ep 4 -> unscrew the nut counterclockwise on the bolt     (the cycle repeats)
  100 episodes: 25 / 25 / 25 / 25
```

It is a cycle, not an independent draw, for two reasons. A draw leaves the split to chance
(21/24/27/28 on the first task tried) and the rarest phrasing is the one a policy learns
worst. And a cycle depends on the episode index alone, so parallel render shards label
themselves and still agree once aggregated. The cycle also interleaves the four phrasings
evenly through the recording order.

Both the ordering and the evaluation draw are SHA-256-derived rather than seeded-RNG-derived:
the assignment is baked into datasets that several groups render independently, and it has
to come out the same on any machine, any Python, any year. `ASSIGNMENT_SHA256` in
[`tests/test_dex_task_instructions.py`](../tests/test_dex_task_instructions.py) pins the
entire 30 × 100 assignment so it cannot move by accident; it was verified identical on
CPython 3.9/macOS and 3.10/Linux.

Both hashes take the task's `instruction_key`, not its env id. The key is fixed when the
task's list is published and does not follow env id renames, so renaming an env changes no
dataset label and no evaluation draw; the keys in the table are the env ids the tasks carried
when the list was published.

### How an evaluation episode gets its instruction

`episode_instruction(env_id, environment_seed)`: SHA-256 of `(instruction_key, seed)` indexes
the list. Independent of the recording-side cycle by design — tying the test-time wording to the
training label would defeat the purpose. Over the standard 100-episode schedule all four
phrasings appear, at roughly 28/27/24/21 rather than exactly 25 each; the same seed always
gives the same instruction, so the run is reproducible.

## Where the label lives

| | tasks table | per episode | per frame |
| --- | --- | --- | --- |
| v3.0 (PI0.5) | `meta/tasks.parquet` — 4 rows, instruction strings as the pandas index | `meta/episodes/**/*.parquet` `tasks` | `data/**/*.parquet` `task_index` |
| v2.1 (GR00T) | `meta/tasks.jsonl` — 4 lines | `meta/episodes.jsonl` `tasks` | `data/**/episode_*.parquet` `task_index` |

GR00T reads the language from `annotation.human.task_description` (`task_index` mapped through
the tasks table); its episode loader was read directly and hands the processor four distinct
instructions across a task's episodes, each matching the table. PI0.5 reads the same strings
from the v3.0 layout of the same episodes.

## Relabelling a dataset, and checking one

A dataset rendered before this protocol labels every episode with one instruction. Nothing
about the recording is wrong — the videos, states and actions are what a re-render would
produce — and the labels live entirely in metadata plus one integer column, so they are
moved in place:

```bash
python utils/groot_utils/relabel_instructions.py --dataset <root> --env_id <ENV> --dry-run
python utils/groot_utils/relabel_instructions.py --dataset <root> --env_id <ENV>
python scripts/dex/ops/verify_labels.py       --dataset <root> --env_id <ENV>
```

Atomic per file, idempotent, both layouts. The verifier reads the three places the label lives
the way the trainers do and checks them against the table — run it on any dataset you did not
label yourself before training on it. Do not relabel while a training job is reading the
dataset. Relabelling does not change what an existing checkpoint knows; a policy trained on a
single-instruction dataset has to be retrained.

> The v3.0 tasks table keeps the instruction strings as the DataFrame's pandas index, which
> only survives through the metadata `DataFrame.to_parquet` writes. Written as a bare arrow
> table the columns look right but the index comes back as `0..3` and PI0.5 fails with
> `Cannot extract task strings from LeRobot dataset metadata`. The relabeller writes through
> pandas and the verifier reads through it; check with `pd.read_parquet`, not pyarrow.

## Evaluating GR00T

```bash
python train/train_groot.py --evaluate --model_path <checkpoint> --env_id <ENV_ID> ...
    --instruction_sampling registered     # default: one of the four per episode
    --instruction_sampling canonical      # every episode on the canonical instruction
    --prompt "<registered instruction>"   # optional: pin the canonical role to another phrasing
```

`train/train_groot.py` resolves the task's instructions from the table (a `--prompt` must be
one the task registers), draws one per episode from the episode seed, hands it to the policy
through `GrootAgent.set_instruction()`, and prints one `[episode]` line per episode with the
instruction it ran under, so a score can be split by phrasing afterwards. GR00T's language
input used to be `--exp_name`, i.e. whatever the run was named; it no longer is.

The OpenPI PI0.5 baseline validates dataset episode labels against
`dataset_episode_instruction(env_id, episode_index)` during preparation and
selects evaluation prompts with `episode_instruction(env_id, seed)`. Run it via
`train/train_pi05.py --evaluate`; the instruction table's source hash
is part of the run identity. See [the PI0.5 guide](PI05.md).

## Changing an instruction

- **Adding a phrasing** changes the cycle, so datasets labelled before the change no longer
  match what the table assigns today. Re-run the relabeller (cheap — metadata only) and bump
  `DEX_INSTRUCTIONS_VERSION`.
- **Renaming an env** changes nothing here: the label cycle and the evaluation draw hash the
  task's `instruction_key`, which stays as it is. Never rewrite a key.
- **Rewording `instructions[0]`** invalidates every exported package identified by it.
  `CANONICAL_INSTRUCTIONS` in the tests pins all 30, so a drive-by edit fails the test rather
  than silently breaking a checkpoint.
- **Either way the policy has to be retrained** for the change to reach it.

Known wording to fix at the next retrain: `PipetteTransferDispenseLiquid-Sharpa-v1` says *into
the well*, but the scene dispenses into the test tube standing in the rack
(`tasks/pipette_transfer/pipette_transfer_dispense_liquid.py`). The canonical string is kept until then; its paraphrases name
the tube.
