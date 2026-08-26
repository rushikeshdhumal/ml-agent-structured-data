# Model selection and cost, per agent

Which model each of the eight agents should run, argued from measured per-agent
token usage rather than from how important each agent sounds.

The short version: **the most expensive agent is the best protected one, and the
least protected agent is mid-priced.** A naive "big model for the hard-sounding
job" assignment gets this close to backwards.

## The measurement

MLflow run `e3b692960a8340eb8e7c5657c20f42ba` (`per-agent-usage`), 2026-08-25.

| | |
|---|---|
| Model | `nvidia/nemotron-3-super-120b-a12b` (NVIDIA NIM) |
| Dataset | `data/explain_smoke.csv` -- **500 rows x 5 columns** |
| Mode | `AUTO_APPROVE=1` (no human gates fired) |
| Wall clock | 8.5 min, all 8 agents and 13 tasks completed |
| Totals | 65,664 prompt + 38,645 completion = 104,309 tokens, 29 requests |

Reproduce with:

```bash
MODEL=nvidia/nemotron-3-super-120b-a12b AUTO_APPROVE=1 \
  uv run ds-crew --data data/explain_smoke.csv --target target --task classification
```

Per-agent attribution comes from [`usage_listener.py`](../src/ds_crew/usage_listener.py),
which buckets every `LLMCallCompletedEvent` by the `task_name` and `agent_role`
the event carries. **Per-task sums reconcile exactly against the crew-level
totals** (delta 0 on both prompt and completion), so no events were lost to the
event bus's threaded dispatch. That check is the thing that makes the numbers
below usable; a shortfall would mean dropped events, not cheaper execution.

## Azure pricing

`eastus2`, GlobalStandard, from Azure's public Retail Prices API. The service
name there is **`Foundry Models`**, not "Azure OpenAI", and meters use shorthand
(`5 pp inp Gl 1M Tokens`: `Gl`=Global, `inp`/`opt`=input/output, `cd`=cached).

| model | input /1M | output /1M | cached input /1M |
|---|---|---|---|
| gpt-5 | $2.50 | $20.00 | $0.25 |
| gpt-5-mini | $0.45 | $3.60 | $0.045 |
| gpt-5-nano | $0.05 | $0.40 | $0.005 |

**Output costs 8x input.** Every conclusion below follows from that ratio.

## Measured per-agent usage

| Agent | in | out | out% | retries | safety net | gpt-5 | mini | nano |
|---|---|---|---|---|---|---|---|---|
| `explainer` | 19,865 | 15,516 | **40.2%** | 1 | guardrail + gate | $0.3600 | $0.0648 | $0.0072 |
| `feature_engineer` | 8,691 | 7,236 | 18.7% | 1 | guardrail + gate | $0.1664 | $0.0300 | $0.0033 |
| `model_selector` | 7,449 | 5,181 | 13.4% | 1 | guardrail + gate | $0.1222 | $0.0220 | $0.0024 |
| `evaluator` | 13,365 | 4,255 | 11.0% | 0 | **none** on `evaluation_task` | $0.1185 | $0.0213 | $0.0024 |
| `cleaning_strategist` | 6,863 | 4,209 | 10.9% | 1 | guardrail + gate | $0.1013 | $0.0182 | $0.0020 |
| `ensembler` | 3,471 | 903 | 2.3% | 0 | none | $0.0267 | $0.0048 | $0.0005 |
| `hpo_tuner` | 2,656 | 702 | 1.8% | 0 | none | $0.0207 | $0.0037 | $0.0004 |
| `eda_analyst` | 3,304 | 643 | 1.7% | 0 | none | $0.0211 | $0.0038 | $0.0004 |
| **total** | 65,664 | 38,645 | 100% | 4 | | **$0.9371** | $0.1687 | $0.0187 |

### Safety-net map

Which tasks have a guardrail and/or a human gate, from `crew.py`:

| task | guardrail | human gate |
|---|---|---|
| `propose_cleaning_task` | yes | yes |
| `propose_feature_task` | yes | yes |
| `propose_metric_task` | yes | yes |
| `explanation_task` | yes | yes |
| `finalize_task` | yes | no |
| `eda_task`, `execute_*`, `set_metric_task`, `model_selection_task`, `hpo_task`, `ensembling_task`, **`evaluation_task`** | no | no |

All four retries in the measured run landed on guarded tasks; unguarded tasks
had zero. The guardrails caught every correction the model needed.

## The two findings that drive the assignment

**1. The `explainer` alone is 38% of the all-gpt-5 bill**, and it has the
strongest safety net in the pipeline (grounding guardrail plus human gate). It
is simultaneously the most expensive agent and the hardest one to get silently
wrong.

**2. `evaluation_task` has no guardrail and no human gate**, yet the evaluator's
job is flagging `leakage_suspicion`. The sign-off gate sits on `explanation_task`,
one step later, by design -- so a human does eventually see the numbers, but
nothing programmatic checks the evaluator's skepticism. A missed leakage flag is
the highest-consequence silent failure in the pipeline.

Spend follows exposure, not job title: pay for the agent nothing else is
checking, and economize where a guardrail and a human already are.

## Recommended assignment

| Model | Agents | Why |
|---|---|---|
| **gpt-5** | `evaluator` | The only agent whose critical task has no safety net. Nothing else catches a miss. |
| **gpt-5-mini** | `eda_analyst`, `cleaning_strategist`, `feature_engineer`, `model_selector`, `explainer` | Guardrailed and human-gated, or small enough that cost is irrelevant. |
| ~~**gpt-5-nano**~~ **gpt-5-mini** | `hpo_tuner`, `ensembler` | Analysis says nano: zero judgment by design, since HPO requests a trial count the tool caps anyway and the ensembler is explicitly forbidden from picking members, weights, or strategy. **Overridden by a platform constraint** (below). |

`eda_analyst` is unguarded but gets mini rather than nano deliberately: its
profile seeds every downstream decision and no guardrail catches a fabricated
statistic. At $0.0038/run the cost difference is not worth the risk.

### Platform constraint: nano cannot call tools on Foundry (2026-08-26)

Discovered while building the agents: the `gpt-5-nano` deployment offers **no
custom tools at all**, MCP included. A Foundry agent on nano cannot reach the
tool layer, and both of nano's assigned agents exist solely to call one tool
each. So the tier is unusable here regardless of the reasoning that selected it.

`hpo_tuner` and `ensembler` therefore run on **gpt-5-mini**. The cost of being
overruled is **$0.0076/run**, 2.9% of the recommended mix, or **$3.27 across the
whole $99**. Runs per $99 fall from 383 to 372.

Chasing it further is not worth it: `gpt-5.4-nano` would recover only $1.92 of
that $3.27 over the same budget, and would need its own deployment and its own
capability check.

The analysis stands as written. Where task judgment is genuinely absent, the
cheapest tier is correct reasoning; what changed is that the platform will not
sell it here. Recorded rather than quietly edited, because the same reasoning
applies again on any platform whose cheapest tier does support tool calling.

### Economics

| Config | $/run | Runs per $99 | vs all-gpt-5 |
|---|---|---|---|
| All gpt-5 | $0.9371 | 105 | -- |
| Recommended, as analysed | $0.2583 | 383 | -72% |
| **Recommended, as deployable** | **$0.2659** | **372** | -72% |
| Recommended, `explainer` on gpt-5 | $0.5610 | 176 | -40% |
| All gpt-5-mini | $0.1687 | 586 | -82% |

"As deployable" is the row that matters: it is the analysed mix with
`hpo_tuner` and `ensembler` moved off nano, which Foundry will not let call
tools. The two rows differ by $3.27 over the full budget.

## The one open decision

The `explainer` is $0.0648/run on mini against $0.3600 on gpt-5. **That single
choice is $29.52 per 100 runs**, and it is the largest lever in the whole table.

Its guardrail catches ungrounded claims. It cannot catch an explanation that is
merely bland, and the explanation report is the artifact a human actually reads
before approving a model. So this is worth an A/B on identical runs rather than
a guess in either direction. Start on mini; upgrade only if the difference in
output is visible.

By contrast, downgrading the three cheapest agents (`hpo_tuner`, `ensembler`,
`eda_analyst`) from gpt-5 to nano saves $0.0672/run, about 7% of the bill. Real,
but not where the decision lives.

## Caveats

- **Token counts come from nemotron, not gpt-5.** Absolute numbers will shift
  with a different model's verbosity. The *shape* -- explainer dominant, HPO and
  ensembler negligible -- is driven by what each task has to produce, so it
  should hold.
- **Run-to-run variance is roughly 20%.** This run used 104,309 tokens; an
  earlier identical-configuration run used 130,326. Treat every figure as +/-20%.
- **The dataset is 500 rows x 5 columns.** The EDA report, cleaning plan, and
  feature plan all scale with **column count**, so a realistic 30-50 column
  dataset costs substantially more -- roughly 3-5x on the plan-heavy stages.
- **`AUTO_APPROVE=1`, so no human-feedback loops fired.** Interactive runs re-run
  tasks after feedback and cost more than this.
- **Prompt caching barely helps here.** Because output is ~80% of the gpt-5 bill,
  even a 75% input cache hit moves 105 runs to about 115. Optimizing input is the
  wrong instinct; choosing which agents emit long output is the right one.
