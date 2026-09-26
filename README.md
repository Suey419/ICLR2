# Look Before You Leap: Planning with Risk and Cost Awareness for a Clinical Diagnosis Agent

SPARC is a Sequential Planning with Anticipated Risk and Cost agent for
clinical diagnosis under partial observability.

Clinical diagnosis is a sequence of consequential decisions: every test
reveals information, consumes resources, and changes what should be done next.
Rather than selecting a test only from its immediate information gain, SPARC
looks ahead before acting. It predicts plausible outcomes for candidate tests,
anticipates downstream diagnostic trajectories, and uses stochastic AND-OR
search to jointly consider information gain, cumulative testing cost,
diagnostic error, and the risk of missing a time-critical condition (TCC).

SPARC follows a receding-horizon policy. It executes only the first action of
the selected plan, incorporates the result returned by the execution
environment, updates its diagnostic state, and replans from the new state.
This makes the implementation suitable for studying diagnostic-pathway
planning rather than isolated one-step test selection.

## FISC dataset

FISC is the Fast healthcare interoperability resources-based Interactive
Sequential Clinical case dataset. It is derived from 156 longitudinal
real-world inpatient records obtained from the Xiamen Medical Big Data Center.
The cases are re-encoded as HL7 FHIR JSON files and include:

- structured patient information;
- reference diagnoses used only for evaluation;
- recorded examination results;
- standardized diagnostic actions;
- LOINC-based examination identifiers;
- examination prices; and
- predefined time-critical conditions.

The 156 cases span ten clinical domains. Diagnostic actions are mapped to FHIR
resources so that every compared method uses the same validated action space.
The case data and institutional records are not included in this source
package.

## SPARC method

SPARC separates sequential diagnosis into three code-level components:

- **State** updates the visible diagnostic state using the initial patient
  information and returned examination results.
- **Planner** (`RollingAndOrPlanner`) predicts counterfactual examination
  outcomes, evaluates possible downstream diagnostic states, and selects the
  next examination or the stopping action using anticipated downstream cost,
  diagnostic loss, and TCC risk. Its public decision method is
  `RollingAndOrPlanner.choose()`.
- **ResultEnvironment** executes the selected examination through
  `ResultEnvironment.execute()` and returns either the recorded result or a
  simulated result, depending on the configured environment mode.

At decision step \(t\), the planning state contains the visible clinical
history and the current diagnostic belief. The planner cannot access the hidden
case state, the gold diagnosis, unselected examination results, or the future
action sequence of another method.

The legal action space is a fixed FHIR-compatible examination catalog. Each
action has a unique LOINC code, display name, result schema, and predefined
cost. Previously executed examinations are removed from the candidate set.
The language model returns structured JSON containing candidate LOINC codes;
the controller validates every code against the catalog before execution.

The planning objective minimizes:

- cumulative examination cost;
- general diagnostic error; and
- additional loss from missing a time-critical condition.

Safe stopping is an explicit action. Active stopping requires the diagnostic
belief and expected TCC miss loss to satisfy the configured thresholds. A run
may also terminate because no unused examination remains or because the
maximum number of executed examinations is reached.

## Test-result environments

SPARC uses two execution-environment settings:

- **R: real-result environment** — returns the recorded result when the
  selected examination is present in the original patient record; otherwise
  returns `no result`.
- **R+S: real-plus-simulated environment** — returns the recorded result when
  available and generates a structured simulated result when the selected
  examination has no recorded result.

The simulator is invoked only after an examination has been selected. It is
not used during SPARC's counterfactual planning. The simulator receives only
the selected examination, its expected result schema, and permitted
de-identified hidden context. It must not receive the gold diagnosis,
unselected examination results, the future action sequence, or the planner's
predicted branches.

## Compared methods

The experimental setup compares SPARC with:

- **Greedy** — selects the test with the highest
  information-gain-per-cost score, using
  \(\mathrm{IG}(a)/\sqrt{\mathrm{Cost}(a)}\).
- **ACTMED** — performs uncertainty-driven active test selection. Test prices
  are retained for resource accounting but are not included in its selection
  utility.
- **Self-Depending** — performs direct sequential LLM decision making.
- **Primary-Info** — produces a diagnosis from the full visible case
  information in one pass.

All methods receive the same initial patient information, candidate test
catalog, test-cost table, and result environment. Reference diagnoses are
withheld during decision making and used only for evaluation. Every method
produces Top-1, Top-2, and Top-3 diagnostic predictions.

SPARC planning depth is evaluated at
\(D_{\mathrm{search}}\in\{1,2,3,4,5\}\). Depth one corresponds to
single-step action evaluation; larger depths allow the planner to consider
downstream examinations and possible result branches.

## Evaluation metrics

Diagnostic quality is measured using Top-k accuracy for
k in {1, 2, 3}. A case is correct at Top-k when its reference diagnosis
is covered by the first \(k\) predicted diagnoses.

Safety is measured using the Top-k TCC miss rate among cases with a
reference TCC. A TCC is missed when the reference TCC is not covered by the
first \(k\) predictions.

Resource use is measured by:

- mean cumulative testing cost in CNY; and
- mean number of selected examinations per case.

Accuracy and TCC miss-rate estimates use 95% Wilson score confidence
intervals. Reported statistics are aggregated at the dataset level over the
the 156 cases in the FISC dataset. The current codebase contains one dataset;
the ten clinical domains are groups within FISC rather than separate datasets.

## Getting Started

### 1) Environment

Create and activate the Conda environment:

```bash
conda env create -f sparc_environment.yml
conda activate sparc
```

The source code targets Python 3.11 or newer. The environment specification
contains the core numerical, tabular-data, OpenAI-compatible client, and
environment-loading dependencies.

### 2) Data layout

When running the command-line tools, use the following layout:

```text
SPARC/
    .env
    data/
        FISC/
            <case files or case groups>/
    runs/
    src/
        baseline/
        catalog/
        config/
        core/
        data_ingestion/
        diagnosis/
        execution_environment/
        planning/
        prompts/
        providers/
        retrieval/
```

Each input case should be a FHIR JSON bundle accepted by
`data_ingestion.fhir_bundle`.

### 3) Environment variables

The model providers read credentials, model names, and API endpoints from
`.env` or the shell environment. SPARC sends requests through an
OpenAI-compatible Chat Completions interface, so it can use OpenAI models
directly and can use Claude, DeepSeek, Qwen, or other models through a
compatible provider or gateway.

```bash
# Diagnosis and planning
DOCTOR_API_KEY="your-provider-api-key"
DOCTOR_MODEL="gpt-6-astra"
DOCTOR_API_KEY_BASE_URL="https://api.openai.com/v1"

# Test-result simulation
PROVIDERS_API_KEY="your-provider-api-key"
PROVIDERS_MODEL="gpt-6-astra"
PROVIDERS_API_KEY_BASE_URL="https://api.openai.com/v1"

# Optional ablation experiments
ABLATION_API_KEY="your-provider-api-key"
ABLATION_MODEL="gpt-4o"
ABLATION_BASE_URL="https://api.openai.com/v1"

# Request timeout in seconds
LLM_TIMEOUT_SECONDS="120"
```

Examples of model configurations are:

```bash
# OpenAI
DOCTOR_MODEL="gpt-4o"
DOCTOR_API_KEY_BASE_URL="https://api.openai.com/v1"

# Claude through an OpenAI-compatible gateway
DOCTOR_MODEL="claude-3-7-sonnet-latest"
DOCTOR_API_KEY_BASE_URL="https://your-compatible-gateway.example/v1"

# DeepSeek through an OpenAI-compatible endpoint
DOCTOR_MODEL="deepseek-chat"
DOCTOR_API_KEY_BASE_URL="https://api.deepseek.com/v1"

# Qwen through an OpenAI-compatible endpoint
DOCTOR_MODEL="qwen-plus"
DOCTOR_API_KEY_BASE_URL="https://your-compatible-qwen-endpoint.example/v1"
```



## Running the implemented tools

Run commands from `src/` so that the project modules are importable.

### Self-Depending

```bash
cd src
python -m baseline.self_depending.run_self_depending \
  /path/to/case.json \
  --data-root ../data/FISC \
  --run-root ../runs/self_depending \
  --mode mixed
```

### Primary-Info

```bash
python -m baseline.primary_info.run_primary_info \
  /path/to/case.json \
  --data-root ../data/FISC \
  --run-root ../runs/primary_info
```

### FISC baselines

```bash
python baseline/actmed/src/runFISC_cost_aware.py \
  --case /path/to/case.json \
  --data-root ../data/FISC \
  --run-root ../runs/fisc_actmed
```

Other FISC policy entry points are:

- `baseline/actmed/src/runFISC_cheapest.py`;
- `baseline/greedy/runFISC_greedy.py`; and
- `baseline/evaluate_fisc_top3.py`.

### Build the FHIR examination catalog

```bash
python build_catalog.py \
  --data-root ../data/FISC \
  --output ../runs/fisc_catalog.json
```

The catalog contains normalized examination actions, display names, LOINC
codes, result schemas, and aliases used by retrieval and candidate selection.

### Simulate a held-out examination

```bash
python run_simulation_examples.py \
  --data-root ../data/FISC/Group\ 8 \
  --count 3 \
  --target-action "Complete blood count"
```

This command masks a recorded target examination, constructs a reviewed
de-identified context, and asks the execution simulator to generate a
structured result. The masked real result can be printed for offline
comparison.

## Configuration and experiment artifacts

Configuration files are stored in `src/config/`:

- `default.json` — general planning defaults;
- `full_planning.json` — Full-Planning settings and frozen case labels; and
- `main_experiment_labels.json` — compact label schema for the main experiment.

The main planning parameters are:

- `max_steps` — maximum number of executed examinations in one trajectory;
- `search_depth` — maximum lookahead depth per decision;
- `candidate_action_count` or `max_actions` — maximum candidate tests retained;
- `branches_per_action` — number of counterfactual outcome branches;
- `belief_threshold` — minimum diagnostic belief for active stopping;
- `critical_loss_threshold` — maximum acceptable expected TCC miss loss;
- `diagnostic_error_cost` — general diagnostic error penalty; and
- `critical_extra_cost` — additional TCC miss penalty.

Run outputs are stored below `--run-root`. A case directory may contain:

- `checkpoint.json` — resumable state and completion status;
- `events.jsonl` — append-only step-level events;
- `audit/` or `interactions/` — model requests and validated responses; and
- result JSON files containing selected actions, returned results, diagnostic
  predictions, termination status, and cumulative testing cost.

Reusing a `run_id` resumes its checkpoint. Use a new `run_id` for an
independent repetition.

## Repository structure

- `src/` — source code and command-line entry points.
- `src/baseline/` — SPARC comparison methods and experiment runners.
- `src/config/` — experiment settings and frozen labels.
- `src/prompts/` — English prompt templates for diagnosis, planning,
  execution, summarization, and final diagnosis.
- `src/data_ingestion/` — FHIR parsing and case construction.
- `src/catalog/` — FHIR examination-catalog construction.
- `src/planning/` — counterfactual AND-OR planning and stopping logic.
- `src/execution_environment/` — real and simulated result execution.
- `data/` — the single local FISC dataset, not included in this package.
- `runs/` — generated checkpoints, logs, audits, and result artifacts.
- `sparc_environment.yml` — Conda environment specification.
- `1.tex` — manuscript describing the method, evaluation protocol, and
  experimental findings.

## Limitations

The method is evaluated on retrospective cases and has not been prospectively
validated in clinical practice. Some unrecorded examination results in R+S are
simulated, and simulation errors may affect later decisions. Results also
depend on the examination catalog, cost estimates, diagnostic-belief quality,
and counterfactual result predictions.

