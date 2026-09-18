# GridWise — Smart Campus Energy Optimization

LLM-assisted operator-note interpretation + deterministic guardrails + linear-program optimization.

This implementation follows the challenge's canonical pipeline:

```text
operator_notes
    ↓
language-capable generative model
    ↓
deterministic directive guardrails
    ↓
96-variable linear-program optimizer
    ↓
public hourly_plan formatting
    ↓
independent schedule validation
    ↓
exact API response
```

The Problem Statement is the canonical source for endpoint names, request/response fields, directive semantics, battery behavior, and optimization validity. The Participant Guide is canonical for deployment, reproducibility, security, and evaluation procedure.

## Project structure

```text
gridwise/
├── main.py
├── llm.py
├── validator.py
├── optimizer.py
├── models.py
├── requirements.txt
├── Dockerfile
├── README.md
├── .env
├── .env.example
├── .gitignore
└── tests/
    ├── test_core.py
    ├── test_llm.py
    └── test_public_samples.py
```

## Prerequisites

- Python 3.11
- At least one LLM API key for real `/optimize-energy` requests
- PowerShell on Windows
- Docker only for container verification/deployment

The optimizer itself is deterministic and uses SciPy/HiGHS. The LLM is used only for the required natural-language operator-note interpretation.

## Free development setup

You do **not** need a paid DeepSeek account to develop this project.

The recommended free setup is to create:

```text
GEMINI_API_KEY
GROQ_API_KEY
```

and let the application use:

```text
gemini → groq → openrouter → deepseek
```

in that order.

Google currently provides a Gemini API Free Tier for eligible models, and Google AI Studio can create API keys for it. New keys created in AI Studio are auth keys under Google's current 2026 key changes. Check your AI Studio quota dashboard for the exact limits assigned to your project/model. 

Groq currently has a Free tier as well. Its exact model/account rate limits are shown in the Groq Limits page; limits are measured in requests/minute, requests/day, and token quotas, and differ by model/account.

OpenRouter is included only as another fallback/development option. Its current Free plan has a platform limit of 50 requests/day, so it should not be your only provider for a repeatedly-called judge endpoint.

DeepSeek remains supported for teams that have DeepSeek API balance or granted credits.

Official provider pages:

- Google AI Studio: https://aistudio.google.com/
- Gemini API key guide: https://ai.google.dev/gemini-api/docs/api-key
- Gemini billing/free tier: https://ai.google.dev/gemini-api/docs/billing
- Groq: https://console.groq.com/
- Groq limits: https://console.groq.com/docs/rate-limits
- OpenRouter keys: https://openrouter.ai/keys
- OpenRouter pricing: https://openrouter.ai/pricing
- DeepSeek API docs: https://api-docs.deepseek.com/

## Environment configuration

Copy the provided `.env.example` if you need to recreate `.env`.

A minimal free setup is:

```text
LLM_PROVIDER_ORDER=gemini,groq,openrouter,deepseek
GEMINI_API_KEY=your_gemini_key
GROQ_API_KEY=your_groq_key
```

Leave other keys blank unless you have them.

Do not commit `.env`. The repository's `.gitignore` already excludes it.

You can verify that Python can load the key without printing the secret:

```powershell
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('Gemini key loaded:', bool(os.getenv('GEMINI_API_KEY'))); print('Groq key loaded:', bool(os.getenv('GROQ_API_KEY')));"
```

## Click-and-run in VS Code

1. Open the **`gridwise` folder** in VS Code.
2. Put at least one real provider key in `.env`.
3. Open `main.py`.
4. Click **Run Python File**.
5. Open `http://127.0.0.1:8000/docs`.

The service listens on port 8000 unless `PORT` is changed.

## PowerShell run

```powershell
python main.py
```

Or:

```powershell
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

## Health check

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Expected:

```json
{"status":"ok"}
```

## Automated tests

Run the local suite:

```powershell
python -m pytest -q
```

The suite covers:

- exact request validation and non-finite-number rejection
- all five actionable directive effects
- signed battery action and no simultaneous charge/discharge
- energy/state transition correctness
- end-of-day battery neutrality
- independent final-plan validation
- reported-total recalculation
- tamper detection
- directive guardrails
- provider configuration/fallback behavior
- malformed/truncated LLM response handling
- `/health`
- `/optimize-energy` integration with the LLM boundary mocked

The public-sample test is skipped unless you explicitly supply the organizer JSON path.

## Verify all 10 public sample cases

From PowerShell:

```powershell
$env:GRIDWISE_PUBLIC_CASES = "C:\path\to\BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json"
python -m pytest -q
```

The test reads the organizer file at runtime. It does not hard-code public note wording or reference schedules.

Equivalent optimal schedules are accepted by the challenge, so the test focuses on validity and documented optimal cost rather than byte-for-byte schedule equality.

## Manual API test

Start the service:

```powershell
python main.py
```

Then use `/docs` for the easiest manual request.

The challenge input has exactly:

- `scenario_id`
- `operator_notes` with 1–3 strings
- `hours` containing exactly hours 0–23
- `battery`

The successful response has:

- `scenario_id`
- `directive_interpretation`
- `hourly_plan`
- `total_grid_kwh`
- `total_cost_bdt`
- `peak_grid_kwh`
- `plan_summary`

## LLM architecture

`llm.py` does not contain phrase matching as the interpreter. It calls a language-capable generative model and asks it to return the required structured interpretation.

The provider layer supports:

```text
Gemini OpenAI-compatible endpoint
Groq OpenAI-compatible endpoint
OpenRouter OpenAI-compatible endpoint
DeepSeek OpenAI-compatible endpoint
```

All four return the same internal JSON envelope. Provider order is configurable through `LLM_PROVIDER_ORDER`.

A provider without a configured key is skipped. If a configured provider fails, returns malformed/truncated JSON, or its output fails the caller-supplied deterministic directive validator, the next configured provider is attempted.

The provider layer does **not** silently turn model failure into regex/keyword classification. The LLM remains directly in the interpretation path as required by the challenge.

## Deterministic guardrails

`validator.py` treats model output as untrusted data.

It verifies:

- exactly one interpretation per operator note
- note indices exactly match note order
- only the six supported directive types
- `no_op` uses `applies=false` and `structured_adjustment=null`
- every non-`no_op` directive uses `applies=true`
- hours are unique integers from 0 through 23 in ascending order
- finite numeric values
- `solar_reduction.factor` is in [0, 1]
- battery reserves are non-negative and do not exceed capacity
- grid caps are non-negative
- exact adjustment object shapes with no extra fields

A directive cannot change demand, tariff, or battery parameters unless the challenge explicitly permits that directive effect.

## Optimization

`optimizer.py` uses 96 LP variables:

```text
grid[h]
solar_used[h]
battery_action[h]
battery_energy[h]
```

where:

```text
battery_action > 0  → charge
battery_action < 0  → discharge
battery_action = 0  → idle
```

This single signed action variable prevents simultaneous charging and discharging by construction.

The objective is:

```text
total_cost_bdt = SUM(grid_kwh[h] * tariff_bdt_per_kwh[h])
```

subject to the challenge's physical and operator constraints, including effective solar availability, battery state bounds and rate limits, no export, directive limits, and final battery energy equal to initial energy.

## Independent schedule validation

After optimization, the solver result is converted into the challenge's public `hourly_plan`.

That public plan is then independently replayed by `validator.py` before the API returns it.

The validator recalculates:

```text
total_grid_kwh
total_cost_bdt
peak_grid_kwh
```

from the public plan itself.

This deliberately matches the challenge's source-of-truth model: the final schedule is what gets checked, not an optimizer-side claim about what the schedule means.

## Failure behavior

Malformed request JSON or structural request validation is handled by FastAPI/Pydantic.

LLM/provider failure, invalid model interpretation, and infeasible optimization are returned as controlled HTTP 500 errors without exposing provider details, secrets, prompts, or stack traces.

The service does not log API keys or raw prompts.

## Docker

Build:

```powershell
docker build -t gridwise-energy-optimizer .
```

Run:

```powershell
docker run --rm -p 8000:8000 --env-file .env gridwise-energy-optimizer
```

Then:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

The image contains no baked-in credentials.

## Team integration

If the teammate branch already contains equivalent Pydantic models, merge by preserving these public contract names and meanings:

```text
OptimizeRequest
DirectiveInterpretation
HourlyPlanEntry
OptimizeResponse
```

Do not maintain two competing request/response schemas.

## Security

Never commit API keys, `.env`, passwords, or tokens. The challenge also prohibits exposing secrets or raw prompts in API responses/logs and requires a reachable, reproducible deployment.

## Known limitation

The LLM interpretation is the only nondeterministic part of the system. The optimizer and final validation are deterministic. Free-provider quotas and availability are account/provider dependent, so the submitted deployment should have a provider key with enough quota for repeated hidden evaluation and, preferably, a second configured provider as fallback.
