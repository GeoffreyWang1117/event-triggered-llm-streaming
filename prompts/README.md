# LLM Prompts

Prompt templates used for the LLM diagnosis step (paper §7.6).

## Static templates (exported here)

- `system_prompt.txt` — the system role for the diagnostic LLM.
- `cmapss_domain_context.txt` — domain context injected for the CMAPSS prognostics task.
- `cicids_domain_context.txt` — domain context injected for the CIC-IDS2017 intrusion task.

These are verbatim exports of the string constants in
[`../src/models/llm/prompts.py`](../src/models/llm/prompts.py).

## Programmatically assembled parts

The **user prompt**, the **severity rubric**, and the **JSON output schema** are
built at call time by `PromptBuilder` in `src/models/llm/prompts.py` from the
per-window sensor features and anomaly state. The **grounding rubric** used to
score diagnoses (`{0, 0.25, 0.5, 0.75, 1.0}`) is specified in the supplementary
material (`../supplementary/supplementary.pdf`, Appendix B / Experimental Details).

To regenerate the exported templates after editing `prompts.py`:

```bash
python - <<'PY'
from src.models.llm.prompts import PromptBuilder
for name in ("SYSTEM_PROMPT", "CMAPSS_DOMAIN_CONTEXT", "CICIDS_DOMAIN_CONTEXT"):
    open(f"prompts/{name.lower()}.txt", "w").write(getattr(PromptBuilder, name).strip() + "\n")
PY
```
