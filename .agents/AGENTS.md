# Project Rules

Whenever you run a terminal command (especially running python scripts, tests, or installations), you MUST run it within the `astra-lm-env` conda environment.

You can do this by prepending the command with `conda run -n astra-lm-env`.
For example:
- `conda run -n astra-lm-env python script.py`
- `conda run -n astra-lm-env pip install ...`
