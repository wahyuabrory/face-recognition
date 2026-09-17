# Repository instructions

These instructions apply to the whole repository.

## Boundaries

- Keep biometric images, identity mappings, credentials, models, and generated artifacts out of Git.
- Keep TensorFlow imports lazy. Configuration loading and scenario listing must work without importing TensorFlow.
- Preserve the 16-scenario matrix in `configs/scenarios.yaml` unless the requested change explicitly changes the experiment design.
- Keep `src/face_recognition` free of comments and docstrings. Prefer clear names and direct code.
- Use Python 3.12 and the locked `uv` environment. Do not change dependency bounds or the lockfile unless the task requires it.

## Changes

- Make the smallest complete change and preserve unrelated behavior.
- Validate data and paths at trust boundaries. Keep errors visible and specific.
- Use the existing tests and fixtures. Create test files or test-only helpers only when the user explicitly approves them.
- Update user-facing documentation when commands, configuration, outputs, or supported behavior change.

## Verification

For Python changes, run:

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest -q
```

Use a direct behavior check when the change affects CLI or runtime behavior. TensorFlow training requires an AVX-capable CPU, so report that hardware limit instead of claiming an unexecuted training check.
