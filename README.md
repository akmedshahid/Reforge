# Advanced Hugging Face Downloader

This project provides an interactive and scriptable downloader for complete Hugging Face repositories (models/datasets/spaces), preserving folder layout.

## What it does

- Accepts Hugging Face links like:
  - `https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/tree/main`
  - `Qwen/Qwen2.5-VL-7B-Instruct`
- Organizes downloads under:
  - `<your_base_folder>/ai_models/<org__repo>/<revision>/...`
- Preserves remote directory/file structure exactly.
- Supports:
  - retry logic
  - resumable downloads
  - include/exclude patterns
  - configurable parallel workers
  - dry-run mode (list files only)
- Generates a `download_manifest.json` per download.

## Quick start

```bash
python -m pip install -r requirements.txt
python hf_model_downloader.py
```

On Windows, you can also double-click:

- `run_hf_downloader.bat`

## Command examples

Download everything (prompt for confirmation):

```bash
python hf_model_downloader.py https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/tree/main --output D:/Downloads
```

Download without prompt:

```bash
python hf_model_downloader.py Qwen/Qwen2.5-VL-7B-Instruct --output D:/Downloads --yes
```

Dry run only:

```bash
python hf_model_downloader.py Qwen/Qwen2.5-VL-7B-Instruct --dry-run --yes
```

Include/exclude patterns:

```bash
python hf_model_downloader.py Qwen/Qwen2.5-VL-7B-Instruct \
  --allow "*.json,*.safetensors" \
  --ignore "*.bin" \
  --yes
```

## Useful options

- `--revision main` branch/tag/commit
- `--repo-type model|dataset|space`
- `--output <folder>` base folder (then `ai_models/` inside it)
- `--subdir ai_models` customize top model folder name
- `--token <hf_token>` private/gated access (or set `HF_TOKEN` env var)
- `--max-workers 8` parallel worker count
- `--retries 4` retry count
- `--dry-run` list files only

## Notes

- Large repositories (50GB+) can take significant time and disk space.
- For private/gated repos, log in or provide a token with access.

## Troubleshooting

### GitHub merge conflict in `hf_model_downloader.py`

If GitHub shows conflict markers like `<<<<<<<`, `=======`, `>>>>>>>`, keep only one import block at the top of the file:

```python
from typing import Any, Iterable

try:
    from huggingface_hub import HfApi, snapshot_download
except ModuleNotFoundError as import_error:
    HfApi = Any  # type: ignore[assignment]
    snapshot_download = None  # type: ignore[assignment]
    _IMPORT_ERROR = import_error
else:
    _IMPORT_ERROR = None
```

Then verify no conflict markers remain:

```bash
python - <<'PY'
from pathlib import Path
text = Path("hf_model_downloader.py").read_text(encoding="utf-8")
markers = ("<<<<<<<", "=======", ">>>>>>>")
print("CONFLICT MARKERS FOUND" if any(m in text for m in markers) else "No conflict markers")
PY
```
