# Python oracles

These scripts produce the golden vectors in `testdata/`. They are never used
at runtime: the Go runtime must reproduce their outputs identically.

```bash
cd tools/oracle
python -m venv .venv && .venv/bin/pip install -r requirements.txt
./download_models.sh                      # v2 models in testdata/models/v2 (pinned commit)
.venv/bin/python dump_ops.py ../../testdata/models/v2/*.tflite -v   # inspection
.venv/bin/python gen_model_parity.py      # testdata/parity/<model>.json
.venv/bin/python gen_frontend_golden.py   # testdata/frontend/<signal>.json
```

Any regeneration must be committed with the package versions pinned in
`requirements.txt`, and the reason for the regeneration in the commit message.
