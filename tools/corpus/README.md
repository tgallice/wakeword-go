# Synthetic validation corpus

Text to speech corpus used to measure the detector functionally (recall, false accepts,
near misses). It is synthetic: Piper voices, not real recordings. See the README results
section for what that does and does not prove.

```sh
cd tools/corpus
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python generate.py                 # corpus/ (downloads 7 Piper voices, ~450 MB, once)
.venv/bin/python select_fixtures.py          # testdata/audio/ (committed, under 2 MB)
cd ../..
go run ./cmd/wakeword-eval --corpus tools/corpus/corpus --models testdata/models/v2 --verbose
```

`generate.py` renders every voice at three speech rates and two noise scales, then
augments each clip in numpy: gain, leading and trailing silence, and for half of the clips
a background (white, pink or brown noise, or TTS babble) at 10 to 20 dB SNR. Sentences,
near misses and embedding phrases live in `sentences.py`; negatives are checked to contain
no whole wake word. The seed fixes the augmentation and the sentence order; Piper samples
its own noise internally, so two runs give the same corpus in distribution but not bit for
bit.

`wakeword-eval` feeds a 3.5 s near-silent preroll before every file, so that the ESPHome
warm-up (100 invocations, 3 s) never hides a detection, and counts a positive as detected
when an event falls between the start of the wake word and 1.5 s after its end. It sweeps
the manifest cutoff plus 0.5, 0.8, 0.9, 0.97 and 0.99, with and without the VAD model, and
with `--verbose` lists every speech false accept with the sentence being spoken. A full run
takes about five minutes on sixteen cores.
