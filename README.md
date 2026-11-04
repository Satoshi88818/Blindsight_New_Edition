# Blindsight - visual-prosthesis phosphene simulator

Simulates what a cortical or retinal prosthesis user might see: a camera image is turned into
features, sampled at electrode locations, passed through temporal dynamics (adaptation, persistence,
noise, fatigue, dead electrodes) and drawn as Gaussian phosphenes.

> A demonstration / research-prototyping tool. Parameters are plausible, not clinically validated.
> For validated models see `pulse2percept` (retinal) and `dynaphos` (cortical).

## Install & run

```bash
pip install -r requirements.txt
python -m blindsight                                   # webcam, default (clear, coloured) view
python -m blindsight --no-webcam --image_path photo.jpg
python -m blindsight --no-webcam --image_path photo.jpg --save_gif --gif_duration 5
python -m blindsight --patient_profile argus_veteran   # realistic, degraded view
python -m blindsight --layout percept --implant_type cortical --feature_extractor dog
```

Live keys: `s` saccade, `r` reset shift, `q`/`w` previous/next extractor, `ESC` quit.
Headless machines: use `--save_gif` (no window needed).

## Defaults and profiles

Built-in defaults are the **clear** view: retinal grid, filled `luma` brightness, colour from the
image, no dropout / noise / fatigue / head shake. Other profiles start from a realistic baseline
(cortical, `dog` edges, dropout, noise, fatigue, head sway) and modify it:

`naive_public_demo`, `argus_veteran`, `argus_veteran_2015_2020`, `optimistic_next_gen_retinal`,
`early_cortical_orion_like`, `training_progression`, `orion_optimist`, `clear_demo` (default).

Profiles only set defaults - any explicit flag wins (`--patient_profile argus_veteran --dropout_prob 0.2`).
Rates (`adaptation_rate`, `persistence_decay`, ...) refer to a 20 fps reference and are frame-rate
independent; `failure_prob` is per second.

## Layout

```
blindsight/
  config.py        CLI parser, profiles, per_frame helper
  layout.py        electrode grid -> visual field (inverse log map), image sampling
  features.py      dog / canny / highpass / luma, grid centre-surround, chroma, pseudo-depth
  motion.py        head sway, microsaccades, saccade offset, motion blur
  dynamics.py      adaptation, persistence, noise, dropout mask, fatigue, failures, training
  phosphenes.py    current -> brightness, shapes, cached Gaussian kernels, splatting
  sonification.py  optional audio (pygame)
  simulator.py     Simulator.process(frame, t, dt) -> (shifted frame, phosphene image)
  sources.py       webcam / static image
  display.py       side-by-side view, live loop, GIF export
  cli.py           main()
tests/             pytest smoke + regression tests
```

## Library use

```python
import cv2
from blindsight import Simulator, parse_config

sim = Simulator(parse_config(["--patient_profile", "argus_veteran"]))
frame = cv2.imread("photo.jpg")                       # BGR
shifted, phosphenes = sim.process(frame, t=0.0, dt=0.05)   # float32 in [0,1], (H,W) or (H,W,3)
```

## Tests

```bash
pip install pytest && pytest
```

## Modelling notes / limitations

- Cortical layout: regular electrode grid mapped to the visual field with the inverse of
  `ecc_cort = a*ln(ecc/e0 + 1)` (Chebyshev radius so the square grid fills the field of view).
  `--layout electrode` draws on the grid (cortical-sheet view; a round scene edge appears square);
  `--layout percept` draws at visual-field positions.
- Retinal mode is a uniform grid with no axon-streak model.
- Image colour is a demonstration feature; real prostheses give very limited colour.
- `--pseudo_depth` uses motion magnitude and is not real depth.
- 
