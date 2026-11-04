import numpy as np
import cv2
import pytest

from blindsight import Simulator, parse_config
from blindsight.features import extract_features


def synthetic_frame():
    img = np.full((480, 640, 3), 60, np.uint8)
    cv2.circle(img, (320, 240), 90, (255, 255, 255), -1)
    cv2.rectangle(img, (80, 80), (220, 200), (30, 160, 220), 6)
    return img


def test_clear_demo_is_default():
    cfg = parse_config([])
    assert cfg.patient_profile == 'clear_demo'
    assert cfg.implant_type == 'retinal' and cfg.feature_extractor == 'luma'
    assert cfg.image_color and cfg.dropout_prob == 0.0 and not cfg.head_movement


def test_realistic_profile_and_explicit_flag_wins():
    cfg = parse_config(['--patient_profile', 'argus_veteran'])
    assert cfg.implant_type == 'cortical' and cfg.dropout_prob == pytest.approx(0.15)
    cfg = parse_config(['--patient_profile', 'argus_veteran', '--dropout_prob', '0.2'])
    assert cfg.dropout_prob == 0.2


def test_dog_does_not_wrap_uint8():
    gray = np.full((320, 320), 200, np.uint8)
    gray[150:170, 150:170] = 20
    f = extract_features(gray, 'dog')
    assert f.dtype == np.float32 and 0.0 <= f.min() and f.max() <= 1.0
    assert f[:100, :100].max() < 0.05          # flat bright area must not light up


@pytest.mark.parametrize("argv", [
    [],
    ['--patient_profile', 'argus_veteran'],
    ['--patient_profile', 'early_cortical_orion_like', '--layout', 'percept'],
    ['--patient_profile', 'training_progression', '--feature_extractor', 'highpass'],
    ['--pseudo_depth', '--feature_extractor', 'canny'],
    ['--color_enabled', '--patient_profile', 'naive_public_demo'],
])
def test_pipeline_runs(argv):
    sim = Simulator(parse_config(argv))
    frame = synthetic_frame()
    for k in range(10):
        _, out = sim.process(frame.copy(), k / 20, 0.05)
    assert out.dtype == np.float32 and np.isfinite(out).all()
    assert 0.0 <= out.min() and out.max() <= 1.0
    assert out.shape[:2] == (cfg_size(argv),) * 2


def cfg_size(argv):
    c = parse_config(argv)
    return c.grid_size * c.spacing


def test_deterministic_for_seed():
    frame = synthetic_frame()
    outs = []
    for _ in range(2):
        sim = Simulator(parse_config(['--patient_profile', 'argus_veteran']))
        for k in range(5):
            _, o = sim.process(frame.copy(), k / 20, 0.05)
        outs.append(o)
    assert np.array_equal(outs[0], outs[1])


def test_image_does_not_fade_to_black_with_motion():
    sim = Simulator(parse_config(['--patient_profile', 'argus_veteran']))
    frame = synthetic_frame()
    means = []
    for k in range(400):
        _, o = sim.process(frame.copy(), k / 20, 0.05)
        means.append(o.mean())
    assert sim.fatigue < 0.5
    assert np.mean(means[-50:]) > 0.3 * np.mean(means[20:70])
