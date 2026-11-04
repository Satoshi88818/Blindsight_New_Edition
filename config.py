"""Configuration: CLI parser, patient profiles and small time-step helpers."""

import argparse

REF_FPS = 20.0            # frame rate the per-frame CLI rates refer to
ADAPT_GAIN = 1.5          # strength of divisive adaptation
FATIGUE_RECOVERY = 0.1    # 1/s
EXTRACTORS = ["dog", "canny", "highpass", "luma"]

# The parser defaults (see build_parser) are the clear, idealised view. REALISTIC_BASE is the degraded
# "patient-like" baseline that every profile except clear_demo starts from.
REALISTIC_BASE = dict(
    implant_type='cortical', feature_extractor='dog', dropout_prob=0.10, radial_elongation=1.3,
    adaptation_rate=0.05, temporal_noise_std=0.25, multiplicative_noise_std=0.15,
    position_jitter_std=0.10, crosstalk_factor=0.10, facilitation_factor=0.05, fatigue_rate=0.01,
    failure_prob=0.05, varied_shapes=True, image_color=False, head_movement=True,
    microsaccade_freq=0.5, persistence_decay=0.8, brightness_gamma=0.6, base_phosphene_sigma=5.0,
    motion_blur_strength=5, filling_in_dilation_factor=1.5, peripheral_dropout_scale=2.0,
    crowding_factor=1.5)

# Profiles: (values to SET, factors to MULTIPLY the REALISTIC_BASE value by)
PROFILES = {
    "naive_public_demo": (dict(dropout_prob=0.4, radial_elongation=2.0, adaptation_rate=0.1,
                               brightness_gamma=0.4, color_enabled=False), {}),
    "argus_veteran": ({}, dict(dropout_prob=1.5, adaptation_rate=1.2, fatigue_rate=1.5,
                               failure_prob=2.0)),
    "argus_veteran_2015_2020": (dict(dropout_prob=0.275, radial_elongation=1.6, motion_blur_strength=7),
                                dict(adaptation_rate=1.5, temporal_noise_std=1.2)),
    "optimistic_next_gen_retinal": (dict(dropout_prob=0.05, radial_elongation=1.1, adaptation_rate=0.02),
                                    dict(fatigue_rate=0.5, temporal_noise_std=0.7)),
    "early_cortical_orion_like": (dict(max_ecc=20.0, radial_elongation=1.5),
                                  dict(temporal_noise_std=1.5, dropout_prob=1.2)),
    "training_progression": (dict(training_mode=True),
                             dict(dropout_prob=2.0, temporal_noise_std=1.5, fatigue_rate=1.5)),
    "orion_optimist": (dict(radial_elongation=1.1, max_ecc=15.0), dict(temporal_noise_std=0.8)),
    # Idealised, easy-to-read view = the parser defaults (no overrides). Not realistic.
    "clear_demo": ({}, {}),
}


# ============================ configuration ============================
def build_parser():
    B = argparse.BooleanOptionalAction
    p = argparse.ArgumentParser(description="Research-style cortical/retinal visual prosthesis simulator (v14)")
    a = p.add_argument
    a('--grid_size', type=int, default=32, help='Electrodes per side')
    a('--base_phosphene_sigma', type=float, default=7.0, help='Foveal phosphene sigma in OUTPUT PIXELS')
    a('--max_sigma_px', type=float, default=30.0, help='Hard cap on phosphene sigma in pixels')
    a('--spacing', type=int, default=24, help='Pixel spacing between electrodes in the output')
    a('--max_current', type=float, default=100.0, help='Max stimulation current (uA)')
    a('--threshold_current', type=float, default=10.0, help='Phosphene detection threshold (uA)')
    a('--saturation_current', type=float, default=80.0, help='Current where brightness saturates (uA)')
    a('--max_sigma_scale', type=float, default=2.0, help='Max sigma growth from current/eccentricity')
    a('--ecc_base_size', type=float, default=0.3, help='Phosphene size (deg) at eccentricity 0')
    a('--ecc_linear_factor', type=float, default=0.15, help='Size growth (deg/deg)')
    a('--implant_type', choices=['cortical', 'retinal'], default='retinal')
    a('--layout', choices=['electrode', 'percept'], default='electrode',
      help="'electrode': phosphenes drawn on the regular electrode grid (cortical sheet view). "
           "'percept': drawn where they appear in the visual field (dense fovea).")
    a('--use_webcam', '--webcam', action=B, default=True, help='Webcam input (--no-webcam for a static image)')
    a('--image_path', default='example_image.jpg')
    a('--feature_extractor', choices=EXTRACTORS, default='luma')
    a('--dropout_prob', type=float, default=0.0, help='Fraction of non-responsive electrodes (foveal value)')
    a('--temporal_noise_std', type=float, default=0.03, help='Additive scintillation (signal dependent)')
    a('--multiplicative_noise_std', type=float, default=0.03)
    a('--position_jitter_std', type=float, default=0.0, help='Per-frame position jitter (fraction of spacing)')
    a('--filling_in_dilation_factor', type=float, default=1.0)
    a('--filling_in_bright_thresh', type=float, default=0.7)
    a('--filling_in_dim_thresh', type=float, default=0.3)
    a('--crosstalk_factor', type=float, default=0.0)
    a('--facilitation_factor', type=float, default=0.0)
    a('--brightness_gamma', type=float, default=0.8)
    a('--save_gif', action='store_true', help='Render offline to a GIF (no window)')
    a('--gif_path', default='blindsight_simulation.gif')
    a('--gif_fps', type=int, default=20)
    a('--gif_duration', type=float, default=10.0)
    a('--gif_scale', type=float, default=0.75, help='Downscale factor for GIF frames')
    a('--head_movement', action=B, default=False)
    a('--head_amplitude', type=float, default=40.0, help='Head movement amplitude (px)')
    a('--head_freq', type=float, default=0.2)
    a('--enable_sonification', action=B, default=False)
    a('--show_vf_overlay', action=B, default=True)
    a('--radial_elongation', type=float, default=1.0)
    a('--persistence_decay', type=float, default=0.5, help='Per-frame (20 fps ref.) persistence')
    a('--adaptation_rate', type=float, default=0.0, help='Per-frame (20 fps ref.) adaptation rate')
    a('--motion_blur_strength', type=float, default=0)
    a('--center_surround_sigma_center', type=float, default=1.0)
    a('--center_surround_sigma_surround', type=float, default=3.0)
    a('--center_surround_weight', type=float, default=0.5)
    a('--crowding_factor', type=float, default=1.0)
    a('--peripheral_dropout_scale', type=float, default=1.0)
    a('--max_ecc', type=float, default=10.0, help='Visual-field radius covered (deg)')
    a('--phosphene_shape_var', type=float, default=0.5)
    a('--color_enabled', action=B, default=False, help='Legacy: brightness-coded hues')
    a('--image_color', action=B, default=True, help='Phosphenes take their colour from the camera image')
    a('--fatigue_rate', type=float, default=0.0, help='Fatigue gain per (spacing/s of motion)')
    a('--microsaccade_freq', type=float, default=0.0, help='Mean microsaccade rate (Hz)')
    a('--patient_profile', choices=list(PROFILES), default='clear_demo')
    a('--training_mode', action=B, default=False)
    a('--failure_prob', type=float, default=0.0, help='Electrode failure probability PER SECOND')
    a('--pseudo_depth', action=B, default=False, help='Experimental: motion-magnitude pseudo-depth')
    a('--varied_shapes', action=B, default=False, help='Some phosphenes are elongated/oriented')
    a('--seed', type=int, default=42)
    return p


def parse_config(argv=None):
    """Parse CLI args (or an explicit list). Profile values act as defaults; explicit flags win."""
    parser = build_parser()
    pre, _ = parser.parse_known_args(argv)
    set_d, mul_d = PROFILES[pre.patient_profile]
    new = {} if pre.patient_profile == 'clear_demo' else dict(REALISTIC_BASE)
    new.update(set_d)
    for k, f in mul_d.items():
        new[k] = REALISTIC_BASE[k] * f
    parser.set_defaults(**new)           # profile = defaults; explicit CLI flags still win
    return parser.parse_args(argv)


def per_frame(rate, dt):
    """Convert a per-frame rate (defined at REF_FPS) to the actual time step."""
    rate = min(max(rate, 0.0), 1.0)
    return 1.0 - (1.0 - rate) ** (dt * REF_FPS)
