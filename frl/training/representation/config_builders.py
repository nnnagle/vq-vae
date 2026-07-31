"""Config-dict builders for representation training.

These functions translate the parsed bindings / training YAML objects into the
plain dicts and loss-setup objects that ``main()`` and ``process_batch()``
consume. Extracted verbatim from ``main()`` — the ``x if a and b else default``
fallback logic and log strings are unchanged.

Heavy dependencies (``build_anchor_sampler``, ``EvtDiffusionMetric``) are imported
function-locally so this module imports without numpy/torch and the pure
dict-builders remain unit-testable.

Note on ``spatial_positive_max_dist``: ``build_spatial_pair_config`` guards on
``max_distance is not None`` (falling back to 8) while ``build_loss_config`` does
not (a present-but-None selection yields None). This is a pre-existing divergence
between the two derivations; both are preserved exactly rather than reconciled,
since real configs set the value and the two agree in practice.
"""

from __future__ import annotations

import logging


def build_spatial_pair_config(bindings_config, type_encoder_feature: str) -> dict:
    """Spatial-InfoNCE params for DataLoader-worker pair precomputation.

    Must match the spatial values in ``build_loss_config``; the fallback path in
    ``process_batch`` reproduces these when a sample has no worker-precomputed pairs.
    """
    _spc_spectral_cfg = bindings_config.get_loss('infonce_type_spectral')
    _spc_spatial_cfg = bindings_config.get_loss('infonce_type_spatial')
    _spc_sampling_cfg = bindings_config.get_sampling_strategy(
        _spc_spectral_cfg.anchor_population if _spc_spectral_cfg else 'grid-plus-supplement'
    )
    _spc_grid = _spc_sampling_cfg.grid if _spc_sampling_cfg and _spc_sampling_cfg.grid else None
    _spc_supp = _spc_sampling_cfg.supplement if _spc_sampling_cfg else None
    _spc_pos = (
        _spc_spatial_cfg.positive_strategy.selection
        if _spc_spatial_cfg and _spc_spatial_cfg.positive_strategy
        and _spc_spatial_cfg.positive_strategy.selection else None
    )
    _spc_neg = (
        _spc_spatial_cfg.negative_strategy.selection
        if _spc_spatial_cfg and _spc_spatial_cfg.negative_strategy
        and _spc_spatial_cfg.negative_strategy.selection else None
    )
    _spc_sw = (
        _spc_spatial_cfg.spectral_weighting
        if _spc_spatial_cfg and _spc_spatial_cfg.spectral_weighting else None
    )
    return {
        'type_encoder_feature': type_encoder_feature,
        'stride': _spc_grid.stride if _spc_grid else 16,
        'border': _spc_grid.exclude_border if _spc_grid else 16,
        'jitter_radius': _spc_grid.jitter.radius if _spc_grid and _spc_grid.jitter else 4,
        'supplement_n': _spc_supp.n if _spc_supp else 104,
        'spatial_positive_k': _spc_pos.k if _spc_pos else 4,
        'spatial_positive_max_dist': _spc_pos.max_distance if _spc_pos and _spc_pos.max_distance is not None else 8,
        'spatial_negative_min_dist': _spc_neg.min_distance if _spc_neg and _spc_neg.min_distance is not None else 96.0,
        'spatial_negative_max_dist': _spc_neg.max_distance if _spc_neg and _spc_neg.max_distance is not None else 192.0,
        'spatial_negatives_per_anchor': _spc_neg.n_per_anchor if _spc_neg and _spc_neg.n_per_anchor is not None else 16,
        'spatial_spectral_tau': _spc_sw.tau if _spc_sw else 200,
        'spatial_min_w': _spc_sw.min_weight if _spc_sw else 0.03,
    }


def build_loss_config(bindings_config, training_config, logger: logging.Logger) -> dict:
    """Assemble the loss/sampling/training ``loss_config`` dict from parsed YAML.

    Includes the optional variance-covariance (type) and phase-VCR sub-configs and
    emits the same startup log lines. EVT keys are added later by ``build_evt``.
    """
    spectral_loss_cfg = bindings_config.get_loss('infonce_type_spectral')
    spatial_loss_cfg = bindings_config.get_loss('infonce_type_spatial')
    sampling_cfg = bindings_config.get_sampling_strategy(
        spectral_loss_cfg.anchor_population if spectral_loss_cfg else 'grid-plus-supplement'
    )

    # Grid/sampling params from sampling strategy
    grid = sampling_cfg.grid if sampling_cfg and sampling_cfg.grid else None
    supplement = sampling_cfg.supplement if sampling_cfg else None

    loss_config = {
        # Sampling (from sampling-strategy config)
        'stride': grid.stride if grid else 16,
        'border': grid.exclude_border if grid else 16,
        'jitter_radius': grid.jitter.radius if grid and grid.jitter else 4,
        'supplement_n': supplement.n if supplement else 104,

        # Spectral InfoNCE loss (from losses.infonce_type_spectral config)
        'positive_k': (
            spectral_loss_cfg.positive_strategy.selection.k
            if spectral_loss_cfg and spectral_loss_cfg.positive_strategy
            and spectral_loss_cfg.positive_strategy.selection
            else 16
        ),
        'positive_min_spatial': (
            spectral_loss_cfg.positive_strategy.selection.min_distance
            if spectral_loss_cfg and spectral_loss_cfg.positive_strategy
            and spectral_loss_cfg.positive_strategy.selection
            else 4.0
        ),
        'negative_quantile_low': (
            spectral_loss_cfg.negative_strategy.selection.range[0]
            if spectral_loss_cfg and spectral_loss_cfg.negative_strategy
            and spectral_loss_cfg.negative_strategy.selection
            and spectral_loss_cfg.negative_strategy.selection.range
            else 0.5
        ),
        'negative_quantile_high': (
            spectral_loss_cfg.negative_strategy.selection.range[1]
            if spectral_loss_cfg and spectral_loss_cfg.negative_strategy
            and spectral_loss_cfg.negative_strategy.selection
            and spectral_loss_cfg.negative_strategy.selection.range
            else 0.75
        ),
        'negative_min_spatial': (
            spectral_loss_cfg.negative_strategy.selection.min_distance
            if spectral_loss_cfg and spectral_loss_cfg.negative_strategy
            and spectral_loss_cfg.negative_strategy.selection
            else 8.0
        ),
        'temperature': (
            spectral_loss_cfg.temperature
            if spectral_loss_cfg and spectral_loss_cfg.temperature is not None
            else 0.07
        ),

        # Spatial InfoNCE loss (from losses.infonce_type_spatial config)
        'spatial_positive_k': (
            spatial_loss_cfg.positive_strategy.selection.k
            if spatial_loss_cfg and spatial_loss_cfg.positive_strategy
            and spatial_loss_cfg.positive_strategy.selection
            else 4
        ),
        'spatial_positive_max_dist': (
            spatial_loss_cfg.positive_strategy.selection.max_distance
            if spatial_loss_cfg and spatial_loss_cfg.positive_strategy
            and spatial_loss_cfg.positive_strategy.selection
            else 8
        ),
        'spatial_negative_min_dist': (
            spatial_loss_cfg.negative_strategy.selection.min_distance
            if spatial_loss_cfg and spatial_loss_cfg.negative_strategy
            and spatial_loss_cfg.negative_strategy.selection
            and spatial_loss_cfg.negative_strategy.selection.min_distance is not None
            else 96.0
        ),
        'spatial_negative_max_dist': (
            spatial_loss_cfg.negative_strategy.selection.max_distance
            if spatial_loss_cfg and spatial_loss_cfg.negative_strategy
            and spatial_loss_cfg.negative_strategy.selection
            and spatial_loss_cfg.negative_strategy.selection.max_distance is not None
            else 192.0
        ),
        'spatial_negatives_per_anchor': (
            spatial_loss_cfg.negative_strategy.selection.n_per_anchor
            if spatial_loss_cfg and spatial_loss_cfg.negative_strategy
            and spatial_loss_cfg.negative_strategy.selection
            and spatial_loss_cfg.negative_strategy.selection.n_per_anchor is not None
            else 16
        ),
        'spatial_spectral_tau': (
            spatial_loss_cfg.spectral_weighting.tau
            if spatial_loss_cfg and spatial_loss_cfg.spectral_weighting
            else 200
        ),
        'spatial_min_w': (
            spatial_loss_cfg.spectral_weighting.min_weight
            if spatial_loss_cfg and spatial_loss_cfg.spectral_weighting
            else 0.03
        ),
        'spatial_temperature': (
            spatial_loss_cfg.temperature
            if spatial_loss_cfg and spatial_loss_cfg.temperature is not None
            else 0.07
        ),

        # Loss weights (from losses config)
        'spectral_loss_weight': spectral_loss_cfg.weight if spectral_loss_cfg else 1.0,
        'spatial_loss_weight': spatial_loss_cfg.weight if spatial_loss_cfg else 1.0,

        # Cross-batch spectral neg sampling: target this many negatives per anchor.
        # With N_total anchors, n_neg = spectral_neg_per_anchor * N_total pairs are sampled.
        'spectral_neg_per_anchor': 20,

        # Training (from training config)
        'gradient_clip_enabled': training_config.training.gradient_clip.enabled,
        'gradient_clip_max_norm': training_config.training.gradient_clip.max_norm,

        # Encoder input feature names (from model section of training config)
        'type_encoder_feature': training_config.model_input.type_encoder_feature,
        'phase_encoder_feature': training_config.model_input.phase_encoder_feature,

        # Chunked encoder forward: number of samples per sub-batch to bound peak GPU memory.
        # enc_chunk_size=1 matches original serial behaviour; larger values are faster
        # but use more GPU memory proportionally.
        'enc_chunk_size': training_config.hardware.enc_chunk_size,
    }

    # Variance-covariance regularization (optional)
    vcr_cfg = bindings_config.get_loss('variance_covariance_type')
    if vcr_cfg is not None:
        loss_config['vcr_enabled'] = True
        loss_config['vcr_weight'] = vcr_cfg.weight if vcr_cfg.weight is not None else 0.1
        loss_config['vcr_variance_weight'] = vcr_cfg.variance_weight if vcr_cfg.variance_weight is not None else 1.0
        loss_config['vcr_covariance_weight'] = vcr_cfg.covariance_weight if vcr_cfg.covariance_weight is not None else 1.0
        loss_config['vcr_variance_target'] = vcr_cfg.variance_target if vcr_cfg.variance_target is not None else 1.0
        logger.info(
            f"Variance-covariance loss enabled: weight={loss_config['vcr_weight']}, "
            f"var_w={loss_config['vcr_variance_weight']}, "
            f"cov_w={loss_config['vcr_covariance_weight']}, "
            f"var_target={loss_config['vcr_variance_target']}"
        )
    else:
        logger.info("Variance-covariance loss (type) disabled (not in config)")

    # Phase VCR (variance-covariance on z_phase)
    phase_vcr_cfg = bindings_config.get_loss('variance_covariance_phase')
    if phase_vcr_cfg is not None:
        loss_config['phase_vcr_config'] = {
            'weight': phase_vcr_cfg.weight if phase_vcr_cfg.weight is not None else 0.1,
            'variance_weight': phase_vcr_cfg.variance_weight if phase_vcr_cfg.variance_weight is not None else 1.0,
            'covariance_weight': phase_vcr_cfg.covariance_weight if phase_vcr_cfg.covariance_weight is not None else 1.0,
            'variance_target': phase_vcr_cfg.variance_target if phase_vcr_cfg.variance_target is not None else 1.0,
        }
        logger.info(
            f"Variance-covariance loss (phase) enabled: "
            f"weight={loss_config['phase_vcr_config']['weight']}, "
            f"var_w={loss_config['phase_vcr_config']['variance_weight']}, "
            f"cov_w={loss_config['phase_vcr_config']['covariance_weight']}, "
            f"var_target={loss_config['phase_vcr_config']['variance_target']}"
        )
    else:
        logger.info("Variance-covariance loss (phase) disabled (not in config)")

    logger.info(
        f"Loss config from bindings: "
        f"stride={loss_config['stride']}, border={loss_config['border']}, "
        f"supplement_n={loss_config['supplement_n']}, "
        f"spectral(k={loss_config['positive_k']}, min_spatial={loss_config['positive_min_spatial']}, "
        f"neg_q=[{loss_config['negative_quantile_low']}, {loss_config['negative_quantile_high']}], "
        f"neg_min_spatial={loss_config['negative_min_spatial']}, temp={loss_config['temperature']}), "
        f"spatial(neg_dist=[{loss_config['spatial_negative_min_dist']}, {loss_config['spatial_negative_max_dist']}], "
        f"neg_per_anchor={loss_config['spatial_negatives_per_anchor']}, "
        f"spec_tau={loss_config['spatial_spectral_tau']}, min_w={loss_config['spatial_min_w']}, "
        f"temp={loss_config['spatial_temperature']}), "
        f"weights(spectral={loss_config['spectral_loss_weight']}, spatial={loss_config['spatial_loss_weight']}), "
        f"cross_patch_neg_k={loss_config.get('cross_patch_negatives_per_anchor', 8)}"
    )

    return loss_config


def build_phase_config(bindings_config, logger: logging.Logger):
    """Build the phase-loss anchor sampler + phase_config dict.

    Returns (phase_sampler, phase_config), both None when soft_neighborhood_phase
    is not configured.
    """
    from data.sampling.anchor_sampling import build_anchor_sampler

    phase_loss_cfg = bindings_config.get_loss('soft_neighborhood_phase')
    phase_sampler = None
    phase_config = None

    if phase_loss_cfg is not None:
        # Build the ysfc-weighted anchor sampler
        phase_anchor_pop = (
            phase_loss_cfg.anchor_population
            if phase_loss_cfg.anchor_population
            else 'grid-plus-supplement-ysfc'
        )
        phase_sampler = build_anchor_sampler(bindings_config, phase_anchor_pop)

        # Extract pair construction + loss params from parsed config
        ps = phase_loss_cfg.pair_strategy
        pw = phase_loss_cfg.pair_weights
        cur = phase_loss_cfg.curriculum
        phase_config = {
            # Pair construction
            'k': ps.type_similarity.k if ps and ps.type_similarity else 16,
            'min_overlap': ps.ysfc_overlap.min_overlap if ps and ps.ysfc_overlap else 3,
            'min_pairs': ps.min_pairs if ps else 5,
            'include_self': ps.include_self if ps else True,
            'sigma': pw.sigma if pw else 5.0,
            'self_pair_weight': pw.self_pair_weight if pw else 1.0,
            # Loss
            'weight': phase_loss_cfg.weight if phase_loss_cfg.weight is not None else 1.0,
            'tau_ref': phase_loss_cfg.tau_ref if phase_loss_cfg.tau_ref is not None else 0.1,
            'tau_learned': phase_loss_cfg.tau_learned if phase_loss_cfg.tau_learned is not None else 0.1,
            'min_valid_per_row': phase_loss_cfg.min_valid_per_row if phase_loss_cfg.min_valid_per_row is not None else 2,
            'self_similarity_weight': phase_loss_cfg.self_similarity_weight if phase_loss_cfg.self_similarity_weight is not None else 1.0,
            'cross_pixel_weight': phase_loss_cfg.cross_pixel_weight if phase_loss_cfg.cross_pixel_weight is not None else 1.0,
            # Curriculum
            'curriculum_start_epoch': cur.start_epoch if cur else 10,
            'curriculum_ramp_epochs': cur.ramp_epochs if cur else 10,
            # Type-leakage penalty
            'phase_type_leakage_weight': phase_loss_cfg.phase_type_leakage_weight,
        }
        logger.info(
            f"Phase loss enabled: sampler={phase_anchor_pop}, "
            f"k={phase_config['k']}, min_overlap={phase_config['min_overlap']}, "
            f"min_pairs={phase_config['min_pairs']}, sigma={phase_config['sigma']}, "
            f"tau_ref={phase_config['tau_ref']}, tau_learned={phase_config['tau_learned']}, "
            f"weight={phase_config['weight']}, "
            f"curriculum=[start={phase_config['curriculum_start_epoch']}, "
            f"ramp={phase_config['curriculum_ramp_epochs']}], "
            f"leakage_weight={phase_config['phase_type_leakage_weight']}"
        )
    else:
        logger.info("Phase pair construction disabled (no soft_neighborhood_phase loss in config)")

    return phase_sampler, phase_config


def build_spread_config(bindings_config, logger: logging.Logger):
    """Build the phase spread-ranking loss config dict (None if not configured)."""
    spread_loss_cfg = bindings_config.get_loss('phase_spread_ranking')
    spread_config = None
    if spread_loss_cfg is not None:
        cur_s = spread_loss_cfg.curriculum
        spread_config = {
            'weight': spread_loss_cfg.weight if spread_loss_cfg.weight is not None else 0.5,
            'margin': spread_loss_cfg.margin if spread_loss_cfg.margin is not None else 0.1,
            'delta': spread_loss_cfg.delta if spread_loss_cfg.delta is not None else 0.5,
            'curriculum_start_epoch': cur_s.start_epoch if cur_s else 30,
            'curriculum_ramp_epochs': cur_s.ramp_epochs if cur_s else 10,
        }
        logger.info(
            f"Phase spread ranking loss enabled: weight={spread_config['weight']}, "
            f"margin={spread_config['margin']}, delta={spread_config['delta']}, "
            f"curriculum=[start={spread_config['curriculum_start_epoch']}, "
            f"ramp={spread_config['curriculum_ramp_epochs']}]"
        )
    return spread_config


def build_recovery_disc_config(bindings_config, logger: logging.Logger):
    """Build the phase recovery-discrimination loss config dict (None if not configured)."""
    recovery_disc_loss_cfg = bindings_config.get_loss('phase_recovery_discrimination')
    recovery_disc_config = None
    if recovery_disc_loss_cfg is not None:
        cur_rd = recovery_disc_loss_cfg.curriculum
        recovery_disc_config = {
            'weight': recovery_disc_loss_cfg.weight if recovery_disc_loss_cfg.weight is not None else 1.0,
            'margin': recovery_disc_loss_cfg.margin if recovery_disc_loss_cfg.margin is not None else 0.5,
            'low_ysfc_max': recovery_disc_loss_cfg.low_ysfc_max if recovery_disc_loss_cfg.low_ysfc_max is not None else 1.0,
            'high_ysfc_min': recovery_disc_loss_cfg.high_ysfc_min if recovery_disc_loss_cfg.high_ysfc_min is not None else 5.0,
            'curriculum_start_epoch': cur_rd.start_epoch if cur_rd else 30,
            'curriculum_ramp_epochs': cur_rd.ramp_epochs if cur_rd else 10,
        }
        logger.info(
            f"Phase recovery discrimination loss enabled: weight={recovery_disc_config['weight']}, "
            f"margin={recovery_disc_config['margin']}, "
            f"low_ysfc_max={recovery_disc_config['low_ysfc_max']}, "
            f"high_ysfc_min={recovery_disc_config['high_ysfc_min']}, "
            f"curriculum=[start={recovery_disc_config['curriculum_start_epoch']}, "
            f"ramp={recovery_disc_config['curriculum_ramp_epochs']}]"
        )
    return recovery_disc_config


def build_evt(bindings_config, feature_builder, loss_config: dict, device, logger: logging.Logger):
    """Build the EVT soft-neighbourhood metric + stratified anchor sampler.

    Mutates ``loss_config`` in place with ``evt_weight`` / ``evt_tau_ref`` /
    ``evt_tau_learned`` when active. Returns (evt_metric, evt_sampler); both None
    when the EVT loss is disabled (weight 0 or not configured).
    """
    from data.sampling.anchor_sampling import build_anchor_sampler
    from losses.evt_soft_neighborhood import EvtDiffusionMetric

    evt_metric = None
    evt_sampler = None
    evt_loss_cfg = bindings_config.get_loss('soft_neighborhood_evt')
    if evt_loss_cfg is not None and (evt_loss_cfg.weight or 0.0) > 0.0:
        evt_anchor_pop = (
            evt_loss_cfg.anchor_population
            if evt_loss_cfg.anchor_population
            else 'grid-plus-supplement-evt'
        )
        evt_sampler = build_anchor_sampler(bindings_config, evt_anchor_pop)
        # EVT code counts come from the shared stats file, at the path:
        #   stats["evt_class"]["static_categorical.evt"]["counts"]
        # Keys are string codes, values are integer pixel counts.
        evt_code_counts = (
            feature_builder.stats
            .get("evt_class", {})
            .get("static_categorical.evt", {})
            .get("counts", {})
        )
        if not evt_code_counts:
            raise ValueError(
                "EVT code counts not found in stats file. "
                "Run example_compute_stats.py to compute stats first."
            )
        evt_metric = EvtDiffusionMetric(
            confusion_csv=evt_loss_cfg.confusion_matrix_path,
            code_counts=evt_code_counts,
            min_count=evt_loss_cfg.min_count or 100,
            min_confusion_samples=evt_loss_cfg.min_confusion_samples or 30,
            diffusion_steps=evt_loss_cfg.diffusion_steps or 2,
            laplace_smoothing=evt_loss_cfg.laplace_smoothing or 0.0,
            binary_threshold=evt_loss_cfg.binary_threshold or 0.0,
        ).to(device)
        loss_config['evt_weight'] = evt_loss_cfg.weight
        loss_config['evt_tau_ref'] = evt_loss_cfg.tau_ref or 0.5
        loss_config['evt_tau_learned'] = evt_loss_cfg.tau_learned or 0.5
        # Restrict inverse-frequency weighting to valid EVT codes only.
        for spec in evt_sampler.weight_specs:
            if spec.transform == 'inverse-frequency':
                spec.valid_values = evt_metric.valid_codes
        logger.info(
            f"EVT soft neighbourhood loss enabled: "
            f"{evt_metric.n_codes} codes, "
            f"diffusion_steps={evt_loss_cfg.diffusion_steps or 2}, "
            f"min_count={evt_loss_cfg.min_count or 100}, "
            f"weight={loss_config['evt_weight']}, "
            f"tau_ref={loss_config['evt_tau_ref']}, "
            f"tau_learned={loss_config['evt_tau_learned']}, "
            f"anchor_population={evt_anchor_pop}"
        )
    else:
        logger.info("EVT soft neighbourhood loss disabled (weight=0 or not configured)")

    return evt_metric, evt_sampler
