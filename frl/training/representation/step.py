#!/usr/bin/env python3
"""Per-batch training/validation step for representation learning.

``process_batch`` runs one batch end to end: CPU prep (anchors, spatial + phase
pairs, spectral weights), a single batched encoder forward, per-sample loss
computation (spatial InfoNCE, VICReg, phase encoder + phase VCR, EVT), and the
cross-batch spectral and phase losses. Extracted verbatim from
``train_representation.py``.
"""

from __future__ import annotations

import logging
import time

import torch

from data.loaders.builders.feature_builder import FeatureBuilder
from data.sampling import sample_anchors_grid_plus_supplement
from data.sampling.anchor_sampling import AnchorSampler
from models import RepresentationModel
from losses import contrastive_loss, pairs_mutual_knn_chunked
from losses.phase_pairs import build_phase_pairs
from losses.phase_neighborhood import (
    build_phase_neighborhood_batch,
    compute_phase_spread_ranking,
    phase_neighborhood_loss,
)
from losses.triplet_phase import phase_recovery_discrimination_loss
from losses.type_phase_contrastive import type_phase_contrastive_loss
from losses.variance_covariance import variance_covariance_loss
from losses.evt_soft_neighborhood import EvtDiffusionMetric, evt_soft_neighborhood_loss
from utils import (
    extract_at_locations,
    extract_temporal_at_locations,
    spatial_knn_pairs,
    spatial_negative_pairs,
)
from training.representation.curriculum import ramp_weight
from training.representation.profiling import is_profiling

# Keep the original module's logger name so slurm-log records are byte-identical
# to the pre-refactor inline code (the messages moved, the %(name)s should not).
logger = logging.getLogger("training.train_representation")


def pair_l2(a: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
    # a: [N, C], pairs: [P, 2] -> returns [P]
    v1 = a[pairs[:, 0]]
    v2 = a[pairs[:, 1]]
    return torch.norm(v1 - v2, dim=1)


# Per-patch cap on gate values kept for the per-epoch distribution diagnostic.
# The gate tensor is [D,H,W] (~4.2M values); a few thousand samples are plenty
# for mean/std/quantile summaries and avoid a multi-second cat+quantile per batch.
_GATE_STATS_SAMPLES = 4096


def _get_feature(
    feature_name: str,
    batch: dict,
    sample_idx: int,
    sample: dict,
    feature_builder: FeatureBuilder,
) -> 'FeatureResult':
    """Return a precomputed FeatureResult from the batch if available, else build it.

    When ForestDatasetV2 is configured with feature_builder + precompute_features,
    the data/mask arrays arrive already processed in the batch dict. This avoids
    repeating the whitening transform in the main process.
    """
    from data.loaders.builders.feature_builder import FeatureResult
    data_key = f'__feat_{feature_name}_data'
    if data_key in batch:
        return FeatureResult(
            data=batch[data_key][sample_idx].numpy(),
            mask=batch[f'__feat_{feature_name}_mask'][sample_idx].numpy(),
            feature_name=feature_name,
            channel_names=[],
            is_temporal=False,
        )
    return feature_builder.build_feature(feature_name, sample)


def process_batch(
    batch: dict,
    feature_builder: FeatureBuilder,
    model: RepresentationModel,
    device: torch.device,
    config: dict,
    training: bool = True,
    optimizer: torch.optim.Optimizer | None = None,
    phase_sampler: AnchorSampler | None = None,
    phase_config: dict | None = None,
    spread_config: dict | None = None,
    recovery_disc_config: dict | None = None,
    epoch: int = 0,
    evt_metric: EvtDiffusionMetric | None = None,
    evt_sampler: AnchorSampler | None = None,
) -> dict:
    """
    Process a single batch for training or validation.

    Args:
        batch: Batch from dataloader
        feature_builder: FeatureBuilder instance
        model: RepresentationModel (encoder + spatial conv)
        device: Device to use
        config: Loss config dict
        training: If True, run in training mode (gradients, optimizer step).
                  If False, run in eval mode (no gradients, no jitter).
        optimizer: Optimizer (required if training=True)
        phase_sampler: Optional anchor sampler for phase loss pair construction
        phase_config: Optional phase loss config dict (k, min_overlap, etc.)
        epoch: Current epoch (0-indexed), used for curriculum weighting
        evt_sampler: Optional EVT-stratified anchor sampler; oversamples rare EVT codes

    Returns:
        Dict with loss values and stats
    """
    if training:
        if optimizer is None:
            raise ValueError("optimizer is required when training=True")
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()

    # Use jitter only during training
    jitter_radius = config.get('jitter_radius', 4) if training else 0

    # Process each sample in batch (pair generation is per-patch)
    total_loss = 0.0
    total_spectral_loss = 0.0
    total_spatial_loss = 0.0
    total_phase_loss = 0.0
    total_phase_spread_loss = 0.0
    total_phase_recovery_disc_loss = 0.0
    total_phase_anchor_loss = 0.0
    total_phase_ou_loss = 0.0
    last_ou_diag = None
    # Phase-radius diagnostic: RMS ‖z_phase‖ split by recovery state (hub-and-rim
    # check). Accumulate ΣΣ‖z‖² and counts; loops.py forms the epoch RMS.
    mature_r2_sum = 0.0
    mature_r2_count = 0
    disturbed_r2_sum = 0.0
    disturbed_r2_count = 0
    total_vcr_loss = 0.0
    total_phase_vcr_loss = 0.0
    total_evt_loss = 0.0
    all_evt_diag: list[dict] = []  # accumulate per-sample EVT diagnostics
    n_valid = 0
    total_spectral_pos_pairs = 0
    total_spectral_neg_pairs = 0
    total_spatial_pos_pairs = 0
    total_spatial_neg_pairs = 0

    # Collectors for distribution logging (accumulated across samples)
    all_gate_values = []
    all_pos_weights = []
    all_neg_weights = []
    all_pos_sims = []
    all_neg_sims = []
    all_pos_spec_dists = []
    all_neg_spec_dists = []
    _TAU_SWEEP = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    tau_sweep_pos: dict[float, list] = {t: [] for t in _TAU_SWEEP}
    tau_sweep_neg: dict[float, list] = {t: [] for t in _TAU_SWEEP}

    # Phase pair stats accumulators
    all_phase_pair_stats = []
    all_phase_loss_stats = []

    # Timing accumulators (seconds)
    t_sample_build = 0.0
    t_feature_build = 0.0
    t_anchor_sample = 0.0
    t_spatial_pairs = 0.0
    t_spectral_weights = 0.0
    t_gpu_forward = 0.0
    t_phase_pairs = 0.0
    t_phase_forward = 0.0
    t_loss_compute = 0.0
    t_backward = 0.0
    t_pass1 = 0.0  # whole per-sample feature/pair build loop (superset of feature_build etc.)
    t_pass2 = 0.0  # whole per-sample loss loop (superset of phase_forward+loss_compute)
    t_temporal_build = 0.0  # subset of pass2: per-sample full-grid phase_ccdc/dynamism builds

    # RFF-bandwidth coverage diagnostic (mean leverage per batch).
    all_readout_leverage = []
    # Held-out readout μ-fit accumulators (pooled R² = 1 − Σss_res/Σss_tot).
    readout_ss_res = 0.0
    readout_ss_tot = 0.0
    # z_phase→z_type leakage guard: z_phase mean-pooled over T, and z_type.
    all_pre_film_h_mean = []   # [N, zp] per batch (z_phase mean over T; was pre-FiLM h)
    all_z_type_at_phase = []   # [N, z_type_dim] per batch

    # Collectors for global spectral loss (pairs built cross-batch after loop)
    cross_patch_z_anchors: list[torch.Tensor] = []
    cross_patch_spec_features: list[torch.Tensor] = []
    cross_patch_anchor_coords: list[torch.Tensor] = []

    # Collectors for cross-batch phase loss (assembled after loop like spectral loss)
    cross_phase_z_type: list[torch.Tensor] = []
    cross_phase_spec: list[torch.Tensor] = []
    cross_phase_embeddings: list[torch.Tensor] = []
    cross_phase_ysfc: list[torch.Tensor] = []
    cross_phase_pairs: list[torch.Tensor] = []
    cross_phase_weights: list[torch.Tensor] = []
    cross_phase_dynamism: list[torch.Tensor] = []
    cross_phase_h: list[torch.Tensor] = []  # non-detached h for Frobenius loss
    cross_phase_flow: list[torch.Tensor] = []       # [N,T,2C] anomaly (a,Δa) — Step-5 flow-state
    cross_phase_disturbed: list[torch.Tensor] = []  # [N,T] disturbed pixel-times (Step-5 filter)
    cross_phase_n_offset: int = 0  # running pixel count for pair index remapping

    # Hoist epoch-dependent curriculum weights (same for every patch in the batch)
    if phase_config is not None:
        curriculum_w = ramp_weight(
            epoch,
            phase_config.get('curriculum_start_epoch', 10),
            phase_config.get('curriculum_ramp_epochs', 10),
        )
    else:
        curriculum_w = 0.0

    # Mature-timestep selector (readout fit + anchor loss) and anchor-loss weight.
    mature_ysfc_threshold = (
        phase_config.get('mature_ysfc_threshold', 12.0) if phase_config else 12.0)
    anchor_weight = (
        phase_config.get('anchor_weight', 0.05) if phase_config else 0.05)
    ou_weight = (
        phase_config.get('ou_weight', 1.0) if phase_config else 1.0)

    if spread_config is not None:
        spread_w = ramp_weight(
            epoch,
            spread_config['curriculum_start_epoch'],
            spread_config['curriculum_ramp_epochs'],
        )
    else:
        spread_w = 0.0

    if recovery_disc_config is not None:
        rd_w = ramp_weight(
            epoch,
            recovery_disc_config['curriculum_start_epoch'],
            recovery_disc_config['curriculum_ramp_epochs'],
        )
    else:
        rd_w = 0.0

    batch_size = len(batch['metadata'])

    # ------------------------------------------------------------------
    # PASS 1 — CPU prep for every sample (anchors, spatial pairs, weights).
    # The encoder forward is deferred to a single batched call below so the
    # B sequential [1,C,H,W] forwards become one [B,C,H,W] forward.
    # ------------------------------------------------------------------
    prep_list: list[dict | None] = [None] * batch_size

    _t_pass1 = time.perf_counter()
    for i in range(batch_size):
        # Extract single sample from batch
        _t0 = time.perf_counter()
        sample = {
            key: val[i].numpy() if isinstance(val, torch.Tensor) else val[i]
            for key, val in batch.items()
            if key != 'metadata' and not key.startswith('__spatial_')
        }
        sample['metadata'] = batch['metadata'][i]
        t_sample_build += time.perf_counter() - _t0

        # Build features — encoder_data stays CPU for batched forward
        _t0 = time.perf_counter()
        encoder_feature = _get_feature(config['type_encoder_feature'], batch, i, sample, feature_builder)
        spec_dist_feature = _get_feature('infonce_type_spectral', batch, i, sample, feature_builder)

        # Convert to tensors. encoder_data stays on CPU until the batched forward.
        encoder_data = torch.from_numpy(encoder_feature.data).float()
        spec_dist_data = torch.from_numpy(spec_dist_feature.data).float().to(device)
        mask = torch.from_numpy(encoder_feature.mask).to(device)

        # Also apply distance feature mask
        spec_dist_mask = torch.from_numpy(spec_dist_feature.mask).to(device)
        combined_mask = mask & spec_dist_mask
        t_feature_build += time.perf_counter() - _t0

        # Worker-precomputed spatial pairs are only valid when anchors were drawn
        # by the same grid+supplement sampler. With an EVT-stratified sampler the
        # anchors differ, so fall back to recomputing on the main process.
        _spatial_precomputed = (
            evt_sampler is None
            and '__spatial_valid' in batch
            and batch['__spatial_valid'][i] is not None
            and bool(batch['__spatial_valid'][i].item())
        )
        if not _spatial_precomputed and i == 0 and epoch == 0:
            logger.info(
                f"[precompute diag sample 0] "
                f"evt_sampler={'set' if evt_sampler is not None else 'None'}, "
                f"'__spatial_valid' in batch={('__spatial_valid' in batch)}, "
                f"val[0]={batch['__spatial_valid'][0] if '__spatial_valid' in batch else 'MISSING'}"
            )

        _t0 = time.perf_counter()
        if _spatial_precomputed:
            anchors          = batch['__spatial_anchors'][i].to(device)
            unique_coords    = batch['__spatial_unique_coords'][i].to(device)
            spatial_pos_pairs = batch['__spatial_pos_pairs'][i].to(device)
            spatial_neg_pairs = batch['__spatial_neg_pairs'][i].to(device)
            pos_weights      = batch['__spatial_pos_weights'][i].to(device)
            neg_weights      = batch['__spatial_neg_weights'][i].to(device)
            spec_dist_at_anchors = batch['__spatial_spec_dist_at_anchors'][i].to(device)

            n_pos = spatial_pos_pairs.shape[0]
            n_neg = spatial_neg_pairs.shape[0]
            if n_pos > 0:
                all_pos_weights.append(pos_weights.detach().cpu())
            if n_neg > 0:
                all_neg_weights.append(neg_weights.detach().cpu())
            t_anchor_sample += time.perf_counter() - _t0
        else:
            # Sample anchor locations — use EVT-stratified sampler when available so
            # rare forest types are represented for cross-batch kNN pair construction.
            if evt_sampler is not None:
                anchors = evt_sampler(combined_mask, training=training, sample=sample)
            else:
                anchors = sample_anchors_grid_plus_supplement(
                    combined_mask,
                    stride=config.get('stride', 16),
                    border=config.get('border', 16),
                    jitter_radius=jitter_radius,
                    supplement_n=config.get('supplement_n', 104),
                )
            t_anchor_sample += time.perf_counter() - _t0

            if anchors.shape[0] < 10:
                continue

            # Extract features at anchor locations
            # spec_dist_at_anchors collected here; pairs built cross-batch after loop.
            spec_dist_at_anchors = extract_at_locations(spec_dist_data, anchors)

            # --- Spatial InfoNCE pair generation (offset-based, no full matrix) ---
            _t0 = time.perf_counter()
            pos_anchor_idx, pos_neighbor_coords = spatial_knn_pairs(
                anchors,
                combined_mask,
                k=config.get('spatial_positive_k', 4),
                max_radius=int(config.get('spatial_positive_max_dist', 8)),
            )

            neg_anchor_idx, neg_neighbor_coords = spatial_negative_pairs(
                anchors,
                combined_mask,
                min_distance=config.get('spatial_negative_min_dist', 16.0),
                max_distance=config.get('spatial_negative_max_dist', None),
                n_per_anchor=config.get('spatial_negatives_per_anchor', 4),
            )

            # Build coordinate-to-index mapping for spatial loss
            all_spatial_coords = [anchors]
            if pos_neighbor_coords.numel() > 0:
                all_spatial_coords.append(pos_neighbor_coords)
            if neg_neighbor_coords.numel() > 0:
                all_spatial_coords.append(neg_neighbor_coords)

            all_spatial_coords = torch.cat(all_spatial_coords, dim=0)  # [N+M+K, 2]

            unique_coords, inverse_indices = torch.unique(
                all_spatial_coords, dim=0, return_inverse=True
            )

            n_anchors_spatial = anchors.shape[0]
            anchor_to_unique = inverse_indices[:n_anchors_spatial]

            n_pos = pos_neighbor_coords.shape[0] if pos_neighbor_coords.numel() > 0 else 0
            n_neg = neg_neighbor_coords.shape[0] if neg_neighbor_coords.numel() > 0 else 0

            spatial_pos_pairs = torch.zeros((0, 2), dtype=torch.long, device=device)
            spatial_neg_pairs = torch.zeros((0, 2), dtype=torch.long, device=device)

            if n_pos > 0:
                pos_neighbor_unique = inverse_indices[n_anchors_spatial : n_anchors_spatial + n_pos]
                pos_anchor_unique = anchor_to_unique[pos_anchor_idx]
                spatial_pos_pairs = torch.stack([pos_anchor_unique, pos_neighbor_unique], dim=1)

            if n_neg > 0:
                neg_neighbor_unique = inverse_indices[n_anchors_spatial + n_pos :]
                neg_anchor_unique = anchor_to_unique[neg_anchor_idx]
                spatial_neg_pairs = torch.stack([neg_anchor_unique, neg_neighbor_unique], dim=1)
            t_spatial_pairs += time.perf_counter() - _t0

            # --- Spectral weighting for spatial pairs ---
            _t0 = time.perf_counter()
            spec_dist_unique = extract_at_locations(spec_dist_data, unique_coords)  # [Nuniq, Cdist]

            tau = config.get("spatial_spectral_tau", 1.0)
            min_w = config.get("spatial_min_w", 0.05)

            pos_weights = None
            neg_weights = None

            if spatial_pos_pairs.numel() > 0:
                dpos = pair_l2(spec_dist_unique, spatial_pos_pairs)
                pos_weights = torch.exp(-dpos / tau).clamp(min=min_w, max=1.0)
                all_pos_weights.append(pos_weights.detach().cpu())
                all_pos_spec_dists.append(dpos.detach().cpu())
                dpos_cpu = dpos.detach().cpu()
                if epoch == 0:
                    for t in _TAU_SWEEP:
                        tau_sweep_pos[t].append(torch.exp(-dpos_cpu / t).clamp(min=min_w, max=1.0))

            if spatial_neg_pairs.numel() > 0:
                dneg = pair_l2(spec_dist_unique, spatial_neg_pairs)
                neg_weights = (1.0 - torch.exp(-dneg / tau)).clamp(min=min_w, max=1.0)
                all_neg_weights.append(neg_weights.detach().cpu())
                all_neg_spec_dists.append(dneg.detach().cpu())
                dneg_cpu = dneg.detach().cpu()
                if epoch == 0:
                    for t in _TAU_SWEEP:
                        tau_sweep_neg[t].append((1.0 - torch.exp(-dneg_cpu / t)).clamp(min=min_w, max=1.0))
            t_spectral_weights += time.perf_counter() - _t0

        # Check if we have valid pairs for losses
        # Spectral: pairs are built cross-batch after the loop; just need valid anchors.
        has_spectral = spec_dist_at_anchors.shape[0] > 0
        has_spatial  = spatial_pos_pairs.shape[0] > 0 and spatial_neg_pairs.shape[0] > 0

        if not has_spectral and not has_spatial:
            continue  # prep_list[i] stays None

        # --- Phase pair construction (CPU; TCN forward deferred to Pass 2) ---
        phase_prep = None
        if phase_sampler is not None and phase_config is not None:
            ysfc_feature = _get_feature('ysfc', batch, i, sample, feature_builder)
            ysfc_data = torch.from_numpy(ysfc_feature.data).float()
            ysfc_mask = torch.from_numpy(ysfc_feature.mask)
            if ysfc_mask.ndim == 3:
                ysfc_spatial_mask = ysfc_mask.all(dim=0)
            else:
                ysfc_spatial_mask = ysfc_mask
            phase_mask = combined_mask.cpu() & ysfc_spatial_mask

            phase_anchors = phase_sampler(phase_mask, training=training, sample=sample)

            if phase_anchors.shape[0] >= 10:
                phase_spec_at_anchors = extract_at_locations(
                    spec_dist_data.cpu(), phase_anchors)
                ysfc_at_anchors = extract_temporal_at_locations(
                    ysfc_data, phase_anchors).squeeze(-1)

                _t0 = time.perf_counter()
                phase_pairs, phase_weights, phase_stats = build_phase_pairs(
                    spec_features=phase_spec_at_anchors,
                    ysfc=ysfc_at_anchors,
                    k=phase_config.get('k', 16),
                    min_overlap=phase_config.get('min_overlap', 3),
                    min_pairs=phase_config.get('min_pairs', 5),
                    include_self=phase_config.get('include_self', True),
                    sigma=phase_config.get('sigma', 5.0),
                    self_pair_weight=phase_config.get('self_pair_weight', 1.0),
                )
                t_phase_pairs += time.perf_counter() - _t0
                all_phase_pair_stats.append(phase_stats)

                phase_prep = {
                    'phase_anchors':   phase_anchors,
                    'phase_pairs':     phase_pairs,
                    'phase_weights':   phase_weights,
                    'ysfc_at_anchors': ysfc_at_anchors,
                }

        prep_list[i] = {
            'sample':             sample,
            'encoder_data':       encoder_data,        # CPU [C,H,W]
            'spec_dist_data':     spec_dist_data,      # GPU tensor
            'combined_mask':      combined_mask,       # GPU bool
            'anchors':            anchors,
            'unique_coords':      unique_coords,
            'spatial_pos_pairs':  spatial_pos_pairs,
            'spatial_neg_pairs':  spatial_neg_pairs,
            'pos_weights':        pos_weights,
            'neg_weights':        neg_weights,
            'spec_dist_at_anchors': spec_dist_at_anchors,
            'has_spectral':       has_spectral,
            'has_spatial':        has_spatial,
            'phase_prep':         phase_prep,
        }

    if is_profiling() and device.type == 'cuda':
        torch.cuda.synchronize()
    t_pass1 = time.perf_counter() - _t_pass1

    # ── BATCHED GPU FORWARD ───────────────────────────────────────────────
    # Chunk the forward pass to bound peak GPU memory. Each chunk processes
    # enc_chunk_size samples; results are concatenated before Pass 2.
    valid_prep = [(idx, p) for idx, p in enumerate(prep_list) if p is not None]
    z_batch = gate_batch = None
    enc_chunk_size = config.get('enc_chunk_size', 4)
    if valid_prep:
        _t0 = time.perf_counter()
        z_chunks, gate_chunks = [], []
        all_enc_inputs = [p['encoder_data'] for _, p in valid_prep]
        for chunk_start in range(0, len(all_enc_inputs), enc_chunk_size):
            chunk = torch.stack(all_enc_inputs[chunk_start:chunk_start + enc_chunk_size]).to(device)
            z_c, gate_c = model(chunk, return_gate=True)
            z_chunks.append(z_c)
            gate_chunks.append(gate_c)
        z_batch = torch.cat(z_chunks, dim=0)
        gate_batch = torch.cat(gate_chunks, dim=0)
        if is_profiling() and device.type == 'cuda':
            torch.cuda.synchronize()
        t_gpu_forward = time.perf_counter() - _t0

    # ── PASS 2: PER-SAMPLE LOSS COMPUTATION ──────────────────────────────
    _t_pass2 = time.perf_counter()
    for out_idx, (i, prep) in enumerate(valid_prep):
        sample = prep['sample']
        z_full = z_batch[out_idx]    # [D, H, W]
        gate   = gate_batch[out_idx]  # [D, H, W]

        anchors           = prep['anchors'].to(device)
        unique_coords     = prep['unique_coords'].to(device)
        spatial_pos_pairs = prep['spatial_pos_pairs'].to(device)
        spatial_neg_pairs = prep['spatial_neg_pairs'].to(device)
        pos_weights       = prep['pos_weights'].to(device) if prep['pos_weights'] is not None else None
        neg_weights       = prep['neg_weights'].to(device) if prep['neg_weights'] is not None else None
        has_spatial           = prep['has_spatial']
        has_spectral          = prep['has_spectral']
        combined_mask_cpu     = prep['combined_mask'].cpu()
        spec_dist_at_anchors  = prep['spec_dist_at_anchors']  # already on device

        # Collect gate values on CPU for the per-epoch distribution log.
        # Subsample: gate is [D,H,W] ~4.2M values/patch, so keeping all of them
        # meant a 16x4.2M=67M-element cat + randperm(67M) + quantile in
        # compute_stats *every batch* (~3.5s/batch, and it only feeds a diagnostic
        # that is frozen at 1.0 during the smoothing curriculum). A few thousand
        # random values per patch give the same distribution summary for free.
        _gate_flat = gate.detach().flatten()
        if _gate_flat.numel() > _GATE_STATS_SAMPLES:
            _gsub = torch.randint(_gate_flat.numel(), (_GATE_STATS_SAMPLES,), device=_gate_flat.device)
            _gate_flat = _gate_flat[_gsub]
        all_gate_values.append(_gate_flat.cpu())

        # Extract embeddings at anchor locations
        z_anchors = extract_at_locations(z_full, anchors)  # [num_anchors, D]

        # EVT soft neighbourhood loss
        evt_loss_val = torch.tensor(0.0, device=device)
        if evt_metric is not None:
            evt_feature = _get_feature('evt_class', batch, i, sample, feature_builder)
            evt_data = torch.from_numpy(evt_feature.data).long().to(device)
            if evt_sampler is not None:
                evt_anchors = evt_sampler(combined_mask_cpu.to(device), training=training, sample=sample)
                z_evt = extract_at_locations(z_full, evt_anchors)
                evt_at_anchors = extract_at_locations(evt_data, evt_anchors).squeeze(1)
            else:
                z_evt = z_anchors
                evt_at_anchors = extract_at_locations(evt_data, anchors).squeeze(1)
            evt_raw, evt_diag = evt_soft_neighborhood_loss(
                z_evt, evt_at_anchors, evt_metric,
                tau_ref=config.get('evt_tau_ref', 0.5),
                tau_learned=config.get('evt_tau_learned', 0.5),
            )
            all_evt_diag.append(evt_diag)
            evt_loss_val = config.get('evt_weight', 0.0) * evt_raw

        # Variance-covariance regularization
        vcr_loss_val = torch.tensor(0.0, device=device)
        if config.get('vcr_enabled', False) and z_anchors.shape[0] >= 2:
            vcr_total, _, _ = variance_covariance_loss(
                z_anchors,
                variance_weight=config.get('vcr_variance_weight', 1.0),
                covariance_weight=config.get('vcr_covariance_weight', 1.0),
                variance_target=config.get('vcr_variance_target', 1.0),
            )
            vcr_loss_val = config.get('vcr_weight', 0.1) * vcr_total

        # Spatial loss
        spatial_loss_val = torch.tensor(0.0, device=device)
        if has_spatial:
            z_spatial = extract_at_locations(z_full, unique_coords)
            spatial_loss_val = contrastive_loss(
                z_spatial, spatial_pos_pairs, spatial_neg_pairs,
                pos_weights=pos_weights, neg_weights=neg_weights,
                temperature=config.get('spatial_temperature', 0.07),
                similarity='l2',
            )
            with torch.no_grad():
                p_a, p_b = z_spatial[spatial_pos_pairs[:, 0]], z_spatial[spatial_pos_pairs[:, 1]]
                n_a, n_b = z_spatial[spatial_neg_pairs[:, 0]], z_spatial[spatial_neg_pairs[:, 1]]
                D = z_spatial.shape[1]
                all_pos_sims.append((-(p_a - p_b).pow(2).sum(1) / D).cpu())
                all_neg_sims.append((-(n_a - n_b).pow(2).sum(1) / D).cpu())

        # Phase pathway (needs z_type from z_full — must stay in Pass 2).
        #
        # Two-part structure (see docs/phase_rethink_design.md Step 3):
        #   ALWAYS (incl. type-only warmup) — build the anomaly input, settle the
        #     μ/σ readout on mature timesteps, and let the transform observe the Δ
        #     scale.  This is what lets the readout settle before the encoder turns on.
        #   GATED on curriculum_w>0 — run the FiLM-free encoder → z_phase, the anchor
        #     loss, phase VCR, and the cross-batch phase-loss accumulation.
        phase_loss_val = torch.tensor(0.0, device=device)
        phase_spread_loss_val = torch.tensor(0.0, device=device)
        phase_recovery_disc_loss_val = torch.tensor(0.0, device=device)
        phase_vcr_loss_val = torch.tensor(0.0, device=device)
        phase_anchor_loss_val = torch.tensor(0.0, device=device)
        phase_ou_loss_val = torch.tensor(0.0, device=device)
        pp = prep['phase_prep']
        if pp is not None:
                phase_anchors   = pp['phase_anchors']
                ysfc_at_anchors = pp['ysfc_at_anchors']            # [N, T]
                phase_anchors_dev = phase_anchors.to(device)

                # Raw phase feature x at anchors (anchor-only build; [N, T, C]).
                _t_tb = time.perf_counter()
                x_np, _ = feature_builder.build_feature_at_locations(
                    config['phase_encoder_feature'], sample, phase_anchors)   # [N, T, C]
                t_temporal_build += time.perf_counter() - _t_tb
                x_tc = torch.from_numpy(x_np).float().to(device)             # [N, T, C]
                x = x_tc.permute(0, 2, 1)                                    # [N, C, T]

                # Stop-grad z_type at phase anchors (readout also detaches internally).
                z_type_at_anchors = extract_at_locations(z_full.detach(), phase_anchors_dev)  # [N, d]

                ysfc_dev = ysfc_at_anchors.to(device)                       # [N, T]
                mature = ysfc_dev > mature_ysfc_threshold                   # [N, T]

                # μ/σ (pre-update readout) → anomaly [a ; Δa]; the transform observes
                # the Δ scale during warmup (training & unlocked).  Runs every batch.
                mu, sigma = model.mature_baseline.predict(z_type_at_anchors)   # [N, C]
                _t0 = time.perf_counter()
                anomaly_feats, phase_valid = model.anomaly_transform(x, mu, sigma)  # [N, 2C, T], [N, T]
                t_phase_forward += time.perf_counter() - _t0
                # ‖Δa‖ per timestep (the Δ block of the anomaly input) — OU gate signal.
                _C = x.shape[1]
                delta_a_norm = anomaly_feats[:, _C:, :].norm(dim=1)         # [N, T]

                # Held-out readout fit (μ-R²) on this batch's mature samples, measured
                # BEFORE folding them in → genuine held-out. R²>0 ⇔ bandwidth is right
                # (the diagnostic that catches a mis-calibrated h; leverage can't).
                if bool(mature.any()):
                    T_ = x.shape[2]
                    z_rep = z_type_at_anchors.unsqueeze(1).expand(-1, T_, -1)   # [N, T, d]
                    z_m, x_m = z_rep[mature], x_tc[mature]                      # [M, d], [M, C]
                    _sr, _st = model.mature_baseline.heldout_r2(z_m, x_m)
                    readout_ss_res += float(_sr); readout_ss_tot += float(_st)
                    # Settle the readout on these mature samples (training only; the
                    # predict above used the pre-update readout → no within-batch leak).
                    if training:
                        model.mature_baseline.update(z_m, x_m)

                # RFF-bandwidth coverage diagnostic (mean leverage = prior-reliance).
                all_readout_leverage.append(
                    model.mature_baseline.leverage(z_type_at_anchors).mean().detach())

                # Encoder + phase losses: gated on the phase curriculum.
                if pp['phase_pairs'].shape[0] > 0 and curriculum_w > 0.0:
                    phase_pairs   = pp['phase_pairs']
                    phase_weights = pp['phase_weights']

                    _t0 = time.perf_counter()
                    z_phase_at_anchors = model.encode_phase(anomaly_feats)  # [N, T, zp]
                    if is_profiling() and device.type == 'cuda':
                        torch.cuda.synchronize()
                    t_phase_forward += time.perf_counter() - _t0

                    # Anchor loss: pin mature z_phase to the single shared origin.
                    if bool(mature.any()):
                        zp_mature = z_phase_at_anchors[mature]              # [M, zp]
                        phase_anchor_loss_val = (
                            anchor_weight * curriculum_w
                            * zp_mature.pow(2).sum(dim=-1).mean()
                        )

                    # Radius diagnostic: RMS ‖z_phase‖ for mature vs disturbed
                    # (non-mature) timesteps. Mature should pin toward the origin;
                    # disturbed should eject. Grad-free; accumulated across samples.
                    with torch.no_grad():
                        zp_r2 = z_phase_at_anchors.pow(2).sum(dim=-1)      # [N, T]
                        m_sel = mature & phase_valid
                        d_sel = (~mature) & phase_valid
                        if bool(m_sel.any()):
                            mature_r2_sum += float(zp_r2[m_sel].sum())
                            mature_r2_count += int(m_sel.sum())
                        if bool(d_sel.any()):
                            disturbed_r2_sum += float(zp_r2[d_sel].sum())
                            disturbed_r2_count += int(d_sel.sum())

                    # Step-4 within-pixel OU dynamics (disturbance-gated transition NLL).
                    ou_raw, ou_diag = model.ou_dynamics(
                        z_phase_at_anchors, delta_a_norm, phase_valid)
                    phase_ou_loss_val = ou_weight * curriculum_w * ou_raw
                    last_ou_diag = ou_diag

                    # z_phase → z_type leakage guard (was pre-FiLM h; h == z_phase now):
                    # with the type-agnostic encoder this should stay ≈0.
                    zphase_mean = z_phase_at_anchors.mean(dim=1)           # [N, zp]
                    all_pre_film_h_mean.append(zphase_mean.detach())
                    all_z_type_at_phase.append(z_type_at_anchors.detach())
                    cross_phase_h.append(zphase_mean)

                    # Phase VCR (per-patch; retired in Step 5).
                    phase_vcr_cfg = config.get('phase_vcr_config')
                    if phase_vcr_cfg is not None:
                        z_phase_flat = z_phase_at_anchors.reshape(-1, model.z_phase_dim)
                        pvcr_total, _, _ = variance_covariance_loss(
                            z_phase_flat,
                            variance_weight=phase_vcr_cfg.get('variance_weight', 1.0),
                            covariance_weight=phase_vcr_cfg.get('covariance_weight', 1.0),
                            variance_target=phase_vcr_cfg.get('variance_target', 1.0),
                        )
                        phase_vcr_loss_val = phase_vcr_cfg.get('weight', 0.1) * curriculum_w * pvcr_total

                    # Cross-batch phase loss accumulation (offset pair indices). The
                    # raw feature x_tc is the pair-construction space (unchanged).
                    cross_phase_z_type.append(z_type_at_anchors)
                    cross_phase_spec.append(x_tc)
                    cross_phase_embeddings.append(z_phase_at_anchors)
                    cross_phase_ysfc.append(ysfc_dev)
                    # Step-5 flow-state (a,Δa) per pixel-time + recovery-tube pool.
                    cross_phase_flow.append(anomaly_feats.permute(0, 2, 1))   # [N, T, 2C]
                    # Pool = ALL valid timesteps of any pixel disturbed somewhere in
                    # the window (any non-mature ysfc). This pulls each such pixel's
                    # own mature (pre-disturbance / recovered) timesteps in as
                    # origin-anchored negatives, so the InfoNCE forms the
                    # mature↔disturbed triangle and the fixed τ pins the ejection
                    # radius (not just the inter-disturbed spread). Always-mature
                    # pixels carry no tube signal and are excluded.
                    informative = (~mature).any(dim=1, keepdim=True)          # [N, 1]
                    cross_phase_disturbed.append(informative & phase_valid)   # [N, T]
                    cross_phase_pairs.append(phase_pairs.to(device) + cross_phase_n_offset)
                    cross_phase_weights.append(phase_weights.to(device))
                    cross_phase_n_offset += z_type_at_anchors.shape[0]

                    # Accumulate dynamism for spread loss.
                    if spread_config is not None:
                        _t_tb = time.perf_counter()
                        dynamism_np, _ = feature_builder.build_feature_at_locations(
                            'phase_dynamism_supervision', sample, phase_anchors)   # [N, C]
                        cross_phase_dynamism.append(torch.from_numpy(dynamism_np).float())
                        t_temporal_build += time.perf_counter() - _t_tb

        # Combine per-patch losses (phase losses computed cross-batch after loop)
        _t0 = time.perf_counter()
        spatial_weight = config.get('spatial_loss_weight', 1.0)
        loss = (spatial_weight * spatial_loss_val
                + vcr_loss_val
                + phase_vcr_loss_val
                + phase_anchor_loss_val
                + phase_ou_loss_val
                + evt_loss_val)
        t_loss_compute += time.perf_counter() - _t0

        # Skip if loss is NaN or Inf (numerical instability)
        if not torch.isfinite(loss):
            md = sample['metadata']
            sw = md.get('spatial_window')
            origin = (sw.row_start, sw.col_start) if sw is not None else '?'
            patch_idx = md.get('patch_idx', '?')
            def _fmt(t):
                v = t.item() if hasattr(t, 'item') else float(t)
                return f"{v:.4f}"
            logger.warning(
                f"Skipping sample with non-finite loss: {loss.item()} | "
                f"patch_idx={patch_idx} origin={origin} | "
                f"spat={_fmt(spatial_weight * spatial_loss_val)} "
                f"phase={_fmt(phase_loss_val)} "
                f"spr={_fmt(phase_spread_loss_val)} "
                f"vcr={_fmt(vcr_loss_val)} "
                f"pvcr={_fmt(phase_vcr_loss_val)}"
            )
            continue

        # Accumulate cross-batch spectral inputs only for finite-loss samples.
        if has_spectral:
            cross_patch_z_anchors.append(model.project_type(z_anchors))
            cross_patch_spec_features.append(spec_dist_at_anchors)
            cross_patch_anchor_coords.append(anchors)

        # Accumulate: keep as tensor for training (backward), use .item() for validation
        if training:
            total_loss += loss
            total_spatial_loss += spatial_weight * spatial_loss_val
            total_vcr_loss += vcr_loss_val
            total_phase_vcr_loss += phase_vcr_loss_val
            total_phase_anchor_loss += phase_anchor_loss_val
            total_phase_ou_loss += phase_ou_loss_val
            total_evt_loss += evt_loss_val
        else:
            total_loss += loss.item()
            total_spatial_loss += (spatial_weight * spatial_loss_val).item()
            total_vcr_loss += vcr_loss_val.item()
            total_phase_vcr_loss += phase_vcr_loss_val.item()
            total_phase_anchor_loss += float(phase_anchor_loss_val)
            total_phase_ou_loss += float(phase_ou_loss_val)
            total_evt_loss += evt_loss_val.item()
        n_valid += 1
        total_spatial_pos_pairs += spatial_pos_pairs.shape[0]
        total_spatial_neg_pairs += spatial_neg_pairs.shape[0]

    if is_profiling() and device.type == 'cuda':
        torch.cuda.synchronize()
    t_pass2 = time.perf_counter() - _t_pass2

    # --- Global Spectral InfoNCE with Cross-Batch kNN ---
    # All anchors from all samples in the batch are pooled into a single
    # N_total × N_total spectral distance matrix. Positive pairs are mutual
    # kNN across the full pool — spectrally similar pixels from *different*
    # patches can now be positives, teaching location-invariant forest type.
    # Negatives are sampled from the [q_low, q_high) quantile range of spectral
    # distances and weighted by 1 - exp(-d/tau) to suppress false negatives
    # (cross-patch pairs that happen to be spectrally similar).
    spectral_weight = config.get('spectral_loss_weight', 1.0)
    global_spectral_loss_val = torch.tensor(0.0, device=device)
    spectral_neg_tau_sweep: dict = {}
    spectral_sim_stats: dict | None = None
    t_cross_spectral = 0.0
    t_cross_phase = 0.0
    if cross_patch_z_anchors:
        _t0 = time.perf_counter()
        n_patches = len(cross_patch_z_anchors)
        z_all = torch.cat(cross_patch_z_anchors, dim=0)        # [N_total, D]
        spec_all = torch.cat(cross_patch_spec_features, dim=0) # [N_total, C]

        offsets: list[int] = [0]
        for z in cross_patch_z_anchors:
            offsets.append(offsets[-1] + z.shape[0])

        # --- Positive pairs: chunked mutual kNN ---
        # Processes chunk_size queries at a time — peak memory O(chunk × N_total).
        # Keeps all anchors; no subsampling needed.
        global_pos = pairs_mutual_knn_chunked(
            spec_all,
            cross_patch_anchor_coords,
            offsets,
            k=config.get('positive_k', 16),
            pos_min_spatial=config.get('positive_min_spatial', 4.0),
            chunk_size=config.get('spectral_knn_chunk_size', 128),
        )

        # --- Negative pairs: random cross-patch sampling ---
        # Cross-patch pairs automatically satisfy the spatial distance constraint.
        # We sample randomly rather than computing the full distance matrix;
        # spectral weights handle false negatives (low weight ≈ spectrally similar).
        tau_neg = config.get('spectral_neg_tau', 1.0)
        min_w = config.get('spectral_neg_min_weight', 0.05)
        # Scale neg count by N_total so each anchor gets ~neg_per_anchor negatives on average.
        # (Analogous to spatial InfoNCE which uses spatial_negatives_per_anchor * N.)
        N_total = z_all.shape[0]
        neg_per_anchor = config.get('spectral_neg_per_anchor', 20)
        n_neg = neg_per_anchor * N_total
        n_patch_pairs = n_patches * (n_patches - 1)
        n_per = max(1, n_neg // n_patch_pairs) if n_patch_pairs > 0 else 0

        neg_i_parts, neg_j_parts = [], []
        for pi in range(n_patches):
            for pj in range(n_patches):
                if pi == pj:
                    continue
                is_s, is_e = offsets[pi], offsets[pi + 1]
                js_s, js_e = offsets[pj], offsets[pj + 1]
                neg_i_parts.append(torch.randint(is_s, is_e, (n_per,), device=device))
                neg_j_parts.append(torch.randint(js_s, js_e, (n_per,), device=device))

        global_neg_i = torch.cat(neg_i_parts)
        global_neg_j = torch.cat(neg_j_parts)
        global_neg = torch.stack([global_neg_i, global_neg_j], dim=1)

        # Distances computed only for sampled pairs — O(n_neg × C), not O(N²)
        neg_spec_dist = torch.norm(spec_all[global_neg_i] - spec_all[global_neg_j], dim=1)
        neg_weights = (1.0 - torch.exp(-neg_spec_dist / tau_neg)).clamp(min=min_w, max=1.0)
        if epoch == 0:
            _nsd_cpu = neg_spec_dist.detach().cpu()
            spectral_neg_tau_sweep = {
                t: {
                    'neg_mean': (1.0 - torch.exp(-_nsd_cpu / t)).clamp(min=min_w, max=1.0).mean().item(),
                    'neg_q25':  torch.quantile((1.0 - torch.exp(-_nsd_cpu / t)).clamp(min=min_w, max=1.0), 0.25).item(),
                    'neg_q50':  torch.quantile((1.0 - torch.exp(-_nsd_cpu / t)).clamp(min=min_w, max=1.0), 0.50).item(),
                }
                for t in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
            }
        else:
            spectral_neg_tau_sweep = {}

        global_spectral_loss_val = contrastive_loss(
            z_all, global_pos, global_neg,
            temperature=config.get('temperature', 0.07),
            similarity='l2',
            neg_weights=neg_weights,
        )
        # Kernel-sizing diagnostic: pos/neg similarity in the same -||a-b||^2/D
        # units the softmax sees. After the exp032 projection-head removal the
        # spectral loss runs on raw z_type (D=z_type_dim), so the gap between
        # these means and the spectral temperature reveals whether the exp(sim/T)
        # kernel is saturated (gap/T >> the spatial loss's healthy ~2-3).
        if global_pos.numel() > 0 and global_neg.numel() > 0:
            with torch.no_grad():
                _Dspec = z_all.shape[1]
                _sp = -(z_all[global_pos[:, 0]] - z_all[global_pos[:, 1]]).pow(2).sum(1) / _Dspec
                _sn = -(z_all[global_neg[:, 0]] - z_all[global_neg[:, 1]]).pow(2).sum(1) / _Dspec
                spectral_sim_stats = {
                    'pos_mean': _sp.mean().item(), 'pos_std': _sp.std().item(),
                    'neg_mean': _sn.mean().item(), 'neg_std': _sn.std().item(),
                    'temperature': float(config.get('temperature', 0.07)),
                }
        total_spectral_pos_pairs = global_pos.shape[0]
        total_spectral_neg_pairs = global_neg.shape[0]
        if is_profiling() and device.type == 'cuda':
            torch.cuda.synchronize()
        t_cross_spectral = time.perf_counter() - _t0

    empty_stats = {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0,
                   'q25': 0.0, 'q50': 0.0, 'q75': 0.0}
    empty_phase_stats = {
        'n_anchors': 0, 'n_anchors_surviving': 0,
        'n_candidates': 0, 'n_after_overlap': 0,
        'n_self_pairs': 0, 'n_total_pairs': 0,
        'overlap_mean': 0.0, 'overlap_min': 0,
        'weight_mean': 0.0, 'weight_std': 0.0,
        'dist_mean': 0.0, 'dist_std': 0.0,
        'dist_q25': 0.0, 'dist_q50': 0.0, 'dist_q75': 0.0,
        'dist_min': 0.0, 'dist_max': 0.0,
    }
    empty_phase_loss_stats = {
        'n_pairs_input': 0, 'n_pairs_sufficient_overlap': 0,
        'loss_self': 0.0, 'loss_cross': 0.0,
        'curriculum_w': 0.0,
        # Entropy of reference (p) and learned (q) distributions
        'self_mean_entropy_p': 0.0, 'self_mean_entropy_q': 0.0,
        'self_mean_overlap': 0.0,
        'cross_mean_entropy_p': 0.0, 'cross_mean_entropy_q': 0.0,
        'cross_mean_overlap': 0.0,
        # Distance distributions that tau operates on
        'd_ref_self_mean': 0.0, 'd_ref_self_std': 0.0,
        'd_ref_self_q25': 0.0, 'd_ref_self_q50': 0.0, 'd_ref_self_q75': 0.0,
        'd_ref_cross_mean': 0.0, 'd_ref_cross_std': 0.0,
        'd_ref_cross_q25': 0.0, 'd_ref_cross_q50': 0.0, 'd_ref_cross_q75': 0.0,
    }

    # Aggregate phase pair stats across samples
    def aggregate_phase_stats(stats_list: list[dict]) -> dict:
        if not stats_list:
            return empty_phase_stats
        agg = {}
        for key in empty_phase_stats:
            vals = [s[key] for s in stats_list]
            if key.startswith('n_'):
                agg[key] = sum(vals) / len(vals)  # mean per sample
            else:
                agg[key] = sum(vals) / len(vals)  # mean
        return agg

    def aggregate_phase_loss_stats(stats_list: list[dict]) -> dict:
        if not stats_list:
            return empty_phase_loss_stats
        agg = {}
        for key in empty_phase_loss_stats:
            vals = [s.get(key, 0.0) for s in stats_list]
            agg[key] = sum(vals) / len(vals)
        return agg

    _empty_evt_diag = dict(
        mean_entropy_ref=0.0, mean_entropy_learned=0.0,
        median_d_learned=0.0, n_anchors_valid=0, mean_kl=0.0,
        d_lrn_confused=0.0, d_lrn_noncf=0.0,
        n_confused_pairs=0.0, mean_rank_confused=0.5, eff_n_ref=1.0,
    )
    if n_valid == 0:
        return {
            'loss': 0.0, 'spectral_loss': 0.0, 'spatial_loss': 0.0,
            'phase_loss': 0.0, 'phase_spread_loss': 0.0, 'phase_recovery_disc_loss': 0.0,
            'vcr_loss': 0.0, 'phase_vcr_loss': 0.0,
            'evt_loss': 0.0, 'evt_diag': _empty_evt_diag,
            'n_valid': 0,
            'spectral_pos_pairs': 0, 'spectral_neg_pairs': 0,
            'spatial_pos_pairs': 0, 'spatial_neg_pairs': 0,
            'gate_stats': empty_stats, 'pos_weight_stats': empty_stats,
            'neg_weight_stats': empty_stats,
            'phase_pair_stats': empty_phase_stats,
            'phase_loss_stats': empty_phase_loss_stats,
        }

    # --- Cross-batch phase losses ---
    # Phase neighborhood, spread, and recovery discrimination losses are computed
    # once over all patches in the batch. Spectral reference features are demeaned
    # by a type-local baseline computed via SVD rank reduction + kNN in type space.
    cross_phase_loss_val = torch.tensor(0.0, device=device)
    cross_phase_spread_val = torch.tensor(0.0, device=device)   # retired (Step 5)
    cross_phase_rd_val = torch.tensor(0.0, device=device)        # retired (Step 5)
    cross_phase_leakage_val = torch.tensor(0.0, device=device)   # retired (diagnostic only)
    phase_contrastive_diag = None

    cp_timing: dict[str, float] = {}
    t_cross_phase = 0.0
    if cross_phase_embeddings and curriculum_w > 0.0:
        _t0 = time.perf_counter()
        Z = torch.cat(cross_phase_z_type, dim=0)               # [P, dt]
        z_phase_all = torch.cat(cross_phase_embeddings, dim=0)  # [P, T, zp]
        flow_all = torch.cat(cross_phase_flow, dim=0)           # [P, T, 2C]
        disturbed_all = torch.cat(cross_phase_disturbed, dim=0)  # [P, T]
        P, T_all, zp = z_phase_all.shape

        # Flatten to pixel-times (n,t); z_type broadcast over T.
        zphase_flat = z_phase_all.reshape(-1, zp)
        ztype_flat = Z.unsqueeze(1).expand(-1, T_all, -1).reshape(-1, Z.shape[1])
        flow_flat = flow_all.reshape(-1, flow_all.shape[-1])
        pixel_flat = torch.arange(P, device=device).unsqueeze(1).expand(-1, T_all).reshape(-1)
        dist_flat = disturbed_all.reshape(-1)

        # Filter to disturbed / recovering pixel-times (the mature majority carries
        # no ray signal), then cap for the O(M²) kernel.
        sel = dist_flat.nonzero(as_tuple=True)[0]
        max_samples = phase_config.get('contrastive_max_samples', 2000)
        if sel.numel() > max_samples:
            sel = sel[torch.randperm(sel.numel(), device=device)[:max_samples]]
        cp_timing['cat'] = time.perf_counter() - _t0

        if sel.numel() >= phase_config.get('contrastive_min_samples', 32):
            _t0 = time.perf_counter()
            zt = ztype_flat[sel]                                # standardize z_type
            zt = (zt - zt.mean(0, keepdim=True)) / (zt.std(0, keepdim=True) + 1e-6)
            c_loss, phase_contrastive_diag = type_phase_contrastive_loss(
                z_phase=zphase_flat[sel],
                z_type=zt,
                flow_state=flow_flat[sel],
                pixel_id=pixel_flat[sel],
                tau_phase=phase_config.get('contrastive_tau', 1.0),
                sigma_type=phase_config.get('contrastive_sigma_type', 1.0),
                sigma_flow=phase_config.get('contrastive_sigma_flow', 1.0),
                n_pos=phase_config.get('contrastive_n_pos', 5),
                n_neg=phase_config.get('contrastive_n_neg', 20),
            )
            cross_phase_loss_val = phase_config.get('contrastive_weight', 1.0) * curriculum_w * c_loss
            if phase_contrastive_diag is not None:
                phase_contrastive_diag['n_disturbed'] = int(sel.numel())
            if is_profiling() and device.type == 'cuda':
                torch.cuda.synchronize()
            cp_timing['contrastive_loss'] = time.perf_counter() - _t0

        t_cross_phase = sum(cp_timing.values())

    # Average losses over valid samples in batch.
    # Spectral and phase losses are computed globally (cross-patch) and added on top.
    _scalar = lambda t: t.item() if hasattr(t, 'item') else float(t)
    if training:
        mean_loss = (total_loss / n_valid
                     + spectral_weight * global_spectral_loss_val
                     + cross_phase_loss_val
                     + cross_phase_spread_val
                     + cross_phase_rd_val
                     + cross_phase_leakage_val)
        mean_spectral_loss = spectral_weight * global_spectral_loss_val
    else:
        mean_loss = (total_loss / n_valid
                     + spectral_weight * _scalar(global_spectral_loss_val)
                     + _scalar(cross_phase_loss_val)
                     + _scalar(cross_phase_spread_val)
                     + _scalar(cross_phase_rd_val)
                     + _scalar(cross_phase_leakage_val))
        mean_spectral_loss = spectral_weight * _scalar(global_spectral_loss_val)
    mean_spatial_loss = total_spatial_loss / n_valid
    mean_phase_loss = _scalar(cross_phase_loss_val)
    mean_phase_spread_loss = _scalar(cross_phase_spread_val)
    mean_phase_recovery_disc_loss = _scalar(cross_phase_rd_val)
    mean_phase_leakage_loss = _scalar(cross_phase_leakage_val)
    mean_vcr_loss = total_vcr_loss / n_valid
    mean_phase_vcr_loss = total_phase_vcr_loss / n_valid
    mean_phase_anchor_loss = total_phase_anchor_loss / n_valid
    mean_phase_ou_loss = total_phase_ou_loss / n_valid
    mean_evt_loss = total_evt_loss / n_valid

    if training:
        # Final NaN check before backward
        if not torch.isfinite(mean_loss):
            logger.warning(f"Skipping batch with non-finite mean loss: {mean_loss.item()}")
            return {
                'loss': float('nan'), 'spectral_loss': float('nan'),
                'spatial_loss': float('nan'), 'phase_loss': float('nan'),
                'phase_spread_loss': float('nan'), 'phase_recovery_disc_loss': float('nan'),
                'vcr_loss': float('nan'), 'phase_vcr_loss': float('nan'),
                'evt_loss': float('nan'), 'evt_diag': _empty_evt_diag,
                'n_valid': 0,
                'spectral_pos_pairs': 0, 'spectral_neg_pairs': 0,
                'spatial_pos_pairs': 0, 'spatial_neg_pairs': 0,
                'gate_stats': empty_stats, 'pos_weight_stats': empty_stats,
                'neg_weight_stats': empty_stats,
                'phase_pair_stats': empty_phase_stats,
                'phase_loss_stats': empty_phase_loss_stats,
            }

        # Backward
        _t0 = time.perf_counter()
        mean_loss.backward()

        # Gradient clipping
        if config.get('gradient_clip_enabled', True):
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=config.get('gradient_clip_max_norm', 1.0)
            )

        optimizer.step()
        if is_profiling() and device.type == 'cuda':
            torch.cuda.synchronize()
        t_backward += time.perf_counter() - _t0
        mean_loss = mean_loss.item()
        mean_spectral_loss = mean_spectral_loss.item()
        mean_spatial_loss = float(mean_spatial_loss)
        mean_phase_loss = float(mean_phase_loss)
        mean_phase_spread_loss = float(mean_phase_spread_loss)
        mean_phase_recovery_disc_loss = float(mean_phase_recovery_disc_loss)
        mean_phase_leakage_loss = float(mean_phase_leakage_loss)
        mean_vcr_loss = float(mean_vcr_loss)
        mean_phase_vcr_loss = float(mean_phase_vcr_loss)
        mean_phase_anchor_loss = float(mean_phase_anchor_loss)
        mean_phase_ou_loss = float(mean_phase_ou_loss)
        mean_evt_loss = float(mean_evt_loss) if not hasattr(mean_evt_loss, 'item') else mean_evt_loss.item()

    # Compute distribution statistics for gate values and weights
    def compute_stats(tensors: list[torch.Tensor]) -> dict:
        """Compute summary stats from list of tensors."""
        if not tensors:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0,
                    'q25': 0.0, 'q50': 0.0, 'q75': 0.0}
        combined = torch.cat(tensors)
        # torch.quantile requires < 2^24 elements; subsample if needed
        _MAX_QUANTILE_ELEMS = 2 ** 23
        q_input = (
            combined[torch.randperm(len(combined), device=combined.device)[:_MAX_QUANTILE_ELEMS]]
            if len(combined) > _MAX_QUANTILE_ELEMS
            else combined
        )
        return {
            'mean': combined.mean().item(),
            'std':  combined.std().item(),
            'min':  combined.min().item(),
            'max':  combined.max().item(),
            'q25':  torch.quantile(q_input, 0.25).item(),
            'q50':  torch.quantile(q_input, 0.50).item(),
            'q75':  torch.quantile(q_input, 0.75).item(),
        }

    # FiLM removed — no gamma/beta stats.  RFF-bandwidth coverage diagnostic:
    # mean leverage over the epoch's phase anchors (0 = data-pinned, →1 =
    # prior-dominated; a high value means the readout bandwidth h is leaving many
    # types under-supported).
    film_stats = None
    readout_leverage = (
        torch.stack(all_readout_leverage).mean().item() if all_readout_leverage else 0.0)

    # Type-leakage diagnostics: how much z_type is linearly recoverable from
    # z_phase (mean over T)?  With the type-agnostic encoder this should stay ≈0.
    type_leakage_stats = None
    if all_pre_film_h_mean and all_z_type_at_phase:
        h_cat = torch.cat(all_pre_film_h_mean, dim=0).float()   # [N, zp]
        zt_cat = torch.cat(all_z_type_at_phase, dim=0).float()  # [N, z_type_dim]
        N = h_cat.shape[0]

        # Option 1: Cross-covariance Frobenius norm
        # Demean both to get unbiased cross-covariance
        h_c = h_cat - h_cat.mean(dim=0, keepdim=True)
        zt_c = zt_cat - zt_cat.mean(dim=0, keepdim=True)
        cross_cov = (h_c.T @ zt_c) / (N - 1)  # [zp, z_type_dim]
        cross_cov_frob = cross_cov.pow(2).sum().sqrt().item()

        # Option 2: Ridge regression R² of z_type predicted from h
        # Fit closed-form ridge: W = (h^T h + λI)^{-1} h^T zt
        lam = 1e-3
        A = h_c.T @ h_c + lam * torch.eye(h_c.shape[1], device=h_c.device)
        B = h_c.T @ zt_c
        W = torch.linalg.solve(A, B)  # [zp, z_type_dim]
        pred = h_c @ W               # [N, z_type_dim]
        ss_res = (zt_c - pred).pow(2).sum(dim=0)   # [z_type_dim]
        ss_tot = zt_c.pow(2).sum(dim=0).clamp(min=1e-8)
        r2_per_dim = (1.0 - ss_res / ss_tot)       # [z_type_dim]
        r2_mean = r2_per_dim.mean().item()
        r2_max = r2_per_dim.max().item()

        type_leakage_stats = {
            'cross_cov_frob': cross_cov_frob,
            'r2_mean': r2_mean,
            'r2_max': r2_max,
        }

    # Aggregate EVT diagnostics across samples
    empty_evt_diag = dict(
        mean_entropy_ref=0.0, mean_entropy_learned=0.0,
        median_d_learned=0.0, n_anchors_valid=0, mean_kl=0.0,
        d_lrn_confused=0.0, d_lrn_noncf=0.0,
        n_confused_pairs=0.0, mean_rank_confused=0.5, eff_n_ref=1.0,
    )
    if all_evt_diag:
        evt_diag_agg = {
            k: sum(d.get(k, empty_evt_diag[k]) for d in all_evt_diag) / len(all_evt_diag)
            for k in empty_evt_diag
        }
    else:
        evt_diag_agg = empty_evt_diag

    return {
        'loss': mean_loss,
        'spectral_loss': mean_spectral_loss,
        'spatial_loss': mean_spatial_loss,
        'phase_loss': mean_phase_loss,
        'phase_spread_loss': mean_phase_spread_loss,
        'phase_recovery_disc_loss': mean_phase_recovery_disc_loss if not hasattr(mean_phase_recovery_disc_loss, 'item') else mean_phase_recovery_disc_loss.item(),
        'phase_leakage_loss': mean_phase_leakage_loss,
        'vcr_loss': mean_vcr_loss,
        'phase_vcr_loss': mean_phase_vcr_loss,
        'phase_anchor_loss': mean_phase_anchor_loss,
        'phase_ou_loss': mean_phase_ou_loss,
        'mature_r2_sum': mature_r2_sum,
        'mature_r2_count': mature_r2_count,
        'disturbed_r2_sum': disturbed_r2_sum,
        'disturbed_r2_count': disturbed_r2_count,
        'ou_diag': last_ou_diag,
        'phase_contrastive_diag': phase_contrastive_diag,
        'readout_leverage': readout_leverage,
        'readout_ss_res': readout_ss_res,
        'readout_ss_tot': readout_ss_tot,
        'readout_median_dz': float(model.mature_baseline.median_dz),
        'readout_bandwidth': float(model.mature_baseline.active_bandwidth),
        'evt_loss': mean_evt_loss if not hasattr(mean_evt_loss, 'item') else mean_evt_loss.item(),
        'evt_diag': evt_diag_agg,
        'n_valid': n_valid,
        # Spectral pairs are cross-batch totals (not per-sample); report raw counts.
        'spectral_pos_pairs': total_spectral_pos_pairs,
        'spectral_neg_pairs': total_spectral_neg_pairs,
        'spatial_pos_pairs': total_spatial_pos_pairs // n_valid if n_valid > 0 else 0,
        'spatial_neg_pairs': total_spatial_neg_pairs // n_valid if n_valid > 0 else 0,
        'gate_stats': compute_stats(all_gate_values),
        'pos_weight_stats': compute_stats(all_pos_weights),
        'neg_weight_stats': compute_stats(all_neg_weights),
        'pos_sim_stats': compute_stats(all_pos_sims),
        'neg_sim_stats': compute_stats(all_neg_sims),
        'spectral_sim_stats': spectral_sim_stats,
        'pos_spec_dist_stats': compute_stats(all_pos_spec_dists),
        'neg_spec_dist_stats': compute_stats(all_neg_spec_dists),
        'tau_sweep': {
            t: {
                'pos_mean': torch.cat(tau_sweep_pos[t]).mean().item() if tau_sweep_pos[t] else 0.0,
                'pos_q25':  torch.quantile(torch.cat(tau_sweep_pos[t]), 0.25).item() if tau_sweep_pos[t] else 0.0,
                'pos_q50':  torch.quantile(torch.cat(tau_sweep_pos[t]), 0.50).item() if tau_sweep_pos[t] else 0.0,
                'neg_mean': torch.cat(tau_sweep_neg[t]).mean().item() if tau_sweep_neg[t] else 0.0,
            }
            for t in _TAU_SWEEP
        },
        'phase_pair_stats': aggregate_phase_stats(all_phase_pair_stats),
        'phase_loss_stats': aggregate_phase_loss_stats(all_phase_loss_stats),
        'film_stats': film_stats,
        'type_leakage_stats': type_leakage_stats,
        'timing': {
            'pass1_total':      t_pass1,
            'sample_build':     t_sample_build,
            'feature_build':    t_feature_build,
            'anchor_sample':    t_anchor_sample,
            'spatial_pairs':    t_spatial_pairs,
            'spectral_weights': t_spectral_weights,
            'gpu_forward':      t_gpu_forward,
            'phase_pairs':      t_phase_pairs,
            'phase_forward':    t_phase_forward,
            'loss_compute':     t_loss_compute,
            'temporal_build':   t_temporal_build,
            'pass2_total':      t_pass2,
            'backward':         t_backward,
            'cross_spectral':   t_cross_spectral,
            'cross_phase':      t_cross_phase,
            'cross_phase_detail': cp_timing,
        },
    }


