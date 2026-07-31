"""Representation-model training internals.

Modules extracted from the former monolithic ``train_representation.py`` so that
the training step, config plumbing, scheduler, checkpointing, logging, and epoch
loops each live in a focused, testable unit. ``train_representation.py`` remains
the CLI entry point and wires these together.
"""
