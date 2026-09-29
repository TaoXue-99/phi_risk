"""Editable ESMM defaults, available without any training dependency."""


def baseline_template():
    return """# Resolve input dimensions from the TRAIN-fitted feature schema before start_run.
# This file is configuration data; construct Python objects explicitly.
model:
  num_continuous: null
  categorical_cardinalities: []
  embedding_dim: 4
  share_dim: 64
  base_dim: 32
  expert_units: 16
  num_experts: 4
  dropout: 0.1
  batch_norm: false
  normalize_shared: false
  use_expert_bias: true
  use_gate_bias: true
  expert_activation: elu
loss:
  ctr_weight: 1.0
  ctcvr_weight: 1.0
optimizer:
  name: Adam
  lr: 0.001
  weight_decay: 0.0
scheduler:
  name: StepLR
  step_size: 10
  gamma: 0.5
  interval: epoch
data:
  continuous_features: []
  categorical_features: []
  labels:
    ctr: y_ctr
    ctcvr: y_ctcvr
  batch_size: 128
  num_workers: 0
  drop_last: false
train:
  epochs: 30
  seed: 2026
  monitor: val_loss
  mode: min
  patience: 5
  max_grad_norm: 1.0
torchkeras:
  plot: false
  quiet: true
  cpu: true
  mixed_precision: 'no'
  gradient_accumulation_steps: 1
"""
