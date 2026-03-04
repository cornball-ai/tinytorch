# Rtorch TODO

## High Priority

- [ ] **linalg/fft/special namespaces** — Codegen extension for `torch.linalg.*`, `torch.fft.*`, `torch.special.*`
- [ ] **Quantization APIs** — `quantize_per_tensor`, `dequantize`, QScheme bindings for consumer inference
- [ ] **DataLoader / Dataset** — Pure R abstractions for batching, shuffling, iteration

## Medium Priority

- [ ] **LR schedulers** — Pure R (StepLR, CosineAnnealing, etc.)
- [ ] **Model save/load** — `torch::serialize` / `torch::load` bindings for model persistence

## Low Priority

- [ ] **Sparse tensor APIs** — Codegen for sparse ops (COO, CSR)
- [ ] **Custom autograd functions** — `torch::autograd::Function` subclassing from R
- [ ] **Distributed** — Multi-GPU/multi-node (NCCL, Gloo) — not consumer-facing
