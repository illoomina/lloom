# LFM2 MoE metadata fixture (no weights)

`config.json` is the unmodified config of `LiquidAI/LFM2.5-24B-A2B`
at revision `1769516ac588c4c558089c94f27d96b0dc67b183`.
SHA-256: `b0c46662afe3881e49cf542e683bf5ea8a5e0b780a429b6141a48defc33531bf`.

`tensor-names.txt` is the sorted `weight_map` key set from that revision's
`model.safetensors.index.json` (7,610 keys). Original index SHA-256:
`b3c3d025ee3b7e78ca3d6160d1be872d85ff03272c64b0c2b5fc85a6f36c622f`.
Its `metadata.total_size` is 47,943,180,288 bytes; this is index metadata,
not a measured resident/peak footprint and not a shape or dtype header audit.

Primary references:
- https://huggingface.co/LiquidAI/LFM2.5-24B-A2B/blob/1769516ac588c4c558089c94f27d96b0dc67b183/config.json
- https://huggingface.co/LiquidAI/LFM2.5-24B-A2B/blob/1769516ac588c4c558089c94f27d96b0dc67b183/model.safetensors.index.json
- https://github.com/huggingface/transformers/blob/276d3fac6973c0e5c179e665af662d81a7df6c98/src/transformers/models/lfm2_moe/modular_lfm2_moe.py
- https://github.com/huggingface/transformers/blob/276d3fac6973c0e5c179e665af662d81a7df6c98/src/transformers/models/lfm2_moe/modeling_lfm2_moe.py

These files test adapter/config/schema provenance only. Scalar value tests and
mixed-model state/file/consumer tests live in `../../lfm2_moe.rs`; neither
metadata nor synthetic values establish support for the real checkpoint.
