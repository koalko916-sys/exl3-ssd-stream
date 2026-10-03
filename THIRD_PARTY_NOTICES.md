# Third-party notices

This repository distributes the adapter and its tests, not model weights or
ExLlamaV3/CUDA binaries.

- **ExLlamaV3**: https://github.com/turboderp-org/exllamav3, MIT,
  Copyright (c) 2025 Turboderp. Its license is reproduced in
  `docs/EXLLAMAV3-LICENSE.txt`. The adapter imports its architecture and kernels.
- **WARP**: https://github.com/sqliteai/warp, Apache-2.0. Inspiration for the
  larger-than-RAM experiment. No WARP C source or container format is included.
- **PyTorch**, **Transformers** and **psutil** are installed separately and retain
  their upstream licenses. The Windows installer downloads official distributions.
- **GLM model weights** and quantization are separate artifacts. The verified
  checkpoint is https://huggingface.co/Infatoshi/GLM-5.3-UNCENSORED-EXL3-3.0bpw,
  derived from dealignai and Z.ai releases; consult that model card and its upstream
  terms. This adapter's license does not replace model or dependency licenses.

No affiliation with or endorsement by these projects or authors is claimed.
