<div align ="center">
    <h2>
        PISA: Piecewise Sparse Attention Is Wiser for Efficient Diffusion Transformers
    </h2>
</div>

<!-- <p align="center">
    <a href="https://arxiv.org/abs/2602.01077"><b>Paper</b></a>
</p> -->

<p align="center">
    <img src="assets/PISA_Title_Page.jpg" width=100%>
<p>


## 🔥 News
- [2026/02] [Paper](https://arxiv.org/abs/2602.01077) is released.


## 💡 TL;DR
We propose a training-free Piecewise Sparse Attention (PISA) that covers the full attention span with sub-quadratic complexity.

Unlike the standard ***keep-or-drop*** paradigm that directly drop the non-critical blocks, PISA introduces a novel ***exact-or-approximate*** strategy: it maintains exact computation for critical blocks while efficiently approximating the remainder through block-wise Taylor expansion.


## ⏰ Plans
- [x] Release triton kernel
- [x] Release flux.1-dev inference demo
- [ ] Release wan/hyvideo inference script


## 🔧 Installation
Requirements:
* `torch >= 2.7.1`
* `triton >= 3.5.1`


Install:
```shell
git clone https://github.com/hp-l33/piecewise-sparse-attention.git
cd piecewise-sparse-attention
pip install -e .
```

Note: Our kernels are currently primarily optimized for the NVIDIA Hopper architecture (e.g., H100, H800).

## 🎮 Quick Start
### Text-to-Image Generation
```diff
from diffusers import AutoPipelineForText2Image
+ from pisa.models.flux.flux_processor import FluxAttnProcessor, set_processor
+ from pisa.kernels.piecewise_sparse_attn_tma import piecewise_sparse_attention

pipeline = AutoPipelineForText2Image.from_pretrained(
    "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16,
).to("cuda")

+ pipeline = set_processor(pipeline, piecewise_sparse_attention, density=0.15)

prompt = "A portrait of a human growing colorful flowers from her hair. Hyperrealistic oil painting. Intricate details."
image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=50,
    max_sequence_length=512,
).images[0]
```


<!-- ## 🔗 BibTeX
If this work is helpful for your research, please give it a star or cite it:
```bibtex

``` -->