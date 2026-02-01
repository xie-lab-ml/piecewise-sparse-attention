<!-- # PISA: Piecewise Sparse Attention Is Wiser for Efficient Diffusion Transformers
 -->
<div align ="center">
<h1>PISA: Piecewise Sparse Attention Is Wiser for Efficient Diffusion Transformers</h3>
</div>

<p align="center">
<img src="assets/PISA_Title_Page.jpg" width=95%>
<p>

## TL;DR
We propose a training-free Piecewise Sparse Attention (PISA) that covers the full attention span with sub-quadratic complexity.

Unlike the standard ***keep-or-drop*** paradigm that directly drop the non-critical blocks, PISA introduces a novel ***exact-or-approximate*** strategy: it maintains exact computation for critical blocks while efficiently approximating the remainder through block-wise Taylor expansion.

## News
- [2026/02/02] Our paper is released.

## Plans
We currently only provide a minimal implementation for the community to understand and modify. We will release inference scripts for video and image generation in the coming weeks.

- [x] Release triton kernel
- [ ] Release inference script
