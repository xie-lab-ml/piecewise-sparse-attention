<!-- # PISA: Piecewise Sparse Attention Is Wiser for Efficient Diffusion Transformers
 -->
<div align ="center">
<h1>PISA: Piecewise Sparse Attention Is Wiser for Efficient Diffusion Transformers</h3>
</div>

<p align="center">
<img src="title.png" width=95%>
<p>

## TL;DR
We propose a training-free Piecewise Sparse Attention (PISA) that covers the full attention span with sub-quadratic complexity.

Unlike the standard ***keep-or-drop*** paradigm that directly drop the non-critical blocks, PISA introduces a novel ***exact-or-approximate*** strategy: it maintains exact computation for critical blocks while efficiently approximating the remainder through block-wise Taylor expansion.

## News

## Plans
- [ ] Release triton kernel
- [ ] Release inference script
