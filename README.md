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

| Method | Paradigm | Strategy |
| :--- | :--- | :--- |
| Standard Sparse Attention | Keep-or-drop | Directly drop non-critical block. |
| **Piecewise Sparse Attention** | **Exact-or-approximate** | Maintains exact computation for critical blocks while efficiently approximating the remainder through block-wise Taylor expansion. |

## News

## Plans
- [ ] Release triton kernel
- [ ] Release inference script
