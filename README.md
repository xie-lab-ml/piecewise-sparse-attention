<!-- # PISA: Piecewise Sparse Attention Is Wiser for Efficient Diffusion Transformers
 -->
<div align ="center">
<h2>PISA: Piecewise Sparse Attention Is Wiser for Efficient Diffusion Transformers</h2>
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



## Plans
- [x] Release triton kernel
- [x] Release flux.1-dev inference demo
- [ ] Release wan/hyvideo inference script


<!-- ## 🔗 BibTeX
```bibtex

``` -->