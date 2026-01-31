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

<table style="width: 100%; table-layout: fixed; border-collapse: collapse;">
  <thead>
    <tr>
      <th style="width: 33.33%; text-align: left; border-bottom: 2px solid #ccc; padding: 10px;">Method</th>
      <th style="width: 33.33%; text-align: left; border-bottom: 2px solid #ccc; padding: 10px;">Paradigm</th>
      <th style="width: 33.33%; text-align: left; border-bottom: 2px solid #ccc; padding: 10px;">Strategy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding: 10px; border-bottom: 1px solid #eee;">Standard Sparse Attention</td>
      <td style="padding: 10px; border-bottom: 1px solid #eee;">Keep-or-drop</td>
      <td style="padding: 10px; border-bottom: 1px solid #eee;">Directly drop non-critical block.</td>
    </tr>
    <tr>
      <td style="padding: 10px; border-bottom: 1px solid #eee;"><strong>Piecewise Sparse Attention</strong></td>
      <td style="padding: 10px; border-bottom: 1px solid #eee;"><strong>Exact-or-approximate</strong></td>
      <td style="padding: 10px; border-bottom: 1px solid #eee;">Maintains exact computation for critical blocks while efficiently approximating the remainder through block-wise Taylor expansion.</td>
    </tr>
  </tbody>
</table>
## News

## Plans
- [ ] Release triton kernel
- [ ] Release inference script
