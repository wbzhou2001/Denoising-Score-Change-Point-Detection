# DSM-CUSUM
The official code of the paper [Sequential Change Point Detection via Denoising Score Matching](https://arxiv.org/abs/2501.12667). 

## Abstract

Sequential change-point detection plays a critical role in numerous real-world applications, where timely identification of distributional shifts can greatly mitigate adverse outcomes. Classical methods commonly rely on parametric density assumptions of pre- and post-change distributions, limiting their effectiveness for high-dimensional, complex data streams. This paper proposes a score-based CUSUM change-point detection, in which the score functions of the data distribution are estimated by injecting noise and applying denoising score matching. We consider both offline and online versions of score estimation. Through theoretical analysis, we demonstrate that denoising score matching can enhance detection power by effectively controlling the injected noise scale. Finally, we validate the practical efficacy of our method through numerical experiments on two synthetic datasets and a real-world earthquake precursor detection task, demonstrating its effectiveness in challenging scenarios.
