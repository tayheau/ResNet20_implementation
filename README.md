# ResNet20 recreation with PyTorch

This repo is designed to be used on a google collab instance.

You can play with the hyperparameters in the `train.py` file. 
According to the original ResNet paper [1][1], he author discusses the case of CIFAR10 in section 4.2 CIFAR-10 and Analysis, providing the following table:

$$
\begin{array}{|c|c|c|c|}
\hline
\textbf{output map size} & \textbf{32 \times 32} & \textbf{16 \times 16} & \textbf{8 \times 8} \\ \hline
\# \text{layers} & 1 + 2n & 2n & 2n \\ \hline
\# \text{filters} & 16 & 32 & 64 \\ \hline
\end{array}
$$

When $a \ne 0$, there are two solutions to $(ax^2 + bx + c = 0)$ and they are
$$ x = {-b \pm \sqrt{b^2-4ac} \over 2a} $$

## Bibliography
[1] : Kaiming He et al. *Deep Residual Learning for Image Recognition.* 2015. arXiv: 1512. 03385 [cs.CV](https://arxiv.org/abs/1512.03385).
