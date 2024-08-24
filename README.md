# ResNet20 recreation with PyTorch

This repo is designed to be used on a google collab instance.

You can play with the hyperparameters in the `train.py` file. 
According to the original ResNet paper [1][1], he author discusses the case of CIFAR10 in section 4.2 CIFAR-10 and Analysis, providing the following table:

$$\begin{table}[h]
\centering
\begin{tabular}{|c|c|c|c|}
\hline
\textbf{output map size} & \textbf{32$\times$32} & \textbf{16$\times$16} & \textbf{8$\times$8} \\ \hline
\# layers & $1 + 2n$ & $2n$ & $2n$ \\ \hline
\# filters & 16 & 32 & 64 \\ \hline
\end{tabular}
\caption{The ResNet parameters for the CIFAR10 according to the researchers}
\end{table}$$

## Bibliography
[1] : Kaiming He et al. *Deep Residual Learning for Image Recognition.* 2015. arXiv: 1512. 03385 [cs.CV](https://arxiv.org/abs/1512.03385).
