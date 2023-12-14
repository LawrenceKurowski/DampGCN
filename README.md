# DampGCN
This was the basis of my Master thesis at Tsinghua University.

# Abstract
The theory of Ordinary Differential Equations (ODEs) has long inspired constructing more explainable Deep Learning models. Starting from the ResNet model which has been seen as akin in its form to a simple first order ODE, many successful attempts have been made in recent years at connecting the mathematically well-understood theory of Differential Equations with Deep Neural Networks (DNNs).

In this work we propose and evaluate a novel ODE-inspired deep model, termed DampGCN, which we derive from the widely used Graph Convolutional Network model. We show that both the baseline GCN, and our proposed DampGCN models are in fact special cases of a more general framework. We then demonstrate that the DampGCN
model is provably more robust to adversarial attacks.

The proposed network is theoretically motivated by the Lyapunov analysis of the ODE
model, while the output analysis computationally validates the improved robustness. Extensive experiments reveal that adding the damping term significantly improves robustness of the GCN in response to several adversarial attack methods targeting graph node features. In addition, the proposed model improves robustness to sophisticated structure-only attack methods, including the Fast-Gradient Attack method.
Moreover, our model is robust to both training and testing time attacks, providing a
simple but surprisingly generalisable defense mechanism for GCN models.

# How this repo is structured
* `data` contains the citation network datasets we used.
* `models` are the different adverssarial attack models.
* `tests` contains code for testing models on the data.

# References
This repo is building up on the [DeepRobust](https://github.com/DSE-MSU/DeepRobust) repository.

