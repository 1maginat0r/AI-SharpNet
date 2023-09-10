Welcome to AI-SharpNet, a state-of-the-art Open-source Deep Learning library designed in C# 10.0.

Features:
 - Residual Networks [v1](https://arxiv.org/pdf/1512.03385.pdf), [v2](https://arxiv.org/pdf/1603.05027.pdf), [WideResNet](https://arxiv.org/pdf/1605.07146.pdf) and [EfficientNet](https://arxiv.org/pdf/1905.11946.pdf)
 - [DenseNet](https://arxiv.org/pdf/1608.06993.pdf)
 - Broad range of layers including BatchNorm, Conv1D, Conv2D, Dense, Dropout, Embedding, GRU, LSTM, Pooling, RNN, Shortcut, and SimpleRNN layers
 - Variety of activations like Elu, Relu, Leaky Relu, Sigmoid, Softmax, Swish, and Tanh
 - SGD & Adam optimizers
 - Image Data Augmentation (with Cutout/CutMix/Mixup)
 - Ensemble Learning

AI-SharpNet can run on both GPU (using NVIDIA cuDNN) and CPU (using MKL Blas) making the most out of your hardware.

Performance-wise, it is between 1.5x (batch size = 128) and 3x time (batch size = 32) faster than TensorFlow 1.x when running on ResNet18 v1.

Dependencies:
- [Visual Studio 2022](https://visualstudio.microsoft.com/downloads/)
- [DotNetCore 6.0](https://dotnet.microsoft.com/download/dotnet-core/6.0)
- [CUDA Toolkit 10.1,  10.2 or 11.0](https://developer.nvidia.com/cuda-downloads)
- [CuDNN 8.0](https://developer.nvidia.com/rdp/cudnn-download)
- [Intel MKL](https://software.intel.com/en-us/mkl)

For further inquiries and contribution guidelines, contact 1maginat0r.