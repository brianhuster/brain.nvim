# nvim-brain

This is a Lua implementation of the neural network described in the book "Neural Network and Deep Learning" by Michael Nielsen. The code is designed to be used with Neovim. The neural network is trained on MNIST dataset (https://github.com/Eventual-Inc/mnist-json)

To train and evaluate a MNIST model, run `nvim -l mnist.lua`. The trained weights will be saved to `mnist_weights.mpk` (a MessagePack file).
