This repository includes a Keras callback which can be used to find an optimal learning rate for a Keras model,as described in Leslie Smith's paper: https://arxiv.org/abs/1506.01186

Choosing the right learning rate for a deep network can be tricky. A low learning rate may take very long to converge against an optimal solution, while a higher learning rate quickly converges, but may never find the best solution.

The fast.ai library for pytorch offers a Learning Rate Finder to quickly find a good learning rate.
In Keras, a similar solution can be realised by using a callback.

The callback can be used with any Keras Models and increases the learning rate while training the model.
The learning rate which yields the minimal training loss is supposed to perform well in training.