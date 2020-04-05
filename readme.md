### Advantage Actor Critic with CuLE

Train atari games with A2C with CULE, wich is CUDA Learning Environment.A deeper neural network architecture from [1] is used. It has convolution layer followed by
max pooling layer and two residual blocks. In addition, we added batch normalization layer between two convolution layers in the residual block. This repeats for 16,32,32 channels
and at the end is fully connected layer.

[1] https://arxiv.org/abs/1802.01561 (IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures)

Trained  Pong game with 1152 environments on V100 GPU and reached average score of 19.8 out of 21 after 40 minutes of tranining and for 4 GPUs
it took 15 minutes.
