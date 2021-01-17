# ALPS_2021
Repository for the Explainable AI track in the ALPS winter school 2021

Please, keep in mind that changes are possible to the notebooks and/or source code in the few upcoming days.

# Lab 1
The first part of the lab is on the topic of explainability.

You can find the corresponding notebook on [Colab](https://colab.research.google.com/drive/1M1-O8FnSARfE8nS35A_s3RQEyCUKdKNi?usp=sharing).

The solutions for the tasks can be found in [this notebook](https://colab.research.google.com/drive/1-aZ9-Kzkb_BVb-8vcvHBAAYy2iBk0khV?usp=sharing).

# Lab 2
The second lab focuses explorative interpretability via acivation maximization - i.e. TX-Ray https://arxiv.org/abs/1912.00982. Activation maximization works for supervised and *self/un-supervised* settings alike, but the lab focuses analyzing CNN filters in a simple supervised setting.
## Lab code
[CoLAB2](https://colab.research.google.com/drive/1lg6Xj9QM33lekmSkdzjxHIBlGnexBTY3?usp=sharing) <- copy this Colab notebook and add code to it for the exercises.
There are two types of exercises:

<font color='Gold'>**Familiarization exercise:**</font> to 'play with and understand' the technique. These allow quickly changes data collection and visualization parameters. They are intended for explorative analysis.

<font color='orange'>**Advanced Exercises:**</font>, these are optional and concern applications of the technique. They have no solution, but give solution outlines (starter code). Opt-EX1: XAI based pruning with hooks, Opt-EX2 influence of overparameterization (wider CNN with more filters), Opt-Ex3: filter redundancy. Opt-Ex2-3 belong together
## <font color='LawnGreen'>Goals of LAB2</font>:
+ learn how to explore and interpret activations in NNs using 'activation maximization' principles
+ learn how to extract activations via forward_hooks
+ exercise how to usefully interpret and visualize activation behaviour
+ exercise how to prune activations -- advanced
+ analyze neuron/ filter redundancy, specialization, generalization - advanced
+ Overall: explore/ develop ideas towards 'model understanding' -- see https://arxiv.org/abs/1907.10739 for a great introduction of 'decision understanding' vs. 'model understanding'
	+ this tutorial focuses on explorative 'model understanding' via TX-Ray https://arxiv.org/abs/1912.00982
