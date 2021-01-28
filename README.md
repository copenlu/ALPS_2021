# ALPS_2021 - LAB 2: XAI in NLP - January 22 2021
Repository for the Explainable AI track in the ALPS winter school 2021 - [schedule](http://lig-alps.imag.fr/index.php/schedule/).

This lab consists of two parts - one on explainability and one on explorative interpretability.
You should try to split your time between the two labs.

# Lab 2.1
The first part of the lab focuses on explainability for Natural Language Processing Models.
In this part, we will lay the foundations of post-hoc explainability techniques and ways of evaluating them.

## Lab 2.1 code
[CoLAB](https://colab.research.google.com/drive/14pIyfxlX9D9js966kjHoBFmNXnRd1-1T?usp=sharing) <- copy this Colab notebook and add code to it for the exercises.
[CoLAB Solutions](https://colab.research.google.com/drive/1-aZ9-Kzkb_BVb-8vcvHBAAYy2iBk0khV?usp=sharing)

For this notebook of the lab, we encourage you to work in groups, so that you could split the work and discuss the outcomes.

### <font color='LawnGreen'>Goals of LAB 2.1</font>:
+ learn how to implement two basic and commonly used types of gradient-based explainability techniques
+ learn how to implement an approximation-based explainability technique
+ exercise how to apply explainability techniques to discover flaws of machine learning models and construct adversarial explamples using them
+ learn how to evaluate explainability techniques with common diagnostic properties [(based on this paper)](https://www.aclweb.org/anthology/2020.emnlp-main.263.pdf)
+ exercise using the diagnostic properties to find which architecture parameters of a model make it harder to explain

If you find this code useful for your research, please consider citing:

```
@inproceedings{atanasova2020diagnostic,
title={A Diagnostic Study of Explainability Techniques for Text Classification},
author={Pepa Atanasova and Jakob Grue Simonsen and Christina Lioma and Isabelle Augenstein},
booktitle = {Proceedings of EMNLP},
publisher = {Association for Computational Linguistics},
year = 2020
}
```

# Lab 2.2
The second lab focuses explorative interpretability via acivation maximization - i.e. TX-Ray https://arxiv.org/abs/1912.00982. Activation maximization works for supervised and *self/un-supervised* settings alike, but the lab focuses analyzing CNN filters in a simple supervised setting.
## Lab 2.2 code
[CoLAB2](https://colab.research.google.com/drive/1lg6Xj9QM33lekmSkdzjxHIBlGnexBTY3?usp=sharing) <- copy this Colab notebook and add code to it for the exercises.
There are two types of exercises:

<font color='Gold'>**Familiarization exercise:**</font> to 'play with and understand' the technique. These allow quickly changes data collection and visualization parameters. They are intended for explorative analysis.

<font color='orange'>**Advanced Exercises:**</font> these are optional and concern applications of the technique. They have no solution, but give solution outlines (starter code). Opt-EX1: XAI based pruning with hooks, Opt-EX2 influence of overparameterization (wider CNN with more filters), Opt-Ex3: filter redundancy. Opt-Ex2-3 belong together
### <font color='LawnGreen'>Goals of LAB 2.2</font>:
+ learn how to explore and interpret activations in NNs using 'activation maximization' principles
+ learn how to extract activations via forward_hooks
+ exercise how to usefully interpret and visualize activation behaviour
+ exercise how to prune activations -- advanced
+ analyze neuron/ filter redundancy, specialization, generalization - advanced
+ Overall: explore/ develop ideas towards 'model understanding' -- see https://arxiv.org/abs/1907.10739 for a great introduction of 'decision understanding' vs. 'model understanding'
	+ this tutorial focuses on explorative 'model understanding' via TX-Ray https://arxiv.org/abs/1912.00982

If you find this code useful for your research, please consider citing:

```
@inproceedings{Rethmeier19TX-Ray,
title = {TX-Ray: Quantifying and Explaining Model-Knowledge Transfer in (Un-)Supervised NLP},
author = {Rethmeier, Nils and Kumar Saxena, Vageesh and Augenstein, Isabelle},
booktitle = {Proceedings of the Conference on Uncertainty in Artificial Intelligence (UAI)},
year = 2020
}
```
