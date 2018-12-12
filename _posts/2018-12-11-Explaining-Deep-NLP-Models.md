---
layout: post
title: Explaining Deep NLP Models
published: true
---

In this post, I will describe a technique I have been using at [Kunumi](https://kunumi.ai) to explain the output of Deep Neural Networks trained to solve NLP tasks.

Some of the diagrams and part of the Jupyter Notebook examples are based on the excelent set of tutorials by Ben Trevett ( https://github.com/bentrevett/pytorch-sentiment-analysis ), released under the MIT license.

## Deep Learnng applied to Natural Language Processing

Over the last 5 years [Deep Learning](https://www.cs.toronto.edu/~hinton/absps/NatureDeepReview.pdf) has become the essential ingredient in [state-of-the-art NLP models](https://www.stateoftheart.ai/?area=Natural%20Language%20Processing). 

[Natural Language Processing (NLP)](https://en.wikipedia.org/wiki/Natural_language_processing) is "... is a subfield of computer science, information engineering, and artificial intelligence concerned with the interactions between computers and human (natural) languages..." . In practice, NLP today focuses on creating Machine Learning models capable of understanding specific aspects of text such as sentiment analysis, part-of-speech tagging, named entity recognition, language translation, etc. 

The Deep Learning term refers to the application of Deep Neural Networks usually trained using stochastic gradient descent. The Neural Networks most frequently applyed to NLP problems are [Recurrent Neural Network](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) variants (such as LSTM and GRU) and [Convolutional Neural Networks](http://cs231n.github.io/convolutional-networks/). Although RNNs are a natural fit for NLP problems due to their sequential data processing nature, CNNs are increasingly being adopted in the field as they provide essentially the same level of accuracy at a fraction of the processing cost.

## Interpreting Deep Neural Network outputs

Model interpretation has started to receive a lot of interest over the last couple of years because as models transition from prototyping to production it becomes important to understand and justify the probability distributions generated as the output of these models.

When talking about model interpretation one usually means to obtain some kind of "explanation" for a particular model output. This explanation is usually framed in terms of the importante of the input features: one would like to see some "evidence" in the input features the model predictions.

Neural Networks have been traditionally seen as uninterpretable black-boxes by the Machine Learning community. However, many new techniques have been developed in order to try and understand the information processed by these models. Techniques such as [gradients](CITATION NEEDED), [LIME](https://www.oreilly.com/learning/introduction-to-local-interpretable-model-agnostic-explanations-lime), [GuidedBackProp](CITATION NEEDED) and [LRP](CITATION NEEDED), among others, provide ways to identify relevant input features by analyzing how information propagates inside Neural Networks. 

The gradients technique will be the focus of this post. This technique uses the information contained in the gradients generated during the backwards pass of the backpropagation algorithm as a proxy for feature importance. [Even though it does not always yield the most interpretable explanations](CITATION NEEDED) it is straightforward to implement and computationally unexpensive when compared to some of the other methods. 

## Gradient-based model explanations

The backpropagation algorithm used to train modern neural networks has two phases: forward and backward. In the forward phase, the input signal is propagated through the layers of the network until an output layer is reached. A loss function is then used to compute the error between the network output and the ground truth (label).

In supervised learning, this loss function is differentiable and therefore can produce a [gradient](https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives/partial-derivative-and-gradient-articles/a/the-gradient), which can be used to correct the weights of the last layer of the neural network in order to make the output closer to the ground truth. Since every signal transformation performed by the neural network during the forward pass is also differentiable, the gradient of the output with regrd to the error can be used to compute a gradient for the last layer, then of the penultimate layer, then of the layer before that and so on until the input signal. These gradients are then used to make a small update to every weight of the neural network.
![backpropagation.gif]({{site.baseurl}}/_posts/images/backpropagation.gif)
*Figure 1: illustration of the forward and backward passes. Source: http://www.cnblogs.com/daniel-D/archive/2013/06/03/3116278.html*







