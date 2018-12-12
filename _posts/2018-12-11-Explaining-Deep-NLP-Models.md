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







