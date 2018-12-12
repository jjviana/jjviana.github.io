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

The Deep Learning term refers to the application of Deep Neural Networks usually trained using stochastic gradient descent. The Neural Networks most frequently applyed to NLP problems are Recurrent Neural Network variants (such as LSTM and GRU) and Convolutional Neural Networks. Although RNNs are a natural fit for NLP problems due to their sequential data processing nature, CNNs are increasingly being adopted in the field as they provide essentially the same level of accuracy at a fraction of the processing cost.




