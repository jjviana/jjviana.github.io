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
*Illustration of the forward and backward passes. Source: http://www.cnblogs.com/daniel-D/archive/2013/06/03/3116278.html*

The gradients of the error w.r.t the input are usually disregarded, as the input is not updated by the backpropagation algorithm as the weights are. However, these gradients still provide some very useful information: **they describe how each signal component in the input would need to change in order to make the network output closer to the label**. The way backpropagation shapes the neural network during training makes it so that "uninteresting" parts of the input signal get less gradient signal than "interesting" ones. Thus, measuring the magnitude of the gradients received by each input feature during a backwards pass w.r.t a specific outpu can be used as a proxy to how important this feature is to the identification of the label under analysis.

![explanation_methods.png]({{site.baseurl}}/_posts/images/explanation_methods.png)
*A compairson of different explanation methods for visual classification. Most of the methods are based on the interpretation of the backpropagation gradients. Source: "Sanity Checks for Saliency Maps" https://arxiv.org/abs/1810.03292*

As the above figure shows, the gradient information by itself seldom generates understandable explanations on images. Some post-processing of these gradients is needed in order to turn them into meaningful salience maps. Nevertheless, they provide the cheapest way to get some explanation about the reasoning of a deep learning model.

## Interpreting gradient information in NLP models

Most of the work developed in feature interpretation for Deep Neural Networks using gradient information has been done in the context of processing images. The focus is usually in identifying regions of an input image most relevant for classification in a particular class. In order to understand how these methods need to be adapted to work with textual input, we need to understan first how natural language is usually represented as input to a Deep Neural Network.

### Word Vector Representation

Natural language documents are sequences of sentences, which in turn are themselves sequences of words that in their turn are sequences of sylabes (n-grams) that can be broken down into individual characteres. A NLP model usually makes some simplification by looking at natural language at a specific zoom level, ignoring the lower-level components. Even though there are NLP models that work at the character or n-gram level, most models work at the word level.  

Once you choose to look at the input word by word, you still need to choose a way to represent each word. There are many ways to do that, but in the Deep Learning community one particular way became the de-facto standard for that task: word vectors.

Since [their first publication in 2013](https://arxiv.org/abs/1301.3781), word vectors have taken the Deep Learning world by storm. This representation models each word as a multi-dimentional feature vector. The vector representation for each word is jointly learnt from a large corpus of text in an unsupervised way by a neural network solving an auxiliary prediction task such as phrase completion. After training the original neural network is discarded and only the internal representation it has learnt for each word is kept. It has been demonstrated that this representation contains syntatic and semantic information about each word.

![linear-relationships.png]({{site.baseurl}}/_posts/images/linear-relationships.png)
*Word vectors capture semantic information. This can be demonstrated by using them to solve analogy tasks such as: King-Man+Woman=Queen. Source: https://www.tensorflow.org/images/linear-relationships.png*

Word2Vec is the origial word vector learning technique. There are currently other learning techniques such as [Glove](https://nlp.stanford.edu/projects/glove/) and [FastText](https://fasttext.cc/). All these techniques learn semantically signifficand multi-dimensional word embeddings and therefore are considered equivalent from the point of view of this post.

## Phrase-level representation with word vectors

In NLP one is usually not interested on interpreting individual words. The units of interest are phrases, parahraphs and documents. Therefore, it is neccessary to find a way of combining word vectors into higher-level representations. Many different techniques for solving this particular problem are under development. In this post, we will focus on the most common one which is simply word vector concatenation.

In this technique the phrase representation is constructed by concatenating the word vector representing each word. Word vectors are usually high-dimensional, usually having 300 dimentions or more. We can imagine the word vectors as being piled up on top of one another forming a structure aking to an image:

![word_vector_concatenation.png]({{site.baseurl}}/_posts/images/word_vector_concatenation.png)
*An example of a phrase represented as a concatenation of word vectors*

Assuming our word vectors have 300 dimentions, the phrase "I hate this film" can be represented as an image of 300x4 "pixels", where each line of the image correspond to a word and each column to a word vector feature. 


