---
published: false
---
# Manipulating sentiment using Deep NLP Explanations

In my [last post](CITATION_NEEDED) I described a method for feature importance that can be applied to Deep Neural Networks trained to solve NLP tasks. Today I will show how the featue importance and direction information recovered by this method can be used to guide editing movie reviews in a desired sentiment direction (positive or negative)

## Defining the goal

The interpretation of the input gradients provided in the Deep NLP Explanations post claims that there are two components in the feature importance: for each word, there is a magnitude (how "important" this word is) and also a direction with respect to the optimization objective. In the sentiment analysis objective, the positive and negative directions in feature importance correlate with the positive and negative evaluations of the review. 

For instance, given this movie review:

```
Sentiment: 0.10220414400100708+
This is a ridiculous movie and you should never see it 
````

we get the following feature explanations:

````
This (0.202) is (0.569) a (0.262) ridiculous (-3.093) movie (-0.136) and (0.720) you (0.426) should (0.427) never (0.156) see (0.156) it (0.156) . (0.156) 
````

From this explanation we see that the word 'ridiculous' has the largest importance by far, and also that is contributes negatively for the sentiment. 

In this technique, the feature importance is derived from the input gradient. Since the gradient can be thought of as describing a transformation in the word vector representing the input word, can we somehow use this information in order to edit the text in a desired direction? 

Suppose we wanted to transform the above review from negative to positive. Surely, we know that we need to change the word 'ridiculous' to a more positive word. We know that because, knowing English, we recognize that 'ridiculous' is an adjective of negative conotation. Which word should we choose to replace it? We would most certainly choose a positive adjective in order to keep the general syntax of the phrase coherent. 

Can we create a procedure to do that automatically, based on the interpretation of the gradient information? It turns out we can.


