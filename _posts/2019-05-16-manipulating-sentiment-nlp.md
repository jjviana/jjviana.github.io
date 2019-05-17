---
published: true
title: Manipulating sentiment in NLP
---
In my [last post](https://jjviana.github.io/Explaining-Deep-NLP-Models/) I described a method for feature importance that can be applied to Deep Neural Networks trained to solve NLP tasks. Today I will show how the feature importance and direction information recovered by this method can be used to guide editing movie reviews in the desired sentiment direction (positive or negative).

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

From this explanation we see that the word 'ridiculous' has the largest importance by far, and also that contributes negatively to the sentiment. 

In this technique, the feature importance is derived from the input gradient. Since the gradient can be thought of as describing a transformation in the word vector representing the input word, can we somehow use this information in order to edit the text in the desired direction? 

Suppose we wanted to transform the above review from negative to positive. Surely, we know that we need to change the word 'ridiculous' to a more positive word. We know that because, knowing English, we recognize that 'ridiculous' is an adjective of negative connotation. Which word should we choose to replace it? We would most certainly choose a positive adjective in order to keep the general syntax of the phrase coherent. 

Can we create a procedure to do that automatically, based on the interpretation of the gradient information? It turns out we can.


## The input gradient as a transformation

The backpropagation algorithm used to train Deep Neural Networks relies on gradient information in order to update the neural network parameters. From the point of view of this algorithm, there is nothing special about the neural network input. In fact, backpropagation makes no distinction between the neurons that represent the input data and the neurons of the other layers. The only distinction exists in practice when parameter updates happen: neuron parameters are updated in the direction provided by the gradients, but the input parameters are not. That makes sense: gradient descent can act to change the interpretation of the reality (network weights), but not reality itself.

Now, what would that mean if we apply gradient descent to the input data? The gradient optimization step of SGD requires us to update parameters as follows:

````
param = param - lr * grad
````

Where ``lr`` is a learning rate constant and ``grad`` is the gradient information computed in the backward pass.

Since the input space of these neural networks consists of word vectors, this transformation can be seen as moving the word in the word vector space in a specific direction. 

![ridiculous_gradient.png]({{site.baseurl}}/post-images/ridiculous_gradient.png)
*Visualization courtesy of projector.tensorflow.org*

The above illustration shows a 2D projection of some word vectors in the vicinity of the word 'ridiculous'. The original space has 300 dimensions, so this projection is necessarily an approximation. The blue arrow illustrates the direction indicated by the input gradient for the word 'ridiculous'. It is clear that in this case following the gradient direction will approximate this word vector to the representation of more positive adjectives.

The question that remains is how to choose a suitable word to replace this negative adjective since in the real world words are not continuous vectors but discrete features. One possible solution is to find the nearest neighbor of the original word when moving in the direction indicated by the input gradient.

Putting it all together, we have the following algorithm:

```
given: phrase to improve (f)
       neural network that classifies f (m)
       desired classification target (t)
returns: a new phrase (f') constructed by replacing a single
         word from f so that f' is closer to the target than f
         
1. let grads =  gradients for each word in f with regard to target t
2. let w = the word in f that has the largest importance magnitude in the negative direction
3. let wvec= word vector representation of w
4. let wgrad = gradient of word w with regard to target t
3. Repeat:
   3.1 let new_vec= wvec - wgrad
   3.2 let new_word = nearest neighbor of new_vec in the vector space
   3.3 if new_word is different from w:
       3.4 exit repeat
4. let f' = f with the replacement of w by new_word
5. return f'

```

This algorithm is implemented in my [original collaboratory notebook](https://colab.research.google.com/drive/1w2gqazjaS2HGSjxLp3vWvSy3FwKGbAmj), in the function ``predict_and_make_it_better``

In theory, each time a word is replaced according to the above algorithm one would end up with a new phrase that scores better than before in the desired direction. How well does that work in practice?

You can play with this algorithm by running the [notebook](https://colab.research.google.com/drive/1w2gqazjaS2HGSjxLp3vWvSy3FwKGbAmj). Below are some demonstrations of this algorithm in action:

### Small phrase, neutral to positive

<iframe width="806" height="348" src="https://www.youtube.com/embed/pnSy7pnFY0E" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

### Small phrase, neutral to negative

<iframe width="806" height="340" src="https://www.youtube.com/embed/g-4wFIjqZbw" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

### Complete movie review, positive to negative

<iframe width="806" height="344" src="https://www.youtube.com/embed/VvesHwGdL80" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

### Complete movie review, negative to positive

<iframe width="806" height="336" src="https://www.youtube.com/embed/Ur6LfII_lRU" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>


## Future directions

The fact that this sentence transformation heuristic works at all is a further indication of the validity of interpreting input gradients in the way described in my [previous post](https://jjviana.github.io/Explaining-Deep-NLP-Models/). 
It is indeed possible to generate gradual word-by-word transformations of a phrase or paragraph of text in the desired direction.

However, if you look closely at the sentences generated you will notice that they are not always syntactically or semantically correct. This is only natural, as the heuristic implemented here does not take into account any such linguistic restrictions. 

Would it be possible to combine this technique with a language model (maybe even the powerful [gpt-2 model](https://github.com/openai/gpt-2) ) in order to generate valid sentences that still represent a transformation in the desired direction? Maybe this will be the theme of one of my future posts :).
