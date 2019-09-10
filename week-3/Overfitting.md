# The problem of Overfitting

### Underfitting Vs Overfitting  

- Underfitting, or high bias, is when the form of our hypothesis function h maps poorly to the trend of the data. It is usually caused by a function that is too simple or uses too few features. <br>
- At the other extreme, overfitting, or high variance, is caused by a hypothesis function that fits the available data but does not generalize well to predict new data. It is usually caused by a complicated function that creates a lot of unnecessary curves and angles unrelated to the data.<br>


There are two main options to address the issue of overfitting:

1) Reduce the number of features:

-   Manually select which features to keep.
-   Use a model selection algorithm (studied later in the course).

2) Regularization

-   Keep all the features, but reduce the magnitude of parameters  θj​.
-   Regularization works well when we have a lot of slightly useful features.

## Regularization Cost Function

We could also regularize all of our theta parameters in a single summation as:<br>
<a href="https://www.codecogs.com/eqnedit.php?latex=\min_\theta&space;\sum_{i=1}^m&space;(h_\theta(x^{(i)})&space;-&space;y^{(i)})^2&space;&plus;&space;\lambda&space;\sum_{j=1}^n&space;\theta_j^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\min_\theta&space;\sum_{i=1}^m&space;(h_\theta(x^{(i)})&space;-&space;y^{(i)})^2&space;&plus;&space;\lambda&space;\sum_{j=1}^n&space;\theta_j^2" title="\min_\theta \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2 + \lambda \sum_{j=1}^n \theta_j^2" /></a>

The λ, or lambda, is the  **regularization parameter**. It determines how much the costs of our theta parameters are inflated.

Using the above cost function with the extra summation, we can smooth the output of our hypothesis function to reduce overfitting. If lambda is chosen to be too large, it may smooth out the function too much and cause underfitting.
