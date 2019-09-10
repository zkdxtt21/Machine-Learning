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
<a href="https://www.codecogs.com/eqnedit.php?latex=\min_\theta&space;\frac{1}{2m}&space;\left[\sum_{i=1}^m&space;(h_\theta(x^{(i)})&space;-&space;y^{(i)})^2&space;&plus;&space;\lambda&space;\sum_{j=1}^n&space;\theta_j^2\right]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\min_\theta&space;\frac{1}{2m}&space;\left[\sum_{i=1}^m&space;(h_\theta(x^{(i)})&space;-&space;y^{(i)})^2&space;&plus;&space;\lambda&space;\sum_{j=1}^n&space;\theta_j^2\right]" title="\min_\theta \frac{1}{2m} \left[\sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2 + \lambda \sum_{j=1}^n \theta_j^2\right]" /></a>

The λ, or lambda, is the  **regularization parameter**. It determines how much the costs of our theta parameters are inflated.

Using the above cost function with the extra summation, we can smooth the output of our hypothesis function to reduce overfitting. If lambda is chosen to be too large, it may smooth out the function too much and cause underfitting.


## Regularized Linear Regression

The **cost function** for regularized linear regression is <br>

<a href="https://www.codecogs.com/eqnedit.php?latex=J(\theta)&space;=&space;\min_\theta&space;\frac{1}{2m}&space;\left[\sum_{i=1}^m&space;(h_\theta(x^{(i)})&space;-&space;y^{(i)})^2&space;&plus;&space;\lambda&space;\sum_{j=1}^n&space;\theta_j^2\right]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?J(\theta)&space;=&space;\min_\theta&space;\frac{1}{2m}&space;\left[\sum_{i=1}^m&space;(h_\theta(x^{(i)})&space;-&space;y^{(i)})^2&space;&plus;&space;\lambda&space;\sum_{j=1}^n&space;\theta_j^2\right]" title="J(\theta) = \min_\theta \frac{1}{2m} \left[\sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2 + \lambda \sum_{j=1}^n \theta_j^2\right]" /></a>

#### Gradient descent procedure: <br>
We will modify our gradient descent function to separate out ​<a href="https://www.codecogs.com/eqnedit.php?latex=\theta_0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\theta_0" title="\theta_0" /></a> from the rest of the parameters because we do not want to penalize <a href="https://www.codecogs.com/eqnedit.php?latex=\theta_0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\theta_0" title="\theta_0" /></a>.<br>
>Repeat{<br>
><a href="https://www.codecogs.com/eqnedit.php?latex=\theta_0&space;:=&space;\theta_0&space;-&space;\alpha&space;\frac{1}{m}&space;\sum_{i=1}^m(h_\theta(x^{(i)})-&space;y^{(i)})x_0^{(i)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\theta_0&space;:=&space;\theta_0&space;-&space;\alpha&space;\frac{1}{m}&space;\sum_{i=1}^m(h_\theta(x^{(i)})-&space;y^{(i)})x_0^{(i)}" title="\theta_0 := \theta_0 - \alpha \frac{1}{m} \sum_{i=1}^m(h_\theta(x^{(i)})- y^{(i)})x_0^{(i)}" /></a>
><a href="https://www.codecogs.com/eqnedit.php?latex=\theta_j&space;:=&space;\theta_j&space;-&space;\alpha&space;\left[\left(\frac{1}{m}&space;\sum_{i=1}^m(h_\theta(x^{(i)})-&space;y^{(i)})x_j^{(i)}&space;\right)&plus;&space;\frac{\lambda}{m}\theta_j&space;\right]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\theta_j&space;:=&space;\theta_j&space;-&space;\alpha&space;\left[\left(\frac{1}{m}&space;\sum_{i=1}^m(h_\theta(x^{(i)})-&space;y^{(i)})x_j^{(i)}&space;\right)&plus;&space;\frac{\lambda}{m}\theta_j&space;\right]" title="\theta_j := \theta_j - \alpha \left[\left(\frac{1}{m} \sum_{i=1}^m(h_\theta(x^{(i)})- y^{(i)})x_j^{(i)} \right)+ \frac{\lambda}{m}\theta_j \right]" /></a><br>
>}
#### Normal Equation<br>
<a href="https://www.codecogs.com/eqnedit.php?latex=\theta&space;=&space;(X^TX&space;&plus;&space;\lambda&space;\cdot&space;L)^{-1}&space;X^Ty" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\theta&space;=&space;(X^TX&space;&plus;&space;\lambda&space;\cdot&space;L)^{-1}&space;X^Ty" title="\theta = (X^TX + \lambda \cdot L)^{-1} X^Ty" /></a> <br>
where<br>
<a href="https://www.codecogs.com/eqnedit.php?latex=L&space;=&space;\left\begin{bmatrix}&space;0&space;&&space;&&space;&&space;\\&space;&&space;1&space;&&space;&&space;\\&space;&&space;&&space;\ddots&space;&&space;\\&space;&&space;&&space;&&space;1&space;\end{bmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?L&space;=&space;\left\begin{bmatrix}&space;0&space;&&space;&&space;&&space;\\&space;&&space;1&space;&&space;&&space;\\&space;&&space;&&space;\ddots&space;&&space;\\&space;&&space;&&space;&&space;1&space;\end{bmatrix}" title="L = \left\begin{bmatrix} 0 & & & \\ & 1 & & \\ & & \ddots & \\ & & & 1 \end{bmatrix}" /></a>

## Regularized Logistic Regression

The **cost function** for regularized logistic regression is <br>
<a href="https://www.codecogs.com/eqnedit.php?latex=J(\theta)&space;=&space;-\frac{1}{m}\sum_{i=1}^m&space;\left[y^{(i)}log(h_\theta(x^{(i)}))&space;&plus;&space;(1-y^{(i)})log(1-h_\theta(x^{(i)}))\right]&space;&plus;&space;\frac{\lambda}{2m}&space;\sum_{j=1}^n&space;\theta_j^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?J(\theta)&space;=&space;-\frac{1}{m}\sum_{i=1}^m&space;\left[y^{(i)}log(h_\theta(x^{(i)}))&space;&plus;&space;(1-y^{(i)})log(1-h_\theta(x^{(i)}))\right]&space;&plus;&space;\frac{\lambda}{2m}&space;\sum_{j=1}^n&space;\theta_j^2" title="J(\theta) = -\frac{1}{m}\sum_{i=1}^m \left[y^{(i)}log(h_\theta(x^{(i)})) + (1-y^{(i)})log(1-h_\theta(x^{(i)}))\right] + \frac{\lambda}{2m} \sum_{j=1}^n \theta_j^2" /></a>

#### Gradient descent procedure: <br>
>Repeat{<br>
><a href="https://www.codecogs.com/eqnedit.php?latex=\theta_0&space;:=&space;\theta_0&space;-&space;\alpha&space;\frac{1}{m}&space;\sum_{i=1}^m(h_\theta(x^{(i)})-&space;y^{(i)})x_0^{(i)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\theta_0&space;:=&space;\theta_0&space;-&space;\alpha&space;\frac{1}{m}&space;\sum_{i=1}^m(h_\theta(x^{(i)})-&space;y^{(i)})x_0^{(i)}" title="\theta_0 := \theta_0 - \alpha \frac{1}{m} \sum_{i=1}^m(h_\theta(x^{(i)})- y^{(i)})x_0^{(i)}" /></a>
><a href="https://www.codecogs.com/eqnedit.php?latex=\theta_j&space;:=&space;\theta_j&space;-&space;\alpha&space;\left[\left(\frac{1}{m}&space;\sum_{i=1}^m(h_\theta(x^{(i)})-&space;y^{(i)})x_j^{(i)}&space;\right)&plus;&space;\frac{\lambda}{m}\theta_j&space;\right]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\theta_j&space;:=&space;\theta_j&space;-&space;\alpha&space;\left[\left(\frac{1}{m}&space;\sum_{i=1}^m(h_\theta(x^{(i)})-&space;y^{(i)})x_j^{(i)}&space;\right)&plus;&space;\frac{\lambda}{m}\theta_j&space;\right]" title="\theta_j := \theta_j - \alpha \left[\left(\frac{1}{m} \sum_{i=1}^m(h_\theta(x^{(i)})- y^{(i)})x_j^{(i)} \right)+ \frac{\lambda}{m}\theta_j \right]" /></a><br>
>}
