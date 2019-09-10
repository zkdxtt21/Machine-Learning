# Logistic Regression
### Logistic Regression Model <br>
**Idea**: We want our prediction between 0 and 1.<br>
**Model**:<br>
 <a href="https://www.codecogs.com/eqnedit.php?latex=h_\theta(x)&space;=&space;g(\theta^Tx)&space;~where~&space;g(z)&space;=&space;\frac{1}{1&space;&plus;&space;e^{-z}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?h_\theta(x)&space;=&space;g(\theta^Tx)&space;~where~&space;g(z)&space;=&space;\frac{1}{1&space;&plus;&space;e^{-z}}" title="h_\theta(x) = g(\theta^Tx) ~where~ g(z) = \frac{1}{1 + e^{-z}}" /></a> <br>
<a href="https://www.codecogs.com/eqnedit.php?latex=h_\theta(x)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?h_\theta(x)" title="h_\theta(x)" /></a> will give us the **probability** that our output is 1.

### Decision Boundary <br>
If we predict y as 1 when hθ(x)≥0.5.<br>
 Then The **decision boundary** is <br>
> <a href="https://www.codecogs.com/eqnedit.php?latex=\theta^TX\geq&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\theta^TX\geq&space;0" title="\theta^TX\geq 0" /></a>  , then we predict y = 1.<br>
>  <a href="https://www.codecogs.com/eqnedit.php?latex=\theta^TX&space;<&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\theta^TX&space;<&space;0" title="\theta^TX < 0" /></a>, then we predict y = 0.

The **decision boundary** is the line that separates the area where y = 0 and where y = 1.

Again, the input to the sigmoid function g(z) doesn't need to be linear, and could be a function that describes a circle (e.g. z = -1 + x1^2 + x2^2​) or any shape to fit our data.

### Cost Function<br>

If we use the same form in Linear Regression, then the cost function is non-convex, which is not something we want.

Instead, we define<br>
<a href="https://www.codecogs.com/eqnedit.php?latex=J(\theta)&space;=&space;\frac{1}{m}\sum_{i=1}^m&space;\frac{1}{2}cost(h_\theta(x^{(i)}),&space;y^{(i)})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?J(\theta)&space;=&space;\frac{1}{m}\sum_{i=1}^m&space;\frac{1}{2}cost(h_\theta(x^{(i)}),&space;y^{(i)})" title="J(\theta) = \frac{1}{m}\sum_{i=1}^m \frac{1}{2}cost(h_\theta(x^{(i)}), y^{(i)})" /></a><br>
<a href="https://www.codecogs.com/eqnedit.php?latex=cost(h_\theta(x),&space;y)&space;=&space;\begin{cases}&space;-\log(h_\theta(x)),~if~&space;y&space;=&space;1;\\&space;-\log(1-h_\theta(x)),&space;~if~y=0.&space;\end{cases}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?cost(h_\theta(x),&space;y)&space;=&space;\begin{cases}&space;-\log(h_\theta(x)),~if~&space;y&space;=&space;1;\\&space;-\log(1-h_\theta(x)),&space;~if~y=0.&space;\end{cases}" title="cost(h_\theta(x), y) = \begin{cases} -\log(h_\theta(x)),~if~ y = 1;\\ -\log(1-h_\theta(x)), ~if~y=0. \end{cases}" /></a>

If y = 1, then cost function looks like<br>
![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/Q9sX8nnxEeamDApmnD43Fw_1cb67ecfac77b134606532f5caf98ee4_Logistic_regression_cost_function_positive_class.png?expiry=1568073600000&hmac=eFsagYpqa-lGR_ORNA5tQmfNaijQgQKneUheJlYwmzE)

If y = 0, the cost function looks like<br>
![](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/Ut7vvXnxEead-BJkoDOYOw_f719f2858d78dd66d80c5ec0d8e6b3fa_Logistic_regression_cost_function_negative_class.png?expiry=1568073600000&hmac=x49tgPnwfducHkzt3e_TRkm22MbFKuOSAcDY20KhyBI)

If our correct answer 'y' is 0, then the cost function will be 0 if our hypothesis function also outputs 0. If our hypothesis approaches 1, then the cost function will approach infinity.

If our correct answer 'y' is 1, then the cost function will be 0 if our hypothesis function outputs 1. If our hypothesis approaches 0, then the cost function will approach infinity.

Note that writing the cost function in this way guarantees that J(θ) is convex for logistic regression.

### Gradient Descent <br>
The **Total Cost Function** is<br>
<a href="https://www.codecogs.com/eqnedit.php?latex=J(\theta)&space;=&space;-\frac{1}{m}\sum_{i=1}^m&space;y^{(i)}log(h_\theta(x^{(i)}))&space;&plus;&space;(1-y^{(i)})log(1-h_\theta(x^{(i)}))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?J(\theta)&space;=&space;-\frac{1}{m}\sum_{i=1}^m&space;y^{(i)}log(h_\theta(x^{(i)}))&space;&plus;&space;(1-y^{(i)})log(1-h_\theta(x^{(i)}))" title="J(\theta) = -\frac{1}{m}\sum_{i=1}^m y^{(i)}log(h_\theta(x^{(i)})) + (1-y^{(i)})log(1-h_\theta(x^{(i)}))" /></a><br>
We want to minimize J(θ).<br>
Repeat{<br>
<a href="https://www.codecogs.com/eqnedit.php?latex=\theta_j&space;=&space;\theta_j&space;-&space;\alpha&space;\frac{1}{m}\frac{\partial}{\partial\theta_j}J(\theta)&space;=&space;\theta_j&space;-&space;\alpha&space;\frac{1}{m}&space;\sum_{i=1}^m(h_\theta(x^{(i)})-&space;y^{(i)})x_j^{(i)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\theta_j&space;=&space;\theta_j&space;-&space;\alpha&space;\frac{1}{m}\frac{\partial}{\partial\theta_j}J(\theta)&space;=&space;\theta_j&space;-&space;\alpha&space;\frac{1}{m}&space;\sum_{i=1}^m(h_\theta(x^{(i)})-&space;y^{(i)})x_j^{(i)}" title="\theta_j = \theta_j - \alpha \frac{1}{m}\frac{\partial}{\partial\theta_j}J(\theta) = \theta_j - \alpha \frac{1}{m} \sum_{i=1}^m(h_\theta(x^{(i)})- y^{(i)})x_j^{(i)}" /></a>
}<br>
Vectorization: <br>
<a href="https://www.codecogs.com/eqnedit.php?latex=\theta&space;=&space;\theta&space;-&space;\alpha&space;\frac{1}{m}&space;\sum_{i=1}^m(h_\theta(x^{(i)})-&space;y^{(i)})&space;\cdot&space;x^{(i)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\theta&space;=&space;\theta&space;-&space;\alpha&space;\frac{1}{m}&space;\sum_{i=1}^m(h_\theta(x^{(i)})-&space;y^{(i)})&space;\cdot&space;x^{(i)}" title="\theta = \theta - \alpha \frac{1}{m} \sum_{i=1}^m(h_\theta(x^{(i)})- y^{(i)}) \cdot x^{(i)}" /></a>

###  Advanced Optimization<br>
"Conjugate gradient", "BFGS", and "L-BFGS" are more sophisticated, faster ways to optimize θ that can be used instead of gradient descent. We suggest that you should not write these more sophisticated algorithms yourself (unless you are an expert in numerical computing) but use the libraries instead, as they're already tested and highly optimized.<br>
We can write a single function that returns both of the cost function and the gradient of the cost function:<br>

``` matlab
function [jVal, gradient] = costFunction(theta)
  jVal = [...code to compute J(theta)...];
  gradient = [...code to compute derivative of J(theta)...];
end
```

Then we can use octave's "fminunc()" optimization algorithm along with the "optimset()" function that creates an object containing the options we want to send to "fminunc()".

```matlab
options = optimset('GradObj', 'on', 'MaxIter', 100);
initialTheta = zeros(2,1);
[optTheta, functionVal, exitFlag] = fminunc(@costFunction, initialTheta, options);
```
We give to the function "fminunc()" our cost function, our initial vector of theta values, and the "options" object that we created beforehand.


## Multiclass Classification

Now we will approach the classification of data when we have more than two categories. Instead of y = {0,1} we will expand our definition so that y = {0,1...n}.

Since y = {0,1...n}, we divide our problem into n+1 (+1 because the index starts at 0) binary classification problems; in each one, we predict the probability that 'y' is a member of one of our classes.

**Method:**<br>
1. Train a logistic regression classifier  <a href="https://www.codecogs.com/eqnedit.php?latex=h_\theta^{(i)}(x)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?h_\theta^{(i)}(x)" title="h_\theta(x^{(i)})" /></a>  for each class￼ to predict the probability that ￼ ￼y = i￼ ￼.

2. To make a prediction on a new x, pick the class ￼that maximizes <a href="https://www.codecogs.com/eqnedit.php?latex=h_\theta^{(i)}(x)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?h_\theta^{(i)}(x)" title="h_\theta(x^{(i)})" /></a>.
