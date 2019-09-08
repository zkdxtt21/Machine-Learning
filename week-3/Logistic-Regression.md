# Logistic Regression
### Logistic Regression Model <br>
**Idea**: We want our prediction between 0 and 1.<br>
**Model**:
 <a href="https://www.codecogs.com/eqnedit.php?latex=h_\theta(x)&space;=&space;g(\theta^Tx)&space;~where~&space;g(z)&space;=&space;\frac{1}{1&space;&plus;&space;e^{-z}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?h_\theta(x)&space;=&space;g(\theta^Tx)&space;~where~&space;g(z)&space;=&space;\frac{1}{1&space;&plus;&space;e^{-z}}" title="h_\theta(x) = g(\theta^Tx) ~where~ g(z) = \frac{1}{1 + e^{-z}}" /></a> <br>
<a href="https://www.codecogs.com/eqnedit.php?latex=h_\theta(x)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?h_\theta(x)" title="h_\theta(x)" /></a> will give us the **probability** that our output is 1.

### Decision Boundary <br>
If we predict y as 1 when hθ(x)≥0.5.
 Then The **decision boundary** is <br>
> <a href="https://www.codecogs.com/eqnedit.php?latex=\theta^TX\geq&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\theta^TX\geq&space;0" title="\theta^TX\geq 0" /></a>  , then we predict y = 1.
>  <a href="https://www.codecogs.com/eqnedit.php?latex=\theta^TX&space;<&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\theta^TX&space;<&space;0" title="\theta^TX < 0" /></a>, then we predict y = 0.

The **decision boundary** is the line that separates the area where y = 0 and where y = 1.

Again, the input to the sigmoid function g(z) doesn't need to be linear, and could be a function that describes a circle (e.g. z = -1 + x1^2 + x2^2​) or any shape to fit our data.
