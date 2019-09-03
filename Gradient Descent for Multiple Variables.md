
Gradient Descent
=====
Cost Function : $J(\theta)$ <br>
Gradient Descent :
Repeat{$$\theta_j := \theta_j - \alpha\frac{\partial}{\partial \theta_j} J(\theta_1, \cdots, \theta_n) $$}

For linear regression cases,
Hypothesis: $h_\theta(x) = \boldsymbol{\theta}^T\boldsymbol{x} = \theta_0 x_0 + \theta_1 x_1 +\cdots + \theta_n x_n$
Cost Function: $$J(\theta) = \frac{1}{2m} \sum_{i=1}^m (h_\theta(x^{(i)} ) -y^{(i)})^2$$
#### If n = 1,
Cost Function: $$\begin{aligned}
J(\theta) &= \frac{1}{2m} \sum_{i=1}^m (h_\theta(x^{(i)} ) -y^{(i)})^2 \\
&= \frac{1}{2m} \sum_{i=1}^m (\theta_0  + \theta_1 x^{(i)} - y^{(i)})^2
\end{aligned}$$
Repeat{$$
\begin{aligned}
\theta_0 :&= \theta_0 - \alpha \frac{1}{m} \sum_{i=1}^m (h_\theta(x^{(i)} ) -y^{(i)})\\
\theta_1 :&= \theta_1 - \alpha \frac{1}{m} \sum_{i=1}^m (h_\theta(x^{(i)} ) -y^{(i)}) x^{(i)}
\end{aligned}
$$}
#### Now suppose $n \geq 1$:
Repeat{$$
\theta_j := \theta_j - \alpha \frac{1}{m} \sum_{i=1}^m (h_\theta(x^{(i)} ) -y^{(i)}) x_j^{(i)}
$$(Simultaneously updated $\theta_j$ for $j = 1, \cdots, n$)
}
### Practical Tricks for Gradient Descent
Idea: Make sure features are on a similar scale.
This is because θ will descend quickly on small ranges and slowly on large ranges, and so will oscillate inefficiently down to the optimum when the variables are very uneven.
1. **Feature Scaling**
$$x_i := \frac{x_i - \mu_i}{s_i}$$
Where $μ_i$ is the average of all the values for feature (i) and $s_i$ is the range of values (max - min).
2. **Mean normalization**
$$x_i := \frac{x_i - \mu_i}{s_i}$$
Where $μ_i$ is the average of all the values for feature (i) and  $s_i$ is the standard deviation.
