# PC3

## EXERCISE 1 (LINK SURROGATE LOSS/PREDICTION)
We want to show that the minimizer of
$$\mathbb{E}[\overline{\ell}(Y,f(\underline{X}))]=\mathbb{E}[l(Yf(\underline{X}))]$$
, where $l$ is a convex non-increasing function such that $l(0)=1$, $l$ is differentiable at 0 and $l^{\prime}(0)=-1$, is the Bayes classifier $f^{*}=sign(2\mathbb{P}(Y=1|\underline{X})-1)$. $y\in\{-1,1\}$ and $\underline{X}\in\mathbb{R}^{d}.$

1.  Write $\mathbb{E}[\overline{\ell}(Y,f(\underline{X}))]$ as a function $H(f,\eta(\underline{X}))$ where $\eta(\underline{X})=\mathbb{P}(Y=1|\underline{X}).$
Solution: We have
$$
\begin{align*}
\mathbb{E}[\overline{\ell}(Y,f(\underline{X}))] &= \mathbb{E}[\mathbb{E}[l(Yf(\underline{X}))|\underline{X}]] \\
&= \mathbb{E}[\mathbb{E}[l(f(\underline{X}))\mathbb{1}_{Y=1} + l(-f(\underline{X}))\mathbb{1}_{Y=-1}|\underline{X}]] \\
&= \mathbb{E}[\eta(\underline{X})l(f(\underline{X})) + (1-\eta(\underline{X}))l(-f(\underline{X}))] \\
&= \mathbb{E}[H(f(\underline{X}),\eta(\underline{X}))].
\end{align*}
$$
So we can define
$$H(f,\eta) = \eta l(f) + (1-\eta)l(-f).$$

2.  Prove that the optimal $\tilde{f}$ of $H(f,\eta)$ for a given $\eta$ has the same sign as $2\eta-1$. 
Solution: We want to minimize $H(f,\eta)$ with respect to $f$. We can not compute the derivative of $H$ with respect to $f$, because $l$ is not differentiable everywhere. However, we can compute the subdifferential of $H$ with respect to $f$:
$$\partial_{f}H(f,\eta) = \eta \partial l(f) - (1-\eta)\partial l(-f).$$
When the optimal $\tilde{f}$ is reached, $0\in \partial_{f}H(\tilde{f},\eta)$.
We introduce $\xi_+ := \partial l(f)$ and $\xi_- := \partial l(-f)$. So at the optimum, we have
$$0\in \eta \xi_+ - (1-\eta)\xi_-.$$

We recall the properties of $l$:
- $l$ is non-decreasing, thus $\xi_+ <0$ and $\xi_- <0$.
- $l$ is convex, thus if $f < -f$, $\xi_+ < \xi_-$ and if $f > -f$, $\xi_+ > \xi_-$.
- $l\in \mathcal{C}^{1}(0)$, thus $\xi_+ = -1$ and $\xi_- = -1$ if $\tilde{f}=0$ and $2\eta - 1 = 0$. This tells us that $\tilde{f}$ and $2\eta - 1$ have the same sign.

We now analyze the sign: If $\tilde{f} > 0$, then $\tilde{f} > -\tilde{f}$, thus $\xi_+ > -\xi_-$. So we have
$$0\in \eta \xi_+ - (1-\eta)\xi_-\geq(2\eta - 1)\xi_+.$$
We know that $\xi_+ < 0$, thus $2\eta - 1 > 0$. In this case $\tilde{f}$ and $2\eta - 1$ have the same sign.

Similarly, we can prove the case $\tilde{f} < 0$.

3.  Conclude.

Solution: We have proved that for a given $\eta$, the optimal $\tilde{f}$ of $H(f,\eta)$ has the same sign as $2\eta - 1$. Thus, the optimal $\tilde{f}(\underline{X})$ of $\mathbb{E}[H(f(\underline{X}),\eta(\underline{X}))]$ has the same sign as $2\eta(\underline{X}) - 1$. We know that the Bayes classifier is defined by $f^{*}(\underline{X}) = sign(2\eta(\underline{X}) - 1)$. Thus, $\tilde{f}$ and $f^{*}$ have the same sign.

## EXERCISE 2 (BACKPROP)
Let f be a neural network with L hiddern layers parametrized by $W_{1},b_{1},...,W_{L},b_{L},W_{O},b_{O}$ by
$$z_{1}(x)=W_{1}x+b_{1}$$
$$h_{1}(x)=g_{1}(z_{1}(x))$$
$$\dots$$
$$z_{l}(x)=W_{l}h_{l-1}(x)+b_{l}$$
$$h_{l}(x)=g_{l}(z_{l}(x))$$
$$\dots$$
$$z_{O}(x)=W_{O}h_{L}(x)+b_{O}$$
$$f(x)=g_{O}(z_{0}(x))$$

For the sake of simplicity, we do not denote the dependency on the parameters in the functions. We are nevertheless interested in computing the derivative of
$$F_{i}=l(Y_{i},f(X_{i}))$$
with respect to those parameters.

### 1 Warmup. 
Let $u(x)=u_{out}(u_{cur}(u_{in}(x,\theta_{in}),\theta_{cur}),\theta_{out})$.
    
(a) Verify that
$$\frac{\partial u^{(d)}}{d\theta_{cur}^{(d^{\prime})}}(x)=\sum_{k}\frac{\partial u_{out}^{(d)}}{du_{cur}^{(k)}}(u_{cur}(u_{in}(x,\theta_{in}),\theta_{cur}),\theta_{out})\frac{\partial u_{cur}^{(k)}}{d\theta_{cur}^{(d^{\prime})}}(u_{in}(x,\theta_{in}),\theta_{cur})$$

Pf: Use the chain rule. If we denote $u^{(d)}=u^{(d)}(v^{(1)}(x),v^{(2)}(x),\ldots,v^{(p)}(x))$, we can apply the chain rule to obtain the desired result.

$$\frac{\partial u^{(d)}}{\partial x}(v(x))=\sum_{k}\frac{\partial u^{(d)}}{\partial v^{(k)}}(v(x))\frac{\partial v^{(k)}}{\partial x}(x)$$

(b) Using Jacobian matrix notation where $\frac{Dv}{dw}$ is defined by
$(\frac{Dv}{dw})_{d,^{\prime}}=\frac{\partial v^{d}}{dw^{d^{\prime}}}$
verify that this can be rewritten as
$$\frac{Du}{d\theta_{cur}}(x)=\frac{Du_{out}}{dx_{cur}}(u_{cur}(x,\theta_{in}),\theta_{cur}),\theta_{out})\times\frac{Du_{cur}}{d\theta_{cur}}(u_{in}(x,\theta_{in}),\theta_{cur}).$$

Pf: noticed that
$$(\frac{\partial u_{out}^{(d)}}{du_{cur}^{(k)}}(u_{cur}(u_{in}(x,\theta_{in}),\theta_{cur}),\theta_{out}))_{d,k}=\frac{D u_{out}^{(d)}}{du_{cur}^{(k)}}(u_{cur}(u_{in}(x,\theta_{in}),\theta_{cur}),\theta_{out})$$
and
$$(\frac{\partial u_{cur}^{(k)}}{d\theta_{cur}^{(d^{\prime})}}(u_{in}(x,\theta_{in}),\theta_{cur}))_{k,d^{\prime}}=\frac{D u_{cur}^{(k)}}{d\theta_{cur}^{(d^{\prime})}}(u_{in}(x,\theta_{in}),\theta_{cur})$$
Thus, we can rewrite the previous result in matrix form as
$$\frac{Du}{d\theta_{cur}}(x)=\frac{Du_{out}}{du_{cur}}(u_{cur}(u_{in}(x,\theta_{in}),\theta_{cur}),\theta_{out})\times\frac{Du_{cur}}{d\theta_{cur}}(u_{in}(x,\theta_{in}),\theta_{cur}).$$

### 2
Using $\theta_{l}=(flatten(W_{l}),b_{l})$, where flatten is an operator which transforms $n\times m$ matrix into a $1\times(n\times m)$ vector.

(a) Deduce that
$$\frac{DF_{i}}{d\theta_{O}}=\frac{Dl}{df}(f(X_{i}))\times\frac{Df}{dz_{O}}(z_{O}(X_{i}))\times\frac{Dz_{O}}{d\theta_{O}}(h_{L}(X_{i}))$$

$$\frac{DF_{i}}{d\theta_{l}}=\frac{Dl}{df}(f(X_{i}))\times\frac{Df}{dz_{O}}(z_{O}(X_{i}))\times\frac{Dz_{O}}{dh_{L}}(h_{L}(X_{i}))\times\frac{Dh_{L}}{dz_{L}}(z_{L}(X_{i}))\times\frac{Dz_{L}}{dh_{L-1}}(h_{L-1}(X_{i}))$$

$$\times\cdot\cdot\cdot\times\frac{Dh_{l+1}}{dz_{l+1}}(z_{l+1}(X_{i}))\times\frac{Dz_{l+1}}{dh_{l}}(h_{l}(X_{i}))\times\frac{Dh_{l}}{dz_{l}}(z_{l}(X_{i}))\times\frac{Dz_{l}}{d\theta_{l}}(h_{l-1}(X_{i}))$$

with some abuse of notations if $l>L-1$ and $l=1.$



(b) Verifiy that
$$\nabla_{\theta_{l}}F_{i}=\frac{DF_{i}}{d\theta_{l}}^{\top}$$

(c) Compute
$$\frac{Dl}{Df}; \frac{Df}{dz_{O}}, \frac{Dh_{l}}{dz_{l}}, \frac{Dz_{l}}{dh_{l-1}} \text{ and } \frac{dz_{l}}{d\theta_{l}}$$

### 3
We are now interested in the complexity of such a computation.

(a) What are the sizes of those matrices?

(b) What is the best way to compute the products of those matrices? From left to right or right to left?

(c) Why is the left to right direction called backward and the right to left direction called forward?

(d) Explain why the backward solution is even better when we compute all the derivatives with all the $\theta_{I}.$