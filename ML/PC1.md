# PC1

## exo1
$(x_i,y_i) \sim P_{(x,y)}$, $x_i \in X$ et $y_i \in Y = \{-1,1\}$.

classification: $f: X \rightarrow \{-1,1\}$

$$L(f(x_i),y_i)=\mathbb{1}_{y_i\neq f(x_i)}$$

$$R(f)=\mathbb{E}_{(x_i,y_i)\sim P_{(x,y)}}[L(f(x_i),y_i)]=\mathbb{P}_{(x_i,y_i)\sim P_{(x,y)}}[y_i\neq f(x_i)]$$

$$f^* \in \arg\min_{f} R(f)$$ 
Bayes classifier

$$f^*(x)=\begin{cases} 1 & \text{if } P(y=1|x)\geq 1/2 \\ -1 & \text{else} \end{cases}$$

proof:

$$
\begin{align*}
    R(f)&=\mathbb{E}_{(x_i,y_i)\sim P_{(x,y)}}[L(f(x_i),y_i)]\\
&=P(y_i\neq f(x_i))\\
&=P(y=1,f(x)=-1)+P(y=-1,f(x)=1)\\
&=E[1_{y=1}1_{f(x)=-1}]+E[1_{y=-1}1_{f(x)=1}]\\
&=E[E[1_{y=1}1_{f(x)=-1}|x]]+E[E[1_{y=-1}1_{f(x)=1}|x]]\\
&=E[1_{f(x)=-1}P(y=1|x)]+E[1_{f(x)=1}P(y=-1|x)]\\
&=E[(1-1_{f(x)=1})P(y=1|x)]+E[1_{f(x)=1}(1-P(y=1|x))]\\
&=E[P(y=1|x)+1_{f(x)=1}(1-2P(y=1|x))]\\
\end{align*}
$$

$$
\begin{align*}
    R(f)-R(f^*)&=E[P(y=1|x)+1_{f(x)=1}(1-2P(y=1|x))]\\
    &-(E[P(y=1|x)+1_{f^*(x)=1}(1-2P(y=1|x))])\\
   &=E[1_{f(x)=1}(1-2P(y=1|x))]+E[1_{f^*(x)=1}(1-2P(y=1|x))]\\
    &=E[(1_{f(x)=1}-1_{f^*(x)=1})(1-2P(y=1|x))]\\
\end{align*}
$$

noticed that $E[1_{f^*(x)=1}(1-2P(y=1|x))]\in \{-1,0,1\}$.

* if $E[1_{f^*(x)=1}(1-2P(y=1|x))]=1$ than $1_{f(x)=1}=1=1_{f^*(x)=1}$. Thus, $P(y=1|x)\leq P(x=-1|x)=1-P(y=1|x) \Leftrightarrow 0 \leq 1-2P(y=1|x)$ and hence $R(f)-R(f^*)\geq 0$.



## exo2

Let $((x_i,y_i))$ be a sequence of i.i.d. random vectors. $f_\beta$ fit by least squares to a set of training data: ($x_i,y_i$) for $i=1,...,n$ drawn from a population. let $\hat\beta\in R^d$ be the least squares estimator, i.e
$$\hat\beta \in \arg\min_{\beta\in R^d} \frac{1}{N}\sum_{i=1}^n (y_i - f_\beta(x_i))^2$$

Suppose we have some test data $(\tilde x_i, \tilde y_i)$ for $i=1,...,M$ drawn from the same population as the training data. 

### 1 
We define

$$R_{train}=\frac{1}{N}\sum_{i=1}^n (y_i - f_{\hat\beta}(x_i))^2$$

and

$$R_{test}=\frac{1}{M}\sum_{i=1}^M (\tilde y_i - f_{\hat\beta}(\tilde x_i))^2.$$

Prove that
$$E[R_{train}(\hat\beta)] \leq E[R_{test}(\hat\beta)]$$

where the expectations are over all that is random in each expression.

Solution:
For all $\beta \in R^d$,

$E[R_{train}(\beta)]=E[(y_i - f_{\hat\beta}(x_i))^2]$ as $x_i,y_i$ are i.i.d..

$E[R_{test}(\beta)]=E[(\tilde y_i - f_{\hat\beta}(\tilde x_i))^2]=E[(y_i - f_{\hat\beta}(x_i))^2]=E[R_{train}(\beta)]=R(f_\beta)$.

Moreover, $\hat\beta \in \arg\min_{\beta\in R^d} f_\beta$.

So, for all $\beta \in R^d$, $R_{train}(\beta) \geq R_{train}(\hat\beta) \Rightarrow \forall \beta \in R^d,E[R_{test}(\beta)]=E[R_{train}(\beta)] \geq E[R_{train}(\hat\beta)]$.

Define $\beta^* =\arg\min_{\beta\in R^d} E[R_{test}(\beta)]=E[R_{test}(\beta)|(x_i,y_i)^N_1]$.

because $(x,y)$ and $(\tilde x,\tilde y)$ are independent, $E[R_{test}(\beta)|(x_i,y_i)^N_1]\geq E[R_{test}(\beta^*)]$.

But $\hat\beta$ is $\sigma-R^d$ measurable, so $E[R_{test}(\hat\beta)|(x_i,y_i)^N_1]\geq E[R_{test}(\beta^*)]$.

And thus, $E[R_{test}(\hat\beta)]\geq E[R_{test}(\beta^*)]$.

Finally, $E[R_{train}(\hat\beta)] \leq E[R_{test}(\hat\beta)]$.

### 2

Can we replace $R_{test}(\beta)$ by the risk $R(\beta)=E[(\tilde y-f_{\beta}(\tilde x))^2]$ where $(\tilde x,\tilde y)$ follows the population law ?

### 3
Let $\beta^*$ be the minimizer of the risk $R(\beta)$, prove that
   $$E[R_{train}(\hat\beta)]\leq R(\beta^*) \leq E[R(\beta^*)]$$

## exo3

Let $(\Omega,F,P)$ be a probability space. Assume that $(X,Y)$ is a couple of random variables defined on $(\Omega,F,P)$ and taking values in $\Chi\times \{-1,1\}$ where $\chi$ is a given state space. One aim of supervised classification is to build a function $h:\Chi \rightarrow \{-1,1\}$ such that $h(X)$ is the best prediction of $Y$ in a given context. For instance, the probability of misclassification of $h$ is
$$L_{miss}(h) = P(h(X) \neq Y) = E[1_{h(X) \neq Y}].$$

Note that $E[Y|X]$ is a random variable measurable with respect ot the $\sigma-$algebra $\sigma(X)$. Therefore, there exists a funciton $\eta:\Chi \rightarrow [-1,1]$ such that $E[Y|X] = \eta(X)$ almost surely.

In Exo1, we have shown that $h_*$, defined for all $x\in \Chi$ by
$$h_*(x) = \begin{cases} 1 & \text{if } \eta(x) \geq 0 \\ -1 & \text{else} \end{cases}$$
is such that
$$h_* = \argmin_{h:\Chi \rightarrow \{-1,1\}} L_{miss}(h).$$

### 1 
In practice, the minimization of $L_{miss}(h)$ holds on a specific set $\mathcal{H}$ of classifiers(Often called the dictionary), which may possibly not contain the Bayes classifier. Moreover, since in most cases, the clasification problem then boil down to solving

$$\hat{L}^n_{miss}(h)=\frac{1}{n}\sum_{i=1}^n 1_{Y_i\neq h(X_i)},$$

where $(X_i,Y_i)_{1\leq i\leq n}$ are i.i.d. copies of $(X,Y)$. The clasification problem then boil down to solving

$$\hat{h}^n_{\mathcal{H}} \in \argmin_{h \in \mathcal{H}} \hat{L}^n_{miss}(h)$$

Prove that for all set $\mathcal{H}$ of classifiers and $n\geq 1$,
$$L_{miss}(\hat{h}^n_{\mathcal{H}}) - \inf_{h \in \mathcal{H}} L_{miss}(h) \leq 2\sup_{h \in \mathcal{H}} |L_{miss}(h) - \hat{L}^n_{miss}(h)|.$$

You’re being asked to prove the standard “ERM excess risk ≤ 2 × uniform deviation” inequality for 0–1 loss.

Proof:Let

$$
\hat h \equiv \hat h^{\,n}_{\mathcal H}\in\arg\min_{h\in\mathcal H}\hat L^{\,n}_{\text{miss}}(h),\qquad 
L(h)\equiv L_{\text{miss}}(h),\quad \hat L(h)\equiv \hat L^{\,n}_{\text{miss}}(h).
$$

Pick $h^*\in\arg\min_{h\in\mathcal H} L(h)$ if the minimum is attained (if not, use an $\varepsilon$-minimizer; see the remark at the end). Then decompose:

$$
\begin{align*}
L(\hat h)-L(h^*)
&=\underbrace{[L(\hat h)-\hat L(\hat h)]}_{(A)}
+\underbrace{[\hat L(\hat h)-\hat L(h^*)]}_{(B)}
+\underbrace{[\hat L(h^*)-L(h^*)]}_{(C)}.
\end{align*}
$$

By definition of $\hat h$ as an ERM, term (B) $\le 0$. Hence

$$
L(\hat h)-L(h^*) \;\le\; |L(\hat h)-\hat L(\hat h)| + |\hat L(h^*)-L(h^*)|
\;\le\; 2\sup_{h\in\mathcal H}\big|L(h)-\hat L(h)\big|.
$$

Since $L(h^*)=\inf_{h\in\mathcal H}L(h)$, this yields

$$
L_{\text{miss}}\big(\hat h^{\,n}_{\mathcal H}\big)
-\inf_{h\in\mathcal H}L_{\text{miss}}(h)
\;\le\; 2\sup_{h\in\mathcal H}\big|L_{\text{miss}}(h)-\hat L^{\,n}_{\text{miss}}(h)\big|.
$$

**If the infimum isn’t attained.** For any $\varepsilon>0$, choose $h_\varepsilon\in\mathcal H$ with $L(h_\varepsilon)\le \inf_{h\in\mathcal H}L(h)+\varepsilon$. Repeat the same steps with $h_\varepsilon$:

$$
L(\hat h)-\inf L \;\le\; \big(L(\hat h)-L(h_\varepsilon)\big)+\varepsilon
\;\le\; 2\sup_{h\in\mathcal H}|L(h)-\hat L(h)|+\varepsilon,
$$

then let $\varepsilon\downarrow 0$.

Voilà. The ERM can only be so bad: its **excess risk** is at most twice how much empirical risks deviate uniformly from true risks. (If only croissants deviated so nicely.)



### 2
Using Hoeffding's inequality, prove that when $\mathcal{H}=\{h_1,\dot,h_M\}$ for a given $M\geq 1$, then, for all $\delta > 0$,
$$P(L_{miss}(\hat{h}^n_{\mathcal{H}}) \leq \min_{1\leq i \leq M} L_{miss}(h_i) + \sqrt{\frac{2}{n}\log(2M/\delta)}) \geq 1 -\delta$$

下面先用一句话介绍 Hoeffding 不等式，然后给出一步到位的证明。

# Hoeffding 不等式（有界独立情形）

设 $Z_1,\dots,Z_n$ 独立且几乎处处落在区间 $[a,b]$，$\mu=\mathbb E[Z_i]$，$\bar Z_n=\frac1n\sum_i Z_i$。则对任意 $t>0$,

$$
\Pr\!\big(|\bar Z_n-\mu|\ge t\big)\ \le\ 2\exp\!\left(\!-\frac{2n t^2}{(b-a)^2}\right).
$$

对于 0–1 变量（即 $[a,b]=[0,1]$），变成

$$
\Pr\!\big(|\bar Z_n-\mu|\ge t\big)\ \le\ 2\exp(-2nt^2).
$$

# 题目证明（$\mathcal H$ 有限、0–1 损失）

记

$$
L(h)=\Pr(h(X)\ne Y)=\mathbb E[\,\mathbf1_{h(X)\ne Y}\,],\quad
\hat L(h)=\frac1n\sum_{i=1}^n \mathbf1_{h(X_i)\ne Y_i}.
$$

对任意固定 $h$，$\mathbf1_{h(X_i)\ne Y_i}\in[0,1]$ 独立同分布，套用 Hoeffding：

$$
\Pr\!\big(|\hat L(h)-L(h)|>t\big)\ \le\ 2e^{-2nt^2}.
$$

对有限类 $\mathcal H=\{h_1,\dots,h_M\}$ 做并联合（union bound）：

$$
\Pr\!\Big(\sup_{h\in\mathcal H}|\hat L(h)-L(h)|>t\Big)
\ \le\ \sum_{j=1}^M \Pr\!\big(|\hat L(h_j)-L(h_j)|>t\big)
\ \le\ 2M e^{-2nt^2}.
$$

令右边 $\le \delta$，解得

$$
t\ =\ \sqrt{\frac{1}{2n}\log\frac{2M}{\delta}}.
$$

因此以概率至少 $1-\delta$ 有

$$
\sup_{h\in\mathcal H}|\hat L(h)-L(h)|\ \le\ \sqrt{\frac{1}{2n}\log\frac{2M}{\delta}}.
$$

最后调用你在第 1 问已经证明的 **ERM 过剩风险不等式**

$$
L(\hat h^n_{\mathcal H})-\inf_{h\in\mathcal H}L(h)
\ \le\ 2\sup_{h\in\mathcal H}|\hat L(h)-L(h)|.
$$

把上面的上界代入，得到以概率至少 $1-\delta$,

$$
L_{miss}(\hat h^n_{\mathcal H})
\ \le\ \min_{1\le i\le M} L_{miss}(h_i)\ +\ 2\sqrt{\frac{1}{2n}\log\frac{2M}{\delta}}
\ =\ \min_{i} L_{miss}(h_i)\ +\ \sqrt{\frac{2}{n}\log\frac{2M}{\delta}},
$$

证毕。

一句话版总结：**有限假设类 + Hoeffding + 并联合 + ERM 分解**，三板斧下去，泛化差距就老实地缩到 $\sqrt{\tfrac{\log M}{n}}$ 量级了（外加一个 $\log(1/\delta)$）。
