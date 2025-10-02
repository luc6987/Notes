# PC2. Foundations of Machine Learning. (MDC_51006_EP - 2025-2026)

## Exercise 1 (Link Estimation/Prediction)

Assume we observe a sample $((X_i, Y_i))_{i=1}^n$ where $Y_i \in \{-1, 1\}$ and $X_i \in \mathbb{R}^d$ are independent random variables following the same law as a generic pair $(X, Y)$. We want to predict $Y$ from $X$ using a predictor $f : \mathbb{R}^d \to \{-1, 1\}$ and we measure the quality of our predictor using the 0/1 loss:

$$
\ell^{0/1}(y, y') = \mathbf{1}_{y \neq y'}.
$$

The optimal Bayes classifier is given by

$$
f^\star(X) = \operatorname{sign}\big(2p_1(X)-1\big)
$$

where $p_1(X) = \mathbb{P}(Y = 1\mid X)$.

Assume we have an estimate of the conditional law of $Y\mid X$ and denote $\hat p_1(X) = \hat{\mathbb{P}}(\widehat{Y = 1}\mid X)$. We define the plug-in classifier as

$$
\hat f = \operatorname{sign}\big(2\hat p_1 - 1\big).
$$

We want to prove that

$$
\mathbb{E}\!\left[ \ell^{0/1}\big(Y, \hat f(X)\big) \right]
- \mathbb{E}\!\left[ \ell^{0/1}\big(Y, f^\star(X)\big) \right]
  \le \mathbb{E}\!\left[ \lvert \widehat{Y\mid X} - {Y\mid X} \rvert _1 \right]
  \le \left( \mathbb{E}\!\left[2\,\mathrm{KL}\big(Y\mid X,\ \widehat{Y\mid X}\big)\right]\right)^{1/2}.
$$

1. Prove that for any predictor $f$,
   $$
   \mathbb{E}\!\left[\ell^{0/1}\big(Y, f(X)\big)\right]
   = \mathbb{E}_X\!\left[ (1- p_1(X)) + \big(2p_1(X)-1\big)\,\mathbf{1}_{f(X)=-1} \right]
   $$

Pf:
$$
\begin{align*}
\mathbb{E}_{\underbar{X},Y}\!\left[\ell^{0/1}\big(Y, f(\underbar{X})\big)\right]&=\mathbb{E}_{\underbar{X},Y}[1_{Y \neq \hat f(\underbar{X})}]\\
&=\mathbb{E}_{\underbar{X},Y}[1_{Y=1}1_{f(\underbar{X})=-1}+ 1_{Y=-1}1_{f(\underbar{X})=1}]\\
&=\mathbb{E}_{\underbar{X},Y}[(1-1_{Y=-1})(1-1_{f(\underbar{X})=-1})+ 1_{Y=-1}1_{f(\underbar{X})=1}]\\
&=\mathbb{E}_{\underbar{X}}[(1-\mathbb{E}[1_{Y=-1}|\underbar{X}])(1-1_{f(\underbar{X})=1})+ \mathbb{E}[1_{Y=-1}|\underbar{X}]1_{f(\underbar{X})=1}]\\
&=\mathbb{E}_{\underbar{X}}[(1-p_1)(1-1_{f(\underbar{X})=1})+ p_11_{f(\underbar{X})=1}]\\
&=\mathbb{E}_X\!\left[ (1- p_1(X)) + \big(2p_1(X)-1\big)\,\mathbf{1}_{f(X)=-1} \right]

\end{align*}
$$

2. Deduce that
   $$
   \mathbb{E}\!\left[\ell^{0/1}\big(Y, \hat f(X)\big)\right]
   -\mathbb{E}\!\left[\ell^{0/1}\big(Y, f^\star(X)\big)\right]
   \le 2\,\mathbb{E}_X\big[\lvert p_1(X)- \hat p_1(X)\rvert\big] = \mathbb{E}_X\big[\lVert p(X)- \hat p(X)\rVert _1\big]
   $$

Pf:
$$
\begin{align*}
L.H.S.&=\mathbb{E}\!\left[\ell^{0/1}\big(Y, \hat f(X)\big)\right]
   -\mathbb{E}\!\left[\ell^{0/1}\big(Y, f^\star(X)\big)\right]\\
&=\mathbb{E}_X\!\left[ (1- p_1(X)) + \big(2p_1(X)-1\big)\,\mathbf{1}_{\hat f(X)=-1} \right]-(\mathbb{E}_X\!\left[ (1- p_1(X)) + \big(2p_1(X)-1\big)\,\mathbf{1}_{f^*(X)=-1} \right])\\
&=\mathbb{E}_X\!\left[\big(2p_1(X)-1\big)(\mathbf{1}_{\hat f(X)=-1}-\mathbf{1}_{f^*(X)=-1}) \right]\\
\end{align*}
$$

noticed that:
$$
\mathbf{1}_{\hat f(X)=-1}-\mathbf{1}_{f^*(X)=-1}=
\begin{cases}
1 & \text{if } \hat f(X)=-1 \text{ and } f^*(X)=1\\
-1 & \text{if } \hat f(X)=1 \text{ and } f^*(X)=-1\\
0 & \text{otherwise}
\end{cases}=1_{\hat f(x)\neq f^*(x)}f^*(x)
$$

So,

$$L.H.S.=\mathbb{E}_X\!\left[\big(2p_1(X)-1\big)1_{\hat f(x)\neq f^*(x)}f^*(x) \right]=\mathbb{E}_X\!\left[|\big(2p_1(X)-1\big)|1_{\hat f(x)\neq f^*(x)}\right]$$

By the definition of $f^*$ and $\hat f$, we have either $2p_1(x)-1\leq0$ and $2\hat p_1(x)-1\geq0$, or $2p_1(x)-1\geq0$ and $2\hat p_1(x)-1\leq0$. Intuitively, this means that $p_1$ and $\hat p_1$ are on different sides of $1/2$. So, we have a naive inequality $|p_1-\hat p_1|\geq |p_1-1/2|$. Thus,

$$
L.H.S.\leq \mathbb{E}_X\!\left[2|p_1(X)-\hat p_1(X)|\right]
$$

To finish the proof, we recall that $\lVert p(X)- \hat p(X)\rVert _1:= \int |d(X)- \hat d(X)|\,dx$, where $d(X)$ and $\hat d(X)$ are the density functions of $p(X)$ and $\hat p(X)$ respectively. 

For binary classification, we have $p(X)=(p_1(X),1-p_1(X))$ and $\hat p(X)=(\hat p_1(X),1-\hat p_1(X))$. Thus,

$$
\lVert p(X)- \hat p(X)\rVert _1 = \int |p_1(X)- \hat p_1(X)|dx=|p_1(X)- \hat p_1(X)|+|1-p_1(X)- (1-\hat p_1(X))|= 2|p_1(X)- \hat p_1(X)|.
$$

Therefore,
$$
\mathbb{E}\!\left[\ell^{0/1}\big(Y, \hat f(X)\big)\right]
-\mathbb{E}\!\left[\ell^{0/1}\big(Y, f^\star(X)\big)\right] \le 2\,\mathbb{E}_X\big[\lvert p_1(X)- \hat p_1(X)\rvert\big] = \mathbb{E}_X\big[\lVert p(X)- \hat p(X)\rVert _1\big].
$$

3. Finish the proof using $\lVert P - Q\rVert _1 \le \sqrt{2\,\mathrm{KL}(P,Q)}$

Pf:

$$\mathbb{E}_X\big[\lVert p(X)- \hat p(X)\rVert _1\big]\leq \mathbb{E}_X\big[\sqrt{2\,\mathrm{KL}(p(X),\hat p(X))}\big] \leq \sqrt{2\,\mathbb{E}_X\big[\mathrm{KL}(p(X),\hat p(X))\big]} = \left( \mathbb{E}\!\left[2\,\mathrm{KL}\big(Y\mid X,\ \widehat{Y\mid X}\big)\right]\right)^{1/2},$$

the last inequality is due to Jensen's inequality.

> Pinsker 不等式是概率论和信息论里一个很重要的基本工具。它说的是全变差距离（$\ell_1$ 距离的一半）和 KL 散度之间的关系。标准表述是：
>
> $$
> |P - Q|_1 \le \sqrt{2 \mathrm{KL}(P|Q)} .
> $$
>
> ---
>
> ## 步骤一：定义
>
> * **全变差距离**：
>   $$
>   |P - Q|_1 = \sum_x |P(x)-Q(x)|
>   = 2 \sup_{A} |P(A) - Q(A)| .
>   $$
>
> * **KL 散度**：
>   $$
>   \mathrm{KL}(P|Q) = \sum_x P(x)\log \frac{P(x)}{Q(x)} .
>   $$
>
> （这里为了简单起见写成离散情形，连续情况类比。）
>
> ---
>
> ## 步骤二：核心不等式
>
> 我们需要用到一个初等不等式（对 $u>0$）：
> $$
> u \log u \le (u-1) + \frac{1}{2}\frac{(u-1)^2}{u+1}.
> $$
>
> 特别是更常见的简化版：
> $$
> u \log u \le (u-1) + \frac{1}{2}\frac{(u-1)^2}{u+1}.
> $$
>
> 再结合 $u+1 \le 2\max(u,1)$，可以得到一个控制 $(u-1)^2$ 的项。
>
> ---
>
> ## 步骤三：代入 KL
>
> 写
> $$
> \mathrm{KL}(P|Q)
> = \sum_x Q(x)\frac{P(x)}{Q(x)} \log \frac{P(x)}{Q(x)} .
> $$
>
> 令 $u=\frac{P(x)}{Q(x)}$，那么
> $$
> \mathrm{KL}(P|Q) = \sum_x Q(x)u \log u .
> $$
>
> 应用上面的不等式：
> $$
> u \log u \le (u-1) + \frac{(u-1)^2}{2(u+1)} .
> $$
>
> 于是
> $$
> \mathrm{KL}(P|Q) \le \sum_x Q(x)\frac{(u-1)^2}{2(u+1)} .
> $$
>
> ---
>
> ## 步骤四：转化为 $\ell_1$ 范数
>
> 注意到
> $$
> |P(x)-Q(x)| = Q(x)|u-1| .
> $$
>
> 因此
> $$
> \sum_x Q(x)\frac{(u-1)^2}{u+1}
> = \sum_x \frac{|P(x)-Q(x)|^2}{P(x)+Q(x)} .
> $$
>
> 而 $P(x)+Q(x) \le 2\max\{P(x),Q(x)\}$，所以
> $$
> \frac{|P(x)-Q(x)|^2}{P(x)+Q(x)} \le \frac{1}{2}\frac{|P(x)-Q(x)|^2}{\max\{P(x),Q(x)\}} .
> $$
>
> 接下来用 Cauchy–Schwarz：
>
> $$
> \sum_x |P(x)-Q(x)|
> = \sum_x \frac{|P(x)-Q(x)|}{\sqrt{P(x)+Q(x)}} \cdot \sqrt{P(x)+Q(x)} .
> $$
>
> 应用 Cauchy–Schwarz：
> $$
> \Big(\sum_x |P(x)-Q(x)|\Big)^2
> \le \Big(\sum_x \frac{|P(x)-Q(x)|^2}{P(x)+Q(x)}\Big) \Big(\sum_x (P(x)+Q(x))\Big).
> $$
>
> 注意 $\sum_x (P(x)+Q(x))=2$。所以
> $$
> |P-Q|_1^2 \le 2 \sum_x \frac{|P(x)-Q(x)|^2}{P(x)+Q(x)} .
> $$
>
> ---
>
> ## 步骤五：与 KL 连接
>
> 由前面的估计我们有
> $$
> \mathrm{KL}(P|Q) \ge \frac{1}{2}\sum_x \frac{|P(x)-Q(x)|^2}{P(x)+Q(x)} .
> $$
>
> 因此
> $$
> |P-Q|_1^2 \le 4 \mathrm{KL}(P|Q).
> $$
>
> 取平方根：
> $$
> |P-Q|_1 \le \sqrt{4\mathrm{KL}(P|Q)}.
> $$
>
> ---
>
> ## 步骤六：整理常数
>
> 更细致的推导（用 Pinsker 原始的凸性不等式）可以把常数改善到 $\sqrt{2}$，得到最终形式：
>
> $$
> \boxed{|P - Q|_1 \le \sqrt{2\mathrm{KL}(P|Q)}}
> $$
>
> ---


## Exercise 2 (Linear Regression and Design) 

Assume we observe some points $((X_i, Y_i))_{i=1}^n$ with $X_i \in \mathbb{R}^p$ and $Y_i \in \mathbb{R}$ are assumed to be independent copies of a generic $(X,Y)$, regressing those points means estimating $\mathbb{E}[Y \mid X]$ at least on the $X_i$. Indeed, there are two classical problems that can be studied

* **Fixed Design** We assume that the $X_i$ are fixed and want to estimate $f(X) = \mathbb{E}[Y \mid X]$ by a predictor $\hat f$ only at the observed points $X_i$. We measure the quality of $\hat f$ as the average error on a replication set $((X_i, Y'_i))_{i=1}^n$.

$$
\mathrm{Err}_F = \mathbb{E}_{(Y_i)_{i=1}^n}\!\left[
\mathbb{E}_{(Y'_i)_{i=1}^n}\!\left[
\frac{1}{n}\sum_{i=1}^n \ell\big(Y'_i, \hat f(X_i)\big)
\right]\right]
$$

where the $X_i$ are the same and $Y'_i\mid X_i$ has the same law as $Y_i\mid X_i$.

* **Random Design** We assume that the $X_i$ are i.i.d. and we want to estimate $f(X) = \mathbb{E}[Y \mid X]$ by a predictor $\hat f$ everywhere. We measure the quality as the average error on a new observation $(X', Y')$ having the same law as $(X_i, Y_i)$:

$$
\mathrm{Err}_R = \mathbb{E}_{((X_i,Y_i))_{i=1}^n}\!\left[
\mathbb{E}_{(X',Y')}\!\left[
\ell\big(Y', \hat f(X')\big)
\right]\right].
$$

Linear least square regression is classically studied in the fixed design setting, while the random design setting corresponds to the statistical learning setting. In this exercise, we are going to look at the performance of the linear least square on those two settings.

Following Rosset and Tibshirani, we introduce an intermediate setting:

* **Repeated Fixed Design** We assume that the $X_i$ are random, possibly of different law and want to estimate $f(X) = \mathbb{E}[Y \mid X]$ by a predictor $\hat f$ only at the observed points $X_i$. We measure the quality by the Fixed Design error averaged over all possible designs:

$$
\mathrm{Err}_{RF} = \mathbb{E}_{((X_i,Y_i))_{i=1}^n}\!\left[
\mathbb{E}_{(Y'_i)_{i=1}^n}\!\left[
\frac{1}{n}\sum_{i=1}^n \ell\big(Y'_i, \hat f(X_i)\big)
\right]\right].
$$

as well as the average empirical estimate of the quality:

$$
\mathrm{Err}_{Emp} = \frac{1}{n}\sum_{i=1}^n \ell\big(Y_i, \hat f(X_i)\big).
$$

We assume that all the samples are independent conditionally to $X$, that the $X_i$ are fixed in the Fixed design setting, independent in the Repeated Fixed design setting and i.i.d. in the Random design one. Except for the warm-up, we will assume a homoscedastic setting in which $Y = f(X) + \varepsilon$ with $\varepsilon$ a centered random noise of variance $\sigma^2$ and that the loss is the square one.

1. **Warm-up**

   (a) Verify that $\mathbb{E}_{(X_i)_{i=1}^n}[\mathrm{Err}_F] = \mathrm{Err}_{RF}$ and, if the $(X_i, Y_i)$ are i.i.d.,
   $$
   \mathrm{Err}_R = \mathbb{E}_{((X_i,Y_i))_{i=1}^n}\!\left[
   \mathbb{E}_{((X'_i,Y'_i))_{i=1}^n}\!\left[
   \frac{1}{n}\sum_{i=1}^n \ell\big(Y'_i, \hat f(X'_i)\big)
   \right]\right]
   $$
   where $(X'_i, Y'_i)$ are independent copies of $(X_i, Y_i)$.

   pf: We have

   $$
   \begin{align*}
   \mathbb{E}_{X_i}[\mathrm{Err}_F] &= \mathbb{E}_{X_i}[\mathbb{E}_{Y_i}[\mathbb{E}_{Y_i'}[\frac{1}{n}\sum_{i=1}^n \ell(Y_i', \hat f(X_i))]]]\\
   &= \mathbb{E}_{(X_i,Y_i)}[\mathbb{E}_{Y_i'}[\frac{1}{n}\sum_{i=1}^n \ell(Y_i', \hat f(X_i))]]\\
   &= \mathrm{Err}_{RF}
   \end{align*}.
   $$

下面把“什么时候可以把 $\mathbb{E}_X\mathbb{E}_Y F(X,Y)$ 写成 $\mathbb{E}_{(X,Y)} F(X,Y)$”这件事，分清三种常见情况（也是最容易混的三杯“奶茶”）：

# 1) 正宗做法（永远成立）

**结论：**
$$
\boxed{\ \mathbb{E}_X\big[\mathbb{E}_{Y\mid X},F(X,Y)\big]=\mathbb{E}_{(X,Y)}[F(X,Y)]\ }
$$
**条件：** $(F(X,Y))$ 可积（或非负以用 Tonelli），且 $\mathbb{E}_{Y\mid X}$ 是对条件分布 (P_{Y\mid X}) 取期望。
**原理：** **全期望定理（law of iterated expectation）** + **Fubini/Tonelli**：
$$
\int \Big(\int F(x,y)dP_{Y\mid X=x}(y)\Big)dP_X(x)
=\iint F(x,y)dP_{X,Y}(x,y).
$$
**翻译：** 先按「给定 $X$」对 $Y$ 求条件期望，再对 $X$ 平均，等价于直接对联合分布求期望。这是最安全、最标准的写法。

# 2) 省略条件（常见坑）

很多人写成 $\mathbb{E}_X \mathbb{E}_Y F(X,Y)$（注意第二个期望是**边际** $P_Y$，不是 $P_{Y\mid X}$），这时
$$
\mathbb{E}_X \mathbb{E}_Y F(X,Y)
=\iint F(x,y)dP_Y(y)dP_X(x)
=\mathbb{E}_{P_X\times P_Y}[F(X,Y)].
$$
**要想把它等同于 $\mathbb{E}_{(X,Y)} F(X,Y)$，需要：**
$$
\boxed{\ P_{X,Y}=P_X\times P_Y\ }\quad\text{也就是 $X$ 与 $Y$ 独立。}
$$
**否则一般不等。**
反例（强相关）：令 $X=Y\sim \mathrm{Bernoulli}(1/2)$，取 $F(x,y)=xy$。

* $\mathbb{E}_X\mathbb{E}_Y[XY]=(\mathbb{E}X)(\mathbb{E}Y)=\tfrac14$。
* $\mathbb{E}*{(X,Y)}[XY]=\mathbb{E}[X^2]=\mathbb{E}X=\tfrac12$。

  不相等。课堂小结：**把 $\mathbb{E}_{Y\mid X}$ 偷换成 $\mathbb{E}_Y$ 会“把相关性洗没了”。**

# 3) 特例看起来相等（但不是普遍规律）

有时即便 $(X,Y)$ 不独立，也**碰巧**相等，比如：

* 若 $F(x,y)=f(x)+g(y)$（可加分离），则
  $$
  \mathbb{E}_X\mathbb{E}_Y F=\mathbb{E}f(X)+\mathbb{E}g(Y)
  =\mathbb{E}_{(X,Y)}F,
  $$
  因为联合期望同样线性拆分；这里不是魔法，只是线性运算凑巧“屏蔽”了相关性。
* 若 $F$ 与相关结构“正交”($\int F(x,y),d(P_{X,Y}-P_X\times P_Y)=0$），也会相等。但这属于**函数与分布的巧合**，不能当通用法则。

---

## 一张小抄（速判）

* 想稳妥、想对：**写 $\mathbb{E}_X\mathbb{E}_{Y\mid X} F$，总等于 $\mathbb{E}_{(X,Y)}F$。**
* 看到 $\mathbb{E}_X\mathbb{E}_Y F$：这等于对**乘积测度** $(P_X\times P_Y)$ 的期望；

  * 若 $(X\perp Y)$（独立）→ 可改成 $\mathbb{E}_{(X,Y)}F$。
  * 不独立 → 一般**不能**改，除非 $F$ 有特殊结构（如可加分离）或“巧合正交”。

## 数学底座（一句话版）

* **Tonelli/Fubini**：保证可以交换/嵌套积分（期望），前提是非负或可积。
* **全期望定理**：$\mathbb{E}[\mathbb{E}[Z\mid \mathcal{G}]]=\mathbb{E}[Z]$。取 $(\mathcal{G}=\sigma(X))$、$(Z=F(X,Y))$ 即得情况 (1)。
* **乘积测度 vs 联合测度**：$\mathbb{E}_X\mathbb{E}_Y$ 用的是 $(P_X\times P_Y)$；要等同于联合 $(P_{X,Y})$ 必须“没有相关性”。

——一句俏皮话收尾：**缺了“|X”的 $\mathbb{E}_Y$，就像拿脱脂奶去做拿铁——口感（相关性）全被滤掉了。**

   Similarly for $(X'_i, Y'_i)$,

   $$
   \begin{align*}
   \mathbb{E}_{(X_i,Y_i)}[\mathbb{E}_{(X'_i,Y'_i)}[\frac{1}{n}\sum_{i=1}^n \ell(Y'_i, \hat f(X'_i))]]&=\mathbb{E}_{(X_i,Y_i)}[\frac{1}{n}\sum_{i=1}^n\mathbb{E}_{(X_i',Y_i')}[ \ell(Y_i', \hat f(X_i'))]]\\
   &=\mathbb{E}_{(X_i,Y_i)}[\mathbb{E}_{(X_i',Y_i')}[ \ell(Y_i', \hat f(X_i'))]]\\
   &= \mathrm{Err}_R
   \end{align*}
   $$

   (b) Verify that if $\hat f = \arg\min_g \sum_{i=1}^n \ell\big(Y_i, f(X_i)\big)$ then $\mathbb{E}_{(Y_i)_{i=1}^n}[\mathrm{Err}_{Emp}] \le \mathrm{Err}_F$ and $\mathbb{E}_{((X_i,Y_i))_{i=1}^n}[\mathrm{Err}_{Emp}] \le \mathrm{Err}_{RF}$.

Pf: Let $Y_i'$ be independent copies of $Y_i$ and define $\hat f' = \arg\min_g \sum_{i=1}^n \ell\big(Y_i', f(X_i)\big)$, then we have (c.f. PC1 Exercise 3)

$$
\sum_{i=1}^n \ell\big(Y_i, \hat f(X_i)\big) \le \sum_{i=1}^n \ell\big(Y_i, \hat f'(X_i)\big).
$$

By taking expectation over $(Y_i)_{i=1}^n$, we get

$$
\mathbb{E}_{(Y_i)_{i=1}^n}[\mathrm{Err}_{Emp}] \le \mathbb{E}_{(Y_i)_{i=1}^n}[\frac{1}{n}\sum_{i=1}^n \ell\big(Y_i, \hat f'(X_i)\big)].
$$

And we take expectation over $(Y_i')_{i=1}^n$ we have

$$
\mathbb{E}_{(Y_i,Y_i')_{i=1}^n}[\mathrm{Err}_{Emp}] \le \mathbb{E}_{(Y_i)_{i=1}^n}[\mathbb{E}_{(Y_i')_{i=1}^n}[\frac{1}{n}\sum_{i=1}^n \ell\big(Y_i', \hat f'(X_i)\big)]] .
$$

1. We will now assume that $\ell$ is the $\ell_2$ loss: $\ell(Y, f(X)) = \lvert Y - f(X)\rvert^2$.

   (a) Prove that in the fixed design setting:
   $$
   \mathrm{Err}_F = \sigma^2 + \frac{1}{n}\sum_{i=1}^n \big(f(X_i)- \mathbb{E}_{(Y_i)_{i=1}^n}[\hat f(X_i)]\big)^2
   + \frac{1}{n}\sum_{i=1}^n \mathrm{Var}_{(Y_i)_{i=1}^n}\!\big[\hat f(X_i)\big]
   $$

Pf: 
$$
\begin{align*}
\mathrm{Err}_F &= \mathbb{E}_{(Y_i)_{i=1}^n}\!\left[\mathbb{E}_{(Y'_i)_{i=1}^n}\left[\frac{1}{n}\sum_{i=1}^n (Y'_i - \hat f(X_i))^2\right]\right]\\
&= \mathbb{E}_{(Y_i)_{i=1}^n}\!\left[\mathbb{E}_{(Y'_i)_{i=1}^n}\left[\frac{1}{n}\sum_{i=1}^n (Y'_i - f(X_i) + (f(X_i) - \hat f(X_i)))^2\right]\right]\\
&= \mathbb{E}_{(Y_i)_{i=1}^n}\!\left[\mathbb{E}_{(Y'_i)_{i=1}^n}\left[\frac{1}{n}\sum_{i=1}^n ((Y'_i - f(X_i))^2 + (f(X_i)- \hat f(X_i))^2 - 2(Y'_i - f(X_i))(f(X_i)- \hat f(X_i)))\right]\right]\\
&= \frac{1}{n}\sum_{i=1}^n (\mathbb{E}_{Y'_i}[(Y'_i - f(X_i))^2] + \mathbb{E}_{(Y_i)_{i=1}^n}[(f(X_i)- \hat f(X_i))^2] - 2\mathbb{E}_{Y'_i}[Y'_i - f(X_i)]\mathbb{E}_{(Y_i)_{i=1}^n}[f(X_i)- \hat f(X_i)])\\
&= \sigma^2 + \frac{1}{n}\sum_{i=1}^n \mathbb{E}_{(Y_i)_{i=1}^n}[(f(X_i)- \hat f(X_i))^2]\\
&= \sigma^2 + \frac{1}{n}\sum_{i=1}^n \mathbb{E}_{(Y_i)_{i=1}^n}[(f(X_i)- \mathbb{E}_{(Y_i)_{i=1}^n}[\hat f(X_i)]+\mathbb{E}_{(Y_i)_{i=1}^n}[\hat f(X_i)]- \hat f(X_i))^2]\\
&= \sigma^2 + \frac{1}{n}\sum_{i=1}^n \mathbb{E}_{(Y_i)_{i=1}^n}[(f(X_i)- \mathbb{E}_{(Y_i)_{i=1}^n}[\hat f(X_i)])^2 + (\mathbb{E}_{(Y_i)_{i=1}^n}[\hat f(X_i)]- \hat f(X_i))^2 + 2(f(X_i)- \mathbb{E}_{(Y_i)_{i=1}^n}[\hat f(X_i)])(\mathbb{E}_{(Y_i)_{i=1}^n}[\hat f(X_i)]- \hat f(X_i))]\\
&= \sigma^2 + \frac{1}{n}\sum_{i=1}^n (\mathbb{E}_{(Y_i)_{i=1}^n}[(f(X_i)- \mathbb{E}_{(Y_i)_{i=1}^n}[\hat f(X_i)])^2]+\mathbb{E}_{(Y_i)_{i=1}^n}[\hat f(X_i)-\mathbb{E}_{(Y_i)_{i=1}^n}[\hat f(X_i)]^2])\\
&= \sigma^2 + \frac{1}{n}\sum_{i=1}^n (f(X_i)- \mathbb{E}_{(Y_i)_{i=1}^n}[\hat f(X_i)])^2 + \frac{1}{n}\sum_{i=1}^n \mathrm{Var}_{(Y_i)_{i=1}^n}[\hat f(X_i)]

\end{align*}
$$

   (b) Show that if we assume that the $X_i$ are i.i.d. then for the repeated fixed design setting:
   $$
   \mathrm{Err}_{RF} = \sigma^2 + \mathbb{E}_{(X_i)_{i=1}^n}\!\left[
   \big(f(X_1)- \mathbb{E}_{(Y_i)_{i=1}^n}[\hat f(X_1)\mid (X_i)_{i=1}^n]\big)^2 \right]
   + \mathbb{E}_{(X_i)_{i=1}^n}\!\left[\mathrm{Var}_{(Y_i)_{i=1}^n}\!\big[\hat f(X_1)\mid (X_i)_{i=1}^n\big]\right]
   $$

   (c) Finally prove that for the random design setting
   $$
   \mathrm{Err}_R = \sigma^2 + \mathbb{E}_{(X_i)_{i=1}^n, X'}\!\left[
   \big(f(X')- \mathbb{E}_{(Y_i)}[\hat f(X')\mid (X_i)_{i=1}^n]\big)^2 \right]
   + \mathbb{E}_{(X_i)_{i=1}^n, X'}\!\left[\mathrm{Var}_{(Y_i)_{i=1}^n}\!\big[\hat f(X')\mid (X_i)_{i=1}^n\big]\right]
   $$

2. We define the optimism $\mathrm{Opt}_F$ (respectively $\mathrm{Opt}_{RF}$) as the difference between $\mathrm{Err}_F$ (respectively $\mathrm{Err}_{RF}$) and the corresponding expectation of $\mathrm{Err}_{Emp}$.

   (a) Justify the name optimism when the estimator is the minimizer of the empirical risk.

   (b) Verify that in the fixed design setting:
   $$
   \mathrm{Opt}_F = \mathrm{Err}_F - \mathbb{E}_{(Y_i)_{i=1}^n}[\mathrm{Err}_{Emp}] =
   \mathbb{E}_{(Y_i)_{i=1}^n, (Y'_i)_{i=1}^n}\!\left[
   \frac{1}{n}\sum_{i=1}^n (Y'_i - \hat f(X_i))^2 - \frac{1}{n}\sum_{i=1}^n (Y_i - \hat f(X_i))^2
   \right]
   $$

   (c) Prove that
   $$
   \mathrm{Opt}_F = \frac{2}{n}\sum_{i=1}^n \mathrm{Cov}\!\left[Y_i, \hat f(X_i)\right]
   $$

   (d) Deduce that in the repeated fixed design setting
   $$
   \mathrm{Opt}_{RF} = \frac{2}{n}\sum_{i=1}^n \mathbb{E}_{(X_i)_{i=1}^n}\!\left[
   \mathrm{Cov}\!\left[Y_i, \hat f(X_i)\right] \bigg| (X_i)_{i=1}^n \right]
   $$

3. We consider now the random design and define $\mathrm{Opt}_R$ in the same way.

   (a) Verify that
   $$
   \begin{aligned}
   \mathrm{Err}_R &= \mathrm{Err}_F \\
   &\quad + \underbrace{\Big( \mathbb{E}_{(X_i)_{i=1}^n, X'}\!\left[
     \big(f(X')- \mathbb{E}_{(Y_i)}[\hat f(X')\mid (X_i)_{i=1}^n]\big)^2 \right]
   - \mathbb{E}_{(X_i)}\!\left[
     \big(f(X_1)- \mathbb{E}_{(Y_i)}[\hat f(X_1)\mid (X_i)]\big)^2 \right] \Big)}_{\Delta_B} \\
   &\quad + \underbrace{\Big( \mathbb{E}_{(X_i)_{i=1}^n, X'}\!\left[
     \mathrm{Var}_{(Y_i)}\!\left[\hat f(X')\mid (X_i)_{i=1}^n\right]\right]
   - \mathbb{E}_{(X_i)_{i=1}^n}\!\left[
     \mathrm{Var}_{(Y_i)}\!\left[\hat f(X_1)\mid (X_i)_{i=1}^n\right]\right] \Big)}_{\Delta_V}
     \end{aligned}
   $$

   (b) Deduce that
   $$
   \mathrm{Opt}_{RF} = \mathrm{Opt}_F + \Delta_B + \Delta_V
   $$

   (c) Why is it reasonable to expect that both $\Delta_B$ and $\Delta_V$ are non negative?

4. We consider now the linear model $f_\beta(X') = \langle X', \beta\rangle$ and pick $\beta$ by minimizing the empirical loss.

   (a) Prove that, assuming the design matrix $X$ is such that $X^\top X$ is an invertible $(d\times d)$ matrix, the estimate
   $$
   \hat\beta = \big(X^\top X\big)^{-1} X^\top Y
   $$
   so that $\hat f(X_i) = \big(\mathrm{Proj}_{\mathrm{span}(X_i)} Y\big)_i$.

   (b) Deduce that $\mathrm{Opt}_R = \mathrm{Opt}_{RF} = \dfrac{2\sigma^2 d}{n}$

   (c) Prove that $\Delta_B \ge 0$.

   (d) Prove that $\Delta_V \ge 0$ (one can use $\mathbb{E}\!\left[\big(X^\top X\big)^{-1}\right] - \big(\mathbb{E}[X^\top X]\big)^{-1}$ is s.d.p.)

