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

### 1. **Warm-up**

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

#### 正宗做法（永远成立）

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

#### 2) 省略条件（常见坑）

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

#### 3) 特例看起来相等（但不是普遍规律）

有时即便 $(X,Y)$ 不独立，也**碰巧**相等，比如：

* 若 $F(x,y)=f(x)+g(y)$（可加分离），则
  $$
  \mathbb{E}_X\mathbb{E}_Y F=\mathbb{E}f(X)+\mathbb{E}g(Y)
  =\mathbb{E}_{(X,Y)}F,
  $$
  因为联合期望同样线性拆分；这里不是魔法，只是线性运算凑巧“屏蔽”了相关性。
* 若 $F$ 与相关结构“正交”($\int F(x,y),d(P_{X,Y}-P_X\times P_Y)=0$），也会相等。但这属于**函数与分布的巧合**，不能当通用法则。

#### 一张小抄（速判）:

* 想稳妥、想对：**写 $\mathbb{E}_X\mathbb{E}_{Y\mid X} F$，总等于 $\mathbb{E}_{(X,Y)}F$。**
* 看到 $\mathbb{E}_X\mathbb{E}_Y F$：这等于对**乘积测度** $(P_X\times P_Y)$ 的期望；

  * 若 $(X\perp Y)$（独立）→ 可改成 $\mathbb{E}_{(X,Y)}F$。
  * 不独立 → 一般**不能**改，除非 $F$ 有特殊结构（如可加分离）或“巧合正交”。

数学底座（一句话版）

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

### 2. We will now assume that $\ell$ is the $\ell_2$ loss: $\ell(Y, f(X)) = \lvert Y - f(X)\rvert^2$.

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

$$
\begin{align*}
\mathrm{Err}_{RF}
&= \mathbb{E}_{(X_i,Y_i)}\left[\mathbb{E}_{(Y'_i)}\left[\frac1n\sum_{i=1}^n\big(Y'_i-\hat f(X_i)\big)^2\right]\right] \\
&= \mathbb{E}_{(X_i)}\left[\mathbb{E}_{(Y_i)}\left[\mathbb{E}_{(Y'_i)}\left[\frac1n\sum_{i=1}^n\big(Y'_i-\hat f(X_i)\big)^2\ \big|\ (X_i)\right]\right]\right] \\
&= \mathbb{E}_{(X_i)}\left[\frac1n\sum_{i=1}^n
\mathbb{E}_{(Y_i)}\left[\mathbb{E}_{Y'_i}\left[\big(Y'_i-f(X_i)+f(X_i)-\hat f(X_i)\big)^2\ \big|\ (X_i)\right]\right]\right] \\
&= \mathbb{E}_{(X_i)}\left[\frac1n\sum_{i=1}^n\left(
\underbrace{\mathbb{E}_{Y'_i}\left[(Y'_i-f(X_i))^2\mid (X_i)\right]}_{=\ \sigma^2}
+\mathbb{E}_{(Y_i)}\left[(f(X_i)-\hat f(X_i))^2\mid (X_i)\right]\right)\right] \\
&= \sigma^2
+\mathbb{E}_{(X_i)}\left[\frac1n\sum_{i=1}^n
\left(\big(f(X_i)-\mathbb{E}_{(Y_i)}[\hat f(X_i)\mid (X_i)]\big)^2
+\mathrm{Var}_{(Y_i)}\big[\hat f(X_i)\mid (X_i)\big]\right)\right] \\
&\overset{X_i\ \text{i.i.d.}}{=}
\sigma^2
+\mathbb{E}_{(X_i)}\left[\big(f(X_1)-\mathbb{E}_{(Y_i)}[\hat f(X_1)\mid (X_i)]\big)^2\right]
+\mathbb{E}_{(X_i)}\left[\mathrm{Var}_{(Y_i)}\big[\hat f(X_1)\mid (X_i)\big]\right].
\end{align*}
$$




(c) Finally prove that for the random design setting
$$
\mathrm{Err}_R = \sigma^2 + \mathbb{E}_{(X_i)_{i=1}^n, X'}\!\left[
\big(f(X')- \mathbb{E}_{(Y_i)}[\hat f(X')\mid (X_i)_{i=1}^n]\big)^2 \right]
+ \mathbb{E}_{(X_i)_{i=1}^n, X'}\!\left[\mathrm{Var}_{(Y_i)_{i=1}^n}\!\big[\hat f(X')\mid (X_i)_{i=1}^n\big]\right]
$$

Pf:
$$
\begin{align*}
\mathrm{Err}_R
&= \mathbb{E}_{(X_i,Y_i)}\left[\mathbb{E}_{X',Y'}\left[(Y'-\hat f(X'))^2\right]\right] \\
&= \mathbb{E}_{(X_i)}\left[\mathbb{E}_{(Y_i)}\left(\mathbb{E}_{X'}\left[\mathbb{E}_{Y'\mid X'}\left[(Y'-\hat f(X'))^2\ \big|\ X',(X_i)\right]\right]\right)\right] \\
&= \mathbb{E}_{(X_i)}\left[\mathbb{E}_{(Y_i)}\left(\mathbb{E}_{X'}\left[\mathbb{E}_{Y'\mid X'}\left[(Y'-f(X')+f(X')-\hat f(X'))^2\ \big|\ X',(X_i)\right]\right]\right)\right] \\
&= \mathbb{E}_{(X_i)}\left[\mathbb{E}_{(Y_i)}\left(\mathbb{E}_{X'}\left[\underbrace{\mathbb{E}_{Y'\mid X'}[(Y'-f(X'))^2\mid X']}_{=\ \sigma^2}
+\ (f(X')-\hat f(X'))^2\right]\right)\right] \\
&= \sigma^2+\mathbb{E}_{(X_i)}\left[\mathbb{E}_{(Y_i)}\left(\mathbb{E}_{X'}\left[(f(X')-\hat f(X'))^2\right]\right)\right] \\
&= \sigma^2+\mathbb{E}_{(X_i)}\left[\mathbb{E}_{X'}\left(\mathbb{E}_{(Y_i)}\left[(f(X')-\hat f(X'))^2\ \big|\ X',(X_i)\right]\right)\right] \\
&= \sigma^2+\mathbb{E}_{(X_i),X'}\left[\ \big(f(X')-\mathbb{E}_{(Y_i)}[\hat f(X')\mid X',(X_i)]\big)^2
+\mathrm{Var}_{(Y_i)}\big[\hat f(X')\mid X',(X_i)\big]\ \right] \\
&= \sigma^2+\mathbb{E}_{(X_i),X'}\left[\big(f(X')-\mathbb{E}_{(Y_i)}[\hat f(X')\mid (X_i)]\big)^2\right]
+\mathbb{E}_{(X_i),X'}\left[\mathrm{Var}_{(Y_i)}\big[\hat f(X')\mid (X_i)\big]\right].
\end{align*}
$$

### 3. We define the optimism $\mathrm{Opt}_F$ (respectively $\mathrm{Opt}_{RF}$) as the difference between $\mathrm{Err}_F$ (respectively $\mathrm{Err}_{RF}$) and the corresponding expectation of $\mathrm{Err}_{Emp}$.

   (a) Justify the name optimism when the estimator is the minimizer of the empirical risk.

**答：**之所以叫 *optimism*（“乐观偏差”），是因为当 (\hat f) 是**经验风险最小化**（ERM）解时，训练集上的经验误差系统性**低估**真实误差：

* 由 ERM 的极小性，
  $$
  \mathbb{E}[\mathrm{Err}_{Emp}] \le \mathrm{Err}_{F},\qquad
  \mathbb{E}[\mathrm{Err}_{Emp}] \le \mathrm{Err}_{RF}.
  $$
  因而定义的
  $$
  \mathrm{Opt}_{F} = \mathrm{Err}_{F}-\mathbb{E}[\mathrm{Err}_{Emp}] \ge 0,\quad
  \mathrm{Opt}_{RF} = \mathrm{Err}_{RF}-\mathbb{E}[\mathrm{Err}_{Emp}] \ge 0,
  $$
  是一个**非负的“补偿量”**：把“过于乐观”的训练误差校正回真实误差。

* 等价地，
  $$
  \mathbb{E}[\mathrm{Err}_{Emp}] + \mathrm{Opt}_F = \mathrm{Err}_F,
  $$
  说明 $\mathrm{Opt}_F$ 正是训练误差对真实误差的**系统性低估**（即“乐观”）的期望大小。

* 典型地（如 OLS、同方差噪声），
  $$
  \mathrm{Opt}_{F} = \frac{2\sigma^2\mathrm{df}}{n}\ (\ge 0),
  $$
  $$
  \mathrm{Opt}_{RF} = \frac{2\sigma^2\mathrm{df}}{n}\ (\ge 0),
  $$
  进一步体现训练误差的“乐观”随模型自由度增大而增大。


(b) Verify that in the fixed design setting:
$$
\mathrm{Opt}_F = \mathrm{Err}_F - \mathbb{E}_{(Y_i)_{i=1}^n}[\mathrm{Err}_{Emp}] =
\mathbb{E}_{(Y_i)_{i=1}^n, (Y'_i)_{i=1}^n}\!\left[
\frac{1}{n}\sum_{i=1}^n (Y'_i - \hat f(X_i))^2 - \frac{1}{n}\sum_{i=1}^n (Y_i - \hat f(X_i))^2
\right]
$$
Pf:

$$
\begin{align*}
\mathrm{Opt}_F
&= \mathrm{Err}_F-\mathbb{E}_{(Y_i)_{i=1}^n}[\mathrm{Err}_{Emp}]\\
&= \mathbb{E}_{(Y_i)_{i=1}^n}\left[\ \mathbb{E}_{(Y'_i)_{i=1}^n}\left[\frac{1}{n}\sum_{i=1}^n\big(Y'_i-\hat f(X_i)\big)^2\right]\right]
-\ \mathbb{E}_{(Y_i)_{i=1}^n}\left[\frac{1}{n}\sum_{i=1}^n\big(Y_i-\hat f(X_i)\big)^2\right]\\
&= \mathbb{E}_{(Y_i)_{i=1}^n}\left[\ \mathbb{E}_{(Y'_i)_{i=1}^n}\left[\frac{1}{n}\sum_{i=1}^n\big(Y'_i-\hat f(X_i)\big)^2\ -\ \frac{1}{n}\sum_{i=1}^n\big(Y_i-\hat f(X_i)\big)^2\right]\right]\\
&= \mathbb{E}_{(Y_i)_{i=1}^n,\ (Y'_i)_{i=1}^n}\left[\frac{1}{n}\sum_{i=1}^n\big(Y'_i-\hat f(X_i)\big)^2\ -\ \frac{1}{n}\sum_{i=1}^n\big(Y_i-\hat f(X_i)\big)^2\right],
\end{align*}
$$

其中第三行把对 $(Y_i)_1^n$ 的外层期望视作条件化的常量，再把内外层期望合并为对 $(Y_i)_1^n,(Y'_i)_1^n$ 的联合期望即可。这正是所需结论。


(c) Prove that
$$
\mathrm{Opt}_F = \frac{2}{n}\sum_{i=1}^n \mathrm{Cov}\!\left[Y_i, \hat f(X_i)\right]
$$

 Pf:

$$
\begin{align*}
\mathrm{Opt}_F
&= \mathbb{E}_{(Y_i),(Y'_i)}\left[\frac{1}{n}\sum_{i=1}^n\big((Y'_i-\hat f(X_i))^2-(Y_i-\hat f(X_i))^2\big)\right] \\
&= \frac{1}{n}\sum_{i=1}^n\Big(\underbrace{\mathbb{E}[Y_i'^2]-\mathbb{E}[Y_i^2]}_{=0}
\ -\ 2\big(\mathbb{E}[Y_i'\hat f(X_i)]-\mathbb{E}[Y_i\hat f(X_i)]\big)\Big).
\end{align*}
$$

在 **fixed design** 下，$X_i$ 为常数，$Y'_i$ 与训练数据（从而与 $\hat f(X_i)$）独立，且 $\mathbb{E}[Y'_i]=\mathbb{E}[Y_i]=f(X_i)$。因此
$$
\mathbb{E}[Y_i'\hat f(X_i)]=\mathbb{E}[Y'_i]\mathbb{E}[\hat f(X_i)]=f(X_i)\mathbb{E}[\hat f(X_i)].
$$
于是对每个 $i$，
$$
\mathbb{E}\big[(Y'_i-\hat f(X_i))^2\big]-\mathbb{E}\big[(Y_i-\hat f(X_i))^2\big]
=2\Big(\mathbb{E}[Y_i\hat f(X_i)]-f(X_i),\mathbb{E}[\hat f(X_i)]\Big)
=2\mathrm{Cov}\big(Y_i,\hat f(X_i)\big).
$$

把上式在 $i=1,\dots,n$ 上求和并除以 $n$，得到
$$
\boxed{\ \mathrm{Opt}_F = \frac{2}{n}\sum_{i=1}^n \mathrm{Cov}\left[Y_i, \hat f(X_i)\right]\ }.
$$
这正是所求。

(d) Deduce that in the repeated fixed design setting
$$
\mathrm{Opt}_{RF} = \frac{2}{n}\sum_{i=1}^n \mathbb{E}_{(X_i)_{i=1}^n}\!\left[
\mathrm{Cov}\!\left[Y_i, \hat f(X_i)\right] \bigg| (X_i)_{i=1}^n \right]
$$

### 4. We consider now the random design and define $\mathrm{Opt}_R$ in the same way.

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

### 5. We consider now the linear model $f_\beta(X') = \langle X', \beta\rangle$ and pick $\beta$ by minimizing the empirical loss.

   (a) Prove that, assuming the design matrix $X$ is such that $X^\top X$ is an invertible $(d\times d)$ matrix, the estimate
   $$
   \hat\beta = \big(X^\top X\big)^{-1} X^\top Y
   $$
   so that $\hat f(X_i) = \big(\mathrm{Proj}_{\mathrm{span}(X_i)} Y\big)_i$.

   (b) Deduce that $\mathrm{Opt}_R = \mathrm{Opt}_{RF} = \dfrac{2\sigma^2 d}{n}$

   (c) Prove that $\Delta_B \ge 0$.

   (d) Prove that $\Delta_V \ge 0$ (one can use $\mathbb{E}\!\left[\big(X^\top X\big)^{-1}\right] - \big(\mathbb{E}[X^\top X]\big)^{-1}$ is s.d.p.)

下面用一张表 + 要点速览，把三位“表亲”讲清楚：$\mathrm{Err}_F$、$\mathrm{Err}_{RF}$、$\mathrm{Err}_R$。

### 一张表看全局

| 名称                                  | 评估点（X 的来源）                               | 随机性来自哪里                            | 条件/设置                      | 典型公式（平方损失）                                                                  | 直觉定位            |
| ----------------------------------- | ---------------------------------------- | ---------------------------------- | -------------------------- | --------------------------------------------------------------------------- | --------------- |
| $\mathrm{Err}_F$（Fixed）             | **只在训练的这组固定 $X_1,\dots,X_n$** 上评估        | 来自 $Y,Y'$（噪声），**不**平均 $X$          | 设计点固定                      | $\sigma^2+\frac1n\sum_i \big(\text{Bias}_i^2+\text{Var}_i\big)$             | “这批点上我到底好不好？”   |
| $\mathrm{Err}_{RF}$（Repeated Fixed） | **仍在各自训练集的 (X_i)** 上评估，但**对设计$X$** 再取平均 | 来自 $Y,Y'$ 和 **随机的训练设计 $X$**        | (X_i) 独立（不一定同分布也常设 i.i.d.） | $\sigma^2+\mathbb{E}_{(X_i)}\big[\text{Bias}(X_1)^2+\text{Var}(X_1)\big]$  | “不同训练集的平均训练点表现” |
| $\mathrm{Err}_R$（Random）            | **在新点 $X'\sim \mathcal{L}(X)$** 上评估      | 来自 $Y,Y'$、训练设计 $X$、以及**新样本点 $X'$** | $X_i$ i.i.d.（统计学习常用）       | $\sigma^2+\mathbb{E}_{(X_i),X'}\big[\text{Bias}(X')^2+\text{Var}(X')\big]$ | “泛化到分布上任意新点的表现” |

这里 $\text{Bias}(x)=f(x)-\mathbb{E}[\hat f(x)\mid \text{训练 }X]$，$\text{Var}(x)=\mathrm{Var}[\hat f(x)\mid \text{训练 }X]$。

---

### 三者的**相同点**

* 都度量同一个学习器 $\hat f$ 的均方误差，只是**对哪些随机量取期望**不同。
* 在同样的建模假设下，都可分解为
  $$\quad\boxed{\ \text{噪声 } \sigma^2\ +\ \text{（条件）偏差}^2\ +\ \text{（条件）方差}\ }.$$
* 若考察的是同一个训练器（同一训练流程），它们之间可以用统一的框架比较：$\mathrm{Err}_R$ 与 $\mathrm{Err}_F$ 的差额就是“换点”的**偏差增量 ($\Delta_B$)** 和**方差增量 ($\Delta_V$)**。

### 三者的**不同点**

* **评估点不同**：
  $\mathrm{Err}_F$ 只看**这批**训练点；
  $\mathrm{Err}_{RF}$ 还是训练点，但**对训练点本身再平均**；
  $\mathrm{Err}_R$ 看**分布里的新点 (X')**，是真正的“泛化误差”。
* **取期望的层次不同**：
  $\mathrm{Err}_F$ 不对 (X) 取期望；
  $\mathrm{Err}_{RF}$ 对训练设计 (X) 取期望；
  $\mathrm{Err}_R$ 既对训练设计 (X) 又对**新点** (X') 取期望。
* **量级关系（通常）**：
  $$
  \mathrm{Err}_F\ \le\ \mathrm{Err}_{RF}\ \le\ \mathrm{Err}_R
  $$
  直觉：在“见过的点”上最容易，换不同训练集稍难，换到“新点”最难。形式上：
  $$
  \mathrm{Err}_R
  = \mathrm{Err}_F\ +\ \underbrace{\Delta_B}_{\ge 0}\ +\ \underbrace{\Delta_V}_{\ge 0}.
  $$
* **用途场景**：
  $\mathrm{Err}_F$ 更像“同点复测”的误差（工程里常用作**重测误差**）；
  $\mathrm{Err}_{RF}$ 是对“不同训练集”的平均（**学习流程稳定性**的量化）；
  $\mathrm{Err}_R$ 则是我们关心的**泛化性能**。

### 线性最小二乘（OLS）中的一个“共识”

* 在 OLS（满秩）且同方差噪声下，三者的 **optimism**（真实误差相对训练误差的系统性低估量）都等于
  $$
  \mathrm{Opt}_F=\mathrm{Opt}_{RF}=\mathrm{Opt}_R=\frac{2\sigma^2 d}{n},
  $$
  其中 (d) 是参数维度。这给了训练误差到泛化误差的一个干净的“校正项”。

---

一句话总结（端水大师版）：
**同锅同汤（同一分解），不同碗口（取期望对象不同）**；从 $\mathrm{Err}_F$ 到 $\mathrm{Err}_R$，评估越来越“严格”，也更贴近我们真正关心的泛化表现。

  $$
  \mathrm{Opt}_F=\mathrm{Opt}_{RF}=\mathrm{Opt}_R=\frac{2\sigma^2 d}{n},
  $$
  其中 (d) 是参数维度。这给了训练误差到泛化误差的一个干净的“校正项”。

---

一句话总结（端水大师版）：
**同锅同汤（同一分解），不同碗口（取期望对象不同）**；从 $\mathrm{Err}_F$ 到 $\mathrm{Err}_R$，评估越来越“严格”，也更贴近我们真正关心的泛化表现。
