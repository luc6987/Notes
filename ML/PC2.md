来源：

PC2. Foundations of Machine Learning. (MDC_51006_EP - 2025-2026)

**Exercise 1 (Link Estimation/Prediction)** 

Assume we observe a sample $((X_i, Y_i))_{i=1}^n$ where $Y_i \in \{-1, 1\}$ and $X_i \in \mathbb{R}^d$ are independent random variables following the same law as a generic pair $(X, Y)$. We want to predict $Y$ from $X$ using a predictor $f : \mathbb{R}^d \to \{-1, 1\}$ and we measure the quality of our predictor using the 0/1 loss:

$$
\ell_{0/1}(y, y') = \mathbf{1}_{y \neq y'}.
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
\mathbb{E}\!\left[ \ell_{0/1}\big(Y, \hat f(X)\big) \right]
- \mathbb{E}\!\left[ \ell_{0/1}\big(Y, f^\star(X)\big) \right]
  \le \mathbb{E}\!\left[ \lvert \widehat{Y\mid X} - {Y\mid X} \rvert_1 \right]
  \le \left( \mathbb{E}\!\left[2\,\mathrm{KL}\big(Y\mid X,\ \widehat{Y\mid X}\big)\right]\right)^{1/2}.
$$

1. Prove that for any predictor $f$,
   $$
   \mathbb{E}\!\left[\ell_{0/1}\big(Y, f(X)\big)\right]
   = \mathbb{E}_X\!\left[ (1- p_1(X)) + \big(2p_1(X)-1\big)\,\mathbf{1}_{f(X)=-1} \right]
   $$

2. Deduce that
   $$
   \mathbb{E}\!\left[\ell_{0/1}\big(Y, \hat f(X)\big)\right]
   -\mathbb{E}\!\left[\ell_{0/1}\big(Y, f^\star(X)\big)\right]
   \le 2\,\mathbb{E}_X\big[\lvert p_1(X)- \hat p_1(X)\rvert\big] = \mathbb{E}_X\big[\lvert p(X)- \hat p(X)\rvert_1\big]
   $$

3. Finish the proof using $\lvert P - Q\rvert_1 \le \sqrt{2\,\mathrm{KL}(P,Q)}$

---

**Exercise 2 (Linear Regression and Design)** Assume we observe some points $((X_i, Y_i))_{i=1}^n$ with

$X_i \in \mathbb{R}^p$ and $Y_i \in \mathbb{R}$ are assumed to be independent copies of a generic $(X,Y)$, regressing those points means estimating $\mathbb{E}[Y \mid X]$ at least on the $X_i$. Indeed, there are two classical problems that can be studied

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

   (b) Verify that if $\hat f = \arg\min_g \sum_{i=1}^n \ell\big(Y_i, f(X_i)\big)$ then $\mathbb{E}_{(Y_i)_{i=1}^n}[\mathrm{Err}_{Emp}] \le \mathrm{Err}_F$ and $\mathbb{E}_{((X_i,Y_i))_{i=1}^n}[\mathrm{Err}_{Emp}] \le \mathrm{Err}_{RF}$.

2. We will now assume that $\ell$ is the $\ell_2$ loss: $\ell(Y, f(X)) = \lvert Y - f(X)\rvert^2$.

   (a) Prove that in the fixed design setting:
   $$
   \mathrm{Err}_F = \sigma^2 + \frac{1}{n}\sum_{i=1}^n \big(f(X_i)- \mathbb{E}_{(Y_i)_{i=1}^n}[\hat f(X_i)]\big)^2
   + \frac{1}{n}\sum_{i=1}^n \mathrm{Var}_{(Y_i)_{i=1}^n}\!\big[\hat f(X_i)\big]
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

3. We define the optimism $\mathrm{Opt}_F$ (respectively $\mathrm{Opt}_{RF}$) as the difference between $\mathrm{Err}_F$ (respectively $\mathrm{Err}_{RF}$) and the corresponding expectation of $\mathrm{Err}_{Emp}$.

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

4. We consider now the random design and define $\mathrm{Opt}_R$ in the same way.

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

5. We consider now the linear model $f_\beta(X') = \langle X', \beta\rangle$ and pick $\beta$ by minimizing the empirical loss.

   (a) Prove that, assuming the design matrix $X$ is such that $X^\top X$ is an invertible $(d\times d)$ matrix, the estimate
   $$
   \hat\beta = \big(X^\top X\big)^{-1} X^\top Y
   $$
   so that $\hat f(X_i) = \big(\mathrm{Proj}_{\mathrm{span}(X_i)} Y\big)_i$.

   (b) Deduce that $\mathrm{Opt}_R = \mathrm{Opt}_{RF} = \dfrac{2\sigma^2 d}{n}$

   (c) Prove that $\Delta_B \ge 0$.

   (d) Prove that $\Delta_V \ge 0$ (one can use $\mathbb{E}\!\left[\big(X^\top X\big)^{-1}\right] - \big(\mathbb{E}[X^\top X]\big)^{-1}$ is s.d.p.)

