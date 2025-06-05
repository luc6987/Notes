# Book

# chapter 1

A bandit problem: a sequential game between a learner and an environment.

Horizon: The game is played over $n$ rounds, where $n$ is a positive natural number called the horizon.

Arms : set $\mathcal{A}$ where learner choose $A_t \in \mathcal{A}$ at round $t \in [n]$.

Rewards: at round $t$, the environment returns a reward $X_t \in \mathbb{R}$, which is a function of the arm chosen by the learner and the state of the environment.

History: $H_t = (A_1, X_1, A_2, X_2, \ldots, A_t, X_t)$.

Policy: a function $\pi$ that maps the history $H_t$ to an arm $A_t = \pi(H_t)$.

Environment: a function $E$ that maps the history $H_t$ to a reward $X_t = E(H_t)$.

Objective: maximize $\mathbb{E}[X_n]$.

Regret: the regret of the learner relative to a policy $\pi$(not necessarily that followed by the learner) is the difference between the total expected reward using policy $\pi$ fot $n$ rounds and the total expected reward collected by the learner over n rounds.  

$$\text{Regret}(\pi) = \mathbb{E}\left[\sum_{t=1}^{n} E(r^\pi_t) - \sum_{t=1}^{n} r^{learner}_t\right],$$

where $r^\pi_t$ is the reward at round $t$ if the learner had followed policy $\pi$, and $r^{learner}_t$ is the reward at round $t$ collected by the learner.

The regret relative to a set of policies $\Pi$ is the maximum regret relative to any policy $\pi \in \Pi$ in the set.
$$\text{Regret}(\Pi) = \max_{\pi \in \Pi} \text{Regret}(\pi).$$

Competitor class: $\Pi$.

## Example-stochastic Bernoulli bandit
 $\mathcal{A}=\{1,2,...,k\}$, 

 $X_t \in \{0,1\}$,

 $\exists \mu \in [0,1]^k$ s.t. $P(X_t=1|A_t=a) = \mu_a$.

 If we know the mean vector, our policy will be $\pi_t^*(a) = \arg\max_{a \in \mathcal{A}} \mu_a$. This gives us a natural competitor class $\Pi^* = \{\pi^*_t\}$, where $\pi_t$ is the policy that chooses the arm with the highest mean at round $t$.

 For any given policy $\pi$, the regret is thus:
$$\text{Regret}(\pi) = n\arg\max_{a \in \mathcal{A}} \mu_a - \mathbb{E}\left[\sum_{t=1}^{n} \mu_{\pi_t}\right].$$

Worst-case regret: the maximum regret over all policies in the competitor class $\Pi^*$.
$$\text{Regret}(\Pi^*) = \max_{\pi \in \Pi^*} \text{Regret}(\pi) = n\max_{a \in \mathcal{A}} \mu_a - \min_{\pi \in \Pi^*}\mathbb{E}\left[\sum_{t=1}^{n} \mu_{\pi_t}\right].$$

The objective is to study the behavior of the regret as a function of $n$.

A good learner achieves a regret that grows sublinearly with $n$, i.e., $\text{Regret}(\Pi^*) = o(n)$.

Insight: A large environment class corresponds to less knowledge by the learner. A large competitor class means the regret is a more demanding criteria. Some care is sometimes required to choose these sets appropriately so that 
* (a) guarantees on the regret are meaningful and
* (b) there exist policies that
make the regret small.

stochastic stationary bandit: the environment is independent of the history, i.e., $X_t$ is independent of $H_{t-1}, H_{t-2}, \ldots$.

adversarial bandit: the environment is not independent of the history, i.e., $X_t$ can depend on $H_{t-1}, H_{t-2}, \ldots$.

## Some applications

1. A/B testing: the learner is a website that wants to maximize the click-through rate of its users. The arms are different versions of the website, and the rewards are the click-through rates. The environment is the user behavior, which can change over time.
2. Advert placement: the learner is a website that wants to maximize the revenue from ads. The arms are different ad placements, and the rewards are the revenue from the ads. The environment is the user behavior, which can change over time.
3. Recommendation Services: the learner is a recommendation system that wants to maximize the click-through rate of its users. The arms are different recommendations, and the rewards are the click-through rates. The environment is the user behavior, which can change over time.
4. Network routing: the learner is a network that wants to maximize the throughput. The arms are different routing policies, and the rewards are the throughput. The environment is the network traffic, which can change over time.
5. Dynamic pricing: the learner is a seller that wants to maximize the revenue. The arms are different prices, and the rewards are the revenue. The environment is the demand, which can change over time.
6. Waiting problems: the learner is a system that wants to minimize the waiting time. The arms are different policies, and the rewards are the waiting times. The environment is the arrival process, which can change over time.
7. Resource allocation: the learner is a system that wants to maximize the throughput. The arms are different resource allocation policies, and the rewards are the throughput. The environment is the resource availability, which can change over time.
8. Tree search: the learner is a search engine that wants to maximize the click-through rate of its users. The arms are different search results, and the rewards are the click-through rates. The environment is the user behavior, which can change over time.


# chapter 2

# chapter 3

# chapter 4 Stochastic Bandits

## 4.1 Core assumptions

A stochastic bandit is a collection of distributions $\nu =(P_a:a\in \mathcal{A})$, where $\mathcal{A}$ is the set of available actions.

The learner and the environment interact sequentially over $n$ rounds. In each round $t\in \{1,2,\ldots,n\}$, the learner chooses an action $A_t \in \mathcal{A}$, which is fed to the environment. The intereaction between the learner (or policy) and the environment induces a probability measure on the sequence of outcomes $(A_1, X_1, A_2, X_2, \ldots, A_n, X_n)$. Usually the horizon $n$ is finite, but sometimes we allow the interaction to continue indefinitely, i.e., $n=\infty$. THe sequence of outcomes should satisfy the following assumptions:

1. the conditional distribution of reward $X_t$ given $A_1,X_1,\ldots,A_t,X_t$ is $P_{A_t}$, which captures the intuition that the environment samples $X_t$ from $P_{A_t}$ in round $t$.
2. The conditional law of action $A_t$ given $X_t$ given $A_1,X_1,\ldots,A_t,X_t$ is $\pi_t(•|X_t,A_1,X_1,\ldots,A_t,X_t)$, where $\pi_1,\ldots,\pi_t$ is a sequence of probability kernels that characterise the learner. The most important element of this assumption is the intuitive fact that the learner cannot use the future observations in current decisions.
   
The existence of such space is proven in 4.6

## The learning Objective

Maximise $S_n= \sum_{t=1}^{n} X_t$.

This is not a optimisation problem for three reasons:

1. $n$ is not fixed.
2. $S_n$ is random.
3. $P_{A_t}$ is not known by the learner.

for 1. is not a problem, since we can always fix $n$ and then let it grow.
for 2. we need to choose a proper $S_n$ that fix the reality.
for 3. we will need to involve the new conception.(see below)

## Knowledge and Environment Classes

Bandit instance $\nu=(P_a:a\in \mathcal{A})$.

Environment class: $\mathcal{E}$ is a set of distributions $\nu =(P_a:a\in \mathcal{A})$.

### Unstructured Bandits

$\mathcal{E}$ is unstructured if $\mathcal{A}$ is finite and there exist sets of distributions $\mathcal{M}$ for each $a\in \mathcal{A}$ s.t.

$$\mathcal{E}=\{\nu=(P_a:a\in \mathcal{A}):P_a\in\mathcal{M}\}$$

or, in short, $\mathcal{E}=\times_{a\in \mathcal{A}} \mathcal{M}$.

| Name                    | Symbol              | Definition                                                                                      |
|-------------------------|---------------------|-------------------------------------------------------------------------------------------------|
| Bernoulli               | $\mathcal{E}_k^B$   | $\left\{ (B(\mu_i))_i : \mu \in [0,1]^k \right\}$                                              |
| Uniform                 | $\mathcal{E}_k^U$   | $\left\{ (U(a_i,b_i))_i : a, b \in \mathbb{R}^k \text{ with } a_i \leq b_i \text{ for all } i \right\}$ |
| Gaussian (known var.)   | $\mathcal{E}_k^{N(\sigma^2)}$ | $\left\{ (N(\mu_i, \sigma^2))_i : \mu \in \mathbb{R}^k \right\}$                               |
| Gaussian (unknown var.) | $\mathcal{E}_k^N$   | $\left\{ (N(\mu_i, \sigma_i^2))_i : \mu \in \mathbb{R}^k \text{ and } \sigma^2 \in [0,\infty)^k \right\}$ |
| Finite variance         | $\mathcal{E}_k^{V(\sigma^2)}$ | $\left\{ (P_i)_i : \operatorname{Var}_{X \sim P_i}[X] \leq \sigma^2 \text{ for all } i \right\}$ |
| Finite kurtosis         | $\mathcal{E}_k^{\text{Kurt}(\kappa)}$ | $\left\{ (P_i)_i : \operatorname{Kurt}_{X \sim P_i}[X] \leq \kappa \text{ for all } i \right\}$ |
| Bounded support         | $\mathcal{E}_k^{[a,b]}$ | $\left\{ (P_i)_i : \operatorname{Supp}(P_i) \subseteq [a,b] \right\}$                           |
| Subgaussian             | $\mathcal{E}_k^{\text{SG}(\sigma^2)}$ | $\left\{ (P_i)_i : P_i \text{ is } \sigma^2\text{-subgaussian for all } i \right\}$             |

### Structured Bandits
$\mathcal{E}$ is structured if it is not unstructured.

Example: $\mathcal{A}=\{1,2\}$ and $\mathcal{E}=\{(\mathcal{B}(\theta),\mathcal{B}(1-\theta)):\theta\in[0,1]\}$

Example: (Stochastic linear bandit) let $\mathcal{A}\subset \mathbb{R}^d$ and an unknown parameter $\theta \in \mathbb{R}^d$ and

$$\mathcal{E}=\{N(\langle a,\theta\rangle):a\in\mathcal{A}\} \text{ and } \mathcal{E}=\{\nu_\theta:\theta\in\mathbb{R}^d\}$$

Example: Consider an undirected graph $G=(V,E)$ with $|V|=k$. In each round the learner chooses a path from $v_1\in V$ to $v_k\in V$. Then each edge $e\in E$ is removed with ptobability $1-\theta_e$ for unknown $\theta \in[0,1]^{|E|}$. The learner succeeds if all the paths they chose are present. The probleme can be formalised by letting $\mathcal{A}$ be the set of paths and 

$$\nu_\theta = (\mathcal{B}(\prod_{e\in a}\theta_e):a\in \mathcal{A}) \text{ and } \mathcal{E}=\{\nu_\theta:\theta\in[0,1]^{|E|}\}$$

### The Regret

Lemma: Let $\nu$ be a stochastic bandit environment. Then:

* $R_n(\pi,\nu)\geq 0$ for all policies $\pi$.
* the policy $\pi$ choosing $A_t\in \arg\max_a \mu_a$ for all $t$ satisfies $R_n(\pi,\nu)=0$;(existence of optimal policy)
* if $R_n(\pi,\nu)=0$ for some policy $\pi$, then $\mathbb{P}(\mu_{A_t}=\mu^*)=1$ for all $t\in [n]$.(uniqueness of optimal policy)

We dont always know all the environment. A relatively weak objective is to find a policy $\pi$ with sublinear regret such that
$$\lim_{n\to\infty} \frac{R_n(\pi,\nu)}{n} = 0$$
for all $\nu\in\mathcal{E}$.

Furthermore, we want to find a policy $\pi$ such that
$$R_n(\pi,\nu) \leq C n^{p}$$
for all $\nu\in\mathcal{E}$.

And another alternative objective is to find a function $C:\mathcal{E}\to\mathbb{R}^+$ and $f:\mathbb{N}\to\mathbb{R}^+$ such that
$$R_n(\pi,\nu) \leq C(\nu)f(n)$$
for all $\nu\in\mathcal{E}$.

Bayesian regret:Let $Q$ be a prior probability measure on $\mathcal{E}$. The Bayesian regret of a policy $\pi$ is defined as
$$BR_n(\pi,Q) = \mathbb{E}_Q[R_n(\pi,\nu)] = \int_{\mathcal{E}} R_n(\pi,\nu) dQ(\nu).$$

## Decomposing the Regret

suboptimality gap(or action gap or immediate regret): 

$$\Delta_a(\nu) = \mu^*(\nu) - \mu_a(\nu).$$

Regret decomposition lemma: For any policy $\pi$ and stochastic bandit environment $\nu$ with $\mathcal{A}$ finite or countable and horizon $n\in\mathbb{N}$, the regret $R_n$ of policy $\pi$ in $\nu$ satisfies

$$R_n(\pi,\nu) = \sum_{a\in\mathcal{A}} \Delta_{a}\mathbb{E}[T_{A_t}(n)]$$

where $T_a(t):=\sum_{i=1}^t \mathbb{I}(A_i=a)$ is the number of times action $a$ is selected up to time $t$.

For the case when $\mathcal{A}$ is uncountable, we need to assume $(\mathcal{A},\mathcal{G})$ is a measurable space. Given a bandit $\nu$ and a policy $\pi$, we can define the measure $G$ on $(\mathcal{A},\mathcal{G})$ by
$$G(U) = \mathbb{E}[\sum_{t=1}^n \mathbb{I}(A_t \in U)].$$

Lemma: For any policy $\pi$ and stochastic bandit environment $\nu$ with $(\mathcal{A},\mathcal{G})$ a measurable space and horizon $n\in\mathbb{N}$, the regret $R_n$ of policy $\pi$ in $\nu$ satisfies
$$R_n(\pi,\nu) = \int_{\mathcal{A}} \Delta_a(\nu) dG(a) .$$