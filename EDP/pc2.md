

# 偏微分方程的变分分析

## PC2：泛函的最小化和变分公式

### 练习1：格林公式

设 $\Omega \subset \mathbb{R}^N$ 是一个类 $C^1$ 的有界正则开集。令 $w$ 为 $C^1(\Omega)$ 中的函数，则满足以下格林公式：
$$
\int_{\Omega} \frac{\partial w}{\partial x_i}(x) dx = \int_{\partial \Omega} w(x)n_i(x) ds, \tag{G1}
$$
其中 $n_i$ 是外法向量 $\vec{n}$ 的第 $i$ 分量。

基于此公式，证明：
$$
\int_{\Omega} v(x) \frac{\partial u}{\partial x_i}(x) dx + \int_{\Omega} \frac{\partial v}{\partial x_i}(x) u(x) dx = \int_{\partial \Omega} v(x)u(x)n_i(x) ds, \quad \forall v, u \in C^1(\Omega). \tag{G2}
$$

推导出：
$$
\int_{\Omega} v(x) \text{div} \vec{u}(x) dx + \int_{\Omega} \nabla v(x) \cdot \vec{u}(x) dx = \int_{\partial \Omega} v(x) \vec{u}(x) \cdot \vec{n}(x) ds, \quad \forall v \in C^1(\Omega), \vec{u} \in C^1(\Omega)^N, \tag{G3}
$$
以及：
$$
\int_{\Omega} v(x) \Delta u(x) dx + \int_{\Omega} \nabla v(x) \cdot \nabla u(x) dx = \int_{\partial \Omega} v(x) \frac{\partial u}{\partial n}(x) ds, \quad \forall v \in C^1(\Omega), u \in C^2(\Omega), \tag{G4}
$$
其中：
- $\text{div} \vec{u} = \sum_{1 \leq i \leq N} \frac{\partial u_i}{\partial x_i}$ 是向量场的散度；
- $\nabla u = \left( \frac{\partial u}{\partial x_i} \right)_{1 \leq i \leq N}$ 是标量场的梯度；
- $\frac{\partial u}{\partial n} = \vec{n} \cdot \nabla u$ 是 $\Omega$ 边界上的法向导数。

---

### 练习2：二次能量

1. 设 $E$ 是一个 $R$-向量空间。令 $a$ 是 $E$ 上的对称正定双线性形式，$L$ 是 $E$ 上的线性形式，定义能量泛函 $F : E \to R$ 为：
   $$
   \forall x \in E, \ F(x) = \frac{1}{2}a(x, x) - L(x).
   $$
   证明：$x^\star \in E$ 是 $F$ 的极小点，当且仅当：
   $$
   \forall y \in E, \ a(x^\star, y) = L(y).
   $$

证明梗概: 充分条件直接代入求导即可,必要条件须注意到$a(x,x)\geq 0$. 以及此处能使得$F$满足此性质的条件是$F$的凸性,这可以使$F$推广到非二次的情况.

2. 假设 $E = \mathbb{R}^N$。此时存在对称正定矩阵 $A \in M_N(\mathbb{R})$ 和向量 $z \in \mathbb{R}^N$，使得：
   $$
   F(x) = \frac{1}{2}x^T A x - x^T z.
   $$
   求解 $F$ 在 $\mathbb{R}^N$ 上的极小值对应的方程，并指出在什么条件下存在唯一的极小值。

直接计算即可.

---

### 练习3：一维有限元

设域 $\Omega = (0, 1)$，$f \in C(\Omega)$，$k \in C^1(\Omega)$ 且满足 $k_0 = \inf k > 0$，以及一个实数 $\alpha > 0$。考虑以下边界值问题：
$$
\begin{cases}
-\left(k(x)u'(x)\right)' + \alpha u(x) = f(x), & \text{在 } (0, 1) \text{ 上}, \\
u(0) = u(1) = 0.
\end{cases} \tag{1}
$$
假设该问题存在唯一解 $u \in C^2(\Omega)$，且存在常数 $C_1 > 0$，使得对任意 $f \in C(\Omega)$，有：
$$
\|u''\|_{L^2(0,1)} \leq C_1 \|f\|_{L^2(0,1)}.
$$

#### 1. **函数空间的框架**

定义以下函数空间：
$$
V = \{v \in C(\Omega) : \text{存在分区 } (\omega_i) \text{ 使得 } v \in C^1(\omega_i)\},
$$
以及 $V_0$ 的子空间：
$$
V_0 = \{v \in V : v(0) = v(1) = 0\}.
$$

证明：若 $u, v \in V$，则有：
$$
\int_0^1 u'(x)v(x)dx = -\int_0^1 u(x)v'(x)dx + \left[u(x)v(x)\right]_0^1.
$$

证:分段用分部积分即可.

---

#### 2. **变分形式的书写**

a. 将问题(1)在 $V_0$ 上写为以下变分形式：
$$
\text{求 } u \in V_0 \text{ 使得 } a(u, v) = \ell(v), \quad \forall v \in V_0, \tag{2}
$$
其中 $a$ 是一个双线性形式，$\ell$ 是一个线性形式。

解:经典套路,两边乘$v$后用分部积分公式再使用$v$的在边界为$0$即可.

b. 假设问题(2)的解 $u \in C^2(\Omega)$，证明问题(1)与问题(2)是等价的。

证:经典套路,用边界项为0的条件和密度定理即可.

---

#### 3. **内部变分近似**

设 $V_{h,0}$ 是 $V_0$ 的一个有限维子空间。

a. 证明：映射
$$
u \in V \mapsto \left(\int_0^1 (u(x))^2dx + \int_0^1 (u'(x))^2dx\right)^{1/2}
$$
定义了 $V$ 上的一个范数，记为 $\| \cdot \|_V$。

证:考虑其二次型,显然是双线性对称的,因此其诱导的范数是良定义的.

b. 证明：双线性形式 $a$（定义于(2)中）在 $V_0$ 上是强制的。

由此推导出：近似变分问题
$$
\text{求 } u_h \in V_{h,0} \text{ 使得 } a(u_h, v_h) = \ell(v_h), \quad \forall v_h \in V_{h,0}, \tag{3}
$$
有唯一解。

证:回顾Coercive的定义:

Une forme bilinéaire $X\times X\to \mathbb {R}$ est dite coercive si elle vérifie :

$$ \exists \,\alpha >0,\quad \forall \,x\in X:\qquad a(x,x)\geqslant \alpha \|x\|^{2}.$$

简单来说coercive就是在无穷处取值无穷,并注意到:

$$
a(u, u) = \int_0^1 k(x) (u'(x))^2 \, dx + \alpha \int_0^1 u^2(x) \, dx \geq k_0 \|u'\|^2_{L^2(0,1)} + \alpha \|u\|^2_{L^2(0,1)} \geq \min(k_0, \alpha) \|u\|^2_V.
$$

最后使用Lax-Milgram定理即可得到结果.

c. 当 $\alpha = 0$ 时，如何证明(3)存在唯一解？

证:此时Coercivité条件化为:

$$
a(u, u)\geq k_0 \|u'\|^2_{L^2(0,1)} .
$$

我们只需要证明$\exist C$, t.q. $k_0 \|u'\|^2_{L^2(0,1)}\geq C\|u\|^2_{L^2(0,1)}$.

接下来用LN公式和柯西不等式放缩即可.

注意这里有两种可能性,分别是$V$和$V_0$中的范数,证明方法一样.

---

#### 4. **有限元离散化**

设 $[0, 1]$ 的均匀网格 $(x_j)_{0 \leq j \leq N+1}$，网格步长 $h = \frac{1}{N+1}$，并定义近似空间：
$$
V_{h,0} = \{v \in C^0([0, 1]) : v|_{[x_j, x_{j+1}]} \in P_1, \ \forall 0 \leq j \leq N, \ v(0) = v(1) = 0\} \subset V_0. \tag{4}
$$

a. 证明：求解变分问题(3)等价于求解一个线性系统：
$$
A_h U_h = b_h,
$$
其中：
$$
A_h = K_h + \alpha M_h,
$$
并明确矩阵 $K_h$、$M_h$ 以及向量 $U_h$、$b_h$ 的表达式。

b. 实际计算中，如何构造矩阵 $K_h$、$M_h$ 和右端项 $b_h$？

c. 证明：$A_h$ 是对称正定矩阵。

d. 在使用 $P_2$ 有限元时，定义矩阵 $K_h$ 和 $M_h$。

---

#### 5. **误差估计**

证明：存在常数 $C_2 > 0$，使得：
$$
\|u - u_h\|_{L^2(0,1)} \leq \|u - u_h\|_V \leq C_2 h \|f\|_{L^2(0,1)},
$$
其中 $u$ 是问题(2)的解，$u_h$ 是问题(3)在空间 $V_{h,0}$ 中的解。
