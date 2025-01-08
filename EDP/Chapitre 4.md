# EDP
## 4.2 平方可积函数
定理 4.2.1 配备了由(4.3)式定义的标量积的$L^2(\Omega)$是希尔伯特空间.

结论 4.2.6 令$u \in L^2()\Omega$. 若存在常数$C$使得,对$i\in[1,i]\cap \mathbb N$有:
$$\forall \phi \in D(\Omega), |\int_\Omega \frac{\partial \phi}{\partial x_i}d\Omega|\leq C ||\phi||_{L^2(\Omega)},$$
那么$u$在$e_i$方向上若可导.
## 4.3 索博列夫空间
### 4.3.1 $H^1(\Omega)$空间和$H^1_0(\Omega)$空间
定义 4.3.1 $H^1(\Omega)$空间是在所有方向都有弱导数的$L^2(\Omega)$函数的集合.
$$H^1(\Omega):=\{v\in L^2(\Omega),\frac{v}{x_i}\in L^2(\Omega) \forall i \in [0,1] \}.$$

例 对于$\Omega:=]0,1[$, $x\to x^\beta \in H^1(]0,1[)$ 当且仅当 $\beta > 1/2.$

例 若$\Omega$开且有界, $C^1(\overline{\Omega})\subset H^1(\Omega)$. 

事实上,如果$f \in C^0(\overline{\Omega})$, 那么$f \in L^2(\Omega)$.在$f \in C^1(\overline{\Omega})$的情况下,其作为与导数有相同形式的弱导数是$C^0$的,因此$f$与其导数都是$L^2$的.

一个更有趣的情况是,只要一个函数是分段$C^1$的,那么它就是$H^1$的.

定理 4.3.1 若$\Omega$在$\mathbb R^d$中开且有界,我们做一个划分$\overline{\Omega}=\overline{\Omega_1}\cup\overline{\Omega_2}$,其中$\Omega_1$,$\Omega_2$是互不相交的两个开集.我们记$\Gamma$为$\Omega_1$与$\Omega_2$的分界面:$\Gamma:=\overline{\Omega_1}\cap\overline{\Omega_2}$.

我们有
$$v\in  C^0(\overline{\Omega}) ⇔ v\in H^1(\Omega).$$
其中$\nabla v :=\chi_{\Omega_1}\nabla v_1+\chi_{\Omega_1}\nabla v_1$. $\chi_{\Omega_i}$是$\Omega_i$上的示性函数,$v_i:=v|_{\Omega_1}\in C^1(\overline{\Omega_i})$.

证明梗概:我们验证$v$满足结论4.2.6中的判据:首先,$v\in  C^0(\overline{\Omega})$ 因此 $v,v_i \in L^2(\Omega)$. 现在我们验证$v$满足结论4.2.6中的不等式,利用$v$的定义,格林公式和边界条件:
$$
\int_{\Omega} v \frac{\partial \varphi}{\partial x_i} d\Omega = -\int_{\Omega_1} \frac{\partial v_1}{\partial x_i} \varphi d\Omega  - \int_{\Omega_2} \frac{\partial v_2}{\partial x_i} \varphi d\Omega + \int_{\Gamma} (v_1 - v_2) \varphi n_1 \cdot e_i d\Gamma.$$ 

前两项显然满足(柯西不等式),我们接下来利用反证法证明:

如果$v$不是连续的,那么不存在$f\in L^2(\Omega)$使得
$$\int_{\partial \Omega}(v_1-v_2)\varphi n_1\cdot e_i d\Gamma=\int_\Omega f \varphi d\Omega.$$

因此命题得证.

通过使用完全相同的证明方法以及稍后推广到 $H^1$ 中的函数的Green 公式 ，可以将前面的结果推广到在每个子域中都是 $H^1$ 的函数.在这个定理中，我们还使用了 $H^1$ 函数的迹 $v_\Gamma$ 的概念，这将在第 4.3.3 节中介绍。

定理 4.3.2 设 $\Omega$ 是 $\mathbb{R}^d$ 中的一个有界开集，被划分为两个不相交的子域 $\Omega_1$ 和 $\Omega_2$，使得 $\Omega = \Omega_1 \cup \Omega_2$。我们记 $\Gamma$ 为 $\Omega_1$ 和 $\Omega_2$ 之间的界面：$\Gamma \equiv \Omega_1 \cap \Omega_2$。

设 $v$ 满足 $v_i \equiv v|_{\Omega_i} \in H^1(\Omega_i)$，对于 $i = 1, 2$，则
$$
v|_{\Gamma}^1 = v|_{\Gamma}^2 \iff v \in H^1(\Omega)
$$
并且在这种情况下
$$
\nabla v = \chi_{\Omega_1} \nabla v_1 + \chi_{\Omega_2} \nabla v_2
$$
其中 $\chi_{\Omega_i}$ 是 $\Omega_i$ 的特征函数。

我们现在引入标量积,

定义 $\forall f, g \in H^1(\Omega)$,我们称: 
$$  (f, g)_{H^1(\Omega)} := \int_{\Omega} f g + \nabla f \cdot \nabla g \, d\Omega,
$$
为标量积.

Remarque:这是良定义的,因为$f,g\in H^1(\Omega)$,因此$f,g\in L^2(\Omega)$ 且 $\nabla f, \nabla g\in (L^2(\Omega))^d.$

定义 $\forall f \in H^1(\Omega)$,我们定义模: 
$$ || f||_{H^1(\Omega)} := (\int_{\Omega} (|f|^2  + |\nabla f |^2 )d\Omega)^{1/2}.
$$

显然:$\forall f \in H^1(\Omega)$, $|| f||_{L^2(\Omega)} \leq || f||_{H^1(\Omega)}$ 且 $|| \nabla f||_{L^2(\Omega)} \leq || f||_{H^1(\Omega)}$.

定理4.3.3 配备了上述标量积的$H^1(\Omega)$ 是希尔伯特空间.

证:我们只需要证明$H^1(\Omega)$对由标量积引入的模是完备的. 取一$H^1(\Omega)$中的柯西列$(f_n)_{n\in \mathbb N}$,因为$|| f||_{L^2(\Omega)} \leq || f||_{H^1(\Omega)}$,则此柯西列显然也是$L^2(\Omega)$中的柯西列,并令其收敛到$f$. 类似地,由定理4.2.1, $(\nabla f_n)_{n\in \mathbb N}\in (L^2(\Omega))^d$. 我们只需要证明$f$存在弱梯度$\nabla f = F$. 事实上,$\forall \phi \in (L^2(\Omega))^d$, 我们都有

$$
(f, \nabla \cdot \varphi)_{L^2(\Omega)}=\lim_{n\to+\infty}(f_n, \nabla \cdot \varphi)_{L^2(\Omega)}=-\lim_{n\to+\infty}(\nabla f_n, \varphi)_{(L^2(\Omega))^d}=(-F, \varphi)_{(L^2(\Omega))^d}.
$$

我们证明了
$$
||f-f_n||_{H^1(\Omega)}^2=||f-f_n||_{L^2(\Omega)}^2+||F-\nabla f_n||_{L^2(\Omega)}^2 \to 0
$$

因此命题得证.

我们因此得到以下关于稠密性的结论:

定理 4.3.4 $D(\bar \Omega) 在H^1(\Omega)$中稠密.

Les fonctions de $D(\Omega)$ sont $C^\infty$ à support compact dans $\bar \Omega$. Elles ne s'annulent donc pas nécessairement sur le bord de $\Omega$ quand $\Omega$ a un bord. En revanche, les fonctions de $D(\Omega)$ s'annulent sur le bord de $\Omega$ (le support, fermé, de ces fonctions est inclus dans l'ouvert $\Omega$). Par conséquent, si $\Omega \subsetneq \mathbb{R}^d$, alors $D_0(\Omega) \subsetneq D(\Omega)$.

定义 我们称$H_0^1(\Omega)$为$D(\Omega)$在$H^1(\Omega)$中的陪集, 也就是
$$H_0^1(\Omega):=\overline{D(\Omega)}^{H^1(\Omega)}.$$

如此,由定义$D(\bar \Omega)$ 在$H_0^1(\Omega)$中稠密, 且如果$\Omega=\mathbb R^d$,$H_0^1(\mathbb R^d)=H^1(\mathbb R^d).$ 反之则未必.

命题 4.3.5 配备了上述标量积的$H_0^1(\Omega)$ 是希尔伯特空间.

### 4.3.2 $H(\Omega,div)$,$H^m(\Omega)$和$H_0^m(\Omega)$空间

我们先引入分量都是$L^2(\Omega)$且存在弱散度的向量场:

定义 4.3.3 $H(\Omega,div)$是一个$(L^2(\Omega))^d$向量场的集合,且存在弱散度:
$$H(\Omega,div):=\{U\in (L^2(\Omega))^d, \nabla \cdot U \in L^2(\Omega) \}.$$

注意,$(H^1(\Omega))^d \subset H(\Omega,div).$

我们进一步地有如下结果:

定理 4.3.6 $\forall U,V \in H(\Omega,div)$,配备了如下标量积
$$(U,V)_{ H(\Omega,div)}:=(U,V)_{ (L^2(\Omega))^d}+(\nabla \cdot U,\nabla \cdot V)_{ L^2(\Omega)}$$
的$H(\Omega,div)$是希尔伯特空间.

定理 4.3.7 $(D(\bar \Omega))^d$ 在$H(\Omega,div)$中稠密.

我们现在用递归法定义$H^m(\Omega)$索博列夫空间.

定义 4.3.4 $H^m(\Omega)$ 是$H^{m-1}(\Omega)$中所有$m-1$阶在所有方向弱可导的函数的集合:
$$H^m(\Omega):=\{v\in H^{m-1}(\Omega),\partial^\alpha v \in L^2(\Omega)\forall \alpha \in \mathbb N ^m, |\alpha|=m\}.$$

这里我们回忆一下记号 :
$$
\forall \alpha = (\alpha_1, \ldots, \alpha_m) \in \mathbb{N}^m, \quad |\alpha| \coloneqq \sum_{k=1}^{n} |\alpha_k|,
$$
以及
$$
\frac{\partial^{\alpha}}{\partial \alpha} \coloneqq \frac{\partial^{|\alpha|}}{\partial x_1^{\alpha_1}\cdots \partial x_n^{\alpha_n}}.
$$

例 对于$\Omega:=]0,1[$, $x\to x^\beta \in H^m(]0,1[)$ 当且仅当 $\beta > - 1/2+m.$

进一步的,若$\Omega$为紧开集,$C^m(\overline{\Omega})\subset H^m(\Omega)$. 并且我们有以下序列:
$$D(\Omega)\subset H^m(\Omega)\subset H^1(\Omega)\subset L^2(\Omega).$$

Remarque 4.3.1 使用插值定理可以进一步定义索博列夫泛函 $H^s(\Omega)$, $s \in \mathbb{R}$ (参考 [7]). 这超出了课程内容, 但我们会回到
$$
s < s' \implies H^{s'}(\Omega) \subset H^s(\Omega).
$$
进一步地, $H^s(\Omega)$ 中满足 $m < s < m+1$, $m \in \mathbb{N}$ 的函数比在 $H^m(\Omega)$ 而不在 $H^{m+1}(\Omega)$中的函数更规则 :
$$
H^{m+1}(\Omega) \subset H^s(\Omega) \subset H^m(\Omega) \quad , \quad m < s < m+1.
$$

定理 4.3.8 $\forall f, g \in H^m(\Omega)$ 配备了标量积
$$
(f, g)_{H^m(\Omega)} \coloneqq (f, g)_{H^{m-1}(\Omega)} + \sum_{\alpha \in \mathbb{N}^m, |\alpha| = m} (\partial^\alpha f, \partial^\alpha g)_{L^2(\Omega)}
$$
的$H^m(\Omega)$ 是希尔伯特空间.

