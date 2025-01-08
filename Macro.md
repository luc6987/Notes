# Macroéconomie

## 生产函数和资本积累
这是一切的起点,关键是建立产品(output)和生产(inputs)的关系. 模型假设这样一个函数存在:

$Y(t)=F(K(t),L(t),A(t)),$

其中$ t \in [0,+ \infty[ $, $Y(t)$为生产,$K(t)$为资本积累,$L(t)$是工人数量,$A(t)$量化科技.所有的企业都遵循这个公式. 为了简单,我们将公司数量归一化,这样$Y(t)$就代表GDP.

我们假设$L(t)$是工人数量,$A(t)$科技是外因(exogène)的,因此满足:

$L(t)=L(0)\exp(nt),A(t)=A(0)\exp(g_At)$.

$K(t)$资本积累的演化反之则是endogène的,也就是由一个动力学模型决定. 在一个时间区间$[t,t+dt]$里大致减少$\delta \cdot dt$. 为了看到资本积累如何演化,我们首先注意到资本积累在单位时间的净变化是两个流的平衡导致的:在时刻$t$的投资流$I(t)$和减少流$\delta K(t)$,

$\dot K(t)=I(t)-\delta K(t),\delta > 0$,

其中$\dot K(t):=\partial K/\partial t$.

我们称$\dot K(t)$为净投资,$I(t)$为生投资.

定义：经济模型的内生变量(endogène)是模型试图解释的变量；而外生变量(exogène)则假定是在所研究的系 统之外决定的。这种区分是方法论上的，因为某一变量在某一类模型中（通常）是外生的，但在另一类模型中却是内生的（例如，技术进步在索洛模型及其变体中是外生的，但在第 2 章研究的研发模型中却是内生的）。

定义 如果技术进步只直接影响劳动生产率（在 $Y (t) = F(K(t) ,A(t) L(t))$ 型生产函数中），那么它就是哈罗德意义上的中性技术进步、 如果只直接影响资本生产率（$Y (t) = F(A(t) K(t) ,L(t))$），则为索洛意义上的中性；如果考虑到其他要素，对总产出产生乘法影响（ $Y(t) =A(t)F(K(t) ,L(t))$)，则为希克斯意义上的中性。因此，希克斯意义上的中性技术进步的理论概念对应于经验上的全要素生产率概念，也称为索洛残差。

大部分增长模型认为技术增长是哈罗德中性的.

## 生产函数假设

我们假设$F:\mathbb R ^3_+ \to \mathbb R_+$是一个对$K,L$二阶可导且满足以下假设

假设1:

$F(0,L(t),A(t))=F(K(t),0,A(t))=0$,

$F_K(\cdot),F_L(\cdot)>0$,

$F_{KK}(\cdot),F_{LL}(\cdot)<0$,

第一个条件是“基本条件“.

假设2:

$F$满足Inada条件:

$\forall L>0, \lim _{K\to 0} F_K(\cdot)=\infty$ and $\lim _{K\to \infty} F_K(\cdot)=0$,

$\forall K>0, \lim _{L\to 0} F_L(\cdot)=\infty$ and $\lim _{L\to \infty} F_L(\cdot)=0$.

假设3:
F有一个资本和劳动力规模收益不变(REC)的产出,也就是说

$\forall \lambda>0,F(\lambda K(t),\lambda L(t),A(t))=\lambda F(K(t),L(t),A(t))$.

REC也被等价地称为一阶齐次的.

从REC中,我们可以获得这样一些重要结果:

* 欧拉定理

$F_K(\cdot)K(t)+F_L(\cdot)L(t)=F(K(t),L(t),A(t))$

证明:对REC两边对$\lambda$微分并取$\lambda=1$.

* 两个变量的边际产出是零阶其次的

$F_K(\lambda K(t),\lambda L(t),A(t))=F_K(K(t),L(t),A(t))$
$F_L(\lambda K(t),\lambda L(t),A(t))=F_L(K(t),L(t),A(t))$

证明:对REC两边对K,L微分.