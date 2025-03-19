# Liste des conceptions

## PC1

Court terme: période assez courte pour que l’ajustement de certaines variables à des chocs exogènes soit incomplète.

Long terme: période assez longue pour que les chocs exogènes aient été absorbés.

Ménages Ricardiens
Des agents qui peuvent librement réaffecter la consommation intertemporelle. Les ménages Ricardiens choisissent de ne pas consommer davantage aujourd’hui, afin de consommer davantage demain.

Et qui cherche à maximaliser la fontion d'utilité comme:

$$U(C_1,C_2)=log(C_1)+\beta log(C_2),$$

sous contreint:

$$C_2P_2\leq (Y_1P_1-C_1P_1)(1+i)+Y_2P_2$$

où

$$C_1(1+r)+C_2\leq Y_1(1+r)+Y_2.$$

Et finalement, la comsommation optimale est

$$C_1=\frac{1}{1+\beta}(Y_1+\frac{1}{1+r}Y_2).$$

Prennons l'approximation d'ordre première:

$$\frac{\Delta C}{\Delta r}=-1-\frac{1}{1+\beta}\frac{1}{(1+r)^2}Y_2.$$

Ménages Keynésiens
Des agents dont la consommation est limitée par une contrainte de crédit contraignante. Soit ils ne peuvent pas du tout emprunter, soit le montant qu’ils peuvent emprunter est limité. Les ménages Keynésiens consomment aujourd’hui autant qu’ils le peuvent.

## PC2

### Rappel: PIB Nominal / PIB Réel

$Y_t$: PIB réel:le nombre de paniers de bien produits par l’économie.

$P_t$: Niveau des prix: le prix d'un panier de biens donné

$\pi_t=\frac{P_t-P_{t-1}}{P_{t-1}}$: Taux d'infaltion

PIB nominal: $P_tY_t$.

Le taux d'intérêt réel: $1+r_t=\frac{1+i_t}{1+\pi_{t+1}}$.(définition)

En logarithmes : $r_t=i_t-\pi_{t+1}$ (l'équation de Fisher).

### Les composantes de la demande

Rappelons la décomposition du PIB par les dépenses:
$$𝑌 = 𝐶 + 𝐼 + 𝐺 + 𝑁𝑋$$
* $𝑌$ : PIB nominal
* $𝐶$: Consommation
* $𝐼$: Investissement
* $𝐺$: Dépenses du gouvernement
* $𝑁𝑋$: Exportations nettes

Qu’est ce qui détermine les composants de la demande?
Les différents composants de la demande dépendent du niveau de revenu réel $𝑌$ et du taux d’intérêt réel $𝑟$.

Pour les ménages keynésiens :$C^K(Y^K)=Y^K$, pour les ménages ricardien $C^R(Y^R,r)=Y^R$. Et pour simplifier, o.s.q. $C=C(Y,r),C'_Y>0,C'_r<0$.

Une entreprise produit $𝑓(𝐾)$ en utilisant du capital $K$. o.s.q. $f'>0,f''<0$.Supposons que l’entreprise emprunte une quantité $𝐼$ pour financer $𝐾$, c’est-à-dire $𝐾 = 𝐼$. Après la production, elle doit rembourser $𝐼(1 + 𝑟)$ dans la période suivante. Le programme de maximisation:$max \pi=f(I)-I(1+r)$. c'est à dire: $f'(I)=1+r$ ou $I=I(r)=f'^{-1}(1+r)$ qui est une fonction décroissante. 

### Dérivation tion de la Demande Agrégée

le comportement des consommateurs et des entreprises:
$$C(Y,r)=C_0+C_YY+C_rr$$
$$I(r)=I_0+I_rr$$

équilibre Investissement-Épargne (courbe IS):
$$Y=C(Y,r)+I(r)+G+NX.$$

À l’équilibre du PIB: 
$$Y_0=C_0+I_0+G_0+NX_0.$$

On peut différentier l’équation:

$$Y_0+\Delta Y=C_0+C_Y\Delta Y+C_r\Delta r+I_0+I_r\Delta r+G_0+NX_0+\Delta S$$

ou $\Delta S=\Delta G +\Delta NX$ et

$$\Delta Y=C_Y\Delta Y+C_r\Delta r+I_r\Delta r+\Delta S.$$

Considérons la différencier de la courbe IS:

$$\Delta Y=C_Y\Delta Y+C_r\Delta r+I_r\Delta r.$$

En logarithmes :

$$y=-\sigma(r-r^*)$$

où $y=\frac{\Delta Y}{Y}$, $\Delta r=r-r^*$ et $\sigma = \frac{-(\frac{C_r}{Y}+\frac{I_r}{Y})}{1-C_Y}>0.$

une version légèrement modifiée de la courbe IS:

$$y=-\sigma(r_r-r^*)+\theta_t.$$

où $\theta_t$ est un choc de demande.

### Le rôle de la Banque Centrale

L’objectif principal de la banque centrale consiste à stabiliser l’inflation $\pi$ autour de sa cible $\pi^*$.

la fonction de réponse de la banque centrale:

$$r_r=r^*+\gamma (\pi_t-\pi^*)$$

### Demande Agrégée

## PC 3

### Concurrence imparfaite
 contexte: concurrence monopolistique
* le prix de vente $p=mc$ est indépendant de la demande
* toutes les entreprises facturent leur cout marginal mc
* la quantité que les entreprises peuvent produire à leur coût marginal est indépendante du prix global

Tarification monopolisitique:
* Un monopole fait face à une demande élastique $y(p)$ avec un coût de production linéaire $mc$
* Elle maximise le profit en fixant le bon prix:$max_p(y(p)p-wy(p))$
* Résultat: l'entreprise facture $p=(1+\mu)$ où $\mu$ dépend de l'élasticité de la demande.
* $\mu$ est une marge par rapport aux coût de production

Alors,
* De nombreux producteurs $(N>>1)$ qui prennent les prix comme donnés mais n'interagissent pas directement
* ils produisent des biens qui sont des substituts imparfaits
* Par conséquent, chaque producteur a un peu de pouvoir de marché
* applique un markup $\mu$ sur le coût de production $p=(1+\mu)mc$ un peu comme un monopole
* le mark up dépend(négativement) de l'élasticité de la demane et (negativement) de l'intensité de la concurrence(le nombre d'entreprise)

### L'économie de l'offre
La spirale de l'inflation
* les prix augmentent
* Les travailleurs demandent des salaires plus élevés
* les coût de production augmentent
* les prix augmentent

Peu probable ajd:
* les agents anticipent correctement l'infaltion future
* ils intègrent ces attentes dans la fixation des salaires et des prix
* lorsque tous les marchés sont en équilibre, l'inflation retourne vers l'équilibre

Objectif: Établir pourquoi même si les marchés des biens sont à l'équilibre, les entreprises monopolistiques choisissent de produire davantage lorsque les prix augmentent.

O.s.q. les entreprises produisent en utilisant la main-d'oeuvre $L$ louée au salaire horaire $W$ à l'aide d'une fonction de production simple

$$Y=L$$

Le coût marginal de production d'une unité est simplement $W$.

Sous la concurrence monopolistique, le prix optimal fixé par les entreprises est donc :
$$P^*=(1+\mu)W$$
où $\mu$ est un markup qui mesure l'intensité de la concurrence, comme vu avant.

Le coût des travailleurs:
* le coût est plus élevé lorsque les prix agrégés sont plus élevés
* le coût est plus élevé lorsque la quantité de travail est plus élevée

(theorie du chomage)

Offre de travail:
Un travailleur fournit du travail $L \leq 1$.
* consommer un panier de biens $C$ au niveau de prix $P$
* profiter du temps libre $U=1-L$

Nous pouvons écrire la contrainte budgétaire:

$$W.1 \geq PC+WU$$

L'utilité à maximiser est

$$V(C,u)=log(C)-\xi(1-U)^{1\xi}.$$

Le résultat de l'optimisation donne : $ L^S=(\frac{W}{P})^\xi $, avec la condition $C=Y=L$.

Coût du travail
Par inverser la condision au-dessus, on a:
$$W(L)=PL^{1/\xi}$$

Le lien salaire-prix

Rappelons la fonction de production $Y=L$, de sorte que $W(L)=W(Y)$.

Résumons ce que nous avons jusqu'à présent:

* Marché des biens : $P^*=(1+\mu)W(Y)$
* Marché du travail : $W(Y)=PY^{1/\xi}$

### L'équilibre naturel

Équilibre naturel : niveau de production lorsque tous les prix sont flexibles ou ont eu suffisamment de telps pour s'ajouter. C'est aussi l'équilibre de long terme.

Ici, cela signifie que le prix optimal $P^*$ est égal au niveau général des prix $P$ :
$$P^*=(1+\mu)PY^{1/\xi}.$$
Ou $1=(1+\mu)Y^{1/\xi}.$

le niveau de production naturelle est :
$$Y^{nt}=(1+\mu)^{-\xi}.$$

### Rigité nominales
Frictions dans le marché des biens

fixation des prix échelonnée.

Soit $\omega \in [0,1]$ friction des entreprise.

$P_{t-1}$: ancien prix, toujours utilisé par les entreprises qui n'ont pas ajusté
$P^*_{t}$: nouveau prix par les entreprises qui ajustent

le prix d’un panier de consommation qui est une moyenne des deux prix:$P_t=P_{t-1}^{1-\omega}(P^*_{t})^\omega$.

$P^*_{t}=(1+\mu)P_tY_t^{1/\xi}=(1+\mu)P_{t-1}^{1-\omega}(P^*_{t})^\omega Y_t^{1/\xi}$

Alors:$Y_t = \left(\frac{1}{1+\mu}\right)^{\xi} \left(\frac{P^*}{P_{t-1}}\right)^{\xi(1 - \omega)}$

en reécrivant $P_t=P_{t-1}^{1-\omega}(P^*_{t})^\omega$, on a $\frac{P^*}{P_{t-1}} = \left(\frac{P_t}{P_{t-1}}\right)^{\frac{1}{\omega}}$, puis

$Y_t = \left(\frac{1}{1+\mu}\right)^{\xi} \left(\frac{P^*}{P_{t-1}}\right)^{\xi\frac{1 - \omega}{\omega}}$

en fonction de l'inflation:

$Y_t = \left(\frac{1}{1+\mu}\right)^{\xi} \left(1 + \pi _t \right)^{\xi\frac{1 - \omega}{\omega}}$

prenez les logarithmes:
$$y_t - y^{nt}_t = \xi \frac{1 - \omega}{\omega} \pi_t$$

en posant $\kappa= \frac{ \omega}{\xi(1 -\omega)}$, on a

La courbe de phillips:

$$\kappa (y_t - y^{nt}_t) = \pi_t$$

- On a omis les chocs de productivité.
  - Nous les réintroduirons comme chocs dans $\frac{y^n_t}{t}$.
- On n’a incorporé aucune “anticipation” dans le comportement des entreprises. En principe, elles devraient :
  - Faire des prévisions de prix rationnelles pour fixer leurs prix.
  - Maximiser leur profit intertemporel.
- En fonction du choix de modélisation, on obtient des termes dans la courbe de Phillips en :
  - $\pi_{t-1}$ si optimisation statique et extrapolation du trend ($E_t \pi_{t+1} = \pi_t$).
  - $\pi_{t+1}$ si optimisation dynamique et anticipations rationnelles (modèle standard).

## PC4

Rappel: 

Demande Agrégée:
$$y_t=\theta_t-\sigma\gamma(\pi-\bar \pi)$$
Mécanisme: les presssions inflationnistes $(\pi_t>\bar \pi)$ poussent la banque centrale à adopter une politique monétaire restrictive, ce qui augmente le taux d’intérêt réel donc réduit la demande et la production

Offre Agrégée:

$$\pi_t =\bar \pi + \kappa(y_t - y^n_t)$$

Mécanisme : un écart de production élevé engendre des tensions sur le marché du travail qui élèvent le salaire réel d’équilibre ; les entreprises qui le peuvent répercutent ce coût en élevant leur prix.

À long terme: $$y_\infty=y_\infty^n(=-\mu^*)=\thata_\infty$$

Équilibre de court terme:
DA:$y_t=\theta_t-\sigma\gamma(\pi-\bar \pi)$
OA:$\pi_t =\bar \pi + \kappa(y_t - y^n_t)$

endogènes: $y_t$ et $\pi_t$ 
exogènes:$\theta_t$ et $y^n_t$

En resolvant le système linaire:

$y_t = \text{cte}_1 + \left(\frac{1}{1 + \sigma \gamma \kappa}\right) \theta_t + \left(\frac{\sigma \gamma \kappa}{1 + \sigma \gamma \kappa}\right) y^n_t$

$\pi_t = \text{cte}_2 + \left(\frac{\kappa}{1 + \sigma \gamma \kappa}\right) \theta_t - \left(\frac{\kappa}{1 + \sigma \gamma \kappa}\right) y^n_t$

Les termes entre parenthèses sont génériquement appelés “multiplicateurs” :
- Ils quantifient la réaction de court terme à des chocs de $\theta_t$ et de $y^n_t$.
- Ils ont bien le signe attendu. Voyons comment on peut interpréter leur effet.

## PC5

Les outils de la politique monétaire


les principaux outils de la politique monétaire:

- **Opérations d’open market**
  - La banque centrale (BC) échange des liquidités (cash) en échange d’actifs moins liquides (obligations/bons du trésor).
  - La BC prête sur le marché interbancaire.
  
- **Taux de réserves obligatoires**

- **Taux d’intérêt sur les réserves détenues par les banques auprès de la BC**
  - “Discount rate” aux États-Unis.
  - “Main Refinancing Operations” (MRO) en Zone Euro.

- **D’autres outils non-conventionnels** (non couverts ici)

- **M1 : Monnaie étroite**
  - Monnaie fiduciaire (pièces, billets) en circulation et dépôts à vue (par exemple, les comptes chèques des consommateurs).

- **M2 :**
  - M1 + dépôts avec une maturité convenue de jusqu’à deux ans et dépôts remboursables avec un préavis de jusqu’à trois mois.

- **M3 : Monnaie large**
  - M2 + accords de rachat, parts/unités de fonds du marché monétaire et titres de créance avec une maturité de jusqu’à deux ans.

les banques commerciales doivent conserver une fraction $\lambda$ de leurs passif sous forme de réserves, Le montant total d’argent étroit que le système financier peut créer à partir d’un dépôt de 1 unité est appelé multiplicateur monétaire¹ :$\frac{1}{\mu} = \lambda - 1$

