# Notes on probabilities

#### Math

$0+1+2+...n = \frac{n(n+1)}{2}$\
$0^2+1^2+2^2+...n^2 = \frac{1}{6}n(n+1)(2n+1)$\
Sequence: a function such that sequence(S::Set{T},i::N) -> s_i::T (define an order)\
Serie: the sum of the elements of a sequence (the cumsum over a given order)\
Given $|x| < 1$ → $\sum_{i=0}^\infty x^i = \frac{1}{1-x}$ ;
$\int e^{ax}dx = \frac{1}{a}e^{ax}~~~$ ;
$\int u(x) v'(x) dx = u(x)v(x) - \int u'(x) v(x) dx~~~$
$\int_{-\infty}^{+\infty} e^{-x^2/2} dx = \sqrt{2\pi}~~~$
$\int 1/x dx = ln(x) + c~~~$
$\int ln(ax) dx = x ln(ax) - x~~~$ ;
$ln(x)+ln(y) = ln(xy)~~~$
$log_b (a^c) = c log_b (a) ~~~\log_{b2}x = \frac{\log_{b1}x}{\log_{b1}b2}$
Circumference: $(x-a)^2+(y-b)^2=r^2~~~$ ; $sin(0) = sin(\pi) = 0$
${{a+b+c}\choose{a,b,c}} = \frac{(a+b+c)!}{a! * b! * c!}$
$x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$
$lim_{n \to \infty} \left( 1 + \frac{\lambda}{n} \right)^n = e^\lambda$
- Hessian: second derivatives matrix; Gradient: vector of first derivatives; Jacobian: $I \times J$ matrix of first derivative of the i equation for the j variable
- $x'Hx \leq 0 ~~ \forall x \in R^d$  or H is diagonal and all elements negative ↔ H negative semidefinite, all eigenvalues non-positive
- Positive definite: $|D_i| > 0 ~~ \forall  i \leq n$ with $D_i$ the _i_-th leading principal minor of the Hessian
- Negative definite: $(-1)^i |D_i| > 0 ~~ \forall  i \leq n$
- Vector products:
  - Inner ("dot") product: $X \in R^d \cdot Y \in R^d \to OUT = ||X|| * ||Y|| * cos(\theta) \in R^1$
  - Hadamard ("elementwise") product:   $X \in R^d \odot Y \in R^d \to OUT \in R^d$
  - Outer product: $X \in R^d \otimes Y \in R^d \to OUT \in R^{(d^2)}$
  - Cross product: $X \in R^3 \times Y \in R^3 \to OUT = ||X|| * ||Y|| * sin(\theta) * n \in R^3$ (where $n$ is the unit vector perpendicular to the plane containing X and Y)
- $\frac{\partial \int_a^x f(t) dt}{\partial x} = f(x)$; $\frac{\partial \int_x^a f(t) dt}{\partial x} = -f(x)$

- "positive semidefinite matrix" := square matrix such that $x^TAx \geq 0 ~~ \forall x \in R^d$
  - In particular are spd matrices all diagolan matrices with al non-negative entries and those that can be decomposed as $A = P^TDP$ with D a diagonal matrix with only non-negative entries and P invertible
- "positive semi-def square root": a matrix $A^{\frac{1}{2}}$ such that $A^{\frac{1}{2}}A^{\frac{1}{2}} = A$ with A being spd
  - the roots themselves are positive semi-def
  - for any positive (semi-)definite matrix, the positive (semi-)definite square root is unique.
- "ortogonal" matrix: $M^T = M^{-1}$ 
- (AB)ᵀ = BᵀAᵀ
- $(A^{-1})^T = (A^T)^{-1}$
- $AIB = AB$ with $I$ the identity matrix
- $ABsC = sABC = ABCs = \dots$ with $s$ a scalar value
 - $trace(x^Tx) = trace(xx^T)$
 - $E[trace(\cdot)] = trace(E[\cdot])$S
  
Norm

- $\mid \mid v \mid \mid^2 = v^Tv = trace(v v^T) = \sum_i v_i^2$
- $\mid \mid v \mid \mid_l = (\sum_i v_i^l)^{1/l}$

Vector space
- Projection of vector $a$ on vector $b$: $c = \frac{a \cdot b}{\|b\|} * \frac{b}{\|b\|}$.
- Distance of a point $x$ from a plane identified by $\theta$ and its offset $\theta_0$:  $\|\vec{d}\| = \frac{\vec{x} \cdot \vec{\theta} + \theta_0}{\|\vec{\theta}\|}$
- Orthogonal projection of a point $x$ on plane identified by $\theta$ and its offset $\theta_0$: $\vec{x_p} = \vec{x} - \frac{\vec{x} \cdot \vec{\theta} + \theta_0}{\|\vec{\theta}\|} * \frac{\vec{\theta}}{\| \vec{\theta} \|}$

Vector independence:
A set of J vectors $v_j$ are linear dependent i.f.f. there exist a vector $c$ not all zeros such that $\sum_{j=0}^J c_j v_j = 0$ (note that each individual $c_j$ is a scalar while $v_j$ is a vector).

If a partition of the set of vectors is linearly dependent the whole set is said to be linear dependent.

Any set of J vectors of D elements with J > D is linearly dependent.

rank(AB) ≤ min(rank(A),rank(B))

#### Models and axioms

<!--Prob model: (1) Which are the poss outcomes? (2) Which do we believe are their likelihoods ? (3) Which is the event of interest ?
Countable additivity axiom: each outcome is mutually exclusive (disjoint, only one outcome happened), are collectively exhaustive (union is Ω, exhaust all the possibilities), consider only the element of interest of the experiment.

With A₁, A₂,.. sequence of disjoint events ( $A_i \cap A_j = \emptyset ~ \forall((i,j))$ ): $P(A_1 \cup A_2 \cup...) = P(A_1) + P(A_2) + ...$ (countable additivity axiom)
In general: $P(A_1 or A_2) =   P(A_1 \cup A_2) = P(A_1) + P(A_2) - P(A_1 \cap A_2)$

union bound: $P(A_1 \cup A_2) \leq P(A_1) + P(A_2)$
$P(A_1 \cup A_2 \cup A_3 \cup...) = P(A_1) + P(A_2 \cap A_1^C) + P(A_3 \cap A_1^C \cap A_2^C) + ...$
$P(A_1 and A_2) = P(A_1 \cap A_2) = P(A_1) * P(A_2|A_1) = P(A_2) * P(A_1|A_2)$ $P(A_1 \cap A_2 \cap ... A_n) = P(A_1) * Π_{i=2}^n P(A_i|A_1 \cap ... A_{i-1})$
Discrete uniform: P(A) = #elements in A / #elements in Ω

set: discrete (countable) or continuous (uncountable), finite or infinite
Union->Or->Any
Intersection->And->Any
$(\cup_n A_n)^c = \cap_n A_n^C$ $(\cap_n A_n)^c = \cup_n A_n^C$

Sequence: When there is a relationship between the elements, i.e. elements come from some defined set.
Sequence: a function such that sequence(S::Set{T},i::N) -> s_i::T  converge if lim(i→∞) sequence(S::Set{T},i::N) = a::T
Serie: the sum of the elements of a sequence

Sequence convergence: given $\epsilon>0, \{a_i\}$, there exists $i_0$ such that $\forall i >= i_0 \rightarrow |a_i -a| < \epsilon$.
If $g$ is continuous and $a-i \rightarrow a$: $g(a_i) → g(a)$
Monotonic sequences (functions) converge either to infinite or to a number.

Inf serie: $\lim_{n \to \infty} \sum_{i=1}^n a_i$ if limit exist

Multiple indexes: as long the sum of the abs values of the elements of the sequence is finite, the order of summation doesn't matter.

Countable set: if all elements can be put in a 1 to 1 correspondence with a pos integer, that is we can arrange all elements of Ω in a sequence. Including rational q with 0 ≦ q ≦ 1.
Real: uncountable-->



$(\cup_n A_n)^c = \cap_n A_n^C~~$ $(\cap_n A_n)^c = \cup_n A_n^C~~$
$P(A_1 \cup A_2 \cup A_3 \cup...) = P(A_1) + P(A_2 \cap A_1^C) + P(A_3 \cap A_1^C \cap A_2^C) + ...~~$


${\bf P}\Big((A \cap B^ c) \cup (A^ c \cap B)\Big)={\bf P}(A)+{\bf P}(B)-2\cdot {\bf P}(A\cap B)~~$
${\bf P}(A_1 \cap A_2 \cap \cdots \cap A_ n)\geq {\bf P}(A_1)+{\bf P}(A_2)+\cdots +{\bf P}(A_ n)-(n-1)$

#### Conditioning and independence
**Partition** of a space: array of mutually exclusive ("disjoint") sets whose members are exhaustive ("complementary") of the space.\
**Joint**: $P(A ~ \text{and} ~ B)$ = $P(A \cap B)$ = $P(A,B)$\
→ note that joint PMF/PDF are multidimensional aka multivariate (x is a vector)\
**Marginal** (unconditional): $P(A)$\
→ for PMF (PDF): we sum (integrate) over all or some dimensions to "remove" them and move from the joint toward the marginal\
**Conditional**: $P(A|B) := P(A,B)/P(B)$\
→ Valid also for PMF and PDF with respect to an event
Union: $P(A ~ \text{or} ~ B) = P(A \cup B) = P(A) + P(B) - P(A\cap B)$
Note the _memoryless_ of geometric/exponential: $Pr(X>t+s|X>t)=Pr(X>s)$. This is the _remaining_ time, not the total time, so it is NOT the independence concept.


**Multiplication rule**:
- $P(A_1 ~ \text{and} ~ A_2) = P(A_1 \cap A_2) = P(A_1) * P(A_2|A_1) = P(A_2) * P(A_1|A_2)$
- $P(A_1 \cap A_2 \cap ... A_n) = P(A_1) * Π_{i=2}^n P(A_i|A_1 \cap ... A_{i-1})$ <!--Attn! _not_ $P(A_1)* P(A_2|A_1)* P(A_3|A_2)* ...$--> → also for PMF, PDF

**Total probability/expectation theorem**:
- _given A being a partition_:  $P(B) = \sum_i P(A)* P(B|A_i)$ → also for PMF, PDF, CDF and expectations

![probability tree](./imgs/probTree.png)\

**Bayes' rule**: _given A a partition_ $P(A_i|B) = \frac{P(A_i,B)}{P(B)} = \frac{P(A_i)P(B|A_i)}{\sum_j P(A_j)P(B|A_j)}$ where the first relation is by definition and the second one is for the Multiplication rule on the nominator and the total prob. theorem on te denominator
→ also for PMF, PDF

**Independence**: A, B indep. iff $P(A ~ \text{and}~ B) \equiv P(A \cap B) = P(A)* P(B)$ eq. $P(A|B) = P(A)$, equiv. $P(B|A) = P(B)$
(a) Indep is symmetric. (b) A collection of event is indep if _every_ collection of distinct indices of such collection is indep. (oth. could be pairwise indep.)
→ also for PMF, PDF, CDF and expectations (but for all $x,y$!)
<!--Pairwise but not independent events: exp: 2 tosses fair coin. A=H first toss; B = H second toss; C= both tosses the same result. A,B,C are all pairwise indep. but if I know both A and B I know C result.-->
<!-- Note that 2 rV NOT independent can be independent in a conditional world-->

Union rule (De Morgan's law again):
$P(A_1 or A_2 or A_3 or...) = P(A_1 \cup A_2 \cup A_3 \cup...) = P(A_1) + P(A_2 \cap A_1^C) + P(A_3 \cap A_1^C \cap A_2^C) + ...$
$P(A_1 \cup A_2 \cup A_3 \cup...) = 1- P(A_1^C \cap A_2^C \cap A_3^C ...)$

##### Counting
<!--Basic counting principle: For a selection that can be done in $r$ stages, with **fixed** $n_i$ choices at each stage $i$, the number of possible selections in $n_1 * n_2 * ...* n_r$.-->

Ways to _order_ n elements ("permutations"): $n!$ \
Ways to _partition_ $n$ elements:
(**a**) in 2 subsets: (**a.1**) defining $n_1$: ${n}\choose{n_1}$; (**a.2**) Without defining $n_1$: $2^n = \sum_{i=0}^n {{n}\choose{i}}$; (**b**) in $K$ subsets: (**b.1**) Defining the $k_1,k_2,...,k_K$ elements of each subset: ${n} \choose{k_1,k_2,...,k_K}$; (**b.2**) without defining the number of elements of each subset: $\left\{{n \atop K}\right\}$ (Sterling); (**c**) without specifying the number of subsets $K$: Bell numbers

Note that the partitioning problem with the ks all 1 is the problem of ordering a unique set considering each position a "slot".

Ways to sample $k$ elements from a $n$ elements bin: (**a**) with replacement: (**a.1**) order matters: $n^k$; (**a.2**)order doesn't matter: ${n+k-1} \choose {k}$; (**b**) without replacement:  (**b.1**) order matters: $k! * {{n} \choose {k}}$; (**b.2**) order doesn't matter: ${n} \choose {k}$.

Probability to sample in $n$ attempts $x$ elements of a given type from a bin of $s$ elements of that type out of total $k$ elements: (**a**) with replacement: `Binomial(x;n,s/k)`; (**b**) without replacement: `Hypergeometric(x; s, k-s, n)`, i.e. $\frac{ {s \choose x} { k-s \choose n-x } }{ k \choose n  }$ (this reduces to $\frac{\frac{s!}{(s-n)!}}{\frac{k!}{(k-n)!}}$ for the probabilities to have _all_ $n$ elements sampled of the given type)

##### Distributions

**Random variable**
→ Associate a numerical value to every possible outcome\
→ "Discrete" refers to finite or countable infinite values of X, not necessarily integers\
→ "Mixed": those rv that for some ranges are continuous but for some other values have mass concentrated on that values\
- $p_X(x)$: PMF: Probability Mass Function (discrete) $P(X=x) = P(\{\omega \in \Omega: X(\omega) = x \}) = p_X(x)$ ("such that")
- $f_X(x)$: PDF: Probability Density Function (continuous) $P(a \leq X \leq b) = \int_{a}^{b} f_X(x)$ (prob per "unit length" - or area, they give the rate at which probability accumulates in the vicinity of a point.) PDF can be discontinue.
- $F_X(x)$: CDF: Cumulative density function (discrete, continuous or mixed) $P(X \leq x) = F_X(x)$
- $Quantile(f) = CDF^{-1}(f)$ BUT by convention the  $q_\alpha$ quantile indicates $quantile(1-\alpha)$ not the quantile of $\alpha$.

  - | $\alpha$ | 0.025 | 0.05 | 0.1 |
    | -------- | ----- | ---- | --- |
    | $q_\alpha$ | 1.96 | 1.645 | 1.282 |
  - "95% CI": μ ± 1.96 st.dev

- $\sum_{i=-\infty}^x p_X(i) = F_X(x)$; $\int_{-\infty}^x p_X(i) di = F_X(x)$; $p_X(i) = \frac{dF(x)}{di}$

→ "Random vector" a multivariate random variable in $R^k$. The PDF of the random vector is the joint of all its individual components.
- **Gaussian vector** All elements and any linear combination of them is gaussian distributed (e.g. are independent)

**Discrete distributions**:

- **Discrete Uniform** : Complete ignorance
- **Bernoulli** : Single binary trial
- **Binomial** : Number of successes in independent binary trials
- **Categorical** : Individual categorical trial
- **Multinomial** : Number of successes of the various categories in independent multinomial trials
- **Geometric** : Number of independent binary trials until (and including) the first success (discrete time to first success)
- **Hypergeometric** : Number of successes sampling without replacement from a bin with given initial number of items representing successes  
- **Multivariate hypergeometric** : Number of elements sampled in the various categories from a bin without replacement
- **Poisson** : Number of independent arrivals in a given period given their average rate per that period length (or, alternatively, rate per period multiplied by number of periods)
- **Pascal** : Number of independent binary trials until (and including) the n-th success (discrete time to n-th success).

| Name     | Parameters   | Support   | PMF      | Expectations       | Variance    | CDF    |
| -------- | ------------ | --------- | -------- | ------------------ | ----------- | ------ |
| **D. Unif** | a,b ∈ Z with b ≧ a | $x \in \{a,a+1,...,b\}$| $\frac{1}{b-a+1}$ | $\frac{a+b}{2}$ | $\frac{(b-a)(b-a+2)}{12}$ |$\frac{x-a+1}{b-a+1}$ |
| **Bern** | p ∈ [0,1] | x ∈ {0,1} | $p^x(1-p)^{1-x}$ | $p$ |  $p(1-p)$ | $\sum_{i=0}^x p^i(1-p)^{1-i}$ |
| **Bin** | p ∈ [0,1], n in N⁺ | $x \in \{0,...,n\}$ | ${{n} \choose {x}} p^x(1-p)^{1-x}$ | $np$ | $n p(1-p)$ |  $\sum_{i=0}^{x} {{n} \choose {i}} p^i(1-p)^{1-i}$  |
| **Cat** | $p_1,p_2,...,p_K$ with $p_k \in [0,1]$ and $\sum_{k=1}^K p_k =1$ | x ∈ {1,2,...,K} | $\prod_{k=1}^K p_k^{\mathbb{1}(k=x)}$ | | |
| **Multin** | $n, p_1,p_2,...,p_K$ with $p_k \in [0,1]$, $\sum_{k=1}^K p_k =1$ and $n \in N^+$| $x \in \mathbb{N}_{0}^K$ | ${{n} \choose {x_1, x_2,...,x_K}} \prod_{k=1}^K p_k^{x_K}$ | | | |
| **Geom** | p ∈ [0,1] | x ∈ N⁺|  $(1-p)^{x-1}p$ | $\frac{1}{p}$ | $\frac{1-p}{p^2}$ | $1-(1-p)^x$  |
| **Hyperg** | $n_s,n_f, n \in \mathbb{N}_{0}$ |  $x \in \mathbb{N}_{0}$ with $x \leq n_s$ | $\frac{{n_s \choose x} {n_f \choose n-x} }{  (n_s + n_f) \choose n }$ | $n \frac{n_s}{n_s+n_f}$ | $n\frac{n_s}{n_s+n_f}\frac{n_f}{n_s+n_f}\frac{n_s+n_f+n}{n_s+n_f+1}$ | |
| **Multiv hyperg** | $n_1,n_2,...,n_K$, $n$ with $n \in \mathbb{N}_{+}, n_i \in \mathbb{N}_{0}$ |  $x \in \mathbb{N}_{0}^K$ with $x_i \leq n_i ~ \forall i$, $\sum_{i=1}^K x_i = n$ | $\frac{\prod_{i=1}^K {n_i \choose x_i} }{ \sum_{i=1}^K n_i \choose  n }$ | $n\frac{n_i}{\sum_{i=1}^K n_i}$ | $n\frac{\sum_{j=1}^K n_j - n}{\sum_{j=1}^K n_j - 1} \frac{n_i}{\sum_{j=1}^K n_j} \left(1 - \frac{n_i}{\sum_{j=1}^K n_j} \right)$ |  |
| **Pois** | λ in R⁺ | x ∈ N₀ | $\frac{\lambda^xe^{-\lambda}}{x!}$ | $\lambda$ | $\lambda$ |  |
| **Pasc** | n ∈ N⁺, p in [0,1] | x ∈ [n, n+1, ..., ∞) | ${x-1 \choose n-1} p^n (1-p)^{x-n}$ | $\frac{n}{p}$ | $\frac{n(1-p)}{p^2}$ | |



**Continuous distributions**:

- **Uniform** Complete ignorance, pick at random, all equally likely outcomes
- **Exponential** Waiting time to first event whose rate is λ (continuous time to first success)
- **Laplace** Difference between two iid exponential r.v.
- **Normal** The asymptotic distribution of a sample means  
- **Erlang** Time of the n-th arrival
- **Cauchy** The ratio of two independent zero-means normal r.v.
- **Chi-squared** The sum of the squared of iid standard normal r.v.
- **T distribution** The distribution of a sample means
- **F distribution** : The ratio of the ratio of two indep Χ² r.v. with their relative parameter
- **Beta distribution** The Beta distribution
- **Gamma distribution** Generalisation of the exponential, Erlang and chi-square distributions

| Name     | Parameters   | Support   | PMF      | Expectations       | Variance    | CDF    |
| -------- | ------------ | --------- | -------- | ------------------ | ----------- | ------ |
| **Unif** | a,b ∈ R with b ≧ a | x \in [a,b] | $\frac{1}{b-a}$ | $\frac{a+b}{2}$ | $\frac{(b-a)^2}{12}$ | $\frac{x-a}{b-a}$ |
| **Expo** | λ ∈ R⁺ | x ∈ R⁺ | $\lambda e^{-\lambda x}$ | $\frac{1}{\lambda}$ | $\frac{1}{\lambda^2}$ | $1-e^{-\lambda x}$ |
| **Laplace** | μ ∈ R (location), b ∈ R⁺ (scale) | x ∈ R | $\frac{1}{2b} e^{-\frac{\mid x - \mu \mid}{b}}$ | $\mu$ | $2b^2$ | |
| **Normal** | μ ∈R, σ² ∈ R⁺ | x ∈ R | $\frac{1}{\sigma \sqrt{2 \pi}}e^\frac{-(x-\mu)^2}{2\sigma^2}$ | $\mu$ | $\sigma^2$ | |
| **Multiv. Normal** | $\mu \in R^d, \Sigma \in R^{d \times d}$ | x ∈ Rᵈ | $\frac{1}{\sqrt{(2 \pi)^d \text{det}(\Sigma)  } }   e^{-\frac{1}{2} (x-\mu)'\Sigma^{-1}(x-\mu) }$ | $\mu$ | $\Sigma$ | |
| **Erlang** | n ∈ N⁺, λ ∈ R⁺ | x ∈ R₊ | $\frac{\lambda^n x^{n-1} e^{-\lambda x} }{(n - 1) !}$ | $\frac{n}{\lambda}$ | $\frac{n}{\lambda^2}$ | |
| **Cauchy** | x₀ ∈ R (_location_), γ ∈ R⁺ (_scale_) | x ∈ R | $\frac{1}{\pi \gamma (1+(\frac{x-x_0}{\gamma})^2) }$ | NDEF | NDEF | |
| **Chi-sq** | d ∈ N⁺ | x ∈ R⁺ | $\frac{1}{2^{}\frac{d}{2}\Gamma(\frac{d}{2})} x^{\frac{d}{2})-1}e^{-\frac{x}{2}}$ | $d$ | $2d$
| **T** | ν ∈ R⁺ | x ∈ R | $\frac{ \Gamma(\frac{\nu +1}{2})}{\sqrt{\nu \pi} \Gamma(\frac{\nu}{2})} \left( 1 + \frac{x^2}{\nu} \right)^{- \frac{\nu + 1}{2}}$ |
| **F** | d₁ ∈ N⁺ d₂ ∈ N⁺ | x ∈ R⁺ |  $\frac {\sqrt {\frac {(d_1 x)^{d_1} d_2^{d_2} } {(d_1 x + d_2)^{d_1 + d_2} } }} {x \mathrm {B} \left( \frac{d_1}{2},\frac {d_2}{2} \right) }$ | $\frac{d_2}{d_2 -2}$ for $d_2 > 2$ | $\frac{2 d_2^2 (d_1 + d_2 -2)}{d_1 (d_2 -2)^2 (d_2 -4)}$ for $d_2 > 4$ | |
| **Beta** | α, β ∈ R⁺ | x ∈ [1,0] | $\frac{1}{B(\alpha,\beta)}x^{\alpha-1}(1-x)^{\beta-1}$ | $\frac{\alpha}{\alpha+\beta}$ | $\frac{\alpha \beta}{(\alpha + \beta)^2 (\alpha + \beta + 1)}$| |
| **Gamma** | α ∈ R⁺ (_shape_), β ∈ R⁺ (_rate_) | x ∈ R⁺ | $\frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha-1} e^{-\beta x}$ | $\frac{\alpha}{\beta}$ |  $\frac{\alpha}{\beta^2}$ | |

**Beta function** : $B(\alpha,\beta) = \frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)} = \frac{\alpha + \beta}{\alpha \beta}$ \
**Gamma function**: $\Gamma(x)=(x-1)! ~ \forall x \in N$







**Expected value**
→ The mean we would get running an experiment many times\
- $E[X] := \sum_x x p_X(x) := \int_{-\infty}^{+\infty} x f_X(x) dx$
- **Expected value rule**: $E[Y=g(X)] = \sum_y Y p_Y(y) = \sum_x g(x) p_X(x) \neq g(\sum_x x p_X(x)) = g(E[X])$ (in general)
- **Linearity of expectations**: $E[aX+b] = aE[X]+b$; $E[X+Y+Z] = E[X] + E[Y] + E[Z]$
- **X,Y independent**: $E[XY] = E[X]E[Y]$, $E[g(X)h(Y)] = E[g(X)]E[h(Y)]$
- **Law of Iterated Expectations**: $E[E[X|Y]] := \sum_Y E[X|Y] p_Y(y) = E[X]$ ($E[X|Y]$ is seen as a function $g(Y)$)
- Expectations of convex functions are convex
- The expectations of an indicator function is the prob that the event indicated is true


**Variance**
- $Var(X) := E[(X-\mu)^2] = \sum_x(x-\mu)^2 p_X(x) = \int_{-\infty}^{+\infty} (x-\mu)^2 f_X(x)$
- $Var(X) = E[X^2] - (E[X])^2$ 
- $Var(g(X)) = E[g(X)^2] - (E[g(X)])^2$ 
- $Var[aX+b] = a^2 Var[X]$;
- var of sum of r.v.:
  - X,Y independent: $Var(X+Y) = Var(X)+Var(Y)$
  - in general: $\textrm{Var}\left(\sum_{i=1}^{n} X_i\right)=\sum_{i=1}^{n} \textrm{Var}(X_i)+2 \sum_{i<j} \textrm{Cov}(X_i,X_j) = \sum_{i,j} \textrm{Cov}(X_i,X_j)$
- **Law of total Variance**: $var(X) = E[var(X|Y)]+var(E[X|Y])$ (expected value of the variances of X _within_ each group Y + variance of the means _between_ groups)

**Moments**
- $M_n = \int_{-\infty}^{\infty} (x-c)^n f_X(x) dx$
- c = 0 → "raw moment" (e.g. E[X], E[X²], ...) 
- c = μ → "central moment" (e.g. var(X) (second), skewness (third), kurstsis (fourth)) 
- "Mode" -> argmax(fₓ)
- "Median" -> $k: \int_{-\infty}^{k} f_X(x) dx = \int_{k}^{\infty} f_X(x) dx$
- "Mean" -> the expected value

**Covariance and correlation**
- 2 r.v.: $Cov(X,Y) := E[(X-E[X])(Y-E[Y])] = E[(X)(Y-E[Y])] = E[XY]-E[X]E[Y]$
- $Cov(aX+bY+c, eZ) = a e ~Cov(X,Z) + b e ~Cov(Y,Z)$
- X random vector: 
  - $Cov(X):= E[(X-E[X])(X-E[X])'] = E[XX']-E[X]E[X]'$
  - $Cov(AX+b) = A Cov(X) A'$ (all cov matrix are positive definite and so it's ok to take square roots of them)
- Correlation coeff.: $\rho :=\frac{cov(X,Y)}{\sqrt{var(X)var(Y)}} = E[\frac{X-E[X]}{\sigma_X}*\frac{Y-E[Y]}{\sigma_Y}]$ with $-1 \leq \rho \leq +1$ and $\sigma_X, \sigma_Y \neq 0$\
$(X,Y) ~ \text{indep.} \to cov(X,Y)=0 \leftrightarrow \rho = 0$ (but not $\leftarrow$)\
$|\rho| = 1 \leftrightarrow (X-E[X]) = c(Y-E[Y])$ (i.e. X,Y linearly correlated)\
$\rho(aX+b,Y) = sign(a) * \rho(X,Y)$ (because of dimensionless)\
$X = Z+V, Y= Z+W, Z,V,W \text{indep.} \to \rho(X,Y) = \frac{\sigma_Z^2}{\sigma_Z^2 + \sigma_Z \sigma_W + \sigma_V\sigma_Z + \sigma_V \sigma_W}$

**Normal RV**\
$X \sim N(\mu,\sigma^2)$ → $Y = aX+b \sim N(a\mu+b,a^2\sigma^2)$\
$X \sim N(\mu,\sigma^2), Z \sim N(0,1) \to P(a \leq X \leq b) = P(\frac{a-\mu}{\sigma} \leq \frac{X-\mu}{\sigma} \leq \frac{b-\mu}{\sigma}) = P(\frac{a-\mu}{\sigma} \leq Z \leq \frac{b-\mu}{\sigma}) = \Phi(\frac{b-\mu}{\sigma}) - \Phi(\frac{a-\mu}{\sigma})$ \
$X \sim N(0,\sigma^2) \to P(X < -a) = 1-P(X<a)$\
$X_i \sim N(\mu_i,\sigma_i^2), ~ X_i ~ \text{i.i.d.} \to Y = \sum_i X_i \sim N(\sum_i \mu_i, \sum_i \sigma_i^2)$\
$E[Z]=0,E[Z^2]=1,E[Z^3]=0,E[Z^4]=3,E[Z^5]=0,E[Z^6]=13$

####Derived Distributions

##### Function of a single R.V. #####
**Linear function of a r.v.**: $Y = aX+b$
- $p_Y(y)=p_X(\frac{y-b}{a})$ (where $\frac{y-b}{a}$ is the value of $X$ that raises $y$)
- $f_Y(y)= f_{aX+b}(y) = \frac{1}{|a|}f_X \left(\frac{y-b}{a} \right)$ (area must be constant)\
![linear function rv](./imgs/linear-function-rv.png)\
**Monotonic**: $Y = g(x)$ with g(x) monotonic and continuous\
$f_Y(y) = f_X(g^{-1}(y)) * |\frac{dg^{-1}}{dy}(y)|$
**Probability Integral Transformation**
Considering as "function" the CDF, this is uniformely distributed for any r.v.: $Y = g(X) = CDF_X(X) \sim U(0,1)$
$Y = F_X(x) \to F_Y(y) = P(Y \leq y) = P(F_X(x) \leq y) = p(X \leq F_X^{-1}(y)) = F_X(F_X^{-1}(y)) = y$
For a sample $k \sim U(0,1)$ the corresponding value over X is $x = F_X^{-1}(k)$, i.e. the inverse CDF evaluated on k.

**General**: $Y = g(X)$\
$p_Y(y) = \sum_{\{x:g(x)=y\}} p_X(x)$ \
$f_Y(y) = \frac{dF_Y}{dy}(y)$ with $F_Y(y)=P(g(X)\leq y) = \int_{\{x:g(x)\leq y\}} f_X(x)dx$ (express the CDF of Y in terms of the CDF of X and then derive to find the PDF)

##### Function of multiple R.V. #####
**Sum of 2 independent R.V., discrete:**:
- $Z = X+Y$ $p_Z(z) = \sum_{x} p_X(X=x) p_{Z|X}(Z=z|X=x) = \sum_{x} p_X(x) p_Y(z-x)$
**Sum of 2 independent R.V., continue (convolution)**:
- $Z=z$ in all occasions where $X=x$ and $Y=z-x$
- $f_Z(z) = \int_{max(x_{min},z-y_{max})}^{min(x_{max},z-y_{min})} f_X(x) * f_Y(z-x) dx$\
**General**: $Z = g(X,Y,...)$ Find (e.g. geometrically) the CDF of $Z$ and differentiate for $Z$ to find the PDF.

**Sum of random number of i.i.d. R.V.** $Y = \sum_{i=1}^N X_i$ with $X_i ~\forall i$ i.i.d and indep to $N$\
$E[Y] = E[E[Y|N]] = E[N*E[X]] = E[N]*E[X]$\
$var(Y) = E[N]*var(X)+(E[X])^2 * var(N)$\
$X\sim bern(p); N \sim bin(m,q) \to Y \sim bin(m,pq)$\
$X\sim bern(p); N \sim pois(\lambda) \to Y \sim pois(p\lambda)$\
$X\sim geom(p); N \sim geom(q) \to Y \sim geom(pq)$\
$X\sim exp(\lambda); N \sim geom(q) \to Y \sim exp(\lambda q)$\

**Order statistics**
Y = max(X), X i.i.d -> $F_Y(y) = P(X_i \leq y) \forall_i \in [1,N] = F_X(y)^N \to f_Y(y) = NF_X(y)^{N-1} f_X(y)$
Y = min(X), X i.i.d -> $F_Y(y) = 1 - P(X_i \geq y) \forall_i \in [1,N] = (1 - (1-F_X(y))^N) \to f_Y(y) = n (1-NF_X(y))^{N-1} f_X(y)$

#### Limits ####

**Properties of the sample mean**: $\bar X_n = \frac{\sum_{i=1}^n X_i}{n}$ → $E[\bar X_n] = E[X]$, $var(\bar X_n) = var(X)/n$ (from properties of expectation and variance)
**Markov Inequality**: $X$ non neg r.v., $t>0$ → $P(X ≥ t) ≤ E[X]/t$ \
**Chebyshev Inequality**: $X$ a r.v., $t > 0$ → $P(|X-E[X]| \geq t) \leq Var(X)/t^2$
- proof: from Markov in. by considering a new r.v. $Y = (X-E[X])^2$
- corollary: the prob that a r.v. is $k$ st.dev. away from the mean is less than $1/k^2$, whatever its distribution
- the $t$ is the "accuracy" and the probability itself is the "confidence" in reaching the given accuracy

**Hoffding's Inequality**: $X,_,X_2,...,X_n$ i.i.d. with $E[X_i] = \mu$ and $X \in [a,b]$ almost surely, and $a < b$ ⇒ $P(|\bar X_n - \mu| \geq \epsilon) \leq 2 e^{-\frac{2n\epsilon^2}{(b-a)^2}} ~~ \forall \epsilon > 0$ (with $n$ not necessarily large)


##### Def of Convergences
We are interested in r.v. whose distribution is parametrised by n, i.e. in sequences of a r.v.

**Of a deterministic sequence:**: The sequence $a_n$ converge to the value $a$ if for any $\epsilon > 0$ it exists a value $n_0$ such that $|a_n - a| \leq \epsilon ~~ \forall n \geq n_0$, i.e. whatever small we choose $\epsilon$, we can always find a n limit where subsequent sequences values are lees than $\epsilon$ far away to $a$

**In distribution**: A sequence $Y_n$ of r.v. (not necessarily indep.) converges in distribution to the r.v. $Y$ iff $\lim_{n \to \infty} E[f(Y_n)] = E[f(Y)]$ for any function $f$ _continuous_ and _bounded_ (so, it's not true in general, and in particular you may have convergence in distribution without having $E[Y_n]$ converging to $E[Y]$) or, equivalently, that $Y$ iff $\lim_{n \to \infty} P(Y_n \leq y) = P(Y \leq y)$. That is, the $Y_n$ CDF converges to a limiting CDF.
Converging of a function. The output of the function depends on n, but with less and less variations. Base for the CLT.
Aka the "convergence in law" or "weak convergence".

**In probability**:  A sequence $Y_n$ of r.v. (not necessarily indep.)  converges in probability to the value $a$ if for every $\epsilon > 0$ we have $\lim_{n \to \infty} P(|Y_n - a| \geq \epsilon) = 0$. Base for the Weak l.l.n.

- The "bulk" of the PMF is within a range of $a$, but nothing is said for the values that are not in the range ⇒ conv. in p. doesn't implies conv. in expectations (as expectations are sensitive to the tail of the distribution).
- To verify, first guess the $\epsilon$ (or the "a" ???) and then find the limit of the CDF for the relevant bound goes to 0

**With probability 1 (almost sure)**:  A sequence $Y_n$ of r.v. (not necessarily indep.)  converges w.p. 1 to $c$ if $P(\lim_{n \to \infty} Y_n =c) = 1$ or, more in general, $P(\{w: \lim_{n \to \infty} Y_n(w) = Y(w)\}) = 1$. Base for the strong l.l.n. The closest version to a deterministic convergence, whatever is the outcome of a random experiment all they have to converge.

**Convergence theorems:**

- $X_n \xrightarrow{p/a.s.} a, Y_n \xrightarrow{p/a.s.} b$ ⇒ $X_n + Y_n \xrightarrow{p/a.s} a+b$; $X_n * Y_n \xrightarrow{p/a.s.} a*b$
- Continuous Mapping Theorem: $X_n \xrightarrow{d/p/a.s.} a$, $g$ is a continuous function, $\Rightarrow g(X_n) \xrightarrow{d/p/a.s.} g(a)$
- Stutsky's Theorem: $X_n \xrightarrow{d} X$, $Y_n \xrightarrow{p/a.s.} y$
  - $X_n + Y_n \xrightarrow{d} X + y$
  - $X_n * Y_n \xrightarrow{d} X * y$


**Law of large numbers**: The sample mean converge to the pop mean
- weak l.l.n.: $\lim_{n \to \infty} P(|\bar X_n - E[X]| > ϵ) = 0$
  - from the Chebyshev Inequality by using $\bar X_n$ and taking the limit for $n \to \infty$
  - it is a convergence in p. of $\bar X_n$ to $E[X]$.
- strong l.l.n.: $\lim_{n \to \infty} P(\bar X_n = E[X]) = 1$

Note that given $\bar X_n = \frac{1}{n} \sum X_i$:
- $\bar X_n \xrightarrow{n \to \infty} E[X]$ (LLN)
- $g(\bar X_n) \xrightarrow{n \to \infty} g(E[X])$ (cont. map. theorem)
- $Y = g(x) \Rightarrow \bar Y_n = \frac{1}{n} \sum Y_i = \frac{1}{n} \sum g(X_i) \xrightarrow{n \to \infty} E[Y] = E[g(X)]$ (LLN)


**Central Limit Theorem**: The distribution of the mean from i.i.d. samples converges in distribution to a Normal distribution with mean equal to the population mean and variance of the population variance divided by the sample size:
$\bar{X}_n ∼ N(E[X],σ_X^2/n)$
- formally the CLT is stated in terms of $\frac{S_n-nE[X]}{\sqrt{n} \sigma_{x} } \xrightarrow{dist} \sim N(0,1)$
- multivariate CLT: $\sqrt{n} * \Sigma_X^{-\frac{1}{2}} (\bar X_n -\mu) \xrightarrow{dist} \sim N_d(0,I_d)$
- versions exists for identically distributed $X_i$ or "weakly dependent" ones (dependence only local between neighbour $X_i$)
- X integer: consider $S_{n+1/2}$
- Approximation to the binomial: $P(k \leq S_n \leq l) ≈ \Phi(\frac{l + \frac{1}{2} -np}{\sqrt{(n(1-p))}}) - \Phi(\frac{k - \frac{1}{2} -np}{\sqrt{(n(1-p))}})$



#### Bernoulli and Poisson random processes

Stochastic processes: a probabilistic phenomenon that evolves in time,i.e. an infinite sequence of r.v.
We need to characterise it with informations on the individual r.v. but also on how they relate (joint)
Bernoulli, Poisson → Assumptions: independence ( → memoryless), time-homogeneity

|                     | Bernoulli            | Poisson               |
| ------------------- | -------------------- | --------------------- |
| Time of arrival `t` | Discrete             | Continuous            |
| Arrival rate        | `p` per trial        | `λ` per unit time     |
| N# of arrivals      | Binomial `pₙ(n;t,p)` | Poisson `pₙ(n;t;λ)`   |
| Interarrival time   | Geometric `pₜ(t;p)`  | Exponential `fₜ(t;λ)` |
| Time to nᵗʰ arrival | Pascal `pₜ(k,p)`     | Erlang `fₜ(t;n,λ)`    |

Fresh start: The Bernoulli or Poisson process after time N, where N is a r.v. causally determined from the history of the process, is a new Bernoulli/Poisson process with the same probabilistic characteristics as the original one.

##### The poisson as approximation of the binomial

Given $p$ the probability of a successes in a single slot and $n$ the number of slots, the expected number of successes $\lambda$ is given by $\lambda = pn$.

The poisson PDF can be seen as the limit of the Bernoulli pdf when we consider smaller and smaller time slots, keeping constant the total expected number of successes for the period (that is p - on the single period - becomes smaller and smaller and the number of periods tends to ∞).
The poisson process can hence be seen as a limiting case of a Bernoulli process or, alternatively, as the process deriving from a sequence of exponential r.v..

In a small interval $\delta$, the probability of 1 success is $\lambda \delta$ and of 0 successes is $1-\lambda \delta$ (and negligible probabilities for more than one).

##### Merging

The process made by a sequence of r.v. functions of other sequences of r.v.

###### Merging of Bernoulli processes

$X_1^i \sim Bern(p)$, $X_2 \sim Bern(q)$, $X_1$ indep $X_2$

$Y^i = X_1^i \text{or} X_2^i$ ⇒ $Y^i \sim Bern(p+q-pq)$\
$Y^i = X_1^i \text{and} X_2^i$ ⇒ $Y^i \sim Bern(pq)$ (both new Bernoulli processes)

The probability that observing a success in the merged process we have a success also in  the orignal process 1 is:
- $Y^i = X_1^i \text{or} X_2^i$ ⇒ $P(X_1^i | Y^i) = \frac{P(X_1^i,Y^i)}{P(Y^i)} =  \frac{P(X_1^i)}{P(Y^i)} = \frac{p}{p+q-pq}$
- $Y^i = X_1^i \text{and} X_2^i$ ⇒ $P(X_1^i | Y^i) = 1$

###### Merging of Poisson processes

$X_1^i \sim Poisson(\lambda_1)$, $X_2 \sim Poisson(\lambda_2)$, $X_1$ indep $X_2$\
Note that differently from Bernoulli case, here the change of a match is zero.

$Y^i = X_1^i \text{or} X_2^i$ ⇒ $Y^i \sim Poisson(\lambda_1+\lambda_2)$ 

The probability that observing a success in the merged process we have a success also in  the orignal process 1 is:
- $Y^i = X_1^i \text{or} X_2^i$ ⇒ $P(X_1^i | Y^i) = \frac{P(X_1^i,Y^i)}{P(Y^i)} = \frac{\lambda_1}{\lambda_1+\lambda_2}$

##### Splitting

###### Splitting of Bernoulli processes

Given a Bernoulli process X with probability $p$ and an other independent Bernoulli process Y that assigns each success of X to $Z_1$ with probability $q$ and to $Z_2$ with probability $(1-q)$, we have:

- $Z_1^i = X^i \text{and} Y^i, Z_1^i \sim Bern(pq)$
- $Z_2^i = X^i \text{and not} Y^i, Z_2^i \sim Bern(p(1-q))$

$Z_1$ and $Z_2$ are _not_ independent !


###### Splitting of Poisson processes

Given a Poisson process X with rate $\lambda$ and an other independent Bernoulli process Y that assigns each success of X to $Z_1$ with probability $q$ and to $Z_2$ with probability $(1-q)$, we have:

- $Z_1^i = X^i \text{and} Y^i, Z_1^i \sim Poisson(\lambda q)$
- $Z_2^i = X^i \text{and not} Y^i, Z_2^i \sim Poisson(\lambda (1-q))$

Note that differently from Bernoulli case, here the  the two processes _are_ independent, as the probability of an arrival at any given _point_ in time is zero.


##### Summing Poisson rv
Given $X_1 \sim Poisson(p)$ (the distribution, not the process) and $X_2 \sim Poisson(q)$, and $X_1, X_2$ i.i.d. ⇒ $Y = (X_1 + X_2) \sim Poisson(p+q)$ (think as the two input r.v. as representing numbers of arrivals in disjoint time intervals).

##### Random incidence ("Inspection paradox")
