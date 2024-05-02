---
marp: true
theme: rinkou
math: mathjax
size: 16:9
---
<!--
_class: title
-->
# Deep Learning: Foundations and Concepts 2024

Section: 4.1.2 - 4.2
2024/5/9 Daiki Yoshikawa

---
<!--
class: slides
footer: 2024/04/18<span style="margin-left:250px;">Deep Learning:Foundations and Concepts 2024</span>
paginate: true
-->
# 目次
- 4.1 Linear Regression
  - 4.1.2 Likelihood function
  - 4.1.3 Maximum likelihood
  - 4.1.4 Geometry of least squares
  - 4.1.5 Sequential learning
  - 4.1.6 Regularized least squares
  - 4.1.7 Multiple outputs
- 4.2 Decision theory
---
# 4.1.2 Likelihood function
- 入力変数xと目的変数tの関係をモデル化したい
- tは決定論的関数y(x,w)と加法性ノイズεによって決まると仮定する:
$$t = y(x, w) + \epsilon$$

---
# 4.1.2 Likelihood function
- ノイズ項εはゼロ平均、分散σ^2のガウス分布に従うと仮定する
- これにより、xが与えられた時のtの条件付き分布は:
$$p(t|x, w, \sigma^2) = \mathcal{N}(t|y(x, w), \sigma^2)$$

---
# 4.1.2 Likelihood function

- 入力とターゲットのペアN個からなるデータセット𝓓 = {x_n, t_n}_{n=1}^Nを考える
- データ点は独立同一に分布していると仮定する
- 尤度関数は次のようになる:
$$p(t|X, w, \sigma^2) = \prod_{n=1}^N \mathcal{N}(t_n|y(x_n, w), \sigma^2)$$

---
# 4.1.2 Likelihood function
- 尤度関数の対数をとると:
$$\begin{align}
\ln p(t|X, w, \sigma^2) &= \sum_{n=1}^N \ln \mathcal{N}(t_n|y(x_n, w), \sigma^2)\\
&= -\frac{N}{2}\ln(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{n=1}^N(t_n - y(x_n, w))^2\\
&= -\frac{N}{2}\ln(2\pi\sigma^2) - \frac{1}{2\sigma^2}E_D(w)
\end{align}$$
- ここで、$E_D(w) = \frac{1}{2}\sum_{n=1}^N(t_n - y(x_n, w))^2$は誤差自乗和関数

---
# 4.1.3 Maximum likelihood
- 最尤解を求めるには、尤度関数のwに関する対数の勾配を0とおく
- 勾配を計算すると:
$$\frac{\partial}{\partial w}\ln p(t|X, w, \sigma^2) = \frac{1}{\sigma^2}\sum_{n=1}^N(t_n - w^T\phi(x_n))\phi(x_n)^T = 0$$

---
# 4.1.3 Maximum likelihood
- これにより正規方程式が得られる:
$$\sum_{n=1}^N t_n\phi(x_n)^T = \sum_{n=1}^Nw^T\phi(x_n)\phi(x_n)^T$$  
- N×M行列Φ(デザイン行列)を定義する。Φ_{nj} = \phi_j(x_n)
- 解は次のように求まる:
$$w_{ML} = (\Phi^T\Phi)^{-1}\Phi^Tt$$
- $\Phi^\dagger = (\Phi^T\Phi)^{-1}\Phi^T$はΦのムーア・ペンローズ疑似逆行列

---
# 4.1.3 Maximum likelihood
- ノイズ分散の最尤推定値は残差分散になる:
$$\sigma^2_{ML} = \frac{1}{N}\sum_{n=1}^N (t_n - w_{ML}^T\phi(x_n))^2$$
- つまり、モデルフィッティング後の残差の分散

---
# 4.1.3 Maximum likelihood
- 幾何学的解釈: 最小二乗解w_{ML}は、ターゲットベクトルtを基底ベクトル{ϕ_j(x)}で張られる部分空間Sに正射影したベクトル

---
# 4.1.4 Geometry of least squares
- 正規方程式の解は一括処理であり、全データセットが必要
- 大規模データセットの場合、逐次/オンライン法が有用
- 1データ点ずつ処理し、各観測後にパラメータを更新

---
# 4.1.4 Geometry of least squares
- 確率的勾配降下法を用いて誤差関数を最適化
- 誤差関数E = Σ_nE_nのとき、パラメータ更新式は:  
$$w^{(\tau+1)} = w^{(\tau)} - \eta \nabla E_n(w^{(\tau)})$$
- ここで、ηは学習率

---
# 4.1.4 Geometry of least squares
- 線形回帰モデルでは:
$$w^{(\tau+1)} = w^{(\tau)} + \eta(t_n - w^{(\tau)T}\phi_n)\phi_n$$
- 最小平均二乗(LMS)アルゴリズムと呼ばれる
- 観測を逐次処理し、パラメータを徐々に更新

---
# 4.1.5 Regularized least squares
- 最尤法では、複雑なモデルでデータ数が少ない場合、過学習が起こりがち
- 正則化によりモデルの複雑さを制御して過学習を防ぐ
- 誤差関数に正則化項を加える:
$$E(w) = E_D(w) + \lambda E_W(w)$$  

---
# 4.1.5 Regularized least squares
- $E_D(w)$はデータ依存の誤差項(例:誤差自乗和)
- $E_W(w)$は正則化項でモデル複雑さを罰する  
- $\lambda > 0$で正則化の強さを調整

---
# 4.1.5 Regularized least squares
- よく用いられる正則化項はウェイトDecay:
$$E_W(w) = \frac{1}{2}w^Tw = \frac{1}{2}\sum_j w_j^2$$
- これにより、重み値はゼロに収束する

---
# 4.1.5 Regularized least squares
- この正則化項を用いると、誤差関数は:
$$E(w) = \frac{1}{2}\sum_{n=1}^N(t_n - w^T\phi(x_n))^2 + \frac{\lambda}{2}w^Tw$$
- wに関して2次形式なので、閉形解が得られる

---
# 4.1.5 Regularized least squares
- 正則化された解は:
$$w = (\lambda I + \Phi^T\Phi)^{-1}\Phi^Tt$$
- これは通常の最小二乗解の拡張形
- 正則化項により、基底に redundancy があっても行列は特異でない

---
# 4.1.6 Multiple outputs
- 今までは目的変数tが1変数の場合を扱った
- 場合によっては、複数の変数 $\mathbf{t} = (t_1, \ldots, t_K)^T$ を予測したい

---
# 4.1.6 Multiple outputs
- 同じ基底関数を使って全ての目的変数をモデル化する:
$$\mathbf{y}(x, W) = W^T\phi(x)$$
- $W$は$M\times K$の重み行列
- 単層の重みをもつニューラルネットワークモデル

---
# 4.1.6 Multiple outputs
- 目的変数の条件付き分布として次を仮定する:
$$p(\mathbf{t}|x, W, \sigma^2) = \mathcal{N}(\mathbf{t}|W^T\phi(x), \sigma^2I)$$
- ターゲットデータ$T = [\mathbf{t}_1, \ldots, \mathbf{t}_N]^T$に対する対数尤度は:
$$\ln p(T|X, W, \sigma^2) = -\frac{NK}{2}\ln(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{n=1}^N||\mathbf{t}_n - W^T\phi(x_n)||^2$$

---
# 4.1.6 Multiple outputs
- $W$について最大化すると解は:
$$W_{ML} = (\Phi^T\Phi)^{-1}\Phi^TT$$
- しかしこれは$K$個の別々の回帰問題に分解される

---
# 4.2 Decision theory
- 条件付き分布$p(t|x)$を使って最適な予測$f(x)$を行いたい
- 期待損失$\mathbb{E}[L] = \int\int L(t, f(x))p(x,t)dxdt$を最小化する必要がある

---
# 4.2 Decision theory
- 二乗損失$L(t, f(x)) = \{f(x) - t\}^2$の場合:
$$f^*(x) = \mathop{\mathrm{arg\,min}}\limits_f \mathbb{E}[L] = \mathbb{E}[t|x] = \int tp(t|x)dt$$
- つまり、最適な予測器は$t$の条件付き平均値

---
# 4.2 Decision theory
- 別の導出法:二乗損失を次のように展開する
$$\{f(x) - t\}^2 = \{f(x) - \mathbb{E}[t|x]\}^2 + 2\{f(x) - \mathbb{E}[t|x]\}\{\mathbb{E}[t|x] - t\} + \{\mathbb{E}[t|x] - t\}^2$$

---
# 4.2 Decision theory
- $t$について積分すると交差項はゼロになり:
$$\mathbb{E}[L] = \int \{f(x) - \mathbb{E}[t|x]\}^2p(x)dx + \int \mathrm{var}[t|x]p(x)dx$$

---
# 4.2 Decision theory
- 第1項のみが$f(x)$に依存し、$f(x) = \mathbb{E}[t|x]$のとき最小化される
- 第2項はxに関するtの分散の平均値で、不可逆的なノイズ

---
# 4.2 Decision theory
- 二乗損失以外の損失関数も考えられる
- 一般化したMinkowski損失:
$$\mathbb{E}[L_q] = \int\int |f(x) - t|^qp(x,t)dxdt$$
- $q=2$のとき通常の二乗損失

---
# 4.2 Decision theory
- $\mathbb{E}[L_q]$の最小値は:
    - $q=2$のとき条件付き平均値
    - $q=1$のとき条件付き中央値
    - $q\rightarrow 0$のとき条件付きモード値

---
# 4.2 Decision theory
- これまでは基底関数の形と数が与えられていた前提
- 最尤法では、データ数が少ない場合、過学習が起こりやすい
- 基底関数の数を減らすと過学習は抑えられるが、モデルの自由度が下がる

---
# 4.2 Decision theory
- 正則化は過学習を制御できるが、正則化係数$\lambda$の決め方が問題
- 適切な$\lambda$を選ぶために、バイアス・分散トレードオフを理解する必要がある

---
# 4.2 Decision theory
- バイアス・分散トレードオフについては次章で詳しく説明する
