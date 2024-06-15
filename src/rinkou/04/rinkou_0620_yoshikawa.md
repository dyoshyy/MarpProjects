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

Section: 9.3.2 ~ 9.4
2024/6/20 Daiki Yoshikawa

---
<!--
class: slides
footer: 2024/06/06<span style="margin-left:250px;">Deep Learning : Foundations and Concepts 2024</span>
paginate: true
-->
# 目次
- 9.3 Learning Curves
  - 9.3.2 Double descent
- 9.4 Parameter Sharing
  - 9.4.1 Soft weight sharing

---
<!--
_class: eyecatch
-->
# 9.3.2 Double descent

---
# 9.3.2 Double descent
## パラメータ数に対する従来の解釈

- モデルのパラメータ数を増やすと
  1. 表現力が向上し、検証エラーが減少(低バイアス)
  2. さらに増やすと過学習し、検証エラーが増加(高バリアンス)

  ⇒ **古典統計学における古説**
  - パラメータ数はデータセットの大きさに応じて制限
  - 非常に大きなモデルは性能が低い

---
# 9.3.2 Double descent
## DNNの実際の振る舞い
- 必要なパラメータ数を超えても高い性能を発揮することがある (Zhang et al., 2016)
- Early Stoppingが使用されて誤差ゼロまで学習されることがあり、その場合でも良好な性能を発揮

  ⇒従来の解釈と実際の振る舞いが矛盾？

---
# 9.3.2 Double descent
## 二重降下 (Double Descent, Belkin et al., 2019)
- 学習曲線,モデルの複雑と汎化性能の関係から矛盾は解消される

<div style="margin: -20px auto; text-align: center; font-size: 24px">
<img src="images/9_9.png" width="700" ><br>
<span style="margin: -0px auto;"> 図 9.9 パラメータ数と誤差の関係 </span>
</div>

---
# 9.3.2 Double descent
## 二重降下 (Double Descent, Belkin et al., 2019)
- 二段階に分けることができる
  1. 小から中程度のパラメータ数
    → バイアス-バリアンストレードオフが成り立つ
  2. 非常に大きなパラメータ数
    → テスト誤差が再び減少する

⇒ 二つの段階の境界における
　パラメータ数は？

<img src="images/9_9.png" width="560" style="margin: -230px 650;">

---
# 9.3.2 Double descent
## 二重降下 (Double Descent, Belkin et al., 2019)
- 訓練データに正確に適合するのに十分なパラメータ数のとき二回目の降下が発生 (Belkin et al., 2019)
- **effective model complexity** (Nakkiran et al., 2019)
  - 訓練誤差がゼロになる最大の訓練データ数
  - この値が訓練データ中のデータ数を超えると二重降下が発生

---
# 9.3.2 Double descent
## 二重降下と同様の振る舞い
- early stoppingを用いてモデルの複雑さを制御することで同様の振る舞いが観測される

<div style="margin: -20px auto; text-align: center; font-size: 24px">
<img src="images/9_10.png" width="600" ><br>
<span style="margin: -0px auto;"> 図 9.10 エポック数とテスト誤差の関係 </span>
</div>

---
# 9.3.2 Double descent
## 二重降下と同様の振る舞い
- $1/\lambda$ (正則化パラメータ) に対するテスト誤差の関係においても二重降下が観測される
  - $\lambda$が大きいとモデルの複雑さが抑制されるため

---
# 9.3.2 Double descent
## データ数と汎化性能
- Transformerにて埋め込み次元を増やすとモデルの複雑さが上昇
- 埋め込み次元を増やすと
テスト誤差は(図9.11)
  - 全体的に減少
  - 臨界領域では上昇

⇒ **データ数を増やして汎化性能が
減少する可能性がある**

<div style="position: absolute; top:260px; left: 650px; text-align: center;">
<div>
<img src="images/9_11.png" width="600" ><br>
<span style="font-size: 24px" >図 9.11 埋め込み次元数とテスト誤差の関係 </span>
</div>
</div>

---
<!--
_class: eyecatch
-->
# 9.4 Parameter Sharing

---
# 9.4 Parameter Sharing
## パラメータ共有
- $L_2$正則化は重みを小さくすることで過学習を抑制
  - ネットワークの複雑さを制限する別の方法としてパラメータ共有
- 重みをグループに分け、各グループ内で重みが同じ値を取るようにする
- これによりネットワークの自由度が下がり、モデル複雑さが制限される
- 例えば畳み込みニューラルネットワークでは、この手法を用いてデータの平行移動などの不変性をエンコーディングする

---
<!--
_class: eyecatch
-->
# 9.4.1 Soft weight sharing
---
# 9.4.1 Soft weight sharing
## Soft Weight Sharing: 概要

- 重みをガウス混合モデルでクラスタリングし、各グループ内で重みが類似した値を取るよう正則化する柔軟なアプローチ
- ガウス成分の混合比率 $\{\pi_j\}$、平均 $\{\mu_j\}$、分散 $\{\sigma_j^2\}$ もデータから学習される

--- 
# 9.4.1 Soft weight sharing
## Soft Weight Sharing: 重みの事前分布

- 重みwの事前分布をガウス混合モデルで定義
  - $p(w) = \prod_i \sum_j \pi_j \mathcal{N}(w_i|\mu_j, \sigma_j^2)$ (式9.21)
- この事前分布の対数の負値を正則化項 $\Omega(w)$ とする
  - $\Omega(w) = -\sum_i \ln(\sum_j \pi_j \mathcal{N}(w_i|\mu_j, \sigma_j^2))$ (式9.22)

---
# 9.4.1 Soft weight sharing
## Soft Weight Sharing: 損失関数

- 総損失関数は誤差関数とSoft Weight Sharing の正則化項の和
  - $\tilde{E}(w) = E(w) + \lambda\Omega(w)$ (式9.23)
- この損失関数を{wi}と{πj, μj, σj}それぞれについて勾配降下法で最小化する

---
# 9.4.1 Soft weight sharing
## Soft Weight Sharing: 更新則

- 重み $w_i$ の勾配
  - $\frac{\partial\tilde{E}}{\partial w_i} = \frac{\partial E}{\partial w_i} + \lambda\sum_j \gamma_j(w_i)\frac{w_i - \mu_j}{\sigma_j^2}$ (式9.25)
- 平均 $\mu_j$ の勾配 
  - $\frac{\partial\tilde{E}}{\partial\mu_j} = \lambda\sum_i \gamma_j(w_i)\frac{\mu_j - w_i}{\sigma_j^2}$ (式9.26)  
- 分散 $\sigma_j^2$ の勾配 (ロバストな実装で $\sigma_j^2 = e^{\xi_j}$ と置く)
  - $\frac{\partial\tilde{E}}{\partial\xi_j} = \frac{\lambda}{2}\sum_i \gamma_j(w_i)\left(1 - \frac{(w_i - \mu_j)^2}{\sigma_j^2}\right)$ (式9.28)

$\gamma_j(w_i)$ は $w_i$ が成分 $j$ から生成された事後確率

---
# 9.4.1 Soft weight sharing
## Soft Weight Sharing: 長所

- 硬直的なパラメータ共有の制約を緩和し、柔軟なグループ分けが可能
- グループ内の重み分布の特性も自動で最適化できる
- 確率モデルに基づく理論的根拠が明確
- 教師なしデータからの事前知識を教師ありモデルに取り入れるハイブリッドモデルへの応用も可能

正則化手法として現在でも広く用いられている有力な手法である
