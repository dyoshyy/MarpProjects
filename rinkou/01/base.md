---
marp: true
theme: rinkou
math: mathjax
---
<!--
class: title
-->
# Deep Learning: Foundations and Concepts 2024

2024/4/18 Daiki Yoshikawa

---
<!--
class: slides
-->
## 目次
- 2 Probabilities  
  - 2.1 The Rules of Probability  
    - 2.1.1 A medical screening example
    - 2.1.2 The sum and product rules
    - 2.1.3 Bayes’theorem
    - 2.1.4 Medical screening revisited
    - 2.1.5 Prior and posterior probabilities
    - 2.1.6 Independent variables

---
<!--
class: slides
-->

## 2 Probabilities

- 機械学習では不確実性に常に直面する
- 不確実性には2種類ある
  - **認識論的不確実性**
    - 有限のデータセットサイズに起因
    - より多くのデータを観測することで軽減可能
  - **随伴的不確実性**
    - 世界の部分的な観測可能性に起因
    - 異なる種類のデータを集めることで軽減可能

---
<!--
class: slides
-->
# 2.1 The Rules of Probability

- 確率論は不確実性を定量化するための一貫したフレームワークを提供する
- 2つの簡単な法則に支配される
  - 加法則
  - 乗法則

---
<!--
class: slides
-->
## 2.1.1 A medical screening example

- がん早期発見のための集団検診
- 1%の人が実際にがんを患っている
- 検査:
  - 偽陽性率 = 3%
  - 偽陰性率 = 10%

---
<!--
class: slides
-->
## 2.1.1 A medical screening example

- 質問:
  1. 陽性反応の確率は?
  2. 陽性の場合、がんである確率は?

---
<!--
class: slides
-->
## 2.1.2 The sum and product rules

- **加法則**:
  $$p(X) = \sum_Y p(X, Y)$$

- **乗法則**:
  $$p(X, Y) = p(Y|X)p(X)$$
- 確率を支配する基本的な法則

---
<!--
class: slides
-->
## 2.1.3 Bayes' theorem

$$p(Y|X) = \frac{p(X|Y)p(Y)}{p(X)}$$

- "逆向き"の条件付き確率の関係を示す

---
<!--
class: slides
-->
## 2.1.3 Bayes' theorem

- 事前確率: $p(C)$
- 事後確率: $p(C|T)$
  - データを観測した後の確率

---
<!--
class: slides
-->
## 2.1.4 Medical screening revisited

- 加法則と乗法則を使って:
  - 陽性反応の確率: $p(T=1) = 0.0387$

---
<!--
class: slides
-->
## 2.1.4 Medical screening revisited

- ベイズの定理を使って:
  - 陽性の場合のがん確率: $p(C=1|T=1) = 0.23$
- 直感に反するが正しい
  - がんの事前確率が低い(1%)ため

---
<!--
class: slides
-->
## 2.1.5 Prior and Posterior Probabilities

- 事前確率: $p(C)$
  - 検査結果を観測する前の確率

---
<!--
class: slides
-->
## 2.1.5 Prior and Posterior Probabilities

- 事後確率: $p(C|T)$
  - 検査結果を観測した後の確率
- 例:
  - 事前確率 $p(C) = 1\%$
  - 事後確率 $p(C|T=1) = 23\%$
    - 陽性の場合、はるかに高い

---
<!--
class: slides
-->
## 2.1.6 Independent Variables

- $p(X, Y) = p(X)p(Y)$ の場合、$X$と$Y$は独立
- 例: コインを続けて投げる

---
<!--
class: slides
-->
## 2.1.6 Independent Variables

- 条件付き確率分布は変化しない:
  $$p(Y|X) = p(Y)$$
- 独立変数の場合:
  $$p(C|T) = p(C)$$
  - 検査結果はがんについて何も教えてくれない

---
<!--
class: slides
-->
# まとめ

- 確率論は不確実性を定量化するための一貫したフレームワークを提供する
- 主要な概念:
  - 加法則と乗法則
  - ベイズの定理
  - 事前確率と事後確率
  - 独立変数
- 多くの機械学習手法の基礎となる