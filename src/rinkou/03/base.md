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

Section: 6.3.4 ~ 6.4
2024/6/6 Daiki Yoshikawa

---
<!--
class: slides
footer: 2024/06/06<span style="margin-left:250px;">Deep Learning : Foundations and Concepts 2024</span>
paginate: true
-->
# 目次
- 6.3 Deep Networks
  - 6.3.4 Transfer learning
  - 6.3.5 Contrastive learning
  - 6.3.6 General network architectures
  - 6.3.7 Tensors
- 6.4 Error Functions
  - 6.4.1 Regression
  - 6.4.2 BInary classification
  - 6.4.3 multiclass classification

---
<!--
_class: eyecatch
-->
# 6.3.4 Transfer learning

---
# 6.3.4 Transfer learning

- 1つのタスクで学習した内部表現を、関連するタスクに転移させる手法
- 大量のデータから学習した表現を、少数のデータで別のタスクに転移
  - 例: 大量のラベル付き一般物体画像から学習したネットワークを、皮膚病変検出に転移学習
- 少量のデータだけで学習するより高い精度を実現可能

---

![transfer_learning](./images/6_13.png)
図6.13 転移学習の模式図

---
# 6.3.4 Transfer learning
## 転移学習の要件

- 入力データが同種である (画像、テキストなど)
- 低次の特徴量が共通している
    - 例: 物体認識と皮膚病変検出では、低次の特徴が共通

適切に転移学習を行えば、ターゲットタスクの性能が大幅に向上する

---
# 6.3.4 Transfer learning
## 事前学習 (*pre-training*)
- あるタスクでパラメータを学習し、他のタスクに適用するプロセス
- 新しいタスクにおいては、識別層を含む一部の層のみを再学習
- ファインチューニング (*fine-tuning*) では、全ての層を再学習
  - 非常に小さい学習率とイテレーション数で学習することで
    過学習を防ぐ

---
# 6.3.4 Transfer learning
## マルチタスク学習 (*multitask learning*)
- 複数のタスクを同時に学習する手法
- 例： スパムメールフィルターをユーザーごとに学習したい場合
  - ユーザーごとのデータのみで学習するにはデータが不十分
  - 浅い層は共通、深い層はユーザーごとのパラメータをもつ１つのネットワークを学習
  - タスク間の共通性を利用することが可能

---
# 6.3.4 Transfer learning
## メタ学習 (*meta-learning*)
- タスク間の内部表現や学習アルゴリズム自体を学習する手法
- 新しいクラスのラベル付きデータが少ない場合に有効
  - **few-shot learning**: ラベル付きデータが少量の場合
  - **one-shot learning**: ラベル付きデータが1つの場合

---
<!--
_class: eyecatch
-->
# 6.3.5 Contrastive learning

---

# 6.3.5 Contrastive learning
- 最も一般的で強力な表現学習の手法の1つ
- 入力ペアのうち、ポジティブ（類似）なペアを近くに、ネガティブ（非類似）なペアを遠くに配置するように学習
- 分類などの下流タスクを容易にする

---
# 6.3.5 Contrastive learning
### $\mathbf{x}$ (*anchor*)が与えられたとき
  - $\mathbf{x^+}$: ポジティブペアを成すデータ点
  - $\{\mathbf{x}_1^-,\ldots,\mathbf{x}_N^-\}$: ネガティブペアを成すデータ点の集合
  - $\mathbf{x}$と$\mathbf{x}^+$の近さに報酬、$\{\mathbf{x}, \mathbf{x}_n^-\}$の近さにペナルティを与えるような損失関数が必要

**InfoNCE (noise contrastive estimation)損失関数:**
$$
{E}(\mathbf{w}) = -\ln \frac{\exp\{\mathbf{f}_w(\mathbf{x})^\top \mathbf{f_w}(\mathbf{x}^+)\}}{\exp\{\mathbf{f_w}(\mathbf{x})^\top \mathbf{f_w}(\mathbf{x}^+)\} + \sum_{n=1}^N \exp\{\mathbf{f_w}(\mathbf{x})^\top \mathbf{f_w}(\mathbf{x}_n^-)\}}
$$

---
# 6.3.5 Contrastive learning

- 対照学習のアルゴリズムはポジティブ、ネガティブのペアの選び方で主に決まる
  →事前知識を使って良い表現がどうあるべきかを指定
## 画像の場合
- 意味的な情報を保存しつつ入力画像を改変しポジティブペアとする
- データ拡張 (*data augmentation*) に密接に関連
  - 回転、平行移動、色変換など
- その他の画像はネガティブペアとする

→ **Instance Discrimination**と呼ばれる

---
# 6.3.5 Contrastive learning

## 教師付き対比学習
- 同一クラスの画像ペアをポジティブ、異なるクラスの画像ペアをネガティブペアとする
  - 拡張方法への依存の緩和
  - 意味的に近いペアをネガティブペアにすることを防ぐ
- クロスエントロピー損失を用いた通常の学習よりも性能が向上

---

# 6.3.5 Contrastive learning
## CLIP (Contrastive Language-Image Pre-training)
- 複数のモダリティからのデータを同じ表現空間に写す手法
- 画像とその説明文のペアをポジティブペアとする
- 画像とミスマッチな説明文のペアをネガティブペアとする
- 弱教師あり (*weakly supervised*) と呼ばれる

$$
\begin{align*}
E(\mathbf{w}) &= -\frac{1}{2}\ln\frac{\exp\{{f_w(x)^\top g_\theta(y^+)}\}}{\exp\{{f_w(x^+)^\top g_\theta(y^+)}\} + \sum_n \exp\{{f_w(x_n^-)^\top g_\theta(y^+)}\}} \\
                       &-\frac{1}{2}\ln\frac{\exp\{{f_w(x)^\top g_\theta(y^+)}\}}{\exp\{{f_w(x)^\top g_\theta(y^+)}\} + \sum_m \exp\{{f_w(x)^\top g_\theta(y_m^-)}\}}
                       \ \ \ \ (6.21)
\end{align*}
$$
---
# 6.3.5 Contrastive learning
![w:1200](./images/6_14.png)
図6.14 ３つの対照学習の模式図

---
<!--
_class: eyecatch
-->
# 6.3.6 General network architectures

---
# 6.3.6 General network architectures

- 層と層の間の接続関係を自由に設計可能
- 各ユニットでは、親ユニットの加重和に活性化関数を適用

$$z_k = h\left(\sum_{j \in \mathcal{A}(k)} w_{kj}z_j + b_k\right)$$

- 順伝播計算で全ユニットの活性値を求める
- フィードフォワード構造 (閉路がない) に限定される

---
<!--
_class: eyecatch
-->
# 6.3.7 Tensors

---
# 6.3.7 Tensors

- スカラー、ベクトル、行列を一般化した多次元配列
- 深層学習では様々な階数のテンソルを扱う
- 例: カラー画像データセット $\mathcal{X}$
    - $x_{ijkn}$: 画像 $n$, 色チャンネル $k$, 画素位置 $(i, j)$ の値
- GPUはテンソル演算に特化した並列アーキテクチャ

---

# 6.4 Error Functions

適切な誤差関数と出力活性化関数の組み合わせを選択

---
<!--
_class: eyecatch
-->
# 6.4.1 Regression

---

# 6.4.1 Regression

- 出力活性化関数: 恒等写像 $y_k = a_k$
- 誤差関数: 二乗和誤差

$$\mathcal{E}(w) = \frac{1}{2}\sum_{n=1}^N \|y(x_n, w) - t_n\|^2$$

- 出力層での勾配:

$$\frac{\partial \mathcal{E}}{\partial a_k} = y_k - t_k$$

---

### 最尤推定の観点

- 目標変数 $t$ の条件付き分布を正規分布と仮定:
    $$p(t|x, w) = \mathcal{N}(t|y(x, w), \sigma^2)$$
- 負の対数尤度を最小化することで二乗和誤差が導出される
- 分散 $\sigma^2$ は、最適な $w$ を求めた後に推定可能

---

### 複数の目標変数

- 目標変数が独立に正規分布に従うと仮定すれば、

$$p(t|x, w) = \mathcal{N}(t|y(x, w), \sigma^2I)$$

- 誤差関数は同じ形になる
- 独立性の仮定を外せば、もう少し一般的な問題設定

---
<!--
_class: eyecatch
-->
# 6.4.2 Binary classification

---
# 6.4.2 Binary classification

出力活性化関数: シグモイド関数 $y_k = \sigma(a_k)$
誤差関数: 交差エントロピー誤差

$$
E(\mathbf{w}) = -\sum_{n=1}^N \{t_n \text{ln} y_n + (1-t_n) \text{ln} (1-y_n)\}
$$

出力層での勾配:

$$\frac{\partial \mathcal{E}}{\partial a_k} = y_k - t_k$$

最尤推定の観点

目標変数 $t$ のベルヌーイ分布を仮定:
$$p(t|x, w) = y(x, w)^t (1 - y(x, w))^{1-t}$$
負の対数尤度を最小化することで交差エントロピー誤差が導出される
混同行列によるラベルノイズを許容することも可能


複数の二値分類

各クラスに対してシグモイド出力を用意する
全クラスの交差エントロピー誤差を足し合わせた形になる

$$\mathcal{E}(w) = -\sum_{n=1}^N\sum_{k=1}^K \left{t_{nk}\log y_{nk} + (1-t_{nk})\log(1-y_{nk})\right}$$

---
<!--
_class: eyecatch
-->
# 6.4.3 Multiclass classification

---
# 6.4.3 Multiclass classification

出力活性化関数: ソフトマックス関数

$$y_k(x, w) = \frac{e^{a_k(x, w)}}{\sum_j e^{a_j(x, w)}}$$

誤差関数: 多クラス交差エントロピー誤差

$$\mathcal{E}(w) = -\sum_{n=1}^N\sum_{k=1}^K t_{kn}\log y_k(x_n, w)$$

出力層での勾配:

$$\frac{\partial \mathcal{E}}{\partial a_k} = y_k - t_k$$

最尤推定の観点

目標変数 $t$ の多項分布を仮定:
$$p(t|x, w) = \prod_{k=1}^K y_k(x, w)^{t_k}$$
負の対数尤度を最小化することで多クラス交差エントロピー誤差が導出される

---
# まとめ

転移学習、対比学習により、有用な内部表現を学習可能
一般的な深層ネットワークでは、層間の接続関係が柔軟に設計可能
回帰、分類などの問題に応じた適切な出力活性化関数と誤差関数を選択
複数のモーダルなデータには、変種の対比学習手法が有効
テンソル演算は深層学習の基礎となる重要な概念