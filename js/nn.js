/* =====================================================================
 * nn.js — 全結合ニューラルネットワーク本体（自前実装）
 *
 * このファイルがアプリの核。外部ライブラリを使わず、
 * 順伝播 (forward propagation)・逆伝播 (backpropagation)・
 * ミニバッチ勾配降下法 (mini-batch SGD) を素朴に実装している。
 *
 * 構成:
 *   - 入力は2次元 (x, y)。データ空間 [-6, 6] を /6 して [-1, 1] に
 *     正規化してから入力する(呼び出し側 main.js の責務)。
 *   - 隠れ層の活性化関数は relu / tanh / sigmoid から選択。
 *   - 出力層は常に sigmoid 1ニューロン。「クラス1である確率」を出す。
 *   - 損失は二値交差エントロピー (binary cross-entropy)。
 *     sigmoid + 交差エントロピーの組み合わせでは、出力層の勾配が
 *     (予測値 - 正解) というシンプルな形になり、数値的にも安定する。
 *
 * グローバル名前空間 `NN` として公開する。
 * (file:// で開けるよう ES modules を使わないため)
 * ===================================================================== */

const NN = (() => {
  "use strict";

  /* ----------------------------------------------------------------
   * 活性化関数とその導関数
   * df は「出力値 y (= f(z)) から導関数を計算する」形にしてある。
   * tanh と sigmoid は出力値だけで導関数が書けるので効率がよい。
   * relu だけは入力 z の符号が必要なので z も受け取る。
   * ---------------------------------------------------------------- */
  const ACTIVATIONS = {
    relu: {
      f: (z) => (z > 0 ? z : 0),
      df: (z, y) => (z > 0 ? 1 : 0),
    },
    tanh: {
      f: Math.tanh,
      df: (z, y) => 1 - y * y, // tanh'(z) = 1 - tanh(z)^2
    },
    sigmoid: {
      f: (z) => 1 / (1 + Math.exp(-z)),
      df: (z, y) => y * (1 - y), // σ'(z) = σ(z)(1 - σ(z))
    },
  };

  /** 標準正規分布に従う乱数 (Box-Muller法) */
  function randn() {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  /* ----------------------------------------------------------------
   * Network クラス
   *   layerSizes: 各層のニューロン数の配列。例 [2, 4, 4, 1]
   *               (入力2 → 隠れ4 → 隠れ4 → 出力1)
   *   activationName: 隠れ層の活性化関数名 ("relu" | "tanh" | "sigmoid")
   * ---------------------------------------------------------------- */
  class Network {
    constructor(layerSizes, activationName) {
      this.sizes = layerSizes.slice();
      this.activationName = activationName;
      this.act = ACTIVATIONS[activationName];
      if (!this.act) throw new Error("未知の活性化関数: " + activationName);

      const L = layerSizes.length - 1; // 重み行列の数(層間の接続数)

      // weights[l][j][i] = 第l層の i番ニューロン → 第l+1層の j番ニューロン の重み
      // biases[l][j]     = 第l+1層の j番ニューロン のバイアス
      this.weights = [];
      this.biases = [];

      for (let l = 0; l < L; l++) {
        const nIn = layerSizes[l];
        const nOut = layerSizes[l + 1];
        // Xavier (Glorot) 初期化:
        // 分散 2/(nIn+nOut) の正規分布で初期化すると、信号の大きさが
        // 層を通っても保たれやすく、学習初期から安定して収束する。
        const scale = Math.sqrt(2 / (nIn + nOut));
        const W = [];
        const b = [];
        for (let j = 0; j < nOut; j++) {
          const row = [];
          for (let i = 0; i < nIn; i++) row.push(randn() * scale);
          W.push(row);
          b.push(0); // バイアスは0で初期化するのが定石
        }
        this.weights.push(W);
        this.biases.push(b);
      }
    }

    /* --------------------------------------------------------------
     * 順伝播 (forward propagation)
     * 入力ベクトルを層ごとに変換していき、最終出力(確率)を得る。
     * 逆伝播で使うため、各層の z (線形和) と a (活性化後) を保存して返す。
     *
     * 戻り値: { zs, as }
     *   as[0] = 入力, as[最後] = 出力(確率)
     * -------------------------------------------------------------- */
    forward(input) {
      const L = this.weights.length;
      const zs = [];          // 各層の線形和 z = Wa + b
      const as = [input];     // 各層の活性化後の値

      let a = input;
      for (let l = 0; l < L; l++) {
        const W = this.weights[l];
        const b = this.biases[l];
        const nOut = W.length;
        const z = new Array(nOut);
        const aNext = new Array(nOut);
        // 出力層(最後の層)は sigmoid 固定、隠れ層は選択された活性化関数
        const isOutput = l === L - 1;
        const f = isOutput ? ACTIVATIONS.sigmoid.f : this.act.f;

        for (let j = 0; j < nOut; j++) {
          let sum = b[j];
          const Wj = W[j];
          for (let i = 0; i < a.length; i++) sum += Wj[i] * a[i];
          z[j] = sum;
          aNext[j] = f(sum);
        }
        zs.push(z);
        as.push(aNext);
        a = aNext;
      }
      return { zs, as };
    }

    /** 1点だけ予測して確率(0〜1)を返す。可視化用の軽量版。 */
    predict(input) {
      const { as } = this.forward(input);
      return as[as.length - 1][0];
    }

    /* --------------------------------------------------------------
     * ミニバッチ学習を1回実行する
     *   batch: [{ input: [x, y], label: 0 | 1 }, ...]
     *   lr: 学習率
     * 戻り値: このバッチの平均損失(交差エントロピー)
     *
     * 逆伝播 (backpropagation) の流れ:
     *   1. 順伝播して各層の z, a を記録
     *   2. 出力層の誤差 δ = (予測 p - 正解 t) を計算
     *      (sigmoid + 交差エントロピー だとこの形に簡約される)
     *   3. δ を後ろの層から前の層へ伝播:
     *      δ_prev = (Wᵀ δ) ⊙ f'(z_prev)
     *   4. 勾配 ∂Loss/∂W = δ ⊗ a_prev をバッチ全体で平均
     *   5. 勾配降下法で更新: W ← W - lr * 勾配
     * -------------------------------------------------------------- */
    trainBatch(batch, lr) {
      const L = this.weights.length;

      // 勾配の累積バッファ(重みと同じ形でゼロ初期化)
      const gradW = this.weights.map((W) => W.map((row) => row.map(() => 0)));
      const gradB = this.biases.map((b) => b.map(() => 0));

      let totalLoss = 0;

      for (const sample of batch) {
        const { zs, as } = this.forward(sample.input);
        const p = as[L][0];     // 予測確率
        const t = sample.label; // 正解ラベル (0 または 1)

        // 二値交差エントロピー損失。log(0) を避けるためクランプする。
        const eps = 1e-7;
        const pc = Math.min(1 - eps, Math.max(eps, p));
        totalLoss += -(t * Math.log(pc) + (1 - t) * Math.log(1 - pc));

        // --- 逆伝播 ---
        // 出力層の誤差。sigmoid+交差エントロピーの簡約形。
        let delta = [p - t];

        for (let l = L - 1; l >= 0; l--) {
          const aPrev = as[l]; // この層への入力

          // この層の勾配を累積
          for (let j = 0; j < delta.length; j++) {
            const gWj = gradW[l][j];
            const d = delta[j];
            for (let i = 0; i < aPrev.length; i++) gWj[i] += d * aPrev[i];
            gradB[l][j] += d;
          }

          // 1つ前の層へ誤差を伝播 (入力層まで来たら不要)
          if (l > 0) {
            const W = this.weights[l];
            const zPrev = zs[l - 1];
            const newDelta = new Array(aPrev.length);
            for (let i = 0; i < aPrev.length; i++) {
              let sum = 0;
              for (let j = 0; j < delta.length; j++) sum += W[j][i] * delta[j];
              // 活性化関数の導関数を掛ける(連鎖律)
              newDelta[i] = sum * this.act.df(zPrev[i], aPrev[i]);
            }
            delta = newDelta;
          }
        }
      }

      // --- 勾配降下法による更新 (バッチ平均) ---
      const n = batch.length;
      for (let l = 0; l < L; l++) {
        const W = this.weights[l];
        const b = this.biases[l];
        for (let j = 0; j < W.length; j++) {
          const Wj = W[j];
          const gWj = gradW[l][j];
          for (let i = 0; i < Wj.length; i++) Wj[i] -= (lr * gWj[i]) / n;
          b[j] -= (lr * gradB[l][j]) / n;
        }
      }

      return totalLoss / n;
    }

    /* --------------------------------------------------------------
     * データセット全体の平均損失を計算する(学習はしない)。
     * 損失グラフのテストデータ曲線などに使う。
     * -------------------------------------------------------------- */
    evaluate(samples) {
      if (samples.length === 0) return 0;
      const eps = 1e-7;
      let total = 0;
      for (const s of samples) {
        const p = this.predict(s.input);
        const pc = Math.min(1 - eps, Math.max(eps, p));
        total += -(s.label * Math.log(pc) + (1 - s.label) * Math.log(1 - pc));
      }
      return total / samples.length;
    }

    /** 重みのどこかに NaN / Infinity が混ざっていないか(発散検知用) */
    hasInvalidWeights() {
      for (const W of this.weights)
        for (const row of W)
          for (const w of row) if (!isFinite(w)) return true;
      for (const b of this.biases)
        for (const v of b) if (!isFinite(v)) return true;
      return false;
    }
  }

  return { Network, ACTIVATIONS };
})();
