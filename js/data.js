/* =====================================================================
 * data.js — 2次元分類用データセット生成
 *
 * 4種類のデータセット(2つの円 / XOR / 渦巻き / 2つのガウス分布)を生成する。
 * 座標空間は TensorFlow Playground と同じ [-6, 6] × [-6, 6]。
 *
 * 各データ点: { x, y, label }  (label: 0 = 青, 1 = オレンジ)
 *
 * グローバル名前空間 `DataGen` として公開。
 * ===================================================================== */

const DataGen = (() => {
  "use strict";

  /** 標準正規分布に従う乱数 (Box-Muller法) */
  function randn() {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  /** min〜max の一様乱数 */
  function rand(min, max) {
    return min + Math.random() * (max - min);
  }

  /* ----------------------------------------------------------------
   * 「2つの円」: 内側の円(クラス0)と外側のリング(クラス1)
   * noise はデータ点を正解領域からどれだけ揺らすか (0〜0.5)
   * ---------------------------------------------------------------- */
  function genCircle(n, noise) {
    const points = [];
    const radius = 5;
    for (let k = 0; k < n; k++) {
      const inner = k % 2 === 0; // 半々ずつ生成
      const r = inner ? rand(0, radius * 0.5) : rand(radius * 0.75, radius);
      const angle = rand(0, 2 * Math.PI);
      const x = r * Math.cos(angle) + randn() * noise * radius * 0.5;
      const y = r * Math.sin(angle) + randn() * noise * radius * 0.5;
      points.push({ x, y, label: inner ? 0 : 1 });
    }
    return points;
  }

  /* ----------------------------------------------------------------
   * 「XOR」: 第1・第3象限がクラス1、第2・第4象限がクラス0
   * ---------------------------------------------------------------- */
  function genXor(n, noise) {
    const points = [];
    for (let k = 0; k < n; k++) {
      let x = rand(-5, 5);
      let y = rand(-5, 5);
      // 原点付近にラベルが曖昧な点が密集しないよう少し離す
      x += x > 0 ? 0.3 : -0.3;
      y += y > 0 ? 0.3 : -0.3;
      const label = x * y >= 0 ? 1 : 0;
      points.push({
        x: x + randn() * noise * 2.5,
        y: y + randn() * noise * 2.5,
        label,
      });
    }
    return points;
  }

  /* ----------------------------------------------------------------
   * 「渦巻き(スパイラル)」: 2本の螺旋が絡み合う、最難関のデータセット
   * ---------------------------------------------------------------- */
  function genSpiral(n, noise) {
    const points = [];
    const half = Math.floor(n / 2);
    for (let label = 0; label <= 1; label++) {
      const deltaT = label === 0 ? 0 : Math.PI; // 2本目は180度ずらす
      for (let k = 0; k < half; k++) {
        const r = (k / half) * 5;             // 半径 0 → 5
        const t = ((1.75 * k) / half) * 2 * Math.PI + deltaT; // 回転角
        const x = r * Math.sin(t) + randn() * noise * 1.5;
        const y = r * Math.cos(t) + randn() * noise * 1.5;
        points.push({ x, y, label });
      }
    }
    return points;
  }

  /* ----------------------------------------------------------------
   * 「2つのガウス分布」: 左下と右上に分かれた2つの塊(最も簡単)
   * ---------------------------------------------------------------- */
  function genGauss(n, noise) {
    const points = [];
    // noise が大きいほど2つの山が広がって重なる
    const sigma = 0.5 + noise * 3.5;
    for (let k = 0; k < n; k++) {
      const label = k % 2;
      const cx = label === 1 ? 2.5 : -2.5;
      const cy = label === 1 ? 2.5 : -2.5;
      points.push({
        x: cx + randn() * sigma,
        y: cy + randn() * sigma,
        label,
      });
    }
    return points;
  }

  const GENERATORS = {
    circle: genCircle,
    xor: genXor,
    spiral: genSpiral,
    gauss: genGauss,
  };

  /* ----------------------------------------------------------------
   * データセットを生成し、学習用とテスト用に 50:50 で分割して返す。
   *   name:  "circle" | "xor" | "spiral" | "gauss"
   *   n:     総データ点数
   *   noise: ノイズ量 (0〜0.5)
   * 戻り値: { train: [...], test: [...] }
   * ---------------------------------------------------------------- */
  function generate(name, n, noise) {
    const gen = GENERATORS[name];
    if (!gen) throw new Error("未知のデータセット: " + name);
    const points = gen(n, noise);

    // シャッフル (Fisher–Yates)
    for (let i = points.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [points[i], points[j]] = [points[j], points[i]];
    }

    const half = Math.floor(points.length / 2);
    return { train: points.slice(0, half), test: points.slice(half) };
  }

  return { generate };
})();
