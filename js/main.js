/* =====================================================================
 * main.js — ニューラルネットワーク・プレイグラウンド メインロジック
 *
 * 依存: nn.js (NN.Network) / data.js (DataGen.generate)
 *       ← HTML側で先に読み込まれる前提
 *
 * 構成:
 *   1. 定数・設定
 *   2. 状態 (state)
 *   3. 初期化・リセット
 *   4. 学習 (training)
 *   5. 描画 (rendering)
 *   6. UIイベント
 *   7. メインループ
 * ===================================================================== */

const App = (() => {
  "use strict";

  /* ================================================================
   * 1. 定数・設定
   * ================================================================ */

  /** データセット設定 (種類とノイズ量は state で可変) */
  const DATASET_N       = 300;

  /** 学習ループ設定 */
  const EPOCHS_PER_FRAME = 1;   // 1フレームに何エポック学習するか (境界の変化をゆっくり見せる)
  const BATCH_SIZE       = 10;  // ミニバッチサイズ

  /** ヒートマップ解像度 */
  const HEATMAP_GRID = 56;

  /** 座標空間の範囲 [-COORD_RANGE, COORD_RANGE] */
  const COORD_RANGE = 6;

  /** データ点の半径 (px, CSS空間) */
  const DOT_RADIUS = 3.5;

  /** ヒートマップ最大 alpha */
  const HEATMAP_MAX_ALPHA = 0.55;

  /* ================================================================
   * 2. 状態 (state) — 全状態をここに集約
   * ================================================================ */
  const state = {
    /** ニューラルネットワークインスタンス */
    net: null,

    /** 隠れ層のニューロン数配列 (カスタマイズ可変) */
    hidden: [4, 4],

    /** 活性化関数名 ("relu" | "tanh" | "sigmoid") */
    activation: "tanh",

    /** 学習率 (0.001〜1.0) */
    learningRate: 0.3,

    /** 選択中のデータセット名 ("circle" | "xor" | "spiral" | "gauss") */
    dataset: "xor",

    /** ノイズ量 (0〜0.5) */
    noise: 0.1,

    /** データセット */
    trainData: [],
    testData:  [],

    /** 現在のエポック数 */
    epoch: 0,

    /** 直近の学習損失 */
    trainLoss: null,

    /** 直近のテスト損失 */
    testLoss: null,

    /** 再生中か否か */
    isPlaying: false,

    /** requestAnimationFrame の ID (キャンセル用) */
    rafId: null,

    /** トースト表示タイマーID */
    toastTimer: null,

    /** 一時停止中でも次フレームで再描画するか (アイドル時のCPU節約) */
    dirty: true,

    /** アニメーション位相の基準時刻 */
    startTime: performance.now(),
  };

  /* ================================================================
   * 3. 初期化・リセット
   * ================================================================ */

  /**
   * データセットを生成し、各点に正規化済み input を付与する。
   * input = [p.x / COORD_RANGE, p.y / COORD_RANGE]
   */
  function initData() {
    const { train, test } = DataGen.generate(state.dataset, DATASET_N, state.noise);

    // ネットワーク入力用に座標を正規化して input プロパティとして付与
    const normalize = (pts) =>
      pts.map((p) => ({ ...p, input: [p.x / COORD_RANGE, p.y / COORD_RANGE] }));

    state.trainData = normalize(train);
    state.testData  = normalize(test);
  }

  /**
   * ネットワークをリセット (新規インスタンス生成 + カウンタ初期化)。
   * ボタン操作、発散時の自動リセット共通で呼ぶ。
   */
  function resetNetwork() {
    // 入力2 + 隠れ層 + 出力1 でレイヤー配列を構成
    const layerSizes = [2, ...state.hidden, 1];
    state.net       = new NN.Network(layerSizes, state.activation);
    state.epoch     = 0;
    state.trainLoss = null;
    state.testLoss  = null;
    heatFrame       = 0; // 次の描画で必ずヒートマップを再計算する
    // 損失グラフ履歴もクリア
    LossView.clear();
    updateStatsUI();
  }

  /* ================================================================
   * 4. 学習 (training)
   * ================================================================ */

  /**
   * Fisher–Yates シャッフル (インプレース)
   * @param {Array} arr
   */
  function shuffle(arr) {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = (Math.random() * (i + 1)) | 0;
      const tmp = arr[i];
      arr[i] = arr[j];
      arr[j] = tmp;
    }
  }

  /**
   * 1エポック学習を実行する。
   * 学習データをシャッフルしてミニバッチに分割、全件1周。
   * @returns {number} そのエポックの平均損失
   */
  function trainEpoch() {
    shuffle(state.trainData);
    let totalLoss = 0;
    let batchCount = 0;

    for (let i = 0; i < state.trainData.length; i += BATCH_SIZE) {
      const batch = state.trainData.slice(i, i + BATCH_SIZE);
      totalLoss += state.net.trainBatch(batch, state.learningRate);
      batchCount++;
    }

    return batchCount > 0 ? totalLoss / batchCount : 0;
  }

  /**
   * 発散チェック。発散していたら自動リセットしてトーストを表示。
   * @param {number} loss
   * @returns {boolean} 発散した場合 true
   */
  function checkDivergence(loss) {
    if (!isFinite(loss) || state.net.hasInvalidWeights()) {
      const wasPlaying = state.isPlaying;
      // 再生を一旦止めてリセット
      state.isPlaying = false;
      resetNetwork();
      initData(); // データも再生成
      // 再生状態を復元 (発散前と同じ)
      state.isPlaying = wasPlaying;
      updatePlayPauseBtn();
      showToast("数値が発散したためリセットしました", 3000);
      return true;
    }
    return false;
  }

  /**
   * 複数エポック学習 (フレームごとに呼ぶ)。
   * @param {number} count 学習するエポック数
   */
  function trainMultipleEpochs(count) {
    for (let i = 0; i < count; i++) {
      const loss = trainEpoch();
      state.epoch++;
      state.trainLoss = loss;
      // テスト損失はエポックごとに計算
      state.testLoss = state.net.evaluate(state.testData);

      // 損失グラフ履歴に追加
      LossView.push(state.epoch, loss, state.testLoss);

      // 発散したらループ中断
      if (checkDivergence(loss)) return;
    }
  }

  /* ================================================================
   * 5. 描画 (rendering)
   * ================================================================ */

  // Canvas と 2D コンテキスト
  const canvas = document.getElementById("main-canvas");
  const ctx    = canvas.getContext("2d");

  // ネットワーク図 Canvas と 2D コンテキスト (NetView が描画)
  const netCanvas = document.getElementById("network-canvas");
  const netCtx    = netCanvas.getContext("2d");

  // 損失グラフ Canvas と 2D コンテキスト (LossView が描画)
  const lossCanvas = document.getElementById("loss-canvas");
  const lossCtx    = lossCanvas.getContext("2d");

  // ヒートマップ用オフスクリーン Canvas
  const offscreen    = document.createElement("canvas");
  offscreen.width    = HEATMAP_GRID;
  offscreen.height   = HEATMAP_GRID;
  const offscreenCtx = offscreen.getContext("2d");

  // 決定境界ヒートマップの間引きカウンタ
  // (56×56 = 3136回の順伝播は深い構成では重いので、再生中は2フレームに1回)
  let heatFrame = 0;

  /**
   * devicePixelRatio を考慮して canvas の内部解像度を設定する。
   * ぼやけ防止のため CSS サイズとは分離する。
   */
  function resizeCanvas() {
    const dpr  = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width  = Math.round(rect.width  * dpr);
    canvas.height = Math.round(rect.height * dpr);
    ctx.scale(dpr, dpr);
  }

  /**
   * ネットワーク図 canvas の内部解像度を devicePixelRatio に合わせる。
   * resizeCanvas と同じ方式 (CSS サイズと内部解像度を分離)。
   */
  function resizeNetCanvas() {
    const dpr  = window.devicePixelRatio || 1;
    const rect = netCanvas.getBoundingClientRect();
    netCanvas.width  = Math.round(rect.width  * dpr);
    netCanvas.height = Math.round(rect.height * dpr);
    netCtx.scale(dpr, dpr);
  }

  /**
   * 損失グラフ canvas の内部解像度を devicePixelRatio に合わせる。
   * resizeCanvas / resizeNetCanvas と同じ方式。
   */
  function resizeLossCanvas() {
    const dpr  = window.devicePixelRatio || 1;
    const rect = lossCanvas.getBoundingClientRect();
    lossCanvas.width  = Math.round(rect.width  * dpr);
    lossCanvas.height = Math.round(rect.height * dpr);
    lossCtx.scale(dpr, dpr);
  }

  /**
   * データ空間座標 [-COORD_RANGE, COORD_RANGE] → canvas CSS 座標 に変換。
   * y軸は上が正なので反転する。
   * @param {number} x
   * @param {number} y
   * @param {number} cssW  canvas の CSS 幅
   * @param {number} cssH  canvas の CSS 高さ
   */
  function dataToCanvas(x, y, cssW, cssH) {
    return {
      cx: ((x + COORD_RANGE) / (COORD_RANGE * 2)) * cssW,
      // y軸反転: データ空間の上(+)が画面の上になるよう
      cy: ((COORD_RANGE - y) / (COORD_RANGE * 2)) * cssH,
    };
  }

  /**
   * 確率値 p (0〜1) から RGBA 配列 [r, g, b, a] を計算する。
   *   クラス0側 (p→0) = AppTheme.c0 / クラス1側 (p→1) = AppTheme.c1
   *   p=0.5 付近: alpha ≈ 0 (プロット下地色が透けて見える)
   * @param {number} p 0〜1
   * @returns {number[]} [r, g, b, a0〜255]
   */
  function probToRgba(p) {
    // γ=0.75 で持ち上げ、p=0.5 付近の弱い判断も淡く読めるようにする
    const intensity = Math.pow(Math.abs(p - 0.5) * 2, 0.75);
    const alpha = intensity * HEATMAP_MAX_ALPHA;

    const rgb = p < 0.5 ? window.AppTheme.c0.rgb : window.AppTheme.c1.rgb;
    return [rgb[0], rgb[1], rgb[2], Math.round(alpha * 255)];
  }

  /**
   * HEATMAP_GRID四方のオフスクリーン canvas にヒートマップを描画する。
   * プロット下地色 (AppTheme.bgPlot) の上に確率に応じた色を重ねる。
   */
  function drawHeatmap() {
    const size      = HEATMAP_GRID;
    const imageData = offscreenCtx.createImageData(size, size);
    const data      = imageData.data;

    for (let row = 0; row < size; row++) {
      for (let col = 0; col < size; col++) {
        // グリッドセルの中心座標をデータ空間に変換
        const dataX = ((col + 0.5) / size) * (COORD_RANGE * 2) - COORD_RANGE;
        // row 0 = 上 = y 正方向 なので反転
        const dataY = COORD_RANGE - ((row + 0.5) / size) * (COORD_RANGE * 2);

        const p = state.net.predict([dataX / COORD_RANGE, dataY / COORD_RANGE]);
        const [r, g, b, a] = probToRgba(p);

        const idx = (row * size + col) * 4;

        // プロット下地色の上に alpha ブレンド
        const [bgR, bgG, bgB] = window.AppTheme.bgPlot;
        const fa  = a / 255; // 0〜1
        data[idx    ] = Math.round(bgR * (1 - fa) + r * fa);
        data[idx + 1] = Math.round(bgG * (1 - fa) + g * fa);
        data[idx + 2] = Math.round(bgB * (1 - fa) + b * fa);
        data[idx + 3] = 255; // 完全不透明(背景込みで合成済み)
      }
    }

    offscreenCtx.putImageData(imageData, 0, 0);
  }

  /**
   * メイン canvas に全てを描画する。
   *   1. 背景塗りつぶし
   *   2. ヒートマップ (オフスクリーン → 拡大描画)
   *   3. 学習データ点の描画
   */
  function render() {
    const dpr  = window.devicePixelRatio || 1;
    const cssW = canvas.width  / dpr;
    const cssH = canvas.height / dpr;

    // --- 背景 (プロット下地色 = --bg-plot) ---
    ctx.fillStyle = "#0E0F12";
    ctx.fillRect(0, 0, cssW, cssH);

    // --- ヒートマップ ---
    // 再生中は2フレームに1回だけ再計算し、間のフレームは
    // オフスクリーンを使い回す (1エポック分の境界変化は目視できないため)。
    // 一時停止中の単発描画 (設定変更・1ステップ後など) では必ず再計算する。
    if (!state.isPlaying || heatFrame % 2 === 0) drawHeatmap();
    heatFrame++;
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(offscreen, 0, 0, cssW, cssH);

    // --- グリッド線 (薄い) ---
    drawGrid(cssW, cssH);

    // --- データ点 (学習データのみ) ---
    drawDataPoints(cssW, cssH);

    // --- ネットワーク図 (NetView モジュールに委譲) ---
    const netCssW = netCanvas.width  / dpr;
    const netCssH = netCanvas.height / dpr;
    NetView.render(netCtx, state.net, netCssW, netCssH, {
      time: (performance.now() - state.startTime) / 1000,
      active: state.isPlaying,
      flow: window.AppFlags.flow,
      force: !state.isPlaying, // 一時停止中の単発描画では必ず最新を計算
    });

    // --- 損失グラフ (LossView モジュールに委譲) ---
    const lossCssW = lossCanvas.width  / dpr;
    const lossCssH = lossCanvas.height / dpr;
    LossView.render(lossCtx, lossCssW, lossCssH);

    state.dirty = false;
  }

  /**
   * 座標グリッドを描画する (Tweaksで非表示可)。
   * 2単位ごとの細線 + x=0 / y=0 の軸線。
   */
  function drawGrid(cssW, cssH) {
    if (!window.AppFlags.grid) return;
    ctx.save();
    ctx.lineWidth = 1;

    // 細グリッド (2単位ごと、軸は除く)
    ctx.strokeStyle = "rgba(255, 255, 255, 0.035)";
    for (let v = -4; v <= 4; v += 2) {
      if (v === 0) continue;
      const { cx } = dataToCanvas(v, 0, cssW, cssH);
      const { cy } = dataToCanvas(0, v, cssW, cssH);
      ctx.beginPath();
      ctx.moveTo(cx, 0);
      ctx.lineTo(cx, cssH);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, cy);
      ctx.lineTo(cssW, cy);
      ctx.stroke();
    }

    // 軸線 (x=0, y=0)
    const { cx: axisX, cy: axisY } = dataToCanvas(0, 0, cssW, cssH);
    ctx.strokeStyle = "rgba(255, 255, 255, 0.10)";
    ctx.beginPath();
    ctx.moveTo(axisX, 0);
    ctx.lineTo(axisX, cssH);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(0, axisY);
    ctx.lineTo(cssW, axisY);
    ctx.stroke();

    ctx.restore();
  }

  /**
   * 学習データ点を canvas に描画する。
   * 塗り = クラス色、縁取り = 地色の暗いリング (ヒートマップから点を浮かせる)。
   */
  function drawDataPoints(cssW, cssH) {
    const theme = window.AppTheme;
    ctx.save();
    ctx.lineWidth = 1.2;
    ctx.strokeStyle = "rgba(10, 11, 13, 0.75)";

    for (const p of state.trainData) {
      const { cx, cy } = dataToCanvas(p.x, p.y, cssW, cssH);
      ctx.beginPath();
      ctx.arc(cx, cy, DOT_RADIUS, 0, Math.PI * 2);
      ctx.fillStyle = p.label === 1 ? theme.c1.hex : theme.c0.hex;
      ctx.fill();
      ctx.stroke();
    }

    ctx.restore();
  }

  /* ================================================================
   * 6. UI 更新
   * ================================================================ */

  // DOM 要素キャッシュ
  const elEpoch      = document.getElementById("stat-epoch");
  const elTrainLoss  = document.getElementById("stat-train-loss");
  const elTestLoss   = document.getElementById("stat-test-loss");
  const btnPlayPause = document.getElementById("btn-playpause");
  const btnStep      = document.getElementById("btn-step");
  const btnReset     = document.getElementById("btn-reset");
  const toastEl      = document.getElementById("toast");

  // ネットワーク構造コントロール
  const btnLayerMinus    = document.getElementById("btn-layer-minus");
  const btnLayerPlus     = document.getElementById("btn-layer-plus");
  const elLayerCount     = document.getElementById("layer-count-display");
  const elNeuronChips    = document.getElementById("neuron-chips");
  const elArch           = document.getElementById("stat-arch");

  // 活性化関数セグメントボタン
  const activationBtns   = document.querySelectorAll(".activation-btn");

  // 学習率スライダー
  const lrSlider         = document.getElementById("lr-slider");
  const lrValue          = document.getElementById("lr-value");

  /** 統計表示を state に合わせて更新する */
  function updateStatsUI() {
    elEpoch.textContent = state.epoch;

    const hasTrain = state.trainLoss !== null && isFinite(state.trainLoss);
    elTrainLoss.textContent = hasTrain ? state.trainLoss.toFixed(4) : "—";
    elTrainLoss.classList.toggle("is-empty", !hasTrain);

    const hasTest = state.testLoss !== null && isFinite(state.testLoss);
    elTestLoss.textContent = hasTest ? state.testLoss.toFixed(4) : "—";
    elTestLoss.classList.toggle("is-empty", !hasTest);
  }

  /**
   * 学習率の値を読みやすい形式にフォーマットする。
   * @param {number} lr
   * @returns {string}
   */
  function formatLR(lr) {
    if (lr >= 0.1)  return lr.toFixed(2);
    if (lr >= 0.01) return lr.toFixed(3);
    return lr.toFixed(4);
  }

  /**
   * ネットワーク構造コントロールを state.hidden に合わせて再構築する。
   * 隠れ層の数表示・各層のニューロン数チップを再生成する。
   */
  function updateNetworkControls() {
    const count = state.hidden.length;

    // 隠れ層数表示
    elLayerCount.textContent = count + " 層";

    // 読み出しストリップの構成表示 (例 "2 · 4 · 4 · 1")
    if (elArch) {
      elArch.textContent = [2, ...state.hidden, 1].join(" · ");
    }

    // ステッパーの上下限 disabled
    btnLayerMinus.disabled = (count <= 0);
    btnLayerPlus.disabled  = (count >= 4);

    // ニューロン数チップを再生成
    elNeuronChips.innerHTML = "";
    state.hidden.forEach((n, i) => {
      const chip = document.createElement("div");
      chip.className = "neuron-chip";

      const label = document.createElement("span");
      label.className = "neuron-chip-label";
      label.textContent = "L" + (i + 1);

      const btnMinus = document.createElement("button");
      btnMinus.className = "stepper-btn";
      btnMinus.textContent = "−";
      btnMinus.disabled = (n <= 1);
      btnMinus.setAttribute("aria-label", "隠れ層" + (i + 1) + "のニューロン数を減らす");
      btnMinus.addEventListener("click", () => {
        if (state.hidden[i] > 1) {
          state.hidden[i]--;
          updateNetworkControls();
          applySettingsChange();
        }
      });

      const elCount = document.createElement("span");
      elCount.className = "neuron-chip-count";
      elCount.textContent = n;

      const btnPlus = document.createElement("button");
      btnPlus.className = "stepper-btn";
      btnPlus.textContent = "+";
      btnPlus.disabled = (n >= 8);
      btnPlus.setAttribute("aria-label", "隠れ層" + (i + 1) + "のニューロン数を増やす");
      btnPlus.addEventListener("click", () => {
        if (state.hidden[i] < 8) {
          state.hidden[i]++;
          updateNetworkControls();
          applySettingsChange();
        }
      });

      chip.appendChild(label);
      chip.appendChild(btnMinus);
      chip.appendChild(elCount);
      chip.appendChild(btnPlus);
      elNeuronChips.appendChild(chip);
    });
  }

  /** 再生/一時停止ボタンの外観を state.isPlaying に合わせて更新 */
  function updatePlayPauseBtn() {
    if (state.isPlaying) {
      btnPlayPause.classList.add("is-playing");
      btnPlayPause.querySelector(".btn-icon").textContent = "⏸";
      btnPlayPause.querySelector(".btn-text").textContent = "一時停止";
    } else {
      btnPlayPause.classList.remove("is-playing");
      btnPlayPause.querySelector(".btn-icon").textContent = "▶";
      btnPlayPause.querySelector(".btn-text").textContent = "再生";
    }
  }

  /**
   * トースト通知を表示する。
   * @param {string} message 表示テキスト
   * @param {number} duration 表示時間(ms)
   */
  function showToast(message, duration) {
    // 既存タイマーをクリア
    if (state.toastTimer) clearTimeout(state.toastTimer);

    toastEl.textContent = message;
    toastEl.classList.add("is-visible");

    state.toastTimer = setTimeout(() => {
      toastEl.classList.remove("is-visible");
      state.toastTimer = null;
    }, duration);
  }

  /* ================================================================
   * 6.5 データセットサムネイル
   *   ボタン内の小さな canvas に、実際のデータ生成器による
   *   散布図プレビューを描く。データは初回生成してキャッシュ、
   *   テーマ変更時は再描画のみ行う。
   * ================================================================ */
  const thumbCache = {};

  function initThumbs() {
    document.querySelectorAll(".dataset-thumb").forEach((cv) => {
      const name = cv.dataset.thumb;
      if (!thumbCache[name]) {
        const { train } = DataGen.generate(name, 110, 0.05);
        thumbCache[name] = train;
      }
    });
    drawThumbs();
  }

  function drawThumbs() {
    const theme = window.AppTheme;
    const size = 22;
    const pad = 2;
    const dpr = window.devicePixelRatio || 1;

    document.querySelectorAll(".dataset-thumb").forEach((cv) => {
      const pts = thumbCache[cv.dataset.thumb];
      if (!pts) return;

      cv.width = size * dpr;
      cv.height = size * dpr;
      const c = cv.getContext("2d");
      c.scale(dpr, dpr);
      c.clearRect(0, 0, size, size);

      const span = size - pad * 2;
      for (const p of pts) {
        const x = pad + ((p.x + COORD_RANGE) / (COORD_RANGE * 2)) * span;
        const y = pad + ((COORD_RANGE - p.y) / (COORD_RANGE * 2)) * span;
        c.beginPath();
        c.arc(x, y, 1, 0, Math.PI * 2);
        c.fillStyle = p.label === 1 ? theme.c1.hex : theme.c0.hex;
        c.fill();
      }
    });
  }

  /**
   * 外部 (Tweaksパネル・テーマ変更) からの再描画要求。
   */
  function invalidate() {
    state.dirty = true;
    drawThumbs();
  }

  /* ================================================================
   * 7. イベントハンドラ
   * ================================================================ */

  /** 再生 / 一時停止 トグル */
  btnPlayPause.addEventListener("click", () => {
    state.isPlaying = !state.isPlaying;
    updatePlayPauseBtn();
    // 一時停止直後に静止フレーム (フローなし) を1枚描き直す
    state.dirty = true;
  });

  /** 1ステップ: 一時停止中に1エポックだけ進む */
  btnStep.addEventListener("click", () => {
    // 再生中でも押せるが、一時停止状態に切り替えてから1ステップ
    state.isPlaying = false;
    updatePlayPauseBtn();

    const loss = trainEpoch();
    state.epoch++;
    state.trainLoss = loss;
    state.testLoss  = state.net.evaluate(state.testData);

    // 損失グラフ履歴に追加
    LossView.push(state.epoch, loss, state.testLoss);

    if (!checkDivergence(loss)) {
      updateStatsUI();
      render();
    }
  });

  /** リセット: ネットワーク・エポック・損失を初期化 */
  btnReset.addEventListener("click", () => {
    state.isPlaying = false;
    updatePlayPauseBtn();
    resetNetwork();
    initData(); // データも再生成してシャッフル
    render();
  });

  /**
   * 設定変更時の共通処理: データ再生成 + ネットワーク再初期化。
   * 再生中なら再生したまま新しい設定で学習が続く (TF Playground と同じ挙動)。
   */
  function applySettingsChange() {
    initData();
    resetNetwork();
    render();
  }

  /** 隠れ層の数: [−] ボタン */
  btnLayerMinus.addEventListener("click", () => {
    if (state.hidden.length > 0) {
      state.hidden.pop();
      updateNetworkControls();
      applySettingsChange();
    }
  });

  /** 隠れ層の数: [+] ボタン */
  btnLayerPlus.addEventListener("click", () => {
    if (state.hidden.length < 4) {
      state.hidden.push(4);
      updateNetworkControls();
      applySettingsChange();
    }
  });

  /** 活性化関数セグメントボタン (イベントデリゲーション) */
  activationBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const fn = btn.dataset.activation;
      if (fn === state.activation) return;
      state.activation = fn;
      activationBtns.forEach((b) => {
        b.classList.toggle("is-active", b === btn);
      });
      applySettingsChange();
    });
  });

  /** 学習率スライダー (対数スケール) — リセットしない */
  lrSlider.addEventListener("input", () => {
    state.learningRate = Math.pow(10, parseFloat(lrSlider.value));
    lrValue.textContent = formatLR(state.learningRate);
  });

  /** データセット選択ボタン (イベントデリゲーション) */
  const datasetGrid = document.getElementById("dataset-grid");
  datasetGrid.addEventListener("click", (e) => {
    const btn = e.target.closest(".dataset-btn");
    if (!btn || btn.dataset.dataset === state.dataset) return;

    state.dataset = btn.dataset.dataset;

    // 選択状態の見た目を更新
    datasetGrid.querySelectorAll(".dataset-btn").forEach((b) => {
      b.classList.toggle("is-active", b === btn);
    });

    applySettingsChange();
  });

  /** ノイズスライダー */
  const noiseSlider = document.getElementById("noise-slider");
  const noiseValue  = document.getElementById("noise-value");
  noiseSlider.addEventListener("input", () => {
    state.noise = parseFloat(noiseSlider.value);
    noiseValue.textContent = state.noise.toFixed(2);
    applySettingsChange();
  });

  /** ウィンドウリサイズ時に canvas 解像度を再設定 */
  window.addEventListener("resize", () => {
    resizeCanvas();
    resizeNetCanvas();
    resizeLossCanvas();
    state.dirty = true;
    render();
  });

  /* ================================================================
   * 8. メインループ (requestAnimationFrame)
   * ================================================================ */

  /**
   * アニメーションフレームごとに呼ばれるループ関数。
   * 再生中なら EPOCHS_PER_FRAME エポック学習 → 描画 → 統計更新。
   */
  function loop() {
    if (state.isPlaying) {
      trainMultipleEpochs(EPOCHS_PER_FRAME);
      updateStatsUI();
      render();
    } else if (state.dirty) {
      // 一時停止中は変更があったフレームだけ描画 (アイドルCPUを使わない)
      render();
    }

    // 次フレームを予約
    state.rafId = requestAnimationFrame(loop);
  }

  /* ================================================================
   * 9. 起動
   * ================================================================ */

  /**
   * アプリ起動処理。
   * DOM 読み込み完了後に呼ぶ (script は body 末尾なので即時可)。
   */
  function init() {
    // ネットワーク構造コントロールの初期化
    updateNetworkControls();

    // データセットサムネイルの描画
    initThumbs();

    // 学習率スライダーの初期値を設定 (log10(0.3) ≈ -0.523)
    lrSlider.value = Math.log10(state.learningRate).toString();
    lrValue.textContent = formatLR(state.learningRate);

    // データ生成
    initData();

    // ネットワーク初期化
    resetNetwork();

    // Canvas サイズを設定 (getBoundingClientRect が有効な状態で)
    resizeCanvas();
    resizeNetCanvas();
    resizeLossCanvas();

    // 初期描画
    render();

    // メインループ開始
    state.rafId = requestAnimationFrame(loop);
  }

  // DOMContentLoaded を待たずに呼べるが、念のため遅延チェック
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }

  // 外部から参照できるように公開 (Tweaks・デバッグ用)
  return { state, resetNetwork, render, invalidate };

})();
