/* =====================================================================
 * netview.js — ネットワーク図のリアルタイム可視化モジュール
 *
 * 「ニューロン=円」「重み=線」。各ニューロンの円の中には、その
 * ニューロンが入力空間 [-6,6]² をどう見ているかのミニヒートマップを
 * 描く (net.forward を1回呼ぶと全ニューロンの活性化が手に入る)。
 *
 * デザイン:
 *   - 重み線は2パス描画。下層=常時表示の実線(控えめ)、
 *     上層=学習中のみ流れるダッシュ(信号が前方へ伝播する表現)。
 *   - 色は window.AppTheme から毎フレーム取得 (Tweaksで差し替え可)。
 *   - prefers-reduced-motion 環境ではダッシュを流さない。
 *
 * 公開 API:
 *   NetView.render(ctx, net, cssW, cssH, opts)
 *     opts.time   — 秒 (アニメーション位相)
 *     opts.active — 学習中フラグ (ダッシュ表示)
 *     opts.force  — ヒートマップを必ず再計算 (一時停止中の単発描画用)
 * ===================================================================== */

const NetView = (() => {
  "use strict";

  /* ----------------------------------------------------------------
   * 定数
   * ---------------------------------------------------------------- */

  /** 座標空間の範囲 [-COORD_RANGE, COORD_RANGE] (main.js と同じ) */
  const COORD_RANGE = 6;

  /** ミニヒートマップの解像度 */
  const HEAT_GRID = 15;

  /** 学習中のヒートマップ再計算間隔 (Nフレームに1回) */
  const HEAT_INTERVAL = 3;

  /** 円の半径の上限・下限 (px, CSS空間) */
  const RADIUS_MAX = 21;
  const RADIUS_MIN = 14;

  /** ラベル色 */
  const LABEL_COLOR = "rgba(255, 255, 255, 0.32)";

  const LABEL_FONT =
    "500 9px 'IBM Plex Mono', 'IBM Plex Sans JP', system-ui, sans-serif";

  /** ダッシュフローの速度 (px/秒) */
  const FLOW_SPEED = 26;

  /** OSの「視差効果を減らす」設定 */
  const reducedMotion =
    window.matchMedia &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ----------------------------------------------------------------
   * モジュール内状態
   * ---------------------------------------------------------------- */
  const stateView = {
    sizesKey: null,        // 構成キー "2,4,4,1" — 変化したら作り直す
    cells: null,           // [layer][index] => オフスクリーンcanvas
    cellCtxs: null,
    frameCount: 0,
    heatStale: true,       // 強制再計算フラグ
  };

  /* ----------------------------------------------------------------
   * レイアウト計算
   * ---------------------------------------------------------------- */
  function computeLayout(sizes, cssW, cssH) {
    const numLayers = sizes.length;
    const maxNeurons = Math.max(...sizes);

    const padTop = 36;
    const padBottom = 14;
    const usableH = Math.max(20, cssH - padTop - padBottom);

    let radius = usableH / (maxNeurons * 3.2);
    radius = Math.max(RADIUS_MIN, Math.min(RADIUS_MAX, radius));

    const padLeft = radius + 28;
    const padRight = radius + 12;
    const innerW = Math.max(1, cssW - padLeft - padRight);
    const colXs = [];
    if (numLayers === 1) {
      colXs.push(padLeft + innerW / 2);
    } else {
      for (let l = 0; l < numLayers; l++) {
        colXs.push(padLeft + (innerW * l) / (numLayers - 1));
      }
    }

    const positions = [];
    const centerY = padTop + usableH / 2;
    for (let l = 0; l < numLayers; l++) {
      const n = sizes[l];
      const layerPos = [];
      const pitch = maxNeurons > 1 ? usableH / maxNeurons : 0;
      const spanTop = centerY - (pitch * (n - 1)) / 2;
      for (let j = 0; j < n; j++) {
        layerPos.push({ x: colXs[l], y: spanTop + pitch * j });
      }
      positions.push(layerPos);
    }

    return { positions, radius, colXs };
  }

  /* ----------------------------------------------------------------
   * オフスクリーン canvas の準備
   * ---------------------------------------------------------------- */
  function ensureCells(sizes) {
    const key = sizes.join(",");
    if (stateView.sizesKey === key && stateView.cells) return;

    stateView.sizesKey = key;
    stateView.cells = [];
    stateView.cellCtxs = [];
    stateView.heatStale = true;

    for (let l = 0; l < sizes.length; l++) {
      const layerCells = [];
      const layerCtxs = [];
      for (let j = 0; j < sizes[l]; j++) {
        const cv = document.createElement("canvas");
        cv.width = HEAT_GRID;
        cv.height = HEAT_GRID;
        layerCells.push(cv);
        layerCtxs.push(cv.getContext("2d"));
      }
      stateView.cells.push(layerCells);
      stateView.cellCtxs.push(layerCtxs);
    }
  }

  /* ----------------------------------------------------------------
   * ミニヒートマップの再計算
   *   v<0 → クラス0色 / v>0 → クラス1色、|v| を濃さに地色とブレンド。
   * ---------------------------------------------------------------- */
  function recomputeHeatmaps(net) {
    const theme = window.AppTheme;
    const HEAT_POS = theme.c1.rgb;
    const HEAT_NEG = theme.c0.rgb;
    const HEAT_BG = theme.bgPlot;

    const sizes = net.sizes;
    const numLayers = sizes.length;
    const G = HEAT_GRID;

    const buffers = [];
    for (let l = 0; l < numLayers; l++) {
      const layerBufs = [];
      for (let j = 0; j < sizes[l]; j++) {
        layerBufs.push(stateView.cellCtxs[l][j].createImageData(G, G));
      }
      buffers.push(layerBufs);
    }

    for (let row = 0; row < G; row++) {
      const dataY = COORD_RANGE - ((row + 0.5) / G) * (COORD_RANGE * 2);
      for (let col = 0; col < G; col++) {
        const dataX = ((col + 0.5) / G) * (COORD_RANGE * 2) - COORD_RANGE;

        const nx = dataX / COORD_RANGE;
        const ny = dataY / COORD_RANGE;
        const { as } = net.forward([nx, ny]);

        const pixIdx = (row * G + col) * 4;

        for (let l = 0; l < numLayers; l++) {
          const isInput = l === 0;
          const isOutput = l === numLayers - 1;
          const layerAct = as[l];
          for (let j = 0; j < sizes[l]; j++) {
            let v;
            if (isInput) {
              v = j === 0 ? nx : ny;
            } else if (isOutput) {
              v = 2 * layerAct[j] - 1;
            } else {
              v = Math.tanh(layerAct[j]);
            }
            if (v > 1) v = 1; else if (v < -1) v = -1;

            const color = v >= 0 ? HEAT_POS : HEAT_NEG;
            // 弱い活性も読めるよう、濃さはやや持ち上げる (γ=0.8)
            const mag = Math.pow(Math.abs(v), 0.8);
            const r = Math.round(HEAT_BG[0] * (1 - mag) + color[0] * mag);
            const g = Math.round(HEAT_BG[1] * (1 - mag) + color[1] * mag);
            const b = Math.round(HEAT_BG[2] * (1 - mag) + color[2] * mag);

            const data = buffers[l][j].data;
            data[pixIdx] = r;
            data[pixIdx + 1] = g;
            data[pixIdx + 2] = b;
            data[pixIdx + 3] = 255;
          }
        }
      }
    }

    for (let l = 0; l < numLayers; l++) {
      for (let j = 0; j < sizes[l]; j++) {
        stateView.cellCtxs[l][j].putImageData(buffers[l][j], 0, 0);
      }
    }
  }

  /* ----------------------------------------------------------------
   * 重み線 (2パス)
   *   パス1: 実線。太さ・透明度は |w| に応じて控えめに。
   *   パス2: 学習中のみ、前方 (入力→出力) へ流れるダッシュを重ねる。
   * ---------------------------------------------------------------- */
  function drawWeights(ctx, net, layout, opts) {
    const theme = window.AppTheme;
    const { positions, radius } = layout;
    const weights = net.weights; // weights[l][j][i]

    const showFlow =
      opts.active && opts.flow && !reducedMotion;
    const t = opts.time || 0;

    for (let l = 0; l < weights.length; l++) {
      const W = weights[l];
      const fromPos = positions[l];
      const toPos = positions[l + 1];

      for (let j = 0; j < W.length; j++) {
        const Wj = W[j];
        const to = toPos[j];
        for (let i = 0; i < Wj.length; i++) {
          const w = Wj[i];
          const from = fromPos[i];
          const aw = Math.abs(w);
          const sat = Math.min(1, aw); // 0〜1 に飽和

          const rgb = w >= 0 ? theme.c1.rgb : theme.c0.rgb;

          const x1 = from.x + radius;
          const y1 = from.y;
          const x2 = to.x - radius;
          const y2 = to.y;
          const midX = (x1 + x2) / 2;

          // --- パス1: 実線 ---
          const width = 0.6 + sat * 2.6;
          const alpha = 0.10 + sat * 0.5;

          ctx.beginPath();
          ctx.moveTo(x1, y1);
          ctx.bezierCurveTo(midX, y1, midX, y2, x2, y2);
          ctx.strokeStyle =
            "rgba(" + rgb[0] + "," + rgb[1] + "," + rgb[2] + "," + alpha + ")";
          ctx.lineWidth = width;
          ctx.stroke();

          // --- パス2: 流れるダッシュ (学習中のみ) ---
          // 弱い重みの線まで光らせると煩いので |w| > 0.15 のみ
          if (showFlow && aw > 0.15) {
            const phase = (l * 17 + j * 7 + i * 13) % 19; // 線ごとに位相をずらす
            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.bezierCurveTo(midX, y1, midX, y2, x2, y2);
            ctx.setLineDash([3, 17]);
            ctx.lineDashOffset = -(t * FLOW_SPEED + phase);
            ctx.strokeStyle =
              "rgba(" + rgb[0] + "," + rgb[1] + "," + rgb[2] + "," +
              (0.35 + sat * 0.55) + ")";
            ctx.lineWidth = width + 0.4;
            ctx.stroke();
            ctx.setLineDash([]);
          }
        }
      }
    }
  }

  /* ----------------------------------------------------------------
   * ニューロン円 (ミニヒートマップを円形クリップで貼る)
   * ---------------------------------------------------------------- */
  function drawNeurons(ctx, net, layout) {
    const { positions, radius } = layout;
    const sizes = net.sizes;

    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";

    for (let l = 0; l < sizes.length; l++) {
      for (let j = 0; j < sizes[l]; j++) {
        const { x, y } = positions[l][j];
        const cell = stateView.cells[l][j];

        ctx.save();
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.clip();
        ctx.drawImage(cell, x - radius, y - radius, radius * 2, radius * 2);
        ctx.restore();

        // ヘアラインの縁取り
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.lineWidth = 1;
        ctx.strokeStyle = "rgba(255, 255, 255, 0.22)";
        ctx.stroke();
      }
    }
  }

  /* ----------------------------------------------------------------
   * ラベル (層名 / 入力 x y)
   * ---------------------------------------------------------------- */
  function drawLabels(ctx, net, layout) {
    const { positions, radius, colXs } = layout;
    const sizes = net.sizes;
    const numLayers = sizes.length;

    ctx.fillStyle = LABEL_COLOR;
    ctx.font = LABEL_FONT;
    ctx.textAlign = "center";
    ctx.textBaseline = "alphabetic";

    let hiddenCount = 0;
    for (let l = 0; l < numLayers; l++) {
      let label;
      if (l === 0) {
        label = "入力";
      } else if (l === numLayers - 1) {
        label = "出力";
      } else {
        hiddenCount++;
        label = "隠れ層 " + hiddenCount;
      }
      ctx.fillText(label, colXs[l], 16);
    }

    // 入力ニューロンの x / y ラベル
    if (sizes[0] >= 1) {
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      ctx.fillStyle = "rgba(255, 255, 255, 0.45)";
      ctx.font = "400 11px 'IBM Plex Mono', monospace";
      const inputPos = positions[0];
      const labels = ["x", "y"];
      for (let j = 0; j < inputPos.length && j < labels.length; j++) {
        ctx.fillText(labels[j], inputPos[j].x - radius - 8, inputPos[j].y);
      }
    }
  }

  /* ----------------------------------------------------------------
   * 公開 API: render
   * ---------------------------------------------------------------- */
  function render(ctx, net, cssW, cssH, opts) {
    if (!net) return;
    opts = opts || {};

    ctx.clearRect(0, 0, cssW, cssH);

    ensureCells(net.sizes);

    // ヒートマップ再計算:
    //   - 強制フラグ (一時停止中の単発描画・設定変更直後)
    //   - 学習中は HEAT_INTERVAL フレームに1回
    if (opts.force || stateView.heatStale ||
        stateView.frameCount % HEAT_INTERVAL === 0) {
      recomputeHeatmaps(net);
      stateView.heatStale = false;
    }
    stateView.frameCount++;

    const layout = computeLayout(net.sizes, cssW, cssH);

    drawWeights(ctx, net, layout, opts);
    drawNeurons(ctx, net, layout);
    drawLabels(ctx, net, layout);
  }

  return { render };
})();
