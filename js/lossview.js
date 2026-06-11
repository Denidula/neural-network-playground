/* =====================================================================
 * lossview.js — 損失グラフ描画モジュール
 *
 * デザイン:
 *   - 学習損失 = 白の実線 / テスト損失 = グレーの破線 (色相を使わない)
 *   - ラベル・数値はすべて等幅 (IBM Plex Mono)
 *   - 右端に現在値を数値で添える (計器の読み出し)
 *
 * 公開 API:
 *   LossView.push(epoch, trainLoss, testLoss)
 *   LossView.clear()
 *   LossView.render(ctx, cssW, cssH)
 * ===================================================================== */

const LossView = (() => {
  "use strict";

  /* ----------------------------------------------------------------
   * 定数
   * ---------------------------------------------------------------- */
  const MAX_POINTS = 400;

  const PAD_LEFT   = 40;  // y軸ラベル用
  const PAD_BOTTOM = 20;  // x軸ラベル用
  const PAD_TOP    = 8;
  const PAD_RIGHT  = 58;  // 現在値ラベル用

  const GRID_LINES = 4;

  const LABEL_FONT = "400 9px 'IBM Plex Mono', monospace";
  const VALUE_FONT = "500 10px 'IBM Plex Mono', monospace";

  const COLOR_TRAIN = "#E7E9EC";
  const COLOR_TEST  = "#84878E";
  const COLOR_GRID  = "rgba(255, 255, 255, 0.05)";
  const COLOR_AXIS  = "rgba(255, 255, 255, 0.14)";
  const COLOR_LABEL = "rgba(255, 255, 255, 0.32)";

  /* ----------------------------------------------------------------
   * 履歴データ
   * ---------------------------------------------------------------- */
  const epochs      = [];
  const trainLosses = [];
  const testLosses  = [];

  /* ----------------------------------------------------------------
   * 公開 API
   * ---------------------------------------------------------------- */
  function push(epoch, trainLoss, testLoss) {
    epochs.push(epoch);
    trainLosses.push(trainLoss);
    testLosses.push(testLoss);
  }

  function clear() {
    epochs.length      = 0;
    trainLosses.length = 0;
    testLosses.length  = 0;
  }

  function render(ctx, cssW, cssH) {
    ctx.clearRect(0, 0, cssW, cssH);

    const n = epochs.length;

    // --- 空状態 ---
    if (n < 2) {
      ctx.save();
      ctx.font = LABEL_FONT;
      ctx.fillStyle = COLOR_LABEL;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("グラフは学習開始後に表示されます", cssW / 2, cssH / 2);
      ctx.restore();
      return;
    }

    /* ---- スケール ---- */
    let maxLoss = 0.8;
    for (let i = 0; i < n; i++) {
      if (trainLosses[i] > maxLoss) maxLoss = trainLosses[i];
      if (testLosses[i]  > maxLoss) maxLoss = testLosses[i];
    }
    maxLoss *= 1.05;
    if (maxLoss < 0.8) maxLoss = 0.8;

    const plotX = PAD_LEFT;
    const plotY = PAD_TOP;
    const plotW = cssW - PAD_LEFT - PAD_RIGHT;
    const plotH = cssH - PAD_TOP - PAD_BOTTOM;

    const maxEpoch = epochs[n - 1] || 1;

    function toX(epoch) { return plotX + (epoch / maxEpoch) * plotW; }
    function toY(loss)  { return plotY + plotH - (loss / maxLoss) * plotH; }

    /* ---- グリッド & y軸ラベル ---- */
    ctx.save();
    ctx.font = LABEL_FONT;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.lineWidth = 1;

    for (let k = 0; k <= GRID_LINES; k++) {
      const val = (maxLoss * k) / GRID_LINES;
      const cy = Math.round(toY(val)) + 0.5;

      ctx.strokeStyle = k === 0 ? COLOR_AXIS : COLOR_GRID;
      ctx.beginPath();
      ctx.moveTo(plotX, cy);
      ctx.lineTo(plotX + plotW, cy);
      ctx.stroke();

      ctx.fillStyle = COLOR_LABEL;
      ctx.fillText(val.toFixed(1), plotX - 8, cy);
    }
    ctx.restore();

    /* ---- x軸ラベル ---- */
    ctx.save();
    ctx.font = LABEL_FONT;
    ctx.fillStyle = COLOR_LABEL;
    ctx.textAlign = "right";
    ctx.textBaseline = "top";
    ctx.fillText("EPOCH " + maxEpoch, plotX + plotW, plotY + plotH + 6);
    ctx.textAlign = "left";
    ctx.fillText("0", plotX, plotY + plotH + 6);
    ctx.restore();

    /* ---- 折れ線 ---- */
    const stride = Math.ceil(n / MAX_POINTS);

    _drawLine(ctx, toX, toY, stride, n, testLosses, COLOR_TEST, 1.2, [4, 4]);
    _drawLine(ctx, toX, toY, stride, n, trainLosses, COLOR_TRAIN, 1.5, null);

    /* ---- 右端: 現在値の読み出し ---- */
    const xEnd = toX(epochs[n - 1]);
    const trainY = toY(trainLosses[n - 1]);
    const testY  = toY(testLosses[n - 1]);

    _drawDot(ctx, xEnd, trainY, COLOR_TRAIN);
    _drawDot(ctx, xEnd, testY, COLOR_TEST);

    // 値ラベルが重ならないよう最小間隔を確保する
    let labelTrainY = trainY;
    let labelTestY  = testY;
    const MIN_GAP = 12;
    if (Math.abs(labelTrainY - labelTestY) < MIN_GAP) {
      const mid = (labelTrainY + labelTestY) / 2;
      if (labelTrainY <= labelTestY) {
        labelTrainY = mid - MIN_GAP / 2;
        labelTestY  = mid + MIN_GAP / 2;
      } else {
        labelTrainY = mid + MIN_GAP / 2;
        labelTestY  = mid - MIN_GAP / 2;
      }
    }

    ctx.save();
    ctx.font = VALUE_FONT;
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillStyle = COLOR_TRAIN;
    ctx.fillText(trainLosses[n - 1].toFixed(3), xEnd + 9, labelTrainY);
    ctx.fillStyle = COLOR_TEST;
    ctx.fillText(testLosses[n - 1].toFixed(3), xEnd + 9, labelTestY);
    ctx.restore();
  }

  /* ----------------------------------------------------------------
   * 内部ヘルパー
   * ---------------------------------------------------------------- */
  function _drawLine(ctx, toX, toY, stride, n, values, color, width, dash) {
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    if (dash) ctx.setLineDash(dash);

    ctx.beginPath();
    let started = false;

    for (let i = 0; i < n; i += stride) {
      const x = toX(epochs[i]);
      const y = toY(values[i]);
      if (!started) {
        ctx.moveTo(x, y);
        started = true;
      } else {
        ctx.lineTo(x, y);
      }
    }

    const last = n - 1;
    if (last % stride !== 0) {
      ctx.lineTo(toX(epochs[last]), toY(values[last]));
    }

    ctx.stroke();
    ctx.restore();
  }

  function _drawDot(ctx, x, y, color) {
    ctx.save();
    ctx.beginPath();
    ctx.arc(x, y, 2.5, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.restore();
  }

  return { push, clear, render };
})();
