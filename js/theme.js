/* =====================================================================
 * theme.js — クラス色テーマとグローバル描画フラグ
 *
 * canvas描画(JS)とCSS(カスタムプロパティ)の両方が同じ色を参照する
 * ための単一の情報源。Tweaksパネルが AppTheme.set() を呼ぶと
 * CSS変数とcanvasの再描画が同期する。
 *
 * 読み込み順: 全モジュールより先 (index.html 参照)
 * ===================================================================== */

window.AppTheme = (() => {
  "use strict";

  function hexToRgb(hex) {
    const h = hex.replace("#", "");
    return [
      parseInt(h.slice(0, 2), 16),
      parseInt(h.slice(2, 4), 16),
      parseInt(h.slice(4, 6), 16),
    ];
  }

  const theme = {
    /** クラス0 (青系) / クラス1 (オレンジ系) */
    c0: { hex: "#7DA2FF", rgb: [125, 162, 255] },
    c1: { hex: "#FFA94D", rgb: [255, 169, 77] },

    /** 地色 (style.css の --bg / --bg-plot と一致させること) */
    bg:     [10, 11, 13],
    bgPlot: [14, 15, 18],

    /**
     * クラス色を差し替える (Tweaksパネルから呼ばれる)。
     * CSS変数を更新し、canvas側へ再描画を要求する。
     */
    set(hex0, hex1) {
      theme.c0 = { hex: hex0, rgb: hexToRgb(hex0) };
      theme.c1 = { hex: hex1, rgb: hexToRgb(hex1) };
      const root = document.documentElement.style;
      root.setProperty("--c0", hex0);
      root.setProperty("--c1", hex1);
      if (window.App && window.App.invalidate) window.App.invalidate();
    },
  };

  return theme;
})();

/** 描画フラグ (Tweaksパネルが書き換える) */
window.AppFlags = {
  /** 学習中、重み線に沿って信号が流れるアニメーション */
  flow: true,
  /** 決定境界の座標グリッド表示 */
  grid: true,
};
