// Aortica landing page — animated ECG waveform backgrounds (canvas, no deps).
(function () {
  'use strict';

  var prefersReduced =
    window.matchMedia &&
    window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // Generate one PQRST-ish beat centred at x=0, over [-0.5, 0.5].
  function beat(t) {
    // t in [0, 1) within a single beat cycle.
    var y = 0;
    // P wave
    y += 0.12 * Math.exp(-Math.pow((t - 0.18) / 0.028, 2));
    // Q dip
    y -= 0.09 * Math.exp(-Math.pow((t - 0.30) / 0.012, 2));
    // R spike
    y += 1.0 * Math.exp(-Math.pow((t - 0.33) / 0.010, 2));
    // S dip
    y -= 0.24 * Math.exp(-Math.pow((t - 0.36) / 0.014, 2));
    // T wave
    y += 0.28 * Math.exp(-Math.pow((t - 0.58) / 0.045, 2));
    return y;
  }

  function fitCanvas(canvas) {
    var dpr = window.devicePixelRatio || 1;
    var rect = canvas.getBoundingClientRect();
    canvas.width = Math.max(1, Math.floor(rect.width * dpr));
    canvas.height = Math.max(1, Math.floor(rect.height * dpr));
    var ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx: ctx, w: rect.width, h: rect.height };
  }

  // ── Hero: scrolling ECG line across the whole background ──────────
  function heroECG() {
    var canvas = document.getElementById('ecg-canvas');
    if (!canvas) return;
    var view = fitCanvas(canvas);
    var ctx = view.ctx;
    var phase = 0;
    var beatsAcross = 6; // beats visible across the width

    function draw() {
      var w = view.w, h = view.h;
      ctx.clearRect(0, 0, w, h);
      var mid = h * 0.55;
      var amp = Math.min(h * 0.22, 120);

      ctx.lineWidth = 2;
      ctx.strokeStyle = 'rgba(230, 57, 70, 0.55)';
      ctx.shadowColor = 'rgba(230, 57, 70, 0.5)';
      ctx.shadowBlur = 12;
      ctx.beginPath();
      for (var px = 0; px <= w; px++) {
        var u = (px / w) * beatsAcross + phase;
        var frac = u - Math.floor(u);
        var y = mid - beat(frac) * amp;
        if (px === 0) ctx.moveTo(px, y);
        else ctx.lineTo(px, y);
      }
      ctx.stroke();
      ctx.shadowBlur = 0;
    }

    function loop() {
      phase += 0.0022;
      draw();
      raf = window.requestAnimationFrame(loop);
    }

    var raf;
    draw();
    if (!prefersReduced) loop();

    window.addEventListener('resize', function () {
      view = fitCanvas(canvas);
      ctx = view.ctx;
      draw();
    });
  }

  // ── Demo: static ECG strip + attribution overlay ──────────────────
  function demoECG() {
    var canvas = document.getElementById('demo-canvas');
    if (!canvas) return;
    var view = fitCanvas(canvas);

    function draw() {
      var ctx = view.ctx, w = view.w, h = view.h;
      ctx.clearRect(0, 0, w, h);

      // Grid
      ctx.strokeStyle = 'rgba(230, 57, 70, 0.08)';
      ctx.lineWidth = 1;
      for (var gx = 0; gx <= w; gx += 20) {
        ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, h); ctx.stroke();
      }
      for (var gy = 0; gy <= h; gy += 20) {
        ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(w, gy); ctx.stroke();
      }

      var mid = h * 0.55;
      var amp = Math.min(h * 0.32, 80);
      var beatsAcross = 4;

      // Attribution band under the R waves (illustrative XAI overlay)
      for (var b = 0; b < beatsAcross; b++) {
        var cx = ((b + 0.33) / beatsAcross) * w;
        var grad = ctx.createRadialGradient(cx, mid, 2, cx, mid, 46);
        grad.addColorStop(0, 'rgba(52, 152, 219, 0.35)');
        grad.addColorStop(1, 'rgba(52, 152, 219, 0)');
        ctx.fillStyle = grad;
        ctx.fillRect(cx - 46, 0, 92, h);
      }

      // Trace
      ctx.lineWidth = 2;
      ctx.strokeStyle = 'rgba(230, 57, 70, 0.9)';
      ctx.beginPath();
      for (var px = 0; px <= w; px++) {
        var u = (px / w) * beatsAcross;
        var frac = u - Math.floor(u);
        var y = mid - beat(frac) * amp;
        if (px === 0) ctx.moveTo(px, y);
        else ctx.lineTo(px, y);
      }
      ctx.stroke();
    }

    draw();
    window.addEventListener('resize', function () {
      view = fitCanvas(canvas);
      draw();
    });
  }

  function init() {
    heroECG();
    demoECG();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
