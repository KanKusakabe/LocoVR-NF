/* Browser/Node port of the trained C3 StepFlow (locovrnf/model.py + zuko NSF).
 * Runs the REAL trained weights: CropCNN -> MLP -> merge -> conditional
 * neural-spline-flow inverse sampling.  Verified against PyTorch to <1e-4 by
 * webdemo/testvecs.json (see verify() / `node stepflow.js`).
 */
(function (root) {
  "use strict";

  function b64f32(s) {
    var bin = (typeof atob === "function") ? atob(s) : Buffer.from(s, "base64").toString("binary");
    var n = bin.length, u = new Uint8Array(n);
    for (var i = 0; i < n; i++) u[i] = bin.charCodeAt(i);
    return new Float32Array(u.buffer);
  }
  function b64u8(s) {
    var bin = (typeof atob === "function") ? atob(s) : Buffer.from(s, "base64").toString("binary");
    var n = bin.length, u = new Uint8Array(n);
    for (var i = 0; i < n; i++) u[i] = bin.charCodeAt(i);
    return u;
  }
  function lin(L) { return { w: b64f32(L.w), b: b64f32(L.b), o: L.shape[0], i: L.shape[1] }; }
  // y = W x + b ; W is [o,i] row-major
  function linF(L, x, relu) {
    var y = new Float32Array(L.o);
    for (var o = 0; o < L.o; o++) {
      var s = L.b[o], base = o * L.i;
      for (var k = 0; k < L.i; k++) s += L.w[base + k] * x[k];
      y[o] = relu && s < 0 ? 0 : s;
    }
    return y;
  }
  // conv2d stride2 pad1 kernel3, in [Ci,H,W] flat, W weight [Co,Ci,3,3]
  function conv(x, Ci, H, Wd, cw, cb, Co, relu) {
    var Ho = Math.floor((H + 2 - 3) / 2) + 1, Wo = Math.floor((Wd + 2 - 3) / 2) + 1;
    var out = new Float32Array(Co * Ho * Wo);
    for (var oc = 0; oc < Co; oc++) {
      var wOc = oc * Ci * 9, outOc = oc * Ho * Wo, bias = cb[oc];
      for (var oh = 0; oh < Ho; oh++) {
        var ih0 = oh * 2 - 1;
        var kh0 = ih0 < 0 ? 1 : 0, kh1 = ih0 + 3 > H ? H - ih0 : 3;
        for (var ow = 0; ow < Wo; ow++) {
          var iw0 = ow * 2 - 1;
          var kw0 = iw0 < 0 ? 1 : 0, kw1 = iw0 + 3 > Wd ? Wd - iw0 : 3;
          var s = bias;
          for (var ic = 0; ic < Ci; ic++) {
            var xic = ic * H, wIc = wOc + ic * 9;
            for (var kh = kh0; kh < kh1; kh++) {
              var xrow = (xic + ih0 + kh) * Wd + iw0, wrow = wIc + kh * 3;
              for (var kw = kw0; kw < kw1; kw++) s += cw[wrow + kw] * x[xrow + kw];
            }
          }
          out[outOc + oh * Wo + ow] = relu && s < 0 ? 0 : s;
        }
      }
    }
    return { d: out, H: Ho, W: Wo };
  }
  function softmax(a) {
    var m = -Infinity, i; for (i = 0; i < a.length; i++) if (a[i] > m) m = a[i];
    var s = 0, e = new Float64Array(a.length);
    for (i = 0; i < a.length; i++) { e[i] = Math.exp(a[i] - m); s += e[i]; }
    for (i = 0; i < a.length; i++) e[i] /= s; return e;
  }

  function StepFlow(json) {
    this.cfg = json.config;
    this.conv = json.cnn.conv.map(function (c) {
      return { w: b64f32(c.w), b: b64f32(c.b), Co: c.shape[0], Ci: c.shape[1] };
    });
    this.cnnLin = lin(json.cnn.lin);
    this.mlp = json.mlp.map(lin);
    this.merge = lin(json.merge);
    this.tf = json.transforms.map(function (t) { return t.map(lin); });
  }

  StepFlow.prototype.cnn = function (crop48) {           // crop48: Float32Array(48*48) in {0,1}
    var a = conv(crop48, 1, 48, 48, this.conv[0].w, this.conv[0].b, this.conv[0].Co, true);
    var b = conv(a.d, this.conv[1].Ci, a.H, a.W, this.conv[1].w, this.conv[1].b, this.conv[1].Co, true);
    var c = conv(b.d, this.conv[2].Ci, b.H, b.W, this.conv[2].w, this.conv[2].b, this.conv[2].Co, true);
    return linF(this.cnnLin, c.d, true);                 // 48
  };
  StepFlow.prototype.context = function (crop48, extra4) {
    var cf = this.cnn(crop48);
    var m = linF(this.mlp[0], extra4, true); m = linF(this.mlp[1], m, true);
    var cat = new Float32Array(cf.length + m.length);
    cat.set(cf, 0); cat.set(m, cf.length);
    return linF(this.merge, cat, true);                  // 64
  };
  StepFlow.prototype.hyper = function (t, input66) {      // masked-linear MLP
    var h = linF(t[0], input66, true); h = linF(t[1], h, true); return linF(t[2], h, false); // 46
  };

  // monotonic rational-quadratic spline inverse (zuko MonotonicRQSTransform._inverse)
  StepFlow.prototype.rqsInv = function (y, p, off) {
    var bins = this.cfg.bins, B = this.cfg.bound, ls = Math.log(this.cfg.slope);
    var w = new Float64Array(bins), h = new Float64Array(bins), d = new Float64Array(bins - 1), i;
    for (i = 0; i < bins; i++) { var v = p[off + i]; w[i] = v / (1 + Math.abs(2 * v / ls)); }
    for (i = 0; i < bins; i++) { var v2 = p[off + bins + i]; h[i] = v2 / (1 + Math.abs(2 * v2 / ls)); }
    for (i = 0; i < bins - 1; i++) { var v3 = p[off + 2 * bins + i]; d[i] = v3 / (1 + Math.abs(v3 / ls)); }
    var wsm = softmax(w), hsm = softmax(h);
    // padded cumulative knots (length bins+1)
    var hor = new Float64Array(bins + 1), ver = new Float64Array(bins + 1), der = new Float64Array(bins + 1);
    var cw = 0, ch = 0;
    hor[0] = B * (2 * 0 - 1); ver[0] = B * (2 * 0 - 1); der[0] = Math.exp(0);
    for (i = 0; i < bins; i++) {
      cw += wsm[i]; ch += hsm[i];
      hor[i + 1] = B * (2 * cw - 1); ver[i + 1] = B * (2 * ch - 1);
      der[i + 1] = Math.exp(i < bins - 1 ? d[i] : 0);    // derivatives padded (1,1) with 0 then exp
    }
    // searchsorted(vertical, y) - 1
    var k = 0; for (i = 0; i < bins + 1; i++) if (ver[i] < y) k++; k -= 1;
    if (k < 0 || k >= bins) return y;                    // outside domain -> identity
    var x0 = hor[k], x1 = hor[k + 1], y0 = ver[k], y1 = ver[k + 1], d0 = der[k], d1 = der[k + 1];
    var s = (y1 - y0) / (x1 - x0), y_ = y - y0;
    var a = (y1 - y0) * (s - d0) + y_ * (d0 + d1 - 2 * s);
    var b = (y1 - y0) * d0 - y_ * (d0 + d1 - 2 * s);
    var c = -s * y_;
    var z = 2 * c / (-b - Math.sqrt(b * b - 4 * a * c));
    return x0 + z * (x1 - x0);
  };

  // one AutoregressiveTransform inverse (passes iterations)
  StepFlow.prototype.autoInv = function (t, y, ctx) {
    var x = new Float32Array(2), P = this.cfg.passes, per = 23;
    var input = new Float32Array(2 + ctx.length);
    input.set(ctx, 2);
    for (var pass = 0; pass < P; pass++) {
      input[0] = x[0]; input[1] = x[1];
      var phi = this.hyper(t, input);
      for (var j = 0; j < 2; j++) x[j] = this.rqsInv(y[j], phi, j * per);
    }
    return x;
  };
  // full flow inverse: composed transforms applied in reverse
  StepFlow.prototype.flowInv = function (z, ctx) {
    var x = new Float32Array(z);
    for (var i = this.tf.length - 1; i >= 0; i--) x = this.autoInv(this.tf[i], x, ctx);
    return x;
  };
  // sample a step (ego lat,fwd in scaled units) given crop, extra and base noise z
  StepFlow.prototype.sampleStep = function (crop48, extra4, z) {
    return this.flowInv(z, this.context(crop48, extra4));
  };

  StepFlow.b64u8 = b64u8;
  root.StepFlow = StepFlow;

  // rational-quadratic spline knots from 23 unconstrained params (shared)
  function rqsKnots(p, off, bins, B, slope) {
    var ls = Math.log(slope), i;
    var w = new Float64Array(bins), h = new Float64Array(bins), d = new Float64Array(bins - 1);
    for (i = 0; i < bins; i++) { var v = p[off + i]; w[i] = v / (1 + Math.abs(2 * v / ls)); }
    for (i = 0; i < bins; i++) { var v2 = p[off + bins + i]; h[i] = v2 / (1 + Math.abs(2 * v2 / ls)); }
    for (i = 0; i < bins - 1; i++) { var v3 = p[off + 2 * bins + i]; d[i] = v3 / (1 + Math.abs(v3 / ls)); }
    var wsm = softmax(w), hsm = softmax(h);
    var hor = new Float64Array(bins + 1), ver = new Float64Array(bins + 1), der = new Float64Array(bins + 1);
    var cw = 0, ch = 0; hor[0] = -B; ver[0] = -B; der[0] = 1;
    for (i = 0; i < bins; i++) { cw += wsm[i]; ch += hsm[i]; hor[i + 1] = B * (2 * cw - 1); ver[i + 1] = B * (2 * ch - 1); der[i + 1] = Math.exp(i < bins - 1 ? d[i] : 0); }
    return { hor: hor, ver: ver, der: der };
  }

  // Affordance flow (C1): log p(visit offset | occupancy crop). Uses the RQS
  // FORWARD (data->latent) + jacobian for exact log-density. Same CropCNN/NSF
  // skeleton as StepFlow; context comes straight from the CNN (no goal MLP).
  function AffordanceFlow(json) {
    this.cfg = json.config;
    this.conv = json.cnn.conv.map(function (c) { return { w: b64f32(c.w), b: b64f32(c.b), Co: c.shape[0], Ci: c.shape[1] }; });
    this.cnnLin = lin(json.cnn.lin);
    this.tf = json.transforms.map(function (t) { return t.map(lin); });
  }
  AffordanceFlow.prototype.context = function (crop48) {
    var a = conv(crop48, 1, 48, 48, this.conv[0].w, this.conv[0].b, this.conv[0].Co, true);
    var b = conv(a.d, this.conv[1].Ci, a.H, a.W, this.conv[1].w, this.conv[1].b, this.conv[1].Co, true);
    var c = conv(b.d, this.conv[2].Ci, b.H, b.W, this.conv[2].w, this.conv[2].b, this.conv[2].Co, true);
    return linF(this.cnnLin, c.d, true);
  };
  AffordanceFlow.prototype.hyper = function (t, input) {
    var h = linF(t[0], input, true); h = linF(t[1], h, true); return linF(t[2], h, false);
  };
  AffordanceFlow.prototype.rqsFwd = function (x, p, off) {   // returns [y, log|dy/dx|]
    var bins = this.cfg.bins, B = this.cfg.bound, K = rqsKnots(p, off, bins, B, this.cfg.slope);
    var k = 0, i; for (i = 0; i < bins + 1; i++) if (K.hor[i] < x) k++; k -= 1;
    if (k < 0 || k >= bins) return [x, 0];
    var x0 = K.hor[k], x1 = K.hor[k + 1], y0 = K.ver[k], y1 = K.ver[k + 1], d0 = K.der[k], d1 = K.der[k + 1];
    var s = (y1 - y0) / (x1 - x0), z = (x - x0) / (x1 - x0), denom = s + (d0 + d1 - 2 * s) * z * (1 - z);
    var y = y0 + (y1 - y0) * (s * z * z + d0 * z * (1 - z)) / denom;
    var jac = s * s * (2 * s * z * (1 - z) + d0 * (1 - z) * (1 - z) + d1 * z * z) / (denom * denom);
    return [y, Math.log(jac)];
  };
  AffordanceFlow.prototype.logProb = function (crop48, x) {  // x = [x0,x1]
    var ctx = this.context(crop48), cur = [x[0], x[1]], ladj = 0, per = 23;
    var input = new Float32Array(2 + ctx.length); input.set(ctx, 2);
    for (var i = 0; i < this.tf.length; i++) {
      input[0] = cur[0]; input[1] = cur[1];
      var phi = this.hyper(this.tf[i], input), nx = [0, 0];
      for (var j = 0; j < 2; j++) { var r = this.rqsFwd(cur[j], phi, j * per); nx[j] = r[0]; ladj += r[1]; }
      cur = nx;
    }
    var LOG2PI = Math.log(2 * Math.PI);
    return -0.5 * (cur[0] * cur[0] + cur[1] * cur[1] + 2 * LOG2PI) + ladj;
  };
  root.AffordanceFlow = AffordanceFlow;

  // ---- node self-verification against PyTorch test vectors ----
  if (typeof module !== "undefined" && module.exports) {
    module.exports = { StepFlow: StepFlow, AffordanceFlow: AffordanceFlow };
    module.exports.StepFlow = StepFlow;
    if (require.main === module) {
      var fs = require("fs"), path = require("path"), dir = __dirname;
      var w = JSON.parse(fs.readFileSync(path.join(dir, "stepflow.json")));
      var tv = JSON.parse(fs.readFileSync(path.join(dir, "testvecs.json")));
      var sf = new StepFlow(w), maxErr = 0, n = 0;
      tv.forEach(function (v) {
        var u = b64u8(v.crop), crop = new Float32Array(u.length);
        for (var i = 0; i < u.length; i++) crop[i] = u[i];
        var step = sf.sampleStep(crop, Float32Array.from(v.extra), Float32Array.from(v.z));
        var e = Math.max(Math.abs(step[0] - v.step[0]), Math.abs(step[1] - v.step[1]));
        maxErr = Math.max(maxErr, e); n++;
      });
      console.log("StepFlow: verified " + n + " vectors vs PyTorch | max abs error = " + maxErr.toExponential(3));
      var ok = maxErr < 1e-3;
      if (fs.existsSync(path.join(dir, "affordance.json"))) {
        var af = new AffordanceFlow(JSON.parse(fs.readFileSync(path.join(dir, "affordance.json"))));
        var atv = JSON.parse(fs.readFileSync(path.join(dir, "aff_testvecs.json"))), aerr = 0;
        atv.forEach(function (v) {
          var u = b64u8(v.crop), crop = new Float32Array(u.length);
          for (var i = 0; i < u.length; i++) crop[i] = u[i];
          aerr = Math.max(aerr, Math.abs(af.logProb(crop, v.x) - v.logp));
        });
        console.log("AffordanceFlow: verified " + atv.length + " logp vs PyTorch | max abs error = " + aerr.toExponential(3));
        ok = ok && aerr < 1e-3;
      }
      process.exit(ok ? 0 : 1);
    }
  }
})(typeof window !== "undefined" ? window : globalThis);
