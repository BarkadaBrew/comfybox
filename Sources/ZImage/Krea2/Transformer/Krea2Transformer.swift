// Krea2Transformer.swift — MLX-Swift port of Krea-2 SingleStreamDiT.
//
// Faithful port of the reference MLX impl (docs/krea2-reference/krea2/transformer.py),
// itself a line-by-line port of krea-2-official/mmdit.py. @ModuleInfo keys match the
// `turbo.safetensors` tensor names exactly (verified: 430 keys), so weights load with
// no remapping. RMSNorm effective weight = stored `scale` + 1.0, math in float32.

import Foundation
import MLX
import MLXNN
import MLXFast

// MARK: - Configuration

public struct Krea2Config: Sendable {
  public var features: Int = 6144
  public var tdim: Int = 256
  public var txtdim: Int = 2560
  public var heads: Int = 48
  public var kvheads: Int = 12
  public var multiplier: Int = 4
  public var layers: Int = 28
  public var patch: Int = 2
  public var channels: Int = 16
  public var theta: Float = 1000.0
  public var txtheads: Int = 20
  public var txtkvheads: Int = 20
  public var txtlayers: Int = 12  // number of selected encoder hidden-state layers

  public init() {}

  public var headDim: Int { features / heads }
  /// 3-axis RoPE split [t, h, w], sum == headDim.
  public var axes: [Int] {
    let hd = headDim
    return [hd - 12 * (hd / 16), 6 * (hd / 16), 6 * (hd / 16)]
  }
}

// MARK: - Activation

@inline(__always)
func kreaGeluTanh(_ x: MLXArray) -> MLXArray {
  let c = Float((2.0 / Double.pi).squareRoot())
  return 0.5 * x * (1.0 + MLX.tanh(c * (x + 0.044715 * x * x * x)))
}

/// Parameter-less GELU(tanh) module (occupies a weight-less slot in sequences).
public final class Krea2GELUTanh: Module {
  public override init() { super.init() }
  public func callAsFunction(_ x: MLXArray) -> MLXArray { kreaGeluTanh(x) }
}

// MARK: - Norms

/// Krea2 RMSNorm: effective weight = (scale + 1), float32 math, eps 1e-5.
public final class Krea2RMSNorm: Module {
  @ModuleInfo(key: "scale") var scale: MLXArray
  let eps: Float

  public init(_ features: Int, eps: Float = 1e-5) {
    self.eps = eps
    self._scale.wrappedValue = MLX.zeros([features])
  }

  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    let dt = x.dtype
    var t = x.asType(.float32)
    t = t * MLX.rsqrt(MLX.mean(t * t, axis: -1, keepDims: true) + MLXArray(eps))
    t = t * (1.0 + scale.asType(.float32))
    return t.asType(dt)
  }
}

public final class Krea2QKNorm: Module {
  @ModuleInfo(key: "qnorm") var qnorm: Krea2RMSNorm
  @ModuleInfo(key: "knorm") var knorm: Krea2RMSNorm
  public init(_ dim: Int) {
    self._qnorm.wrappedValue = Krea2RMSNorm(dim)
    self._knorm.wrappedValue = Krea2RMSNorm(dim)
  }
}

// MARK: - SwiGLU

public final class Krea2SwiGLU: Module {
  @ModuleInfo(key: "gate") var gate: Linear
  @ModuleInfo(key: "up") var up: Linear
  @ModuleInfo(key: "down") var down: Linear

  public init(features: Int, multiplier: Int, multiple: Int = 128) {
    var mlpdim = (2 * features / 3) * multiplier
    mlpdim = multiple * ((mlpdim + multiple - 1) / multiple)
    self._gate.wrappedValue = Linear(features, mlpdim, bias: false)
    self._up.wrappedValue = Linear(features, mlpdim, bias: false)
    self._down.wrappedValue = Linear(mlpdim, features, bias: false)
  }

  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    down(silu(gate(x)) * up(x))
  }
}

// MARK: - RoPE (3-axis, interleaved adjacent-pair rotation)

enum Krea2Rope {
  /// pos: (L,3) -> (cos, sin) each (L, sum(axes)/2).
  ///
  /// `scales` applies NTK-aware frequency widening per axis (DyPE). A scale > 1
  /// widens theta so the model's trained frequency range covers a larger token
  /// grid — this is what keeps 2K renders structurally coherent. Axis 0 is the
  /// text/frame axis and is never scaled; widening it would break text-image
  /// alignment. Formula matches ZImageRopeEmbedder.computeNTKFreqTable exactly.
  static func make(
    pos: MLXArray, axes: [Int], theta: Float, scales: [Float] = [1, 1, 1]
  ) -> (MLXArray, MLXArray) {
    var cosParts: [MLXArray] = []
    var sinParts: [MLXArray] = []
    for (i, d) in axes.enumerated() {
      let axisScale = i < scales.count ? scales[i] : 1.0
      let axisTheta = (i == 0 || axisScale <= 1.0)
        ? theta
        : theta * pow(axisScale, Float(d) / Float(d - 2))
      let scale = MLXArray(stride(from: 0, to: d, by: 2).map { Float($0) }) / Float(d)
      let omega = 1.0 / MLX.pow(MLXArray(axisTheta), scale)  // (d/2,)
      let posCol = pos[0..., i ..< (i + 1)]  // (L,1)
      let freqs = posCol * omega.expandedDimensions(axis: 0)  // (L, d/2)
      cosParts.append(MLX.cos(freqs))
      sinParts.append(MLX.sin(freqs))
    }
    return (MLX.concatenated(cosParts, axis: -1), MLX.concatenated(sinParts, axis: -1))
  }

  /// x: (B,H,L,D); cos/sin: (L, D/2). Interleaved rotation.
  static func apply(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
    let b = x.dim(0), h = x.dim(1), l = x.dim(2), d = x.dim(3)
    let xf = x.asType(.float32).reshaped(b, h, l, d / 2, 2)
    let x0 = xf[.ellipsis, 0]  // (b,h,l,d/2)
    let x1 = xf[.ellipsis, 1]
    let c = cos.expandedDimensions(axis: 0).expandedDimensions(axis: 0)  // (1,1,L,D/2)
    let s = sin.expandedDimensions(axis: 0).expandedDimensions(axis: 0)
    let o0 = x0 * c - x1 * s
    let o1 = x0 * s + x1 * c
    let out = MLX.stacked([o0, o1], axis: -1).reshaped(b, h, l, d)
    return out.asType(x.dtype)
  }
}

// MARK: - Attention (GQA, QK-norm, sigmoid-gated output)

public final class Krea2Attention: Module {
  let heads: Int
  let kvheads: Int
  let headdim: Int
  let scale: Float

  @ModuleInfo(key: "wq") var wq: Linear
  @ModuleInfo(key: "wk") var wk: Linear
  @ModuleInfo(key: "wv") var wv: Linear
  @ModuleInfo(key: "gate") var gate: Linear
  @ModuleInfo(key: "qknorm") var qknorm: Krea2QKNorm
  @ModuleInfo(key: "wo") var wo: Linear

  public init(dim: Int, heads: Int, kvheads: Int? = nil) {
    self.heads = heads
    self.kvheads = kvheads ?? heads
    self.headdim = dim / heads
    self.scale = 1.0 / Float(dim / heads).squareRoot()
    self._wq.wrappedValue = Linear(dim, headdim * heads, bias: false)
    self._wk.wrappedValue = Linear(dim, headdim * self.kvheads, bias: false)
    self._wv.wrappedValue = Linear(dim, headdim * self.kvheads, bias: false)
    self._gate.wrappedValue = Linear(dim, dim, bias: false)
    self._qknorm.wrappedValue = Krea2QKNorm(headdim)
    self._wo.wrappedValue = Linear(dim, dim, bias: false)
  }

  public func callAsFunction(
    _ qkv: MLXArray, cos: MLXArray?, sin: MLXArray?, mask: MLXArray?
  ) -> MLXArray {
    let b = qkv.dim(0), l = qkv.dim(1)
    var q = wq(qkv).reshaped(b, l, heads, headdim).transposed(0, 2, 1, 3)
    var k = wk(qkv).reshaped(b, l, kvheads, headdim).transposed(0, 2, 1, 3)
    var v = wv(qkv).reshaped(b, l, kvheads, headdim).transposed(0, 2, 1, 3)
    let g = gate(qkv)

    q = qknorm.qnorm(q)
    k = qknorm.knorm(k)

    if let cos, let sin {
      q = Krea2Rope.apply(q, cos: cos, sin: sin)
      k = Krea2Rope.apply(k, cos: cos, sin: sin)
    }

    if kvheads != heads {
      let rep = heads / kvheads
      k = repeatKV(k, rep)
      v = repeatKV(v, rep)
    }

    var out = MLXFast.scaledDotProductAttention(
      queries: q, keys: k, values: v, scale: scale,
      mask: mask.map { .array($0) } ?? .none)
    out = out.transposed(0, 2, 1, 3).reshaped(b, l, heads * headdim)
    return wo(out * MLX.sigmoid(g))
  }

  private func repeatKV(_ x: MLXArray, _ rep: Int) -> MLXArray {
    guard rep > 1 else { return x }
    let s = x.shape  // (b, kv, L, hd)
    let e = x.expandedDimensions(axis: 2)  // (b, kv, 1, L, hd)
    return MLX.broadcast(e, to: [s[0], s[1], rep, s[2], s[3]])
      .reshaped(s[0], s[1] * rep, s[2], s[3])
  }
}

// MARK: - Modulation

public final class Krea2DoubleSharedModulation: Module {
  @ModuleInfo(key: "lin") var lin: MLXArray  // (6*dim,)
  public init(_ dim: Int) { self._lin.wrappedValue = MLX.zeros([6 * dim]) }
  /// Returns 6 chunks each (…, dim).
  public func callAsFunction(_ vec: MLXArray) -> [MLXArray] {
    MLX.split(vec + lin, parts: 6, axis: -1)
  }
}

public final class Krea2SimpleModulation: Module {
  @ModuleInfo(key: "lin") var lin: MLXArray  // (2, dim)
  public init(_ dim: Int) { self._lin.wrappedValue = MLX.zeros([2, dim]) }
  /// vec: (B,1,dim) -> (scale, shift) each (B,1,dim).
  public func callAsFunction(_ vec: MLXArray) -> (MLXArray, MLXArray) {
    let out = vec + lin.expandedDimensions(axis: 0)  // (B,2,dim)
    let parts = MLX.split(out, parts: 2, axis: 1)
    return (parts[0], parts[1])
  }
}

// MARK: - Blocks

public final class Krea2SingleStreamBlock: Module {
  @ModuleInfo(key: "mod") var mod: Krea2DoubleSharedModulation
  @ModuleInfo(key: "prenorm") var prenorm: Krea2RMSNorm
  @ModuleInfo(key: "postnorm") var postnorm: Krea2RMSNorm
  @ModuleInfo(key: "attn") var attn: Krea2Attention
  @ModuleInfo(key: "mlp") var mlp: Krea2SwiGLU

  public init(features: Int, heads: Int, multiplier: Int, kvheads: Int) {
    self._mod.wrappedValue = Krea2DoubleSharedModulation(features)
    self._prenorm.wrappedValue = Krea2RMSNorm(features)
    self._postnorm.wrappedValue = Krea2RMSNorm(features)
    self._attn.wrappedValue = Krea2Attention(dim: features, heads: heads, kvheads: kvheads)
    self._mlp.wrappedValue = Krea2SwiGLU(features: features, multiplier: multiplier)
  }

  public func callAsFunction(
    _ x: MLXArray, vec: MLXArray, cos: MLXArray, sin: MLXArray, mask: MLXArray?
  ) -> MLXArray {
    let m = mod(vec)
    let prescale = m[0], preshift = m[1], pregate = m[2]
    let postscale = m[3], postshift = m[4], postgate = m[5]
    var out = x + pregate * attn((1 + prescale) * prenorm(x) + preshift, cos: cos, sin: sin, mask: mask)
    out = out + postgate * mlp((1 + postscale) * postnorm(out) + postshift)
    return out
  }
}

public final class Krea2TextFusionBlock: Module {
  @ModuleInfo(key: "prenorm") var prenorm: Krea2RMSNorm
  @ModuleInfo(key: "postnorm") var postnorm: Krea2RMSNorm
  @ModuleInfo(key: "attn") var attn: Krea2Attention
  @ModuleInfo(key: "mlp") var mlp: Krea2SwiGLU

  public init(features: Int, heads: Int, multiplier: Int, kvheads: Int) {
    self._prenorm.wrappedValue = Krea2RMSNorm(features)
    self._postnorm.wrappedValue = Krea2RMSNorm(features)
    self._attn.wrappedValue = Krea2Attention(dim: features, heads: heads, kvheads: kvheads)
    self._mlp.wrappedValue = Krea2SwiGLU(features: features, multiplier: multiplier)
  }

  public func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
    var out = x + attn(prenorm(x), cos: nil, sin: nil, mask: mask)
    out = out + mlp(postnorm(out))
    return out
  }
}

public final class Krea2TextFusionTransformer: Module {
  @ModuleInfo(key: "layerwise_blocks") var layerwiseBlocks: [Krea2TextFusionBlock]
  @ModuleInfo(key: "projector") var projector: Linear
  @ModuleInfo(key: "refiner_blocks") var refinerBlocks: [Krea2TextFusionBlock]

  public init(numTxtLayers: Int, txtDim: Int, heads: Int, multiplier: Int, kvheads: Int) {
    self._layerwiseBlocks.wrappedValue = (0..<2).map { _ in
      Krea2TextFusionBlock(features: txtDim, heads: heads, multiplier: multiplier, kvheads: kvheads)
    }
    self._projector.wrappedValue = Linear(numTxtLayers, 1, bias: false)
    self._refinerBlocks.wrappedValue = (0..<2).map { _ in
      Krea2TextFusionBlock(features: txtDim, heads: heads, multiplier: multiplier, kvheads: kvheads)
    }
  }

  /// x: (B, L, nLayers, D) -> (B, L, D)
  public func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
    let b = x.dim(0), l = x.dim(1), n = x.dim(2), d = x.dim(3)
    var h = x.reshaped(b * l, n, d)
    for block in layerwiseBlocks { h = block(h, mask: nil) }
    // (b*l, n, d) -> (b, l, d, n)
    h = h.reshaped(b, l, n, d).transposed(0, 1, 3, 2)
    h = projector(h)  // (b, l, d, 1)
    h = h[.ellipsis, 0]  // (b, l, d)
    for block in refinerBlocks { h = block(h, mask: mask) }
    return h
  }
}

// MARK: - Mixed sequences (weight-less slots preserve index-based keys)

/// tmlp = [Linear "0", GELU "1", Linear "2"] (checkpoint keys remapped to lin0/lin2
/// in the loader — numeric path segments unflatten as array indices in MLX-Swift).
public final class Krea2TMLP: Module {
  @ModuleInfo(key: "lin0") var l0: Linear
  @ModuleInfo(key: "lin2") var l2: Linear
  public init(tdim: Int, features: Int) {
    self._l0.wrappedValue = Linear(tdim, features)
    self._l2.wrappedValue = Linear(features, features)
  }
  public func callAsFunction(_ x: MLXArray) -> MLXArray { l2(kreaGeluTanh(l0(x))) }
}

/// tproj = [GELU "0", Linear "1"] (checkpoint key remapped to lin1 in the loader).
public final class Krea2TProj: Module {
  @ModuleInfo(key: "lin1") var l1: Linear
  public init(features: Int) { self._l1.wrappedValue = Linear(features, features * 6) }
  public func callAsFunction(_ x: MLXArray) -> MLXArray { l1(kreaGeluTanh(x)) }
}

/// txtmlp = [RMSNorm "0", Linear "1", GELU "2", Linear "3"] (keys remapped in loader).
public final class Krea2TxtMLP: Module {
  @ModuleInfo(key: "norm0") var norm: Krea2RMSNorm
  @ModuleInfo(key: "lin1") var l1: Linear
  @ModuleInfo(key: "lin3") var l3: Linear
  public init(txtDim: Int, features: Int) {
    self._norm.wrappedValue = Krea2RMSNorm(txtDim)
    self._l1.wrappedValue = Linear(txtDim, features)
    self._l3.wrappedValue = Linear(features, features)
  }
  public func callAsFunction(_ x: MLXArray) -> MLXArray { l3(kreaGeluTanh(l1(norm(x)))) }
}

// MARK: - Last layer

public final class Krea2LastLayer: Module {
  @ModuleInfo(key: "norm") var norm: Krea2RMSNorm
  @ModuleInfo(key: "linear") var linear: Linear
  @ModuleInfo(key: "modulation") var modulation: Krea2SimpleModulation

  public init(features: Int, patch: Int, channels: Int) {
    self._norm.wrappedValue = Krea2RMSNorm(features)
    self._linear.wrappedValue = Linear(features, patch * patch * channels, bias: true)
    self._modulation.wrappedValue = Krea2SimpleModulation(features)
  }

  public func callAsFunction(_ x: MLXArray, tvec: MLXArray) -> MLXArray {
    let (scale, shift) = modulation(tvec)
    return linear((1 + scale) * norm(x) + shift)
  }
}

// MARK: - Helpers

enum Krea2Util {
  static func timestepEmbed(_ t: MLXArray, dim: Int, period: Float = 1e4, tfactor: Float = 1e3) -> MLXArray {
    let half = dim / 2
    let freqs = MLX.exp(-Foundation.log(period) * MLXArray((0..<half).map { Float($0) }) / Float(half))
    // (B,) -> (B,1,1) * (half,) -> (B,1,half)
    let args = (t.asType(.float32) * tfactor).expandedDimensions(axis: -1).expandedDimensions(axis: -1) * freqs
    return MLX.concatenated([MLX.cos(args), MLX.sin(args)], axis: -1)  // (B,1,dim)
  }

  /// valid: (B,L) {0,1} -> additive (B,1,L,L) mask, or nil if all valid.
  static func additiveMask(_ valid: MLXArray, dtype: DType) -> MLXArray? {
    if MLX.all(valid .>= MLXArray(Float(0.5))).item(Bool.self) { return nil }
    let m = valid.asType(.float32)
    let full = m.expandedDimensions(axis: 2) * m.expandedDimensions(axis: 1)  // (B,L,L)
    let add = (1.0 - full) * Float(-1e9)
    return add.expandedDimensions(axis: 1).asType(dtype)
  }
}

// MARK: - SingleStreamDiT

public final class Krea2SingleStreamDiT: Module {
  public let cfg: Krea2Config

  @ModuleInfo(key: "first") var first: Linear
  @ModuleInfo(key: "blocks") var blocks: [Krea2SingleStreamBlock]
  @ModuleInfo(key: "tmlp") var tmlp: Krea2TMLP
  @ModuleInfo(key: "txtfusion") var txtfusion: Krea2TextFusionTransformer
  @ModuleInfo(key: "txtmlp") var txtmlp: Krea2TxtMLP
  @ModuleInfo(key: "last") var last: Krea2LastLayer
  @ModuleInfo(key: "tproj") var tproj: Krea2TProj

  // Depth Control-LoRA (docs/FDD-krea2-depth-controlnet.md). Stored as plain
  // MLXArrays (NOT @ModuleInfo) so they are excluded from q8 quantize and from
  // update(parameters:)/eval traversal — set explicitly at control-LoRA load.
  // Expanded input projection: weight (features, 2C), bias (features,).
  public var controlFirstWeight: MLXArray? = nil
  public var controlFirstBias: MLXArray? = nil

  public init(cfg: Krea2Config = Krea2Config()) {
    self.cfg = cfg
    self._first.wrappedValue = Linear(cfg.channels * cfg.patch * cfg.patch, cfg.features, bias: true)
    self._blocks.wrappedValue = (0..<cfg.layers).map { _ in
      Krea2SingleStreamBlock(features: cfg.features, heads: cfg.heads, multiplier: cfg.multiplier, kvheads: cfg.kvheads)
    }
    self._tmlp.wrappedValue = Krea2TMLP(tdim: cfg.tdim, features: cfg.features)
    self._txtfusion.wrappedValue = Krea2TextFusionTransformer(
      numTxtLayers: cfg.txtlayers, txtDim: cfg.txtdim, heads: cfg.txtheads,
      multiplier: cfg.multiplier, kvheads: cfg.txtkvheads)
    self._txtmlp.wrappedValue = Krea2TxtMLP(txtDim: cfg.txtdim, features: cfg.features)
    self._last.wrappedValue = Krea2LastLayer(features: cfg.features, patch: cfg.patch, channels: cfg.channels)
    self._tproj.wrappedValue = Krea2TProj(features: cfg.features)
  }

  /// img: (B, Limg, channels*patch^2); context: (B, seq, nLayers, txtdim);
  /// t: (B,) in [0,1]; pos: (L,3) for [txt; img]; mask: (B,L) validity.
  public func callAsFunction(
    img imgIn: MLXArray, context contextIn: MLXArray, t: MLXArray, pos: MLXArray, mask: MLXArray,
    control: MLXArray? = nil
  ) -> MLXArray {
    // Control ON: project concat([noisy tokens ‖ control tokens]) (B,L,2C) through the
    // expanded input projection. Control OFF: base `first` path is byte-identical to today.
    let img: MLXArray
    if let cw = controlFirstWeight, let cb = controlFirstBias, let ctrl = control {
      let x = MLX.concatenated([imgIn, ctrl], axis: -1).asType(cw.dtype)  // (B, L, 2C)
      img = (MLX.matmul(x, cw.transposed(1, 0)) + cb).asType(imgIn.dtype) // (B, L, features)
    } else {
      img = first(imgIn)
    }
    let tEmb = tmlp(Krea2Util.timestepEmbed(t, dim: cfg.tdim).asType(img.dtype))  // (B,1,feat)
    let tvec = tproj(tEmb)  // (B,1,6*feat)

    let txtlen = contextIn.dim(1)
    let txtValid = mask[0..., 0..<txtlen]
    let txtMask = Krea2Util.additiveMask(txtValid, dtype: img.dtype)
    var context = txtfusion(contextIn, mask: txtMask)
    context = txtmlp(context)

    var combined = MLX.concatenated([context, img], axis: 1)
    let (cos, sin) = Krea2Rope.make(pos: pos.asType(.float32), axes: cfg.axes, theta: cfg.theta)
    let fullMask = Krea2Util.additiveMask(mask, dtype: img.dtype)

    for block in blocks {
      combined = block(combined, vec: tvec, cos: cos, sin: sin, mask: fullMask)
    }

    let final = last(combined, tvec: tEmb)
    return final[0..., txtlen ..< (txtlen + img.dim(1)), 0...]
  }
}
