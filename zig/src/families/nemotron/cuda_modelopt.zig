//! NVIDIA ModelOpt checkpoints: each projection FP8, NVFP4 or bf16 as its tensors say; the bf16 MTP head made 4-bit.

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const W = @import("cuda_weights.zig");
const kern = @import("cuda_kernels.zig");
const Config = @import("config.zig").Config;
const draft_ids = @import("draft_ids.zig");
const host4 = core.affine4_host;

const Loader = W.Loader;
const Tensor = core.checkpoint.Tensor;
const Nvfp4Experts = cuda.experts.Nvfp4Experts;

pub const Kind = enum { nvfp4, fp8, bf16, q8_0, iq2xxs };

/// Bytes and weights of one ggml block: Q8_0 34 per 32, IQ2_XXS 66 per 256.
fn ggmlBlock(k: Kind) [2]usize {
    return switch (k) {
        .q8_0 => .{ 34, 32 },
        .iq2xxs => .{ 66, 256 },
        else => unreachable,
    };
}

fn kindOf(t: Tensor) !Kind {
    // ggml blocks (Bolt): [rows, blocks, block bytes]; NVFP4 packs pairs as [rows, k / 2]
    if (t.dtype == .u8 and t.rank == 3) return switch (t.dim(2)) {
        34 => .q8_0,
        66 => .iq2xxs,
        else => error.UnsupportedQuantization,
    };
    return switch (t.dtype) {
        .u8 => .nvfp4,
        .f8_e4m3 => .fp8,
        .bf16 => .bf16,
        else => error.UnsupportedQuantization,
    };
}

/// A part's weight, e4m3 block scales (NVFP4), fp32 scale (weight_scale_2 or weight_scale) and input_scale (act).
pub const Part = struct { w: Tensor, blocks: ?Tensor = null, scale: f32 = 1.0, act: f32 = 0, kind: Kind, n: usize, k: usize };

fn name(buf: []u8, comptime fmt: []const u8, args: anytype) ![]const u8 {
    return std.fmt.bufPrint(buf, fmt, args);
}

/// An fp32 tensor of one element, read through the map (a few bytes; the loader's O_DIRECT reads are for bulk).
fn scalar(ck: *core.Checkpoint, n: []const u8) !f32 {
    const t = try ck.get(n);
    if (t.dtype != .f32 or t.bytes.len != 4) return error.UnexpectedTensor;
    return @bitCast(std.mem.readInt(u32, t.bytes[0..4], .little));
}

/// `{pre}.weight` and its scales by the weight's dtype, its static input scale (act); FP8 cache scales set aside.
pub fn part(ck: *core.Checkpoint, pre: []const u8) !Part {
    var buf: [192]u8 = undefined;
    const w = try ck.get(try name(&buf, "{s}.weight", .{pre}));
    if (w.rank != 2 and w.rank != 3) return error.UnexpectedTensor;
    // the KV cache stays bf16; the input scale is read for the checkpoint's math (--precision checkpoint)
    var act: f32 = 0;
    for ([_][]const u8{ "input_scale", "k_scale", "v_scale" }) |extra| {
        const n = try name(&buf, "{s}.{s}", .{ pre, extra });
        if (!ck.has(n)) continue;
        if (extra[0] == 'i') act = try scalar(ck, n) else _ = try ck.get(n);
    }
    const kind = try kindOf(w);
    if ((w.rank == 3) != (kind == .q8_0 or kind == .iq2xxs)) return error.UnexpectedTensor;
    const k = switch (kind) {
        .nvfp4 => w.dim(1) * 2,
        .q8_0, .iq2xxs => w.dim(1) * ggmlBlock(kind)[1],
        else => w.dim(1),
    };
    var p: Part = .{ .w = w, .act = act, .kind = kind, .n = w.dim(0), .k = k };
    switch (kind) {
        .bf16, .q8_0, .iq2xxs => {},
        .fp8 => p.scale = try scalar(ck, try name(&buf, "{s}.weight_scale", .{pre})),
        .nvfp4 => {
            p.blocks = try ck.expect(try name(&buf, "{s}.weight_scale", .{pre}), .f8_e4m3, &.{ p.n, p.k / 16 });
            p.scale = try scalar(ck, try name(&buf, "{s}.weight_scale_2", .{pre}));
        },
    }
    return p;
}

fn kernels(L: *Loader) !*const kern.Modelopt {
    return if (L.ops.k.modelopt) |*m| m else error.ModeloptNotLoaded;
}

/// Row-stacked parts (one projection, or q, k, v) of one kind and one scale as a lane weight, packed on the GPU.
pub fn dense(L: *Loader, ck: *core.Checkpoint, label: []const u8, prefixes: []const []const u8) !kern.QLinear {
    const m = try kernels(L);
    var parts: [3]Part = undefined;
    var n: usize = 0;
    for (prefixes, 0..) |pre, i| {
        parts[i] = try part(ck, pre);
        n += parts[i].n;
    }
    const first = parts[0];
    for (parts[0..prefixes.len]) |p| if (p.kind != first.kind or p.scale != first.scale or p.k != first.k) return error.MixedQuantization;
    // the checkpoint's math: e4m3 rows by the input scale every part of an FP8 projection shares
    const act: f32 = if (L.ops.k.ck != null and first.kind == .fp8) first.act else 0;
    if (act != 0) for (parts[0..prefixes.len]) |p| if (p.act != act) return error.MixedQuantization;
    if (L.ops.k.ck != null and first.kind == .fp8 and act == 0) return error.NoInputScale;
    const k = first.k;
    if (k % 64 != 0) return error.UnexpectedTensor;
    var buf: [128]u8 = undefined;
    const s = L.ops.s;
    switch (first.kind) {
        .q8_0, .iq2xxs => return ggmlDense(L, label, parts[0..prefixes.len], n, k),
        .bf16, .fp8 => {
            // Volta has no block-scaled FP8 matmul: run an FP8 projection W8A16 through the validated
            // bf16 GEMM (dequant e4m3 -> bf16). Blackwell keeps the native FP8 (W8A8 checkpoint math).
            const to_bf16: bool = first.kind == .fp8 and L.ops.k.volta_lane != null;
            const size: usize = if (first.kind == .bf16) 2 else 1;             // raw bytes/element uploaded
            const out_size: usize = if (first.kind == .bf16 or to_bf16) 2 else 1; // packed bytes/element
            const base = try L.tmp(n * k * size);
            var at: usize = 0;
            for (parts[0..prefixes.len]) |p| {
                try L.src.upload(base + at, p.w.bytes);
                at += p.w.bytes.len;
            }
            const out = try L.alloc(try name(&buf, "{s}.weight", .{label}), cuda.modelopt.padded(n) * k * out_size);
            try L.src.flush();
            if (first.kind == .bf16) {
                try m.pack.bf16Lane(s, base, out, n, k);
                return .{ .bf16 = cuda.modelopt.bf16Weight(out, n, k) };
            }
            if (to_bf16) {
                try m.pack.fp8ToBf16Lane(s, base, out, n, k);
                return .{ .bf16 = cuda.modelopt.bf16WeightScaled(out, first.scale, n, k) };
            }
            try m.pack.fp8Lane(s, base, out, n, k);
            var w = cuda.modelopt.fp8Weight(out, first.scale, n, k);
            w.act = act;
            return .{ .fp8 = w };
        },
        .nvfp4 => {
            const codes = n * k / 2;
            const base = try L.tmp(codes + n * k / 16);
            var at = [2]usize{ 0, codes };
            for (parts[0..prefixes.len]) |p| for ([2]Tensor{ p.w, p.blocks.? }, &at) |t, *o| {
                try L.src.upload(base + o.*, t.bytes);
                o.* += t.bytes.len;
            };
            const words = try L.alloc(try name(&buf, "{s}.weight", .{label}), cuda.modelopt.padded(n) * k / 2);
            const tiles = try L.alloc(try name(&buf, "{s}.scales", .{label}), cuda.modelopt.padded(n) * k / 16);
            try L.src.flush();
            try m.pack.fp4Lane(s, base, base + codes, words, tiles, n, k);
            return .{ .nvfp4 = cuda.nvfp4.weight(words, tiles, first.scale, n, k) };
        },
    }
}

/// Row-stacked ggml parts (q, k, v: rows after rows) uploaded as stored, then tiled on the GPU (a byte permutation).
fn ggmlDense(L: *Loader, label: []const u8, parts: []const Part, n: usize, k: usize) !kern.QLinear {
    const g = try ggmlKernels(L);
    const kind = parts[0].kind;
    const base = try L.tmp(n * k / ggmlBlock(kind)[1] * ggmlBlock(kind)[0]);
    var at: usize = 0;
    for (parts) |p| {
        try L.src.upload(base + at, p.w.bytes);
        at += p.w.bytes.len;
    }
    var buf: [128]u8 = undefined;
    const gk: cuda.ggml.Kind = if (kind == .q8_0) .q8_0 else .iq2xxs;
    const out = try L.alloc(try name(&buf, "{s}.weight", .{label}), cuda.ggml.bytes(gk, n, k));
    try L.src.flush();
    try g.pack(L.ops.s, gk, base, out, n, k, 1);
    return if (kind == .q8_0) .{ .q8_0 = cuda.ggml.q8Weight(out, n, k) } else .{ .iq2xxs = cuda.ggml.iq2Weight(out, n, k) };
}

fn ggmlKernels(L: *Loader) !*const cuda.ggml.Kernels {
    return if (L.ops.k.ggml) |*g| g else error.GgmlNotLoaded;
}

/// IQ2_XXS grouped experts from `count` tensors named by `names`, [n, k] each, uploaded back to back and tiled.
fn iq2Experts(L: *Loader, ck: *core.Checkpoint, label: []const u8, count: usize, n: usize, k: usize, names: anytype) !u64 {
    const g = try ggmlKernels(L);
    const one = n * k / 256 * 66;
    const base = try L.tmp(count * one);
    var b: [192]u8 = undefined;
    for (0..count) |i| {
        const p = try part(ck, try names.at(&b, i));
        if (p.kind != .iq2xxs or p.n != n or p.k != k) {
            std.log.err("{s}: {t} [{d}, {d}], expected IQ2_XXS [{d}, {d}]", .{ try names.at(&b, i), p.kind, p.n, p.k, n, k });
            return error.UnexpectedTensor;
        }
        try L.src.upload(base + i * one, p.w.bytes);
    }
    const out = try L.alloc(label, cuda.ggml.bytes(.iq2xxs, n, k) * count);
    try L.src.flush();
    try g.pack(L.ops.s, .iq2xxs, base, out, n, k, count);
    return out;
}

/// A LatentMoE layer (Bolt): router and bias, fc1/fc2_latent_proj, latent IQ2_XXS experts, full-width shared expert.
pub fn latent(L: *Loader, ck: *core.Checkpoint, label: []const u8, pre: []const u8, c: Config, layer: usize) !W.MoE {
    const lm = c.layerMoe(layer);
    const E = lm.experts;
    const Lw = c.latent;
    var a: [128]u8 = undefined;
    var b: [192]u8 = undefined;
    const r = try routerOf(L, try name(&a, "{s}.router", .{label}), try ck.get(try name(&b, "{s}gate.weight", .{pre})), E, c.hidden);
    const bias = try ck.expect(try name(&b, "{s}gate.e_score_correction_bias", .{pre}), .f32, &.{E});
    var m: W.MoE = .{ .router = r, .bias = try L.raw(try name(&a, "{s}.bias", .{label}), bias), .experts = undefined };
    var ex: kern.Experts = .{ .format = .iq2xxs, .up = 0, .down = 0, .width = lm.width, .dims = Lw, .experts = E, .relu2 = true };
    const Names = struct {
        pre: []const u8,
        leaf: []const u8,
        fn at(x: @This(), buf: []u8, i: usize) ![]const u8 {
            return std.fmt.bufPrint(buf, "{s}experts.{d}.{s}", .{ x.pre, i, x.leaf });
        }
    };
    ex.up = try iq2Experts(L, ck, try name(&a, "{s}.experts.up", .{label}), E, lm.width, Lw, Names{ .pre = pre, .leaf = "up_proj" });
    ex.down = try iq2Experts(L, ck, try name(&a, "{s}.experts.down", .{label}), E, Lw, lm.width, Names{ .pre = pre, .leaf = "down_proj" });
    m.experts = ex;
    const Shared = struct {
        pre: []const u8,
        leaf: []const u8,
        fn at(x: @This(), buf: []u8, _: usize) ![]const u8 {
            return std.fmt.bufPrint(buf, "{s}shared_experts.{s}", .{ x.pre, x.leaf });
        }
    };
    var sh: kern.Experts = .{ .format = .iq2xxs, .up = 0, .down = 0, .width = c.shared_width, .dims = c.hidden, .experts = 1, .relu2 = true };
    sh.up = try iq2Experts(L, ck, try name(&a, "{s}.shared.up", .{label}), 1, c.shared_width, c.hidden, Shared{ .pre = pre, .leaf = "up_proj" });
    sh.down = try iq2Experts(L, ck, try name(&a, "{s}.shared.down", .{label}), 1, c.hidden, c.shared_width, Shared{ .pre = pre, .leaf = "down_proj" });
    var p1: [192]u8 = undefined;
    var p2: [192]u8 = undefined;
    m.bolt = .{
        .fc1 = try dense(L, ck, try name(&a, "{s}.fc1_latent", .{label}), &.{try name(&p1, "{s}fc1_latent_proj", .{pre})}),
        .fc2 = try dense(L, ck, try name(&b, "{s}.fc2_latent", .{label}), &.{try name(&p2, "{s}fc2_latent_proj", .{pre})}),
        .shared = sh,
    };
    return m;
}

/// The router as the kernels read it: bf16 [E, D]; an fp32 router is taken only when every value is a bf16.
pub fn router(L: *Loader, label: []const u8, t: Tensor, c: Config) !u64 {
    return routerOf(L, label, t, c.experts, c.hidden);
}

fn routerOf(L: *Loader, label: []const u8, t: Tensor, experts_: usize, hidden: usize) !u64 {
    if (t.rank != 2 or t.dim(0) != experts_ or t.dim(1) != hidden) return error.UnexpectedTensor;
    if (t.dtype == .bf16) return L.raw(label, t);
    if (t.dtype != .f32) return error.UnexpectedTensor;
    const raw = try L.gpa.alloc(u8, t.bytes.len);
    defer L.gpa.free(raw);
    try L.src.read(raw, t.bytes);
    const out = try L.keep(u16, t.bytes.len / 4);
    for (out, 0..) |*o, i| {
        const bits = std.mem.readInt(u32, raw[4 * i ..][0..4], .little);
        if (bits & 0xFFFF != 0) return error.RouterNotBf16; // our router kernel reads bf16 weights
        o.* = @intCast(bits >> 16);
    }
    const ptr = try L.alloc(label, out.len * 2);
    try L.src.flush();
    try L.ops.upload(ptr, std.mem.sliceAsBytes(out));
    return ptr;
}

/// Per-expert fp32 global scales on the GPU.
pub fn scales(L: *Loader, label: []const u8, values: []const f32) !u64 {
    const host = try L.keep(f32, values.len);
    @memcpy(host, values);
    const ptr = try L.alloc(label, values.len * 4);
    try L.src.flush();
    try L.ops.upload(ptr, std.mem.sliceAsBytes(host));
    return ptr;
}

/// A MoE layer: the router, its bias, and NVFP4 experts with the shared expert's halves as experts E and E + 1.
pub fn moe(L: *Loader, ck: *core.Checkpoint, label: []const u8, pre: []const u8, c: Config) !W.MoE {
    var a: [128]u8 = undefined;
    var b: [192]u8 = undefined;
    const r = try router(L, try name(&a, "{s}.router", .{label}), try ck.get(try name(&b, "{s}gate.weight", .{pre})), c);
    const bias = try ck.expect(try name(&b, "{s}gate.e_score_correction_bias", .{pre}), .f32, &.{c.experts});
    return .{ .router = r, .bias = try L.raw(try name(&a, "{s}.bias", .{label}), bias), .experts = try experts(L, ck, label, pre, c) };
}

fn experts(L: *Loader, ck: *core.Checkpoint, label: []const u8, pre: []const u8, c: Config) !kern.Experts {
    const m = try kernels(L);
    const E = c.experts;
    const e = E + 2;
    const width = c.expert_width;
    const D = c.hidden;
    var layer: kern.Experts = .{ .format = .nvfp4, .up = 0, .down = 0, .width = width, .dims = D, .experts = e, .relu2 = true };
    const gs = try L.gpa.alloc(f32, 2 * e);
    defer L.gpa.free(gs);
    var a: [128]u8 = undefined;
    var b: [192]u8 = undefined;
    for ([2]bool{ true, false }) |up| {
        const n = if (up) width else D;
        const k = if (up) D else width;
        const codes = n * k / 2;
        const blocks = n * k / 16;
        const base = try L.tmp(e * (codes + blocks));
        for (0..E) |i| {
            const p = try part(ck, try name(&b, "{s}experts.{d}.{s}", .{ pre, i, if (up) "up_proj" else "down_proj" }));
            if (p.kind != .nvfp4 or p.n != n or p.k != k) return error.UnexpectedTensor;
            try L.src.upload(base + i * codes, p.w.bytes);
            try L.src.upload(base + e * codes + i * blocks, p.blocks.?.bytes);
            gs[@intFromBool(!up) * e + i] = p.scale;
        }
        const s = try part(ck, try name(&b, "{s}shared_experts.{s}", .{ pre, if (up) "up_proj" else "down_proj" }));
        if (s.kind != .nvfp4 or s.n != (if (up) 2 * n else n) or s.k != (if (up) k else 2 * k)) return error.SharedExpertWidth;
        try L.src.upload(base + E * codes, s.w.bytes); // up: rows 0..2W are experts E and E + 1; down: [D, 2W] apart
        try L.src.upload(base + e * codes + E * blocks, s.blocks.?.bytes);
        gs[@intFromBool(!up) * e + E] = s.scale;
        gs[@intFromBool(!up) * e + E + 1] = s.scale;
        const out = try L.alloc(try name(&a, "{s}.experts.{s}", .{ label, if (up) "up" else "down" }), Nvfp4Experts.bytes(e, n, k, 1));
        try L.src.flush();
        const st = L.ops.s;
        if (up) {
            try m.pack.fp4Experts(st, base, base + e * codes, out, e, n, k, 0, k, 1, 0);
            layer.up = out;
        } else {
            try m.pack.fp4Experts(st, base, base + e * codes, out, E, n, k, 0, k, 1, 0);
            const one = Nvfp4Experts.bytes(1, n, k, 1);
            for (0..2) |h| try m.pack.fp4Experts(st, base + E * codes, base + e * codes + E * blocks, out + (E + h) * one, 1, n, 2 * k, h * k, k, 1, 0);
            layer.down = out;
        }
    }
    layer.up_scale = try scales(L, try name(&a, "{s}.experts.up_scale", .{label}), gs[0..e]);
    layer.down_scale = try scales(L, try name(&a, "{s}.experts.down_scale", .{label}), gs[e..]);
    return layer;
}

/// Magnitudes of an e2m1 nibble (bits 0-2), matching core/quant/nvfp4.e2m1.
const e2m1mag = [_]f64{ 0, 0.5, 1, 1.5, 2, 3, 4, 6 };

/// Nearest e2m1 code (bit 3 = sign) to a value; quantizes MTP expert weights on the host.
fn e2m1code(x: f32) u4 {
    const ax = if (x < 0) -x else x;
    const s: u4 = if (x < 0) 1 else 0;
    var bi: u3 = 0;
    var bd: f64 = std.math.inf(f64);
    inline for (e2m1mag, 0..) |v, i| {
        const d = if (v >= ax) v - ax else ax - v;
        if (d < bd) {
            bd = d;
            bi = @as(u3, @intCast(i));
        }
    }
    return bi | (s << 3);
}

/// Nearest e4m3 byte to a non-negative value (a block scale, in [0,1] here); matches core/quant/fp8.e4m3.
fn e4m3byte(x: f32) u8 {
    const ax = if (x < 0) -x else x;
    if (ax == 0) return 0;
    if (ax >= 448) return 0x7E;
    if (ax < 0.015625) {
        var m: i32 = @intFromFloat(std.math.round(@as(f64, ax) * 512.0));
        if (m < 0) m = 0;
        if (m >= 8) return 8; // min normal 2^-6
        return @as(u8, @intCast(m)); // subnormal, exponent field 0
    }
    const e: i32 = if (ax >= 1.0) 0
        else if (ax >= 0.5) -1
        else if (ax >= 0.25) -2
        else if (ax >= 0.125) -3
        else if (ax >= 0.0625) -4
        else if (ax >= 0.03125) -5
        else -6;
    const base: f32 = switch (e) {
        0 => 1.0,
        -1 => 0.5,
        -2 => 0.25,
        -3 => 0.125,
        -4 => 0.0625,
        -5 => 0.03125,
        else => 0.015625,
    };
    var m: i32 = @intFromFloat(std.math.round((@as(f64, ax) / base - 1) * 8));
    if (m < 0) m = 0;
    if (m >= 8) return @as(u8, @intCast((e + 8) << 3)); // rounded up to the next level, mantissa 0
    return @as(u8, @intCast(((e + 7) << 3) | @as(u3, @intCast(m))));
}

/// Quantize one BF16 [n, k] checkpoint matrix into packed NVFP4: e2m1 codes (col 0 = low nibble),
/// one e4m3 scale every 16 inputs, and a global scale (amax / 6). Returns the global scale.
fn quantMatrix(L: *Loader, ck: *core.Checkpoint, tname: []const u8, n: usize, k: usize, codes: []u8, blocks: []u8) !f32 {
    const t = try ck.get(tname);
    if (t.dtype != .bf16 or t.rank != 2 or t.dim(0) != n or t.dim(1) != k or k % 16 != 0) return error.UnexpectedTensor;
    const count = n * k;
    const raw = try L.gpa.alloc(u8, count * 2);
    defer L.gpa.free(raw);
    try L.src.read(raw, t.bytes);
    var amax: f32 = 0;
    for (0..count) |i| {
        const x = host4.f32of(std.mem.readInt(u16, raw[2 * i ..][0..2], .little));
        const a = if (x < 0) -x else x;
        if (a > amax) amax = a;
    }
    const global: f32 = if (amax > 0) amax / 6.0 else 0.0;
    var v: [16]f32 = undefined;
    for (0..n) |r| {
        for (0..k / 16) |bl| {
            const base16 = r * k + bl * 16;
            for (0..16) |j| v[j] = host4.f32of(std.mem.readInt(u16, raw[2 * (base16 + j) ..][0..2], .little));
            var bamax: f32 = 0;
            for (v) |x| {
                const a = if (x < 0) -x else x;
                if (a > bamax) bamax = a;
            }
            const cb = r * (k / 2) + bl * 8;
            const sb = r * (k / 16) + bl;
            if (bamax > 0 and amax > 0) {
                const scale_b = bamax / 6.0;
                blocks[sb] = e4m3byte(bamax / amax);
                for (0..8) |yb| codes[cb + yb] = @as(u8, e2m1code(v[2 * yb] / scale_b)) | (@as(u8, e2m1code(v[2 * yb + 1] / scale_b)) << 4);
            } else {
                blocks[sb] = 0;
                for (0..8) |yb| codes[cb + yb] = 0;
            }
        }
    }
    return global;
}

/// The MTP MoE experts as NVFP4: the MTP weights ship BF16 (unquantized), so they are quantized on the
/// host into the same packed layout the backbone's pre-quantized experts() produces, then packed.
fn mtpExperts(L: *Loader, ck: *core.Checkpoint, c: Config) !kern.Experts {
    const m = try kernels(L);
    const E = c.experts;
    const e = E + 2;
    const width = c.expert_width;
    const D = c.hidden;
    const p1 = "mtp.layers.1.mixer.";
    var layer: kern.Experts = .{ .format = .nvfp4, .up = 0, .down = 0, .width = width, .dims = D, .experts = e, .relu2 = true };
    const gs = try L.gpa.alloc(f32, 2 * e);
    defer L.gpa.free(gs);
    var b: [192]u8 = undefined;
    for ([2]bool{ true, false }) |up| {
        const n = if (up) width else D;
        const k = if (up) D else width;
        const codes = n * k / 2;
        const blocks = n * k / 16;
        const base = try L.tmp(e * (codes + blocks));
        const staging = try L.gpa.alloc(u8, e * (codes + blocks));
        defer L.gpa.free(staging);
        for (0..E) |i| {
            const nm = try name(&b, "{s}experts.{d}.{s}.weight", .{ p1, i, if (up) "up_proj" else "down_proj" });
            gs[@intFromBool(!up) * e + i] = try quantMatrix(L, ck, nm, n, k, staging[i * codes ..][0..codes], staging[e * codes + i * blocks ..][0..blocks]);
        }
        const snm = try name(&b, "{s}shared_experts.{s}.weight", .{ p1, if (up) "up_proj" else "down_proj" });
        const sg = try quantMatrix(L, ck, snm, if (up) 2 * n else n, if (up) k else 2 * k, staging[E * codes ..][0..2 * codes], staging[e * codes + E * blocks ..][0..2 * blocks]);
        gs[@intFromBool(!up) * e + E] = sg;
        gs[@intFromBool(!up) * e + E + 1] = sg;
        try L.ops.upload(base, staging);
        const out = try L.alloc(if (up) "mtp.moe.experts.up" else "mtp.moe.experts.down", Nvfp4Experts.bytes(e, n, k, 1));
        const st = L.ops.s;
        if (up) {
            try m.pack.fp4Experts(st, base, base + e * codes, out, e, n, k, 0, k, 1, 0);
            layer.up = out;
        } else {
            try m.pack.fp4Experts(st, base, base + e * codes, out, E, n, k, 0, k, 1, 0);
            const one = Nvfp4Experts.bytes(1, n, k, 1);
            for (0..2) |h| try m.pack.fp4Experts(st, base + E * codes, base + e * codes + E * blocks, out + (E + h) * one, 1, n, 2 * k, h * k, k, 1, 0);
            layer.down = out;
        }
    }
    layer.up_scale = try scales(L, "mtp.moe.experts.up_scale", gs[0..e]);
    layer.down_scale = try scales(L, "mtp.moe.experts.down_scale", gs[e..]);
    return layer;
}

/// The bf16 token table as stored.
pub fn embed(L: *Loader, ck: *core.Checkpoint, c: Config) !W.Embed {
    var buf: [64]u8 = undefined;
    const t = try ck.expect(try name(&buf, "{s}backbone.embeddings.weight", .{c.prefix()}), .bf16, &.{ c.vocab, c.hidden });
    return .{ .w = try L.raw("embed.weight", t), .n = c.vocab, .k = c.hidden, .bf16 = true };
}

/// A bf16 matrix made MLX affine-4 in groups of 64 on the host: the words, scales and biases the 4-bit loader reads.
fn quantized(L: *Loader, t: Tensor, words: []u32, sc: []u16, bi: []u16) !void {
    if (t.dtype != .bf16 or t.rank != 2 or t.dim(1) % host4.group != 0) return error.UnexpectedTensor;
    const count = t.dim(0) * t.dim(1);
    const raw = try L.gpa.alloc(u8, count * 2);
    defer L.gpa.free(raw);
    try L.src.read(raw, t.bytes);
    const values = try L.gpa.alloc(f32, count);
    defer L.gpa.free(values);
    for (values, 0..) |*v, i| v.* = host4.f32of(std.mem.readInt(u16, raw[2 * i ..][0..2], .little));
    host4.quantize(values, words, sc, bi);
}

/// `count` bf16 matrices [n, k] (`fmt` names each by pre and index) made 4-bit, stacked as the MLX loader reads them.
fn stack4(L: *Loader, ck: *core.Checkpoint, comptime fmt: []const u8, pre: []const u8, count: usize, n: usize, k: usize) ![3]Tensor {
    const per = n * k;
    const words = try L.keep(u32, count * per / 8);
    const sc = try L.keep(u16, count * per / host4.group);
    const bi = try L.keep(u16, count * per / host4.group);
    var buf: [192]u8 = undefined;
    for (0..count) |i| {
        const t = try ck.get(try name(&buf, fmt, .{ pre, i }));
        if (t.dim(0) != n or t.dim(1) != k) return error.UnexpectedTensor;
        try quantized(L, t, words[i * per / 8 ..][0 .. per / 8], sc[i * per / host4.group ..][0 .. per / host4.group], bi[i * per / host4.group ..][0 .. per / host4.group]);
    }
    return .{ W.made(.u32, count * n, k / 8, std.mem.sliceAsBytes(words)), W.made(.bf16, count * n, k / host4.group, std.mem.sliceAsBytes(sc)), W.made(.bf16, count * n, k / host4.group, std.mem.sliceAsBytes(bi)) };
}

/// One bf16 matrix `{pre}{mid}.weight` made 4-bit.
fn one4(L: *Loader, ck: *core.Checkpoint, pre: []const u8, mid: []const u8) ![3]Tensor {
    var buf: [192]u8 = undefined;
    const t = try ck.get(try name(&buf, "{s}{s}.weight", .{ pre, mid }));
    if (t.rank != 2) return error.UnexpectedTensor;
    const n = t.dim(0);
    const k = t.dim(1);
    const words = try L.keep(u32, n * k / 8);
    const sc = try L.keep(u16, n * k / host4.group);
    const bi = try L.keep(u16, n * k / host4.group);
    try quantized(L, t, words, sc, bi);
    return .{ W.made(.u32, n, k / 8, std.mem.sliceAsBytes(words)), W.made(.bf16, n, k / host4.group, std.mem.sliceAsBytes(sc)), W.made(.bf16, n, k / host4.group, std.mem.sliceAsBytes(bi)) };
}

/// The MTP head (bf16 in these checkpoints) made affine-4 as the MLX one is; the draft head is lm_head's NVFP4 rows.
pub fn mtp(L: *Loader, ck: *core.Checkpoint, c: Config) !void {
    const p0 = "mtp.layers.0.";
    const p1 = "mtp.layers.1.mixer.";
    const att = p0 ++ "mixer.";
    const moe_: W.MoE = .{ .router = try router(L, "mtp.moe.router", try ck.get(p1 ++ "gate.weight"), c), .bias = try L.raw("mtp.moe.bias", try ck.expect(p1 ++ "gate.e_score_correction_bias", .f32, &.{c.experts})), .experts = try mtpExperts(L, ck, c) };
    L.w.mtp = .{
        .enorm = try L.raw("mtp.enorm", try ck.get(p0 ++ "enorm.weight")),
        .hnorm = try L.raw("mtp.hnorm", try ck.get(p0 ++ "hnorm.weight")),
        .eh_proj = try dense(L, ck, "mtp.eh_proj", &.{p0 ++ "eh_proj"}),
        .attn_norm = try L.raw("mtp.attn_norm", try ck.get(p0 ++ "norm.weight")),
        .attn = .{ .qkv = try dense(L, ck, "mtp.attn.qkv", &.{att ++ "q_proj", att ++ "k_proj", att ++ "v_proj"}), .o = try dense(L, ck, "mtp.attn.o", &.{att ++ "o_proj"}) },
        .moe_norm = try L.raw("mtp.moe_norm", try ck.get("mtp.layers.1.norm.weight")),
        .moe = moe_,
        .final_norm = try L.raw("mtp.final_norm", try ck.get("mtp.layers.1.final_layernorm.weight")),
    };
    try draftHead(L, ck, c);
}

/// head_rows: lm_head's NVFP4 rows for the draft ids, packed as a head of their own (one global scale).
fn draftHead(L: *Loader, ck: *core.Checkpoint, c: Config) !void {
    const m = try kernels(L);
    const ids = try draft_ids.load(L.gpa, c.vocab);
    defer L.gpa.free(ids);
    const p = try part(ck, "lm_head");
    if (p.kind != .nvfp4 or p.n != c.vocab) return error.UnexpectedTensor;
    const rows = [2]usize{ p.k / 2, p.k / 16 };
    const host = try L.staging(ids.len * (rows[0] + rows[1]));
    try W.gatherRows(L, p.w, ids, host);
    try W.gatherRows(L, p.blocks.?, ids, host[ids.len * rows[0] ..]);
    const base = try L.tmp(host.len);
    try L.ops.upload(base, host);
    const n = ids.len;
    const words = try L.alloc("draft_head.weight", cuda.modelopt.padded(n) * p.k / 2);
    const tiles = try L.alloc("draft_head.scales", cuda.modelopt.padded(n) * p.k / 16);
    try m.pack.fp4Lane(L.ops.s, base, base + n * rows[0], words, tiles, n, p.k);
    L.w.draft_head = .{ .nvfp4 = cuda.nvfp4.weight(words, tiles, p.scale, n, p.k) };
    try W.draftIds(L, ids);
}

/// Super's head_rows: lm_head's FP8 rows for the draft ids, packed as a head of their own (the head's scales).
pub fn fp8DraftHead(L: *Loader, ck: *core.Checkpoint, c: Config) !void {
    const m = try kernels(L);
    const ids = try draft_ids.load(L.gpa, c.vocab);
    defer L.gpa.free(ids);
    var buf: [192]u8 = undefined;
    const p = try part(ck, try name(&buf, "{s}lm_head", .{c.prefix()}));
    if (p.kind != .fp8 or p.n != c.vocab or p.k != c.hidden) return error.UnexpectedTensor;
    const act: f32 = if (L.ops.k.ck != null) p.act else 0;
    if (L.ops.k.ck != null and act == 0) return error.NoInputScale;
    const host = try L.staging(ids.len * p.k);
    try W.gatherRows(L, p.w, ids, host);
    const base = try L.tmp(host.len);
    try L.ops.upload(base, host);
    const out = try L.alloc("draft_head.weight", cuda.modelopt.padded(ids.len) * p.k);
    try m.pack.fp8Lane(L.ops.s, base, out, ids.len, p.k);
    var w = cuda.modelopt.fp8Weight(out, p.scale, ids.len, p.k);
    w.act = act;
    L.w.draft_head = .{ .fp8 = w };
    try W.draftIds(L, ids);
    L.w.draft_coverage = try .init(L.gpa, c.vocab, ids); // the whole head stays resident: streams draft on either
}
