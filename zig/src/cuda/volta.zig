//! Volta (sm_70, Quadro GV100) kernel dispatch.
//!
//! When the device's compute-capability major is 7 the engine loads the
//! *_volta fatbins and resolves their `extern "C"` kernels instead of the
//! sm_80+ images. Every symbol name below is the exact string cuobjdump lists
//! in the built fatbin, so a typo cannot hide behind an unused path: the
//! loader resolves each one at startup and a wrong name is a hard error.
//!
//! The runtime already knows the capability (Driver.capability /
//! Context.capability return 70, 80, 90, 120); callers pass (capability)/10 as
//! `major` to the family loaders, which branch on isVolta(major).
const std = @import("std");
const driver = @import("driver.zig");
const module_ = @import("module.zig");
const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");
const kernels = @import("kernels.zig");
const memory_ = @import("memory.zig");

const Driver = driver.Driver;
const Error = driver.Error;
const Module = module_.Module;
const Function = module_.Function;
const Stream = stream_.Stream;

/// True when the compute-capability major selects the Volta (sm_70) images.
pub fn isVolta(major: u32) bool {
    return major == 7;
}

// ---------------------------------------------------------------------------
// attention_volta: the prefill-attention path (tf_vattn_*).
// A "shared" kernel runs the whole row in one block (128 or 256 threads); a
// "tail" kernel handles the last, partially-filled rows.
// ---------------------------------------------------------------------------
pub const Attention = struct {
    mod: Module,
    shared_128: Function,
    shared_256: Function,
    tail_128: Function,
    tail_256: Function,

    pub fn load(d: *const Driver) Error!Attention {
        var mod = try Module.load(d, kernels.attention_volta);
        return Attention{
            .mod = mod,
            .shared_128 = try mod.function("tf_vattn_shared_128"),
            .shared_256 = try mod.function("tf_vattn_shared_256"),
            .tail_128 = try mod.function("tf_vattn_tail_128"),
            .tail_256 = try mod.function("tf_vattn_tail_256"),
        };
    }

    pub fn unload(self: *Attention) void {
        self.mod.unload();
        self.* = undefined;
    }
};

// ---------------------------------------------------------------------------
// volta_qmm: the lane matmul (tf_vqmm884_*) plus its prep / reduce / scale
// helpers. matmul[xi][y][z] is qmm884_body<xs[xi], y, z> with xs = {1,2,4}.
// ---------------------------------------------------------------------------
pub const Qmm = struct {
    mod: Module,
    /// matmul[xi][y][z]: xi in 0..3 selects xs[xi] in {1,2,4}; y, z in 0..1.
    matmul: [3][2][2]Function,
    prep: Function,
    prep_any: Function,
    prep_frag: Function,
    prep_frag_any: Function,
    reduce: Function,
    dequant: Function,
    unscale: Function,

    fn matmulName(comptime xi: u3, comptime y: u32, comptime z: u32) [:0]const u8 {
        const xs = [_]u32{ 1, 2, 4 };
        return std.fmt.comptimePrint("tf_vqmm884_{d}_{d}_{d}", .{ xs[xi], y, z });
    }

    pub fn load(d: *const Driver) Error!Qmm {
        var mod = try Module.load(d, kernels.volta_qmm);
        var m: [3][2][2]Function = undefined;
        inline for (0..3) |xi| {
            inline for (0..2) |y| {
                inline for (0..2) |z| {
                    m[xi][y][z] = try mod.function(matmulName(@intCast(xi), y, z));
                }
            }
        }
        return Qmm{
            .mod = mod,
            .matmul = m,
            .prep = try mod.function("tf_vprep"),
            .prep_any = try mod.function("tf_vprep_any"),
            .prep_frag = try mod.function("tf_vprep_frag"),
            .prep_frag_any = try mod.function("tf_vprep_frag_any"),
            .reduce = try mod.function("tf_vreduce"),
            .dequant = try mod.function("tf_vdequant"),
            .unscale = try mod.function("tf_vunscale"),
        };
    }

    pub fn unload(self: *Qmm) void {
        self.mod.unload();
        self.* = undefined;
    }
};

// ---------------------------------------------------------------------------
// qmmf_volta: the FP4/FP8/MXFP8 lane matmul (tf_vqmmf884s_*) with its
// dequant / reduce / unscale / gather helpers. matmul[fmt][mt]: fmt in 0..3 is
// {0=fp4,1=fp8,2=mxfp8}; mt in 0..3 is the row tile {1,2,4}.
// ---------------------------------------------------------------------------
pub const Qmmf = struct {
    mod: Module,
    /// matmul[fmt][mti]: fmt in 0..3 is {0,1,2}; mti in 0..3 selects {1,2,4}.
    matmul: [3][3]Function,
    /// bf16[mti]: tf_vbf16_{1,2,4} — dense bf16 rows x the packed bf16-lane weight.
    bf16: [3]Function,
    /// dequant[i]: tf_vqmmf884s_dequant_i for i in 0..3.
    dequant: [3]Function,
    /// tile_x: tf_vqmmf884s_tile_x — X (bf16 rows) into the matmul's input tiles.
    tile_x: Function,
    gather_rows: Function,
    reduce: Function,
    unscale2: Function,

    fn matmulName(comptime fmt: u3, comptime mt: u32) [:0]const u8 {
        return std.fmt.comptimePrint("tf_vqmmf884s_{d}_{d}", .{ fmt, mt });
    }

    fn bf16Name(comptime mt: u32) [:0]const u8 {
        return std.fmt.comptimePrint("tf_vbf16_{d}", .{mt});
    }

    fn dequantName(comptime i: u3) [:0]const u8 {
        return std.fmt.comptimePrint("tf_vqmmf884s_dequant_{d}", .{i});
    }

    pub fn load(d: *const Driver) Error!Qmmf {
        var mod = try Module.load(d, kernels.qmmf_volta);
        const mts = [_]u32{ 1, 2, 4 };
        var m: [3][3]Function = undefined;
        inline for (0..3) |fmt| {
            inline for (0..3) |mti| {
                m[fmt][mti] = try mod.function(matmulName(@intCast(fmt), mts[mti]));
            }
        }
        var dq: [3]Function = undefined;
        inline for (0..3) |i| dq[i] = try mod.function(dequantName(@intCast(i)));
        var bf: [3]Function = undefined;
        inline for (0..3) |mti| bf[mti] = try mod.function(bf16Name(mts[mti]));
        return Qmmf{
            .mod = mod,
            .matmul = m,
            .bf16 = bf,
            .dequant = dq,
            .tile_x = try mod.function("tf_vqmmf884s_tile_x"),
            .gather_rows = try mod.function("tf_vqmmf884s_gather_rows"),
            .reduce = try mod.function("tf_vqmmf884s_reduce"),
            .unscale2 = try mod.function("tf_vqmmf884s_unscale2"),
        };
    }

    pub fn unload(self: *Qmmf) void {
        self.mod.unload();
        self.* = undefined;
    }
};

// ---------------------------------------------------------------------------
// qmmf_volta Lane: the dispatch on top of Qmmf. plan() picks the K slices and
// the row tile from the shape alone (so a row's bits never depend on the row
// count, qmm.split_k884 + _row_tiles); matmul() launches the tensor-core GEMM
// (fp16 in, fp32 sums) and, for sk > 1, the kernel's split-K counter (CNT)
// reduces the slice partials in-kernel (no separate reduce kernel needed).
// The input rows come in fragment order (the tile_x / prep884 layout); rs is
// the per-row scale and alpha the per-column factor, both applied in-kernel.
// ---------------------------------------------------------------------------
pub const Lane = struct {
    d: *const Driver, // the driver that loaded the images; fp4 allocates its scratch with it
    q: Qmmf, // the qmmf_volta kernels (matmul, tile_x, dequant, ...)
    prep_fn: Function, // tf_vprep: bf16 rows -> fp16 rows + row scales (from volta_qmm)
    prep_mod: Module, // the volta_qmm module that owns prep_fn

    // Persistent decode scratch: grown eagerly once (the warmup runs before capture) for
    // max_decode_rows and then reused, so a captured graph's baked-in pointers stay valid and no
    // cuMemAlloc lands inside the capture window. A call with rows > max_decode_rows (an eager
    // prefill) falls back to per-call allocation, which is legal there. sc_part/sc_cnt are shared
    // with the bf16 path (same size formula for a given m, n, sk).
    max_decode_rows: u32 = 0,
    sc_alpha: memory_.DeviceBuffer, // per-column factor [n]
    sc_x16: memory_.DeviceBuffer, // fp16 rows [m, k]
    sc_xt: memory_.DeviceBuffer, // tiled rows
    sc_rs: memory_.DeviceBuffer, // per-row scale [m]
    sc_part: memory_.DeviceBuffer, // split-K partials
    sc_cnt: memory_.DeviceBuffer, // split-K counters

    const COLS: u32 = 128; // n-columns a block covers
    const THREADS: u32 = 128; // threads a block
    const TARGET: u32 = 640; // blocks split_k884 fills
    const MTS = [_]u32{ 1, 2, 4 }; // the row tiles matmul[fmt][mti] was built for

    pub fn load(d: *const Driver) Error!Lane {
        var q = try Qmmf.load(d);
        errdefer q.unload();
        var prep_mod = try Module.load(d, kernels.volta_qmm);
        errdefer prep_mod.unload();
        const prep_fn = try prep_mod.function("tf_vprep");
        return .{
            .d = d,
            .q = q,
            .prep_fn = prep_fn,
            .prep_mod = prep_mod,
            .sc_alpha = emptyBuf(d),
            .sc_x16 = emptyBuf(d),
            .sc_xt = emptyBuf(d),
            .sc_rs = emptyBuf(d),
            .sc_part = emptyBuf(d),
            .sc_cnt = emptyBuf(d),
        };
    }

    fn emptyBuf(d: *const Driver) memory_.DeviceBuffer {
        return .{ .d = d, .ptr = 0, .len = 0 };
    }

    pub fn unload(self: *Lane) void {
        self.sc_alpha.free();
        self.sc_x16.free();
        self.sc_xt.free();
        self.sc_rs.free();
        self.sc_part.free();
        self.sc_cnt.free();
        self.q.unload();
        self.prep_mod.unload();
    }

    /// Grow `b` to at least `need` bytes, reusing it when it already fits. Reached only while not
    /// capturing (the warmup runs eagerly before capture), so cuMemAlloc is legal here.
    pub fn ensure(l: *Lane, b: *memory_.DeviceBuffer, need: usize) Error!void {
        if (b.len >= need) return;
        if (b.len > 0) b.free();
        b.* = try memory_.DeviceBuffer.alloc(l.d, need);
    }

    /// What one call launches, from the shapes alone.
    pub const Plan = struct {
        sk: u32, // K slices
        mt: u32, // the row tile (1, 2, or 4)
    };

    /// K slices (qmm.split_k884) and the row tile (qmm._row_tiles) for an (n, k) weight with m rows.
    pub fn plan(n: u32, k: u32, m: u32) Plan {
        const tiles: u32 = (n + COLS - 1) / COLS;
        const groups: u32 = k / 64;
        var sk: u32 = 1;
        while (sk < 16 and tiles * sk < TARGET and groups % (sk * 2) == 0 and groups / (sk * 2) >= 4) : (sk *= 2) {}
        const mt: u32 = if (m <= 8) 1 else if (m <= 16) 2 else 4;
        return .{ .sk = sk, .mt = mt };
    }

    /// The row-tile index matmul[fmt][mti] was built for (mt 1,2,4 -> mti 0,1,2).
    fn mti(mt: u32) u32 {
        return if (mt == 1) 0 else if (mt == 2) 1 else 2;
    }

    /// Slice partials (fp32 [sk][m][n]) a call with sk > 1 writes.
    pub fn partBytes(m: u32, n: u32, sk: u32) u64 {
        return @as(u64, m) * @as(u64, n) * @as(u64, sk) * 4;
    }

    /// The split-K counters (ints [gz][gx]) a call with sk > 1 uses, zero before the first launch.
    pub fn cntBytes(m: u32, n: u32, p: Plan) u64 {
        const gx: u32 = (n + COLS - 1) / COLS;
        const gz: u32 = (m + 8 * p.mt - 1) / (8 * p.mt);
        return @as(u64, gx) * @as(u64, gz) * 4;
    }

    /// y [m, n] = x16 [m, k] (fp16 fragment-order rows) times the tiled weight, fp32 sums.
    /// `rs` is the per-row scale, `alpha` the per-column factor, `w`/`scales` the tiled weight and its
    /// scales (fmt 0=fp4, 1=fp8, 2=fp16). `part`/`cnt` hold the slice partials and counters when sk > 1
    /// (zero the counters first); pass 0 for both when sk == 1. `f32` writes fp32 rows, else bf16.
    pub fn matmul(l: Lane, s: Stream, x16: u64, rs: u64, w: u64, scales: u64, alpha: u64, out: u64, part: u64, cnt: u64, m: u32, n: u32, k: u32, fmt: u32, out_f32: bool) Error!void {
        const p = Lane.plan(n, k, m);
        const f = l.q.matmul[fmt][Lane.mti(p.mt)];
        const gper: u32 = k / 64 / p.sk; // the kernel strides slices by the per-slice group count (sk == 1: all of K)
        const gx: u32 = (n + COLS - 1) / COLS;
        const gz: u32 = (m + 8 * p.mt - 1) / (8 * p.mt);
        var args: launch_.Args = .{};
        args.add(x16);
        args.add(rs);
        args.add(w);
        args.add(scales);
        args.add(alpha);
        args.add(out);
        args.add(if (p.sk > 1) part else @as(u64, 0));
        args.add(if (p.sk > 1) cnt else @as(u64, 0));
        args.add(@as(i32, @intCast(m)));
        args.add(@as(i32, @intCast(n)));
        args.add(@as(i32, @intCast(k)));
        args.add(@as(i32, @intCast(gper)));
        args.add(@as(i32, @intCast(p.sk)));
        args.add(if (out_f32) @as(bool, true) else @as(bool, false));
        try launch_.launch(f, .{
            .grid = .{ .x = gx, .y = p.sk, .z = gz },
            .block = .{ .x = THREADS, .y = 1, .z = 1 },
        }, s, &args);
    }

    /// y [m, n] = x [m, ldx] (raw bf16 rows, `ldx` apart, ldx % 8 == 0) times the packed bf16-lane
    /// weight `w` under the single factor `scale`, fp32 sums. The sm_70 dense path (tf_vbf16_*): both
    /// operands convert to fp16 in-kernel and accumulate on mma.m8n8k4.f16. `part`/`cnt` hold the slice
    /// partials and counters when sk > 1 (zero the counters first); pass 0 for both when sk == 1.
    /// `out_f32` writes fp32 rows, else bf16.
    pub fn matmulBf16(l: Lane, s: Stream, x: u64, ldx: u32, w: u64, scale: f32, out: u64, part: u64, cnt: u64, m: u32, n: u32, k: u32, out_f32: bool) Error!void {
        const p = Lane.plan(n, k, m);
        const f = l.q.bf16[Lane.mti(p.mt)];
        const gper: u32 = k / 64 / p.sk; // the kernel strides slices by the per-slice group count (sk == 1: all of K)
        const gx: u32 = (n + COLS - 1) / COLS;
        const gz: u32 = (m + 8 * p.mt - 1) / (8 * p.mt);
        var args: launch_.Args = .{};
        args.add(x);
        args.add(w);
        args.add(scale);
        args.add(out);
        args.add(if (p.sk > 1) part else @as(u64, 0));
        args.add(if (p.sk > 1) cnt else @as(u64, 0));
        args.add(@as(i32, @intCast(m)));
        args.add(@as(i32, @intCast(n)));
        args.add(@as(i32, @intCast(k)));
        args.add(@as(i32, @intCast(gper)));
        args.add(@as(i32, @intCast(p.sk)));
        args.add(@as(i32, @intCast(ldx)));
        args.add(if (out_f32) @as(i32, 1) else @as(i32, 0));
        try launch_.launch(f, .{
            .grid = .{ .x = gx, .y = p.sk, .z = gz },
            .block = .{ .x = THREADS, .y = 1, .z = 1 },
        }, s, &args);
    }

    /// Plain X [M][K] (fp16 rows) -> tiled layout (for the matmul kernel).
    /// grid = ceil(ceil(M/8)*K/256), block = 256; the destination is the full tiled shape, ceil(M/8)*K int4 (16 B each).
    pub fn tileX(l: Lane, s: Stream, x: u64, xt: u64, m: u32, k: u32) Error!void {
        const rowtiles: u64 = (@as(u64, m) + 7) / 8;
        const grid: u32 = @intCast((rowtiles * @as(u64, k) + 255) / 256);
        var args: launch_.Args = .{};
        args.add(x);
        args.add(xt);
        args.add(@as(i32, @intCast(m)));
        args.add(@as(i32, @intCast(k)));
        try launch_.launch(l.q.tile_x, .{
            .grid = .{ .x = grid, .y = 1, .z = 1 },
            .block = .{ .x = 256, .y = 1, .z = 1 },
        }, s, &args);
    }

    /// bf16 [m, k] rows (`ldx` apart) -> fp16 [m, k] row-major (`out`) and the row scales (`rs` [m]).
    /// One block per row (512 threads), the volta_qmm prep kernel.
    pub fn prep(l: Lane, s: Stream, x: u64, ldx: u32, m: u32, k: u32, out: u64, rs: u64) Error!void {
        var args: launch_.Args = .{};
        args.add(x);
        args.add(@as(i64, @intCast(ldx)));
        args.add(@as(i32, @intCast(m)));
        args.add(@as(i32, @intCast(k)));
        args.add(out);
        args.add(rs);
        try launch_.launch(l.prep_fn, .{
            .grid = .{ .x = m, .y = 1, .z = 1 },
            .block = .{ .x = 512, .y = 1, .z = 1 },
        }, s, &args);
    }

    /// The whole FP4 decode path: y [m, n] = x [m, k] (bf16 rows, `ldx` apart) times the packed FP4
    /// weight (`w` codes, `scales`), `alpha` the per-column factor, fp32 sums rounded to `out` (bf16,
    /// or fp32 when out_f32). Runs prep (bf16 -> fp16 + row scales), tile_x, and the tensor-core GEMM;
    /// the scratch buffers (fp16, tiled, row scales, slice partials + counters) are allocated here and
    /// freed before the call returns.
    pub fn fp4(l: Lane, s: Stream, x: u64, ldx: u32, m: u32, n: u32, k: u32, w: u64, scales: u64, alpha: u64, out: u64, out_f32: bool) Error!void {
        const p = Lane.plan(n, k, m);
        const xbytes: u64 = @as(u64, m) * @as(u64, k) * 2; // fp16 [m, k]
        const xtbytes: u64 = ((@as(u64, m) + 7) / 8) * @as(u64, k) * 16; // tiled: rowtiles*K int4 (16 B each)
        var x16 = try memory_.DeviceBuffer.alloc(l.d, @intCast(xbytes));
        var xt = try memory_.DeviceBuffer.alloc(l.d, @intCast(xtbytes));
        var rs = try memory_.DeviceBuffer.alloc(l.d, @as(usize, m) * 4); // float [m]
        var part = try memory_.DeviceBuffer.alloc(l.d, if (p.sk > 1) @intCast(Lane.partBytes(m, n, p.sk)) else 0);
        var cnt = try memory_.DeviceBuffer.alloc(l.d, if (p.sk > 1) @intCast(Lane.cntBytes(m, n, p)) else 0);
        errdefer cnt.free();
        errdefer part.free();
        errdefer rs.free();
        errdefer xt.free();
        errdefer x16.free();
        if (p.sk > 1) try cnt.fill32(0, s.handle);
        try l.prep(s, x, ldx, m, k, x16.ptr, rs.ptr);
        try l.tileX(s, x16.ptr, xt.ptr, m, k);
        try l.matmul(s, xt.ptr, rs.ptr, w, scales, alpha, out, if (p.sk > 1) part.ptr else 0, if (p.sk > 1) cnt.ptr else 0, m, n, k, 0, out_f32);
        cnt.free();
        part.free();
        rs.free();
        xt.free();
        x16.free();
    }
};

// ---------------------------------------------------------------------------
// experts_volta: the MoE lane (tf_vexp_*). prep_rows packs the routed inputs;
// the e32/e64 variants are expert_body<BM, WARPS, KGROUP> for the (32,1,0),
// (32,1,3), (32,2,2) and (64,1,0), (64,1,1), (64,1,3), (64,2,2) shapes.
// ---------------------------------------------------------------------------
pub const Experts = struct {
    mod: Module,
    prep_rows: Function,
    e32_1_0: Function,
    e32_1_3: Function,
    e32_2_2: Function,
    e64_1_0: Function,
    e64_1_1: Function,
    e64_1_3: Function,
    e64_2_2: Function,

    pub fn load(d: *const Driver) Error!Experts {
        var mod = try Module.load(d, kernels.experts_volta);
        return Experts{
            .mod = mod,
            .prep_rows = try mod.function("tf_vexp_prep_rows"),
            .e32_1_0 = try mod.function("tf_vexp_32_1_0"),
            .e32_1_3 = try mod.function("tf_vexp_32_1_3"),
            .e32_2_2 = try mod.function("tf_vexp_32_2_2"),
            .e64_1_0 = try mod.function("tf_vexp_64_1_0"),
            .e64_1_1 = try mod.function("tf_vexp_64_1_1"),
            .e64_1_3 = try mod.function("tf_vexp_64_1_3"),
            .e64_2_2 = try mod.function("tf_vexp_64_2_2"),
        };
    }

    pub fn unload(self: *Experts) void {
        self.mod.unload();
        self.* = undefined;
    }
};

// ---------------------------------------------------------------------------
// draft_attention_volta: the speculative draft attention (tf_vdraft_*).
// split_<T> runs the split-KV attention (128 or 64 threads); fold merges the
// split partials back into bf16 output rows.
// ---------------------------------------------------------------------------
pub const Draft = struct {
    mod: Module,
    split_128: Function,
    split_64: Function,
    fold: Function,

    pub fn load(d: *const Driver) Error!Draft {
        var mod = try Module.load(d, kernels.draft_attention_volta);
        return Draft{
            .mod = mod,
            .split_128 = try mod.function("tf_vdraft_split_128"),
            .split_64 = try mod.function("tf_vdraft_split_64"),
            .fold = try mod.function("tf_vdraft_fold"),
        };
    }

    pub fn unload(self: *Draft) void {
        self.mod.unload();
        self.* = undefined;
    }
};

// ---------------------------------------------------------------------------
// Symbol-table checks (comptime, no GPU): the names the loaders resolve must
// be exactly the strings the built fatbins export (verified with cuobjdump).
// ---------------------------------------------------------------------------
test "isVolta selects compute-capability major 7 only" {
    try std.testing.expect(isVolta(7));
    try std.testing.expect(!isVolta(8));
    try std.testing.expect(!isVolta(9));
    try std.testing.expect(!isVolta(12));
}

test "qmm matmul names match the built fatbin" {
    try std.testing.expectEqualStrings("tf_vqmm884_1_0_0", Qmm.matmulName(0, 0, 0));
    try std.testing.expectEqualStrings("tf_vqmm884_2_1_0", Qmm.matmulName(1, 1, 0));
    try std.testing.expectEqualStrings("tf_vqmm884_4_1_1", Qmm.matmulName(2, 1, 1));
}

test "qmmf matmul and dequant names match the built fatbin" {
    try std.testing.expectEqualStrings("tf_vqmmf884s_0_1", Qmmf.matmulName(0, 1));
    try std.testing.expectEqualStrings("tf_vqmmf884s_2_4", Qmmf.matmulName(2, 4));
    try std.testing.expectEqualStrings("tf_vqmmf884s_dequant_2", Qmmf.dequantName(2));
}

test "qmmf volta plan: K slices and row tiles from the shape alone" {
    // split_k884: enough blocks to fill 80 SMs, K groups stay divisible.
    try std.testing.expectEqual(@as(u32, 1), Lane.plan(128, 64, 1).sk);
    try std.testing.expectEqual(@as(u32, 1), Lane.plan(512, 256, 1).sk);
    try std.testing.expectEqual(@as(u32, 2), Lane.plan(1024, 512, 1).sk);
    try std.testing.expectEqual(@as(u32, 16), Lane.plan(4096, 4096, 1).sk);
    // _row_tiles: one for a few rows, up to four so a weight load feeds more tensor-core work.
    try std.testing.expectEqual(@as(u32, 1), Lane.plan(1024, 512, 1).mt);
    try std.testing.expectEqual(@as(u32, 1), Lane.plan(1024, 512, 8).mt);
    try std.testing.expectEqual(@as(u32, 2), Lane.plan(1024, 512, 9).mt);
    try std.testing.expectEqual(@as(u32, 2), Lane.plan(1024, 512, 16).mt);
    try std.testing.expectEqual(@as(u32, 4), Lane.plan(1024, 512, 17).mt);
    try std.testing.expectEqual(@as(u32, 4), Lane.plan(1024, 512, 300).mt);
    // the split-K buffers, in bytes.
    const p = Lane.plan(1024, 512, 1);
    try std.testing.expectEqual(@as(u64, 8192), Lane.partBytes(1, 1024, p.sk));
    try std.testing.expectEqual(@as(u64, 32), Lane.cntBytes(1, 1024, p));
}
