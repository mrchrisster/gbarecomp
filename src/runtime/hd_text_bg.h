// HD redraw of a per-line warped text background (Mother 3 battle backgrounds).
//
// The game builds its swirling battle backgrounds from two 256x256 text BGs
// (BG2 over BG3, alpha blended) whose HOFS/VOFS are rewritten every scanline by
// HBlank DMA. At native resolution the wave can only move in whole pixels per
// line, so curved edges become staircases once the 240x160 frame is magnified
// in a headset. This redraws those two layers at the output scale S:
//   * the per-line offsets are smoothed and interpolated between lines, so the
//     wave is continuous at sub-line precision,
//   * texels are still sampled nearest, so flat colours and hard pixel edges
//     stay exactly as the game draws them (no blur, no new colours),
//   * colours (palette cycling, fades, blend) come from the frame itself.
// Only pixels whose native colour matches a re-composition of those two layers
// are replaced; sprites, text boxes, windows and anything unexplained keep the
// game's own pixel, magnified nearest. Host-only: never touches emulation.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <vector>

#include "gba_color.h"

namespace gbarecomp::hdbg {

constexpr int kLines = 160;

struct Line {
    uint16_t dispcnt = 0;
    uint16_t bgcnt[2] = {};  // BG2CNT, BG3CNT
    uint16_t hofs[2] = {};
    uint16_t vofs[2] = {};
    uint16_t bldcnt = 0, bldalpha = 0, bldy = 0;
};

struct Frame {
    uint64_t seq = 0;
    bool active = false;
    int view_x = 0, view_y = 0;  // native picture's offset inside the render frame
    std::array<Line, kLines> lines{};
    std::array<uint16_t, 256> pal{};
    std::vector<uint8_t> vram;  // first 64 KiB (BG area)
};

// ---- capture (emulation thread) -> publish -> snapshot (host thread) ----
struct Shared {
    std::mutex mtx;
    Frame published;
    std::array<Line, kLines> work{};
};
inline Shared& shared() {
    static Shared s;
    return s;
}

inline uint16_t rd16(const uint8_t* io, unsigned a) {
    return static_cast<uint16_t>(io[a] | (io[a + 1] << 8));
}

// Call before native line y is drawn, with the IO page it will be drawn with.
inline void capture_line(unsigned y, const uint8_t* io) {
    if (y >= static_cast<unsigned>(kLines) || !io) return;
    Line& l = shared().work[y];
    l.dispcnt = rd16(io, 0x00);
    l.bgcnt[0] = rd16(io, 0x0C);
    l.bgcnt[1] = rd16(io, 0x0E);
    l.hofs[0] = rd16(io, 0x18) & 0x1FF;
    l.vofs[0] = rd16(io, 0x1A) & 0x1FF;
    l.hofs[1] = rd16(io, 0x1C) & 0x1FF;
    l.vofs[1] = rd16(io, 0x1E) & 0x1FF;
    l.bldcnt = rd16(io, 0x50);
    l.bldalpha = rd16(io, 0x52);
    l.bldy = rd16(io, 0x54);
}

// Call once the frame is complete. When inactive, only the flag is updated.
inline void publish(bool active, const uint8_t* vram, const uint8_t* pal, int view_x, int view_y) {
    Shared& s = shared();
    std::lock_guard<std::mutex> lk(s.mtx);
    ++s.published.seq;
    s.published.active = active && vram && pal;
    if (!s.published.active) return;
    s.published.view_x = view_x;
    s.published.view_y = view_y;
    s.published.lines = s.work;
    s.published.vram.assign(vram, vram + 0x10000);
    std::memcpy(s.published.pal.data(), pal, 512);
}

// Copies the latest published frame; false when there is no active frame.
inline bool snapshot(Frame& out) {
    Shared& s = shared();
    std::lock_guard<std::mutex> lk(s.mtx);
    if (!s.published.active) {
        out.active = false;
        return false;
    }
    out = s.published;
    return true;
}

// ---- renderer ----
class Renderer {
public:
    // native_topdown: the frame (or depth plane) as handed to the host, RGBA,
    //   top row first, iw x ih. Used only to decide which pixels are background.
    // graded_bottomup: the same image after host colour grading, bottom row
    //   first (what would otherwise be uploaded). Used for every other pixel.
    // lut555: optional 32768-entry table, BGR555 -> graded RGBA bytes (alpha 255).
    // out_bottomup: (iw*S) x (ih*S) RGBA, bottom row first.
    // Returns false (out untouched) when the frame is not a supported layout.
    bool render(const Frame& f, const uint8_t* native_topdown, const uint8_t* graded_bottomup, int iw,
                int ih, int S, const uint32_t* lut555, uint8_t* out_bottomup) {
        if (!f.active || iw <= 0 || ih <= 0 || S < 1 || f.vram.size() < 0x10000) return false;
        if (f.seq != decoded_seq_ || !decoded_ok_) {
            decoded_ok_ = decode_layers(f);
            if (decoded_ok_) build_offsets(f);
            decoded_seq_ = f.seq;
        }
        if (!decoded_ok_) return false;
        build_mask(f, native_topdown, iw, ih);
        if (!any_bg_) return false;

        const int W = iw * S, H = ih * S;
        const uint32_t* graded = reinterpret_cast<const uint32_t*>(graded_bottomup);
        uint32_t* out = reinterpret_cast<uint32_t*>(out_bottomup);
        const double inv_s = 1.0 / S;
        for (int R = 0; R < H; ++R) {
            const int r = R / S;                       // render row
            uint32_t* orow = out + static_cast<size_t>(H - 1 - R) * W;
            const uint32_t* grow = graded + static_cast<size_t>(ih - 1 - r) * iw;
            const uint8_t* mrow = mask_.data() + static_cast<size_t>(r) * iw;
            const int y = r - f.view_y;                // native line
            const bool row_bg = y >= 0 && y < kLines && row_any_[r];
            if (!row_bg) {
                for (int x = 0; x < iw; ++x) std::fill_n(orow + x * S, S, grow[x]);
                continue;
            }
            const Line& ln = f.lines[y];
            // Sub-line position of this output row: line centres sit at integers.
            const double t = y + ((R % S) + 0.5) * inv_s - 0.5;
            int srow[2];
            double hsh[2];
            for (int k = 0; k < 2; ++k) {
                const double u = sample_curve(u_[k], seg_[k], y, t);
                const double h = sample_curve(h_[k], seg_[k], y, t);
                srow[k] = static_cast<int>(std::floor(u + 0.5)) & 255;
                hsh[k] = h - f.view_x;
            }
            const uint16_t* row2 = layer_[0].data() + srow[0] * 256;
            const uint16_t* row3 = layer_[1].data() + srow[1] * 256;
            // Fixed point (16.16) source column per output pixel.
            const int64_t step = static_cast<int64_t>(65536.0 * inv_s + 0.5);
            int64_t c2 = static_cast<int64_t>(std::floor((0.5 * inv_s + hsh[0]) * 65536.0));
            int64_t c3 = static_cast<int64_t>(std::floor((0.5 * inv_s + hsh[1]) * 65536.0));
            const Comp comp = line_comp(ln, f.pal[0]);
            uint16_t last_a = 0, last_b = 0;
            uint32_t last_px = 0;
            {
                const uint16_t c0 = compose(comp, 0, 0);
                last_px = lut555 ? lut555[c0 & 0x7FFF] : pack(c0);
            }
            uint32_t* o = orow;
            for (int x = 0; x < iw; ++x, o += S) {
                if (!mrow[x]) {
                    std::fill_n(o, S, grow[x]);
                    c2 += step * S;
                    c3 += step * S;
                    continue;
                }
                for (int s = 0; s < S; ++s, c2 += step, c3 += step) {
                    const uint16_t a = row2[(c2 >> 16) & 255];
                    const uint16_t b = row3[(c3 >> 16) & 255];
                    // Most texels repeat their neighbour: reuse the last composed colour.
                    if (a != last_a || b != last_b) {
                        last_a = a;
                        last_b = b;
                        const uint16_t c = compose(comp, a, b);
                        last_px = lut555 ? lut555[c & 0x7FFF] : pack(c);
                    }
                    o[s] = last_px;
                }
            }
        }
        return true;
    }

private:
    static constexpr uint16_t kClear = 0xFFFF;

    struct Comp {
        bool on[2];
        int top;  // 0 = BG2 above BG3, 1 = BG3 above BG2
        int effect;
        bool t1[2], t2[2], t2_back;
        uint32_t eva, evb, evy;
        uint16_t back;
    };

    static uint32_t pack(uint16_t c) {
        uint8_t rgb[3];
        gba::color::to_rgb888(c, rgb);
        return static_cast<uint32_t>(rgb[0]) | (static_cast<uint32_t>(rgb[1]) << 8) |
               (static_cast<uint32_t>(rgb[2]) << 16) | 0xFF000000u;
    }

    static Comp line_comp(const Line& l, uint16_t backdrop) {
        Comp c{};
        c.on[0] = (l.dispcnt >> 10) & 1;
        c.on[1] = (l.dispcnt >> 11) & 1;
        const int p2 = l.bgcnt[0] & 3, p3 = l.bgcnt[1] & 3;
        c.top = (p3 < p2) ? 1 : 0;
        c.effect = (l.bldcnt >> 6) & 3;
        c.t1[0] = (l.bldcnt >> 2) & 1;
        c.t1[1] = (l.bldcnt >> 3) & 1;
        c.t2[0] = (l.bldcnt >> 10) & 1;
        c.t2[1] = (l.bldcnt >> 11) & 1;
        c.t2_back = (l.bldcnt >> 13) & 1;
        c.eva = std::min<uint32_t>(l.bldalpha & 31, 16);
        c.evb = std::min<uint32_t>((l.bldalpha >> 8) & 31, 16);
        c.evy = std::min<uint32_t>(l.bldy & 31, 16);
        c.back = backdrop & 0x7FFF;
        return c;
    }

    // a = BG2 texel, b = BG3 texel (kClear when transparent).
    static uint16_t compose(const Comp& c, uint16_t a, uint16_t b) {
        const uint16_t px[2] = {c.on[0] ? a : kClear, c.on[1] ? b : kClear};
        const int first = c.top, second = 1 - c.top;
        int top_layer = -1;  // 0/1, or 2 = backdrop
        uint16_t top = 0, below = 0;
        bool below_t2 = false;
        if (px[first] != kClear) {
            top_layer = first;
            top = px[first];
            if (px[second] != kClear) {
                below = px[second];
                below_t2 = c.t2[second];
            } else {
                below = c.back;
                below_t2 = c.t2_back;
            }
        } else if (px[second] != kClear) {
            top_layer = second;
            top = px[second];
            below = c.back;
            below_t2 = c.t2_back;
        } else {
            return c.back;
        }
        if (!c.t1[top_layer]) return top;
        switch (c.effect) {
            case 1: return below_t2 ? gba::color::blend_alpha_gba555(top, below, c.eva, c.evb) : top;
            case 2: return gba::color::brighten_gba555(top, c.evy);
            case 3: return gba::color::darken_gba555(top, c.evy);
            default: return top;
        }
    }

    bool decode_layers(const Frame& f) {
        const Line& ref = f.lines[kLines / 2];
        if ((ref.dispcnt & 7) != 0) return false;  // text mode only
        for (int k = 0; k < 2; ++k) {
            const uint16_t cnt = ref.bgcnt[k];
            if ((cnt >> 14) != 0) return false;  // 256x256 maps only
            const uint32_t cb = ((cnt >> 2) & 3) * 0x4000u;
            const uint32_t sb = ((cnt >> 8) & 31) * 0x800u;
            const bool c8 = (cnt >> 7) & 1;
            layer_[k].resize(256 * 256);
            uint16_t* dst = layer_[k].data();
            const uint8_t* v = f.vram.data();
            for (int ty = 0; ty < 32; ++ty)
                for (int tx = 0; tx < 32; ++tx) {
                    const uint32_t eo = sb + (ty * 32 + tx) * 2;
                    if (eo + 1 >= 0x10000) return false;
                    const uint16_t e = static_cast<uint16_t>(v[eo] | (v[eo + 1] << 8));
                    const uint32_t tile = e & 0x3FF;
                    const bool hf = (e >> 10) & 1, vf = (e >> 11) & 1;
                    const uint32_t pb = (e >> 12) & 15;
                    for (int py = 0; py < 8; ++py) {
                        const int sy = vf ? 7 - py : py;
                        for (int px = 0; px < 8; ++px) {
                            const int sx = hf ? 7 - px : px;
                            uint32_t ci;
                            if (c8) {
                                const uint32_t o = cb + tile * 64 + sy * 8 + sx;
                                ci = o < 0x10000 ? v[o] : 0;
                            } else {
                                const uint32_t o = cb + tile * 32 + sy * 4 + (sx >> 1);
                                ci = o < 0x10000 ? ((v[o] >> ((sx & 1) * 4)) & 15) : 0;
                                if (ci) ci += pb * 16;
                            }
                            dst[(ty * 8 + py) * 256 + tx * 8 + px] =
                                ci ? static_cast<uint16_t>(f.pal[ci] & 0x7FFF) : kClear;
                        }
                    }
                }
        }
        return true;
    }

    // Per layer: h(y) = HOFS, u(y) = y + VOFS (the source row line y shows),
    // unwrapped, then lightly smoothed within runs of lines that belong to one
    // continuous wave. Runs break where the layer is off, its map changes, or
    // the offsets jump; a run that alternates line to line (an interlaced
    // effect) is left unsmoothed so its look is kept.
    void build_offsets(const Frame& f) {
        for (int k = 0; k < 2; ++k) {
            double rh[kLines], ru[kLines];
            int seg = 0;
            for (int y = 0; y < kLines; ++y) {
                const Line& l = f.lines[y];
                const double h = l.hofs[k], u = y + l.vofs[k];
                bool brk = (y == 0);
                if (!brk) {
                    const Line& p = f.lines[y - 1];
                    const double dh = wrap256(h - raw_h_[k][y - 1]);
                    const double du = wrap256(u - raw_u_[k][y - 1]);
                    brk = p.bgcnt[k] != l.bgcnt[k] || ((p.dispcnt ^ l.dispcnt) & (0x400 << k)) ||
                          std::fabs(dh) > 24 || std::fabs(du - 1) > 24;
                    if (!brk) {
                        rh[y] = rh[y - 1] + dh;
                        ru[y] = ru[y - 1] + du;
                    }
                }
                if (brk) {
                    if (y) ++seg;
                    rh[y] = h;
                    ru[y] = u;
                }
                raw_h_[k][y] = h;
                raw_u_[k][y] = u;
                seg_[k][y] = seg;
            }
            smooth(rh, seg_[k], h_[k]);
            smooth(ru, seg_[k], u_[k]);
        }
    }

    static double wrap256(double d) {
        d = std::fmod(d + 128.0, 256.0);
        if (d < 0) d += 256.0;
        return d - 128.0;
    }

    static void smooth(const double* in, const int* seg, double* out) {
        constexpr double kSigma = 1.2;
        constexpr int kR = 4;
        double w[kR + 1];
        for (int i = 0; i <= kR; ++i) w[i] = std::exp(-0.5 * (i / kSigma) * (i / kSigma));
        int y0 = 0;
        while (y0 < kLines) {
            int y1 = y0;
            while (y1 + 1 < kLines && seg[y1 + 1] == seg[y0]) ++y1;
            // Interlaced / jittery runs keep their raw offsets.
            double jitter = 0;
            int n2 = 0;
            for (int y = y0 + 1; y < y1; ++y, ++n2)
                jitter += std::fabs(in[y + 1] - 2 * in[y] + in[y - 1]);
            const bool keep = n2 > 0 && jitter / n2 > 2.5;
            for (int y = y0; y <= y1; ++y) {
                if (keep) {
                    out[y] = in[y];
                    continue;
                }
                double s = 0, ws = 0;
                for (int d = -kR; d <= kR; ++d) {
                    const int yy = y + d;
                    if (yy < y0 || yy > y1) continue;
                    s += in[yy] * w[std::abs(d)];
                    ws += w[std::abs(d)];
                }
                out[y] = s / ws;
            }
            y0 = y1 + 1;
        }
    }

    // Catmull-Rom through the smoothed values of line y's run, at position t.
    static double sample_curve(const double* v, const int* seg, int y, double t) {
        int i = static_cast<int>(std::floor(t));
        const double fr = t - i;
        auto at = [&](int j) {
            j = std::clamp(j, 0, kLines - 1);
            // stay inside line y's run: clamp toward y
            if (seg[j] != seg[y]) {
                if (j > y) {
                    while (j > y && seg[j] != seg[y]) --j;
                } else {
                    while (j < y && seg[j] != seg[y]) ++j;
                }
            }
            return v[j];
        };
        const double p0 = at(i - 1), p1 = at(i), p2 = at(i + 1), p3 = at(i + 2);
        return p1 + 0.5 * fr * (p2 - p0 + fr * (2 * p0 - 5 * p1 + 4 * p2 - p3 + fr * (3 * (p1 - p2) + p3 - p0)));
    }

    // A pixel is "background" when the native image shows exactly what the two
    // layers compose to with that line's own (unsmoothed) offsets.
    void build_mask(const Frame& f, const uint8_t* native, int iw, int ih) {
        mask_.assign(static_cast<size_t>(iw) * ih, 0);
        row_any_.assign(ih, 0);
        any_bg_ = false;
        for (int r = 0; r < ih; ++r) {
            const int y = r - f.view_y;
            if (y < 0 || y >= kLines) continue;
            const Line& l = f.lines[y];
            const Comp comp = line_comp(l, f.pal[0]);
            if (!comp.on[0] && !comp.on[1]) continue;
            const uint16_t* row2 = layer_[0].data() + ((y + l.vofs[0]) & 255) * 256;
            const uint16_t* row3 = layer_[1].data() + ((y + l.vofs[1]) & 255) * 256;
            const uint8_t* nrow = native + static_cast<size_t>(r) * iw * 4;
            uint8_t* mrow = mask_.data() + static_cast<size_t>(r) * iw;
            int count = 0;
            for (int x = 0; x < iw; ++x) {
                if (nrow[x * 4 + 3] != 255) continue;
                const int hx = x - f.view_x;
                const uint16_t a = row2[(hx + l.hofs[0]) & 255];
                const uint16_t b = row3[(hx + l.hofs[1]) & 255];
                if (a == kClear && b == kClear) continue;  // backdrop only: leave it
                uint8_t rgb[3];
                gba::color::to_rgb888(compose(comp, a, b), rgb);
                if (rgb[0] == nrow[x * 4] && rgb[1] == nrow[x * 4 + 1] && rgb[2] == nrow[x * 4 + 2]) {
                    mrow[x] = 1;
                    ++count;
                }
            }
            row_any_[r] = count > 0;
            any_bg_ |= count > 0;
        }
    }

    std::vector<uint16_t> layer_[2];
    double raw_h_[2][kLines] = {}, raw_u_[2][kLines] = {};
    double h_[2][kLines] = {}, u_[2][kLines] = {};
    int seg_[2][kLines] = {};
    std::vector<uint8_t> mask_, row_any_;
    bool any_bg_ = false;
    uint64_t decoded_seq_ = ~0ull;
    bool decoded_ok_ = false;
};

}  // namespace gbarecomp::hdbg
