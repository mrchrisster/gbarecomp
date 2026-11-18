// STOCK 3D building / roof fix, shared by the Quest OpenXR host and the desktop layer view
// (GBARECOMP_LAYER_VIEW) so both show exactly the same layers. Input: the PPU's stock planes
// (plane = 4 - priority) and the semantic 5-plane frame with its alpha side channel
// (254 solid, 253 authored, 252 both). See docs/QUEST_VR_3D_ARCHITECTURE.md Section 10.
// Interiors (gba::g_layered_interior) stay pure stock apart from authored overrides.
#pragma once
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>
#include "gba_ppu.h"

namespace gbarecomp {

struct Stock3dFix {
    std::vector<uint8_t> stock_fixed;  // STOCK 3D planes after the building fix
    std::vector<int8_t> fix_sem, fix_top;
    std::vector<int> fix_integral;
    std::vector<uint8_t> fix_state;
    std::vector<int> fix_queue, fix_piece;
    std::vector<int8_t> fix_top2;
    std::vector<uint8_t> fix_solid, fix_wall;
    std::vector<int> fix_run, fix_coltop;
    std::vector<uint32_t> void_col;
    double last_ms = 0.0;

    // Returns the fixed planes (valid until the next call).
    const uint8_t* run(const uint8_t* stock, const uint8_t* masked_ptr, int iw, int ih, unsigned used_planes) {
        constexpr unsigned kLayeredPlaneCount = 5;
        const size_t plane_bytes = static_cast<size_t>(iw) * ih * 4u;
        const uint8_t* plane_src = stock;
        Stock3dFix& b = *this;
        const auto fix_t0 = std::chrono::steady_clock::now();
        b.stock_fixed.assign(plane_src, plane_src + used_planes * plane_bytes);
        const size_t n_px = static_cast<size_t>(iw) * ih;
        uint8_t* fx = b.stock_fixed.data();
        // Interiors stay pure STOCK 3D (furniture, mirrors, characters as the game layers
        // them); only hand-authored overrides below apply there.
        const bool interior = gba::g_layered_interior != 0;
        if (!interior) {
        // per-pixel top plane of the semantic and stock stacks
        b.fix_sem.assign(n_px, -1);
        b.fix_top.assign(n_px, -1);
        for (size_t i = 0; i < n_px; ++i) {
            const size_t o = i * 4 + 3;
            for (int q = static_cast<int>(kLayeredPlaneCount) - 1; q >= 0; --q)
                if (masked_ptr[q * plane_bytes + o]) { b.fix_sem[i] = static_cast<int8_t>(q); break; }
            for (int q = static_cast<int>(used_planes) - 1; q >= 0; --q)
                if (fx[q * plane_bytes + o]) { b.fix_top[i] = static_cast<int8_t>(q); break; }
        }
        // Classifier promote/demote (semantic Plane 2 up, stray stock Plane 2 down) is OFF:
        // owner wants stock layers everywhere except the roof fix + hand-authored fixes
        // (Nov 17 2026). Kept for reference.
        constexpr bool kClassifierPromote = false;
        if (kClassifierPromote) {
        // integral image of "stock Plane 2 that the classifier agrees with"
        const int W1 = iw + 1;
        b.fix_integral.assign(static_cast<size_t>(W1) * (ih + 1), 0);
        for (int y = 0; y < ih; ++y) {
            int row = 0;
            for (int x = 0; x < iw; ++x) {
                const size_t i = static_cast<size_t>(y) * iw + x;
                row += (b.fix_top[i] == 2 && b.fix_sem[i] == 2) ? 1 : 0;
                b.fix_integral[static_cast<size_t>(y + 1) * W1 + x + 1] =
                    b.fix_integral[static_cast<size_t>(y) * W1 + x + 1] + row;
            }
        }
        constexpr int kR = 12;
        auto agreed_plane2_share_high = [&](int x, int y) {
            const int x0 = std::max(0, x - kR), x1 = std::min(iw, x + kR + 1);
            const int y0 = std::max(0, y - kR), y1 = std::min(ih, y + kR + 1);
            const int keep = b.fix_integral[static_cast<size_t>(y1) * W1 + x1] -
                             b.fix_integral[static_cast<size_t>(y0) * W1 + x1] -
                             b.fix_integral[static_cast<size_t>(y1) * W1 + x0] +
                             b.fix_integral[static_cast<size_t>(y0) * W1 + x0];
            return keep * 10 >= (x1 - x0) * (y1 - y0) * 3;
        };
        // Reverse case candidates: stock Plane 2 where the classifier says floor.
        // 0 = not a candidate, 1 = candidate, 2 = protected (inside agreed Plane 2,
        // e.g. a coarse 16x16 classifier block in a tree canopy), 3 = demote.
        b.fix_state.assign(n_px, 0);
        for (int y = 0; y < ih; ++y)
            for (int x = 0; x < iw; ++x) {
                const size_t i = static_cast<size_t>(y) * iw + x;
                const int sem = b.fix_sem[i];
                if (b.fix_top[i] == 2 && (sem == 0 || sem == 1))
                    b.fix_state[i] = agreed_plane2_share_high(x, y) ? 2 : 1;
            }
        // Each remaining connected piece goes down unless it stands on something the
        // classifier calls an object (a burning tree's top on its trunk) or on a
        // protected block. Roof edges and door arches stand on walls/floor -> down.
        b.fix_queue.clear();
        b.fix_piece.clear();
        for (size_t s0 = 0; s0 < n_px; ++s0) {
            if (b.fix_state[s0] != 1) continue;
            b.fix_piece.clear();
            b.fix_queue.assign(1, static_cast<int>(s0));
            b.fix_state[s0] = 4;  // visited
            bool supported = false;
            for (size_t qi = 0; qi < b.fix_queue.size(); ++qi) {
                const int p = b.fix_queue[qi];
                b.fix_piece.push_back(p);
                const int py = p / iw, px = p % iw;
                if (py + 1 < ih) {
                    const int below = p + iw;
                    const uint8_t st = b.fix_state[below];
                    if (st != 1 && st != 4 && (b.fix_sem[below] == 2 || st == 2)) supported = true;
                }
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx) {
                        const int yy = py + dy, xx = px + dx;
                        if (yy < 0 || yy >= ih || xx < 0 || xx >= iw) continue;
                        const int q = yy * iw + xx;
                        if (b.fix_state[q] == 1) { b.fix_state[q] = 4; b.fix_queue.push_back(q); }
                    }
            }
            if (!supported)
                for (int p : b.fix_piece) b.fix_state[p] = 3;
        }
        for (size_t i = 0; i < n_px; ++i) {
            const int sem = b.fix_sem[i], top = b.fix_top[i];
            if (sem == 2) {
                if (top < 0 || top >= 2) continue;
                std::memcpy(fx + 2 * plane_bytes + i * 4, fx + top * plane_bytes + i * 4, 4);
            } else if (b.fix_state[i] == 3) {
                // stray building piece (roof edge, arch): down onto the floor plane so
                // head motion never shows the texture under it as thin lines
                std::memcpy(fx + sem * plane_bytes + i * 4, fx + 2 * plane_bytes + i * 4, 4);
                std::memset(fx + 2 * plane_bytes + i * 4, 0, 4);
            }
        }
        }  // kClassifierPromote
        // Houses: "roof stands out, walls stay on the ground" (owner's choice,
        // Nov 16 2026). Mother 3 paints roofs mostly on BG2 (stock Plane 2).
        // Roof = wide (>= 48 px), tall (>= 24 px) stock-Plane-2 pieces (>= 48 solid px) the
        // fix kept standing, with open walkable ground above most of their top
        // edge (indoor shelves sit under more wall). Holes inside one roof are
        // filled row-wise when the roof bounds them below within 48 px; solid
        // pixels from the roof's lower edge down to walkable ground (<= 128 px)
        // go flat when they form a wide (>= 24 px) wall. Mirrors
        // tools/map_sweep/wall_rule.py roof_only(). Collision comes from the PPU
        // side channel: alpha 254 on the top opaque semantic pixel of a solid cell.
        {
            constexpr int kMinRoofW = 48, kMinRoofH = 24, kMaxWall = 128, kMinWall = 24,
                          kEave = 48, kMaxGap = 96;
            const int npx = static_cast<int>(n_px);
            auto top_of = [&](size_t i) -> int {
                const size_t o = i * 4 + 3;
                for (int q = static_cast<int>(used_planes) - 1; q >= 0; --q)
                    if (fx[q * plane_bytes + o]) return q;
                return -1;
            };
            b.fix_top2.assign(n_px, -1);
            b.fix_solid.assign(n_px, 0);
            for (size_t i = 0; i < n_px; ++i) {
                b.fix_top2[i] = static_cast<int8_t>(top_of(i));
                const size_t o = i * 4 + 3;
                for (int q = static_cast<int>(kLayeredPlaneCount) - 1; q >= 0; --q) {
                    const uint8_t a = masked_ptr[q * plane_bytes + o];
                    if (a) { b.fix_solid[i] = (a == 254u || a == 252u) ? 1 : 0; break; }
                }
            }
            // roof seeds and their components
            b.fix_wall.assign(n_px, 0);  // 1 = seed, 2 = roof
            for (size_t i = 0; i < n_px; ++i)
                if (b.fix_top[i] == 2 && b.fix_top2[i] >= 2) b.fix_wall[i] = 1;
            b.fix_run.assign(n_px, 0);  // roof component id (1-based), 0 = none
            b.fix_coltop.assign(iw, 0);
            int comp_id = 0;
            for (int s0 = 0; s0 < npx; ++s0) {
                if (b.fix_wall[s0] != 1 || b.fix_run[s0]) continue;
                ++comp_id;
                b.fix_piece.clear();
                b.fix_queue.assign(1, s0);
                b.fix_run[s0] = comp_id;
                int minx = iw, maxx = -1, miny = ih, maxy = -1;
                for (size_t qi = 0; qi < b.fix_queue.size(); ++qi) {
                    const int p = b.fix_queue[qi];
                    b.fix_piece.push_back(p);
                    const int py = p / iw, px = p % iw;
                    minx = std::min(minx, px); maxx = std::max(maxx, px);
                    miny = std::min(miny, py); maxy = std::max(maxy, py);
                    for (int dy = -1; dy <= 1; ++dy)
                        for (int dx = -1; dx <= 1; ++dx) {
                            const int yy = py + dy, xx = px + dx;
                            if (yy < 0 || yy >= ih || xx < 0 || xx >= iw) continue;
                            const int q = yy * iw + xx;
                            if (b.fix_wall[q] == 1 && !b.fix_run[q]) { b.fix_run[q] = comp_id; b.fix_queue.push_back(q); }
                        }
                }
                // the walkable-ground edge of a roof is solid; its upper part often is not
                // (beyond the walkable area), so a piece needs some collision, not all
                int solid_px = 0;
                for (int p : b.fix_piece) solid_px += b.fix_solid[p] ? 1 : 0;
                bool keep = (maxx - minx + 1) >= kMinRoofW && (maxy - miny + 1) >= kMinRoofH &&
                            solid_px >= kMinRoofW;
                if (keep) {
                    for (int x = minx; x <= maxx; ++x) b.fix_coltop[x] = ih;
                    for (int p : b.fix_piece) {
                        const int py = p / iw, px = p % iw;
                        b.fix_coltop[px] = std::min(b.fix_coltop[px], py);
                    }
                    int cols = 0, open = 0;
                    for (int x = minx; x <= maxx; ++x) {
                        const int ty = b.fix_coltop[x];
                        if (ty >= ih) continue;
                        ++cols;
                        // a roof running off the top of the view (or into the letterbox
                        // band / HUD on Plane 4, or nothing drawn) counts as open sky above
                        if (ty == 0) { ++open; continue; }
                        const size_t a = static_cast<size_t>(ty - 1) * iw + x;
                        if (b.fix_top2[a] < 0 || b.fix_top2[a] >= 4 || !b.fix_solid[a]) ++open;
                    }
                    keep = open * 2 >= cols;
                }
                if (keep) for (int p : b.fix_piece) b.fix_wall[p] = 2;
            }
            // roof below within kEave px (per column, scanning bottom-up)
            b.fix_state.assign(n_px, 0);  // 1 = roof within kEave below
            for (int x = 0; x < iw; ++x) {
                int last = -100000;
                for (int y = ih - 1; y >= 0; --y) {
                    const size_t i = static_cast<size_t>(y) * iw + x;
                    if (b.fix_wall[i] == 2) last = y;
                    else if (last - y <= kEave) b.fix_state[i] = 1;
                }
            }
            // fill holes inside one roof, row-wise
            for (int y = 0; y < ih; ++y) {
                int prev = -1;
                for (int x = 0; x < iw; ++x) {
                    const size_t i = static_cast<size_t>(y) * iw + x;
                    if (b.fix_wall[i] != 2) continue;
                    if (prev >= 0 && x - prev > 1 && x - prev <= kMaxGap &&
                        b.fix_run[static_cast<size_t>(y) * iw + prev] == b.fix_run[i]) {
                        bool ok = true;
                        for (int k = prev + 1; k < x && ok; ++k) {
                            const size_t j = static_cast<size_t>(y) * iw + k;
                            ok = b.fix_solid[j] && b.fix_state[j];
                        }
                        if (ok)
                            for (int k = prev + 1; k < x; ++k) {
                                const size_t j = static_cast<size_t>(y) * iw + k;
                                if (b.fix_wall[j] != 2) b.fix_wall[j] = 3;  // filled roof
                            }
                    }
                    prev = x;
                }
            }
            // raise filled roof pixels
            for (size_t i = 0; i < n_px; ++i) {
                if (b.fix_wall[i] != 3) continue;
                const int top = b.fix_top2[i];
                if (top >= 0 && top < 2) {
                    std::memcpy(fx + 2 * plane_bytes + i * 4, fx + top * plane_bytes + i * 4, 4);
                    b.fix_top2[i] = 2;
                }
                b.fix_wall[i] = 2;
            }
            // walls under roofs: standing solid pixels from the roof's lower edge
            // down to walkable ground
            for (int x = 0; x < iw; ++x) {
                int y = 0;
                while (y < ih - 1) {
                    const size_t i = static_cast<size_t>(y) * iw + x, j = i + iw;
                    if (b.fix_wall[i] == 2 && b.fix_wall[j] != 2 && b.fix_solid[j]) {
                        int y1 = y + 1;
                        while (y1 < ih && y1 - y <= kMaxWall) {
                            const size_t k = static_cast<size_t>(y1) * iw + x;
                            if (!b.fix_solid[k] || b.fix_wall[k] == 2 || b.fix_top[k] >= 4) break;
                            ++y1;
                        }
                        // ends on walkable ground, or runs out of the view (HUD band / bottom)
                        const size_t e = static_cast<size_t>(std::min(y1, ih - 1)) * iw + x;
                        if (y1 >= ih || b.fix_top[e] >= 4 ||
                            (!b.fix_solid[e] && b.fix_wall[e] != 2)) {
                            for (int k = y + 1; k < y1; ++k) {
                                const size_t m = static_cast<size_t>(k) * iw + x;
                                if (b.fix_top2[m] >= 2 && b.fix_top2[m] < 4 && b.fix_top[m] < 2)
                                    b.fix_wall[m] = 4;  // wall candidate; BG2 details (signs) keep standing
                            }
                            // the house's own walkable front (porch, steps) right below the wall:
                            // lifted by the classifier, not by the game -> flat as well
                            for (int k = y1; k < ih && k - y1 <= kEave; ++k) {
                                const size_t m = static_cast<size_t>(k) * iw + x;
                                if (b.fix_top[m] >= 4) break;
                                if (b.fix_top2[m] >= 2 && b.fix_top2[m] < 4 && b.fix_top[m] < 2) b.fix_wall[m] = 4;
                            }
                        }
                        y = y1;
                    } else {
                        ++y;
                    }
                }
            }
            for (int s0 = 0; s0 < npx; ++s0) {
                if (b.fix_wall[s0] != 4) continue;
                b.fix_piece.clear();
                b.fix_queue.assign(1, s0);
                b.fix_wall[s0] = 5;
                int minx = iw, maxx = -1;
                for (size_t qi = 0; qi < b.fix_queue.size(); ++qi) {
                    const int p = b.fix_queue[qi];
                    b.fix_piece.push_back(p);
                    const int py = p / iw, px = p % iw;
                    minx = std::min(minx, px); maxx = std::max(maxx, px);
                    for (int dy = -1; dy <= 1; ++dy)
                        for (int dx = -1; dx <= 1; ++dx) {
                            const int yy = py + dy, xx = px + dx;
                            if (yy < 0 || yy >= ih || xx < 0 || xx >= iw) continue;
                            const int q = yy * iw + xx;
                            if (b.fix_wall[q] == 4) { b.fix_wall[q] = 5; b.fix_queue.push_back(q); }
                        }
                }
                if (maxx - minx + 1 < kMinWall) continue;
                for (int p : b.fix_piece) {
                    // down to the floor plane: drop the standing copy; if nothing is
                    // below it, move the pixel to Plane 1
                    // the visible pixel moves down onto Plane 1 (overwriting what BG3
                    // has underneath, which would otherwise show through)
                    uint8_t* up = fx + 2 * plane_bytes + static_cast<size_t>(p) * 4;
                    std::memcpy(fx + 1 * plane_bytes + static_cast<size_t>(p) * 4, up, 4);
                    std::memset(up, 0, 4);
                    b.fix_wall[p] = 6;  // flattened wall
                    b.fix_top2[p] = 1;
                }
            }
            // A lean-to / shed built against a flattened wall belongs to the house: grow the
            // flat area through touching solid pixels that only the classifier lifted.
            b.fix_queue.clear();
            for (int p = 0; p < npx; ++p)
                if (b.fix_wall[p] == 6) b.fix_queue.push_back(p);
            for (size_t qi = 0; qi < b.fix_queue.size(); ++qi) {
                const int p = b.fix_queue[qi];
                const int py = p / iw, px = p % iw;
                const int nb[4][2] = {{0, -1}, {0, 1}, {-1, 0}, {1, 0}};
                for (const auto& d : nb) {
                    const int yy = py + d[1], xx = px + d[0];
                    if (yy < 0 || yy >= ih || xx < 0 || xx >= iw) continue;
                    const int q = yy * iw + xx;
                    if (b.fix_wall[q] == 6 || b.fix_wall[q] == 2 || !b.fix_solid[q] ||
                        b.fix_top2[q] != 2 || b.fix_top[q] >= 2) continue;
                    uint8_t* up = fx + 2 * plane_bytes + static_cast<size_t>(q) * 4;
                    std::memcpy(fx + 1 * plane_bytes + static_cast<size_t>(q) * 4, up, 4);
                    std::memset(up, 0, 4);
                    b.fix_wall[q] = 6;
                    b.fix_top2[q] = 1;
                    b.fix_queue.push_back(q);
                }
            }
        }
        }  // !interior
        // Interiors: the black void around the room (dark BG tiles the game puts on BG2,
        // i.e. stock Plane 2, in front of the room) goes to the back (Plane 0). The void
        // is the dark colour most common on the screen border; flood from the border
        // through pixels of exactly that colour.
        if (interior) {
            std::vector<uint32_t>& col = b.void_col;
            col.assign(n_px, 0);
            auto top_px = [&](size_t i) -> const uint8_t* {
                for (int q = static_cast<int>(used_planes) - 1; q >= 1; --q) {
                    const uint8_t* px = fx + static_cast<size_t>(q) * plane_bytes + i * 4;
                    if (px[3]) return px;
                }
                return nullptr;
            };
            uint32_t best = 0; int best_n = 0;
            {
                std::vector<std::pair<uint32_t, int>> counts;
                auto add = [&](size_t i) {
                    const uint8_t* px = top_px(i);
                    if (!px || px[3] != 255u || std::max({px[0], px[1], px[2]}) > 72u) return;
                    const uint32_t c = (px[0] << 16) | (px[1] << 8) | px[2];
                    for (auto& kv : counts) if (kv.first == c) { ++kv.second; return; }
                    counts.push_back({c, 1});
                };
                for (int x = 0; x < iw; ++x) { add(static_cast<size_t>(x)); add(static_cast<size_t>(ih - 1) * iw + x); }
                for (int y = 0; y < ih; ++y) { add(static_cast<size_t>(y) * iw); add(static_cast<size_t>(y) * iw + iw - 1); }
                for (auto& kv : counts) if (kv.second > best_n) { best = kv.first; best_n = kv.second; }
            }
            if (best_n >= 8) {
                auto is_void = [&](size_t i) {
                    const uint8_t* px = top_px(i);
                    return px && px[3] == 255u && ((px[0] << 16) | (px[1] << 8) | px[2]) == best;
                };
                b.fix_state.assign(n_px, 0);
                b.fix_queue.clear();
                auto seed = [&](size_t i) { if (!b.fix_state[i] && is_void(i)) { b.fix_state[i] = 1; b.fix_queue.push_back(static_cast<int>(i)); } };
                for (int x = 0; x < iw; ++x) { seed(static_cast<size_t>(x)); seed(static_cast<size_t>(ih - 1) * iw + x); }
                for (int y = 0; y < ih; ++y) { seed(static_cast<size_t>(y) * iw); seed(static_cast<size_t>(y) * iw + iw - 1); }
                for (size_t qi = 0; qi < b.fix_queue.size(); ++qi) {
                    const int p = b.fix_queue[qi];
                    const int py = p / iw, px = p % iw;
                    if (px > 0) seed(static_cast<size_t>(p - 1));
                    if (px + 1 < iw) seed(static_cast<size_t>(p + 1));
                    if (py > 0) seed(static_cast<size_t>(p - iw));
                    if (py + 1 < ih) seed(static_cast<size_t>(p + iw));
                }
                for (int p : b.fix_queue) {
                    const size_t i = static_cast<size_t>(p);
                    const uint8_t* src = top_px(i);
                    uint8_t* p0 = fx + i * 4;
                    p0[0] = src[0]; p0[1] = src[1]; p0[2] = src[2]; p0[3] = 255u;
                    for (unsigned q = 1; q < used_planes; ++q) std::memset(fx + q * plane_bytes + i * 4, 0, 4);
                }
            }
        }
        // Hand-authored building masks (tools/map_sweep/authored) win over every
        // rule above: alpha 253/252 on the top semantic pixel. Stand (semantic
        // Plane 2) -> on stock Plane 2; ground (0/1) -> off Plane 2.
        for (size_t i = 0; i < n_px; ++i) {
            const size_t o = i * 4 + 3;
            int sq = -1;
            uint8_t sa = 0;
            for (int q = static_cast<int>(kLayeredPlaneCount) - 1; q >= 0; --q) {
                const uint8_t a = masked_ptr[q * plane_bytes + o];
                if (a && a < 252u) continue;  // see through translucent glass/overlays
                if (a) { sq = q; sa = a; break; }
            }
            if (sa != 253u && sa != 252u) continue;
            int top = -1;
            for (int q = static_cast<int>(used_planes) - 1; q >= 0; --q)
                if (fx[q * plane_bytes + o]) { top = q; break; }
            // actors / HUD above stay as they are; an authored-ground BG pixel the game
            // put on Plane 3 (a room mirror's glass) comes down too
            if (top < 0 || top > 3 || (top == 3 && sq == 2)) continue;
            uint8_t* up = fx + 2 * plane_bytes + i * 4;
            if (sq == 2) {
                if (top < 2) std::memcpy(up, fx + top * plane_bytes + i * 4, 4);
            } else if (top >= 2) {
                // flatten: composite Planes 1..top (straight alpha, so see-through mirror
                // glass blends onto what it covers) into Plane 1 and clear the planes above;
                // just clearing Plane 2 would show the BG3 tile underneath (grass, not wall)
                uint8_t* p1 = fx + 1 * plane_bytes + i * 4;
                for (int q = 2; q <= top; ++q) {
                    uint8_t* hq = fx + static_cast<size_t>(q) * plane_bytes + i * 4;
                    const unsigned al = hq[3];
                    if (al == 255u || p1[3] == 0u) {
                        if (al) std::memcpy(p1, hq, 4);
                    } else if (al) {
                        for (int c = 0; c < 3; ++c)
                            p1[c] = static_cast<uint8_t>((hq[c] * al + p1[c] * (255u - al) + 127u) / 255u);
                    }
                    std::memset(hq, 0, 4);
                }
            }
        }
        // Behind an authored standing piece set into an authored wall (the YADO
        // sign), the ground plane holds whatever BG3 has there (grass). Fill it
        // with the nearest wall pixel of the same row, so a head move shows wall.
        {
            constexpr int kReach = 64;
            b.fix_state.assign(n_px, 0);  // 1 = authored stand, 2 = authored ground
            for (size_t i = 0; i < n_px; ++i) {
                const size_t o = i * 4 + 3;
                for (int q = static_cast<int>(kLayeredPlaneCount) - 1; q >= 0; --q) {
                    const uint8_t a = masked_ptr[q * plane_bytes + o];
                    if (!a) continue;
                    if (a == 253u || a == 252u) b.fix_state[i] = (q == 2) ? 1 : 2;
                    break;
                }
            }
            for (int y = 0; y < ih; ++y) {
                const size_t row = static_cast<size_t>(y) * iw;
                for (int x = 0; x < iw; ++x) {
                    if (b.fix_state[row + x] != 1) continue;
                    int l = -1, r = -1;
                    for (int k = x - 1; k >= 0 && x - k <= kReach; --k)
                        if (b.fix_state[row + k] == 2) { l = k; break; }
                    for (int k = x + 1; k < iw && k - x <= kReach; ++k)
                        if (b.fix_state[row + k] == 2) { r = k; break; }
                    if (l < 0 || r < 0) continue;  // not set into a wall (roof edges)
                    const int src = (x - l <= r - x) ? l : r;
                    std::memcpy(fx + 1 * plane_bytes + (row + x) * 4,
                                fx + 1 * plane_bytes + (row + src) * 4, 4);
                }
            }
        }
        last_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - fix_t0).count();
        return fx;
    }
};

}  // namespace gbarecomp
