// quest_popup_layout.h — CPU scene analysis for the Quest "POP-UP 3D" mode.
//
// Input: the 5 semantic RGBA planes produced by gba_ppu.cpp build_masked_5plane_frame
//   P0 backdrop/void, P1 ground/walls, P2 props, P3 actors, P4 overhangs/foreground/HUD.
// Output: layers ready for the pop-up renderer:
//   - props  : P2 objects + P4 pieces glued to them (roofs, canopies)
//   - actors : P3 sprites
//   - front  : loose P4 pieces (room frame, south rims, fog, HUD) drawn in front of everything
// plus, for props and actors, a per-pixel "base row" = the stock-image row where the
// object touches the floor. The renderer stands every pixel on a screen-facing card at the
// floor depth of its base row, so objects are grounded instead of floating.
//
// Stepped ("bent") cards: per object, each column's lowest pixel is smoothed with a
// lower envelope (fills gaps between posts/legs); the dominant base B is the median over
// columns. Pixels above B stand at B (house body on its walls); pixels below B in columns
// that stick out stand at their own column bottom (stairs become a step in front).
//
// Header-only and free of GL/OpenXR so it can be unit-tested on desktop.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

namespace gbarecomp {
namespace popup {

struct Group {
    uint16_t base = 0;  // base row (1..h); 0 is never used for a group
    int x0 = 0, x1 = 0, y0 = 0, y1 = 0;  // stock-pixel bbox, [x0,x1) x [y0,y1)
};

struct Layout {
    int w = 0, h = 0;
    std::vector<uint8_t> props_rgba, actors_rgba, front_rgba;
    std::vector<uint16_t> prop_base, actor_base;  // 0 = empty
    std::vector<Group> prop_groups, actor_groups;
    bool front_any = false;

    // scratch
    std::vector<int32_t> lab2, lab4, lab3, lab_obj;
    std::vector<int32_t> queue;
    std::vector<int32_t> comp_size;
    std::vector<int32_t> col_bottom, col_env;
    std::vector<int32_t> tmp;
};

namespace detail {

// 8-connected labelling with a square bridge radius. Returns component count; labels 1..n.
inline int label(const uint8_t* rgba, int w, int h, int bridge, std::vector<int32_t>& lab,
                 std::vector<int32_t>& q, std::vector<int32_t>& sizes) {
    const int n_px = w * h;
    lab.assign(n_px, 0);
    sizes.assign(1, 0);
    q.resize(n_px);
    int n = 0;
    for (int i = 0; i < n_px; ++i) {
        if (rgba[i * 4 + 3] == 0 || lab[i]) continue;
        ++n;
        sizes.push_back(0);
        int head = 0, tail = 0;
        q[tail++] = i;
        lab[i] = n;
        while (head < tail) {
            const int p = q[head++];
            ++sizes[n];
            const int py = p / w, px = p % w;
            const int ya = std::max(0, py - bridge), yb = std::min(h - 1, py + bridge);
            const int xa = std::max(0, px - bridge), xb = std::min(w - 1, px + bridge);
            for (int yy = ya; yy <= yb; ++yy) {
                for (int xx = xa; xx <= xb; ++xx) {
                    const int j = yy * w + xx;
                    if (rgba[j * 4 + 3] != 0 && !lab[j]) {
                        lab[j] = n;
                        q[tail++] = j;
                    }
                }
            }
        }
    }
    return n;
}

inline void add_group(std::vector<Group>& groups, std::vector<int32_t>& index_of_base, uint16_t base,
                      int x, int y) {
    int gi = index_of_base[base];
    if (gi < 0) {
        gi = static_cast<int>(groups.size());
        index_of_base[base] = gi;
        Group g;
        g.base = base;
        g.x0 = x; g.x1 = x + 1; g.y0 = y; g.y1 = y + 1;
        groups.push_back(g);
        return;
    }
    Group& g = groups[gi];
    g.x0 = std::min(g.x0, x); g.x1 = std::max(g.x1, x + 1);
    g.y0 = std::min(g.y0, y); g.y1 = std::max(g.y1, y + 1);
}

}  // namespace detail

// planes: 5 consecutive w*h RGBA planes (P0..P4), top row first.
inline void build_layout(const uint8_t* planes, int w, int h, Layout& L,
                         int min_px = 6, int smooth = 8) {
    using namespace detail;
    const int n_px = w * h;
    const size_t pb = static_cast<size_t>(n_px) * 4;
    const uint8_t* p2 = planes + 2 * pb;
    const uint8_t* p3 = planes + 3 * pb;
    const uint8_t* p4 = planes + 4 * pb;
    L.w = w; L.h = h;
    L.props_rgba.assign(pb, 0);
    L.actors_rgba.assign(pb, 0);
    L.front_rgba.assign(pb, 0);
    L.prop_base.assign(n_px, 0);
    L.actor_base.assign(n_px, 0);
    L.prop_groups.clear();
    L.actor_groups.clear();
    L.front_any = false;

    // ---- props: P2 components, glue P4 pieces that touch a (larger) P2 object
    std::vector<int32_t> size2, size4, size3;
    const int n2 = label(p2, w, h, 1, L.lab2, L.queue, size2);
    const int n4 = label(p4, w, h, 1, L.lab4, L.queue, size4);
    L.lab_obj = L.lab2;  // object id per pixel (P2 ids, glued P4 joins them)
    std::vector<int32_t> votes(n2 + 1, 0);
    // collect P4 pixels per component
    std::vector<std::vector<int32_t>> p4_pixels(n4 + 1);
    for (int i = 0; i < n_px; ++i)
        if (L.lab4[i]) p4_pixels[L.lab4[i]].push_back(i);
    int next_obj = n2;
    std::vector<int32_t> obj_size(size2.begin(), size2.end());
    for (int c = 1; c <= n4; ++c) {
        const auto& px = p4_pixels[c];
        if (static_cast<int>(px.size()) < min_px) continue;
        std::fill(votes.begin(), votes.end(), 0);
        for (int i : px) {  // neighbours within 2 px
            const int y = i / w, x = i % w;
            for (int yy = std::max(0, y - 2); yy <= std::min(h - 1, y + 2); ++yy)
                for (int xx = std::max(0, x - 2); xx <= std::min(w - 1, x + 2); ++xx) {
                    const int t = L.lab2[yy * w + xx];
                    if (t) ++votes[t];
                }
        }
        int best = 0;
        for (int t = 1; t <= n2; ++t)
            if (votes[t] > votes[best]) best = t;
        // only glue overhangs that are not much bigger than their object (keeps the
        // full-screen room frame / fog out of furniture)
        if (best && static_cast<int>(px.size()) <= obj_size[best] * 3 / 2) {
            for (int i : px) {
                if (L.lab_obj[i]) continue;
                L.lab_obj[i] = best;
                std::memcpy(&L.props_rgba[i * 4], &p4[i * 4], 4);
            }
            obj_size[best] += static_cast<int>(px.size());
        } else {
            for (int i : px) std::memcpy(&L.front_rgba[i * 4], &p4[i * 4], 4);
            L.front_any = true;
        }
    }
    (void)next_obj;
    for (int i = 0; i < n_px; ++i)
        if (L.lab2[i]) std::memcpy(&L.props_rgba[i * 4], &p2[i * 4], 4);

    // per object pixel lists
    std::vector<std::vector<int32_t>> obj_px(n2 + 1);
    for (int i = 0; i < n_px; ++i)
        if (L.lab_obj[i]) obj_px[L.lab_obj[i]].push_back(i);

    L.col_bottom.assign(w, -1);
    L.col_env.assign(w, -1);
    std::vector<int32_t> cols;
    for (int o = 1; o <= n2; ++o) {
        const auto& px = obj_px[o];
        if (static_cast<int>(px.size()) < min_px) {
            // tiny specks: still draw them, standing at their own bottom
            int yb = 0;
            for (int i : px) yb = std::max(yb, i / w + 1);
            for (int i : px) L.prop_base[i] = static_cast<uint16_t>(yb);
            continue;
        }
        cols.clear();
        for (int i : px) {
            const int x = i % w, y = i / w;
            if (L.col_bottom[x] < 0) cols.push_back(x);
            L.col_bottom[x] = std::max(L.col_bottom[x], y + 1);
        }
        std::sort(cols.begin(), cols.end());
        const int span = cols.back() - cols.front() + 1;
        if (span > 2 * smooth) {
            for (int x : cols) {
                int m = -1;
                for (int xx = std::max(0, x - smooth); xx <= std::min(w - 1, x + smooth); ++xx)
                    m = std::max(m, L.col_bottom[xx]);
                L.col_env[x] = m;
            }
            L.tmp.clear();
            for (int x : cols) L.tmp.push_back(L.col_env[x]);
            std::nth_element(L.tmp.begin(), L.tmp.begin() + L.tmp.size() / 2, L.tmp.end());
            const int B = L.tmp[L.tmp.size() / 2];
            // A protrusion (run of neighbouring columns reaching >= 6 rows below B) steps
            // forward as ONE card at the run's lowest row, so it never shows vertical stripes.
            for (size_t k = 0; k < cols.size();) {
                if (L.col_env[cols[k]] < B + 6) { ++k; continue; }
                size_t e = k;
                int m = L.col_env[cols[k]];
                while (e + 1 < cols.size() && cols[e + 1] == cols[e] + 1 && L.col_env[cols[e + 1]] >= B + 6)
                    m = std::max(m, L.col_env[cols[++e]]);
                for (size_t j = k; j <= e; ++j) L.col_env[cols[j]] = m;
                k = e + 1;
            }
            for (int i : px) {
                const int x = i % w, y = i / w;
                const int c = L.col_env[x];
                // Only forward protrusions (stairs, porch) get their own step; columns that end
                // above the dominant base (recessed walls, gaps) stay on the body so a building
                // never splits into a front and a back half.
                int base = B;
                // ...and only real ones (>= 6 rows), so a bed foot or table legs don't split
                if (c >= B + 6 && y >= B) base = c;
                L.prop_base[i] = static_cast<uint16_t>(base);
            }
        } else {
            int yb = 0;
            for (int x : cols) yb = std::max(yb, L.col_bottom[x]);
            for (int i : px) L.prop_base[i] = static_cast<uint16_t>(yb);
        }
        for (int x : cols) { L.col_bottom[x] = -1; L.col_env[x] = -1; }
    }

    // ---- stacking: a piece that sits on top of another object (roof on walls, upper storey,
    // P2 pieces cut apart by a P1 strip) stands on that object's base instead of its own.
    {
        struct Obj { int x0 = 1 << 30, x1 = -1, y0 = 1 << 30, y1 = -1, base = 0; };
        std::vector<Obj> ob(n2 + 1);
        for (int o = 1; o <= n2; ++o) {
            for (int i : obj_px[o]) {
                const int x = i % w, y = i / w;
                Obj& b = ob[o];
                b.x0 = std::min(b.x0, x); b.x1 = std::max(b.x1, x + 1);
                b.y0 = std::min(b.y0, y); b.y1 = std::max(b.y1, y + 1);
                b.base = std::max(b.base, static_cast<int>(L.prop_base[i]));
            }
        }
        std::vector<int> order;
        for (int o = 1; o <= n2; ++o)
            if (static_cast<int>(obj_px[o].size()) >= min_px) order.push_back(o);
        std::sort(order.begin(), order.end(), [&](int a, int b) { return ob[a].y1 > ob[b].y1; });
        std::vector<int> dominant(n2 + 1, 0);
        for (int o : order) {
            int d = 0;  // dominant (most common) base of the object
            L.tmp.clear();
            for (int i : obj_px[o]) L.tmp.push_back(L.prop_base[i]);
            std::nth_element(L.tmp.begin(), L.tmp.begin() + L.tmp.size() / 2, L.tmp.end());
            d = L.tmp[L.tmp.size() / 2];
            dominant[o] = d;
        }
        for (size_t k = 0; k < order.size(); ++k) {
            const int a = order[k];
            const Obj& A = ob[a];
            const int aw = A.x1 - A.x0;
            int best = 0, best_base = dominant[a];
            for (size_t j = 0; j < k; ++j) {  // candidates end lower on screen (processed first)
                const int s2 = order[j];
                const Obj& S = ob[s2];
                if (dominant[s2] <= dominant[a]) continue;
                if (S.y0 > A.y1 + 2 || S.y0 < A.y0) continue;  // must start at/just below A's bottom
                const int ov = std::min(A.x1, S.x1) - std::max(A.x0, S.x0);
                if (ov * 2 < aw) continue;  // A must sit mostly on S
                // touching: some A pixel within 2 px above an S pixel
                bool touch = false;
                for (int i : obj_px[a]) {
                    const int x = i % w, y = i / w;
                    for (int dy = 0; dy <= 3 && !touch; ++dy) {
                        if (y + dy >= h) break;
                        for (int dx = -1; dx <= 1 && !touch; ++dx) {
                            if (x + dx < 0 || x + dx >= w) continue;
                            touch = L.lab_obj[(y + dy) * w + x + dx] == s2;
                        }
                    }
                    if (touch) break;
                }
                if (touch && dominant[s2] > best_base) { best = s2; best_base = dominant[s2]; }
            }
            if (!best) continue;
            for (int i : obj_px[a])
                if (L.prop_base[i] < best_base) L.prop_base[i] = static_cast<uint16_t>(best_base);
            dominant[a] = best_base;
        }
    }

    // ---- actors: P3 sprites at their foot row; a sprite drawn over a prop must stand in front of it
    const int n3 = label(p3, w, h, 2, L.lab3, L.queue, size3);
    std::vector<int32_t> foot(n3 + 1, 0), front_of(n3 + 1, 0);
    for (int i = 0; i < n_px; ++i) {
        const int c = L.lab3[i];
        if (!c) continue;
        foot[c] = std::max(foot[c], i / w + 1);
        if (L.prop_base[i]) front_of[c] = std::max(front_of[c], static_cast<int>(L.prop_base[i]) + 1);
    }
    for (int i = 0; i < n_px; ++i) {
        const int c = L.lab3[i];
        if (!c) continue;
        int base = std::max(foot[c], front_of[c]);
        L.actor_base[i] = static_cast<uint16_t>(std::min(base, 0xFFFF));
        std::memcpy(&L.actors_rgba[i * 4], &p3[i * 4], 4);
    }

    // ---- groups (one screen-facing card per distinct base row)
    std::vector<int32_t> idx(65536, -1);
    for (int i = 0; i < n_px; ++i)
        if (L.prop_base[i]) add_group(L.prop_groups, idx, L.prop_base[i], i % w, i / w);
    std::fill(idx.begin(), idx.end(), -1);
    for (int i = 0; i < n_px; ++i)
        if (L.actor_base[i]) add_group(L.actor_groups, idx, L.actor_base[i], i % w, i / w);
}

}  // namespace popup
}  // namespace gbarecomp
