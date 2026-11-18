// gba_ppu.cpp — see gba_ppu.h.

#include "gba_ppu.h"
#include "gba_color.h"
#include "foreign_presentation_internal.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include "snapshot.h"

namespace gba {

// Widescreen margin tilemap provider (Step C). When set (by the runtime-side
// sidecar) and the wide path samples a margin column (hardware x outside
// 0..239), the BG text-tilemap entry is sourced from this hook (the true
// extended world) instead of the wrapped 256px ring. Returns nonzero + fills
// *out_entry when data is available. nullptr (default) = vanilla wide behavior.
// Layering-clean: the PPU only calls a nullable pointer; the sidecar owns it.
extern "C" int (*g_ws_tilemap_provider)(int bg, int hw_x, int screen_y,
                                        uint16_t* out_entry) = nullptr;
extern "C" int (*g_ws_bg_x_provider)(int bg, int output_x, int screen_y,
                                     int* out_hw_x) = nullptr;
extern "C" unsigned g_ws_bg_x_provider_layers = 0xFu;
extern "C" int (*g_ws_bg_xy_provider)(int, int, int, int*, int*) = nullptr;
extern "C" unsigned g_ws_bg_xy_provider_layers = 0xFu;
extern "C" int (*g_ws_ui_pixel_provider)(int, int, std::uint16_t*) = nullptr;
extern "C" int g_ws_affine_filter_enabled = 0;
extern "C" int (*g_ws_affine_filter_provider)(int, int) = nullptr;
extern "C" int g_ws_authored_margin_layers = 0;
extern "C" int g_layered_pop_objects = 0;
extern "C" int g_layered_unify_objects = 1;
extern "C" int g_layered_split_ties = 0;
extern "C" int g_layered_obj_forward = 0;
extern "C" int g_layered_obj_plane = 0;
extern "C" int g_layered_flat_bitmap = 0;
extern "C" int g_layered_sprite_layer = 0;
extern "C" int g_layered_fog_alpha = 0;
extern "C" float g_layered_split_push = 0.0f;
extern "C" void (*g_layered_bg_split_row)(int bg, int y, const uint8_t* io, const uint8_t* vram,
                                          uint8_t* out, int width) = nullptr;
extern "C" float g_layered_sprite_depth = 3.5f;
extern "C" float g_layered_sprite_depth_live = 3.5f;
extern "C" float g_layered_parallax_depth = 0.0f;
extern "C" float g_layered_plane_depth_bias[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
extern "C" int g_ws_native_view_left = -1;
extern "C" int g_ws_native_view_top = -1;
extern "C" void (*g_ws_frame_start)() = nullptr;

// Widescreen pillarbox (Step C policy): when nonzero, the wide path renders the
// margin columns (outside the central 240) as solid black instead of extended
// world — used for non-overworld screens (menus, battles, transitions) so they
// are letterboxed rather than garbled. Set per-frame by the runtime policy.
extern "C" int g_ws_pillarbox = 0;
extern "C" int g_ws_pillarbox_left = 0;
extern "C" int g_ws_pillarbox_right = 0;
extern "C" int (*g_ws_obj_x_provider)(int, int*) = nullptr;
extern "C" int (*g_ws_obj_attr_x_provider)(int, std::uint16_t,
                                            std::uint16_t, std::uint16_t,
                                            int*) = nullptr;
// Opt-in OBJ margin policy: clip OBJ pixels to the native 240px viewport in
// expanded rendering. Games routinely park sprites just past the visible edge
// (signed 9-bit OAM X, or Y-wrapped); without this clip those parked sprites
// become visible inside widescreen margins. Default off preserves each
// established per-game behavior. g_ws_obj_attr_x_provider/g_ws_obj_x_provider
// still run first, so a game can re-place known-safe HUD sprites and clip the
// rest.
extern "C" int g_ws_obj_native_clip = 0;
extern "C" int (*g_ws_obj_attr_y_provider)(int, std::uint16_t,
                                            std::uint16_t, std::uint16_t,
                                            int*) = nullptr;
extern "C" const WsMarginObjPixel* (*g_ws_obj_margin_provider)(int, int*, int*) = nullptr;
extern "C" int g_ppu_record_motion = 0;
extern "C" int (*g_present_interpolator)(const GbaPpu*, const uint8_t*, uint32_t, uint32_t,
                                         uint8_t*, float) = nullptr;
extern "C" int g_layered_solid_provider_wanted = 0;
extern "C" int (*g_layered_solid_provider)(int, int) = nullptr;
extern "C" void (*g_layered_solid_row_provider)(int, int, int, std::int8_t*) = nullptr;
extern "C" int (*g_layered_plane_provider)(int, int, int) = nullptr;
extern "C" void (*g_layered_plane_row_provider)(int, int, int, int, std::int8_t*) = nullptr;
extern "C" int g_layered_bg1_detached = 0;
extern "C" int g_layered_interior = 0;

namespace {

// Tag row for the scanline being composited into the work framebuffer (see
// GbaPpu::MotionLine); null for every other render (screenshots, re-latches).
std::uint16_t* g_tag_row = nullptr;
inline std::uint16_t motion_tag(int key, std::uint8_t layer) {
    if (layer == 5) return 0xFFFFu;
    return static_cast<std::uint16_t>(key < 0 || key > 0xFFFE ? 0xFFFE : key);
}
inline void fill_tags(std::uint32_t width) {
    if (g_tag_row) std::fill(g_tag_row, g_tag_row + width, static_cast<std::uint16_t>(0xFFFFu));
}

// These pointers have no C linkage and are deliberately not declared by the
// public PPU header. Only the engine-internal seam below can publish them.
const std::uint16_t* g_foreign_background = nullptr;
const GbaForeignObjFocusTransform* g_foreign_obj_focus = nullptr;
const GbaForeignScreenOverlay* g_foreign_screen_overlay = nullptr;

}  // namespace

namespace foreign_presentation_internal {

void set_background(const std::uint16_t* pixels) { g_foreign_background = pixels; }
const std::uint16_t* background() { return g_foreign_background; }
void set_obj_focus(const GbaForeignObjFocusTransform* focus) {
    g_foreign_obj_focus = focus;
}
const GbaForeignObjFocusTransform* obj_focus() { return g_foreign_obj_focus; }
void set_screen_overlay(const GbaForeignScreenOverlay* overlay) {
    g_foreign_screen_overlay = overlay;
}
const GbaForeignScreenOverlay* screen_overlay() { return g_foreign_screen_overlay; }
void clear() {
    g_foreign_background = nullptr;
    g_foreign_obj_focus = nullptr;
    g_foreign_screen_overlay = nullptr;
}

}  // namespace foreign_presentation_internal

GbaPpu::GbaPpu()  = default;
GbaPpu::~GbaPpu() = default;

void GbaPpu::serialize(gbarecomp::debug::SnapshotWriter& w) const {
    w.u32(scanline_);
    w.u32(dot_in_scanline_);
    w.u32(cycle_in_dot_);
    w.u16(vcount_);
    w.u64(frame_count_);
    w.boolean(has_latched_fb_);
    // Snapshot layout remains the fixed native 240x160 payload. At native width
    // retain the historical byte-for-byte path. A wide latch is row-major at
    // render_width(), so its first kFramebufferBytes are NOT the native image;
    // explicitly crop the authentic center instead. This also makes a wide save
    // useful when loaded later in faithful mode.
    if (!view_expanded()) {
        w.bytes(latched_fb_.data(), kFramebufferBytes);
    } else {
        std::array<uint8_t, kFramebufferBytes> center{};
        constexpr std::size_t kNativeStride = kScreenWidth * 3u;
        const std::size_t wide_stride = render_width() * 3u;
        const std::size_t center_x = native_view_x() * 3u;
        for (std::size_t y = 0; y < kScreenHeight; ++y) {
            std::memcpy(center.data() + y * kNativeStride,
                        latched_fb_.data() + (y + native_view_y()) * wide_stride + center_x,
                        kNativeStride);
        }
        w.bytes(center.data(), center.size());
    }
}

void GbaPpu::deserialize(gbarecomp::debug::SnapshotReader& r) {
    if (high_resolution_) high_resolution_->invalidate();
    // Foreign presentation is intentionally not save-state data.  A selected
    // trusted mod must explicitly republish a newly valid buffer after load.
    foreign_presentation_internal::clear();
    scanline_        = r.u32();
    dot_in_scanline_ = r.u32();
    cycle_in_dot_    = r.u32();
    vcount_          = r.u16();
    frame_count_     = r.u64();
    const bool serialized_latch = r.boolean();
    if (!view_expanded()) {
        // Historical native-width restore path, deliberately unchanged.
        has_latched_fb_ = serialized_latch;
        r.bytes(latched_fb_.data(), kFramebufferBytes);
        // Seed the scanline compositor with the restored frame: a state saved
        // mid-frame resumes rendering at scanline_, so rows at and below it
        // are not re-rendered until the frame after next. Without the seed
        // those rows of the first latched frame would be stale garbage.
        std::memcpy(work_fb_.data(), latched_fb_.data(), kFramebufferBytes);
    } else {
        // The fixed-size payload has no authored wide margins. Consume it to
        // preserve snapshot framing, then invalidate only the presentation
        // latch; the next present renders a fresh wide frame from restored GBA
        // state. Device timing/frame counters above remain fully restored.
        std::array<uint8_t, kFramebufferBytes> native_latch{};
        r.bytes(native_latch.data(), native_latch.size());
        has_latched_fb_ = false;
        // Seed the compositor's authentic center from the restored native
        // frame (margins stay whatever the pillarbox policy paints), so rows
        // the resumed mid-frame render never revisits before the next latch
        // show the restored image instead of stale pre-load pixels.
        constexpr std::size_t kNativeStride = kScreenWidth * 3u;
        const std::size_t wide_stride = render_width() * 3u;
        const std::size_t center_x = native_view_x() * 3u;
        for (std::size_t y = 0; y < kScreenHeight; ++y) {
            std::memcpy(work_fb_.data() + (y + native_view_y()) * wide_stride + center_x,
                        native_latch.data() + y * kNativeStride,
                        kNativeStride);
        }
        // A host may re-present immediately after load, before a game-owned
        // state-epoch hook has invalidated/rebuilt its margin caches. Fail those
        // unauthored margins closed for that interim render; the game's normal
        // publish policy reauthorizes them once restored-state data is ready.
        g_ws_pillarbox = g_ws_authored_margin_layers ? 0 : 1;
        g_ws_pillarbox_left = 0;
        g_ws_pillarbox_right = 0;
    }
    // The snapshot format predates the PPU's internal affine reference
    // accumulator. Reconstruct it from live BGxX/Y on the next scanline.
    affine_line_ = {};
}

void GbaPpu::reset() {
    if (high_resolution_) high_resolution_->invalidate();
    foreign_presentation_internal::clear();
    scanline_ = 0;
    dot_in_scanline_ = 0;
    cycle_in_dot_ = 0;
    vcount_ = 0;
    frame_count_ = 0;
    has_latched_fb_ = false;
    affine_line_ = {};
    std::memset(latched_fb_.data(), 0xFF, latched_fb_.size());
    std::memset(work_fb_.data(), 0xFF, work_fb_.size());
}

uint32_t GbaPpu::native_view_x() const {
    if (g_ws_native_view_left < 0) return extra_left_;
    return std::min(static_cast<uint32_t>(g_ws_native_view_left), render_width() - kScreenWidth);
}

uint32_t GbaPpu::native_view_y() const {
    if (g_ws_native_view_top < 0) return extra_top_;
    return std::min(static_cast<uint32_t>(g_ws_native_view_top), render_height() - kScreenHeight);
}

void GbaPpu::set_view_margins(uint32_t left, uint32_t right,
                              uint32_t top, uint32_t bottom) {
    // Geometry alone cannot authorize vertical content. The compositor also
    // requires authored providers; otherwise those rows stay black.
    const uint32_t next_left = left > kMaxExtraX ? kMaxExtraX : left;
    const uint32_t next_right = right > kMaxExtraX ? kMaxExtraX : right;
    const uint32_t next_top = top > kMaxExtraY ? kMaxExtraY : top;
    const uint32_t next_bottom = bottom > kMaxExtraY ? kMaxExtraY : bottom;
    if (next_left != extra_left_ || next_right != extra_right_ ||
        next_top != extra_top_ || next_bottom != extra_bottom_) {
        // A latched frame is laid out at the old stride. Force one fresh
        // render after a live resize rather than interpreting it at the new
        // dimensions. Timing state and guest memory remain untouched.
        has_latched_fb_ = false;
        has_latched_layered_fb_ = false;
        has_latched_masked_fb_ = false;
        if (high_resolution_) high_resolution_->invalidate();
    }
    extra_left_ = next_left;
    extra_right_ = next_right;
    extra_top_ = next_top;
    extra_bottom_ = next_bottom;
    if (layered_3d_) {
        work_layered_fb_.assign(layered_framebuffer_bytes(), 0);
        latched_layered_fb_.assign(layered_framebuffer_bytes(), 0);
        work_masked_fb_.assign(layered_framebuffer_bytes(), 0);
        latched_masked_fb_.assign(layered_framebuffer_bytes(), 0);
    }
}

GbaPpu::TickEvents GbaPpu::tick(uint32_t cycles, uint16_t vcount_compare) {
    // Coarse but correct: walk dot-by-dot, advancing scanline counters
    // as we cross dot boundaries. The cycle_in_dot_ accumulator lets
    // sub-dot ticks (less than 4 cycles) stash residue for the next
    // call.
    TickEvents ev{};
    cycle_in_dot_ += cycles;
    while (cycle_in_dot_ >= kCyclesPerDot) {
        cycle_in_dot_ -= kCyclesPerDot;
        bool was_visible = (dot_in_scanline_ < kDotsVisible);
        ++dot_in_scanline_;
        if (was_visible && dot_in_scanline_ == kDotsVisible) {
            ev.hblank_started = true;
        }
        if (dot_in_scanline_ >= kDotsPerScanline) {
            dot_in_scanline_ = 0;
            bool was_in_visible_window = (scanline_ < kLinesVisible);
            ++scanline_;
            if (scanline_ >= kLinesTotal) {
                scanline_ = 0;
                ++frame_count_;
                ev.frame_completed = true;
            }
            vcount_ = static_cast<uint16_t>(scanline_);
            if (was_in_visible_window && scanline_ == kLinesVisible) {
                ev.vblank_started = true;
            }
            if (vcount_ == vcount_compare) {
                ev.vcount_matched = true;
            }
        }
    }
    return ev;
}

uint32_t GbaPpu::cycles_until_next_event() const {
    uint32_t target_dot = (dot_in_scanline_ < kDotsVisible)
        ? kDotsVisible
        : kDotsPerScanline;
    uint32_t cycles = (target_dot - dot_in_scanline_) * kCyclesPerDot;
    if (cycles > cycle_in_dot_) cycles -= cycle_in_dot_;
    else cycles = 1;
    return cycles ? cycles : 1;
}

// ─────────────────────────────────────────────────────────────────────
// Renderer (Phase 2.4)
// ─────────────────────────────────────────────────────────────────────
//
// Scope: enough to draw the GBA BIOS boot intro. The BIOS intro uses
// DISPCNT = 0x1002 (BG mode 2, OBJ enabled, all BGs disabled), so we
// only need OBJ (sprite) compositing for this milestone.
//
// References: GBATEK § "GBA OBJs - OAM", § "GBA Palettes",
//             § "GBA VRAM Character Data".

namespace {

// Sprite shape × size → (width, height) in pixels.
//   shape: 0=square, 1=horizontal, 2=vertical
//   size:  0..3
constexpr int kSpriteWH[3][4][2] = {
    // square
    {{8, 8}, {16, 16}, {32, 32}, {64, 64}},
    // horizontal
    {{16, 8}, {32, 8}, {32, 16}, {64, 32}},
    // vertical
    {{8, 16}, {8, 32}, {16, 32}, {32, 64}},
};

// Keep the descriptor bounded to the visible native PPU domain. This limits
// the trusted presentation input without imposing game-specific coordinates.
bool valid_foreign_obj_focus(const GbaForeignObjFocusTransform* focus) {
    return focus &&
           focus->abi_version == GBA_FOREIGN_OBJ_FOCUS_ABI_VERSION &&
           focus->source_radius_x <= GbaPpu::kScreenWidth &&
           focus->source_radius_y <= GbaPpu::kScreenHeight &&
           (focus->source_obj_scale_q8_8 == 0 ||
            (focus->source_obj_scale_q8_8 >=
                 GBA_FOREIGN_OBJ_FOCUS_SCALE_MIN_Q8_8 &&
             focus->source_obj_scale_q8_8 <=
                 GBA_FOREIGN_OBJ_FOCUS_SCALE_MAX_Q8_8 &&
             (focus->flags & GBA_FOREIGN_OBJ_FOCUS_SOURCE_TILE_RANGE) != 0)) &&
           focus->reserved == 0 &&
           focus->hud_bg_layer_mask <= 0x0fu &&
           focus->hud_bg_map_rect_count <= GBA_FOREIGN_OBJ_FOCUS_MAX_HUD_BG_MAP_RECTS &&
           focus->hud_bg_reserved == 0 &&
           ((focus->hud_bg_map_rect_count == 0 &&
             focus->hud_bg_layer_mask == 0 &&
             focus->hud_bg_map_tile_x == 0 && focus->hud_bg_map_tile_y == 0 &&
             focus->hud_bg_map_tile_width == 0 &&
             focus->hud_bg_map_tile_height == 0 &&
             focus->hud_bg_output_x == 0 && focus->hud_bg_output_y == 0 &&
             focus->hud_bg_output_width == 0 &&
             focus->hud_bg_output_height == 0) ||
            (focus->hud_bg_map_rect_count == 1 &&
             focus->hud_bg_layer_mask != 0 &&
             focus->hud_bg_map_tile_width != 0 &&
             focus->hud_bg_map_tile_height != 0 &&
             focus->hud_bg_map_tile_x < 64 && focus->hud_bg_map_tile_y < 64 &&
             static_cast<unsigned>(focus->hud_bg_map_tile_x) +
                     focus->hud_bg_map_tile_width <= 64 &&
             static_cast<unsigned>(focus->hud_bg_map_tile_y) +
                     focus->hud_bg_map_tile_height <= 64 &&
             focus->hud_bg_output_width != 0 &&
             focus->hud_bg_output_height != 0 &&
             focus->hud_bg_output_x < GbaPpu::kScreenWidth &&
             focus->hud_bg_output_y < GbaPpu::kScreenHeight &&
             static_cast<unsigned>(focus->hud_bg_output_x) +
                     focus->hud_bg_output_width <= GbaPpu::kScreenWidth &&
             static_cast<unsigned>(focus->hud_bg_output_y) +
                     focus->hud_bg_output_height <= GbaPpu::kScreenHeight)) &&
           (focus->flags & ~(GBA_FOREIGN_OBJ_FOCUS_PRESERVE_UNFOCUSED |
                             GBA_FOREIGN_OBJ_FOCUS_ORIGIN_ONLY |
                             GBA_FOREIGN_OBJ_FOCUS_SOURCE_TILE_RANGE |
                             GBA_FOREIGN_OBJ_FOCUS_SUPPRESS_LARGE_NEARBY |
                             GBA_FOREIGN_OBJ_FOCUS_SUPPRESS_NEARBY_NONMATCHING |
                             GBA_FOREIGN_OBJ_FOCUS_SUPPRESS_NONMATCHING_EXCEPT_HUD_OAM)) == 0 &&
           ((focus->flags &
             GBA_FOREIGN_OBJ_FOCUS_SUPPRESS_NONMATCHING_EXCEPT_HUD_OAM) == 0 ||
            (focus->flags & GBA_FOREIGN_OBJ_FOCUS_SOURCE_TILE_RANGE) != 0) &&
           ((focus->flags & GBA_FOREIGN_OBJ_FOCUS_SOURCE_TILE_RANGE) == 0 ||
            (focus->source_obj_tile_count != 0 &&
             focus->source_obj_tile_base < 1024u &&
             static_cast<unsigned>(focus->source_obj_tile_base) +
                 static_cast<unsigned>(focus->source_obj_tile_count) <= 1024u &&
             (focus->source_aux_obj_tile_count == 0 ||
              (focus->source_aux_obj_tile_base < 1024u &&
               static_cast<unsigned>(focus->source_aux_obj_tile_base) +
                   static_cast<unsigned>(focus->source_aux_obj_tile_count) <= 1024u))));
}

// Keep the portal/effect input small enough to be independently reviewable
// and to bound all per-scanline pointer arithmetic. Pointer lifetime remains
// owned by the trusted publishing plugin, just as it does for the existing
// foreign background/focus descriptors.
bool valid_foreign_screen_overlay(const GbaForeignScreenOverlay* overlay) {
    return overlay &&
           overlay->abi_version == GBA_FOREIGN_SCREEN_OVERLAY_ABI_VERSION &&
           overlay->pixels && overlay->alpha_q4 &&
           overlay->width != 0 && overlay->height != 0 &&
           overlay->width <= GBA_FOREIGN_SCREEN_OVERLAY_MAX_WIDTH &&
           overlay->height <= GBA_FOREIGN_SCREEN_OVERLAY_MAX_HEIGHT &&
           overlay->stride >= overlay->width &&
           overlay->stride <= GBA_FOREIGN_SCREEN_OVERLAY_MAX_WIDTH &&
           overlay->reserved16 == 0 && overlay->reserved32 == 0;
}

bool focus_contains(const GbaForeignObjFocusTransform& focus, int x, int y) {
    const int min_x = static_cast<int>(focus.source_link_feet_x) -
                      static_cast<int>(focus.source_radius_x);
    const int max_x = static_cast<int>(focus.source_link_feet_x) +
                      static_cast<int>(focus.source_radius_x);
    const int min_y = static_cast<int>(focus.source_link_feet_y) -
                      static_cast<int>(focus.source_radius_y);
    const int max_y = static_cast<int>(focus.source_link_feet_y) +
                      static_cast<int>(focus.source_radius_y);
    return x >= min_x && x <= max_x && y >= min_y && y <= max_y;
}

bool foreign_hud_bg_map_contains(const GbaForeignObjFocusTransform* focus,
                                 uint32_t layer, uint32_t tile_x,
                                 uint32_t tile_y, uint32_t output_x,
                                 uint32_t output_y) {
    return valid_foreign_obj_focus(focus) &&
           focus->hud_bg_map_rect_count != 0 && layer < 4 &&
           (focus->hud_bg_layer_mask & (1u << layer)) != 0 &&
           tile_x >= focus->hud_bg_map_tile_x &&
           tile_y >= focus->hud_bg_map_tile_y &&
           tile_x < static_cast<uint32_t>(focus->hud_bg_map_tile_x) +
                        focus->hud_bg_map_tile_width &&
           tile_y < static_cast<uint32_t>(focus->hud_bg_map_tile_y) +
                        focus->hud_bg_map_tile_height &&
           output_x >= focus->hud_bg_output_x &&
           output_y >= focus->hud_bg_output_y &&
           output_x < static_cast<uint32_t>(focus->hud_bg_output_x) +
                          focus->hud_bg_output_width &&
           output_y < static_cast<uint32_t>(focus->hud_bg_output_y) +
                          focus->hud_bg_output_height;
}

// Presentation-only translation. It reads a trusted immutable descriptor and
// operates on decoded local coordinates; guest OAM and memory stay untouched.
enum class ForeignObjFocusAction { kKeep, kTranslate, kSuppress };

struct ForeignObjFocusResult {
    ForeignObjFocusAction action = ForeignObjFocusAction::kKeep;
    int scale_q8_8 = GBA_FOREIGN_OBJ_FOCUS_SCALE_IDENTITY_Q8_8;
};

int round_scaled_offset(int offset, int scale_q8_8) {
    const int scaled = offset * scale_q8_8;
    return scaled >= 0 ?
        (scaled + GBA_FOREIGN_OBJ_FOCUS_SCALE_IDENTITY_Q8_8 / 2) /
            GBA_FOREIGN_OBJ_FOCUS_SCALE_IDENTITY_Q8_8 :
        -((-scaled + GBA_FOREIGN_OBJ_FOCUS_SCALE_IDENTITY_Q8_8 / 2) /
          GBA_FOREIGN_OBJ_FOCUS_SCALE_IDENTITY_Q8_8);
}

int scaled_extent(int extent, int scale_q8_8) {
    const int scaled = (extent * scale_q8_8 +
                        static_cast<int>(GBA_FOREIGN_OBJ_FOCUS_SCALE_IDENTITY_Q8_8) / 2) /
                       static_cast<int>(GBA_FOREIGN_OBJ_FOCUS_SCALE_IDENTITY_Q8_8);
    return std::max(1, scaled);
}

int nearest_source_pixel(int destination_pixel, int source_extent,
                         int destination_extent) {
    // Map pixel centers, rather than tile origins, so a nearest-neighbor
    // downscale retains both edges of an 8x8 source tile.
    return ((destination_pixel * 2 + 1) * source_extent) /
           (destination_extent * 2);
}

ForeignObjFocusResult apply_foreign_obj_focus(int* sx, int* sy, int width,
                                              int height, uint32_t tile_num,
                                              bool scale_non_affine,
                                              uint32_t oam_index) {
    const GbaForeignObjFocusTransform* focus =
        foreign_presentation_internal::obj_focus();
    if (!sx || !sy || width <= 0 || height <= 0 ||
        !valid_foreign_obj_focus(focus)) {
        return {};
    }
    const bool origin_matches = focus_contains(*focus, *sx, *sy);
    const bool center_matches =
        focus_contains(*focus, *sx + width / 2, *sy + height / 2);
    const bool in_main_range =
        tile_num >= focus->source_obj_tile_base &&
        tile_num < static_cast<uint32_t>(focus->source_obj_tile_base) +
                       focus->source_obj_tile_count;
    const bool in_aux_range = focus->source_aux_obj_tile_count != 0 &&
        tile_num >= focus->source_aux_obj_tile_base &&
        tile_num < static_cast<uint32_t>(focus->source_aux_obj_tile_base) +
                       focus->source_aux_obj_tile_count;
    const bool tile_matches =
        (focus->flags & GBA_FOREIGN_OBJ_FOCUS_SOURCE_TILE_RANGE) == 0 ||
        in_main_range || in_aux_range;
    if (!tile_matches) {
        if ((focus->flags &
             GBA_FOREIGN_OBJ_FOCUS_SUPPRESS_NONMATCHING_EXCEPT_HUD_OAM) != 0) {
            // The plugin's exact bit is derived from current gHUD metadata and
            // this OAM entry, so a playfield OBJ cannot survive by sharing a
            // broad native priority class with a heart or button glyph.
            const bool is_hud = oam_index < 64u
                ? (focus->hud_oam_mask_lo & (UINT64_C(1) << oam_index)) != 0
                : (focus->hud_oam_mask_hi & (UINT64_C(1) << (oam_index - 64u))) != 0;
            if (is_hud) return {};
            return {ForeignObjFocusAction::kSuppress};
        }
        const bool nearby = origin_matches || center_matches;
        const bool suppress_any =
            (focus->flags & GBA_FOREIGN_OBJ_FOCUS_SUPPRESS_NEARBY_NONMATCHING) != 0;
        const bool suppress_large =
            (focus->flags & GBA_FOREIGN_OBJ_FOCUS_SUPPRESS_LARGE_NEARBY) != 0 &&
            (width >= 32 || height >= 32);
        if (nearby && (suppress_any || suppress_large)) {
            return {ForeignObjFocusAction::kSuppress};
        }
        return {};
    }
    if (!origin_matches &&
        ((focus->flags & GBA_FOREIGN_OBJ_FOCUS_ORIGIN_ONLY) != 0 ||
         !center_matches)) {
        return {};
    }
    // Scaling needs the same explicit, bounded source allocation that selects
    // Link's composite body/shadow. A geometry-only focus may translate but
    // cannot accidentally resample a nearby guest OBJ.
    const bool has_bounded_source_tiles =
        (focus->flags & GBA_FOREIGN_OBJ_FOCUS_SOURCE_TILE_RANGE) != 0;
    const int scale_q8_8 =
        scale_non_affine && has_bounded_source_tiles &&
            focus->source_obj_scale_q8_8 != 0 ?
        focus->source_obj_scale_q8_8 : GBA_FOREIGN_OBJ_FOCUS_SCALE_IDENTITY_Q8_8;
    *sx = static_cast<int>(focus->destination_link_feet_x) +
        round_scaled_offset(*sx - static_cast<int>(focus->source_link_feet_x),
                            scale_q8_8);
    *sy = static_cast<int>(focus->destination_link_feet_y) +
        round_scaled_offset(*sy - static_cast<int>(focus->source_link_feet_y),
                            scale_q8_8);
    return {ForeignObjFocusAction::kTranslate, scale_q8_8};
}

// Convert 16-bit GBA color (0BBBBBGGGGGRRRRR) to 24-bit RGB888.
using color::to_rgb888;
using color::blend_alpha_gba555;
using color::brighten_gba555;
using color::darken_gba555;

inline uint16_t load_u16_le(const uint8_t* p) {
    return static_cast<uint16_t>(p[0] | (p[1] << 8));
}
} // namespace

// Read a signed 16-bit Q8.8 affine parameter from IO.
static inline int32_t read_s16(const uint8_t* io, uint32_t off) {
    int16_t v = static_cast<int16_t>(
        io[off] | (io[off + 1] << 8));
    return static_cast<int32_t>(v);
}

// Read a signed 28-bit Q19.8 reference coord from IO (BGxX/BGxY).
// Bits 27..0 are the value; the field is sign-extended from bit 27.
static inline int32_t read_s28_ref(const uint8_t* io, uint32_t off) {
    uint32_t v = static_cast<uint32_t>(io[off]) |
                 (static_cast<uint32_t>(io[off + 1]) <<  8) |
                 (static_cast<uint32_t>(io[off + 2]) << 16) |
                 (static_cast<uint32_t>(io[off + 3]) << 24);
    v &= 0x0FFFFFFFu;
    if (v & 0x08000000u) v |= 0xF0000000u;
    return static_cast<int32_t>(v);
}

static inline void write_s28_ref(uint8_t* io, uint32_t off, int32_t value) {
    const uint32_t v = static_cast<uint32_t>(value) & 0x0FFFFFFFu;
    io[off + 0] = static_cast<uint8_t>(v);
    io[off + 1] = static_cast<uint8_t>(v >> 8);
    io[off + 2] = static_cast<uint8_t>(v >> 16);
    io[off + 3] = static_cast<uint8_t>(v >> 24);
}

namespace {

bool obj_texel_opaque(const uint8_t* vram,
                      uint32_t obj_tile_base,
                      bool obj_1d_mapping,
                      uint32_t tile_num,
                      int tiles_w,
                      bool color256,
                      int tex_x,
                      int tex_y) {
    int tile_x_in_sprite = tex_x >> 3;
    int tile_y_in_sprite = tex_y >> 3;
    int px_in_tile       = tex_x & 7;
    int py_in_tile       = tex_y & 7;

    uint32_t this_tile;
    if (obj_1d_mapping) {
        this_tile = tile_num + (tile_y_in_sprite * tiles_w + tile_x_in_sprite) *
                                (color256 ? 2u : 1u);
    } else {
        this_tile = tile_num + (tile_y_in_sprite * 32u) +
                    tile_x_in_sprite * (color256 ? 2u : 1u);
    }
    uint32_t tile_off = obj_tile_base + this_tile * 32u;
    if (color256) {
        uint32_t off = tile_off + py_in_tile * 8 + px_in_tile;
        if (off + 1 > 96u * 1024u) return false;
        return vram[off] != 0;
    }
    uint32_t off = tile_off + py_in_tile * 4 + (px_in_tile / 2);
    if (off + 1 > 96u * 1024u) return false;
    uint8_t b = vram[off];
    uint8_t pal_index = (px_in_tile & 1) ? (b >> 4) : (b & 0x0F);
    return pal_index != 0;
}

void mark_obj_window_scanline(bool* mask,
                              uint32_t y,
                              uint16_t dispcnt,
                              const uint8_t* vram,
                              const uint8_t* oam,
                              uint32_t kScreenWidth,
                              uint32_t kScreenHeight) {
    if (y >= kScreenHeight) return;
    if ((dispcnt & 0x8000u) == 0) return;

    uint32_t bg_mode = dispcnt & 0x07u;
    constexpr uint32_t obj_tile_base = 0x10000u;
    bool obj_1d_mapping = (dispcnt & 0x0040u) != 0;

    for (int idx = 127; idx >= 0; --idx) {
        const uint8_t* entry = oam + idx * 8;
        uint16_t attr0 = load_u16_le(&entry[0]);
        uint16_t attr1 = load_u16_le(&entry[2]);
        uint16_t attr2 = load_u16_le(&entry[4]);

        bool rot_scale = (attr0 & 0x0100u) != 0;
        bool disable_or_double = (attr0 & 0x0200u) != 0;
        if (!rot_scale && disable_or_double) continue;
        uint32_t obj_mode = (attr0 >> 10) & 0x3u;
        if (obj_mode != 2) continue;

        uint32_t shape = (attr0 >> 14) & 0x3u;
        if (shape >= 3) continue;
        uint32_t size  = (attr1 >> 14) & 0x3u;
        int sw = kSpriteWH[shape][size][0];
        int sh = kSpriteWH[shape][size][1];

        int sy = static_cast<int>(attr0 & 0xFFu);
        int sx = static_cast<int>(attr1 & 0x1FFu);
        if (sy >= 160) sy -= 256;
        if (sx & 0x100) sx -= 0x200;

        bool color256 = (attr0 & 0x2000u) != 0;
        uint32_t tile_num = attr2 & 0x3FFu;
        // Bitmap BGs consume the lower half of OBJ VRAM. Hardware preserves
        // the normal tile-number origin and ignores tiles 0..511 rather than
        // rebasing tile 0 to 0x14000.
        if (bg_mode >= 3 && tile_num < 512u) continue;
        int tiles_w = sw / 8;
        int tiles_h = sh / 8;

        if (rot_scale) {
            int bw = disable_or_double ? sw * 2 : sw;
            int bh = disable_or_double ? sh * 2 : sh;
            int j = static_cast<int>(y) - sy;
            if (j < 0 || j >= bh) continue;

            int affine_group = (attr1 >> 9) & 0x1Fu;
            const uint8_t* ag = oam + affine_group * 0x20u;
            int32_t pa = read_s16(ag, 0x06);
            int32_t pb = read_s16(ag, 0x0E);
            int32_t pc = read_s16(ag, 0x16);
            int32_t pd = read_s16(ag, 0x1E);

            int half_bw = bw >> 1;
            int half_bh = bh >> 1;
            int half_sw = sw >> 1;
            int half_sh = sh >> 1;
            int dy = j - half_bh;
            for (int i = 0; i < bw; ++i) {
                int screen_x = sx + i;
                if (screen_x < 0 || screen_x >= static_cast<int>(kScreenWidth)) continue;
                int dx = i - half_bw;
                int tex_x = ((pa * dx + pb * dy) >> 8) + half_sw;
                int tex_y = ((pc * dx + pd * dy) >> 8) + half_sh;
                if (tex_x < 0 || tex_x >= sw) continue;
                if (tex_y < 0 || tex_y >= sh) continue;
                if (obj_texel_opaque(vram, obj_tile_base, obj_1d_mapping,
                                     tile_num, tiles_w, color256,
                                     tex_x, tex_y)) {
                    mask[screen_x] = true;
                }
            }
            continue;
        }

        int line = static_cast<int>(y) - sy;
        if (line < 0 || line >= sh) continue;
        bool hflip = (attr1 & 0x1000u) != 0;
        bool vflip = (attr1 & 0x2000u) != 0;
        int ty = line >> 3;
        int py = line & 7;
        int src_ty = vflip ? (tiles_h - 1 - ty) : ty;
        int src_py = vflip ? (7 - py) : py;
        for (int tx = 0; tx < tiles_w; ++tx) {
            int src_tx = hflip ? (tiles_w - 1 - tx) : tx;
            for (int px = 0; px < 8; ++px) {
                int screen_x = sx + tx * 8 + px;
                if (screen_x < 0 || screen_x >= static_cast<int>(kScreenWidth)) continue;
                int src_px = hflip ? (7 - px) : px;
                int tex_x = src_tx * 8 + src_px;
                int tex_y = src_ty * 8 + src_py;
                if (obj_texel_opaque(vram, obj_tile_base, obj_1d_mapping,
                                     tile_num, tiles_w, color256,
                                     tex_x, tex_y)) {
                    mask[screen_x] = true;
                }
            }
        }
    }
}

struct LayeredCandidate {
#if !defined(GBARECOMP_SCANLINE_HLE)
    uint8_t rgb[3] = {0, 0, 0};
#endif
    uint16_t color = 0;
    int key = 0x7FFFFFFF;
    uint8_t layer = 5;
    bool target1 = false;
    bool target2 = false;
    bool valid = false;
};

void clear_layered_scanline_row(uint8_t* layered_rgba,
                                uint32_t y_out,
                                uint32_t width,
                                uint32_t height,
                                uint8_t backdrop_fill) {
    if (!layered_rgba || y_out >= height) return;
    const size_t plane_stride = static_cast<size_t>(width) * height * 4u;
    const size_t row_off = static_cast<size_t>(y_out) * width * 4u;
    uint8_t* plane0 = layered_rgba + row_off;
    if (backdrop_fill == 0xFFu) {
        std::memset(plane0, 0xFF, width * 4u);
    } else {
        for (uint32_t x = 0; x < width; ++x) {
            plane0[x * 4u + 0] = backdrop_fill;
            plane0[x * 4u + 1] = backdrop_fill;
            plane0[x * 4u + 2] = backdrop_fill;
            plane0[x * 4u + 3] = 255u;
        }
    }
    for (uint32_t p = 1; p < GbaPpu::kLayeredPlanes; ++p) {
        std::memset(layered_rgba + p * plane_stride + row_off, 0, width * 4u);
    }
}

// g_layered_split_ties: when enabled BGs share a priority (e.g. a cutscene with every
// BG at priority 0) the stock "plane = 4 - priority" puts them on one plane. Instead rank
// the enabled BGs by (priority, BG index) and give each its own plane, front first.
// An OBJ of priority p goes on the plane of the frontmost BG it is drawn above (priority
// >= p); per-plane key ordering keeps the hardware occlusion. Without ties: stock mapping.
struct LayeredRank {
    bool active = false;
    int8_t bg[4] = {4, 3, 2, 1};
    int8_t obj[4] = {4, 3, 2, 1};
};
static LayeredRank s_rank;
static bool s_flat_scene = false;
// Depth (plane units) of the separate sprite layer on the current scanline. Negative
// g_layered_sprite_depth = automatic: just in front of the frontmost BG the player sprite is
// drawn over (priority >= its own), i.e. behind every BG the GBA draws in front of it.
static float s_line_sprite_depth = 3.5f;

// g_layered_obj_plane: the world sprites (the OBJ priority >= 1 covering the most screen area)
// get a plane of their own, between the BGs drawn in front of them and the BGs they are drawn
// over. The plane is freed by putting the backdrop colour on the backmost BG's plane (plane 0).
static int s_obj_plane_prio = -1;
static int s_first_world_prio = -1;
static void choose_obj_plane_prio(const uint8_t* oam) {
    s_first_world_prio = -1;
    if ((!g_layered_obj_plane && !g_layered_sprite_layer) || !oam) { s_obj_plane_prio = -1; return; }
    static const int kW[3][4] = {{8, 16, 32, 64}, {16, 32, 32, 64}, {8, 8, 16, 32}};
    static const int kH[3][4] = {{8, 16, 32, 64}, {8, 8, 16, 32}, {16, 32, 32, 64}};
    long area[4] = {};
    for (int i = 0; i < 128; ++i) {
        const uint16_t a0 = static_cast<uint16_t>(oam[i * 8] | (oam[i * 8 + 1] << 8));
        const uint16_t a1 = static_cast<uint16_t>(oam[i * 8 + 2] | (oam[i * 8 + 3] << 8));
        const uint16_t a2 = static_cast<uint16_t>(oam[i * 8 + 4] | (oam[i * 8 + 5] << 8));
        const bool affine = a0 & 0x100;
        if (!affine && (a0 & 0x200)) continue;          // hidden
        const int mode = (a0 >> 10) & 3;
        if (mode >= 2) continue;                          // OBJ window / prohibited
        const int shape = (a0 >> 14) & 3, size = (a1 >> 14) & 3;
        if (shape == 3) continue;
        int y = a0 & 0xFF, x = a1 & 0x1FF;
        if (y >= 160) y -= 256;
        if (x >= 240) x -= 512;
        const int w = kW[shape][size], h = kH[shape][size];
        if (x + w <= 0 || x >= 240 || y + h <= 0 || y >= 160) continue;
        area[(a2 >> 10) & 3] += w * h;
        if (s_first_world_prio < 0 && ((a2 >> 10) & 3) >= 1) s_first_world_prio = (a2 >> 10) & 3;
    }
    // The frontmost world-sprite priority with a character-sized area wins: players and
    // enemies sit in front of scenery sprites (WL4's boss door is two big priority-3 OBJs
    // behind the wall arch, Wario is priority 2).
    int best = -1;
    // Sprite layer: the player is drawn first among the world sprites (WL4: OAM right after
    // the priority-0 HUD), so the first visible OBJ of priority >= 1 decides.
    if (g_layered_sprite_layer && s_first_world_prio >= 1) best = s_first_world_prio;
    if (best < 0) for (int p = 1; p < 4; ++p) if (area[p] >= 256) { best = p; break; }
    if (best < 0) for (int p = 1; p < 4; ++p) if (area[p] > 0 && (best < 0 || area[p] > area[best])) best = p;
    s_obj_plane_prio = best;
}

static void update_layered_rank(uint16_t dispcnt, const uint8_t* io) {
    s_rank.active = false;
    // Bitmap modes (3-5) are one picture with sprites placed by the game for that picture
    // (cutscenes, e.g. WL4's tunnel with a sprite city seen through it): splitting them by
    // priority only scatters pieces, so keep the whole scene on the middle plane.
    s_flat_scene = false;
    if (g_layered_flat_bitmap && (dispcnt & 7u) >= 3) {
        for (int i = 0; i < 4; ++i) { s_rank.bg[i] = 2; s_rank.obj[i] = 2; }
        s_rank.active = true;
        s_flat_scene = true;
        return;
    }
    if (!g_layered_split_ties && !g_layered_obj_forward && !g_layered_obj_plane && !g_layered_sprite_layer) return;
    const uint32_t mode = dispcnt & 7u;
    int idx[4], n = 0, prio[4];
    bool tie = false;
    for (int i = 0; i < 4; ++i) {
        prio[i] = (io[0x08 + 2 * i] | (io[0x09 + 2 * i] << 8)) & 3;
        const bool exists = mode == 0 || (mode == 1 && i < 3) || (mode == 2 && i >= 2);
        if (!exists || !(dispcnt & (0x100u << i))) continue;
        for (int k = 0; k < n; ++k) if (prio[idx[k]] == prio[i]) tie = true;
        idx[n++] = i;
    }
    const bool split = g_layered_split_ties && tie;
    const int op = (g_layered_obj_plane && !g_layered_sprite_layer && (dispcnt & 0x1000u)) ? s_obj_plane_prio : -1;
    if (op >= 1) {
        // Front-to-back groups: BGs drawn above the sprites, the sprite plane, BGs they are
        // drawn over. Equal-priority BGs share a group unless split. Planes are handed out
        // from the front (4) down; with 5 groups the backmost BG shares plane 0 with the
        // backdrop colour, which it covers anyway.
        std::sort(idx, idx + n, [&](int a, int b) { return prio[a] != prio[b] ? prio[a] < prio[b] : a < b; });
        int plane = 5;
        int obj_slot_plane = -1;
        int last_prio = -1;
        // Mode 2: the sprite plane goes in FRONT of the nearest BG drawn over the sprites (the
        // terrain they stand on); fix_sprite_plane_occlusion() keeps only visible sprite pixels.
        int slot_prio = op;
        if (g_layered_obj_plane == 2) {
            int below = -1;
            for (int r = 0; r < n; ++r) if (prio[idx[r]] < op) below = prio[idx[r]];
            if (below >= 0) slot_prio = below;
        }
        for (int r = 0; r < n; ++r) {
            const int pr = prio[idx[r]];
            if (obj_slot_plane < 0 && pr >= slot_prio) { --plane; obj_slot_plane = plane; last_prio = -1; }
            if (split || pr != last_prio) --plane;
            last_prio = pr;
            s_rank.bg[idx[r]] = static_cast<int8_t>(std::max(plane, 0));
        }
        if (obj_slot_plane < 0) obj_slot_plane = std::max(plane - 1, 0);  // sprites behind every BG
        const int8_t back = n ? s_rank.bg[idx[n - 1]] : 0;
        for (int p = 0; p < 4; ++p) {
            if (p == op) { s_rank.obj[p] = static_cast<int8_t>(obj_slot_plane); continue; }
            int8_t pl = back;
            for (int r = 0; r < n; ++r) if (prio[idx[r]] >= p) { pl = s_rank.bg[idx[r]]; break; }
            s_rank.obj[p] = pl;
        }
        s_rank.active = true;
        return;
    }
    if (!split && !g_layered_obj_forward) return;
    std::sort(idx, idx + n, [&](int a, int b) { return prio[a] != prio[b] ? prio[a] < prio[b] : a < b; });
    for (int i = 0; i < 4; ++i) s_rank.bg[i] = static_cast<int8_t>(4 - prio[i]);
    if (split) for (int r = 0; r < n; ++r) s_rank.bg[idx[r]] = static_cast<int8_t>(4 - r);
    const int8_t back = n ? s_rank.bg[idx[n - 1]] : 4;
    for (int p = 0; p < 4; ++p) {
        // Hardware placement: with the frontmost BG the OBJ is drawn above (priority >= p).
        int8_t plane = back;
        for (int r = 0; r < n; ++r) if (prio[idx[r]] >= p) { plane = s_rank.bg[idx[r]]; break; }
        if (g_layered_obj_forward) {
            // Forward: join the nearest BG drawn in FRONT of the OBJ (priority < p), so a
            // character stands on the terrain plane instead of the back wall it is drawn over.
            // Per-plane key ordering keeps that BG covering the OBJ where the GBA does.
            int8_t fwd = -1;
            for (int r = n - 1; r >= 0; --r) if (prio[idx[r]] < p) { fwd = s_rank.bg[idx[r]]; break; }
            if (fwd >= 0) plane = fwd;
        }
        s_rank.obj[p] = plane;
    }
    s_rank.active = true;
}

// g_layered_parallax_depth (> 0 enables, scale): estimate each BG's scroll speed relative to
// the fastest-moving BG (the play layer) and push the plane holding a slower BG back to the
// distance that parallax implies (ratio r -> r^-1 times as far). Writes metres per plane
// into g_layered_plane_depth_bias (host adds it behind the plane's normal offset).
struct ParallaxEstimate {
    int prev_h[4] = {}, prev_v[4] = {};
    bool have_prev = false;
    uint16_t prev_cnt[4] = {0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF};
    float ratio[4] = {1, 1, 1, 1};
    float axis_ratio[2][4] = {{1, 1, 1, 1}, {1, 1, 1, 1}};
    bool axis_known[2][4] = {};
    int line_h[4] = {}, line_v[4] = {};
    uint16_t line_dispcnt = 0;
    uint16_t line_cnt[4] = {};
};
static ParallaxEstimate s_parallax;
// Highest plane that received split-off decoration this frame (-1 none), for g_layered_split_push.
static int s_split_target_now = -1, s_split_target_frame = -1;
static float s_parallax_only_bias[5] = {};  // bias from parallax alone (split push excluded)

static void sample_parallax_line(uint16_t dispcnt, const uint8_t* io) {
    s_parallax.line_dispcnt = dispcnt;
    for (int i = 0; i < 4; ++i) {
        s_parallax.line_h[i] = (io[0x10 + 4 * i] | (io[0x11 + 4 * i] << 8)) & 0x1FF;
        s_parallax.line_v[i] = (io[0x12 + 4 * i] | (io[0x13 + 4 * i] << 8)) & 0x1FF;
        s_parallax.line_cnt[i] = static_cast<uint16_t>(io[0x08 + 2 * i] | (io[0x09 + 2 * i] << 8));
    }
}

static void update_parallax_depth(float screen_distance_m) {
    ParallaxEstimate& e = s_parallax;
    const uint16_t dispcnt = e.line_dispcnt;
    const bool on = g_layered_parallax_depth > 0.0f && (dispcnt & 7u) == 0;
    // A BG whose configuration changes (other tiles/map = another scene) or that is switched
    // off forgets its estimate; a room change that keeps the layer keeps its learned distance.
    for (int i = 0; i < 4; ++i) {
        const uint16_t key = static_cast<uint16_t>((e.line_cnt[i] & 0xFFFCu) | ((dispcnt >> (8 + i)) & 1u));
        if (!on || key != e.prev_cnt[i]) {
            e.ratio[i] = 1.0f;
            e.axis_ratio[0][i] = e.axis_ratio[1][i] = 1.0f;
            e.axis_known[0][i] = e.axis_known[1][i] = false;
        }
        e.prev_cnt[i] = key;
    }
    if (!on) e.have_prev = false;
    if (on && e.have_prev) {
        auto wrap = [](int d) { d &= 0x1FF; return d >= 256 ? d - 512 : d; };
        int dh[4], dv[4];
        for (int i = 0; i < 4; ++i) {
            dh[i] = wrap(e.line_h[i] - e.prev_h[i]);
            dv[i] = wrap(e.line_v[i] - e.prev_v[i]);
        }
        // Per axis: the fastest-moving enabled BG is the reference (the play layer). Learn
        // only while it moves a believable amount (not a cut). A layer pinned on one axis
        // (WL4 backgrounds that scroll only horizontally) must not read as "infinitely far",
        // so a BG's distance comes from its NEAREST learned axis.
        for (int axis = 0; axis < 2; ++axis) {
            const int* d = axis == 0 ? dh : dv;
            int ref = -1, ref_mag = 0;
            for (int i = 0; i < 4; ++i)
                if ((dispcnt & (0x100u << i)) && std::abs(d[i]) > ref_mag) { ref_mag = std::abs(d[i]); ref = i; }
            if (ref < 0 || ref_mag < 1 || ref_mag > 16) continue;
            for (int i = 0; i < 4; ++i) {
                // A BG that does not move on this axis is pinned there (or a static
                // menu/HUD layer), not infinitely far: only real motion teaches a ratio.
                if (!(dispcnt & (0x100u << i)) || d[i] == 0) continue;
                const float r = std::clamp(static_cast<float>(d[i]) / static_cast<float>(d[ref]), 0.0f, 1.0f);
                float& a = e.axis_ratio[axis][i];
                a = e.axis_known[axis][i] ? a + (r - a) * 0.15f : r;
                e.axis_known[axis][i] = true;
            }
        }
        for (int i = 0; i < 4; ++i) {
            float r = 1.0f;
            if (e.axis_known[0][i] && e.axis_known[1][i]) r = std::max(e.axis_ratio[0][i], e.axis_ratio[1][i]);
            else if (e.axis_known[0][i]) r = e.axis_ratio[0][i];
            else if (e.axis_known[1][i]) r = e.axis_ratio[1][i];
            e.ratio[i] = r;
        }
        // The nearest-moving enabled BG defines "the play layer" (ratio 1).
        float top = 0.0f;
        for (int i = 0; i < 4; ++i) if (dispcnt & (0x100u << i)) top = std::max(top, e.ratio[i]);
        if (top > 0.05f) for (int i = 0; i < 4; ++i) e.ratio[i] = std::min(1.0f, e.ratio[i] / top);
    }
    for (int i = 0; i < 4; ++i) { e.prev_h[i] = e.line_h[i]; e.prev_v[i] = e.line_v[i]; }
    e.have_prev = on;
    float bias[5] = {0, 0, 0, 0, 0};
    bool set[5] = {};
    if (on) {
        for (int i = 0; i < 4; ++i) {
            if (!(dispcnt & (0x100u << i))) continue;
            const int prio = e.line_cnt[i] & 3;
            const int plane = s_rank.active ? s_rank.bg[i] : 4 - prio;
            const float r = std::max(e.ratio[i], 0.2f);
            float extra = (1.0f / r - 1.0f) * screen_distance_m * g_layered_parallax_depth;
            extra = std::min(extra, 3.0f);
            if (extra < 0.02f) extra = 0.0f;
            // Several BGs on one plane: the nearest one decides.
            if (!set[plane] || extra < bias[plane]) bias[plane] = extra;
            set[plane] = true;
        }
        // Planes behind a pushed plane go at least as far back (keep the order).
        for (int p = 3; p >= 0; --p) bias[p] = std::max(bias[p], bias[p + 1]);
    }
    // g_layered_split_push: when decoration was split off a gameplay layer, the plane holding
    // it (and everything behind) moves further back so the platforms left in front stand out.
    for (int p = 0; p < 5; ++p) s_parallax_only_bias[p] = bias[p];
    if (g_layered_split_push > 0.0f && s_split_target_frame >= 0) {
        for (int p = 0; p <= s_split_target_frame; ++p) bias[p] += g_layered_split_push;
    }
    for (int p = 0; p < 5; ++p) g_layered_plane_depth_bias[p] = bias[p];
    static const bool log = std::getenv("GBARECOMP_PARALLAX_LOG") != nullptr;
    if (log && on)
        std::fprintf(stderr, "parallax h %d %d %d %d v %d %d %d %d ratio %.2f %.2f %.2f %.2f bias %.2f %.2f %.2f %.2f %.2f\n",
                     e.line_h[0], e.line_h[1], e.line_h[2], e.line_h[3], e.line_v[0], e.line_v[1], e.line_v[2], e.line_v[3],
                     e.ratio[0], e.ratio[1], e.ratio[2], e.ratio[3], bias[0], bias[1], bias[2], bias[3], bias[4]);
}

// Sprite-layer candidates per screen column of the current scanline (best = frontmost OBJ of
// the sprite priority); they never enter the plane buckets, so the scenery under a sprite made
// of several overlapping OBJs is not pushed out of the two-deep buckets.
static std::vector<LayeredCandidate> s_sprite_row;
// g_layered_bg_split_row (game hook): per scanline and BG, 1 = this pixel is decoration painted
// into a gameplay layer; it goes one plane further back (unless that plane is parallax-pushed).
static uint8_t s_split_row[4][GbaPpu::kMaxRenderWidth];
static bool s_split_active[4] = {};

void assign_layered_bucket(std::vector<LayeredCandidate>& bucket_top,
                           std::vector<LayeredCandidate>& bucket_second,
                           uint32_t width,
                           uint32_t x,
                           const LayeredCandidate& cand) {
    if (g_layered_sprite_layer && !s_flat_scene && s_obj_plane_prio >= 1 && cand.layer == 4 &&
        cand.key >= 0 && cand.key < 0x70000000 && (cand.key >> 8) >= 1 && (cand.key >> 8) <= s_obj_plane_prio) {
        if (s_sprite_row.size() != width) s_sprite_row.assign(width, LayeredCandidate{});
        if (cand.key < s_sprite_row[x].key) s_sprite_row[x] = cand;
        return;
    }
    uint32_t plane = 0;
    if (cand.layer != 5 && cand.key < 0x70000000) {
        int prio = cand.key >> 8;
        if (prio < 0) prio = 0;
        else if (prio > 3) prio = 3;
        plane = static_cast<uint32_t>(4 - prio);
        if (s_rank.active) {
            plane = cand.layer < 4 ? static_cast<uint32_t>(s_rank.bg[cand.layer])
                                   : static_cast<uint32_t>(s_rank.obj[prio]);
        }
        // Only BGs the player sprite is drawn over (behind the sprite layer): pushing decoration
        // of a BG drawn in front of the player behind him would change what covers him.
        if (cand.layer < 4 && s_split_active[cand.layer] && x < GbaPpu::kMaxRenderWidth &&
            s_split_row[cand.layer][x] && plane >= 1 && s_parallax_only_bias[plane - 1] == 0.0f &&
            (!g_layered_sprite_layer || s_obj_plane_prio < 1 || prio >= s_obj_plane_prio)) {
            plane -= 1;
            if (static_cast<int>(plane) > s_split_target_now) s_split_target_now = static_cast<int>(plane);
        }
    }
    const size_t idx = static_cast<size_t>(plane) * width + x;
    LayeredCandidate& btop = bucket_top[idx];
    LayeredCandidate& bsec = bucket_second[idx];
    if (cand.key < btop.key) {
        bsec = btop;
        btop = cand;
    } else if (cand.key < bsec.key) {
        bsec = cand;
    }
}

// STOCK 3D "standing characters": a world sprite (OBJ, opaque, not HUD) that is the visible
// top pixel but sits on a floor/prop plane (priority 2-3 -> Plane 1-2) is copied up to Plane 3,
// so characters stand in front of the floor and of standing objects. Only visible pixels move,
// so a character hidden behind a table/roof keeps the game's own occlusion; the sprite pixel
// is MOVED (the lower plane shows what the sprite covered), so the sprite never appears on two
// planes at once. Semi-transparent OBJ (shadows) stay flat. Host-selected (3D NEW only).
// Per-frame map of visible opaque world sprites: plane (4 - priority) or -1. Mother 3
// draws a character from several OBJs with different priorities (Lucas's head over
// furniture, body behind), so stock planes split one character across depths;
// unify_world_objects() puts each touching group of sprite pixels on one plane.
std::vector<int8_t> g_obj_plane_map;
uint32_t g_obj_plane_w = 0, g_obj_plane_h = 0;
template <typename TopVec>
void record_world_objects(const TopVec& top, uint32_t width, uint32_t y, uint32_t height) {
    if (g_obj_plane_w != width || g_obj_plane_h != height) {
        g_obj_plane_w = width; g_obj_plane_h = height;
        g_obj_plane_map.assign(static_cast<size_t>(width) * height, -1);
    }
    if (y >= height) return;
    int8_t* row = g_obj_plane_map.data() + static_cast<size_t>(y) * width;
    for (uint32_t x = 0; x < width; ++x) {
        const auto& t = top[x];
        row[x] = -1;
        if (!t.valid || t.layer != 4 || t.target1 || t.key < 0 || t.key >= 0x70000000) continue;
        const int prio = std::clamp(t.key >> 8, 0, 3);
        if (prio == 0) continue;  // Plane 4: HUD / foreground sprites stay
        row[x] = static_cast<int8_t>(4 - prio);
    }
}

void unify_world_objects(uint8_t* layered, uint32_t width, uint32_t height) {
    if (!g_layered_unify_objects || !layered || g_obj_plane_w != width || g_obj_plane_h != height) return;
    const size_t n = static_cast<size_t>(width) * height, plane_bytes = n * 4u;
    thread_local std::vector<int> comp, queue;
    comp.assign(n, -1);
    for (size_t s0 = 0; s0 < n; ++s0) {
        if (g_obj_plane_map[s0] < 0 || comp[s0] >= 0) continue;
        queue.assign(1, static_cast<int>(s0));
        comp[s0] = static_cast<int>(s0);
        int lo = g_obj_plane_map[s0];
        for (size_t qi = 0; qi < queue.size(); ++qi) {
            const int p = queue[qi];
            lo = std::min<int>(lo, g_obj_plane_map[p]);
            const uint32_t px = p % width, py = p / width;
            const int nb[4] = {px > 0 ? p - 1 : -1, px + 1 < width ? p + 1 : -1,
                               py > 0 ? p - static_cast<int>(width) : -1,
                               py + 1 < height ? p + static_cast<int>(width) : -1};
            for (int q : nb)
                if (q >= 0 && g_obj_plane_map[q] >= 0 && comp[q] < 0) { comp[q] = static_cast<int>(s0); queue.push_back(q); }
        }
        // characters stand at floor depth (Plane 1), like the body pieces the game draws
        // behind furniture; only what they hide on screen is hidden in 3D
        lo = std::min(lo, 1);
        for (int p : queue) {
            const int pl = g_obj_plane_map[p];
            if (pl <= lo) continue;
            uint8_t* src = layered + static_cast<size_t>(pl) * plane_bytes + static_cast<size_t>(p) * 4u;
            std::memcpy(layered + static_cast<size_t>(lo) * plane_bytes + static_cast<size_t>(p) * 4u, src, 4);
            std::memset(src, 0, 4);
        }
    }
}

template <typename TopVec>
void pop_world_objects(std::vector<LayeredCandidate>& bucket_top, std::vector<LayeredCandidate>& bucket_second,
                       const TopVec& top, uint32_t width) {
    if (!g_layered_pop_objects) return;
    for (uint32_t x = 0; x < width; ++x) {
        const auto& t = top[x];
        if (!t.valid || t.layer != 4 || t.target1 || t.key < 0 || t.key >= 0x70000000) continue;
        const int prio = t.key >> 8;
        if (prio < 2) continue;  // already on Plane 3/4
        LayeredCandidate c;
#if !defined(GBARECOMP_SCANLINE_HLE)
        c.rgb[0] = t.rgb[0]; c.rgb[1] = t.rgb[1]; c.rgb[2] = t.rgb[2];
#endif
        c.color = t.color; c.key = t.key; c.layer = t.layer;
        c.target1 = t.target1; c.target2 = t.target2; c.valid = true;
        const size_t src = static_cast<size_t>(4 - prio) * width + x;
        if (bucket_top[src].key == t.key) {  // take the sprite off its own plane
            bucket_top[src] = bucket_second[src];
            bucket_second[src] = LayeredCandidate{};
        }
        const size_t idx = static_cast<size_t>(3) * width + x;
        bucket_second[idx] = bucket_top[idx];
        bucket_top[idx] = c;
    }
}

// Mode 2 sprite plane (in front of the terrain): where a BG on a plane BEHIND the sprite plane
// covers the sprite on the GBA, take the sprite pixel off the sprite plane and give it to that
// BG's plane as the pixel underneath (so a translucent fog/water BG still blends over it).
// Per frame: for each pixel where scenery behind the sprite plane covers a sprite, that
// scenery's plane (else -1). keep_sprites_whole() uses it at latch time.
static std::vector<int8_t> s_cover_plane;
static uint32_t s_cover_w = 0, s_cover_h = 0;
static int s_sprite_plane_latched = -1;

template <typename TopVec>
void fix_sprite_plane_occlusion(std::vector<LayeredCandidate>& bucket_top,
                                std::vector<LayeredCandidate>& bucket_second,
                                const TopVec& top, uint32_t width, uint32_t row, uint32_t height) {
    if (g_layered_obj_plane != 2) return;
    if (s_cover_w != width || s_cover_h != height) {
        s_cover_w = width; s_cover_h = height;
        s_cover_plane.assign(static_cast<size_t>(width) * height, -1);
    }
    if (row < height) std::fill_n(s_cover_plane.begin() + static_cast<size_t>(row) * width, width, int8_t(-1));
    if (row == 0) s_sprite_plane_latched = -1;
    if (!s_rank.active || s_obj_plane_prio < 1) return;
    if (s_rank.bg[0] == 2 && s_rank.obj[0] == 2 && s_rank.obj[3] == 2 && s_rank.bg[3] == 2 && g_layered_flat_bitmap) {
        bool flat = true;
        for (int i = 0; i < 4; ++i) flat = flat && s_rank.bg[i] == 2 && s_rank.obj[i] == 2;
        if (flat) return;
    }
    const int sp = s_rank.obj[s_obj_plane_prio];
    s_sprite_plane_latched = sp;
    for (uint32_t x = 0; x < width; ++x) {
        const size_t sidx = static_cast<size_t>(sp) * width + x;
        LayeredCandidate spr = bucket_top[sidx];
        if (!spr.valid || spr.layer != 4 || (spr.key >> 8) != s_obj_plane_prio) continue;
        const auto& t = top[x];
        if (!t.valid || t.layer >= 4 || t.key >= spr.key) continue;  // sprite (or nothing) on top
        const int tp = s_rank.bg[t.layer];
        if (tp >= sp) continue;  // the covering BG is in front of the sprite plane anyway
        bucket_top[sidx] = bucket_second[sidx];
        bucket_second[sidx] = LayeredCandidate{};
        if (row < height) s_cover_plane[static_cast<size_t>(row) * width + x] = static_cast<int8_t>(tp);
        const size_t tidx = static_cast<size_t>(tp) * width + x;
        if (spr.key < bucket_second[tidx].key) bucket_second[tidx] = spr;
    }
}

// Flat bitmap scenes: a sprite seen through a transparent opening of the picture (nothing but
// the backdrop under it, e.g. WL4's city at the end of the intro tunnel) goes on plane 1,
// behind the picture, instead of on the picture's plane.
template <typename TopVec>
void push_sprites_through_holes(std::vector<LayeredCandidate>& bucket_top,
                                std::vector<LayeredCandidate>& bucket_second,
                                const TopVec& top, const TopVec& second, uint32_t width) {
    if (!s_flat_scene) return;
    for (uint32_t x = 0; x < width; ++x) {
        const auto& t = top[x];
        if (!t.valid || t.layer != 4 || !second[x].valid || second[x].layer != 5) continue;
        const size_t src = static_cast<size_t>(2) * width + x;
        if (bucket_top[src].layer != 4 || bucket_top[src].key != t.key) continue;
        const size_t dst = static_cast<size_t>(1) * width + x;
        bucket_second[dst] = bucket_top[dst];
        bucket_top[dst] = bucket_top[src];
        bucket_top[src] = bucket_second[src];
        bucket_second[src] = LayeredCandidate{};
    }
}

// Mode 2: a sprite that scenery partly covers (walking into a doorway, behind a pillar) must
// not be cut in two at different depths. Each touching group of sprite-plane pixels plus its
// covered pixels: when enough of it is covered, all its visible pixels move back onto the
// covering scenery's plane, so the sprite stays whole and coplanar with what hides it.
void keep_sprites_whole(uint8_t* layered, uint32_t width, uint32_t height) {
    if (g_layered_obj_plane != 2 || !layered || s_cover_w != width || s_cover_h != height) return;
    const int sp = s_sprite_plane_latched;
    static const bool log0 = std::getenv("GBARECOMP_SPRITE_LOG") != nullptr;
    if (log0) std::fprintf(stderr, "keep_sprites_whole sp=%d\n", sp);
    if (sp <= 0 || sp > 4) return;
    const size_t n = static_cast<size_t>(width) * height, plane_bytes = n * 4u;
    const uint8_t* spl = layered + static_cast<size_t>(sp) * plane_bytes;
    thread_local std::vector<int> comp, queue;
    comp.assign(n, -1);
    auto member = [&](size_t i) { return spl[i * 4 + 3] != 0 || s_cover_plane[i] >= 0; };
    for (size_t s0 = 0; s0 < n; ++s0) {
        if (comp[s0] >= 0 || !member(s0)) continue;
        queue.assign(1, static_cast<int>(s0));
        comp[s0] = static_cast<int>(s0);
        int covered = 0, visible = 0, votes[5] = {};
        for (size_t qi = 0; qi < queue.size(); ++qi) {
            const int p = queue[qi];
            if (s_cover_plane[p] >= 0) { ++covered; ++votes[s_cover_plane[p]]; } else ++visible;
            const int px = p % static_cast<int>(width), py = p / static_cast<int>(width);
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    const int nx = px + dx, ny = py + dy;
                    if (nx < 0 || ny < 0 || nx >= static_cast<int>(width) || ny >= static_cast<int>(height)) continue;
                    const size_t q = static_cast<size_t>(ny) * width + nx;
                    if (comp[q] >= 0 || !member(q)) continue;
                    comp[q] = static_cast<int>(s0);
                    queue.push_back(static_cast<int>(q));
                }
        }
        static const bool log = std::getenv("GBARECOMP_SPRITE_LOG") != nullptr;
        if (log) std::fprintf(stderr, "sprite group sp=%d covered=%d visible=%d votes=%d,%d,%d,%d,%d\n",
                              sp, covered, visible, votes[0], votes[1], votes[2], votes[3], votes[4]);
        // Any real cut (a few hidden pixels count: a hand behind a wall edge already reads as
        // two layers in VR) sends the whole group back onto the covering scenery's plane.
        if (covered < 4) continue;
        int dest = 0;
        for (int p = 1; p < 5; ++p) if (votes[p] > votes[dest]) dest = p;
        if (dest >= sp) continue;
        uint8_t* dpl = layered + static_cast<size_t>(dest) * plane_bytes;
        uint8_t* srcp = layered + static_cast<size_t>(sp) * plane_bytes;
        for (int p : queue) {
            if (!srcp[p * 4 + 3]) continue;
            std::memcpy(dpl + p * 4, srcp + p * 4, 4);
            std::memset(srcp + p * 4, 0, 4);
        }
    }
}

// g_layered_sprite_layer: the world sprites (priority s_obj_plane_prio) leave the GBA planes and
// go to a separate sixth layer the host shows at g_layered_sprite_depth (plane units). Only
// pixels the GBA shows go there (also under a translucent BG such as fog, which then stays in
// front of the sprite layer); pixels scenery hides are dropped and remembered so a sprite cut
// by scenery can be put back whole onto that scenery's plane at latch time.
static std::vector<uint8_t> s_sprite_work, s_sprite_latched;
static std::vector<int8_t> s_sprite_cover;
static uint32_t s_sprite_w = 0, s_sprite_h = 0;
static bool s_sprite_has_latched = false;

template <typename TopVec>
void extract_sprite_layer(std::vector<LayeredCandidate>& bucket_top,
                          std::vector<LayeredCandidate>& bucket_second,
                          const TopVec& top, const TopVec& second,
                          uint32_t width, uint32_t row, uint32_t height) {
    if (!g_layered_sprite_layer) return;
    if (s_sprite_w != width || s_sprite_h != height) {
        s_sprite_w = width; s_sprite_h = height;
        s_sprite_work.assign(static_cast<size_t>(width) * height * 4u, 0);
        s_sprite_cover.assign(static_cast<size_t>(width) * height, -1);
    }
    if (row >= height) return;
    uint8_t* out = s_sprite_work.data() + static_cast<size_t>(row) * width * 4u;
    int8_t* cov = s_sprite_cover.data() + static_cast<size_t>(row) * width;
    std::memset(out, 0, width * 4u);
    std::fill_n(cov, width, int8_t(-1));
    const int op = s_obj_plane_prio;
    if (op < 1 || s_flat_scene) return;
    const bool ranked = s_rank.active;
    auto is_sprite = [&](const LayeredCandidate& c) {
        return c.valid && c.layer == 4 && c.key >= 0 && c.key < 0x70000000 && (c.key >> 8) >= 1 && (c.key >> 8) <= op;
    };
    if (s_sprite_row.size() != width) { s_sprite_row.assign(width, LayeredCandidate{}); return; }
    for (uint32_t x = 0; x < width; ++x) {
        const LayeredCandidate spr = s_sprite_row[x];
        s_sprite_row[x] = LayeredCandidate{};  // ready for the next scanline
        if (!is_sprite(spr)) continue;
        const auto& t = top[x];
        bool visible = false;
        if (t.valid && t.layer == 4 && t.key == spr.key) visible = true;
        else if (t.valid && t.layer < 4 && t.target1 && second[x].valid && second[x].layer == 4 &&
                 second[x].key == spr.key && second[x].target2) {
            // Under a translucent BG (fog, water): the BG's plane blends with the sprite, as
            // the GBA does, instead of with whatever lies on the planes behind it.
            visible = true;
            const int tp = ranked ? s_rank.bg[t.layer] : 4 - (t.key >> 8);
            const size_t tidx = static_cast<size_t>(tp) * width + x;
            // Only when that BG is shown in front of the sprite layer; a translucent BG behind
            // it (e.g. Fiery Cavern's heat haze) cannot tint the sprite in VR, so the sprite shows
            // untinted and the haze blends with what lies behind it.
            if (!g_layered_fog_alpha && static_cast<float>(tp) > s_line_sprite_depth && bucket_top[tidx].key == t.key)
                bucket_second[tidx] = spr;
        }
        if (visible) {
#if defined(GBARECOMP_SCANLINE_HLE)
            to_rgb888(spr.color, out + x * 4u);
#else
            out[x * 4u + 0] = spr.rgb[0]; out[x * 4u + 1] = spr.rgb[1]; out[x * 4u + 2] = spr.rgb[2];
#endif
            out[x * 4u + 3] = 255;
        } else {
            // Hidden on the GBA. Scenery shown BEHIND the sprite layer that covers the sprite is a
            // cut (handled at latch). Anything in FRONT of the sprite layer (floor, fog, HUD) hides
            // these pixels from straight ahead anyway, so the sprite keeps them: looking around in
            // VR then shows a whole character behind the floor edge instead of cut-off feet.
            int cp = 5;
            if (t.valid && t.layer < 4) cp = ranked ? s_rank.bg[t.layer] : 4 - (t.key >> 8);
            if (static_cast<float>(cp) < s_line_sprite_depth) {
                cov[x] = static_cast<int8_t>(cp);
            } else {
#if defined(GBARECOMP_SCANLINE_HLE)
                to_rgb888(spr.color, out + x * 4u);
#else
                out[x * 4u + 0] = spr.rgb[0]; out[x * 4u + 1] = spr.rgb[1]; out[x * 4u + 2] = spr.rgb[2];
#endif
                out[x * 4u + 3] = 255;
            }
        }
    }
}

// Latch: a sprite group (touching visible + hidden pixels) that scenery cuts goes back whole
// onto the covering scenery's plane, so it is never shown in two pieces at two depths.
void latch_sprite_layer(uint8_t* layered, uint32_t width, uint32_t height) {
    s_sprite_has_latched = false;
    if (!g_layered_sprite_layer || !layered || s_sprite_w != width || s_sprite_h != height) return;
    const size_t n = static_cast<size_t>(width) * height, plane_bytes = n * 4u;
    uint8_t* spl = s_sprite_work.data();
    thread_local std::vector<int> comp, queue;
    comp.assign(n, -1);
    auto member = [&](size_t i) { return spl[i * 4 + 3] != 0 || s_sprite_cover[i] >= 0; };
    for (size_t s0 = 0; s0 < n; ++s0) {
        if (comp[s0] >= 0 || !member(s0)) continue;
        queue.assign(1, static_cast<int>(s0));
        comp[s0] = static_cast<int>(s0);
        int covered = 0, votes[5] = {};
        for (size_t qi = 0; qi < queue.size(); ++qi) {
            const int p = queue[qi];
            if (s_sprite_cover[p] >= 0) { ++covered; ++votes[s_sprite_cover[p]]; }
            const int px = p % static_cast<int>(width), py = p / static_cast<int>(width);
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    const int nx = px + dx, ny = py + dy;
                    if (nx < 0 || ny < 0 || nx >= static_cast<int>(width) || ny >= static_cast<int>(height)) continue;
                    const size_t q = static_cast<size_t>(ny) * width + nx;
                    if (comp[q] >= 0 || !member(q)) continue;
                    comp[q] = static_cast<int>(s0);
                    queue.push_back(static_cast<int>(q));
                }
        }
        if (covered < 4) continue;
        // Feet resting on the ground: grass and sand edges overlapping the lowest rows of a
        // standing sprite are contact, not occlusion. Only cover above the bottom 8 rows of
        // the group (a wall edge across the body, a door frame) sends it back.
        int max_y = 0;
        for (int p : queue) max_y = std::max(max_y, p / static_cast<int>(width));
        int body_cover = 0;
        for (int p = 0; p < 5; ++p) votes[p] = 0;
        for (int p : queue)
            if (s_sprite_cover[p] >= 0 && p / static_cast<int>(width) < max_y - 7) { ++body_cover; ++votes[s_sprite_cover[p]]; }
        if (body_cover < 4) continue;
        int dest = 0;
        for (int p = 1; p < 5; ++p) if (votes[p] > votes[dest]) dest = p;
        uint8_t* dpl = layered + static_cast<size_t>(dest) * plane_bytes;
        for (int p : queue) {
            if (!spl[p * 4 + 3]) continue;
            std::memcpy(dpl + p * 4, spl + p * 4, 4);
            std::memset(spl + p * 4, 0, 4);
        }
    }
    s_sprite_latched = s_sprite_work;
    s_sprite_has_latched = true;
    g_layered_sprite_depth_live = s_line_sprite_depth;
}

void write_layered_scanline_planes(uint8_t* layered_rgba,
                                 uint32_t y_out,
                                 uint32_t width,
                                 uint32_t height,
                                 uint32_t effect,
                                 uint16_t bldalpha,
                                 uint32_t bldy,
                                 const std::vector<LayeredCandidate>& bucket_top,
                                 const std::vector<LayeredCandidate>& bucket_second,
                                 const bool* blackout_cols = nullptr) {
    if (!layered_rgba || y_out >= height) return;
    const uint32_t eva = std::min<uint32_t>(bldalpha & 0x1Fu, 16u);
    const uint32_t evb = std::min<uint32_t>((bldalpha >> 8) & 0x1Fu, 16u);
    const size_t plane_stride = static_cast<size_t>(width) * height * 4u;
    const size_t row_off = static_cast<size_t>(y_out) * width * 4u;
    auto write_candidate_rgb = [](const LayeredCandidate& c, uint8_t* out) {
#if defined(GBARECOMP_SCANLINE_HLE)
        to_rgb888(c.color, out);
#else
        out[0] = c.rgb[0];
        out[1] = c.rgb[1];
        out[2] = c.rgb[2];
#endif
    };

    for (uint32_t p = 0; p < GbaPpu::kLayeredPlanes; ++p) {
        uint8_t* plane_row = layered_rgba + p * plane_stride + row_off;
        for (uint32_t x = 0; x < width; ++x) {
            uint8_t* dst = plane_row + x * 4u;
            if (blackout_cols && blackout_cols[x]) {
                dst[0] = 0;
                dst[1] = 0;
                dst[2] = 0;
                dst[3] = (p == 0) ? 255u : 0u;
                continue;
            }
            const size_t idx = static_cast<size_t>(p) * width + x;
            const LayeredCandidate& btop = bucket_top[idx];
            if (!btop.valid) {
                dst[0] = 0;
                dst[1] = 0;
                dst[2] = 0;
                dst[3] = 0;
                continue;
            }

            const LayeredCandidate& bsec = bucket_second[idx];
            const LayeredCandidate* below = bsec.valid ? &bsec : nullptr;
            const bool same_plane_below = (below != nullptr);
            if (!below && p > 0) {
                for (int q = static_cast<int>(p) - 1; q >= 0; --q) {
                    const LayeredCandidate& cand =
                        bucket_top[static_cast<size_t>(q) * width + x];
                    if (cand.valid) {
                        below = &cand;
                        break;
                    }
                }
            }

            if ((effect == 1u || btop.layer == 4u) &&
                btop.target1 && below && below->target2 &&
                !(btop.layer == 4u && below->layer == 4u)) {
                if (g_layered_fog_alpha && !same_plane_below) {
                    // Translucent BG over other planes: its own see-through layer instead of baked
                    // into the planes behind. PREMULTIPLIED (planes are premultiplied whenever
                    // g_layered_fog_alpha is on): colour = src * EVA/16, alpha = 1 - EVB/16, so the
                    // compositor's src + dst*(1-a) reproduces the GBA blend src*EVA + dst*EVB,
                    // including the additive glow (EVB = 16 -> alpha 0) WL4 uses for mist.
                    uint8_t c[3];
                    write_candidate_rgb(btop, c);
                    for (int k = 0; k < 3; ++k) dst[k] = static_cast<uint8_t>((c[k] * eva + 8u) / 16u);
                    dst[3] = static_cast<uint8_t>(255u - std::min<uint32_t>(255u, (evb * 255u + 8u) / 16u));
                } else if (!same_plane_below && eva == 0u && evb == 16u) {
                    dst[0] = 0;
                    dst[1] = 0;
                    dst[2] = 0;
                    dst[3] = 0;
                } else if (!same_plane_below && evb > 0u && (eva + evb) == 16u) {
                    write_candidate_rgb(btop, dst);
                    dst[3] = static_cast<uint8_t>(
                        std::min<uint32_t>(255u, (eva * 255u + 8u) / 16u));
                } else {
                    const uint16_t blended = blend_alpha_gba555(
                        btop.color, below->color,
                        bldalpha & 0x1Fu, (bldalpha >> 8) & 0x1Fu);
                    to_rgb888(blended, dst);
                    dst[3] = 255;
                }
            } else if ((effect == 2u || effect == 3u) && bldy != 0u && btop.target1) {
                const uint16_t adjusted = (effect == 2u)
                    ? brighten_gba555(btop.color, bldy)
                    : darken_gba555(btop.color, bldy);
                to_rgb888(adjusted, dst);
                dst[3] = 255;
            } else {
                write_candidate_rgb(btop, dst);
                dst[3] = 255;
            }
        }
    }
}

inline void ensure_optb_cap_size(OptionBCap& cap, uint32_t w, uint32_t h) {
    if (cap.width == w && cap.height == h &&
        cap.pixels.size() == static_cast<size_t>(w) * h &&
        cap.lines.size() == h) {
        return;
    }
    cap.width = w;
    cap.height = h;
    cap.pixels.assign(static_cast<size_t>(w) * h, OptionBCap::Pixel{});
    cap.lines.assign(h, OptionBCap::Line{});
}

inline void clear_optb_cap_row(OptionBCap* cap, uint32_t y_out, uint32_t width,
                               uint16_t dispcnt, uint16_t backdrop_color) {
    if (!cap || y_out >= cap->height || width > cap->width) return;
    OptionBCap::Pixel* prow = cap->pixels.data() + static_cast<size_t>(y_out) * cap->width;
    std::fill_n(prow, width, OptionBCap::Pixel{});
    OptionBCap::Line& line = cap->lines[y_out];
    line = OptionBCap::Line{};
    line.dispcnt = dispcnt;
    line.backdrop_color = backdrop_color;
}

inline void record_optb_submit(OptionBCap::Pixel* prow, uint32_t x, uint16_t color,
                               int key, uint8_t layer, bool target1) {
    if (!prow) return;
    OptionBCap::Pixel& p = prow[x];
    if (layer < 4u) {
        const uint8_t bit = static_cast<uint8_t>(1u << layer);
        p.bg_color[layer] = color;
        p.bg_mask |= bit;
        if (target1) p.bg_t1_mask |= bit;
        else p.bg_t1_mask &= static_cast<uint8_t>(~bit);
        const uint32_t prio = static_cast<uint32_t>((key >> 8) & 3);
        p.bg_prio = static_cast<uint8_t>((p.bg_prio & ~(3u << (2u * layer))) | (prio << (2u * layer)));
    } else if (layer == 4u) {
        const uint32_t prio = static_cast<uint32_t>((key >> 8) & 3);
        const uint8_t idx = static_cast<uint8_t>(key & 0x7Fu);
        const int new_key = static_cast<int>(prio * 256u + idx);
        const int cur_key = (p.obj_idx == 0xFFu)
            ? 0x7FFFFFFF
            : static_cast<int>(p.obj_prio * 256u + p.obj_idx);
        if (new_key <= cur_key) {
            p.obj_color = color;
            p.obj_idx = idx;
            p.obj_prio = static_cast<uint8_t>(prio);
            if (target1) p.flags |= 1u;
            else p.flags &= static_cast<uint8_t>(~1u);
        }
    }
}

void render_scanline_internal(uint8_t* rgb,
                              uint32_t y,
                              uint16_t dispcnt,
                              const uint8_t* io,
                              const uint8_t* vram,
                              const uint8_t* oam,
                              const uint8_t* pal,
                              uint32_t kScreenWidth,
                              uint32_t kScreenHeight,
                              uint8_t* layered_rgba = nullptr,
                              OptionBCap* optb_cap = nullptr) {
    if (y >= kScreenHeight) return;

    OptionBCap::Pixel* optb_row = nullptr;
    if (optb_cap && y < optb_cap->height && kScreenWidth <= optb_cap->width) {
        optb_row = optb_cap->pixels.data() + static_cast<size_t>(y) * optb_cap->width;
        std::fill_n(optb_row, kScreenWidth, OptionBCap::Pixel{});
    }

    uint8_t* row = rgb + y * kScreenWidth * 3;
    if (dispcnt & 0x0080u) {
        std::memset(row, 0xFF, kScreenWidth * 3);
        fill_tags(kScreenWidth);
        clear_layered_scanline_row(layered_rgba, y, kScreenWidth, kScreenHeight, 0xFFu);
        clear_optb_cap_row(optb_cap, y, kScreenWidth, dispcnt, 0x7FFFu);
        return;
    }

    using PixelCandidate = LayeredCandidate;

    // ── Windows (GBATEK "LCD I/O Windows") ──────────────────────────────
    // window_control(x) returns the 6-bit per-pixel mask (BG0-3, OBJ, and bit5
    // = color-special-effect enable) chosen by region, priority WIN0 > WIN1 >
    // OBJ-window > outside. WIN0/WIN1 are COORDINATE windows: WINnH gives
    // [X1,X2), WINnV gives [Y1,Y2). The per-scanline WIN0H the game rewrites
    // each line is what carves the circular IRIS used by room transitions — it
    // blanks everything outside the circle to the backdrop, hiding the room's
    // VRAM/palette reload (MC-HP-003). With no window enabled, every layer is on.
    const bool win0_en   = (dispcnt & 0x2000u) != 0;
    const bool win1_en   = (dispcnt & 0x4000u) != 0;
    const bool objwin_en = (dispcnt & 0x8000u) != 0;
    const bool any_window = win0_en || win1_en || objwin_en;
    uint16_t winin  = static_cast<uint16_t>(io[0x48] | (io[0x49] << 8));
    uint16_t winout = static_cast<uint16_t>(io[0x4A] | (io[0x4B] << 8));
    // y within WINnV [Y1,Y2)? GBATEK: Y2>height or Y1>Y2 → Y2=height.
    auto win_v_row = [&](uint32_t vreg) -> bool {
        uint32_t v  = static_cast<uint32_t>(io[vreg] | (io[vreg + 1] << 8));
        uint32_t y1 = (v >> 8) & 0xFFu, y2 = v & 0xFFu;
        if (y2 > kScreenHeight || y1 > y2) y2 = kScreenHeight;
        return y >= y1 && y < y2;
    };
    // x within WINnH [X1,X2)? GBATEK: X2>width or X1>X2 → X2=width.
    auto win_h_in = [&](uint32_t hreg, uint32_t x) -> bool {
        uint32_t h  = static_cast<uint32_t>(io[hreg] | (io[hreg + 1] << 8));
        uint32_t x1 = (h >> 8) & 0xFFu, x2 = h & 0xFFu;
        if (x2 > kScreenWidth || x1 > x2) x2 = kScreenWidth;
        return x >= x1 && x < x2;
    };
    const bool win0_row = win0_en && win_v_row(0x44);
    const bool win1_row = win1_en && win_v_row(0x46);
    bool obj_window_storage[GbaPpu::kScreenWidth] = {};
    bool* obj_window_mask = nullptr;
    if (objwin_en) {
        mark_obj_window_scanline(obj_window_storage, y, dispcnt, vram, oam,
                                 kScreenWidth, kScreenHeight);
        obj_window_mask = obj_window_storage;
    }
    auto window_control = [&](uint32_t x) -> uint16_t {
        if (!any_window) return 0x3Fu;
        if (win0_row && win_h_in(0x40, x)) return winin & 0x3Fu;
        if (win1_row && win_h_in(0x42, x)) return (winin >> 8) & 0x3Fu;
        if (obj_window_mask && obj_window_mask[x])
            return static_cast<uint16_t>((winout >> 8) & 0x3Fu);
        return static_cast<uint16_t>(winout & 0x3Fu);
    };
#if defined(GBARECOMP_SCANLINE_HLE)
    // The caller gives an immutable row snapshot. Resolve windows once, not
    // once per candidate/layer. No cross-row state or VRAM cache is retained.
    uint8_t row_control[GbaPpu::kScreenWidth];
    for (uint32_t x = 0; x < kScreenWidth; ++x)
        row_control[x] = static_cast<uint8_t>(window_control(x));
#endif
    auto layer_enabled = [&](uint32_t x, uint32_t layer_bit) -> bool {
        return (
#if defined(GBARECOMP_SCANLINE_HLE)
                row_control[x]
#else
                window_control(x)
#endif
                & (1u << layer_bit)) != 0;
    };
    auto blend_enabled = [&](uint32_t x) -> bool {
        return (
#if defined(GBARECOMP_SCANLINE_HLE)
                row_control[x]
#else
                window_control(x)
#endif
                & (1u << 5)) != 0;
    };

    uint16_t bldcnt = static_cast<uint16_t>(io[0x50] | (io[0x51] << 8));
    uint16_t bldalpha = static_cast<uint16_t>(io[0x52] | (io[0x53] << 8));
    uint32_t first_targets = bldcnt & 0x3Fu;
    uint32_t second_targets = (bldcnt >> 8) & 0x3Fu;
    uint32_t effect = (bldcnt >> 6) & 0x3u;

    PixelCandidate top[GbaPpu::kScreenWidth];
    PixelCandidate second[GbaPpu::kScreenWidth];
    thread_local std::vector<LayeredCandidate> bucket_top;
    thread_local std::vector<LayeredCandidate> bucket_second;
    if (layered_rgba) {
        const size_t bucket_count =
            static_cast<size_t>(GbaPpu::kLayeredPlanes) * kScreenWidth;
        bucket_top.assign(bucket_count, LayeredCandidate{});
        bucket_second.assign(bucket_count, LayeredCandidate{});
    }
    // This is deliberately a per-layer candidate, not a copy of `top`: a
    // room BG with higher priority may cover the HUD's BG0 cells. Capturing
    // only the explicitly declared source map cells keeps that room terrain
    // out of the foreign frame.
    PixelCandidate foreign_hud_bg[GbaPpu::kScreenWidth] = {};
    const GbaForeignObjFocusTransform* foreign_focus =
        foreign_presentation_internal::obj_focus();
    const uint16_t backdrop_color = load_u16_le(&pal[0]);
#if !defined(GBARECOMP_SCANLINE_HLE)
    uint8_t backdrop_rgb[3];
    to_rgb888(backdrop_color, backdrop_rgb);
#endif
    for (uint32_t x = 0; x < kScreenWidth; ++x) {
#if !defined(GBARECOMP_SCANLINE_HLE)
        top[x].rgb[0] = backdrop_rgb[0];
        top[x].rgb[1] = backdrop_rgb[1];
        top[x].rgb[2] = backdrop_rgb[2];
#endif
        top[x].color = backdrop_color;
        top[x].key = 0x70000000;
        top[x].layer = 5;
        top[x].target1 = blend_enabled(x) && ((first_targets & (1u << 5)) != 0);
        top[x].target2 = (second_targets & (1u << 5)) != 0;
        top[x].valid = true;
        if (layered_rgba) bucket_top[x] = top[x];
    }
    auto submit = [&](uint32_t x,
                      uint16_t color,
                      int key,
                      uint8_t layer,
                      bool target1,
                      bool target2) {
        if (optb_row) record_optb_submit(optb_row, x, color, key, layer, target1);
        PixelCandidate cand;
        cand.color = color;
#if !defined(GBARECOMP_SCANLINE_HLE)
        to_rgb888(color, cand.rgb);
#endif
        cand.key = key;
        cand.layer = layer;
        cand.target1 = target1;
        cand.target2 = target2;
        cand.valid = true;
        if (key < top[x].key) {
            second[x] = top[x];
            top[x] = cand;
        } else if (key < second[x].key) {
            second[x] = cand;
        }
        if (layered_rgba) {
            assign_layered_bucket(bucket_top, bucket_second, kScreenWidth, x, cand);
        }
    };

    uint32_t bg_mode = dispcnt & 0x07u;
    auto render_regular_bg = [&](uint32_t layer,
                                 uint32_t cnt_off,
                                 uint32_t scroll_off) {
        if ((dispcnt & (0x0100u << layer)) == 0) return;

        uint16_t bgcnt = static_cast<uint16_t>(
            io[cnt_off] | (io[cnt_off + 1] << 8));
        uint32_t char_base = ((bgcnt >> 2) & 0x3u) * 0x4000u;
        uint32_t screen_base = ((bgcnt >> 8) & 0x1Fu) * 0x800u;
        bool color256 = (bgcnt & 0x0080u) != 0;
        uint32_t size_code = (bgcnt >> 14) & 0x3u;
        uint32_t bg_priority = bgcnt & 0x3u;
        uint32_t hofs = static_cast<uint16_t>(
            io[scroll_off] | (io[scroll_off + 1] << 8)) & 0x01FFu;
        uint32_t vofs = static_cast<uint16_t>(
            io[scroll_off + 2] | (io[scroll_off + 3] << 8)) & 0x01FFu;

        uint32_t width_tiles = (size_code & 1u) ? 64u : 32u;
        uint32_t height_tiles = (size_code & 2u) ? 64u : 32u;
        uint32_t width_px = width_tiles * 8u;
        uint32_t height_px = height_tiles * 8u;
        uint32_t block_cols = width_tiles / 32u;

#if defined(GBARECOMP_SCANLINE_HLE)
        const uint32_t tex_y = (y + vofs) & (height_px - 1u);
        const uint32_t tile_y = tex_y >> 3;
        const uint32_t map_row = screen_base + (tile_y >> 5) * block_cols * 0x800u +
                                 (tile_y & 31u) * 64u;
        const int key = static_cast<int>(bg_priority * 256u + 128u + layer);
        const bool target1 = (first_targets & (1u << layer)) != 0;
        const bool target2 = (second_targets & (1u << layer)) != 0;
        for (uint32_t x = 0; x < kScreenWidth;) {
            const uint32_t tex_x = (x + hofs) & (width_px - 1u);
            const uint32_t tile_x = tex_x >> 3;
            const uint32_t begin = tex_x & 7u;
            const uint32_t count = std::min(8u - begin, kScreenWidth - x);
            const uint32_t map_off = map_row + (tile_x >> 5) * 0x800u +
                                     (tile_x & 31u) * 2u;
            if (map_off + 1u >= 96u * 1024u) { x += count; continue; }
            const uint16_t entry = load_u16_le(&vram[map_off]);
            const uint32_t py = (entry & 0x0800u) ? 7u - (tex_y & 7u) : tex_y & 7u;
            const uint32_t tile_row = char_base + (entry & 0x03FFu) *
                (color256 ? 64u : 32u) + py * (color256 ? 8u : 4u);
            const uint32_t palette_bank = (entry >> 12) & 15u;
            for (uint32_t j = 0; j < count; ++j) {
                const uint32_t screen_x = x + j;
                if (!layer_enabled(screen_x, layer)) continue;
                const uint32_t px = (entry & 0x0400u) ? 7u - (begin + j) : begin + j;
                const uint32_t addr = tile_row + (color256 ? px : px >> 1);
                if (addr >= 96u * 1024u) continue;
                const uint8_t packed = vram[addr];
                uint32_t index = color256 ? packed : ((px & 1u) ? packed >> 4 : packed & 15u);
                if (!index) continue;
                if (!color256) index |= palette_bank << 4;
                const uint16_t color = load_u16_le(&pal[index * 2]);
                const bool t1 = blend_enabled(screen_x) && target1;
                if (foreign_hud_bg_map_contains(foreign_focus, layer, tile_x,
                                                tile_y, screen_x, y)) {
                    PixelCandidate& hud = foreign_hud_bg[screen_x];
                    hud.color = color; hud.key = key;
                    hud.layer = static_cast<uint8_t>(layer);
                    hud.target1 = t1; hud.target2 = target2; hud.valid = true;
                }
                submit(screen_x, color, key, static_cast<uint8_t>(layer), t1, target2);
            }
            x += count;
        }
#else
        for (uint32_t x = 0; x < kScreenWidth; ++x) {
            if (!layer_enabled(x, layer)) continue;
            uint32_t tex_x = (x + hofs) & (width_px - 1u);
            uint32_t tex_y = (y + vofs) & (height_px - 1u);
            uint32_t tile_x = tex_x >> 3;
            uint32_t tile_y = tex_y >> 3;
            uint32_t block = (tile_x >> 5) + (tile_y >> 5) * block_cols;
            uint32_t map_off = screen_base + block * 0x800u +
                ((tile_y & 31u) * 32u + (tile_x & 31u)) * 2u;
            if (map_off + 1 >= 96u * 1024u) continue;

            uint16_t entry = load_u16_le(&vram[map_off]);
            uint32_t tile_num = entry & 0x03FFu;
            bool hflip = (entry & 0x0400u) != 0;
            bool vflip = (entry & 0x0800u) != 0;
            uint32_t palette_bank = (entry >> 12) & 0x0Fu;
            uint32_t px = tex_x & 7u;
            uint32_t py = tex_y & 7u;
            if (hflip) px = 7u - px;
            if (vflip) py = 7u - py;

            uint8_t pal_idx = 0;
            if (color256) {
                uint32_t tile_addr = char_base + tile_num * 64u + py * 8u + px;
                if (tile_addr >= 96u * 1024u) continue;
                pal_idx = vram[tile_addr];
            } else {
                uint32_t tile_addr = char_base + tile_num * 32u +
                    py * 4u + (px >> 1);
                if (tile_addr >= 96u * 1024u) continue;
                uint8_t packed = vram[tile_addr];
                pal_idx = (px & 1u) ? (packed >> 4) : (packed & 0x0Fu);
                pal_idx = static_cast<uint8_t>(
                    pal_idx | static_cast<uint8_t>(palette_bank << 4));
            }
            if ((pal_idx & (color256 ? 0xFFu : 0x0Fu)) == 0) continue;

            const uint16_t color = load_u16_le(&pal[pal_idx * 2]);
            if (foreign_hud_bg_map_contains(foreign_focus, layer, tile_x,
                                            tile_y, x, y)) {
                PixelCandidate& hud = foreign_hud_bg[x];
                hud.color = color;
#if !defined(GBARECOMP_SCANLINE_HLE)
                to_rgb888(color, hud.rgb);
#endif
                hud.key = static_cast<int>(bg_priority * 256u + 128u + layer);
                hud.layer = static_cast<uint8_t>(layer);
                hud.target1 = blend_enabled(x) &&
                    ((first_targets & (1u << layer)) != 0);
                hud.target2 = (second_targets & (1u << layer)) != 0;
                hud.valid = true;
            }
            submit(x, color,
                   static_cast<int>(bg_priority * 256u + 128u + layer),
                   static_cast<uint8_t>(layer),
                   blend_enabled(x) && ((first_targets & (1u << layer)) != 0),
                   (second_targets & (1u << layer)) != 0);
        }
#endif
    };

    auto render_affine_bg = [&](uint32_t layer,
                                uint32_t cnt_off,
                                uint32_t param_off) {
        if ((dispcnt & (0x0100u << layer)) == 0) return;

        uint16_t bgcnt = static_cast<uint16_t>(io[cnt_off] |
                                               (io[cnt_off + 1] << 8));
        uint32_t char_base   = ((bgcnt >> 2) & 0x3u) * 0x4000u;
        uint32_t screen_base = ((bgcnt >> 8) & 0x1Fu) * 0x800u;
        bool wrap            = (bgcnt & 0x2000u) != 0;
        uint32_t size_code   = (bgcnt >> 14) & 0x3u;
        int bg_pixels = 128 << size_code;
        int bg_tiles  = bg_pixels / 8;
        int bg_priority = static_cast<int>(bgcnt & 0x3u);

        int32_t pa = read_s16(io, param_off + 0x00);
        int32_t pb = read_s16(io, param_off + 0x02);
        int32_t pc = read_s16(io, param_off + 0x04);
        int32_t pd = read_s16(io, param_off + 0x06);
        int32_t refx = read_s28_ref(io, param_off + 0x08);
        int32_t refy = read_s28_ref(io, param_off + 0x0C);
        int32_t xt = refx + static_cast<int32_t>(y) * pb;
        int32_t yt = refy + static_cast<int32_t>(y) * pd;
        for (uint32_t x = 0; x < kScreenWidth; ++x) {
            int32_t tex_x = xt >> 8;
            int32_t tex_y = yt >> 8;
            xt += pa;
            yt += pc;
            if (!layer_enabled(x, layer)) continue;
            if (wrap) {
                tex_x &= (bg_pixels - 1);
                tex_y &= (bg_pixels - 1);
            } else if (tex_x < 0 || tex_x >= bg_pixels ||
                       tex_y < 0 || tex_y >= bg_pixels) {
                continue;
            }
            uint32_t map_off = screen_base + (tex_y >> 3) * bg_tiles + (tex_x >> 3);
            if (map_off >= 96u * 1024u) continue;
            uint8_t tile_index = vram[map_off];
            uint32_t tile_addr = char_base + tile_index * 64u +
                                 (tex_y & 7) * 8 + (tex_x & 7);
            if (tile_addr >= 96u * 1024u) continue;
            uint8_t pal_idx = vram[tile_addr];
            if (pal_idx == 0) continue;
            const uint16_t color = load_u16_le(&pal[pal_idx * 2]);
            submit(x, color,
                   static_cast<int>(bg_priority * 256 + 128 + layer),
                   static_cast<uint8_t>(layer),
                   blend_enabled(x) && ((first_targets & (1u << layer)) != 0),
                   (second_targets & (1u << layer)) != 0);
        }
    };

    // ── Bitmap BG modes 3/4/5 (GBATEK "LCD VRAM Bitmap BG Modes") ───────
    // All three present a single framebuffer through BG2's affine transform,
    // so the sampling loop mirrors render_affine_bg above:
    //   Mode 3  240x160, 16bpp BGR555 direct colour, one frame at 0x00000.
    //   Mode 4  240x160, 8bpp palette indices; frame 0 at 0x00000 and
    //           frame 1 at 0x0A000, selected by DISPCNT bit 4.
    //   Mode 5  160x128, 16bpp direct colour, same two frame bases.
    //
    // Transparency: mode 4 is palette-indexed, so index 0 is transparent
    // exactly like the tiled layers. Modes 3/5 are direct colour and have no
    // transparent index, so every in-bounds texel is opaque. Texels outside
    // the bitmap stay transparent and show the backdrop — that is what
    // letterboxes mode 5's 160x128 image inside the 240x160 screen.
    //
    // BG2CNT bit13 (display-area overflow / wraparound) is defined for the
    // TILED affine BGs; a bitmap is not a power-of-two tile map, so an
    // out-of-range texel here is transparent rather than wrapped.
    auto render_bitmap_bg = [&]() {
        if ((dispcnt & 0x0400u) == 0) return;   // DISPCNT bit10 = BG2 enable
        constexpr uint32_t layer = 2;
        uint16_t bgcnt = static_cast<uint16_t>(io[0x0C] | (io[0x0D] << 8));
        int bg_priority = static_cast<int>(bgcnt & 0x3u);

        const bool     direct = (bg_mode != 4);
        const int      bmp_w  = (bg_mode == 5) ? 160 : 240;
        const int      bmp_h  = (bg_mode == 5) ? 128 : 160;
        const uint32_t bpp    = direct ? 2u : 1u;
        const uint32_t frame_base =
            (bg_mode != 3 && (dispcnt & 0x0010u) != 0) ? 0xA000u : 0x0000u;

        int32_t pa   = read_s16(io, 0x20);
        int32_t pb   = read_s16(io, 0x22);
        int32_t pc   = read_s16(io, 0x24);
        int32_t pd   = read_s16(io, 0x26);
        int32_t refx = read_s28_ref(io, 0x28);
        int32_t refy = read_s28_ref(io, 0x2C);
        int32_t xt = refx + static_cast<int32_t>(y) * pb;
        int32_t yt = refy + static_cast<int32_t>(y) * pd;
        for (uint32_t x = 0; x < kScreenWidth; ++x) {
            int32_t tex_x = xt >> 8;
            int32_t tex_y = yt >> 8;
            xt += pa;
            yt += pc;
            if (!layer_enabled(x, layer)) continue;
            if (tex_x < 0 || tex_x >= bmp_w || tex_y < 0 || tex_y >= bmp_h)
                continue;
            const uint32_t off = frame_base +
                (static_cast<uint32_t>(tex_y) * static_cast<uint32_t>(bmp_w) +
                 static_cast<uint32_t>(tex_x)) * bpp;
            if (off + bpp > 96u * 1024u) continue;
            uint16_t color;
            if (direct) {
                color = load_u16_le(&vram[off]);
            } else {
                uint8_t pal_idx = vram[off];
                if (pal_idx == 0) continue;
                color = load_u16_le(&pal[pal_idx * 2]);
            }
            submit(x, color,
                   static_cast<int>(bg_priority * 256 + 128 + layer),
                   static_cast<uint8_t>(layer),
                   blend_enabled(x) && ((first_targets & (1u << layer)) != 0),
                   (second_targets & (1u << layer)) != 0);
        }
    };

    if (bg_mode == 0) {
        render_regular_bg(3, 0x0E, 0x1C);
        render_regular_bg(2, 0x0C, 0x18);
        render_regular_bg(1, 0x0A, 0x14);
        render_regular_bg(0, 0x08, 0x10);
    } else if (bg_mode == 1) {
        render_affine_bg(2, 0x0C, 0x20);
        render_regular_bg(1, 0x0A, 0x14);
        render_regular_bg(0, 0x08, 0x10);
    } else if (bg_mode == 2) {
        render_affine_bg(3, 0x0E, 0x30);
        render_affine_bg(2, 0x0C, 0x20);
    } else if (bg_mode <= 5) {
        // Modes 6/7 are prohibited (GBATEK); leave them backdrop-only.
        render_bitmap_bg();
    }

    // A bounded host-owned overlay is intentionally a native-world-only
    // presentation layer. It is composed after every guest BG candidate has
    // resolved but before the normal guest OBJ pass, so Link/HUD objects remain
    // in front without any guest OAM/VRAM writes. It retains the synthetic
    // BG2 identity for native WIN0/WIN1/OBJ-window and color-effect routing,
    // so room-authored masks and fades still apply. A full foreign background
    // is authoritative and therefore suppresses this overlay automatically.
    if (foreign_presentation_internal::background() == nullptr) {
        if (const auto* overlay = foreign_presentation_internal::screen_overlay();
            valid_foreign_screen_overlay(overlay)) {
            const int left = overlay->x;
            const int top_y = overlay->y;
            for (unsigned oy = 0; oy < overlay->height; ++oy) {
                const int screen_y = top_y + static_cast<int>(oy);
                if (screen_y != static_cast<int>(y)) continue;
                for (unsigned ox = 0; ox < overlay->width; ++ox) {
                    const int screen_x = left + static_cast<int>(ox);
                    if (screen_x < 0 || screen_x >= static_cast<int>(kScreenWidth))
                        continue;
                    const unsigned x = static_cast<unsigned>(screen_x);
                    // This is a native presentation layer, not a window
                    // bypass. Honor the current BG2 mask at the output pixel.
                    if (!layer_enabled(x, 2)) continue;
                    const std::size_t source =
                        static_cast<std::size_t>(oy) * overlay->stride + ox;
                    const std::uint8_t alpha = overlay->alpha_q4[source];
                    // A mutable/corrupt descriptor must fail closed at the
                    // texel boundary as well as at publish time.
                    if (alpha == 0 || alpha > 16) continue;
                    const PixelCandidate old_top = top[x];
                    PixelCandidate cand;
                    cand.color = blend_alpha_gba555(overlay->pixels[source],
                                                     old_top.color, alpha,
                                                     16u - alpha);
#if !defined(GBARECOMP_SCANLINE_HLE)
                    to_rgb888(cand.color, cand.rgb);
#endif
                    // Same synthetic BG2 identity as a full foreign frame;
                    // regular guest OBJ keys are always in front of 0x10000.
                    cand.key = 0x10000;
                    cand.layer = 2;
                    cand.target1 = blend_enabled(x) &&
                        ((first_targets & (1u << 2)) != 0);
                    cand.target2 = (second_targets & (1u << 2)) != 0;
                    cand.valid = true;
                    top[x] = cand;
                    second[x] = old_top;
                }
            }
        }
    }

    // The foreign background is consumed as immutable data, never through a
    // mod callback.  It replaces all native guest BG/backdrop candidates at
    // the last safe point before OBJ composition.  A deliberately back-most
    // key keeps every guest OBJ visible while preserving authentic OBJ order,
    // OBJ-window masking, and the normal BLDCNT effects pipeline.  Treat it
    // as BG2 for color-effect target bits so authored fades still apply.
    // Unlike a guest BG2, a committed foreign frame is intentionally opaque
    // to native BG/window state: keeping a room-authored window mask here
    // leaked screen-fixed house/door tiles over later foreign rooms. Guest
    // OBJ composition (including the focused Link OBJ) still happens below.
    if (const uint16_t* foreign = foreign_presentation_internal::background()) {
        for (uint32_t x = 0; x < kScreenWidth; ++x) {
            if (foreign_hud_bg[x].valid) {
                top[x] = foreign_hud_bg[x];
                second[x] = PixelCandidate{};
                continue;
            }
            PixelCandidate cand;
            cand.color = foreign[y * GbaPpu::kScreenWidth + x];
#if !defined(GBARECOMP_SCANLINE_HLE)
            to_rgb888(cand.color, cand.rgb);
#endif
            cand.key = 0x10000;
            cand.layer = 2;
            cand.target1 = blend_enabled(x) &&
                ((first_targets & (1u << 2)) != 0);
            cand.target2 = (second_targets & (1u << 2)) != 0;
            cand.valid = true;
            top[x] = cand;
            second[x] = PixelCandidate{};
        }
    }

    if (dispcnt & 0x1000u) {
        constexpr uint32_t obj_tile_base = 0x10000u;
        bool obj_1d_mapping = (dispcnt & 0x0040u) != 0;
        const uint8_t* obj_pal = pal + 0x200;
        for (int idx = 127; idx >= 0; --idx) {
            const uint8_t* entry = oam + idx * 8;
            uint16_t attr0 = load_u16_le(&entry[0]);
            uint16_t attr1 = load_u16_le(&entry[2]);
            uint16_t attr2 = load_u16_le(&entry[4]);
            bool rot_scale = (attr0 & 0x0100u) != 0;
            bool disable_or_double = (attr0 & 0x0200u) != 0;
            if (!rot_scale && disable_or_double) continue;
            uint32_t obj_mode = (attr0 >> 10) & 0x3u;
            if (obj_mode == 2 || obj_mode == 3) continue;
            uint32_t shape = (attr0 >> 14) & 0x3u;
            if (shape >= 3) continue;
            uint32_t size  = (attr1 >> 14) & 0x3u;
            int sw = kSpriteWH[shape][size][0];
            int sh = kSpriteWH[shape][size][1];
            int sy = static_cast<int>(attr0 & 0xFFu);
            int sx = static_cast<int>(attr1 & 0x1FFu);
            if (sy >= 160) sy -= 256;
            if (sx & 0x100) sx -= 0x200;
            bool color256 = (attr0 & 0x2000u) != 0;
            uint32_t tile_num = attr2 & 0x3FFu;
            if (bg_mode >= 3 && tile_num < 512u) continue;
            uint32_t palette_bank = (attr2 >> 12) & 0xFu;
            int tiles_w = sw / 8;
            int tiles_h = sh / 8;
            int priority = static_cast<int>((attr2 >> 10) & 0x3u);
            // Composite key (lower = front). Per-priority stride 256 with OBJ in
            // [p*256, p*256+127] (tie-break by OAM idx) and BG in [p*256+128,
            // p*256+131] (tie-break by layer) yields the exact GBA order
            // OBJ0<BG0<OBJ1<BG1<OBJ2<BG2<OBJ3<BG3: an OBJ is in front of a same-
            // priority BG, but a BG of priority p sits in front of any OBJ of
            // priority p+1 (this is what lets the player walk BEHIND roof/tree
            // tops). The stride MUST exceed 128 so OBJ idx (0..127) can't bleed
            // into the next priority's BG band.
            int key = priority * 256 + idx;
            bool obj_target2 = (second_targets & (1u << 4)) != 0;
            auto emit_obj = [&](int tex_x, int tex_y, int screen_x) {
                if (screen_x < 0 || screen_x >= static_cast<int>(kScreenWidth)) return;
                if (!layer_enabled(static_cast<uint32_t>(screen_x), 4)) return;
                int tile_x_in_sprite = tex_x >> 3;
                int tile_y_in_sprite = tex_y >> 3;
                int px_in_tile = tex_x & 7;
                int py_in_tile = tex_y & 7;
                uint32_t this_tile;
                if (obj_1d_mapping) {
                    this_tile = tile_num + (tile_y_in_sprite * tiles_w + tile_x_in_sprite) *
                                            (color256 ? 2u : 1u);
                } else {
                    this_tile = tile_num + (tile_y_in_sprite * 32u) +
                                tile_x_in_sprite * (color256 ? 2u : 1u);
                }
                uint32_t tile_off = obj_tile_base + this_tile * 32u;
                uint8_t pal_index;
                if (color256) {
                    uint32_t off = tile_off + py_in_tile * 8 + px_in_tile;
                    if (off + 1 > 96u * 1024u) return;
                    pal_index = vram[off];
                    if (pal_index == 0) return;
                } else {
                    uint32_t off = tile_off + py_in_tile * 4 + (px_in_tile / 2);
                    if (off + 1 > 96u * 1024u) return;
                    uint8_t b = vram[off];
                    pal_index = (px_in_tile & 1) ? (b >> 4) : (b & 0x0F);
                    if (pal_index == 0) return;
                    pal_index = static_cast<uint8_t>(pal_index | (palette_bank << 4));
                }
                const uint16_t color = load_u16_le(&obj_pal[pal_index * 2]);
                uint32_t ux = static_cast<uint32_t>(screen_x);
                bool t1 = blend_enabled(ux) &&
                    obj_mode == 1;
                submit(ux, color, key, 4, t1, obj_target2);
            };
            if (rot_scale) {
                int bw = disable_or_double ? sw * 2 : sw;
                int bh = disable_or_double ? sh * 2 : sh;
                if (apply_foreign_obj_focus(&sx, &sy, bw, bh, tile_num,
                                            false, static_cast<uint32_t>(idx)).action ==
                    ForeignObjFocusAction::kSuppress) continue;
                int j = static_cast<int>(y) - sy;
                if (j < 0 || j >= bh) continue;
                int affine_group = (attr1 >> 9) & 0x1Fu;
                const uint8_t* ag = oam + affine_group * 0x20u;
                int32_t pa = read_s16(ag, 0x06);
                int32_t pb = read_s16(ag, 0x0E);
                int32_t pc = read_s16(ag, 0x16);
                int32_t pd = read_s16(ag, 0x1E);
                int half_bw = bw >> 1;
                int half_bh = bh >> 1;
                int half_sw = sw >> 1;
                int half_sh = sh >> 1;
                int dy = j - half_bh;
                for (int i = 0; i < bw; ++i) {
                    int dx = i - half_bw;
                    int tex_x = ((pa * dx + pb * dy) >> 8) + half_sw;
                    int tex_y = ((pc * dx + pd * dy) >> 8) + half_sh;
                    if (tex_x < 0 || tex_x >= sw) continue;
                    if (tex_y < 0 || tex_y >= sh) continue;
                    emit_obj(tex_x, tex_y, sx + i);
                }
                continue;
            }
            const ForeignObjFocusResult focus_result =
                apply_foreign_obj_focus(&sx, &sy, sw, sh, tile_num, true,
                                        static_cast<uint32_t>(idx));
            if (focus_result.action == ForeignObjFocusAction::kSuppress) continue;
            const int draw_w = scaled_extent(sw, focus_result.scale_q8_8);
            const int draw_h = scaled_extent(sh, focus_result.scale_q8_8);
            int line = static_cast<int>(y) - sy;
            if (line < 0 || line >= draw_h) continue;
            bool hflip = (attr1 & 0x1000u) != 0;
            bool vflip = (attr1 & 0x2000u) != 0;
            const int source_y = nearest_source_pixel(line, sh, draw_h);
            int ty = source_y >> 3;
            int py = source_y & 7;
            int src_ty = vflip ? (tiles_h - 1 - ty) : ty;
            int src_py = vflip ? (7 - py) : py;
            for (int dx = 0; dx < draw_w; ++dx) {
                const int source_x = nearest_source_pixel(dx, sw, draw_w);
                const int tx = source_x >> 3;
                int src_tx = hflip ? (tiles_w - 1 - tx) : tx;
                const int px = source_x & 7;
                int src_px = hflip ? (7 - px) : px;
                emit_obj(src_tx * 8 + src_px, src_ty * 8 + src_py, sx + dx);
            }
        }
    }

    // BLDY brightness coefficient (clamped to 16/16 = full).
    uint32_t bldy = static_cast<uint32_t>(io[0x54] | (io[0x55] << 8)) & 0x1Fu;
    if (bldy > 16u) bldy = 16u;
    for (uint32_t x = 0; x < kScreenWidth; ++x) {
        uint8_t* dst = row + x * 3;
        if (g_tag_row) g_tag_row[x] = motion_tag(top[x].key, top[x].layer);
        // Alpha blend top (1st target) with the layer below (2nd target). Per
        // GBATEK this occurs when effect==1 OR top is a semi-transparent OBJ
        // (mode 1 forces alpha regardless of BLDCNT). NOT gated on EVA!=0:
        // EVA=0/EVB=16 is a valid blend (1st target fully fades into the 2nd) —
        // the Oak-intro character fade endpoint that previously snapped back to
        // opaque, leaving body/feet (BG2 + semi-transparent OBJ) out of sync.
        if ((effect == 1 || top[x].layer == 4) &&
            top[x].target1 && second[x].valid && second[x].target2 &&
            !(top[x].layer == 4 && second[x].layer == 4)) {
            const uint16_t blended = blend_alpha_gba555(
                top[x].color, second[x].color,
                bldalpha & 0x1Fu, (bldalpha >> 8) & 0x1Fu);
            to_rgb888(blended, dst);
        } else if ((effect == 2u || effect == 3u) && bldy != 0u && top[x].target1) {
            // Brightness, like alpha, operates on native GBA channels before
            // RGB888 expansion and observes palette bit 15 as green precision.
            const uint16_t adjusted = effect == 2u
                ? brighten_gba555(top[x].color, bldy)
                : darken_gba555(top[x].color, bldy);
            to_rgb888(adjusted, dst);
        } else {
#if defined(GBARECOMP_SCANLINE_HLE)
            to_rgb888(top[x].color, dst);
#else
            dst[0] = top[x].rgb[0];
            dst[1] = top[x].rgb[1];
            dst[2] = top[x].rgb[2];
#endif
        }
    }
    if (optb_cap && y < optb_cap->height) {
        OptionBCap::Line& line = optb_cap->lines[y];
        line.backdrop_color = backdrop_color;
        line.dispcnt = dispcnt;
        line.effect = static_cast<uint8_t>(effect);
        line.eva = static_cast<uint8_t>(bldalpha & 0x1Fu);
        line.evb = static_cast<uint8_t>((bldalpha >> 8) & 0x1Fu);
        line.bldy = static_cast<uint8_t>(bldy);
    }
    if (layered_rgba) {
        pop_world_objects(bucket_top, bucket_second, top, kScreenWidth);
        fix_sprite_plane_occlusion(bucket_top, bucket_second, top, kScreenWidth, y, kScreenHeight);
        push_sprites_through_holes(bucket_top, bucket_second, top, second, kScreenWidth);
        extract_sprite_layer(bucket_top, bucket_second, top, second, kScreenWidth, y, kScreenHeight);
        record_world_objects(top, kScreenWidth, y, kScreenHeight);
        write_layered_scanline_planes(layered_rgba, y, kScreenWidth, kScreenHeight,
                                      effect, bldalpha, bldy,
                                      bucket_top, bucket_second);
    }
    return;
}

// ── Wide (view-expanded) scanline compositor ────────────────────────────────
// Renders output scanline `y` into an `out_w`-pixel row, where output column x
// maps to hardware column hx = x - ox (ox = left margin). This path is ONLY
// entered when view-area expansion is active; the faithful build always takes
// render_scanline_internal above, so OFF-mode stays byte-identical by construc-
// tion. For central hx in [0,240) the per-pixel logic is the same as vanilla;
// margin columns extend the BG scroll / affine extrapolation / OBJ sampling into
// the surrounding map. WIN0/WIN1 register tests are NOT widened — a margin hx
// falls outside [X1,X2) so it naturally resolves to WINOUT (the conservative
// policy for columns the game never authored). OAM X keeps its 9-bit signed
// decode and is tested against the expanded viewport, so wrapped-negative sprites
// (left) and x>=240 sprites (right) both appear in the margins. Signed `y`
// addresses authored vertical rows, where raw OAM/affine/bitmap extrapolation
// is prohibited; only validated game providers may contribute content.
void render_scanline_wide(uint8_t* rgb, int y, uint16_t dispcnt,
                          const uint8_t* io, const uint8_t* vram,
                          const uint8_t* oam, const uint8_t* pal,
                          uint32_t out_w, uint32_t ox, uint32_t oy = 0,
                          GbaHighResolutionFrame* hd = nullptr,
                          uint8_t* layered_rgba = nullptr,
                          uint32_t out_h = GbaPpu::kScreenHeight,
                          OptionBCap* optb_cap = nullptr) {
    constexpr uint32_t kVanW = GbaPpu::kScreenWidth;   // 240
    constexpr uint32_t kVanH = GbaPpu::kScreenHeight;  // 160
    if (g_ws_native_view_left >= 0)
        ox = std::min(static_cast<uint32_t>(g_ws_native_view_left), out_w - kVanW);
    const bool native_row = y >= 0 && y < static_cast<int>(kVanH);
    const uint32_t y_out = static_cast<uint32_t>(y + static_cast<int>(oy));
    OptionBCap::Pixel* optb_row = nullptr;
    if (optb_cap && y_out < optb_cap->height && out_w <= optb_cap->width) {
        optb_row = optb_cap->pixels.data() + static_cast<size_t>(y_out) * optb_cap->width;
        std::fill_n(optb_row, out_w, OptionBCap::Pixel{});
    }
    uint8_t* row = rgb + y_out * out_w * 3;
    if (!native_row && (!g_ws_authored_margin_layers || !g_ws_tilemap_provider || g_ws_pillarbox)) {
        std::memset(row, 0, out_w * 3);
        fill_tags(out_w);
        clear_layered_scanline_row(layered_rgba, y_out, out_w, out_h, 0u);
        clear_optb_cap_row(optb_cap, y_out, out_w, dispcnt, 0u);
        return;
    }
    if (dispcnt & 0x0080u) {
        std::memset(row, 0xFF, out_w * 3);
        fill_tags(out_w);
        clear_layered_scanline_row(layered_rgba, y_out, out_w, out_h, 0xFFu);
        clear_optb_cap_row(optb_cap, y_out, out_w, dispcnt, 0x7FFFu);
        return;
    }

    using PixelCandidate = LayeredCandidate;

    const bool win0_en   = (dispcnt & 0x2000u) != 0;
    const bool win1_en   = (dispcnt & 0x4000u) != 0;
    const bool objwin_en = (dispcnt & 0x8000u) != 0;
    const bool any_window = win0_en || win1_en || objwin_en;
    uint16_t winin  = static_cast<uint16_t>(io[0x48] | (io[0x49] << 8));
    uint16_t winout = static_cast<uint16_t>(io[0x4A] | (io[0x4B] << 8));
    auto win_v_row = [&](uint32_t vreg, int sample_y) -> bool {
        uint32_t v  = static_cast<uint32_t>(io[vreg] | (io[vreg + 1] << 8));
        uint32_t y1 = (v >> 8) & 0xFFu, y2 = v & 0xFFu;
        if (y2 > kVanH || y1 > y2) y2 = kVanH;
        return sample_y >= static_cast<int>(y1) && sample_y < static_cast<int>(y2);
    };
    auto win_h_in = [&](uint32_t hreg, int hx) -> bool {
        uint32_t h  = static_cast<uint32_t>(io[hreg] | (io[hreg + 1] << 8));
        int x1 = static_cast<int>((h >> 8) & 0xFFu), x2 = static_cast<int>(h & 0xFFu);
        if (x2 > static_cast<int>(kVanW) || x1 > x2) x2 = static_cast<int>(kVanW);
        return hx >= x1 && hx < x2;
    };
    const bool win0_row = win0_en && win_v_row(0x44, y);
    const bool win1_row = win1_en && win_v_row(0x46, y);
    // OBJ-window stencil is built in vanilla 240-space; margin columns (hx
    // outside [0,240)) get no OBJ-window (WINOUT), consistent with WIN0/1.
    bool obj_window_storage[GbaPpu::kScreenWidth] = {};
    bool* obj_window_mask = nullptr;
    if (objwin_en && native_row) {
        mark_obj_window_scanline(obj_window_storage, y, dispcnt, vram, oam,
                                 kVanW, kVanH);
        obj_window_mask = obj_window_storage;
    }
    auto window_control_at_hx = [&](int hx) -> uint16_t {
        if (!any_window) return 0x3Fu;
        if (win0_row && win_h_in(0x40, hx)) return winin & 0x3Fu;
        if (win1_row && win_h_in(0x42, hx)) return (winin >> 8) & 0x3Fu;
        if (obj_window_mask && hx >= 0 && hx < static_cast<int>(kVanW) &&
            obj_window_mask[hx])
            return static_cast<uint16_t>((winout >> 8) & 0x3Fu);
        return static_cast<uint16_t>(winout & 0x3Fu);
    };
    auto window_control_at_xy = [&](int hx, int sy) -> uint16_t {
        if (sy == y) return window_control_at_hx(hx);
        if (!any_window) return 0x3Fu;
        if (win0_en && win_v_row(0x44, sy) && win_h_in(0x40, hx)) return winin & 0x3Fu;
        if (win1_en && win_v_row(0x46, sy) && win_h_in(0x42, hx)) return (winin >> 8) & 0x3Fu;
        // Do not invent an OBJ-window stencil for a different scanline.
        return objwin_en ? 0u : static_cast<uint16_t>(winout & 0x3Fu);
    };
    // A guest-authored window has no defined continuation beyond the native
    // 240-pixel scanline. Extend it into the margins only when every authentic
    // pixel selects the same window control. Sampling just an edge and the
    // center misses narrow apertures (including OBJ-window stencils) that lie
    // between those probes and can leak scenery around transitions.
    uint16_t uniform_window_control = window_control_at_hx(0);
    bool window_control_is_uniform = true;
    if (g_ws_tilemap_provider && any_window) {
        for (int hx = 1; hx < static_cast<int>(kVanW); ++hx) {
            if (window_control_at_hx(hx) != uniform_window_control) {
                window_control_is_uniform = false;
                break;
            }
        }
    }
    const bool black_nonuniform_window_margins =
        g_ws_tilemap_provider && any_window && !window_control_is_uniform &&
        !g_ws_authored_margin_layers;
    auto window_control = [&](uint32_t x) -> uint16_t {
        int hx = static_cast<int>(x) - static_cast<int>(ox);
        if (g_ws_tilemap_provider &&
            (hx < 0 || hx >= static_cast<int>(kVanW))) {
            // Minish's room provider treats expanded columns as lying outside
            // every native WIN0/WIN1 rectangle, so WINOUT controls the world
            // there. Other providers retain the established fail-closed rule.
            if (g_ws_authored_margin_layers)
                return window_control_at_hx(hx);
            return window_control_is_uniform ? uniform_window_control : 0u;
        }
        return window_control_at_hx(hx);
    };
    // Every later compositor stage asks the same window questions for the
    // same output column. Resolve them once per scanline instead of repeating
    // WIN0/WIN1/OBJ-window branches for the backdrop, every BG, and every OBJ
    // pixel. The fixed-size stack array covers the renderer's clamped width.
    uint8_t window_controls[GbaPpu::kMaxRenderWidth];
    for (uint32_t x = 0; x < out_w; ++x) {
        window_controls[x] = static_cast<uint8_t>(window_control(x) & 0x3Fu);
        if (hd) hd->window(x, window_controls[x]);
    }
    auto layer_enabled = [&](uint32_t x, uint32_t layer_bit) -> bool {
        return (window_controls[x] & (1u << layer_bit)) != 0;
    };
    auto blend_enabled = [&](uint32_t x) -> bool {
        return (window_controls[x] & (1u << 5)) != 0;
    };

    uint16_t bldcnt = static_cast<uint16_t>(io[0x50] | (io[0x51] << 8));
    uint16_t bldalpha = static_cast<uint16_t>(io[0x52] | (io[0x53] << 8));
    uint32_t first_targets = bldcnt & 0x3Fu;
    uint32_t second_targets = (bldcnt >> 8) & 0x3Fu;
    uint32_t effect = (bldcnt >> 6) & 0x3u;

    PixelCandidate top[GbaPpu::kMaxRenderWidth];
    PixelCandidate second[GbaPpu::kMaxRenderWidth];
    thread_local std::vector<LayeredCandidate> bucket_top;
    thread_local std::vector<LayeredCandidate> bucket_second;
    if (layered_rgba) {
        const size_t bucket_count =
            static_cast<size_t>(GbaPpu::kLayeredPlanes) * out_w;
        bucket_top.assign(bucket_count, LayeredCandidate{});
        bucket_second.assign(bucket_count, LayeredCandidate{});
    }
    const uint16_t backdrop_color = load_u16_le(&pal[0]);
#if !defined(GBARECOMP_SCANLINE_HLE)
    uint8_t backdrop_rgb[3];
    to_rgb888(backdrop_color, backdrop_rgb);
#endif
    for (uint32_t x = 0; x < out_w; ++x) {
#if !defined(GBARECOMP_SCANLINE_HLE)
        top[x].rgb[0] = backdrop_rgb[0];
        top[x].rgb[1] = backdrop_rgb[1];
        top[x].rgb[2] = backdrop_rgb[2];
#endif
        top[x].color = backdrop_color;
        top[x].key = 0x70000000;
        top[x].layer = 5;
        top[x].target1 = blend_enabled(x) && ((first_targets & (1u << 5)) != 0);
        top[x].target2 = (second_targets & (1u << 5)) != 0;
        top[x].valid = true;
        if (layered_rgba) bucket_top[x] = top[x];
    }
    auto submit = [&](uint32_t x, uint16_t color, int key, uint8_t layer,
                      bool target1, bool target2) {
        if (hd) hd->submit(x, color, key, layer, target1, target2);
        if (optb_row) record_optb_submit(optb_row, x, color, key, layer, target1);
        PixelCandidate cand;
        cand.color = color;
#if !defined(GBARECOMP_SCANLINE_HLE)
        to_rgb888(color, cand.rgb);
#endif
        cand.key = key;
        cand.layer = layer;
        cand.target1 = target1;
        cand.target2 = target2;
        cand.valid = true;
        if (key < top[x].key) { second[x] = top[x]; top[x] = cand; }
        else if (key < second[x].key) { second[x] = cand; }
        if (layered_rgba) {
            assign_layered_bucket(bucket_top, bucket_second, out_w, x, cand);
        }
    };

    uint32_t bg_mode = dispcnt & 0x07u;
    auto render_regular_bg = [&](uint32_t layer, uint32_t cnt_off,
                                 uint32_t scroll_off) {
        if ((dispcnt & (0x0100u << layer)) == 0) return;
        uint16_t bgcnt = static_cast<uint16_t>(io[cnt_off] | (io[cnt_off + 1] << 8));
        uint32_t char_base = ((bgcnt >> 2) & 0x3u) * 0x4000u;
        uint32_t screen_base = ((bgcnt >> 8) & 0x1Fu) * 0x800u;
        bool color256 = (bgcnt & 0x0080u) != 0;
        uint32_t size_code = (bgcnt >> 14) & 0x3u;
        uint32_t bg_priority = bgcnt & 0x3u;
        uint32_t hofs = static_cast<uint16_t>(
            io[scroll_off] | (io[scroll_off + 1] << 8)) & 0x01FFu;
        uint32_t vofs = static_cast<uint16_t>(
            io[scroll_off + 2] | (io[scroll_off + 3] << 8)) & 0x01FFu;
        uint32_t width_tiles = (size_code & 1u) ? 64u : 32u;
        uint32_t height_tiles = (size_code & 2u) ? 64u : 32u;
        uint32_t width_px = width_tiles * 8u;
        uint32_t height_px = height_tiles * 8u;
        uint32_t block_cols = width_tiles / 32u;
        for (uint32_t x = 0; x < out_w; ++x) {
            int hx = static_cast<int>(x) - static_cast<int>(ox);
            const bool authored_margin = g_ws_authored_margin_layers &&
                (!native_row || hx < 0 || hx >= static_cast<int>(kVanW));
            // Minish supplies room-map entries for margins independently of
            // the native 240px WIN0/WIN1 HUD/dialog masks. Ignore only their
            // regular-BG layer gate there; unsupported UI BGs still fail in
            // the provider below, so they cannot repeat.
            int sample_hx = hx;
            int sample_y = y;
            bool remapped = false;
            bool xy_remapped = false;
            if (g_ws_bg_xy_provider && (g_ws_bg_xy_provider_layers & (1u << layer))) {
                int provided_x = hx, provided_y = y;
                const int action = g_ws_bg_xy_provider(layer, hx, y, &provided_x, &provided_y);
                if (action < 0) continue;
                if (action > 0) {
                    sample_hx = provided_x; sample_y = provided_y;
                    remapped = xy_remapped = true;
                }
            }
            if (!layer_enabled(x, layer) && !authored_margin && !xy_remapped) continue;
            if (g_ws_bg_x_provider &&
                (g_ws_bg_x_provider_layers & (1u << layer)) && !remapped) {
                int provided_hx = hx;
                const int action = g_ws_bg_x_provider(
                    static_cast<int>(layer), static_cast<int>(x),
                    static_cast<int>(y), &provided_hx);
                if (action < 0) continue;
                if (action > 0) { sample_hx = provided_hx; remapped = true; }
            }
            // A remapped margin sample continues an authentic viewport
            // column, so gate it by that column's own window control instead
            // of the margin's WINOUT fallback. Otherwise reflected
            // continuations vanish exactly when a guest window covers the
            // scenery they mirror (e.g. battle arenas whose WINOUT drops the
            // world layer behind HUD windows). Non-remapped margins keep the
            // established WINOUT/fail-closed behavior.
            if (remapped && (xy_remapped || hx < 0 || hx >= static_cast<int>(kVanW)) &&
                sample_hx >= 0 && sample_hx < static_cast<int>(kVanW) &&
                (window_control_at_xy(sample_hx, sample_y) & (1u << layer)) == 0)
                continue;
            uint32_t tex_x = static_cast<uint32_t>(
                                 sample_hx + static_cast<int>(hofs)) &
                             (width_px - 1u);
            uint32_t tex_y = (sample_y + vofs) & (height_px - 1u);
            uint32_t tile_x = tex_x >> 3;
            uint32_t tile_y = tex_y >> 3;
            uint32_t block = (tile_x >> 5) + (tile_y >> 5) * block_cols;
            uint32_t map_off = screen_base + block * 0x800u +
                ((tile_y & 31u) * 32u + (tile_x & 31u)) * 2u;
            if (map_off + 1 >= 96u * 1024u) continue;
            uint16_t entry = load_u16_le(&vram[map_off]);
            // Widescreen margin (hardware column outside 0..239): the 256px ring
            // entry here is the wrapped/aliased seam. If the sidecar can supply
            // the true off-screen world tile, use it; if it's armed but has no
            // data, leave the pixel transparent (no seam) rather than draw the
            // wrap. With no sidecar, keep vanilla wide behavior.
            if (sample_hx < 0 || sample_hx >= 240 || sample_y < 0 || sample_y >= 160) {
                if (g_ws_tilemap_provider) {
                    uint16_t ext;
                    const int action = g_ws_tilemap_provider(
                        static_cast<int>(layer), sample_hx,
                        sample_y, &ext);
                    if (action == kWsTilemapReplace) {
                        entry = ext;
                    } else if (action != kWsTilemapKeepWrapped) {
                        continue;
                    }
                }
            }
            uint32_t tile_num = entry & 0x03FFu;
            bool hflip = (entry & 0x0400u) != 0;
            bool vflip = (entry & 0x0800u) != 0;
            uint32_t palette_bank = (entry >> 12) & 0x0Fu;
            uint32_t px = tex_x & 7u;
            uint32_t py = tex_y & 7u;
            if (hflip) px = 7u - px;
            if (vflip) py = 7u - py;
            uint8_t pal_idx = 0;
            if (color256) {
                uint32_t tile_addr = char_base + tile_num * 64u + py * 8u + px;
                if (tile_addr >= 96u * 1024u) continue;
                pal_idx = vram[tile_addr];
            } else {
                uint32_t tile_addr = char_base + tile_num * 32u + py * 4u + (px >> 1);
                if (tile_addr >= 96u * 1024u) continue;
                uint8_t packed = vram[tile_addr];
                pal_idx = (px & 1u) ? (packed >> 4) : (packed & 0x0Fu);
                pal_idx = static_cast<uint8_t>(
                    pal_idx | static_cast<uint8_t>(palette_bank << 4));
            }
            if ((pal_idx & (color256 ? 0xFFu : 0x0Fu)) == 0) continue;
            const uint16_t color = load_u16_le(&pal[pal_idx * 2]);
            const bool sample_blend = xy_remapped
                ? (window_control_at_xy(sample_hx, sample_y) & 0x20) != 0
                : blend_enabled(x);
            submit(x, color,
                   static_cast<int>(bg_priority * 256u + 128u + layer),
                   static_cast<uint8_t>(layer),
                   sample_blend && ((first_targets & (1u << layer)) != 0),
                   (second_targets & (1u << layer)) != 0);
        }
    };
    auto render_affine_bg = [&](uint32_t layer, uint32_t cnt_off,
                                uint32_t param_off) {
        if (!native_row) return;
        if ((dispcnt & (0x0100u << layer)) == 0) return;
        uint16_t bgcnt = static_cast<uint16_t>(io[cnt_off] | (io[cnt_off + 1] << 8));
        uint32_t char_base   = ((bgcnt >> 2) & 0x3u) * 0x4000u;
        uint32_t screen_base = ((bgcnt >> 8) & 0x1Fu) * 0x800u;
        bool wrap            = (bgcnt & 0x2000u) != 0;
        uint32_t size_code   = (bgcnt >> 14) & 0x3u;
        int bg_pixels = 128 << size_code;
        int bg_tiles  = bg_pixels / 8;
        int bg_priority = static_cast<int>(bgcnt & 0x3u);
        int32_t pa = read_s16(io, param_off + 0x00);
        int32_t pb = read_s16(io, param_off + 0x02);
        int32_t pc = read_s16(io, param_off + 0x04);
        int32_t pd = read_s16(io, param_off + 0x06);
        int32_t refx = read_s28_ref(io, param_off + 0x08);
        int32_t refy = read_s28_ref(io, param_off + 0x0C);
        // Extrapolate from the scanline reference using the signed logical x:
        // output column 0 is hardware x = -ox, so pre-advance by (-ox)*PA/PC.
        int32_t xt = refx + static_cast<int32_t>(y) * pb +
                     static_cast<int32_t>(-static_cast<int>(ox)) * pa;
        int32_t yt = refy + static_cast<int32_t>(y) * pd +
                     static_cast<int32_t>(-static_cast<int>(ox)) * pc;
        const bool filter =
            g_ws_affine_filter_enabled &&
            g_ws_affine_filter_provider &&
            g_ws_affine_filter_provider(
                static_cast<int>(layer), static_cast<int>(y));
        auto sample_color = [&](int32_t tex_x, int32_t tex_y,
                                uint16_t* out_color) -> bool {
            if (wrap) {
                tex_x &= (bg_pixels - 1);
                tex_y &= (bg_pixels - 1);
            } else if (tex_x < 0 || tex_x >= bg_pixels ||
                       tex_y < 0 || tex_y >= bg_pixels) {
                return false;
            }
            const uint32_t map_off =
                screen_base + (tex_y >> 3) * bg_tiles + (tex_x >> 3);
            if (map_off >= 96u * 1024u) return false;
            const uint8_t tile_index = vram[map_off];
            const uint32_t tile_addr =
                char_base + tile_index * 64u +
                (tex_y & 7) * 8 + (tex_x & 7);
            if (tile_addr >= 96u * 1024u) return false;
            const uint8_t pal_idx = vram[tile_addr];
            if (pal_idx == 0) return false;
            *out_color = load_u16_le(&pal[pal_idx * 2]);
            return true;
        };
        for (uint32_t x = 0; x < out_w; ++x) {
            const int32_t sample_x = xt;
            const int32_t sample_y = yt;
            xt += pa;
            yt += pc;
            if (!layer_enabled(x, layer)) continue;
            const int32_t tex_x = sample_x >> 8;
            const int32_t tex_y = sample_y >> 8;
            uint16_t color = 0;
            if (!filter) {
                if (!sample_color(tex_x, tex_y, &color)) continue;
            } else {
                uint16_t c00 = 0, c10 = 0, c01 = 0, c11 = 0;
                if (!sample_color(tex_x, tex_y, &c00)) continue;
                if (sample_color(tex_x + 1, tex_y, &c10) &&
                    sample_color(tex_x, tex_y + 1, &c01) &&
                    sample_color(tex_x + 1, tex_y + 1, &c11)) {
                    const uint32_t fx =
                        static_cast<uint32_t>(sample_x) & 0xFFu;
                    const uint32_t fy =
                        static_cast<uint32_t>(sample_y) & 0xFFu;
                    const uint32_t ix = 256u - fx;
                    const uint32_t iy = 256u - fy;
                    const uint32_t w00 = ix * iy;
                    const uint32_t w10 = fx * iy;
                    const uint32_t w01 = ix * fy;
                    const uint32_t w11 = fx * fy;
                    auto blend_channel = [&](unsigned shift) {
                        const uint32_t sum =
                            ((c00 >> shift) & 31u) * w00 +
                            ((c10 >> shift) & 31u) * w10 +
                            ((c01 >> shift) & 31u) * w01 +
                            ((c11 >> shift) & 31u) * w11;
                        return (sum + 0x8000u) >> 16;
                    };
                    color = static_cast<uint16_t>(
                        blend_channel(0) |
                        (blend_channel(5) << 5) |
                        (blend_channel(10) << 10));
                } else {
                    color = c00;
                }
            }
            submit(x, color,
                   static_cast<int>(bg_priority * 256 + 128 + layer),
                   static_cast<uint8_t>(layer),
                   blend_enabled(x) && ((first_targets & (1u << layer)) != 0),
                   (second_targets & (1u << layer)) != 0);
        }
    };
    // Bitmap BG modes 3/4/5 — see the commentary on render_bitmap_bg in
    // render_scanline_internal above for the format/transparency rules.
    // A bitmap has a fixed width (240, or 160 in mode 5), so there is no
    // extra image data to reveal in the expanded margins: the affine
    // pre-advance by -ox places the bitmap exactly where it belongs and the
    // margins fall outside its bounds, staying backdrop.
    auto render_bitmap_bg = [&]() {
        if (!native_row) return;
        if ((dispcnt & 0x0400u) == 0) return;   // DISPCNT bit10 = BG2 enable
        constexpr uint32_t layer = 2;
        uint16_t bgcnt = static_cast<uint16_t>(io[0x0C] | (io[0x0D] << 8));
        int bg_priority = static_cast<int>(bgcnt & 0x3u);

        const bool     direct = (bg_mode != 4);
        const int      bmp_w  = (bg_mode == 5) ? 160 : 240;
        const int      bmp_h  = (bg_mode == 5) ? 128 : 160;
        const uint32_t bpp    = direct ? 2u : 1u;
        const uint32_t frame_base =
            (bg_mode != 3 && (dispcnt & 0x0010u) != 0) ? 0xA000u : 0x0000u;

        int32_t pa   = read_s16(io, 0x20);
        int32_t pb   = read_s16(io, 0x22);
        int32_t pc   = read_s16(io, 0x24);
        int32_t pd   = read_s16(io, 0x26);
        int32_t refx = read_s28_ref(io, 0x28);
        int32_t refy = read_s28_ref(io, 0x2C);
        int32_t xt = refx + static_cast<int32_t>(y) * pb +
                     static_cast<int32_t>(-static_cast<int>(ox)) * pa;
        int32_t yt = refy + static_cast<int32_t>(y) * pd +
                     static_cast<int32_t>(-static_cast<int>(ox)) * pc;
        for (uint32_t x = 0; x < out_w; ++x) {
            int32_t tex_x = xt >> 8;
            int32_t tex_y = yt >> 8;
            xt += pa;
            yt += pc;
            if (!layer_enabled(x, layer)) continue;
            if (tex_x < 0 || tex_x >= bmp_w || tex_y < 0 || tex_y >= bmp_h)
                continue;
            const uint32_t off = frame_base +
                (static_cast<uint32_t>(tex_y) * static_cast<uint32_t>(bmp_w) +
                 static_cast<uint32_t>(tex_x)) * bpp;
            if (off + bpp > 96u * 1024u) continue;
            uint16_t color;
            if (direct) {
                color = load_u16_le(&vram[off]);
            } else {
                uint8_t pal_idx = vram[off];
                if (pal_idx == 0) continue;
                color = load_u16_le(&pal[pal_idx * 2]);
            }
            submit(x, color,
                   static_cast<int>(bg_priority * 256 + 128 + layer),
                   static_cast<uint8_t>(layer),
                   blend_enabled(x) && ((first_targets & (1u << layer)) != 0),
                   (second_targets & (1u << layer)) != 0);
        }
    };
    if (bg_mode == 0) {
        render_regular_bg(3, 0x0E, 0x1C);
        render_regular_bg(2, 0x0C, 0x18);
        render_regular_bg(1, 0x0A, 0x14);
        render_regular_bg(0, 0x08, 0x10);
    } else if (bg_mode == 1) {
        render_affine_bg(2, 0x0C, 0x20);
        render_regular_bg(1, 0x0A, 0x14);
        render_regular_bg(0, 0x08, 0x10);
    } else if (bg_mode == 2) {
        render_affine_bg(3, 0x0E, 0x30);
        render_affine_bg(2, 0x0C, 0x20);
    } else if (bg_mode <= 5) {
        // Modes 6/7 are prohibited (GBATEK); leave them backdrop-only.
        render_bitmap_bg();
    }

    // Keep the bounded native-screen overlay in the centered hardware
    // viewport when adaptive view expansion is active. It deliberately does
    // not extend into either margin: its coordinates are still the canonical
    // 240x160 GBA screen domain. As in the faithful path, it is synthetic BG2
    // for window/effect handling, is below guest OBJ, and disappears whenever
    // a full foreign background has been published.
    if (foreign_presentation_internal::background() == nullptr) {
        if (const auto* overlay = foreign_presentation_internal::screen_overlay();
            valid_foreign_screen_overlay(overlay)) {
            const int left = static_cast<int>(ox) + overlay->x;
            const int top_y = overlay->y;
            for (unsigned oy = 0; oy < overlay->height; ++oy) {
                const int screen_y = top_y + static_cast<int>(oy);
                if (screen_y != static_cast<int>(y)) continue;
                for (unsigned ix = 0; ix < overlay->width; ++ix) {
                    const int screen_x = left + static_cast<int>(ix);
                    if (screen_x < 0 || screen_x >= static_cast<int>(out_w))
                        continue;
                    const unsigned x = static_cast<unsigned>(screen_x);
                    if (!layer_enabled(x, 2)) continue;
                    const std::size_t source =
                        static_cast<std::size_t>(oy) * overlay->stride + ix;
                    const std::uint8_t alpha = overlay->alpha_q4[source];
                    if (alpha == 0 || alpha > 16) continue;
                    const PixelCandidate old_top = top[x];
                    PixelCandidate cand;
                    cand.color = blend_alpha_gba555(overlay->pixels[source],
                                                     old_top.color, alpha,
                                                     16u - alpha);
#if !defined(GBARECOMP_SCANLINE_HLE)
                    to_rgb888(cand.color, cand.rgb);
#endif
                    cand.key = 0x10000;
                    cand.layer = 2;
                    cand.target1 = blend_enabled(x) &&
                        ((first_targets & (1u << 2)) != 0);
                    cand.target2 = (second_targets & (1u << 2)) != 0;
                    cand.valid = true;
                    top[x] = cand;
                    second[x] = old_top;
                }
            }
        }
    }

    if ((native_row || g_ws_obj_attr_y_provider) && (dispcnt & 0x1000u)) {
        constexpr uint32_t obj_tile_base = 0x10000u;
        bool obj_1d_mapping = (dispcnt & 0x0040u) != 0;
        const uint8_t* obj_pal = pal + 0x200;
        for (int idx = 127; idx >= 0; --idx) {
            const uint8_t* entry = oam + idx * 8;
            uint16_t attr0 = load_u16_le(&entry[0]);
            uint16_t attr1 = load_u16_le(&entry[2]);
            uint16_t attr2 = load_u16_le(&entry[4]);
            bool rot_scale = (attr0 & 0x0100u) != 0;
            bool disable_or_double = (attr0 & 0x0200u) != 0;
            if (!rot_scale && disable_or_double) continue;
            uint32_t obj_mode = (attr0 >> 10) & 0x3u;
            if (obj_mode == 2 || obj_mode == 3) continue;
            uint32_t shape = (attr0 >> 14) & 0x3u;
            if (shape >= 3) continue;
            uint32_t size  = (attr1 >> 14) & 0x3u;
            int sw = kSpriteWH[shape][size][0];
            int sh = kSpriteWH[shape][size][1];
            int sy = static_cast<int>(attr0 & 0xFFu);
            // OAM X is normally signed 9-bit. An opted-in game may reinterpret
            // values that its widened guest writer emitted beyond hardware X.
            const int raw_sx = static_cast<int>(attr1 & 0x1FFu);
            int sx = raw_sx;
            if (sy >= 160) sy -= 256;
            int authored_sy = sy;
            const bool authored_y = g_ws_obj_attr_y_provider &&
                g_ws_obj_attr_y_provider(idx, attr0, attr1, attr2, &authored_sy);
            if (!native_row && !authored_y) continue;
            if (authored_y) sy = authored_sy;
            auto resolve_sx = [&] {
                int provided_sx = sx;
                if (g_ws_obj_attr_x_provider &&
                    g_ws_obj_attr_x_provider(idx, attr0, attr1, attr2,
                                             &provided_sx)) {
                    sx = provided_sx;
                } else if (g_ws_obj_x_provider &&
                    g_ws_obj_x_provider(raw_sx, &provided_sx)) {
                    sx = provided_sx;
                } else if (sx & 0x100) {
                    sx -= 0x200;
                }
            };
            const bool obj_focus_active =
                valid_foreign_obj_focus(foreign_presentation_internal::obj_focus());
            bool color256 = (attr0 & 0x2000u) != 0;
            uint32_t tile_num = attr2 & 0x3FFu;
            if (bg_mode >= 3 && tile_num < 512u) continue;
            uint32_t palette_bank = (attr2 >> 12) & 0xFu;
            int tiles_w = sw / 8;
            int tiles_h = sh / 8;
            int priority = static_cast<int>((attr2 >> 10) & 0x3u);
            // Composite key (lower = front). Per-priority stride 256 with OBJ in
            // [p*256, p*256+127] (tie-break by OAM idx) and BG in [p*256+128,
            // p*256+131] (tie-break by layer) yields the exact GBA order
            // OBJ0<BG0<OBJ1<BG1<OBJ2<BG2<OBJ3<BG3: an OBJ is in front of a same-
            // priority BG, but a BG of priority p sits in front of any OBJ of
            // priority p+1 (this is what lets the player walk BEHIND roof/tree
            // tops). The stride MUST exceed 128 so OBJ idx (0..127) can't bleed
            // into the next priority's BG band.
            int key = priority * 256 + idx;
            bool obj_target2 = (second_targets & (1u << 4)) != 0;
            auto emit_obj = [&](int tex_x, int tex_y, int screen_x) {
                if (screen_x < 0 || screen_x >= static_cast<int>(out_w)) return;
                // Opt-in policy: hardware-faithful OBJ placement only within
                // the native viewport; margin columns never show sprites the
                // guest believed were off-screen.
                if (g_ws_obj_native_clip &&
                    (screen_x < static_cast<int>(ox) ||
                     screen_x >= static_cast<int>(ox) + static_cast<int>(kVanW)))
                    return;
                const bool authored_margin = authored_y && g_ws_authored_margin_layers &&
                    (!native_row || screen_x < static_cast<int>(ox) ||
                     screen_x >= static_cast<int>(ox) + static_cast<int>(kVanW));
                if (!layer_enabled(static_cast<uint32_t>(screen_x), 4) && !authored_margin) return;
                int tile_x_in_sprite = tex_x >> 3;
                int tile_y_in_sprite = tex_y >> 3;
                int px_in_tile = tex_x & 7;
                int py_in_tile = tex_y & 7;
                uint32_t this_tile;
                if (obj_1d_mapping) {
                    this_tile = tile_num + (tile_y_in_sprite * tiles_w + tile_x_in_sprite) *
                                            (color256 ? 2u : 1u);
                } else {
                    this_tile = tile_num + (tile_y_in_sprite * 32u) +
                                tile_x_in_sprite * (color256 ? 2u : 1u);
                }
                uint32_t tile_off = obj_tile_base + this_tile * 32u;
                uint8_t pal_index;
                if (color256) {
                    uint32_t off = tile_off + py_in_tile * 8 + px_in_tile;
                    if (off + 1 > 96u * 1024u) return;
                    pal_index = vram[off];
                    if (pal_index == 0) return;
                } else {
                    uint32_t off = tile_off + py_in_tile * 4 + (px_in_tile / 2);
                    if (off + 1 > 96u * 1024u) return;
                    uint8_t b = vram[off];
                    pal_index = (px_in_tile & 1) ? (b >> 4) : (b & 0x0F);
                    if (pal_index == 0) return;
                    pal_index = static_cast<uint8_t>(pal_index | (palette_bank << 4));
                }
                const uint16_t color = load_u16_le(&obj_pal[pal_index * 2]);
                uint32_t ux = static_cast<uint32_t>(screen_x);
                bool t1 = (blend_enabled(ux) || authored_margin) && obj_mode == 1;
                submit(ux, color, key, 4, t1, obj_target2);
            };
            if (rot_scale) {
                int bw = disable_or_double ? sw * 2 : sw;
                int bh = disable_or_double ? sh * 2 : sh;
                // Preserve the existing callback timing exactly while focus
                // is inactive. When active, resolve the game-owned widened X
                // first, then apply pure data-only focus before clipping.
                if (obj_focus_active) {
                    resolve_sx();
                    if (apply_foreign_obj_focus(&sx, &sy, bw, bh, tile_num,
                                                false, static_cast<uint32_t>(idx)).action ==
                        ForeignObjFocusAction::kSuppress) continue;
                }
                int j = static_cast<int>(y) - sy;
                if (j < 0 || j >= bh) continue;
                if (!obj_focus_active) resolve_sx();
                int affine_group = (attr1 >> 9) & 0x1Fu;
                const uint8_t* ag = oam + affine_group * 0x20u;
                int32_t pa = read_s16(ag, 0x06);
                int32_t pb = read_s16(ag, 0x0E);
                int32_t pc = read_s16(ag, 0x16);
                int32_t pd = read_s16(ag, 0x1E);
                int half_bw = bw >> 1;
                int half_bh = bh >> 1;
                int half_sw = sw >> 1;
                int half_sh = sh >> 1;
                int dy = j - half_bh;
                for (int i = 0; i < bw; ++i) {
                    int dx = i - half_bw;
                    int tex_x = ((pa * dx + pb * dy) >> 8) + half_sw;
                    int tex_y = ((pc * dx + pd * dy) >> 8) + half_sh;
                    if (tex_x < 0 || tex_x >= sw) continue;
                    if (tex_y < 0 || tex_y >= sh) continue;
                    emit_obj(tex_x, tex_y, sx + i + static_cast<int>(ox));
                }
                continue;
            }
            ForeignObjFocusResult focus_result;
            if (obj_focus_active) {
                resolve_sx();
                focus_result =
                    apply_foreign_obj_focus(&sx, &sy, sw, sh, tile_num, true,
                                            static_cast<uint32_t>(idx));
                if (focus_result.action == ForeignObjFocusAction::kSuppress) continue;
            }
            const int draw_w = scaled_extent(sw, focus_result.scale_q8_8);
            const int draw_h = scaled_extent(sh, focus_result.scale_q8_8);
            int line = static_cast<int>(y) - sy;
            if (line < 0 || line >= draw_h) continue;
            if (!obj_focus_active) resolve_sx();
            bool hflip = (attr1 & 0x1000u) != 0;
            bool vflip = (attr1 & 0x2000u) != 0;
            const int source_y = nearest_source_pixel(line, sh, draw_h);
            int ty = source_y >> 3;
            int py = source_y & 7;
            int src_ty = vflip ? (tiles_h - 1 - ty) : ty;
            int src_py = vflip ? (7 - py) : py;
            for (int dx = 0; dx < draw_w; ++dx) {
                const int source_x = nearest_source_pixel(dx, sw, draw_w);
                const int tx = source_x >> 3;
                int src_tx = hflip ? (tiles_w - 1 - tx) : tx;
                const int px = source_x & 7;
                int src_px = hflip ? (7 - px) : px;
                emit_obj(src_tx * 8 + src_px, src_ty * 8 + src_py,
                         sx + dx + static_cast<int>(ox));
            }
        }
    }

    uint32_t bldy = static_cast<uint32_t>(io[0x54] | (io[0x55] << 8)) & 0x1Fu;
    if (bldy > 16u) bldy = 16u;
    if ((dispcnt & 0x1000u) && g_ws_obj_margin_provider && !g_ws_pillarbox) {
        int left = 0, width = 0;
        const auto* pixels = g_ws_obj_margin_provider(static_cast<int>(y), &left, &width);
        if (pixels && width > 0 && width <= static_cast<int>(GbaPpu::kMaxRenderWidth) &&
            left >= -static_cast<int>(GbaPpu::kMaxRenderWidth) &&
            left <= static_cast<int>(GbaPpu::kMaxRenderWidth)) {
            for (int i = 0; i < width; ++i) {
                const int hx = left + i, x = hx + static_cast<int>(ox);
                if ((native_row && hx >= 0 && hx < 240) || x < 0 || x >= static_cast<int>(out_w)) continue;
                const auto& pixel = pixels[i];
                // Authored world margins are independent of native HUD/window
                // rectangles, just like the regular-BG providers above. A
                // game's WINOUT often disables OBJ beyond the stock screen.
                if ((pixel.color & 0x8000) || pixel.priority > 3 ||
                    (!layer_enabled(x, 4) && !g_ws_authored_margin_layers)) continue;
                submit(x, pixel.color, pixel.priority * 256 + (pixel.order & 127), 4,
                       pixel.semi_transparent && blend_enabled(x),
                       (second_targets & (1u << 4)) != 0);
            }
        }
    }
    bool blackout_cols[GbaPpu::kMaxRenderWidth] = {};
    for (uint32_t x = 0; x < out_w; ++x) {
        uint16_t ui_color = 0;
        if (g_ws_ui_pixel_provider &&
            g_ws_ui_pixel_provider(static_cast<int>(x) - static_cast<int>(ox), y, &ui_color))
            submit(x, ui_color & 0x7FFFu, -1, 0, false, false);
        uint8_t* dst = row + x * 3;
        if (g_tag_row) g_tag_row[x] = motion_tag(top[x].key, top[x].layer);
        // OBJ-only presentations (notably the real GBA BIOS logo) author the
        // whole backdrop through palette entry 0. There is no regular-BG
        // alias to protect in the margins, so preserve that authentic color
        // instead of replacing it with policy black; OBJ remains centered.
        const bool palette_backdrop_only = (dispcnt & 0x0F00u) == 0u;
        // Pillarbox policy: black out the margin columns on non-overworld
        // screens so menus/battles are letterboxed, not garbled.
        const bool left_margin = x < ox;
        const bool right_margin = x >= ox + 240u;
        if ((black_nonuniform_window_margins &&
             (left_margin || right_margin)) ||
            (!palette_backdrop_only &&
             ((g_ws_pillarbox && (left_margin || right_margin)) ||
              (g_ws_pillarbox_left && left_margin) ||
              (g_ws_pillarbox_right && right_margin)))) {
            dst[0] = dst[1] = dst[2] = 0;
            blackout_cols[x] = true;
            if (hd) hd->black_column(x);
            continue;
        }
        // Alpha blend top (1st target) with the layer below (2nd target). Per
        // GBATEK this occurs when effect==1 OR top is a semi-transparent OBJ
        // (mode 1 forces alpha regardless of BLDCNT). NOT gated on EVA!=0:
        // EVA=0/EVB=16 is a valid blend (1st target fully fades into the 2nd) —
        // the Oak-intro character fade endpoint that previously snapped back to
        // opaque, leaving body/feet (BG2 + semi-transparent OBJ) out of sync.
        if ((effect == 1 || top[x].layer == 4) &&
            top[x].target1 && second[x].valid && second[x].target2 &&
            !(top[x].layer == 4 && second[x].layer == 4)) {
            const uint16_t blended = blend_alpha_gba555(
                top[x].color, second[x].color,
                bldalpha & 0x1Fu, (bldalpha >> 8) & 0x1Fu);
            to_rgb888(blended, dst);
        } else if ((effect == 2u || effect == 3u) && bldy != 0u && top[x].target1) {
            const uint16_t adjusted = effect == 2u
                ? brighten_gba555(top[x].color, bldy)
                : darken_gba555(top[x].color, bldy);
            to_rgb888(adjusted, dst);
        } else {
#if defined(GBARECOMP_SCANLINE_HLE)
            to_rgb888(top[x].color, dst);
#else
            dst[0] = top[x].rgb[0];
            dst[1] = top[x].rgb[1];
            dst[2] = top[x].rgb[2];
#endif
        }
    }
    if (optb_cap && y_out < optb_cap->height) {
        OptionBCap::Line& line = optb_cap->lines[y_out];
        line.backdrop_color = backdrop_color;
        line.dispcnt = dispcnt;
        line.effect = static_cast<uint8_t>(effect);
        line.eva = static_cast<uint8_t>(bldalpha & 0x1Fu);
        line.evb = static_cast<uint8_t>((bldalpha >> 8) & 0x1Fu);
        line.bldy = static_cast<uint8_t>(bldy);
        if (optb_row) {
            for (uint32_t x = 0; x < out_w; ++x) {
                if (blackout_cols[x]) optb_row[x].flags |= 4u;
            }
        }
    }
    if (layered_rgba) {
        pop_world_objects(bucket_top, bucket_second, top, out_w);
        fix_sprite_plane_occlusion(bucket_top, bucket_second, top, out_w, y_out, out_h);
        push_sprites_through_holes(bucket_top, bucket_second, top, second, out_w);
        extract_sprite_layer(bucket_top, bucket_second, top, second, out_w, y_out, out_h);
        record_world_objects(top, out_w, y_out, out_h);
        write_layered_scanline_planes(layered_rgba, y_out, out_w, out_h,
                                      effect, bldalpha, bldy,
                                      bucket_top, bucket_second, blackout_cols);
    }
    if (hd) hd->resolve();
}

inline void write_optb_gba555(uint16_t color555, uint8_t effect, bool target1,
                              uint32_t bldy, uint8_t alpha, uint8_t* dst) {
    uint16_t c = color555;
    if ((effect == 2u || effect == 3u) && bldy != 0u && target1) {
        c = (effect == 2u) ? brighten_gba555(c, bldy) : darken_gba555(c, bldy);
    }
    to_rgb888(c, dst);
    dst[3] = alpha;
}

void build_masked_5plane_frame(const OptionBCap& cap,
                               const uint8_t* composite_rgb,
                               std::vector<uint8_t>& masked_fb,
                               bool& out_scene_active,
                               uint32_t ox,
                               uint32_t oy) {
    const uint32_t W = cap.width;
    const uint32_t H = cap.height;
    const size_t N = static_cast<size_t>(W) * H;
    if (W == 0 || H == 0 || cap.pixels.size() < N || cap.lines.size() < H) {
        out_scene_active = false;
        return;
    }

    const size_t rgba_bytes = N * 4u;
    if (masked_fb.size() != rgba_bytes * 5u) masked_fb.resize(rgba_bytes * 5u);
    std::memset(masked_fb.data(), 0, rgba_bytes * 5u);
    uint8_t* p0 = masked_fb.data() + 0u * rgba_bytes;
    uint8_t* p1 = masked_fb.data() + 1u * rgba_bytes;
    uint8_t* p2 = masked_fb.data() + 2u * rgba_bytes;
    uint8_t* p3 = masked_fb.data() + 3u * rgba_bytes;
    uint8_t* p4 = masked_fb.data() + 4u * rgba_bytes;

    // Reusable scratch storage across frames (zero per-frame heap churn).
    thread_local std::vector<int8_t> solid;
    thread_local std::vector<uint8_t> authored;  // hand-authored semantic plane (side channel)
    thread_local std::vector<uint8_t> void_bg3;
    thread_local std::vector<uint8_t> bg1_trans, obj_is_hud;
    thread_local std::vector<int8_t> bg_pl[4];
    thread_local std::vector<uint8_t> vis, floor_set;
    thread_local std::vector<int> bfs_q, comp_pts;
    thread_local std::vector<uint16_t> blk_col;
    thread_local std::vector<uint8_t> blk_valid;
    thread_local std::vector<int> blk_q;
    thread_local std::vector<int> left_clean, right_clean;

    solid.assign(N, -1);
    size_t valid_solid = 0;
    size_t bg3_cnt = 0;
    size_t non_blank_lines = 0;
    size_t bg3_enabled_lines = 0;
    for (uint32_t y = 0; y < H; ++y) {
        const uint16_t d = cap.lines[y].dispcnt;
        if ((d & 0x0080u) == 0u) {
            ++non_blank_lines;
            if ((d & 7u) == 0u && (d & 0x0800u) != 0u) ++bg3_enabled_lines;
        }
    }
    const bool have_solid_provider = (g_layered_solid_row_provider != nullptr) || (g_layered_solid_provider != nullptr);
    if (have_solid_provider && non_blank_lines >= H / 2 && bg3_enabled_lines >= H / 2) {
        for (uint32_t y = 0; y < H; ++y) {
            const int hy = static_cast<int>(y) - static_cast<int>(oy);
            const size_t row_off = static_cast<size_t>(y) * W;
            if (g_layered_solid_row_provider) {
                g_layered_solid_row_provider(-static_cast<int>(ox), hy, static_cast<int>(W), solid.data() + row_off);
                for (uint32_t x = 0; x < W; ++x) {
                    const OptionBCap::Pixel& p = cap.pixels[row_off + x];
                    if (p.bg_mask & 8u) ++bg3_cnt;
                    if (p.flags & 4u) {
                        solid[row_off + x] = -1;
                    } else if (solid[row_off + x] >= 0) {
                        solid[row_off + x] = static_cast<int8_t>(solid[row_off + x] & 1);
                        ++valid_solid;
                    }
                }
            } else {
                for (uint32_t x = 0; x < W; ++x) {
                    const OptionBCap::Pixel& p = cap.pixels[row_off + x];
                    if (p.bg_mask & 8u) ++bg3_cnt;
                    if ((p.flags & 4u) == 0u) {
                        const int s = g_layered_solid_provider(
                            static_cast<int>(x) - static_cast<int>(ox), hy);
                        if (s >= 0) {
                            solid[row_off + x] = static_cast<int8_t>(s & 1);
                            ++valid_solid;
                        }
                    }
                }
            }
        }
    }

    const bool field_scene = (valid_solid >= N / 10u) && (bg3_cnt >= N / 7u);
    if (!field_scene) {
        out_scene_active = false;
        if (composite_rgb) {
            for (size_t i = 0; i < N; ++i) {
                p0[i * 4u + 0] = composite_rgb[i * 3u + 0];
                p0[i * 4u + 1] = composite_rgb[i * 3u + 1];
                p0[i * 4u + 2] = composite_rgb[i * 3u + 2];
                p0[i * 4u + 3] = 255u;
            }
        }
        return;
    }
    out_scene_active = true;

    // 1. HUD extraction (BG0 + UI sprites overlapping/adjacent to BG0 bands) -> Plane 4.
    thread_local std::vector<uint8_t> bg0_row, bg0_band;
    bg0_row.assign(H, 0);
    bg0_band.assign(H, 0);
    bool bg0_any = false;
    for (uint32_t y = 0; y < H; ++y) {
        const size_t row_off = static_cast<size_t>(y) * W;
        for (uint32_t x = 0; x < W; ++x) {
            if ((cap.pixels[row_off + x].bg_mask & 1u) != 0u &&
                (cap.pixels[row_off + x].flags & 4u) == 0u) {
                bg0_row[y] = 1;
                bg0_any = true;
                break;
            }
        }
    }
    if (bg0_any) {
        for (uint32_t y = 0; y < H; ++y) {
            const int y0 = std::max(0, static_cast<int>(y) - 20);
            const int y1 = std::min(static_cast<int>(H) - 1, static_cast<int>(y) + 20);
            for (int yy = y0; yy <= y1; ++yy) {
                if (bg0_row[yy]) { bg0_band[y] = 1; break; }
            }
        }
    }

    obj_is_hud.assign(N, 0);
    vis.assign(N, 0);
    for (uint32_t y = 0; y < H; ++y) {
        for (uint32_t x = 0; x < W; ++x) {
            const size_t start = static_cast<size_t>(y) * W + x;
            const OptionBCap::Pixel& p0_px = cap.pixels[start];
            if (p0_px.obj_idx == 0xFFu || (p0_px.flags & 4u) != 0u || vis[start]) continue;
            vis[start] = 1;
            bfs_q.clear();
            comp_pts.clear();
            bfs_q.push_back(static_cast<int>(start));
            size_t head = 0;
            uint8_t min_prio = p0_px.obj_prio;
            bool in_hud_band = false;
            while (head < bfs_q.size()) {
                const int idx = bfs_q[head++];
                comp_pts.push_back(idx);
                const int cy = idx / static_cast<int>(W);
                const int cx = idx - cy * static_cast<int>(W);
                const OptionBCap::Pixel& cp = cap.pixels[idx];
                if (cp.obj_prio < min_prio) min_prio = cp.obj_prio;
                if ((cp.bg_mask & 1u) != 0u || bg0_band[cy]) in_hud_band = true;
                for (int dy = -2; dy <= 2; ++dy) {
                    const int ny = cy + dy;
                    if (ny < 0 || ny >= static_cast<int>(H)) continue;
                    for (int dx = -2; dx <= 2; ++dx) {
                        const int nx = cx + dx;
                        if (nx < 0 || nx >= static_cast<int>(W)) continue;
                        const size_t nidx = static_cast<size_t>(ny) * W + static_cast<size_t>(nx);
                        const OptionBCap::Pixel& np = cap.pixels[nidx];
                        if (!vis[nidx] && np.obj_idx != 0xFFu && (np.flags & 4u) == 0u) {
                            vis[nidx] = 1;
                            bfs_q.push_back(static_cast<int>(nidx));
                        }
                    }
                }
            }
            if (bg0_any && (min_prio <= 1u) && in_hud_band) {
                for (int pi : comp_pts) obj_is_hud[pi] = 1;
            }
        }
    }

    for (uint32_t y = 0; y < H; ++y) {
        const OptionBCap::Line& line = cap.lines[y];
        const size_t row_off = static_cast<size_t>(y) * W;
        for (uint32_t x = 0; x < W; ++x) {
            const size_t i = row_off + x;
            const OptionBCap::Pixel& p = cap.pixels[i];
            if (p.flags & 4u) continue;
            const bool has_bg0 = (p.bg_mask & 1u) != 0u;
            const bool has_hobj = obj_is_hud[i] != 0u && p.obj_idx != 0xFFu;
            if (!has_bg0 && !has_hobj) continue;
            if (has_hobj && (!has_bg0 ||
                (static_cast<int>(p.obj_prio) * 256 + static_cast<int>(p.obj_idx)) <
                (static_cast<int>(p.bg_prio & 3u) * 256 + 128))) {
                write_optb_gba555(p.obj_color, line.effect, (p.flags & 1u) != 0u,
                                  line.bldy, 255u, &p4[i * 4u]);
            } else {
                write_optb_gba555(p.bg_color[0], line.effect, (p.bg_t1_mask & 1u) != 0u,
                                  line.bldy, 255u, &p4[i * 4u]);
            }
        }
    }

    // 2. Fast BGR555 morphological 3x3-core perimeter void detection on BG3 -> Plane 0.
    //    (Note: BG2 indoor matte/doorway rim stays on BG2/Plane 4 so south room exits frame properly.)
    auto detect_void_layer = [&](uint8_t bg_layer, std::vector<uint8_t>& out_mask) {
        out_mask.assign(N, 0);
        if (W == 0 || H == 0) return;
        blk_valid.assign(N, 0);
        blk_col.resize(N);
        blk_q.clear();
        const uint8_t bit = static_cast<uint8_t>(1u << bg_layer);

        for (uint32_t y = 0; y < H; ++y) {
            const OptionBCap::Line& line = cap.lines[y];
            const size_t row_off = static_cast<size_t>(y) * W;
            for (uint32_t x = 0; x < W; ++x) {
                const size_t idx = row_off + x;
                const OptionBCap::Pixel& p = cap.pixels[idx];
                if ((p.flags & 4u) != 0u || (p.bg_mask & bit) == 0u) continue;
                uint16_t col = p.bg_color[bg_layer];
                if ((line.effect == 2u || line.effect == 3u) && line.bldy != 0u &&
                    (p.bg_t1_mask & bit) != 0u) {
                    col = (line.effect == 2u)
                        ? brighten_gba555(col, line.bldy)
                        : darken_gba555(col, line.bldy);
                }
                col &= 0x7FFFu;
                // (5 * 255 + 15) / 31 = 41, (6 * 255 + 15) / 31 = 49: matches max(r8,g8,b8) <= 45 identically.
                if ((col & 0x1Fu) > 5u || ((col >> 5) & 0x1Fu) > 5u || ((col >> 10) & 0x1Fu) > 5u) continue;
                blk_valid[idx] = 1;
                blk_col[idx] = col;
            }
        }

        for (uint32_t y = 1; y + 1 < H; ++y) {
            const size_t r = static_cast<size_t>(y) * W;
            const size_t rm = r - W;
            const size_t rp = r + W;
            for (uint32_t x = 1; x + 1 < W; ++x) {
                const size_t i = r + x;
                if ((blk_valid[i] & 1u) == 0u) continue;
                const uint16_t c = blk_col[i];
                if (blk_valid[i - 1] && blk_col[i - 1] == c &&
                    blk_valid[i + 1] && blk_col[i + 1] == c &&
                    blk_valid[rm + x] && blk_col[rm + x] == c &&
                    blk_valid[rm + x - 1] && blk_col[rm + x - 1] == c &&
                    blk_valid[rm + x + 1] && blk_col[rm + x + 1] == c &&
                    blk_valid[rp + x] && blk_col[rp + x] == c &&
                    blk_valid[rp + x - 1] && blk_col[rp + x - 1] == c &&
                    blk_valid[rp + x + 1] && blk_col[rp + x + 1] == c) {
                    blk_valid[i] = 3u;
                }
            }
        }
        // Perimeter edges: allow exact-match 3x3 cores up against image border too.
        auto mark_border_core = [&](uint32_t y, uint32_t x) {
            const size_t cidx = static_cast<size_t>(y) * W + x;
            if ((blk_valid[cidx] & 1u) == 0u) return;
            const uint16_t c = blk_col[cidx];
            const uint32_t y0 = (y > 0) ? (y - 1u) : 0u;
            const uint32_t y1 = std::min(H - 1u, y + 1u);
            const uint32_t x0 = (x > 0) ? (x - 1u) : 0u;
            const uint32_t x1 = std::min(W - 1u, x + 1u);
            for (uint32_t ny = y0; ny <= y1; ++ny) {
                for (uint32_t nx = x0; nx <= x1; ++nx) {
                    const size_t ni = static_cast<size_t>(ny) * W + nx;
                    if ((blk_valid[ni] & 1u) == 0u || blk_col[ni] != c) return;
                }
            }
            blk_valid[cidx] = 3u;
        };
        if (H > 0 && W > 0) {
            for (uint32_t x = 0; x < W; ++x) {
                mark_border_core(0u, x);
                if (H > 1u) mark_border_core(H - 1u, x);
            }
            for (uint32_t y = 1; y + 1 < H; ++y) {
                mark_border_core(y, 0u);
                if (W > 1u) mark_border_core(y, W - 1u);
            }
        }

        const uint32_t bw4 = (W + 3u) / 4u;
        const uint32_t bh4 = (H + 3u) / 4u;
        for (uint32_t ty = 0; ty < bh4; ++ty) {
            const uint32_t y0 = ty * 4u;
            const uint32_t y1 = std::min(H, y0 + 4u);
            for (uint32_t tx = 0; tx < bw4; ++tx) {
                const uint32_t x0 = tx * 4u;
                const uint32_t x1 = std::min(W, x0 + 4u);
                bool on_perim = (tx == 0 || tx + 1 == bw4 || ty == 0 || ty + 1 == bh4);
                if (!on_perim) {
                    if (y0 > 0) {
                        for (uint32_t x = x0; x < x1 && !on_perim; ++x) {
                            if ((cap.pixels[static_cast<size_t>(y0 - 1u) * W + x].flags & 4u) != 0u) on_perim = true;
                        }
                    }
                    if (!on_perim && y1 < H) {
                        for (uint32_t x = x0; x < x1 && !on_perim; ++x) {
                            if ((cap.pixels[static_cast<size_t>(y1) * W + x].flags & 4u) != 0u) on_perim = true;
                        }
                    }
                    if (!on_perim && x0 > 0) {
                        for (uint32_t y = y0; y < y1 && !on_perim; ++y) {
                            if ((cap.pixels[static_cast<size_t>(y) * W + (x0 - 1u)].flags & 4u) != 0u) on_perim = true;
                        }
                    }
                    if (!on_perim && x1 < W) {
                        for (uint32_t y = y0; y < y1 && !on_perim; ++y) {
                            if ((cap.pixels[static_cast<size_t>(y) * W + x1].flags & 4u) != 0u) on_perim = true;
                        }
                    }
                }
                if (!on_perim) continue;

                bool ok_patch = true;
                uint16_t c0 = 0xFFFFu;
                uint32_t valid_cnt = 0;
                for (uint32_t y = y0; y < y1 && ok_patch; ++y) {
                    const size_t row_off = static_cast<size_t>(y) * W;
                    for (uint32_t x = x0; x < x1; ++x) {
                        const size_t idx = row_off + x;
                        if ((cap.pixels[idx].flags & 4u) != 0u) continue;
                        if ((blk_valid[idx] & 1u) == 0u) { ok_patch = false; break; }
                        const uint16_t col = blk_col[idx];
                        if (c0 == 0xFFFFu) c0 = col;
                        else if (col != c0) { ok_patch = false; break; }
                        ++valid_cnt;
                    }
                }
                if (!ok_patch || valid_cnt < 4u) continue;
                for (uint32_t y = y0; y < y1; ++y) {
                    const size_t row_off = static_cast<size_t>(y) * W;
                    for (uint32_t x = x0; x < x1; ++x) {
                        const size_t idx = row_off + x;
                        if ((blk_valid[idx] & 2u) == 0u || out_mask[idx]) continue;
                        out_mask[idx] = 1;
                        blk_q.push_back(static_cast<int>(idx));
                    }
                }
            }
        }

        size_t head = 0;
        static constexpr int kDirs[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        while (head < blk_q.size()) {
            const int idx = blk_q[head++];
            const int cy = idx / static_cast<int>(W);
            const int cx = idx - cy * static_cast<int>(W);
            const uint16_t c = blk_col[idx];
            for (const auto& d : kDirs) {
                const int nx = cx + d[0];
                const int ny = cy + d[1];
                if (nx < 0 || nx >= static_cast<int>(W) || ny < 0 || ny >= static_cast<int>(H)) continue;
                const size_t nidx = static_cast<size_t>(ny) * W + static_cast<size_t>(nx);
                if ((blk_valid[nidx] & 2u) == 0u || out_mask[nidx]) continue;
                if (blk_col[nidx] == c) {
                    out_mask[nidx] = 1;
                    blk_q.push_back(static_cast<int>(nidx));
                }
            }
        }

        const size_t core_cnt = blk_q.size();
        for (size_t i = 0; i < core_cnt; ++i) {
            const int idx = blk_q[i];
            const int cy = idx / static_cast<int>(W);
            const int cx = idx - cy * static_cast<int>(W);
            const uint16_t c = blk_col[idx];
            for (int dy = -1; dy <= 1; ++dy) {
                const int ny = cy + dy;
                if (ny < 0 || ny >= static_cast<int>(H)) continue;
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    const int nx = cx + dx;
                    if (nx < 0 || nx >= static_cast<int>(W)) continue;
                    const size_t nidx = static_cast<size_t>(ny) * W + static_cast<size_t>(nx);
                    if ((blk_valid[nidx] & 1u) != 0u && !out_mask[nidx] && blk_col[nidx] == c) {
                        out_mask[nidx] = 1;
                    }
                }
            }
        }
    };

    detect_void_layer(3, void_bg3);

    uint16_t void_fill_col = 0;
    bool has_void_fill = false;
    for (size_t i = 0; i < N && !has_void_fill; ++i) {
        const OptionBCap::Pixel& p = cap.pixels[i];
        if ((p.flags & 4u) != 0u) continue;
        if (void_bg3[i] && (p.bg_mask & 8u)) { void_fill_col = p.bg_color[3]; has_void_fill = true; }
    }
    for (uint32_t y = 0; y < H; ++y) {
        const OptionBCap::Line& line = cap.lines[y];
        const bool active_row = (y >= oy) && (y < oy + GbaPpu::kScreenHeight);
        const size_t row_off = static_cast<size_t>(y) * W;
        for (uint32_t x = 0; x < W; ++x) {
            const size_t i = row_off + x;
            const OptionBCap::Pixel& p = cap.pixels[i];
            if (p.flags & 4u) {
                p0[i * 4u + 3] = 255u;
            } else if (!active_row) {
                write_optb_gba555(line.backdrop_color, line.effect, false, line.bldy, 255u, &p0[i * 4u]);
            } else if (void_bg3[i] && (p.bg_mask & 8u)) {
                write_optb_gba555(p.bg_color[3], line.effect, (p.bg_t1_mask & 8u) != 0u,
                                  line.bldy, 255u, &p0[i * 4u]);
            } else if (has_void_fill) {
                write_optb_gba555(void_fill_col, line.effect, false, line.bldy, 255u, &p0[i * 4u]);
            } else {
                write_optb_gba555(line.backdrop_color, line.effect, false, line.bldy, 255u, &p0[i * 4u]);
            }
        }
    }

    // 3. Per-pixel BG plane assignment using batched row tile/tileset mask provider + auto fallback.
    bg1_trans.assign(N, 0u);
    bg_pl[1].assign(N, -1);
    bg_pl[2].assign(N, -1);
    bg_pl[3].assign(N, -1);
    authored.assign(N, 0);
    floor_set.assign(32768u, 0u);
    bfs_q.clear();
    bool has_plane5 = false;

    for (uint32_t y = 0; y < H; ++y) {
        const int hy = static_cast<int>(y) - static_cast<int>(oy);
        const OptionBCap::Line& line = cap.lines[y];
        const size_t row_off = static_cast<size_t>(y) * W;
        uint8_t row_bg_mask = 0u;
        for (uint32_t x = 0; x < W; ++x) {
            const OptionBCap::Pixel& p = cap.pixels[row_off + x];
            if ((p.flags & 4u) == 0u) row_bg_mask |= p.bg_mask;
        }
        if (g_layered_plane_row_provider) {
            if (row_bg_mask & 2u) {
                g_layered_plane_row_provider(-static_cast<int>(ox), hy, static_cast<int>(W), 1,
                                             bg_pl[1].data() + row_off);
            }
            if (row_bg_mask & 4u) {
                g_layered_plane_row_provider(-static_cast<int>(ox), hy, static_cast<int>(W), 2,
                                             bg_pl[2].data() + row_off);
            }
            if (row_bg_mask & 8u) {
                g_layered_plane_row_provider(-static_cast<int>(ox), hy, static_cast<int>(W), 3,
                                             bg_pl[3].data() + row_off);
            }
            for (uint32_t x = 0; x < W; ++x) {
                const size_t i = row_off + x;
                const OptionBCap::Pixel& p = cap.pixels[i];
                // A provider may flag a hand-authored plane with +16 (Mother 3 building
                // masks); strip it here and remember it for the side channel below.
                for (int b = 1; b <= 3; ++b) {
                    if (bg_pl[b][i] >= 16) {
                        bg_pl[b][i] = static_cast<int8_t>(bg_pl[b][i] - 16);
                        if ((p.bg_mask & (1u << b)) != 0u) authored[i] = 1;
                    }
                }
                if (p.flags & 4u) {
                    bg_pl[1][i] = bg_pl[2][i] = bg_pl[3][i] = -1;
                    continue;
                }
                // Authored pixels: a hidden BG3 pixel under an opaque BG2 one must not lift the
                // pixel above what the visible BG2 pixel was authored as (planes are max-merged).
                if (authored[i] && (p.bg_mask & 4u) && (p.bg_mask & 8u) &&
                    bg_pl[2][i] >= 0 && bg_pl[3][i] > bg_pl[2][i])
                    bg_pl[3][i] = bg_pl[2][i];
                const bool t1_bg1 = (line.effect == 1u) && ((p.bg_t1_mask & 2u) != 0u);
                if ((p.bg_mask & 2u) != 0u && t1_bg1) {
                    bg1_trans[i] = 1;
                    bg_pl[1][i] = -1;
                } else if ((p.bg_mask & 2u) != 0u) {
                    if (bg_pl[1][i] < 0) bg_pl[1][i] = 4;
                    else if (bg_pl[1][i] == 5) has_plane5 = true;
                } else {
                    bg_pl[1][i] = -1;
                }

                if ((p.bg_mask & 4u) != 0u) {
                    if (bg_pl[2][i] < 0) bg_pl[2][i] = 2;
                    else if (bg_pl[2][i] == 5) has_plane5 = true;
                } else {
                    bg_pl[2][i] = -1;
                }

                if ((p.bg_mask & 8u) != 0u && !void_bg3[i]) {
                    if (bg_pl[3][i] < 0) bg_pl[3][i] = 1;
                    else if (bg_pl[3][i] == 5) has_plane5 = true;
                } else {
                    bg_pl[3][i] = -1;
                }
            }
        } else {
            for (uint32_t x = 0; x < W; ++x) {
                const int hx = static_cast<int>(x) - static_cast<int>(ox);
                const size_t i = row_off + x;
                const OptionBCap::Pixel& p = cap.pixels[i];
                if (p.flags & 4u) continue;

                const bool t1_bg1 = (line.effect == 1u) && ((p.bg_t1_mask & 2u) != 0u);
                if ((p.bg_mask & 2u) != 0u && t1_bg1) {
                    bg1_trans[i] = 1;
                } else if ((p.bg_mask & 2u) != 0u) {
                    int pl = g_layered_plane_provider ? g_layered_plane_provider(hx, hy, 1) : -1;
                    if (pl >= 16) { pl -= 16; authored[i] = 1; }
                    if (pl < 0) pl = 4;
                    if (pl == 5) has_plane5 = true;
                    bg_pl[1][i] = static_cast<int8_t>(pl);
                }

                if ((p.bg_mask & 4u) != 0u) {
                    int pl = g_layered_plane_provider ? g_layered_plane_provider(hx, hy, 2) : -1;
                    if (pl >= 16) { pl -= 16; authored[i] = 1; }
                    if (pl < 0) pl = 2;
                    if (pl == 5) has_plane5 = true;
                    bg_pl[2][i] = static_cast<int8_t>(pl);
                }

                if ((p.bg_mask & 8u) != 0u && !void_bg3[i]) {
                    int pl = g_layered_plane_provider ? g_layered_plane_provider(hx, hy, 3) : -1;
                    if (pl >= 16) { pl -= 16; authored[i] = 1; }
                    if (pl < 0) pl = 1;
                    if (pl == 5) has_plane5 = true;
                    bg_pl[3][i] = static_cast<int8_t>(pl);
                }
            }
        }
    }

    // 4. Pixel-accurate exterior peel (only when legacy Plane 5 mixed cells are present;
    //    Option 2 pre-baked metatile masks & clean Option 3 layers are already exact).
    if (has_plane5) {
        for (uint32_t y = oy; y < oy + GbaPpu::kScreenHeight; ++y) {
            const size_t row_off = static_cast<size_t>(y) * W;
            for (uint32_t x = ox; x < ox + GbaPpu::kScreenWidth; ++x) {
                const size_t i = row_off + x;
                const OptionBCap::Pixel& p = cap.pixels[i];
                if (p.flags & 4u) continue;
                const int bg2_pl = bg_pl[2][i];
                const int bg3_pl = bg_pl[3][i];
                if (bg2_pl == 1 || bg3_pl == 1) {
                    const uint16_t shell_col = ((bg2_pl == 1) ? p.bg_color[2] : p.bg_color[3]) & 0x7FFFu;
                    uint8_t rgb[3];
                    to_rgb888(shell_col, rgb);
                    if (std::max({rgb[0], rgb[1], rgb[2]}) > 72u) {
                        bfs_q.push_back(static_cast<int>(i));
                    }
                }
            }
        }
        static constexpr int kDirs[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        thread_local std::vector<uint16_t> floor_cnt;
        thread_local std::vector<uint16_t> wall_cnt;
        thread_local std::vector<uint8_t> wall_set;
        thread_local std::vector<uint8_t> visited;
        floor_cnt.assign(32768u, 0u);
        wall_cnt.assign(32768u, 0u);
        wall_set.assign(32768u, 0u);
        for (int idx : bfs_q) {
            const OptionBCap::Pixel& p = cap.pixels[idx];
            const uint16_t shell_col = ((bg_pl[2][idx] == 1) ? p.bg_color[2] : p.bg_color[3]) & 0x7FFFu;
            if (solid[idx] == 0) {
                if (floor_cnt[shell_col] < 65535u) ++floor_cnt[shell_col];
            } else {
                if (wall_cnt[shell_col] < 65535u) ++wall_cnt[shell_col];
            }
        }
        for (size_t c = 0; c < 32768u; ++c) {
            if (floor_cnt[c] >= 48u) floor_set[c] = 1u;
            if (wall_cnt[c] >= 48u) wall_set[c] = 1u;
        }
        visited.assign(N, 0u);
        for (int idx : bfs_q) visited[idx] = 1u;

        // Also seed mixed (Plane 5) edge pixels directly touching void that carry known wall/floor colors.
        for (uint32_t y = 0; y < H; ++y) {
            const size_t row_off = static_cast<size_t>(y) * W;
            for (uint32_t x = 0; x < W; ++x) {
                const size_t i = row_off + x;
                if (visited[i]) continue;
                if (bg_pl[2][i] != 5 && bg_pl[3][i] != 5) continue;
                bool adj_void = false;
                for (const auto& d : kDirs) {
                    const int nx = static_cast<int>(x) + d[0];
                    const int ny = static_cast<int>(y) + d[1];
                    if (nx < 0 || nx >= static_cast<int>(W) || ny < 0 || ny >= static_cast<int>(H) ||
                        void_bg3[static_cast<size_t>(ny) * W + static_cast<size_t>(nx)]) {
                        adj_void = true;
                        break;
                    }
                }
                if (!adj_void) continue;
                const uint16_t eff_col = ((bg_pl[2][i] == 5) ? cap.pixels[i].bg_color[2] : cap.pixels[i].bg_color[3]) & 0x7FFFu;
                if (floor_set[eff_col] != 0u || wall_set[eff_col] != 0u) {
                    if (bg_pl[2][i] == 5) bg_pl[2][i] = 1;
                    if (bg_pl[3][i] == 5) bg_pl[3][i] = 1;
                    visited[i] = 1u;
                    bfs_q.push_back(static_cast<int>(i));
                }
            }
        }

        size_t head = 0;
        while (head < bfs_q.size()) {
            const int idx = bfs_q[head++];
            const int cy = idx / static_cast<int>(W);
            const int cx = idx - cy * static_cast<int>(W);
            for (const auto& d : kDirs) {
                const int nx = cx + d[0];
                const int ny = cy + d[1];
                if (nx < 0 || nx >= static_cast<int>(W) || ny < 0 || ny >= static_cast<int>(H)) continue;
                const size_t nidx = static_cast<size_t>(ny) * W + static_cast<size_t>(nx);
                if (visited[nidx]) continue;
                const int p2v = bg_pl[2][nidx];
                const int p3v = bg_pl[3][nidx];
                const bool has2 = (p2v == 2 || p2v == 5);
                const bool has3 = (p3v == 2 || p3v == 5);
                if (!has2 && !has3) continue;
                const bool is_mixed = (has2 ? (p2v == 5) : false) || (has3 ? (p3v == 5) : false);
                const uint16_t eff_col = (has2 ? cap.pixels[nidx].bg_color[2] : cap.pixels[nidx].bg_color[3]) & 0x7FFFu;
                const bool peel = (floor_set[eff_col] != 0u) || (is_mixed && wall_set[eff_col] != 0u);
                if (peel) {
                    if (has2) bg_pl[2][nidx] = 1;
                    if (has3) bg_pl[3][nidx] = 1;
                    visited[nidx] = 1u;
                    bfs_q.push_back(static_cast<int>(nidx));
                }
            }
        }

        // Convert any un-peeled Plane 5 mixed prop pixels to Plane 2 before painting.
        for (size_t i = 0; i < N; ++i) {
            if (bg_pl[2][i] == 5) bg_pl[2][i] = 2;
            if (bg_pl[3][i] == 5) bg_pl[3][i] = 2;
        }

        // Reclaim isolated 1px horizontal/vertical floor-plank and wall-log seams exposed between peeled Shell pixels.
        auto is_peeled_shell = [&](size_t j) -> bool {
            return (bg_pl[2][j] != 2 && bg_pl[3][j] != 2) && (bg_pl[2][j] == 1 || bg_pl[3][j] == 1);
        };
        for (int pass = 0; pass < 2; ++pass) {
            for (uint32_t y = 1; y + 1 < H; ++y) {
                const size_t row_off = static_cast<size_t>(y) * W;
                for (uint32_t x = 1; x + 1 < W; ++x) {
                    const size_t i = row_off + x;
                    if (bg_pl[2][i] != 2 && bg_pl[3][i] != 2) continue;
                    const uint16_t eff_col = ((bg_pl[2][i] == 2) ? cap.pixels[i].bg_color[2] : cap.pixels[i].bg_color[3]) & 0x7FFFu;
                    uint8_t rgb[3];
                    to_rgb888(eff_col, rgb);
                    if (std::max({rgb[0], rgb[1], rgb[2]}) > 72u) continue;
                    if ((is_peeled_shell(i - W) && is_peeled_shell(i + W)) ||
                        (is_peeled_shell(i - 1) && is_peeled_shell(i + 1))) {
                        if (bg_pl[2][i] == 2) bg_pl[2][i] = 1;
                        if (bg_pl[3][i] == 2) bg_pl[3][i] = 1;
                    }
                }
            }
        }
    }

    // 5. Paint Planes 0 (Void/Mirror interior), 1 (Shell+Glass), 2 (Furniture), 3 (Sprites OBJ), 4 (FG+HUD).
    vis.assign(N, 0u);
    bool has_p0_scene = false;
    for (uint32_t y = 0; y < H; ++y) {
        const OptionBCap::Line& line = cap.lines[y];
        const size_t row_off = static_cast<size_t>(y) * W;
        auto write_blend_target1 = [&](uint16_t color555, uint8_t* dst) {
            if (line.eva == 0u) return;
            const uint32_t eva = std::min<uint32_t>(line.eva, 16u);
            const uint32_t evb = std::min<uint32_t>(line.evb, 16u);
            uint8_t rgb[3];
            to_rgb888(color555, rgb);
            uint32_t a_eff = eva;
            if (eva + evb < 16u) {
                a_eff = 16u - evb;
                for (int c = 0; c < 3; ++c) {
                    rgb[c] = static_cast<uint8_t>((static_cast<uint32_t>(rgb[c]) * eva + a_eff / 2u) / a_eff);
                }
            }
            dst[0] = rgb[0];
            dst[1] = rgb[1];
            dst[2] = rgb[2];
            dst[3] = static_cast<uint8_t>(std::clamp<uint32_t>((a_eff * 255u + 8u) / 16u, 16u, 255u));
        };
        for (uint32_t x = 0; x < W; ++x) {
            const size_t i = row_off + x;
            const OptionBCap::Pixel& p = cap.pixels[i];
            if (p.flags & 4u) continue;

            // Paint BG layers into their target semantic planes in back-to-front order (BG3 -> BG2 -> BG1),
            // using GBA priority tie-breaking when two visible BG layers share a plane.
            int best_key[5] = {0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF};
            int best_layer[5] = {-1, -1, -1, -1, -1};
            for (int layer = 1; layer <= 3; ++layer) {
                const int pl = bg_pl[layer][i];
                if (pl < 0 || pl > 4 || pl == 3) continue;
                const int prio = static_cast<int>((p.bg_prio >> (2 * layer)) & 3u);
                const int key = prio * 256 + 128 + layer;
                if (key < best_key[pl]) {
                    best_key[pl] = key;
                    best_layer[pl] = layer;
                }
            }

            // Check strict GBA priority occlusion for world sprites (OBJ) against all active opaque BGs.
            bool has_world_obj = (p.obj_idx != 0xFFu && !obj_is_hud[i]);
            bool obj_occluded_by_bg = false;
            int occluding_pl = -1;
            if (has_world_obj) {
                const int obj_key = static_cast<int>(p.obj_prio) * 256 + static_cast<int>(p.obj_idx);
                int best_opaque_bg_key = 0x7FFFFFFF;
                for (int layer = 0; layer <= 3; ++layer) {
                    if ((p.bg_mask & (1u << layer)) == 0u) continue;
                    if (layer == 3 && void_bg3[i]) continue;
                    if (line.effect == 1u && ((p.bg_t1_mask & (1u << layer)) != 0u)) continue;
                    const int pl = (layer == 0) ? ((p4[i * 4u + 3] == 255u) ? 4 : -1) : bg_pl[layer][i];
                    if (pl < 0) continue;
                    const int prio = static_cast<int>((p.bg_prio >> (2 * layer)) & 3u);
                    const int k = prio * 256 + 128 + layer;
                    if (k < best_opaque_bg_key) {
                        best_opaque_bg_key = k;
                        occluding_pl = pl;
                    }
                }
                if (best_opaque_bg_key < obj_key) {
                    obj_occluded_by_bg = true;
                }
            }

            // Wall mirror aperture: translucent BG1 glass over an opening in BG2 showing BG3 (+ behind-BG2 reflection OBJ).
            // Never trigger on detached scrolling overlays (e.g. Map 12 drifting night fog).
            const bool is_mirror_glass = (g_layered_bg1_detached == 0) &&
                                         (bg1_trans[i] != 0u) && (solid[i] == 1) &&
                                         ((p.bg_mask & 4u) == 0u) && ((p.bg_mask & 8u) != 0u) &&
                                         (bg_pl[3][i] >= 0 && bg_pl[3][i] <= 1);
            if (is_mirror_glass) {
                if (has_world_obj && !obj_occluded_by_bg) {
                    write_optb_gba555(p.obj_color, line.effect, (p.flags & 1u) != 0u,
                                      line.bldy, 255u, &p0[i * 4u]);
                } else {
                    write_optb_gba555(p.bg_color[3], line.effect, (p.bg_t1_mask & 8u) != 0u,
                                      line.bldy, 255u, &p0[i * 4u]);
                }
                write_blend_target1(p.bg_color[1], &p1[i * 4u]);
                continue;
            }

            if (best_layer[0] >= 0) {
                const int l = best_layer[0];
                write_optb_gba555(p.bg_color[l], line.effect, (p.bg_t1_mask & (1u << l)) != 0u,
                                  line.bldy, 255u, &p0[i * 4u]);
                vis[i] = 1u;
                has_p0_scene = true;
            }
            if (best_layer[1] >= 0) {
                const int l = best_layer[1];
                write_optb_gba555(p.bg_color[l], line.effect, (p.bg_t1_mask & (1u << l)) != 0u,
                                  line.bldy, 255u, &p1[i * 4u]);
                // Keep Plane 0 populated with behind-wall BG3 + mirror reflection OBJ behind opaque mirror frame
                // so stereo parallax into the mirror reveals interior reflection rather than void.
                if (l == 2 && (p.bg_mask & 8u) != 0u && solid[i] == 1 && !void_bg3[i]) {
                    if (has_world_obj && obj_occluded_by_bg && occluding_pl == 1) {
                        write_optb_gba555(p.obj_color, line.effect, (p.flags & 1u) != 0u,
                                          line.bldy, 255u, &p0[i * 4u]);
                    } else {
                        write_optb_gba555(p.bg_color[3], line.effect, (p.bg_t1_mask & 8u) != 0u,
                                          line.bldy, 255u, &p0[i * 4u]);
                    }
                }
            }
            if (best_layer[2] >= 0) {
                const int l = best_layer[2];
                write_optb_gba555(p.bg_color[l], line.effect, (p.bg_t1_mask & (1u << l)) != 0u,
                                  line.bldy, 255u, &p2[i * 4u]);
            }
            if (best_layer[4] >= 0 && p4[i * 4u + 3] == 0u) {
                const int l = best_layer[4];
                write_optb_gba555(p.bg_color[l], line.effect, (p.bg_t1_mask & (1u << l)) != 0u,
                                  line.bldy, 255u, &p4[i * 4u]);
            }

            // World sprites (OBJ) -> Plane 3 unless occluded behind Plane <= 2 opaque BG geometry.
            if (has_world_obj && (!obj_occluded_by_bg || occluding_pl == 4)) {
                const uint8_t alpha = (p.flags & 1u)
                    ? static_cast<uint8_t>(std::clamp<uint32_t>((line.eva * 255u + 8u) / 16u, 16u, 255u))
                    : 255u;
                write_optb_gba555(p.obj_color, line.effect, (p.flags & 1u) != 0u,
                                  line.bldy, alpha, &p3[i * 4u]);
            }

            // Translucent BG1 overlays (sunbeam / fog) -> Plane 4 if opaque HUD/FG didn't cover it.
            if (bg1_trans[i] && line.eva > 0u && p4[i * 4u + 3] == 0u) {
                write_blend_target1(p.bg_color[1], &p4[i * 4u]);
            }
        }
    }

    // 6. Clean Plane 1 (Shell) continuation behind Plane 2 (Furniture) & Plane 4 (FG) so
    // stereoscopic head parallax never peeks behind furniture/walls into empty cracks.
    left_clean.resize(W);
    right_clean.resize(W);
    thread_local std::vector<int> left_span;
    thread_local std::vector<int> right_span;
    left_span.resize(W);
    right_span.resize(W);
    for (uint32_t y = 0; y < H; ++y) {
        const size_t row_off = static_cast<size_t>(y) * W;
        bool row_has_hole = false;
        for (uint32_t x = 0; x < W; ++x) {
            const size_t i = row_off + x;
            if ((cap.pixels[i].flags & 4u) == 0u && !void_bg3[i] && p1[i * 4u + 3] == 0u &&
                (p2[i * 4u + 3] != 0u || p4[i * 4u + 3] != 0u)) {
                row_has_hole = true;
                break;
            }
        }
        if (!row_has_hole) continue;
        auto is_clean_src = [&](uint32_t cx) -> bool {
            const size_t idx = row_off + cx;
            if ((cap.pixels[idx].flags & 4u) != 0u || void_bg3[idx]) return false;
            if (p1[idx * 4u + 3] != 255u) return false;
            const uint8_t mx = std::max({p1[idx * 4u + 0], p1[idx * 4u + 1], p1[idx * 4u + 2]});
            if (mx > 72u) return true;
            // Allow continuous horizontal floor/wall plank seams away from prop borders.
            if (y > 0 && y + 1 < H) {
                const size_t up = idx - W;
                const size_t dn = idx + W;
                if (p1[up * 4u + 3] == 255u && p1[dn * 4u + 3] == 255u &&
                    p2[up * 4u + 3] == 0u && p2[dn * 4u + 3] == 0u &&
                    std::max({p1[up * 4u + 0], p1[up * 4u + 1], p1[up * 4u + 2]}) > 72u &&
                    std::max({p1[dn * 4u + 0], p1[dn * 4u + 1], p1[dn * 4u + 2]}) > 72u) {
                    return true;
                }
            }
            return false;
        };

        int last = -1;
        int span = 0;
        for (uint32_t x = 0; x < W; ++x) {
            const size_t i = row_off + x;
            if ((cap.pixels[i].flags & 4u) != 0u || void_bg3[i]) {
                last = -1;
                span = 0;
            } else {
                ++span;
                if (is_clean_src(x)) last = static_cast<int>(x);
            }
            left_clean[x] = last;
            left_span[x] = span;
        }
        last = -1;
        span = 0;
        for (int x = static_cast<int>(W) - 1; x >= 0; --x) {
            const size_t i = row_off + static_cast<size_t>(x);
            if ((cap.pixels[i].flags & 4u) != 0u || void_bg3[i]) {
                last = -1;
                span = 0;
            } else {
                ++span;
                if (is_clean_src(static_cast<uint32_t>(x))) last = x;
            }
            right_clean[x] = last;
            right_span[x] = span;
        }
        for (uint32_t x = 0; x < W; ++x) {
            const size_t i = row_off + x;
            if ((cap.pixels[i].flags & 4u) != 0u || void_bg3[i] || p1[i * 4u + 3] != 0u) continue;
            if (p2[i * 4u + 3] == 0u && p4[i * 4u + 3] == 0u) continue;
            const int lc = left_clean[x];
            const int rc = right_clean[x];
            int src_x = -1;
            auto is_floor_src = [&](int sx) -> bool {
                const size_t sidx = row_off + static_cast<size_t>(sx);
                if (solid[sidx] == 0u) return true;
                const uint16_t bgr = static_cast<uint16_t>(
                    ((p1[sidx * 4u + 0] >> 3) & 31u) |
                    (((p1[sidx * 4u + 1] >> 3) & 31u) << 5) |
                    (((p1[sidx * 4u + 2] >> 3) & 31u) << 10));
                return floor_set[bgr] != 0u;
            };
            if (lc >= 0 && rc >= 0) {
                const bool l_floor = is_floor_src(lc);
                const bool r_floor = is_floor_src(rc);
                const bool l_narrow_wall = (left_span[lc] <= 28 && right_span[rc] >= 32) ||
                                           (!l_floor && r_floor && left_span[lc] <= 40);
                const bool r_narrow_wall = (right_span[rc] <= 28 && left_span[lc] >= 32) ||
                                           (!r_floor && l_floor && right_span[rc] <= 40);
                if (l_narrow_wall && !r_narrow_wall) {
                    src_x = rc;
                } else if (r_narrow_wall && !l_narrow_wall) {
                    src_x = lc;
                } else {
                    src_x = ((static_cast<int>(x) - lc) <= (rc - static_cast<int>(x))) ? lc : rc;
                }
            } else if (lc >= 0) {
                src_x = lc;
            } else if (rc >= 0) {
                src_x = rc;
            }
            if (src_x >= 0) {
                const size_t sidx = row_off + static_cast<size_t>(src_x);
                p1[i * 4u + 0] = p1[sidx * 4u + 0];
                p1[i * 4u + 1] = p1[sidx * 4u + 1];
                p1[i * 4u + 2] = p1[sidx * 4u + 2];
                p1[i * 4u + 3] = 255u;
            }
        }
    }
    // Second vertical pass for full-width foreground rows (e.g., south foreground wall).
    for (uint32_t y = 1; y < H; ++y) {
        const size_t row_off = static_cast<size_t>(y) * W;
        for (uint32_t x = 0; x < W; ++x) {
            const size_t i = row_off + x;
            if ((cap.pixels[i].flags & 4u) != 0u || void_bg3[i] || p1[i * 4u + 3] != 0u) continue;
            if (p2[i * 4u + 3] == 0u && p4[i * 4u + 3] == 0u) continue;
            const size_t up = i - W;
            if (!void_bg3[up] && (cap.pixels[up].flags & 4u) == 0u && p1[up * 4u + 3] != 0u) {
                p1[i * 4u + 0] = p1[up * 4u + 0];
                p1[i * 4u + 1] = p1[up * 4u + 1];
                p1[i * 4u + 2] = p1[up * 4u + 2];
                p1[i * 4u + 3] = 255u;
            }
        }
    }

    // Extend Plane 0 backdrop (sky/ocean/window view) slightly behind adjacent Plane 1 shorelines/frames
    // so strong stereo disparity never reveals black seams at edges.
    if (has_p0_scene) {
        constexpr int kBleed = 10;
        for (uint32_t y = 0; y < H; ++y) {
            const size_t row_off = static_cast<size_t>(y) * W;
            int last_x = -1000;
            for (uint32_t x = 0; x < W; ++x) {
                const size_t i = row_off + x;
                if (vis[i] == 1u) {
                    last_x = static_cast<int>(x);
                } else if (vis[i] == 0u && !void_bg3[i] && (cap.pixels[i].flags & 4u) == 0u &&
                           (static_cast<int>(x) - last_x) <= kBleed) {
                    const size_t si = row_off + static_cast<size_t>(last_x);
                    std::memcpy(&p0[i * 4u], &p0[si * 4u], 4u);
                    vis[i] = 2u;
                }
            }
            int next_x = 1000000;
            for (int x = static_cast<int>(W) - 1; x >= 0; --x) {
                const size_t i = row_off + static_cast<size_t>(x);
                if (vis[i] == 1u) {
                    next_x = x;
                } else if ((vis[i] == 0u || vis[i] == 2u) && !void_bg3[i] &&
                           (cap.pixels[i].flags & 4u) == 0u && (next_x - x) <= kBleed) {
                    if (vis[i] == 0u) {
                        const size_t si = row_off + static_cast<size_t>(next_x);
                        std::memcpy(&p0[i * 4u], &p0[si * 4u], 4u);
                        vis[i] = 2u;
                    }
                }
            }
        }
        for (uint32_t x = 0; x < W; ++x) {
            int last_y = -1000;
            for (uint32_t y = 0; y < H; ++y) {
                const size_t i = static_cast<size_t>(y) * W + x;
                if (vis[i] != 0u) {
                    last_y = static_cast<int>(y);
                } else if (!void_bg3[i] && (cap.pixels[i].flags & 4u) == 0u &&
                           (static_cast<int>(y) - last_y) <= kBleed) {
                    const size_t si = static_cast<size_t>(last_y) * W + x;
                    std::memcpy(&p0[i * 4u], &p0[si * 4u], 4u);
                }
            }
        }
    }
    // Collision side channel for the Quest STOCK 3D building fix: the topmost opaque
    // semantic pixel of a solid (collision) cell gets alpha 254 instead of 255, so the host
    // can tell walls from walkable floor without another buffer. Opaque-only, so
    // translucent overlays keep their alpha.
    // Hand-authored pixels (building masks) are flagged too: 253 = authored, 252 =
    // authored + solid.
    for (size_t i = 0; i < N; ++i) {
        const bool is_solid = solid[i] == 1, is_auth = authored[i] != 0;
        if (!is_solid && !is_auth) continue;
        for (int q = 4; q >= 0; --q) {
            uint8_t* a = masked_fb.data() + static_cast<size_t>(q) * rgba_bytes + i * 4u + 3u;
            if (*a == 0u) continue;
            // flag the topmost OPAQUE pixel; see-through overlays (mirror glass) above it
            // keep their alpha, and the host looks through them (stock3d_fix.h)
            if (*a < 252u) continue;
            if (*a == 255u) *a = is_auth ? (is_solid ? 252u : 253u) : 254u;
            break;
        }
    }
}

}  // namespace

void GbaPpu::render(uint8_t* rgb,
                    uint16_t dispcnt,
                    const uint8_t* io,
                    const uint8_t* vram,
                    const uint8_t* oam,
                    const uint8_t* pal) const {
    if (!view_expanded()) {
        // Faithful path — literally unchanged from before view expansion existed.
        for (uint32_t y = 0; y < kScreenHeight; ++y) {
            render_scanline_internal(rgb, y, dispcnt, io, vram, oam, pal,
                                     kScreenWidth, kScreenHeight);
        }
        return;
    }
    const uint32_t ow = render_width();
    const int top = static_cast<int>(native_view_y());
    for (int y = -top; y < static_cast<int>(render_height()) - top; ++y) {
        render_scanline_wide(rgb, y, dispcnt, io, vram, oam, pal, ow, native_view_x(), top);
    }
}

#if 0
    (void)oam;  // referenced below
    // Start with backdrop (palette entry 0) for every pixel. Backdrop
    // is the BG palette index 0, at PAL[0..1].
    uint16_t backdrop = load_u16_le(&pal[0]);
    uint8_t bd_rgb[3];
    to_rgb888(backdrop, bd_rgb);
    for (uint32_t y = 0; y < kScreenHeight; ++y) {
        for (uint32_t x = 0; x < kScreenWidth; ++x) {
            uint8_t* p = rgb + (y * kScreenWidth + x) * 3;
            p[0] = bd_rgb[0]; p[1] = bd_rgb[1]; p[2] = bd_rgb[2];
        }
    }

    // Forced blank (DISPCNT bit 7): output all-white per hardware.
    if (dispcnt & 0x0080u) {
        std::memset(rgb, 0xFF, kFramebufferBytes);
        return;
    }

    // Render background layers BEFORE OBJ so sprites composite on top.
    // Phase 2.5 scope: BG3 affine for BG mode 2 (GBA BIOS Nintendo
    // logo intro uses this). Other BG configurations land later.
    uint32_t bg_mode = dispcnt & 0x07u;
    bool bg3_enabled = (dispcnt & 0x0800u) != 0;
    if (bg3_enabled && (bg_mode == 1 || bg_mode == 2)) {
        // BG3 controls: BG3CNT at 0x0E, params at 0x30..0x3F.
        render_affine_bg(rgb, io, vram, pal,
                         0x0E, 0x30,
                         kScreenWidth, kScreenHeight);
    }

    // OBJ disabled?
    bool obj_enabled = (dispcnt & 0x1000u) != 0;
    if (!obj_enabled) return;

    // OBJ tile data starts at VRAM 0x10000 in tile modes (0/1/2) and
    // at VRAM 0x14000 in bitmap modes (3/4/5).
    uint32_t obj_tile_base = (bg_mode >= 3) ? 0x14000u : 0x10000u;
    bool obj_1d_mapping = (dispcnt & 0x0040u) != 0;

    // OBJ palette starts at PAL[0x200..0x3FF].
    const uint8_t* obj_pal = pal + 0x200;

    // Walk 128 OAM entries in priority-table order. For Phase 2.4 we
    // ignore priority bits and just back-to-front-draw so higher
    // OAM indices appear on top — close enough for the BIOS intro.
    for (int idx = 127; idx >= 0; --idx) {
        const uint8_t* entry = oam + idx * 8;
        uint16_t attr0 = load_u16_le(&entry[0]);
        uint16_t attr1 = load_u16_le(&entry[2]);
        uint16_t attr2 = load_u16_le(&entry[4]);

        bool rot_scale = (attr0 & 0x0100u) != 0;
        bool disable_or_double = (attr0 & 0x0200u) != 0;
        // For non-affine sprites bit 9 is the disable bit; for affine
        // sprites it's the double-size flag (handled in the affine
        // pass). Skip disabled non-affine sprites so the BIOS intro
        // screen blanks correctly once the BIOS clears the wordmark
        // by setting these bits.
        if (!rot_scale && disable_or_double) continue;

        // OBJ Mode (attr0 bits 10-11) per GBATEK § "GBA OBJs — OAM
        // Attribute 0":
        //   0 = Normal  (visible pixel)
        //   1 = Semi-Transparent (alpha-blend with BLDALPHA)
        //   2 = OBJ Window (the sprite's opaque pixels define a
        //       window region; sprite itself is NOT drawn)
        //   3 = Prohibited
        // Without this check the BIOS intro's OBJ-Window stencil
        // sprites leak into the visible frame as garbage pink pixels.
        uint32_t obj_mode = (attr0 >> 10) & 0x3u;
        if (obj_mode == 2) continue;            // window stencil — invisible
        if (obj_mode == 3) continue;            // prohibited
        // Semi-transparent (mode 1) renders as normal for now; proper
        // alpha blending lands when we wire BLDCNT / BLDALPHA.

        uint32_t shape = (attr0 >> 14) & 0x3u;
        if (shape >= 3) continue;
        uint32_t size  = (attr1 >> 14) & 0x3u;
        int sw = kSpriteWH[shape][size][0];
        int sh = kSpriteWH[shape][size][1];

        int sy = static_cast<int>(attr0 & 0xFFu);
        int sx = static_cast<int>(attr1 & 0x1FFu);
        // Y wraps at 256 (GBATEK).
        if (sy >= 160) sy -= 256;
        // X is 9 bits signed.
        if (sx & 0x100) sx -= 0x200;

        bool color256 = (attr0 & 0x2000u) != 0;
        uint32_t tile_num = attr2 & 0x3FFu;
        uint32_t palette_bank = (attr2 >> 12) & 0xFu;

        // For 1D mapping each row of tiles is contiguous in VRAM.
        // For 2D mapping each row of OBJ tiles is 32 tiles wide.
        int tiles_w = sw / 8;
        int tiles_h = sh / 8;

        // Texel sampler shared by affine + non-affine paths. Returns
        // false on transparent / out-of-VRAM, otherwise writes the
        // RGB888 pixel at (screen_x, screen_y).
        auto sample_and_emit = [&](int tex_x, int tex_y,
                                   int screen_x, int screen_y) {
            int tile_x_in_sprite = tex_x >> 3;
            int tile_y_in_sprite = tex_y >> 3;
            int px_in_tile       = tex_x & 7;
            int py_in_tile       = tex_y & 7;

            uint32_t this_tile;
            if (obj_1d_mapping) {
                this_tile = tile_num + (tile_y_in_sprite * tiles_w + tile_x_in_sprite) *
                                            (color256 ? 2u : 1u);
            } else {
                this_tile = tile_num + (tile_y_in_sprite * 32u) +
                            tile_x_in_sprite * (color256 ? 2u : 1u);
            }
            uint32_t tile_off = obj_tile_base + this_tile * 32u;

            uint8_t pal_index;
            if (color256) {
                uint32_t off = tile_off + py_in_tile * 8 + px_in_tile;
                if (off + 1 > 96u * 1024u) return;
                pal_index = vram[off];
                if (pal_index == 0) return;  // transparent
            } else {
                uint32_t off = tile_off + py_in_tile * 4 + (px_in_tile / 2);
                if (off + 1 > 96u * 1024u) return;
                uint8_t b = vram[off];
                pal_index = (px_in_tile & 1) ? (b >> 4) : (b & 0x0F);
                if (pal_index == 0) return;  // transparent
                pal_index = static_cast<uint8_t>(pal_index | (palette_bank << 4));
            }

            uint16_t color = load_u16_le(&obj_pal[pal_index * 2]);
            uint8_t* dst = rgb + (screen_y * kScreenWidth + screen_x) * 3;
            to_rgb888(color, dst);
        };

        if (rot_scale) {
            // Affine sprite. Bounding box is 2x the sprite size when
            // the double-size flag is set, otherwise = sprite size.
            int bw = disable_or_double ? sw * 2 : sw;
            int bh = disable_or_double ? sh * 2 : sh;

            int affine_group = (attr1 >> 9) & 0x1Fu;
            const uint8_t* ag = oam + affine_group * 0x20u;
            // PA/PB/PC/PD live at offsets 0x06, 0x0E, 0x16, 0x1E within
            // the 32-byte affine block (the other bytes belong to OBJ
            // attr0/1/2 entries that share the same 8-byte slots).
            int32_t pa = read_s16(ag, 0x06);
            int32_t pb = read_s16(ag, 0x0E);
            int32_t pc = read_s16(ag, 0x16);
            int32_t pd = read_s16(ag, 0x1E);

            int half_bw = bw >> 1;
            int half_bh = bh >> 1;
            int half_sw = sw >> 1;
            int half_sh = sh >> 1;

            for (int j = 0; j < bh; ++j) {
                int screen_y = sy + j;
                if (screen_y < 0 || screen_y >= static_cast<int>(kScreenHeight)) continue;
                int dy = j - half_bh;
                for (int i = 0; i < bw; ++i) {
                    int screen_x = sx + i;
                    if (screen_x < 0 || screen_x >= static_cast<int>(kScreenWidth)) continue;
                    int dx = i - half_bw;

                    // (tex_x, tex_y) = matrix * (dx, dy) + sprite_center.
                    int tex_x = ((pa * dx + pb * dy) >> 8) + half_sw;
                    int tex_y = ((pc * dx + pd * dy) >> 8) + half_sh;
                    if (tex_x < 0 || tex_x >= sw) continue;
                    if (tex_y < 0 || tex_y >= sh) continue;

                    sample_and_emit(tex_x, tex_y, screen_x, screen_y);
                }
            }
            continue;
        }

        // Non-affine sprite from here on.
        bool hflip = (attr1 & 0x1000u) != 0;
        bool vflip = (attr1 & 0x2000u) != 0;

        for (int ty = 0; ty < tiles_h; ++ty) {
            for (int tx = 0; tx < tiles_w; ++tx) {
                // Compute the source tile index, taking flips into
                // account (flipping the tile *layout* on top of
                // per-pixel flip).
                int src_tx = hflip ? (tiles_w - 1 - tx) : tx;
                int src_ty = vflip ? (tiles_h - 1 - ty) : ty;

                uint32_t this_tile;
                if (obj_1d_mapping) {
                    this_tile = tile_num + (src_ty * tiles_w + src_tx) *
                                                (color256 ? 2 : 1);
                } else {
                    // 2D mapping: the OBJ tile area is always 32
                    // *slots* wide regardless of color depth. Each
                    // 4bpp tile occupies 1 slot; each 8bpp tile
                    // occupies 2 horizontally-adjacent slots.
                    // Row stride in slot units is always 32.
                    this_tile = tile_num + (src_ty * 32u) +
                                src_tx * (color256 ? 2u : 1u);
                }
                // OBJ tile numbers are in 32-byte *slot* units
                // regardless of color depth. An 8bpp visible tile
                // (64 bytes) occupies 2 consecutive slots.
                uint32_t tile_off = obj_tile_base + this_tile * 32u;

                // Per-pixel emit.
                for (int py = 0; py < 8; ++py) {
                    int screen_y = sy + ty * 8 + py;
                    if (screen_y < 0 || screen_y >= static_cast<int>(kScreenHeight)) continue;
                    int src_py = vflip ? (7 - py) : py;
                    for (int px = 0; px < 8; ++px) {
                        int screen_x = sx + tx * 8 + px;
                        if (screen_x < 0 || screen_x >= static_cast<int>(kScreenWidth)) continue;
                        int src_px = hflip ? (7 - px) : px;

                        uint8_t pal_index;
                        if (color256) {
                            // 1 byte per pixel.
                            uint32_t off = tile_off + src_py * 8 + src_px;
                            if (off + 1 > 96 * 1024) continue;
                            pal_index = vram[off];
                        } else {
                            // 4bpp: 4 bytes per row, low nibble = even
                            // pixel, high nibble = odd pixel.
                            uint32_t off = tile_off + src_py * 4 + (src_px / 2);
                            if (off + 1 > 96 * 1024) continue;
                            uint8_t b = vram[off];
                            pal_index = (src_px & 1) ? (b >> 4) : (b & 0x0F);
                            if (pal_index == 0) continue;  // transparent
                            pal_index |= (palette_bank << 4);
                        }
                        if (color256 && pal_index == 0) continue;

                        uint16_t color = load_u16_le(&obj_pal[pal_index * 2]);
                        uint8_t* dst = rgb +
                            (screen_y * kScreenWidth + screen_x) * 3;
                        to_rgb888(color, dst);
                    }
                }
            }
        }
    }
}

#endif

void GbaPpu::render_scanline(uint32_t y,
                             uint16_t dispcnt,
                             const uint8_t* io,
                             const uint8_t* vram,
                             const uint8_t* oam,
                             const uint8_t* pal) {
    if (high_resolution_) {
        high_resolution_->configure(render_width(), render_height());
        if (y == 0) high_resolution_->begin_frame();
    }
    if (presentation_) presentation_->scanline(*this, y, dispcnt, io, vram, oam, pal);
    if (layered_3d_) {
        if (y == 0) choose_obj_plane_prio(oam);
        update_layered_rank(dispcnt, io);
        for (int i = 0; i < 4; ++i) {
            s_split_active[i] = false;
            const int bg_prio = (io[0x08 + 2 * i] | (io[0x09 + 2 * i] << 8)) & 3;
            const bool behind_player = !g_layered_sprite_layer || s_obj_plane_prio < 1 || bg_prio >= s_obj_plane_prio;
            if (g_layered_bg_split_row && behind_player && !s_flat_scene && (dispcnt & 7u) == 0 &&
                (dispcnt & (0x100u << i))) {
                std::memset(s_split_row[i], 0, kScreenWidth);
                g_layered_bg_split_row(i, static_cast<int>(y), io, vram, s_split_row[i], static_cast<int>(kScreenWidth));
                s_split_active[i] = true;
            }
        }
        s_line_sprite_depth = g_layered_sprite_depth;
        if (g_layered_sprite_depth < 0.0f && s_obj_plane_prio >= 1) {
            const uint32_t mode = dispcnt & 7u;
            int best_plane = 0;
            for (int i = 0; i < 4; ++i) {
                const bool exists = mode == 0 || (mode == 1 && i < 3) || (mode == 2 && i >= 2);
                if (!exists || !(dispcnt & (0x100u << i))) continue;
                const int pr = (io[0x08 + 2 * i] | (io[0x09 + 2 * i] << 8)) & 3;
                if (pr < s_obj_plane_prio) continue;
                const int pl = s_rank.active ? s_rank.bg[i] : 4 - pr;
                best_plane = std::max(best_plane, pl);
            }
            s_line_sprite_depth = static_cast<float>(best_plane) + 0.5f;
        }
        static const bool rlog = std::getenv("GBARECOMP_RANK_LOG") != nullptr;
        if (rlog && y == 100)
            std::fprintf(stderr, "rank active=%d op=%d bg=%d,%d,%d,%d obj=%d,%d,%d,%d\n", s_rank.active, s_obj_plane_prio,
                         s_rank.bg[0], s_rank.bg[1], s_rank.bg[2], s_rank.bg[3], s_rank.obj[0], s_rank.obj[1], s_rank.obj[2], s_rank.obj[3]);
        if (y == 80) sample_parallax_line(dispcnt, io);
    }
    // BG2/BG3 affine coordinates are backed by hidden current-reference
    // registers. With constant PB/PD, ref + y*delta is equivalent. Mario Kart
    // HBlank-DMAs PA/PB/PC/PD/X/Y for every road scanline, however: each X/Y
    // write reloads the hidden reference and must be used directly rather than
    // multiplied by the absolute screen Y. Track the hardware accumulator and
    // rewrite a local IO image so the established compositor receives the
    // effective reference for this line through its ref + y*delta interface.
    std::array<uint8_t, 0x60> affine_io{};
    std::memcpy(affine_io.data(), io, affine_io.size());
    for (unsigned index = 0; index < affine_line_.size(); ++index) {
        AffineLineState& state = affine_line_[index];
        const uint32_t base = index == 0 ? 0x20u : 0x30u;
        const int32_t external_x = read_s28_ref(io, base + 0x08u);
        const int32_t external_y = read_s28_ref(io, base + 0x0Cu);
        if (y == 0 || !state.valid_x || state.reload_x) {
            state.x = external_x;
            state.valid_x = true;
        }
        if (y == 0 || !state.valid_y || state.reload_y) {
            state.y = external_y;
            state.valid_y = true;
        }
        state.reload_x = false;
        state.reload_y = false;

        const int32_t pb = read_s16(io, base + 0x02u);
        const int32_t pd = read_s16(io, base + 0x06u);
        write_s28_ref(affine_io.data(), base + 0x08u,
                      state.x - static_cast<int32_t>(y) * pb);
        write_s28_ref(affine_io.data(), base + 0x0Cu,
                      state.y - static_cast<int32_t>(y) * pd);
        state.x += pb;
        state.y += pd;
    }

    if (!rasterization_enabled_) return;
    const bool record = g_ppu_record_motion != 0 && y < kScreenHeight;
    if (record) {
        if (work_tags_.empty()) {
            work_tags_.assign(static_cast<std::size_t>(kMaxRenderWidth) * kMaxRenderHeight, 0xFFFFu);
            latched_tags_ = work_tags_;
            work_lines_.assign(kScreenHeight, MotionLine{});
            latched_lines_ = work_lines_;
        }
        MotionLine& line = work_lines_[y];
        line.dispcnt = dispcnt;
        for (unsigned bg = 0; bg < 4; ++bg) {
            line.bgcnt[bg] = static_cast<uint16_t>(io[0x08 + 2 * bg] | (io[0x09 + 2 * bg] << 8));
            line.hofs[bg] = static_cast<uint16_t>((io[0x10 + 4 * bg] | (io[0x11 + 4 * bg] << 8)) & 0x1FF);
            line.vofs[bg] = static_cast<uint16_t>((io[0x12 + 4 * bg] | (io[0x13 + 4 * bg] << 8)) & 0x1FF);
        }
        if (y == kScreenHeight - 1) std::memcpy(work_oam_.data(), oam, work_oam_.size());
    }
    // The enhanced surface consumes only published raster inputs. Its tile
    // stack and history never replace native pixels or advance device time.
    GbaHighResolutionFrame* hd = high_resolution_.get();
    if (y == 0 && g_ws_frame_start && (view_expanded() || layered_3d_)) g_ws_frame_start();
    const uint32_t ox = view_expanded() ? native_view_x() : 0, oy = view_expanded() ? native_view_y() : 0;
    const bool hd_line = hd && hd->begin_line(int(y), ox, oy,
        dispcnt, affine_io.data(), vram, pal,
        foreign_presentation_internal::background() ||
        foreign_presentation_internal::screen_overlay() ? nullptr : high_resolution_provider_,
        g_ws_affine_filter_enabled != 0);
    if (!view_expanded()) {
        if (record) g_tag_row = work_tags_.data() + static_cast<std::size_t>(y) * kScreenWidth;
        if (layered_3d_ && y == 0) {
            if (work_layered_fb_.size() != layered_framebuffer_bytes()) {
                work_layered_fb_.assign(layered_framebuffer_bytes(), 0);
                latched_layered_fb_.assign(layered_framebuffer_bytes(), 0);
            }
            ensure_optb_cap_size(work_optb_cap_, kScreenWidth, kScreenHeight);
        }
        render_scanline_internal(work_fb_.data(), y, dispcnt,
                                 affine_io.data(), vram, oam, pal,
                                 kScreenWidth, kScreenHeight,
                                 layered_3d_ ? work_layered_fb_.data() : nullptr,
                                 layered_3d_ ? &work_optb_cap_ : nullptr);
        g_tag_row = nullptr;  // motion tags cover the native line only
        if (hd_line) {
            // The expanded compositor may apply optional presentation filters.
            // Keep its native scratch entirely separate from canonical pixels.
            render_scanline_wide(hd->native_scratch(), int(y), dispcnt,
                affine_io.data(), vram, oam, pal, render_width(), 0, 0, hd);
        } else if (hd) hd->upscale_line(y, work_fb_.data() + y * render_width() * 3);
        return;
    }
    if (layered_3d_ && y == 0) {
        if (work_layered_fb_.size() != layered_framebuffer_bytes()) {
            work_layered_fb_.assign(layered_framebuffer_bytes(), 0);
            latched_layered_fb_.assign(layered_framebuffer_bytes(), 0);
            work_masked_fb_.assign(layered_framebuffer_bytes(), 0);
            latched_masked_fb_.assign(layered_framebuffer_bytes(), 0);
        }
        ensure_optb_cap_size(work_optb_cap_, render_width(), render_height());
    }
    // Host-authored extra rows use the published state of the native edge row
    // they continue: rows above at scanline zero, rows below after the last
    // scanline, so per-line scroll effects carry on past both edges. They never
    // run emulated scanline timing or affine updates.
    auto render_extra_rows = [&](int from, int to) {
        for (int sy = from; sy < to; ++sy) {
            render_scanline_wide(work_fb_.data(), sy, dispcnt, io, vram, oam, pal,
                                 render_width(), ox, oy, nullptr,
                                 layered_3d_ ? work_layered_fb_.data() : nullptr,
                                 render_height(),
                                 layered_3d_ ? &work_optb_cap_ : nullptr);
            if (hd) hd->upscale_line(unsigned(sy + int(oy)),
                work_fb_.data() + size_t(sy + int(oy)) * render_width() * 3);
        }
    };
    if (y == 0 && oy) render_extra_rows(-static_cast<int>(oy), 0);
    if (record)
        g_tag_row = work_tags_.data() + static_cast<std::size_t>(y + oy) * render_width();
    render_scanline_wide(work_fb_.data(), y, dispcnt, affine_io.data(),
                         vram, oam, pal, render_width(), ox, oy, hd_line ? hd : nullptr,
                         layered_3d_ ? work_layered_fb_.data() : nullptr,
                         render_height(),
                         layered_3d_ ? &work_optb_cap_ : nullptr);
    g_tag_row = nullptr;
    if (hd && !hd_line) hd->upscale_line(y + oy,
        work_fb_.data() + size_t(y + oy) * render_width() * 3);
    if (y == kScreenHeight - 1 && render_height() > kScreenHeight + oy)
        render_extra_rows(static_cast<int>(kScreenHeight), static_cast<int>(render_height() - oy));
}

void GbaPpu::note_affine_reference_write(unsigned bg, bool y_axis) {
    if (bg < 2 || bg > 3) return;
    AffineLineState& state = affine_line_[bg - 2];
    if (y_axis) state.reload_y = true;
    else state.reload_x = true;
}

void GbaPpu::render_presentation_scanline(const GbaPpu& source, uint32_t y,
                                          uint16_t dispcnt, const uint8_t* io,
                                          const uint8_t* vram, const uint8_t* oam,
                                          const uint8_t* pal) {
    affine_line_ = source.affine_line_;
    render_scanline(y, dispcnt, io, vram, oam, pal);
}

void GbaPpu::latch_framebuffer(uint16_t dispcnt,
                               const uint8_t* io,
                               const uint8_t* vram,
                               const uint8_t* oam,
                               const uint8_t* pal) {
    if (!layered_3d_) {
        render(work_fb_.data(), dispcnt, io, vram, oam, pal);
    } else {
        if (work_layered_fb_.size() != layered_framebuffer_bytes()) {
            work_layered_fb_.assign(layered_framebuffer_bytes(), 0);
            latched_layered_fb_.assign(layered_framebuffer_bytes(), 0);
            work_masked_fb_.assign(layered_framebuffer_bytes(), 0);
            latched_masked_fb_.assign(layered_framebuffer_bytes(), 0);
        }
        ensure_optb_cap_size(work_optb_cap_, render_width(), render_height());
        if (!view_expanded()) {
            for (uint32_t y = 0; y < kScreenHeight; ++y) {
                render_scanline_internal(work_fb_.data(), y, dispcnt, io, vram, oam, pal,
                                         kScreenWidth, kScreenHeight,
                                         work_layered_fb_.data(), &work_optb_cap_);
            }
        } else {
            const uint32_t ow = render_width();
            const uint32_t oh = render_height();
            const uint32_t ox = native_view_x();
            const int top = static_cast<int>(native_view_y());
            for (int y = -top; y < static_cast<int>(oh) - top; ++y) {
                render_scanline_wide(work_fb_.data(), y, dispcnt, io, vram, oam, pal,
                                     ow, ox, static_cast<uint32_t>(top), nullptr,
                                     work_layered_fb_.data(), oh, &work_optb_cap_);
            }
        }
    }
    mark_framebuffer_latched();
}

void GbaPpu::set_layered_3d(bool enabled) {
    if (layered_3d_ == enabled) return;
    layered_3d_ = enabled;
    g_layered_solid_provider_wanted = enabled ? 1 : 0;
    has_latched_layered_fb_ = false;
    has_latched_masked_fb_ = false;
    work_masked_scene_ = false;
    latched_masked_scene_ = false;
    if (enabled) {
        work_layered_fb_.assign(layered_framebuffer_bytes(), 0);
        latched_layered_fb_.assign(layered_framebuffer_bytes(), 0);
        work_masked_fb_.assign(layered_framebuffer_bytes(), 0);
        latched_masked_fb_.assign(layered_framebuffer_bytes(), 0);
        ensure_optb_cap_size(work_optb_cap_, render_width(), render_height());
    } else {
        std::vector<uint8_t>().swap(work_layered_fb_);
        std::vector<uint8_t>().swap(latched_layered_fb_);
        std::vector<uint8_t>().swap(work_masked_fb_);
        std::vector<uint8_t>().swap(latched_masked_fb_);
        std::vector<OptionBCap::Pixel>().swap(work_optb_cap_.pixels);
        std::vector<OptionBCap::Line>().swap(work_optb_cap_.lines);
        work_optb_cap_.width = work_optb_cap_.height = 0;
    }
}

void GbaPpu::mark_framebuffer_latched() {
    if (rasterization_enabled_) {
        std::memcpy(latched_fb_.data(), work_fb_.data(), render_bytes());
        const size_t layered_bytes = layered_framebuffer_bytes();
        if (layered_3d_ && work_layered_fb_.size() == layered_bytes) {
            latched_layered_fb_.resize(layered_bytes);
            std::memcpy(latched_layered_fb_.data(), work_layered_fb_.data(),
                        layered_bytes);
            unify_world_objects(latched_layered_fb_.data(), render_width(), render_height());
            keep_sprites_whole(latched_layered_fb_.data(), render_width(), render_height());
            latch_sprite_layer(latched_layered_fb_.data(), render_width(), render_height());
            s_split_target_frame = s_split_target_now;
            s_split_target_now = -1;
            update_parallax_depth(1.65f);
            has_latched_layered_fb_ = true;
            if (work_optb_cap_.width == render_width() &&
                work_optb_cap_.height == render_height()) {
                const uint32_t ox = view_expanded() ? native_view_x() : 0u;
                const uint32_t oy = view_expanded() ? native_view_y() : 0u;
                build_masked_5plane_frame(work_optb_cap_, work_fb_.data(),
                                          work_masked_fb_, work_masked_scene_, ox, oy);
                latched_masked_fb_.swap(work_masked_fb_);
                latched_masked_scene_ = work_masked_scene_;
                has_latched_masked_fb_ = true;
            }
        }
    }
    if (rasterization_enabled_ && !work_tags_.empty()) {
        std::copy_n(work_tags_.data(), static_cast<std::size_t>(render_width()) * render_height(),
                    latched_tags_.data());
        latched_lines_ = work_lines_;
        latched_oam_ = work_oam_;
    }
    has_latched_fb_ = true;
    if (high_resolution_) high_resolution_->latch();
    if (presentation_) presentation_->frame_ready();
}

void GbaPpu::set_rasterization_enabled(bool enabled) {
    rasterization_enabled_ = enabled;
    if (!enabled) {
        std::fill(work_fb_.begin(),work_fb_.end(),0);
        std::fill(latched_fb_.begin(),latched_fb_.end(),0);
    }
}

}  // namespace gba

namespace gba {
void GbaPpu::set_internal_resolution(unsigned scale, HighResolutionAffineProvider provider) {
    if (scale < 1 || scale > GbaHighResolutionFrame::kMaxScale)
        throw std::invalid_argument("internal resolution must be from 1 through 16");
    high_resolution_provider_ = provider;
    if (scale == internal_resolution()) return;
    if (scale == 1) high_resolution_.reset();
    else {
        high_resolution_ = std::make_unique<GbaHighResolutionFrame>(scale);
        high_resolution_->configure(render_width(), render_height());
    }
}
} // namespace gba

namespace gba {
const uint8_t* latched_sprite_layer() {
    return (g_layered_sprite_layer && s_sprite_has_latched) ? s_sprite_latched.data() : nullptr;
}
}  // namespace gba
