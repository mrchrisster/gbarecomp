// host_window_quest.cpp — Meta Quest OpenXR HostWindow backend.
//
// Renders the GBA framebuffer onto a crisp 4x integer-upscaled OpenXR 5-layer
// quad compositor stack (Backdrop + Priorities 3..0, supporting both 4:3 and
// Mother 3's 16:9 Widescreen, with optional XR_FB_passthrough mixed reality +
// a toggleable control-legend HUD quad below the screen) and translates Quest
// Touch Plus buttons, thumbsticks, and dual-hand haptics into Mother 3 inputs.

#if defined(__ANDROID__) && defined(GBARECOMP_QUEST_XR)

#include "host_window.h"

#include "color_lut.h"
#include "gba_ppu.h"
#include "stock3d_fix.h"
#include "gba_bus.h"
#include "runtime_bus_bridge.h"
#include "hd_text_bg.h"
#include "quest_popup_renderer.h"
#include "quest_window_renderer.h"

#define RECOMP_AUDIO_DRC_IMPL
#include "recomp_audio_drc.h"

#define XR_USE_GRAPHICS_API_OPENGL_ES
#define XR_USE_PLATFORM_ANDROID
#include <jni.h>
#include <android/log.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>

#include <SDL.h>
#include <SDL_system.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <sys/stat.h>

namespace gbarecomp {

namespace {

constexpr const char* kLogTag = "Mother3XR";

void xr_log(int prio, const char* fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    __android_log_print(prio, kLogTag, "%s", buf);
    std::fprintf(stderr, "[%s] %s\n", kLogTag, buf);
    std::fflush(stderr);
}

#define XR_LOGI(...) xr_log(ANDROID_LOG_INFO, __VA_ARGS__)
#define XR_LOGW(...) xr_log(ANDROID_LOG_WARN, __VA_ARGS__)
#define XR_LOGE(...) xr_log(ANDROID_LOG_ERROR, __VA_ARGS__)

constexpr unsigned kLayeredPlaneCount = 5;
constexpr unsigned kMaskedQuadCount = 7; // P0 Void, P1 Ground, P2 Props, P3a/b/c Actors, P4 Overhang/HUD
constexpr int kHudWidth = 1296;
constexpr int kHudHeight = 126;
constexpr int kMenuWidth = 960;   // VR options menu panel (GBARECOMP_XR_MENU=1)
constexpr int kMenuHeight = 720;

// Keep integer texel scaling near ~1200-1280px wide (~native Quest 3 Angular Resolution across ~46 deg FOV)
// so 5 layered OpenXR swapchains stay razor-sharp without 2.25x 1080p TimeWarp fillrate overhead.
constexpr uint32_t target_swapchain_scale(int base_w) {
    return (base_w > 240) ? 4u : 5u;
}
constexpr float kThumbstickDeadzone = 0.32f;
constexpr float kTriggerThreshold = 0.45f;
constexpr float kGripThreshold = 0.55f;
// Cartridge gyro from Touch controller wrist twist (opt-in GBARECOMP_XR_GYRO=1, WarioWare: Twisted!).
constexpr float kGyroDriftDeadzoneRadPerSec = 0.025f;
// 1.00X in the menu = a quarter of the v0.1 response (players found v0.1 far too twitchy for the
// precise microgames; Christoph found a quarter right). 4.00X = the v0.1 response.
constexpr float kQuestGyroRateScale = 0.5f;
constexpr float kGyroSensSteps[] = {0.25f, 0.5f, 0.75f, 1.0f, 1.25f, 1.5f, 2.0f, 3.0f, 4.0f};
constexpr int kGyroSensCount = static_cast<int>(sizeof(kGyroSensSteps) / sizeof(kGyroSensSteps[0]));
constexpr int kGyroSensDefault = 3;

struct Quat {
    float x = 0.0f, y = 0.0f, z = 0.0f, w = 1.0f;
};

XrVector3f rotate_vec(const XrQuaternionf& q, float vx, float vy, float vz) {
    // v' = v + w*t + u x t, where t = 2*(u x v)
    const float tx = 2.0f * (q.y * vz - q.z * vy);
    const float ty = 2.0f * (q.z * vx - q.x * vz);
    const float tz = 2.0f * (q.x * vy - q.y * vx);
    return {
        vx + q.w * tx + (q.y * tz - q.z * ty),
        vy + q.w * ty + (q.z * tx - q.x * tz),
        vz + q.w * tz + (q.x * ty - q.y * tx),
    };
}

XrVector3f v3_add(const XrVector3f& a, const XrVector3f& b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

XrVector3f v3_sub(const XrVector3f& a, const XrVector3f& b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

XrVector3f v3_scale(const XrVector3f& v, float s) {
    return {v.x * s, v.y * s, v.z * s};
}

float v3_dot(const XrVector3f& a, const XrVector3f& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

XrVector3f v3_cross(const XrVector3f& a, const XrVector3f& b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}

float v3_len(const XrVector3f& v) {
    return std::sqrt(v3_dot(v, v));
}

XrVector3f v3_norm(const XrVector3f& v, const XrVector3f& fallback = {0.0f, 0.0f, -1.0f}) {
    const float l = v3_len(v);
    return (l > 1e-5f) ? v3_scale(v, 1.0f / l) : fallback;
}

XrQuaternionf quat_normalize(const XrQuaternionf& q) {
    const float n = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    if (n < 1e-6f) return {0.0f, 0.0f, 0.0f, 1.0f};
    return {q.x / n, q.y / n, q.z / n, q.w / n};
}

XrQuaternionf quat_from_two_unit_vecs(const XrVector3f& u, const XrVector3f& v) {
    const float d = v3_dot(u, v);
    if (d >= 0.99999f) return {0.0f, 0.0f, 0.0f, 1.0f};
    if (d <= -0.99999f) {
        XrVector3f ortho = v3_cross({1.0f, 0.0f, 0.0f}, u);
        if (v3_len(ortho) < 1e-3f) ortho = v3_cross({0.0f, 1.0f, 0.0f}, u);
        ortho = v3_norm(ortho, {0.0f, 1.0f, 0.0f});
        return {ortho.x, ortho.y, ortho.z, 0.0f};
    }
    const XrVector3f c = v3_cross(u, v);
    return quat_normalize({c.x, c.y, c.z, 1.0f + d});
}

// Convert orthonormal right (X), up (Y), forward-normal (Z) basis columns into a unit quaternion.
XrQuaternionf quat_from_basis(const XrVector3f& x_col, const XrVector3f& y_col, const XrVector3f& z_col) {
    const float m00 = x_col.x, m01 = y_col.x, m02 = z_col.x;
    const float m10 = x_col.y, m11 = y_col.y, m12 = z_col.y;
    const float m20 = x_col.z, m21 = y_col.z, m22 = z_col.z;
    const float tr = m00 + m11 + m22;
    XrQuaternionf q{};
    if (tr > 0.0f) {
        const float s = std::sqrt(tr + 1.0f) * 2.0f;
        q.w = 0.25f * s;
        q.x = (m21 - m12) / s;
        q.y = (m02 - m20) / s;
        q.z = (m10 - m01) / s;
    } else if (m00 > m11 && m00 > m22) {
        const float s = std::sqrt(1.0f + m00 - m11 - m22) * 2.0f;
        q.w = (m21 - m12) / s;
        q.x = 0.25f * s;
        q.y = (m01 + m10) / s;
        q.z = (m02 + m20) / s;
    } else if (m11 > m22) {
        const float s = std::sqrt(1.0f + m11 - m00 - m22) * 2.0f;
        q.w = (m02 - m20) / s;
        q.x = (m01 + m10) / s;
        q.y = 0.25f * s;
        q.z = (m12 + m21) / s;
    } else {
        const float s = std::sqrt(1.0f + m22 - m00 - m11) * 2.0f;
        q.w = (m10 - m01) / s;
        q.x = (m02 + m20) / s;
        q.y = (m12 + m21) / s;
        q.z = 0.25f * s;
    }
    return quat_normalize(q);
}

// Build a screen orientation facing the viewer (-to_screen_dir) while blending between
// world-up (when sitting/standing upright) and head/custom up (when laying down).
XrQuaternionf make_screen_orientation(const XrVector3f& to_screen_dir,
                                      const XrVector3f& ref_up_hint,
                                      const XrVector3f& fallback_right) {
    const XrVector3f z_col = v3_norm(v3_scale(to_screen_dir, -1.0f), {0.0f, 0.0f, 1.0f});
    XrVector3f x_col = v3_cross(ref_up_hint, z_col);
    if (v3_len(x_col) < 1e-4f) {
        x_col = v3_sub(fallback_right, v3_scale(z_col, v3_dot(fallback_right, z_col)));
    }
    x_col = v3_norm(x_col, {1.0f, 0.0f, 0.0f});
    const XrVector3f y_col = v3_norm(v3_cross(z_col, x_col), {0.0f, 1.0f, 0.0f});
    return quat_from_basis(x_col, y_col, z_col);
}

// Compact 5x7 uppercase/ASCII bitmap glyphs (A-Z, 0-9, punctuation) for the
// zero-dependency VR control-legend HUD quad.
struct Glyph5x7 {
    char ch;
    uint8_t col[5]; // LSB = top row, bit 6 = bottom row
};

constexpr Glyph5x7 kFont5x7[] = {
    {' ', {0x00, 0x00, 0x00, 0x00, 0x00}},
    {':', {0x00, 0x36, 0x36, 0x00, 0x00}},
    {'/', {0x20, 0x10, 0x08, 0x04, 0x02}},
    {'-', {0x08, 0x08, 0x08, 0x08, 0x08}},
    {'+', {0x08, 0x08, 0x3E, 0x08, 0x08}},
    {'.', {0x00, 0x60, 0x60, 0x00, 0x00}},
    {'!', {0x00, 0x00, 0x5F, 0x00, 0x00}},
    {'|', {0x00, 0x00, 0x7F, 0x00, 0x00}},
    {'=', {0x14, 0x14, 0x14, 0x14, 0x14}},
    {'(', {0x00, 0x1C, 0x22, 0x41, 0x00}},
    {')', {0x00, 0x41, 0x22, 0x1C, 0x00}},
    {'0', {0x3E, 0x51, 0x49, 0x45, 0x3E}},
    {'1', {0x00, 0x42, 0x7F, 0x40, 0x00}},
    {'2', {0x42, 0x61, 0x51, 0x49, 0x46}},
    {'3', {0x21, 0x41, 0x45, 0x4B, 0x31}},
    {'4', {0x18, 0x14, 0x12, 0x7F, 0x10}},
    {'5', {0x27, 0x45, 0x45, 0x45, 0x39}},
    {'6', {0x3C, 0x4A, 0x49, 0x49, 0x30}},
    {'7', {0x01, 0x71, 0x09, 0x05, 0x03}},
    {'8', {0x36, 0x49, 0x49, 0x49, 0x36}},
    {'9', {0x06, 0x49, 0x49, 0x29, 0x1E}},
    {'A', {0x7E, 0x11, 0x11, 0x11, 0x7E}},
    {'B', {0x7F, 0x49, 0x49, 0x49, 0x36}},
    {'C', {0x3E, 0x41, 0x41, 0x41, 0x22}},
    {'D', {0x7F, 0x41, 0x41, 0x22, 0x1C}},
    {'E', {0x7F, 0x49, 0x49, 0x49, 0x41}},
    {'F', {0x7F, 0x09, 0x09, 0x09, 0x01}},
    {'G', {0x3E, 0x41, 0x49, 0x49, 0x7A}},
    {'H', {0x7F, 0x08, 0x08, 0x08, 0x7F}},
    {'I', {0x00, 0x41, 0x7F, 0x41, 0x00}},
    {'J', {0x20, 0x40, 0x41, 0x3F, 0x01}},
    {'K', {0x7F, 0x08, 0x14, 0x22, 0x41}},
    {'L', {0x7F, 0x40, 0x40, 0x40, 0x40}},
    {'M', {0x7F, 0x02, 0x0C, 0x02, 0x7F}},
    {'N', {0x7F, 0x04, 0x08, 0x10, 0x7F}},
    {'O', {0x3E, 0x41, 0x41, 0x41, 0x3E}},
    {'P', {0x7F, 0x09, 0x09, 0x09, 0x06}},
    {'Q', {0x3E, 0x41, 0x51, 0x21, 0x5E}},
    {'R', {0x7F, 0x09, 0x19, 0x29, 0x46}},
    {'S', {0x46, 0x49, 0x49, 0x49, 0x31}},
    {'T', {0x01, 0x01, 0x7F, 0x01, 0x01}},
    {'U', {0x3F, 0x40, 0x40, 0x40, 0x3F}},
    {'V', {0x1F, 0x20, 0x40, 0x20, 0x1F}},
    {'W', {0x3F, 0x40, 0x38, 0x40, 0x3F}},
    {'X', {0x63, 0x14, 0x08, 0x14, 0x63}},
    {'Y', {0x07, 0x08, 0x70, 0x08, 0x07}},
    {'Z', {0x61, 0x51, 0x49, 0x45, 0x43}},
};

const uint8_t* find_glyph5x7(char c) {
    if (c >= 'a' && c <= 'z') c = static_cast<char>(c - 'a' + 'A');
    for (const auto& g : kFont5x7) {
        if (g.ch == c) return g.col;
    }
    return kFont5x7[0].col;
}

void draw_hud_text(std::vector<uint8_t>& rgba, int w, int h, int x0, int y0,
                   int scale, uint32_t color_rgba, const char* text) {
    const uint8_t r = static_cast<uint8_t>((color_rgba >> 24) & 0xFFu);
    const uint8_t g = static_cast<uint8_t>((color_rgba >> 16) & 0xFFu);
    const uint8_t b = static_cast<uint8_t>((color_rgba >> 8) & 0xFFu);
    const uint8_t a = static_cast<uint8_t>(color_rgba & 0xFFu);
    int cx = x0;
    for (const char* p = text; *p; ++p) {
        const uint8_t* cols = find_glyph5x7(*p);
        for (int gx = 0; gx < 5; ++gx) {
            const uint8_t bits = cols[gx];
            for (int gy = 0; gy < 7; ++gy) {
                if (!(bits & (1u << gy))) continue;
                for (int sy = 0; sy < scale; ++sy) {
                    for (int sx = 0; sx < scale; ++sx) {
                        const int px = cx + gx * scale + sx;
                        // Top-down coordinate converted to OpenGL bottom-up row.
                        const int py = (h - 1) - (y0 + gy * scale + sy);
                        if (px < 0 || px >= w || py < 0 || py >= h) continue;
                        const size_t idx = (static_cast<size_t>(py) * w + px) * 4;
                        rgba[idx + 0] = r;
                        rgba[idx + 1] = g;
                        rgba[idx + 2] = b;
                        rgba[idx + 3] = a;
                    }
                }
            }
        }
        cx += 6 * scale;
    }
}

uint8_t premul_channel(uint8_t c, uint8_t a, bool srgb) {
    if (a == 255) return c;
    if (a == 0) return 0;
    if (!srgb) {
        return static_cast<uint8_t>((static_cast<uint32_t>(c) * a + 127u) / 255u);
    }
    static const std::array<uint8_t, 65536> kSrgbPremul = []() {
        std::array<uint8_t, 65536> tbl{};
        auto srgb_to_lin = [](double u) -> double {
            return (u <= 0.04045) ? (u / 12.92) : std::pow((u + 0.055) / 1.055, 2.4);
        };
        auto lin_to_srgb = [](double l) -> double {
            return (l <= 0.0031308) ? (12.92 * l) : (1.055 * std::pow(l, 1.0 / 2.4) - 0.055);
        };
        for (int av = 0; av < 256; ++av) {
            const double af = static_cast<double>(av) / 255.0;
            for (int cv = 0; cv < 256; ++cv) {
                const double lin = srgb_to_lin(static_cast<double>(cv) / 255.0) * af;
                const int enc = static_cast<int>(std::lround(std::clamp(lin_to_srgb(lin), 0.0, 1.0) * 255.0));
                tbl[static_cast<size_t>(av) * 256u + static_cast<size_t>(cv)] =
                    static_cast<uint8_t>(std::clamp(enc, 0, 255));
            }
        }
        return tbl;
    }();
    return kSrgbPremul[static_cast<size_t>(a) * 256u + static_cast<size_t>(c)];
}

struct Backend {
    // GBA logical framebuffer size
    int base_w = 240;
    int base_h = 160;
    uint32_t swap_w = 960;
    uint32_t swap_h = 640;
    std::atomic<int> requested_base_w{240};
    std::atomic<int> requested_base_h{160};
    std::atomic<uint32_t> reported_swap_w{960};
    std::atomic<uint32_t> reported_swap_h{640};
    std::string title = "MOTHER 3 VR";

    // Color LUT + CPU conversion staging buffers (owned by XR thread)
    std::unique_ptr<runtime::ColorLut> color_lut;
    std::vector<uint8_t> graded_fb;
    std::vector<uint8_t> color_lut_rgb;
    std::vector<uint8_t> rgba_composite_fb;
    std::vector<uint8_t> rgba_fb;

    // Thread-safe frame handoff from GBA thread -> dedicated OpenXR compositor thread
    std::mutex frame_mtx;
    std::vector<uint8_t> staging_pixels;
    std::vector<uint8_t> staging_masked_pixels;
    unsigned staging_w = 0;
    unsigned staging_h = 0;
    unsigned staging_planes = 0;
    bool staging_is_rgba = false;
    bool staging_has_masked = false;
    bool staging_masked_scene = false;
    uint64_t pending_seq = 0;

    // XR-thread local copy of latest staged frame
    std::vector<uint8_t> xr_pixels;
    std::vector<uint8_t> xr_masked_pixels;
    gbarecomp::Stock3dFix fix;  // STOCK 3D building fix (shared with the desktop layer view)
    unsigned xr_w = 0;
    unsigned xr_h = 0;
    unsigned xr_planes = 0;
    bool xr_is_rgba = false;
    bool xr_has_masked = false;
    bool xr_masked_scene = false;
    uint64_t uploaded_seq = 0;
    bool force_reblit = false;

    // Battle background HD redraw (hd_text_bg.h): data staged with each frame,
    // rendered at swapchain resolution on the XR thread.
    gbarecomp::hdbg::Frame staging_hdbg, xr_hdbg;
    bool staging_hdbg_ok = false, xr_hdbg_ok = false;
    gbarecomp::hdbg::Renderer hd_renderer;
    std::vector<uint8_t> hd_fb;
    std::vector<uint32_t> hd_lut555;
    bool hd_lut_ready = false;
    GLuint hd_tex = 0, hd_fbo = 0;
    uint32_t hd_tex_w = 0, hd_tex_h = 0;

    // EGL state (owned by XR thread)
    EGLDisplay egl_display = EGL_NO_DISPLAY;
    EGLConfig egl_config = nullptr;
    EGLContext egl_context = EGL_NO_CONTEXT;
    EGLSurface egl_surface = EGL_NO_SURFACE;
    GLuint src_tex = 0;
    GLuint src_fbo = 0;
    GLuint dst_fbo = 0;

    // OpenXR core objects & thread lifecycle
    JavaVM* vm = nullptr;
    jobject activity_global_ref = nullptr;
    std::mutex xr_init_mtx;
    std::condition_variable xr_init_cv;
    bool xr_init_done = false;
    bool xr_init_ok = false;
    std::atomic<bool> stop_xr_thread{false};
    std::thread xr_thread;

    XrInstance instance = XR_NULL_HANDLE;
    XrSystemId system_id = XR_NULL_SYSTEM_ID;
    XrSession session = XR_NULL_HANDLE;
    XrSpace local_space = XR_NULL_HANDLE;
    XrSpace view_space = XR_NULL_HANDLE;
    XrSessionState session_state = XR_SESSION_STATE_UNKNOWN;
    std::atomic<bool> session_running{false};
    std::atomic<bool> exit_requested{false};
    std::atomic<bool> background_pending{false};
    std::atomic<uint16_t> latest_keyinput{0x03FFu};
    // Cartridge gyro (GBARECOMP_XR_GYRO=1): angular rate of the active controller around the
    // forearm axis, handed to the runtime as ev.gyro_rate_z. Hand + sensitivity are menu items.
    bool gyro_enabled = false;
    int active_gyro_hand = 1;  // 1 = right (default), 0 = left
    int gyro_sens_idx = kGyroSensDefault;
    bool gyro_motion_logged = false;
    std::atomic<float> latest_gyro_rate_z{0.0f};

    // Layered GBA swapchains & images (0..4 in STOCK 3D; 0..6 with P3a/b/c actor bands in MASKED 3D)
    XrSwapchain screen_swapchains[kMaskedQuadCount] = {};
    std::vector<XrSwapchainImageOpenGLESKHR> screen_images[kMaskedQuadCount];
    bool active_planes[kMaskedQuadCount] = {true, false, false, false, false, false, false};
    std::vector<uint8_t> p3_bands[3];
    bool p3_band_any[3] = {false, false, false};
    std::vector<uint8_t> p3_vis;
    std::vector<int> p3_bfs_q;
    std::vector<int> p3_comp;
    bool last_frame_layered = false;
    int depth_mode = 1; // 0 = 2D FLAT, 1 = STOCK 3D (2 = POP-UP 3D experiment, disabled Nov 15 2026)
    bool stock_deep = false;
    bool perf_settings_ext = false;  // XR_EXT_performance_settings enabled (GBARECOMP_XR_CPU_HIGH)
    // VR options menu (GBARECOMP_XR_MENU=1): Menu button opens it and pauses the game.
    bool menu_enabled = false;
    bool menu_open = false;
    int menu_sel = 0;
    int menu_slot = 1;
    int menu_prev_dir_y = 0, menu_prev_dir_x = 0;
    bool menu_prev_a = false, menu_prev_b = false;
    bool menu_dirty = true;
    std::string menu_status;
    std::string hud_title = "MOTHER 3 VR";
    XrSwapchain menu_swapchain = XR_NULL_HANDLE;
    std::vector<XrSwapchainImageOpenGLESKHR> menu_images;
    std::atomic<int> pending_save_slot{0};
    std::atomic<int> pending_load_slot{0};
    std::atomic<int> pending_pause_toggles{0};
    // Shadow box (GBARECOMP_XR_SHADOWBOX=<GBA px>): planes get this many pixels of edge-extended
    // margin on every side and a dark frame sits in front with its opening at the screen edge,
    // so leaning never shows the layers' borders.
    int pad = 0;
    XrSwapchain frame_swapchain = XR_NULL_HANDLE;
    std::vector<XrSwapchainImageOpenGLESKHR> frame_images;
    bool frame_uploaded = false;
    bool box_walls = false;  // GBARECOMP_XR_BOXWALLS=1
    // GBARECOMP_XR_WINDOW=1: far planes (0,1) drawn per eye, clipped to the window opening.
    bool window_clip = false;
    bool start_dark = false;
    // GBARECOMP_XR_MARGIN_RENDER=1: the runtime renders `pad` real columns beyond each side of the
    // 240 px screen (wider view); they fill the shadow-box margins instead of stretched edge pixels.
    int wide_margin = 0;
    std::vector<uint8_t> staging_margin, xr_margin;  // [plane][row][2*wide_margin] RGBA, raw
    std::vector<uint8_t> wide_plane;
    bool auto_resume = true;  // menu AUTO RESUME: launch loads the newest state (runtime reads GBARECOMP_AUTO_RESUME)  // GBARECOMP_XR_DARK=1: start with the room (passthrough) off
    bool window_failed = false;
    WindowRenderer winr;
    std::vector<uint8_t> pad_fb;
    std::chrono::steady_clock::time_point r_click_down_at{};
    std::atomic<bool> capture_requested{false}; // STOCK 3D layer spacing: false = 6 cm, true = 11 cm

    // POP-UP 3D: tilted floor + grounded screen-facing cards, rendered per eye into a
    // projection layer (see quest_popup_layout.h / quest_popup_renderer.h).
    popup::Renderer popup;
    bool popup_init_failed = false;
    float popup_tilt_deg = 15.0f;
    XrSwapchain proj_swapchains[2] = {XR_NULL_HANDLE, XR_NULL_HANDLE};
    std::vector<XrSwapchainImageOpenGLESKHR> proj_images[2];
    uint32_t proj_w = 0;
    uint32_t proj_h = 0;
    bool popup_front_active = false;
    bool masked_scene_active = false;
    bool has_submitted_frame = false;
    XrTime last_predicted_time = 0;

    // HUD swapchain & state
    XrSwapchain hud_swapchain = XR_NULL_HANDLE;
    std::vector<XrSwapchainImageOpenGLESKHR> hud_images;
    bool hud_dirty = true;
    int hud_mode = 0; // 0 = MR/Void + HUD, 1 = MR/Void without HUD, 2 = Dark Void only

    // Optional XR_FB_passthrough
    bool passthrough_ext = false;
    bool layer_settings_ext = false;
    XrCompositionLayerSettingsFB popup_layer_settings{XR_TYPE_COMPOSITION_LAYER_SETTINGS_FB};
    bool passthrough_supported = false;
    bool passthrough_active = false;
    XrPassthroughFB passthrough = XR_NULL_HANDLE;
    XrPassthroughLayerFB passthrough_layer = XR_NULL_HANDLE;
    PFN_xrCreatePassthroughFB pfnCreatePassthroughFB = nullptr;
    PFN_xrDestroyPassthroughFB pfnDestroyPassthroughFB = nullptr;
    PFN_xrPassthroughStartFB pfnPassthroughStartFB = nullptr;
    PFN_xrPassthroughPauseFB pfnPassthroughPauseFB = nullptr;
    PFN_xrCreatePassthroughLayerFB pfnCreatePassthroughLayerFB = nullptr;
    PFN_xrDestroyPassthroughLayerFB pfnDestroyPassthroughLayerFB = nullptr;
    PFN_xrPassthroughLayerPauseFB pfnPassthroughLayerPauseFB = nullptr;
    PFN_xrPassthroughLayerResumeFB pfnPassthroughLayerResumeFB = nullptr;

    // Optional XR_FB_display_refresh_rate
    PFN_xrRequestDisplayRefreshRateFB pfnRequestDisplayRefreshRateFB = nullptr;

    // Virtual screen placement in LOCAL space
    XrPosef screen_pose = {{0.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, -1.65f}};
    bool pose_initialized = false;
    XrTime recenter_after_time = 0;
    int recenter_frames_remaining = 6;
    float screen_distance = 1.65f;
    std::atomic<float> screen_width_m{1.42f};
    bool swapchain_srgb = false;

    // Two-handed grip window grab & move state
    bool grip_move_active = false;
    XrVector3f grip_start_grab_dir = {0.0f, 0.0f, -1.0f};
    float grip_start_hand_dist = 0.40f;
    XrVector3f grip_start_screen_offset = {0.0f, 0.0f, -1.65f};
    float grip_start_screen_dist = 1.65f;
    XrVector3f grip_start_screen_up = {0.0f, 1.0f, 0.0f};
    XrVector3f grip_start_screen_right = {1.0f, 0.0f, 0.0f};
    XrVector3f grip_start_hand_right = {1.0f, 0.0f, 0.0f};

    // OpenXR Actions (Touch Plus controls)
    XrActionSet action_set = XR_NULL_HANDLE;
    XrPath hand_paths[2] = {XR_NULL_PATH, XR_NULL_PATH}; // 0=left, 1=right
    XrAction act_aim_pose = XR_NULL_HANDLE;
    XrSpace aim_spaces[2] = {XR_NULL_HANDLE, XR_NULL_HANDLE};
    XrAction act_thumbstick = XR_NULL_HANDLE;
    XrAction act_trigger = XR_NULL_HANDLE;
    XrAction act_squeeze = XR_NULL_HANDLE;
    XrAction act_a = XR_NULL_HANDLE;
    XrAction act_b = XR_NULL_HANDLE;
    XrAction act_x = XR_NULL_HANDLE;
    XrAction act_y = XR_NULL_HANDLE;
    XrAction act_thumb_click = XR_NULL_HANDLE;
    XrAction act_menu = XR_NULL_HANDLE;
    XrAction act_haptic = XR_NULL_HANDLE;
    bool actions_attached = false;

    bool prev_r_stick_click = false;
    bool prev_l_stick_click = false;
    bool prev_both_stick_clicks = false;
    bool stick_chord_consumed = false;
    bool prev_l_menu_click = false;

    // Cartridge rumble & one-shot haptic handoff (GBA thread -> XR thread)
    std::mutex haptic_mtx;
    bool haptic_pulse_queued = false;
    XrDuration queued_haptic_ns = 0;
    float queued_haptic_amp = 0.0f;
    bool desired_rumble = false;
    bool rumble_active = false;
    std::chrono::steady_clock::time_point last_rumble_refresh{};

    // SDL audio callback + resampling audio bridge (Blackman-sinc polyphase + fill servo)
    SDL_AudioDeviceID audio_dev = 0;
    SDL_mutex* audio_mtx = nullptr;
    rab_bridge bridge{};
    bool bridge_ready = false;
    bool sdl_audio_init = false;
    bool audio_enabled = true;
    int volume = 100;
    std::vector<int16_t> volume_buf;

    // Audio probe telemetry (diagnosed under audio_mtx or main thread)
    uint64_t audio_push_count = 0;
    uint64_t audio_push_samples = 0;
    std::chrono::steady_clock::time_point last_push_tp{};
    double max_push_gap_ms = 0.0;
    uint64_t audio_cb_count = 0;
    int max_cb_frames = 0;
    std::chrono::steady_clock::time_point last_cb_tp{};
    double max_cb_gap_ms = 0.0;
    int16_t prev_in_sample = 0;
    int16_t prev_out_sample = 0;
    int32_t in_boundary_max_jump = 0;
    int32_t out_boundary_max_jump = 0;
    int32_t in_max_jump = 0;
    int32_t out_max_jump = 0;
    bool prev_concealing = false;
    uint64_t conceal_exit_jumps = 0;
    int32_t conceal_exit_max_jump = 0;

    // Host flags queried by runtime.cpp
    int scale = 3;
    std::atomic<bool> linear_filter{false};
    bool fps_readout = false;
    bool assist_tools = false;
    int fast_forward_mult = 4;
};

HostWindow* g_quest_window = nullptr;

Backend* as_backend(void* impl) { return static_cast<Backend*>(impl); }
const Backend* as_backend(const void* impl) { return static_cast<const Backend*>(impl); }

runtime::ColorSettings parse_color_settings(const char* screen_arg) {
    runtime::ColorSettings cs;
    const char* env = std::getenv("GBARECOMP_SCREEN");
    const char* chosen = (env && *env) ? env : screen_arg;
    if (chosen && *chosen) {
        runtime::screen_kind_from_name(chosen, cs.screen);
    }
    return cs;
}

bool init_egl(Backend& b) {
    b.egl_display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (b.egl_display == EGL_NO_DISPLAY) {
        XR_LOGE("eglGetDisplay failed");
        return false;
    }
    EGLint major = 0, minor = 0;
    if (!eglInitialize(b.egl_display, &major, &minor)) {
        XR_LOGE("eglInitialize failed");
        return false;
    }
    const EGLint cfg_attrs[] = {
        EGL_RED_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_BLUE_SIZE, 8,
        EGL_ALPHA_SIZE, 8,
        EGL_DEPTH_SIZE, 0,
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
        EGL_NONE,
    };
    EGLint num_cfg = 0;
    if (!eglChooseConfig(b.egl_display, cfg_attrs, &b.egl_config, 1, &num_cfg) ||
        num_cfg < 1) {
        XR_LOGE("eglChooseConfig failed");
        return false;
    }
    const EGLint ctx_attrs[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE};
    b.egl_context = eglCreateContext(b.egl_display, b.egl_config, EGL_NO_CONTEXT, ctx_attrs);
    if (b.egl_context == EGL_NO_CONTEXT) {
        XR_LOGE("eglCreateContext failed: 0x%x", eglGetError());
        return false;
    }
    const EGLint pb_attrs[] = {EGL_WIDTH, 16, EGL_HEIGHT, 16, EGL_NONE};
    b.egl_surface = eglCreatePbufferSurface(b.egl_display, b.egl_config, pb_attrs);
    if (b.egl_surface == EGL_NO_SURFACE) {
        XR_LOGE("eglCreatePbufferSurface failed");
        return false;
    }
    if (!eglMakeCurrent(b.egl_display, b.egl_surface, b.egl_surface, b.egl_context)) {
        XR_LOGE("eglMakeCurrent failed");
        return false;
    }
    glGenTextures(1, &b.src_tex);
    glGenFramebuffers(1, &b.src_fbo);
    glGenFramebuffers(1, &b.dst_fbo);
    XR_LOGI("EGL %d.%d + OpenGL ES 3 initialized (%s)", major, minor,
            glGetString(GL_RENDERER));
    return true;
}

void shutdown_egl(Backend& b) {
    if (b.egl_display != EGL_NO_DISPLAY) {
        if (b.egl_context != EGL_NO_CONTEXT && b.egl_surface != EGL_NO_SURFACE) {
            eglMakeCurrent(b.egl_display, b.egl_surface, b.egl_surface, b.egl_context);
            if (b.src_tex) glDeleteTextures(1, &b.src_tex);
            if (b.src_fbo) glDeleteFramebuffers(1, &b.src_fbo);
            if (b.dst_fbo) glDeleteFramebuffers(1, &b.dst_fbo);
            if (b.hd_tex) glDeleteTextures(1, &b.hd_tex);
            if (b.hd_fbo) glDeleteFramebuffers(1, &b.hd_fbo);
            b.popup.shutdown();
        }
        b.src_tex = b.src_fbo = b.dst_fbo = 0;
        b.hd_tex = b.hd_fbo = 0;
        b.hd_tex_w = b.hd_tex_h = 0;
        eglMakeCurrent(b.egl_display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        if (b.egl_surface != EGL_NO_SURFACE) eglDestroySurface(b.egl_display, b.egl_surface);
        if (b.egl_context != EGL_NO_CONTEXT) eglDestroyContext(b.egl_display, b.egl_context);
        eglTerminate(b.egl_display);
    }
    b.egl_surface = EGL_NO_SURFACE;
    b.egl_context = EGL_NO_CONTEXT;
    b.egl_display = EGL_NO_DISPLAY;
}

void allocate_source_texture(Backend& b, int w, int h) {
    glBindTexture(GL_TEXTURE_2D, b.src_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0,
                 b.swapchain_srgb ? GL_SRGB8_ALPHA8 : GL_RGBA8,
                 w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, b.src_fbo);
    glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, b.src_tex, 0);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
}

bool create_swapchain_with_images(Backend& b, uint32_t w, uint32_t h,
                                  XrSwapchain& out_sc,
                                  std::vector<XrSwapchainImageOpenGLESKHR>& out_imgs) {
    uint32_t fmt_count = 0;
    xrEnumerateSwapchainFormats(b.session, 0, &fmt_count, nullptr);
    std::vector<int64_t> formats(fmt_count);
    xrEnumerateSwapchainFormats(b.session, fmt_count, &fmt_count, formats.data());

    int64_t chosen = GL_RGBA8;
    bool found_rgba8 = false;
    bool found_srgb = false;
    for (int64_t f : formats) {
        if (f == GL_RGBA8) found_rgba8 = true;
        if (f == GL_SRGB8_ALPHA8) found_srgb = true;
    }
    if (found_srgb) chosen = GL_SRGB8_ALPHA8;
    else if (!found_rgba8 && !formats.empty()) chosen = formats[0];
    b.swapchain_srgb = (chosen == GL_SRGB8_ALPHA8);

    XrSwapchainCreateInfo sci{XR_TYPE_SWAPCHAIN_CREATE_INFO};
    sci.usageFlags = XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT |
                     XR_SWAPCHAIN_USAGE_SAMPLED_BIT;
    sci.format = chosen;
    sci.sampleCount = 1;
    sci.width = w;
    sci.height = h;
    sci.faceCount = 1;
    sci.arraySize = 1;
    sci.mipCount = 1;

    XrResult res = xrCreateSwapchain(b.session, &sci, &out_sc);
    if (XR_FAILED(res)) {
        XR_LOGE("xrCreateSwapchain(%ux%u) failed: %d", w, h, res);
        return false;
    }
    uint32_t img_count = 0;
    xrEnumerateSwapchainImages(out_sc, 0, &img_count, nullptr);
    out_imgs.assign(img_count, XrSwapchainImageOpenGLESKHR{XR_TYPE_SWAPCHAIN_IMAGE_OPENGL_ES_KHR});
    xrEnumerateSwapchainImages(out_sc, img_count, &img_count,
                               reinterpret_cast<XrSwapchainImageBaseHeader*>(out_imgs.data()));
    return true;
}

bool init_actions(Backend& b) {
    xrStringToPath(b.instance, "/user/hand/left", &b.hand_paths[0]);
    xrStringToPath(b.instance, "/user/hand/right", &b.hand_paths[1]);

    XrActionSetCreateInfo asci{XR_TYPE_ACTION_SET_CREATE_INFO};
    std::snprintf(asci.actionSetName, sizeof(asci.actionSetName), "gameplay");
    std::snprintf(asci.localizedActionSetName, sizeof(asci.localizedActionSetName),
                  "MOTHER 3 Quest Controls");
    asci.priority = 0;
    if (XR_FAILED(xrCreateActionSet(b.instance, &asci, &b.action_set))) {
        XR_LOGE("xrCreateActionSet failed");
        return false;
    }

    auto make_action = [&](const char* name, const char* loc, XrActionType type,
                           uint32_t n_subpaths, const XrPath* subpaths, XrAction& out) {
        XrActionCreateInfo aci{XR_TYPE_ACTION_CREATE_INFO};
        std::snprintf(aci.actionName, sizeof(aci.actionName), "%s", name);
        std::snprintf(aci.localizedActionName, sizeof(aci.localizedActionName), "%s", loc);
        aci.actionType = type;
        aci.countSubactionPaths = n_subpaths;
        aci.subactionPaths = subpaths;
        return XR_SUCCEEDED(xrCreateAction(b.action_set, &aci, &out));
    };

    if (!make_action("aim_pose", "Controller Aim", XR_ACTION_TYPE_POSE_INPUT, 2, b.hand_paths, b.act_aim_pose) ||
        !make_action("thumbstick", "Thumbstick", XR_ACTION_TYPE_VECTOR2F_INPUT, 2, b.hand_paths, b.act_thumbstick) ||
        !make_action("trigger", "Trigger", XR_ACTION_TYPE_FLOAT_INPUT, 2, b.hand_paths, b.act_trigger) ||
        !make_action("squeeze", "Grip", XR_ACTION_TYPE_FLOAT_INPUT, 2, b.hand_paths, b.act_squeeze) ||
        !make_action("thumb_click", "Thumbstick Click", XR_ACTION_TYPE_BOOLEAN_INPUT, 2, b.hand_paths, b.act_thumb_click) ||
        !make_action("btn_a", "GBA A", XR_ACTION_TYPE_BOOLEAN_INPUT, 0, nullptr, b.act_a) ||
        !make_action("btn_b", "GBA B", XR_ACTION_TYPE_BOOLEAN_INPUT, 0, nullptr, b.act_b) ||
        !make_action("btn_x", "GBA Start", XR_ACTION_TYPE_BOOLEAN_INPUT, 0, nullptr, b.act_x) ||
        !make_action("btn_y", "GBA Select", XR_ACTION_TYPE_BOOLEAN_INPUT, 0, nullptr, b.act_y) ||
        !make_action("menu", "VR Menu", XR_ACTION_TYPE_BOOLEAN_INPUT, 0, nullptr, b.act_menu) ||
        !make_action("haptic", "Haptics", XR_ACTION_TYPE_VIBRATION_OUTPUT, 2, b.hand_paths, b.act_haptic)) {
        XR_LOGE("xrCreateAction failed");
        return false;
    }

    auto path = [&](const char* str) {
        XrPath p = XR_NULL_PATH;
        xrStringToPath(b.instance, str, &p);
        return p;
    };

    std::vector<XrActionSuggestedBinding> bindings = {
        {b.act_aim_pose, path("/user/hand/left/input/aim/pose")},
        {b.act_aim_pose, path("/user/hand/right/input/aim/pose")},
        {b.act_thumbstick, path("/user/hand/left/input/thumbstick")},
        {b.act_thumbstick, path("/user/hand/right/input/thumbstick")},
        {b.act_trigger, path("/user/hand/left/input/trigger/value")},
        {b.act_trigger, path("/user/hand/right/input/trigger/value")},
        {b.act_squeeze, path("/user/hand/left/input/squeeze/value")},
        {b.act_squeeze, path("/user/hand/right/input/squeeze/value")},
        {b.act_thumb_click, path("/user/hand/left/input/thumbstick/click")},
        {b.act_thumb_click, path("/user/hand/right/input/thumbstick/click")},
        {b.act_a, path("/user/hand/right/input/a/click")},
        {b.act_b, path("/user/hand/right/input/b/click")},
        {b.act_x, path("/user/hand/left/input/x/click")},
        {b.act_y, path("/user/hand/left/input/y/click")},
        {b.act_menu, path("/user/hand/left/input/menu/click")},
        {b.act_haptic, path("/user/hand/left/output/haptic")},
        {b.act_haptic, path("/user/hand/right/output/haptic")},
    };

    XrInteractionProfileSuggestedBinding ip{XR_TYPE_INTERACTION_PROFILE_SUGGESTED_BINDING};
    ip.interactionProfile = path("/interaction_profiles/oculus/touch_controller");
    ip.countSuggestedBindings = static_cast<uint32_t>(bindings.size());
    ip.suggestedBindings = bindings.data();
    if (XR_FAILED(xrSuggestInteractionProfileBindings(b.instance, &ip))) {
        XR_LOGE("xrSuggestInteractionProfileBindings failed");
        return false;
    }

    for (int i = 0; i < 2; ++i) {
        XrActionSpaceCreateInfo spci{XR_TYPE_ACTION_SPACE_CREATE_INFO};
        spci.action = b.act_aim_pose;
        spci.subactionPath = b.hand_paths[i];
        spci.poseInActionSpace = {{0.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 0.0f}};
        if (XR_FAILED(xrCreateActionSpace(b.session, &spci, &b.aim_spaces[i]))) {
            XR_LOGE("xrCreateActionSpace failed (%d)", i);
            return false;
        }
    }

    XrSessionActionSetsAttachInfo attach{XR_TYPE_SESSION_ACTION_SETS_ATTACH_INFO};
    attach.countActionSets = 1;
    attach.actionSets = &b.action_set;
    if (XR_FAILED(xrAttachSessionActionSets(b.session, &attach))) {
        XR_LOGE("xrAttachSessionActionSets failed");
        return false;
    }
    b.actions_attached = true;
    return true;
}

void recenter_screen_to_head(Backend& b, XrTime time) {
    if (b.view_space == XR_NULL_HANDLE || b.local_space == XR_NULL_HANDLE || time <= 0) return;
    XrSpaceLocation loc{XR_TYPE_SPACE_LOCATION};
    if (XR_FAILED(xrLocateSpace(b.view_space, b.local_space, time, &loc))) return;
    if ((loc.locationFlags & XR_SPACE_LOCATION_ORIENTATION_VALID_BIT) == 0 ||
        (loc.locationFlags & XR_SPACE_LOCATION_POSITION_VALID_BIT) == 0) {
        return;
    }
    const XrVector3f fwd = v3_norm(rotate_vec(loc.pose.orientation, 0.0f, 0.0f, -1.0f), {0.0f, 0.0f, -1.0f});
    const XrVector3f head_up = v3_norm(rotate_vec(loc.pose.orientation, 0.0f, 1.0f, 0.0f), {0.0f, 1.0f, 0.0f});
    const XrVector3f head_right = v3_norm(rotate_vec(loc.pose.orientation, 1.0f, 0.0f, 0.0f), {1.0f, 0.0f, 0.0f});

    // Place screen along full 3D gaze vector so reclining / laying down looking at the ceiling
    // places the 5-layer 3D window directly overhead in front of the player's eyes.
    const XrVector3f offset = v3_sub(v3_scale(fwd, b.screen_distance), v3_scale(head_up, 0.04f));
    b.screen_pose.position = v3_add(loc.pose.position, offset);

    // When sitting/standing upright, keep the horizon level (world up = +Y).
    // As the player reclines or lays down (looking steeply up/down or lying on their side),
    // smoothly blend toward head_up so ceiling / side-lying play has zero gimbal twist.
    const float pitch_upright = std::clamp((0.82f - std::abs(fwd.y)) / 0.25f, 0.0f, 1.0f);
    const float roll_upright = std::clamp((head_up.y - 0.35f) / 0.35f, 0.0f, 1.0f);
    const float upright_w = pitch_upright * roll_upright;
    const XrVector3f world_up = {0.0f, 1.0f, 0.0f};
    const XrVector3f ref_up = v3_norm(
        v3_add(v3_scale(world_up, upright_w), v3_scale(head_up, 1.0f - upright_w)),
        head_up);

    b.screen_pose.orientation = make_screen_orientation(offset, ref_up, head_right);
    b.pose_initialized = true;
}

void trigger_haptic_ns(Backend& b, int hand, XrDuration ns, float amplitude);

// Remember the player's screen distance and size (two-grip move/pull, window scale) across
// launches: files/xr_view.txt "<distance m> <scale>". Opt-in per game (GBARECOMP_XR_REMEMBER_VIEW=1).
static bool remember_view_enabled() {
    const char* e = std::getenv("GBARECOMP_XR_REMEMBER_VIEW");
    return e && *e == '1';
}
static std::string xr_view_path() {
    const char* dir = SDL_AndroidGetInternalStoragePath();
    return dir ? std::string(dir) + "/xr_view.txt" : std::string();
}
static void save_xr_view(const Backend& b) {
    if (!remember_view_enabled()) return;
    const std::string p = xr_view_path();
    if (p.empty()) return;
    if (std::FILE* f = std::fopen(p.c_str(), "w")) {
        std::fprintf(f, "%.3f %d\n", b.screen_distance, b.scale);
        std::fclose(f);
        XR_LOGI("view saved: distance %.3f m, scale %d", b.screen_distance, b.scale);
    }
}
static void load_xr_view(Backend& b) {
    if (!remember_view_enabled()) return;
    const std::string p = xr_view_path();
    if (p.empty()) return;
    if (std::FILE* f = std::fopen(p.c_str(), "r")) {
        float d = 0.0f; int sc = 0;
        if (std::fscanf(f, "%f %d", &d, &sc) == 2 && d >= 0.55f && d <= 3.2f && sc >= 1 && sc <= 8) {
            b.screen_distance = d;
            b.scale = sc;
            b.screen_width_m.store(0.85f + 0.19f * static_cast<float>(sc), std::memory_order_release);
            XR_LOGI("view restored: distance %.3f m, scale %d", d, sc);
        }
        std::fclose(f);
    }
}

// Menu choices (EDGES, BACKGROUND) remembered across launches: files/xr_prefs.txt "<window 0/1> <dark 0/1>".
// The env defaults (GBARECOMP_XR_WINDOW / _DARK) apply until the player first changes one. Same opt-in.
static std::string xr_prefs_path() {
    const char* dir = SDL_AndroidGetInternalStoragePath();
    return dir ? std::string(dir) + "/xr_prefs.txt" : std::string();
}
static void save_xr_prefs(const Backend& b) {
    if (!remember_view_enabled()) return;
    const std::string p = xr_prefs_path();
    if (p.empty()) return;
    if (std::FILE* f = std::fopen(p.c_str(), "w")) {
        const bool dark = b.passthrough_supported ? (b.hud_mode == 2) : b.start_dark;
        std::fprintf(f, "%d %d %d %d %d\n", b.window_clip ? 1 : 0, dark ? 1 : 0, b.auto_resume ? 1 : 0,
                     static_cast<int>(std::lround(kGyroSensSteps[b.gyro_sens_idx] * 100.0f)), b.active_gyro_hand);
        std::fclose(f);
        XR_LOGI("prefs saved: window %d dark %d", b.window_clip ? 1 : 0, dark ? 1 : 0);
    }
}
static void load_xr_prefs(Backend& b) {
    if (!remember_view_enabled()) return;
    const std::string p = xr_prefs_path();
    if (p.empty()) return;
    if (std::FILE* f = std::fopen(p.c_str(), "r")) {
        int w = 0, d = 0, r = 1, g = kGyroSensDefault, h = 1;
        if (std::fscanf(f, "%d %d %d %d %d", &w, &d, &r, &g, &h) >= 2) {
            // Field 4 = sensitivity in percent of the menu's 1.00X (small values = an older
            // index format with another scale: ignored).
            for (int i = 0; i < kGyroSensCount; ++i)
                if (g >= 10 && std::lround(kGyroSensSteps[i] * 100.0f) == g) b.gyro_sens_idx = i;
            b.active_gyro_hand = h == 0 ? 0 : 1;
            b.window_clip = w == 1;
            b.start_dark = d == 1;
            b.auto_resume = r != 0;
            XR_LOGI("prefs restored: window %d dark %d", w, d);
        }
        std::fclose(f);
    }
}

void update_two_grip_window_move(Backend& b, XrTime time, bool both_grips, float stick_y) {
    if (!both_grips) {
        if (b.grip_move_active) {
            b.grip_move_active = false;
            trigger_haptic_ns(b, 0, 12000000LL, 0.20f);
            trigger_haptic_ns(b, 1, 12000000LL, 0.20f);
            save_xr_view(b);
        }
        return;
    }
    if (b.view_space == XR_NULL_HANDLE || b.local_space == XR_NULL_HANDLE ||
        b.aim_spaces[0] == XR_NULL_HANDLE || b.aim_spaces[1] == XR_NULL_HANDLE || time <= 0) {
        return;
    }

    XrSpaceLocation head_loc{XR_TYPE_SPACE_LOCATION};
    XrSpaceLocation l_loc{XR_TYPE_SPACE_LOCATION};
    XrSpaceLocation r_loc{XR_TYPE_SPACE_LOCATION};
    if (XR_FAILED(xrLocateSpace(b.view_space, b.local_space, time, &head_loc)) ||
        XR_FAILED(xrLocateSpace(b.aim_spaces[0], b.local_space, time, &l_loc)) ||
        XR_FAILED(xrLocateSpace(b.aim_spaces[1], b.local_space, time, &r_loc))) {
        return;
    }
    constexpr XrSpaceLocationFlags kNeedFlags =
        XR_SPACE_LOCATION_ORIENTATION_VALID_BIT | XR_SPACE_LOCATION_POSITION_VALID_BIT;
    if ((head_loc.locationFlags & kNeedFlags) != kNeedFlags ||
        (l_loc.locationFlags & kNeedFlags) != kNeedFlags ||
        (r_loc.locationFlags & kNeedFlags) != kNeedFlags) {
        return;
    }

    const XrVector3f head_pos = head_loc.pose.position;
    const XrVector3f mid_pos = v3_scale(v3_add(l_loc.pose.position, r_loc.pose.position), 0.5f);
    const XrVector3f aim_l = rotate_vec(l_loc.pose.orientation, 0.0f, 0.0f, -1.0f);
    const XrVector3f aim_r = rotate_vec(r_loc.pose.orientation, 0.0f, 0.0f, -1.0f);
    const XrVector3f mid_aim = v3_norm(v3_add(aim_l, aim_r), {0.0f, 0.0f, -1.0f});

    // Virtual grab anchor sits slightly ahead of the two-hand midpoint along average controller aim,
    // so both moving your hands and tilting your wrists smoothly swing the window anywhere (including overhead).
    const XrVector3f grab_pt = v3_add(mid_pos, v3_scale(mid_aim, 0.38f));
    const XrVector3f grab_vec = v3_sub(grab_pt, head_pos);
    const float grab_len = std::max(0.10f, v3_len(grab_vec));
    const XrVector3f grab_dir = v3_scale(grab_vec, 1.0f / grab_len);
    const float hand_dist = std::max(0.10f, v3_len(v3_sub(mid_pos, head_pos)));
    const XrVector3f hand_right = v3_norm(v3_sub(r_loc.pose.position, l_loc.pose.position),
                                          rotate_vec(head_loc.pose.orientation, 1.0f, 0.0f, 0.0f));

    if (!b.grip_move_active) {
        b.grip_move_active = true;
        b.grip_start_grab_dir = grab_dir;
        b.grip_start_hand_dist = hand_dist;
        b.grip_start_screen_offset = v3_sub(b.screen_pose.position, head_pos);
        b.grip_start_screen_dist = std::clamp(v3_len(b.grip_start_screen_offset), 0.55f, 3.20f);
        if (v3_len(b.grip_start_screen_offset) < 0.20f) {
            b.grip_start_screen_offset = v3_scale(grab_dir, b.grip_start_screen_dist);
        }
        b.grip_start_screen_up = v3_norm(rotate_vec(b.screen_pose.orientation, 0.0f, 1.0f, 0.0f), {0.0f, 1.0f, 0.0f});
        b.grip_start_screen_right = v3_norm(rotate_vec(b.screen_pose.orientation, 1.0f, 0.0f, 0.0f), {1.0f, 0.0f, 0.0f});
        b.grip_start_hand_right = hand_right;
        trigger_haptic_ns(b, 0, 18000000LL, 0.35f);
        trigger_haptic_ns(b, 1, 18000000LL, 0.35f);
        return;
    }

    // Allow thumbstick Y while holding both grips to smoothly push/pull the window distance
    if (std::abs(stick_y) > kThumbstickDeadzone) {
        b.grip_start_screen_dist = std::clamp(b.grip_start_screen_dist + stick_y * 0.018f, 0.55f, 3.20f);
    }

    const XrQuaternionf delta_rot = quat_from_two_unit_vecs(b.grip_start_grab_dir, grab_dir);
    const XrVector3f orbited_dir = v3_norm(
        rotate_vec(delta_rot, b.grip_start_screen_offset.x, b.grip_start_screen_offset.y, b.grip_start_screen_offset.z),
        grab_dir);

    const float dist_ratio = std::clamp(hand_dist / std::max(0.12f, b.grip_start_hand_dist), 0.50f, 2.00f);
    const float new_dist = std::clamp(b.grip_start_screen_dist * dist_ratio, 0.55f, 3.20f);
    b.screen_distance = new_dist;

    const XrVector3f new_offset = v3_scale(orbited_dir, new_dist);
    b.screen_pose.position = v3_add(head_pos, new_offset);

    // Follow 3D orbit + two-hand roll delta so the player can tilt the window when lying on their side
    const XrVector3f orbited_up = v3_norm(
        rotate_vec(delta_rot, b.grip_start_screen_up.x, b.grip_start_screen_up.y, b.grip_start_screen_up.z),
        {0.0f, 1.0f, 0.0f});
    const XrVector3f orbited_right = v3_norm(
        rotate_vec(delta_rot, b.grip_start_screen_right.x, b.grip_start_screen_right.y, b.grip_start_screen_right.z),
        {1.0f, 0.0f, 0.0f});
    const XrVector3f orbited_hand_right = v3_norm(
        rotate_vec(delta_rot, b.grip_start_hand_right.x, b.grip_start_hand_right.y, b.grip_start_hand_right.z),
        orbited_right);
    const XrQuaternionf roll_rot = quat_from_two_unit_vecs(orbited_hand_right, hand_right);
    const XrVector3f rolled_up = v3_norm(
        rotate_vec(roll_rot, orbited_up.x, orbited_up.y, orbited_up.z),
        orbited_up);

    // When mostly upright and hands are level, gently stabilize horizon; when overhead/tilted, follow full 3D pose
    const XrVector3f head_up = v3_norm(rotate_vec(head_loc.pose.orientation, 0.0f, 1.0f, 0.0f), {0.0f, 1.0f, 0.0f});
    const float pitch_upright = std::clamp((0.78f - std::abs(orbited_dir.y)) / 0.25f, 0.0f, 1.0f);
    const float roll_upright = std::clamp((head_up.y - 0.45f) / 0.35f, 0.0f, 1.0f);
    const float level_hand = std::clamp((0.25f - std::abs(hand_right.y)) / 0.20f, 0.0f, 1.0f);
    const float world_w = pitch_upright * roll_upright * level_hand * 0.65f;
    const XrVector3f world_up = {0.0f, 1.0f, 0.0f};
    const XrVector3f ref_up = v3_norm(
        v3_add(v3_scale(world_up, world_w), v3_scale(rolled_up, 1.0f - world_w)),
        rolled_up);

    b.screen_pose.orientation = make_screen_orientation(new_offset, ref_up, orbited_right);
    b.pose_initialized = true;
}

void upload_hud_image(Backend& b) {
    if (b.hud_swapchain == XR_NULL_HANDLE || !b.hud_dirty) return;
    b.hud_dirty = false;

    uint32_t img_idx = 0;
    XrSwapchainImageAcquireInfo acq{XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO};
    if (XR_FAILED(xrAcquireSwapchainImage(b.hud_swapchain, &acq, &img_idx))) return;
    XrSwapchainImageWaitInfo wait{XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO};
    wait.timeout = XR_INFINITE_DURATION;
    if (XR_FAILED(xrWaitSwapchainImage(b.hud_swapchain, &wait))) {
        XrSwapchainImageReleaseInfo rel{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
        xrReleaseSwapchainImage(b.hud_swapchain, &rel);
        return;
    }

    std::vector<uint8_t> rgba(static_cast<size_t>(kHudWidth) * kHudHeight * 4, 0);
    // Dark translucent rounded-bar backing (RGBA bottom-up)
    for (int y = 0; y < kHudHeight; ++y) {
        for (int x = 0; x < kHudWidth; ++x) {
            const bool border = (y == 0 || y == kHudHeight - 1 || x == 0 || x == kHudWidth - 1);
            const size_t idx = (static_cast<size_t>(y) * kHudWidth + x) * 4;
            if (border) {
                rgba[idx + 0] = 250;
                rgba[idx + 1] = 210;
                rgba[idx + 2] = 60;
                rgba[idx + 3] = 235;
            } else {
                rgba[idx + 0] = 10;
                rgba[idx + 1] = 14;
                rgba[idx + 2] = 24;
                rgba[idx + 3] = 232;
            }
        }
    }

    const char* depth_name = b.stock_deep ? "STOCK 3D 11CM" : "STOCK 3D 6CM";
    if (b.depth_mode == 0) depth_name = "2D FLAT";
    else if (b.depth_mode == 2) depth_name = "POP-UP 3D";

    const char* view_name = (b.base_w > 240) ? "16:9 WIDE" : "4:3";

    char line1[160];
    std::snprintf(line1, sizeof(line1),
                  "%s | VIEW:%s | DEPTH:%s | MR:%s",
                  b.hud_title.c_str(),
                  view_name,
                  depth_name,
                  (b.passthrough_supported && b.passthrough_active) ? "ON" : "OFF");
    draw_hud_text(rgba, kHudWidth, kHudHeight, 21, 15, 3, 0xFAD23CFFu, line1);
    if (b.menu_enabled) {
        draw_hud_text(rgba, kHudWidth, kHudHeight, 21, 51, 3, 0xFFFFFFFFu,
                      "A/RT: A   B/RG: B   LT: L   LG: R   X: START   Y: SELECT");
        draw_hud_text(rgba, kHudWidth, kHudHeight, 21, 87, 3, 0xB8C4E0FFu,
                      "MENU: OPTIONS  L+R GRIP: MOVE  R-CLK: RECENTER  HOLD R-CLK: CAPTURE");
    } else {
        draw_hud_text(rgba, kHudWidth, kHudHeight, 21, 51, 3, 0xFFFFFFFFu,
                      "A/RT: A(BEAT)  B/RG: B(DASH)  LT: L(1-HAND)  LG: R  X/Y: STA/SEL");
        draw_hud_text(rgba, kHudWidth, kHudHeight, 21, 87, 3, 0xB8C4E0FFu,
                      "L+R GRIP: MOVE WIN  L-CLK: 3D DEPTH  R-CLK: RECENTER  MENU: MR/HUD");
    }

    // Premultiply HUD RGB channels to match OpenXR premultiplied-alpha quad compositing.
    for (size_t i = 0; i < rgba.size(); i += 4) {
        const uint8_t a = rgba[i + 3];
        rgba[i + 0] = premul_channel(rgba[i + 0], a, b.swapchain_srgb);
        rgba[i + 1] = premul_channel(rgba[i + 1], a, b.swapchain_srgb);
        rgba[i + 2] = premul_channel(rgba[i + 2], a, b.swapchain_srgb);
    }

    glBindTexture(GL_TEXTURE_2D, b.hud_images[img_idx].image);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, kHudWidth, kHudHeight,
                    GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());
    glBindTexture(GL_TEXTURE_2D, 0);

    XrSwapchainImageReleaseInfo rel{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
    xrReleaseSwapchainImage(b.hud_swapchain, &rel);
}

// ---- VR options menu (GBARECOMP_XR_MENU=1) ----
enum MenuItem { kMenuResume, kMenuSave, kMenuLoad, kMenuDepth, kMenuGyro, kMenuGyroHand, kMenuBackground, kMenuEdges, kMenuAutoResume, kMenuRecenter, kMenuResetView, kMenuCount };

int menu_depth_index(const Backend& b) { return b.depth_mode == 0 ? 2 : (b.stock_deep ? 1 : 0); }
const char* menu_depth_name(int i) { return i == 0 ? "NORMAL 3D" : (i == 1 ? "DEEP 3D" : "FLAT 2D"); }
// GYRO SENSITIVITY only exists for games with the cartridge gyro.
bool menu_item_visible(const Backend& b, int i) {
    return (i != kMenuGyro && i != kMenuGyroHand) || b.gyro_enabled;
}
int menu_visible_count(const Backend& b) {
    int n = 0;
    for (int i = 0; i < kMenuCount; ++i) n += menu_item_visible(b, i) ? 1 : 0;
    return n;
}
const char* menu_background_name(const Backend& b) {
    if (!b.passthrough_supported) return "DARK";
    return b.hud_mode == 2 ? "DARK" : "ROOM";  // the help bar is not shown with the menu
}

void menu_fill(std::vector<uint8_t>& rgba, int x0, int y0, int w, int h, uint32_t c) {
    for (int y = y0; y < y0 + h; ++y) {
        if (y < 0 || y >= kMenuHeight) continue;
        const int py = kMenuHeight - 1 - y;  // bottom-up rows
        for (int x = x0; x < x0 + w; ++x) {
            if (x < 0 || x >= kMenuWidth) continue;
            uint8_t* d = &rgba[(static_cast<size_t>(py) * kMenuWidth + x) * 4];
            d[0] = static_cast<uint8_t>(c >> 24); d[1] = static_cast<uint8_t>(c >> 16);
            d[2] = static_cast<uint8_t>(c >> 8);  d[3] = static_cast<uint8_t>(c);
        }
    }
}

void upload_menu_image(Backend& b) {
    if (b.menu_swapchain == XR_NULL_HANDLE || !b.menu_dirty) return;
    b.menu_dirty = false;
    uint32_t img_idx = 0;
    XrSwapchainImageAcquireInfo acq{XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO};
    if (XR_FAILED(xrAcquireSwapchainImage(b.menu_swapchain, &acq, &img_idx))) return;
    XrSwapchainImageWaitInfo wait{XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO};
    wait.timeout = XR_INFINITE_DURATION;
    if (XR_FAILED(xrWaitSwapchainImage(b.menu_swapchain, &wait))) {
        XrSwapchainImageReleaseInfo rel{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
        xrReleaseSwapchainImage(b.menu_swapchain, &rel);
        return;
    }
    std::vector<uint8_t> rgba(static_cast<size_t>(kMenuWidth) * kMenuHeight * 4, 0);
    menu_fill(rgba, 0, 0, kMenuWidth, kMenuHeight, 0xFAD23CF0u);            // gold border
    menu_fill(rgba, 6, 6, kMenuWidth - 12, kMenuHeight - 12, 0x0A0E18F0u);   // dark panel
    menu_fill(rgba, 6, 6, kMenuWidth - 12, 150, 0x3A2A08F0u);                // title band
    // Header: the game's own logo when the port provides files/menu_logo.rgba (u16 w, u16 h,
    // RGBA rows top-down; extracted from the player's ROM), else the title text.
    static std::vector<uint8_t> logo;
    static int logo_w = 0, logo_h = 0;
    static bool logo_tried = false;
    if (!logo_tried) {
        logo_tried = true;
        if (const char* dir = SDL_AndroidGetInternalStoragePath()) {
            if (std::FILE* f = std::fopen((std::string(dir) + "/menu_logo.rgba").c_str(), "rb")) {
                uint16_t wh[2] = {};
                if (std::fread(wh, 2, 2, f) == 2 && wh[0] > 0 && wh[1] > 0 && wh[0] <= 480 && wh[1] <= 200) {
                    logo.resize(static_cast<size_t>(wh[0]) * wh[1] * 4);
                    if (std::fread(logo.data(), 1, logo.size(), f) == logo.size()) { logo_w = wh[0]; logo_h = wh[1]; }
                }
                std::fclose(f);
            }
        }
    }
    if (logo_w > 0) {
        const int sc = std::max(1, std::min((kMenuWidth - 60) / logo_w, 136 / logo_h));
        const int ox = (kMenuWidth - logo_w * sc) / 2, oy = 6 + (150 - logo_h * sc) / 2;
        for (int y = 0; y < logo_h * sc; ++y) {
            const int py = kMenuHeight - 1 - (oy + y);
            for (int x = 0; x < logo_w * sc; ++x) {
                const uint8_t* sp = &logo[(static_cast<size_t>(y / sc) * logo_w + x / sc) * 4];
                if (!sp[3]) continue;
                uint8_t* d = &rgba[(static_cast<size_t>(py) * kMenuWidth + ox + x) * 4];
                d[0] = sp[0]; d[1] = sp[1]; d[2] = sp[2]; d[3] = 255;
            }
        }
    } else {
        draw_hud_text(rgba, kMenuWidth, kMenuHeight, 36, 60, 6, 0xFAD23CFFu, b.hud_title.c_str());
    }
    char slot[32];
    std::snprintf(slot, sizeof(slot), "SLOT %d", b.menu_slot);
    char gyro[16];
    std::snprintf(gyro, sizeof(gyro), "%.2fX", static_cast<double>(kGyroSensSteps[b.gyro_sens_idx]));
    const char* labels[kMenuCount] = {"RESUME", "SAVE STATE", "LOAD STATE", "3D DEPTH",
                                      "GYRO SENSITIVITY", "GYRO HAND", "BACKGROUND", "EDGES", "AUTO RESUME", "RECENTER SCREEN", "RESET SCREEN SIZE"};
    const char* values[kMenuCount] = {"", slot, slot, menu_depth_name(menu_depth_index(b)), gyro,
                                      b.active_gyro_hand == 1 ? "RIGHT" : "LEFT",
                                      menu_background_name(b), b.window_clip ? "WINDOW" : "SHADOW BOX",
                                      b.auto_resume ? "ON" : "OFF", "", ""};
    const int row_h = menu_visible_count(b) > 9 ? 40 : 50;
    int row = 0;
    for (int i = 0; i < kMenuCount; ++i) {
        if (!menu_item_visible(b, i)) continue;
        const int y = 178 + (row++) * row_h;
        const bool sel = i == b.menu_sel;
        if (sel) menu_fill(rgba, 20, y - 6, kMenuWidth - 40, row_h - 4, 0xFAD23CFFu);
        const uint32_t fg = sel ? 0x10141CFFu : 0xFFFFFFFFu;
        draw_hud_text(rgba, kMenuWidth, kMenuHeight, 44, y, 5, fg, labels[i]);
        if (values[i][0]) {
            char v[64];
            std::snprintf(v, sizeof(v), "- %s +", values[i]);
            const int vw = static_cast<int>(std::strlen(v)) * 6 * 4;
            draw_hud_text(rgba, kMenuWidth, kMenuHeight, kMenuWidth - 44 - vw, y + 4, 4,
                          sel ? 0x10141CFFu : 0x9FD0FFFFu, v);
        }
    }
    if (!b.menu_status.empty())
        draw_hud_text(rgba, kMenuWidth, kMenuHeight, 44, kMenuHeight - 82, 4, 0x8CF08CFFu, b.menu_status.c_str());
    draw_hud_text(rgba, kMenuWidth, kMenuHeight, 44, kMenuHeight - 42, 3, 0xB8C4E0FFu,
                  "STICK: MOVE / CHANGE    A: OK    B: CLOSE");
    for (size_t i = 0; i < rgba.size(); i += 4) {
        const uint8_t a = rgba[i + 3];
        rgba[i + 0] = premul_channel(rgba[i + 0], a, b.swapchain_srgb);
        rgba[i + 1] = premul_channel(rgba[i + 1], a, b.swapchain_srgb);
        rgba[i + 2] = premul_channel(rgba[i + 2], a, b.swapchain_srgb);
    }
    glBindTexture(GL_TEXTURE_2D, b.menu_images[img_idx].image);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, kMenuWidth, kMenuHeight, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());
    glBindTexture(GL_TEXTURE_2D, 0);
    XrSwapchainImageReleaseInfo rel{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
    xrReleaseSwapchainImage(b.menu_swapchain, &rel);
}

void trigger_haptic_ns(Backend& b, int hand, XrDuration ns, float amplitude) {
    if (!b.actions_attached || b.session == XR_NULL_HANDLE ||
        b.session_state != XR_SESSION_STATE_FOCUSED) {
        return;
    }
    XrHapticActionInfo hai{XR_TYPE_HAPTIC_ACTION_INFO};
    hai.action = b.act_haptic;
    hai.subactionPath = b.hand_paths[hand];
    XrHapticVibration vib{XR_TYPE_HAPTIC_VIBRATION};
    vib.duration = ns;
    vib.frequency = XR_FREQUENCY_UNSPECIFIED;
    vib.amplitude = std::clamp(amplitude, 0.0f, 1.0f);
    xrApplyHapticFeedback(b.session, &hai,
                          reinterpret_cast<const XrHapticBaseHeader*>(&vib));
}

void stop_haptic(Backend& b, int hand) {
    if (!b.actions_attached || b.session == XR_NULL_HANDLE) return;
    XrHapticActionInfo hai{XR_TYPE_HAPTIC_ACTION_INFO};
    hai.action = b.act_haptic;
    hai.subactionPath = b.hand_paths[hand];
    xrStopHapticFeedback(b.session, &hai);
}

void apply_passthrough_state(Backend& b) {
    if (!b.passthrough_supported || b.passthrough == XR_NULL_HANDLE ||
        b.passthrough_layer == XR_NULL_HANDLE) {
        return;
    }
    if (b.passthrough_active) {
        if (b.pfnPassthroughStartFB) b.pfnPassthroughStartFB(b.passthrough);
        if (b.pfnPassthroughLayerResumeFB) b.pfnPassthroughLayerResumeFB(b.passthrough_layer);
    } else {
        if (b.pfnPassthroughLayerPauseFB) b.pfnPassthroughLayerPauseFB(b.passthrough_layer);
        if (b.pfnPassthroughPauseFB) b.pfnPassthroughPauseFB(b.passthrough);
    }
}

void handle_xr_events(Backend& b) {
    for (;;) {
        XrEventDataBuffer ev{XR_TYPE_EVENT_DATA_BUFFER};
        const XrResult res = xrPollEvent(b.instance, &ev);
        if (res == XR_EVENT_UNAVAILABLE) break;
        if (XR_FAILED(res)) {
            XR_LOGW("xrPollEvent error: %d", res);
            break;
        }
        switch (ev.type) {
            case XR_TYPE_EVENT_DATA_INSTANCE_LOSS_PENDING:
                b.exit_requested = true;
                break;
            case XR_TYPE_EVENT_DATA_REFERENCE_SPACE_CHANGE_PENDING: {
                const auto* rsc =
                    reinterpret_cast<const XrEventDataReferenceSpaceChangePending*>(&ev);
                XR_LOGI("OpenXR reference space change pending (space=%d, changeTime=%lld) -> recentering screen",
                        static_cast<int>(rsc->referenceSpaceType),
                        static_cast<long long>(rsc->changeTime));
                b.pose_initialized = false;
                b.recenter_after_time = rsc->changeTime;
                b.recenter_frames_remaining = 6;
                break;
            }
            case XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED: {
                const auto* ssc = reinterpret_cast<const XrEventDataSessionStateChanged*>(&ev);
                b.session_state = ssc->state;
                XR_LOGI("OpenXR session state -> %d", static_cast<int>(ssc->state));
                if (ssc->state == XR_SESSION_STATE_READY) {
                    XrSessionBeginInfo sbi{XR_TYPE_SESSION_BEGIN_INFO};
                    sbi.primaryViewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
                    if (XR_SUCCEEDED(xrBeginSession(b.session, &sbi))) {
                        b.session_running = true;
                        b.pose_initialized = false;
                        b.recenter_after_time = 0;
                        b.recenter_frames_remaining = 6;
                        apply_passthrough_state(b);
                        if (b.perf_settings_ext) {
                            PFN_xrPerfSettingsSetPerformanceLevelEXT set_level = nullptr;
                            xrGetInstanceProcAddr(b.instance, "xrPerfSettingsSetPerformanceLevelEXT",
                                                  reinterpret_cast<PFN_xrVoidFunction*>(&set_level));
                            if (set_level) {
                                const XrResult pr = set_level(b.session, XR_PERF_SETTINGS_DOMAIN_CPU_EXT,
                                                              XR_PERF_SETTINGS_LEVEL_SUSTAINED_HIGH_EXT);
                                XR_LOGI("CPU performance level sustained-high: %d", static_cast<int>(pr));
                            }
                        }
                        XR_LOGI("xrBeginSession succeeded");
                    } else {
                        XR_LOGE("xrBeginSession failed");
                    }
                } else if (ssc->state == XR_SESSION_STATE_STOPPING) {
                    b.background_pending = true;
                    if (b.session_running) {
                        xrEndSession(b.session);
                        b.session_running = false;
                    }
                } else if (ssc->state == XR_SESSION_STATE_EXITING ||
                           ssc->state == XR_SESSION_STATE_LOSS_PENDING) {
                    b.exit_requested = true;
                    b.session_running = false;
                }
                break;
            }
            default:
                break;
        }
    }
}

bool init_openxr(Backend& b) {
    JNIEnv* env = static_cast<JNIEnv*>(SDL_AndroidGetJNIEnv());
    JavaVM* vm = b.vm;
    jobject activity = b.activity_global_ref;
    bool owns_local_activity = false;
    if (!vm && env) env->GetJavaVM(&vm);
    if (!activity) {
        activity = static_cast<jobject>(SDL_AndroidGetActivity());
        owns_local_activity = (activity != nullptr);
    }
    if (!vm || !activity) {
        XR_LOGE("Missing JavaVM (%p) or Activity (%p)", static_cast<void*>(vm),
                static_cast<void*>(activity));
        if (env && owns_local_activity) env->DeleteLocalRef(activity);
        return false;
    }

    PFN_xrInitializeLoaderKHR pfnInitLoader = nullptr;
    if (XR_SUCCEEDED(xrGetInstanceProcAddr(
            XR_NULL_HANDLE, "xrInitializeLoaderKHR",
            reinterpret_cast<PFN_xrVoidFunction*>(&pfnInitLoader))) &&
        pfnInitLoader) {
        XrLoaderInitInfoAndroidKHR lii{XR_TYPE_LOADER_INIT_INFO_ANDROID_KHR};
        lii.applicationVM = vm;
        lii.applicationContext = activity;
        const XrResult lres = pfnInitLoader(
            reinterpret_cast<const XrLoaderInitInfoBaseHeaderKHR*>(&lii));
        if (XR_FAILED(lres)) {
            XR_LOGE("xrInitializeLoaderKHR failed: %d", lres);
            if (env && owns_local_activity) env->DeleteLocalRef(activity);
            return false;
        }
    }

    uint32_t ext_count = 0;
    xrEnumerateInstanceExtensionProperties(nullptr, 0, &ext_count, nullptr);
    std::vector<XrExtensionProperties> props(
        ext_count, XrExtensionProperties{XR_TYPE_EXTENSION_PROPERTIES});
    xrEnumerateInstanceExtensionProperties(nullptr, ext_count, &ext_count, props.data());
    auto has_ext = [&](const char* name) {
        for (const auto& p : props) {
            if (std::strcmp(p.extensionName, name) == 0) return true;
        }
        return false;
    };

    std::vector<const char*> exts = {
        XR_KHR_ANDROID_CREATE_INSTANCE_EXTENSION_NAME,
        XR_KHR_OPENGL_ES_ENABLE_EXTENSION_NAME,
    };
    b.passthrough_ext = has_ext(XR_FB_PASSTHROUGH_EXTENSION_NAME);
    if (b.passthrough_ext) exts.push_back(XR_FB_PASSTHROUGH_EXTENSION_NAME);
    const bool has_refresh = has_ext(XR_FB_DISPLAY_REFRESH_RATE_EXTENSION_NAME);
    if (has_refresh) exts.push_back(XR_FB_DISPLAY_REFRESH_RATE_EXTENSION_NAME);
    b.layer_settings_ext = has_ext(XR_FB_COMPOSITION_LAYER_SETTINGS_EXTENSION_NAME);
    if (b.layer_settings_ext) exts.push_back(XR_FB_COMPOSITION_LAYER_SETTINGS_EXTENSION_NAME);
    // Optional (GBARECOMP_XR_CPU_HIGH=1): ask the runtime for sustained-high CPU clocks so busy
    // scenes do not fall behind the GBA's 59.73 Hz (late frames slow the music down).
    {
        const char* e = std::getenv("GBARECOMP_XR_CPU_HIGH");
        b.perf_settings_ext = e && *e == '1' && has_ext(XR_EXT_PERFORMANCE_SETTINGS_EXTENSION_NAME);
        if (b.perf_settings_ext) exts.push_back(XR_EXT_PERFORMANCE_SETTINGS_EXTENSION_NAME);
    }

    XrInstanceCreateInfoAndroidKHR icia{XR_TYPE_INSTANCE_CREATE_INFO_ANDROID_KHR};
    icia.applicationVM = vm;
    icia.applicationActivity = activity;

    XrInstanceCreateInfo ici{XR_TYPE_INSTANCE_CREATE_INFO};
    ici.next = &icia;
    std::snprintf(ici.applicationInfo.applicationName,
                  sizeof(ici.applicationInfo.applicationName), "%s", b.title.c_str());
    ici.applicationInfo.applicationVersion = 1;
    std::snprintf(ici.applicationInfo.engineName,
                  sizeof(ici.applicationInfo.engineName), "gbarecomp");
    ici.applicationInfo.engineVersion = 1;
    ici.applicationInfo.apiVersion = XR_MAKE_VERSION(1, 0, 26);
    ici.enabledExtensionCount = static_cast<uint32_t>(exts.size());
    ici.enabledExtensionNames = exts.data();

    const XrResult ires = xrCreateInstance(&ici, &b.instance);
    if (env && owns_local_activity) env->DeleteLocalRef(activity);
    if (XR_FAILED(ires)) {
        XR_LOGE("xrCreateInstance failed: %d", ires);
        return false;
    }

    XrSystemGetInfo sgi{XR_TYPE_SYSTEM_GET_INFO};
    sgi.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;
    if (XR_FAILED(xrGetSystem(b.instance, &sgi, &b.system_id))) {
        XR_LOGE("xrGetSystem failed");
        return false;
    }

    // OpenXR validation requires querying OpenGL ES graphics requirements before
    // xrCreateSession.
    PFN_xrGetOpenGLESGraphicsRequirementsKHR pfnGetReq = nullptr;
    xrGetInstanceProcAddr(b.instance, "xrGetOpenGLESGraphicsRequirementsKHR",
                          reinterpret_cast<PFN_xrVoidFunction*>(&pfnGetReq));
    if (pfnGetReq) {
        XrGraphicsRequirementsOpenGLESKHR req{XR_TYPE_GRAPHICS_REQUIREMENTS_OPENGL_ES_KHR};
        pfnGetReq(b.instance, b.system_id, &req);
    }

    XrGraphicsBindingOpenGLESAndroidKHR gles{XR_TYPE_GRAPHICS_BINDING_OPENGL_ES_ANDROID_KHR};
    gles.display = b.egl_display;
    gles.config = b.egl_config;
    gles.context = b.egl_context;

    XrSessionCreateInfo sci{XR_TYPE_SESSION_CREATE_INFO};
    sci.next = &gles;
    sci.systemId = b.system_id;
    if (XR_FAILED(xrCreateSession(b.instance, &sci, &b.session))) {
        XR_LOGE("xrCreateSession failed");
        return false;
    }

    XrReferenceSpaceCreateInfo lsci{XR_TYPE_REFERENCE_SPACE_CREATE_INFO};
    lsci.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_LOCAL;
    lsci.poseInReferenceSpace = {{0, 0, 0, 1}, {0, 0, 0}};
    xrCreateReferenceSpace(b.session, &lsci, &b.local_space);

    XrReferenceSpaceCreateInfo vsci{XR_TYPE_REFERENCE_SPACE_CREATE_INFO};
    vsci.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_VIEW;
    vsci.poseInReferenceSpace = {{0, 0, 0, 1}, {0, 0, 0}};
    xrCreateReferenceSpace(b.session, &vsci, &b.view_space);

    // Check passthrough capability cleanly on Quest
    if (b.passthrough_ext) {
        XrSystemPassthroughProperties2FB pt_props{XR_TYPE_SYSTEM_PASSTHROUGH_PROPERTIES2_FB};
        XrSystemProperties sys_props{XR_TYPE_SYSTEM_PROPERTIES, &pt_props};
        if (XR_SUCCEEDED(xrGetSystemProperties(b.instance, b.system_id, &sys_props)) &&
            (pt_props.capabilities & XR_PASSTHROUGH_CAPABILITY_BIT_FB)) {
            xrGetInstanceProcAddr(b.instance, "xrCreatePassthroughFB",
                                  reinterpret_cast<PFN_xrVoidFunction*>(&b.pfnCreatePassthroughFB));
            xrGetInstanceProcAddr(b.instance, "xrDestroyPassthroughFB",
                                  reinterpret_cast<PFN_xrVoidFunction*>(&b.pfnDestroyPassthroughFB));
            xrGetInstanceProcAddr(b.instance, "xrPassthroughStartFB",
                                  reinterpret_cast<PFN_xrVoidFunction*>(&b.pfnPassthroughStartFB));
            xrGetInstanceProcAddr(b.instance, "xrPassthroughPauseFB",
                                  reinterpret_cast<PFN_xrVoidFunction*>(&b.pfnPassthroughPauseFB));
            xrGetInstanceProcAddr(b.instance, "xrCreatePassthroughLayerFB",
                                  reinterpret_cast<PFN_xrVoidFunction*>(&b.pfnCreatePassthroughLayerFB));
            xrGetInstanceProcAddr(b.instance, "xrDestroyPassthroughLayerFB",
                                  reinterpret_cast<PFN_xrVoidFunction*>(&b.pfnDestroyPassthroughLayerFB));
            xrGetInstanceProcAddr(b.instance, "xrPassthroughLayerPauseFB",
                                  reinterpret_cast<PFN_xrVoidFunction*>(&b.pfnPassthroughLayerPauseFB));
            xrGetInstanceProcAddr(b.instance, "xrPassthroughLayerResumeFB",
                                  reinterpret_cast<PFN_xrVoidFunction*>(&b.pfnPassthroughLayerResumeFB));
            if (b.pfnCreatePassthroughFB && b.pfnCreatePassthroughLayerFB) {
                XrPassthroughCreateInfoFB pci{XR_TYPE_PASSTHROUGH_CREATE_INFO_FB};
                if (XR_SUCCEEDED(b.pfnCreatePassthroughFB(b.session, &pci, &b.passthrough))) {
                    XrPassthroughLayerCreateInfoFB plci{XR_TYPE_PASSTHROUGH_LAYER_CREATE_INFO_FB};
                    plci.passthrough = b.passthrough;
                    plci.purpose = XR_PASSTHROUGH_LAYER_PURPOSE_RECONSTRUCTION_FB;
                    if (XR_SUCCEEDED(b.pfnCreatePassthroughLayerFB(
                            b.session, &plci, &b.passthrough_layer))) {
                        b.passthrough_supported = true;
                        b.passthrough_active = !b.start_dark; // MR on by default unless GBARECOMP_XR_DARK=1
                        if (b.start_dark) b.hud_mode = 2;
                    }
                }
            }
        }
    }

    if (has_refresh) {
        xrGetInstanceProcAddr(b.instance, "xrRequestDisplayRefreshRateFB",
                              reinterpret_cast<PFN_xrVoidFunction*>(&b.pfnRequestDisplayRefreshRateFB));
        if (b.pfnRequestDisplayRefreshRateFB) {
            // 60 Hz gives clean 1:1 cadence with GBA's 59.73 Hz VBlank if supported.
            b.pfnRequestDisplayRefreshRateFB(b.session, 60.0f);
        }
    }

    const uint32_t scale = target_swapchain_scale(b.base_w);
    b.swap_w = static_cast<uint32_t>(b.base_w + 2 * b.pad) * scale;
    b.swap_h = static_cast<uint32_t>(b.base_h + 2 * b.pad) * scale;
    for (unsigned p = 0; p < kMaskedQuadCount; ++p) {
        if (!create_swapchain_with_images(b, b.swap_w, b.swap_h,
                                          b.screen_swapchains[p], b.screen_images[p])) {
            return false;
        }
    }
    if (!create_swapchain_with_images(b, kHudWidth, kHudHeight, b.hud_swapchain, b.hud_images)) {
        XR_LOGW("HUD swapchain creation failed; continuing without HUD quad");
    }
    if (b.menu_enabled &&
        !create_swapchain_with_images(b, kMenuWidth, kMenuHeight, b.menu_swapchain, b.menu_images)) {
        XR_LOGW("menu swapchain creation failed; options menu disabled");
        b.menu_enabled = false;
    }
    allocate_source_texture(b, b.base_w + 2 * b.pad, b.base_h + 2 * b.pad);
    if ((b.pad > 0 || b.box_walls) && !create_swapchain_with_images(b, 16, 16, b.frame_swapchain, b.frame_images)) {
        XR_LOGW("frame swapchain creation failed; shadow box frame disabled");
    }
    b.rgba_fb.assign(static_cast<size_t>(b.base_w) * b.base_h * 4, 255);

    if (!init_actions(b)) return false;

    XR_LOGI("OpenXR initialized (swapchain=%ux%u srgb=%s quads=%u passthrough=%s)",
            b.swap_w, b.swap_h,
            b.swapchain_srgb ? "yes" : "no",
            kMaskedQuadCount,
            b.passthrough_supported ? "yes" : "no");
    return true;
}

void shutdown_openxr(Backend& b) {
    for (int i = 0; i < 2; ++i) {
        if (b.aim_spaces[i] != XR_NULL_HANDLE) {
            xrDestroySpace(b.aim_spaces[i]);
            b.aim_spaces[i] = XR_NULL_HANDLE;
        }
    }
    if (b.action_set != XR_NULL_HANDLE) {
        xrDestroyActionSet(b.action_set);
        b.action_set = XR_NULL_HANDLE;
    }
    if (b.frame_swapchain != XR_NULL_HANDLE) {
        xrDestroySwapchain(b.frame_swapchain);
        b.frame_swapchain = XR_NULL_HANDLE;
    }
    if (b.menu_swapchain != XR_NULL_HANDLE) {
        xrDestroySwapchain(b.menu_swapchain);
        b.menu_swapchain = XR_NULL_HANDLE;
    }
    if (b.hud_swapchain != XR_NULL_HANDLE) {
        xrDestroySwapchain(b.hud_swapchain);
        b.hud_swapchain = XR_NULL_HANDLE;
    }
    for (int i = 0; i < 2; ++i) {
        if (b.proj_swapchains[i] != XR_NULL_HANDLE) {
            xrDestroySwapchain(b.proj_swapchains[i]);
            b.proj_swapchains[i] = XR_NULL_HANDLE;
        }
        b.proj_images[i].clear();
    }
    for (unsigned p = 0; p < kMaskedQuadCount; ++p) {
        if (b.screen_swapchains[p] != XR_NULL_HANDLE) {
            xrDestroySwapchain(b.screen_swapchains[p]);
            b.screen_swapchains[p] = XR_NULL_HANDLE;
        }
    }
    if (b.passthrough_layer != XR_NULL_HANDLE && b.pfnDestroyPassthroughLayerFB) {
        b.pfnDestroyPassthroughLayerFB(b.passthrough_layer);
        b.passthrough_layer = XR_NULL_HANDLE;
    }
    if (b.passthrough != XR_NULL_HANDLE && b.pfnDestroyPassthroughFB) {
        b.pfnDestroyPassthroughFB(b.passthrough);
        b.passthrough = XR_NULL_HANDLE;
    }
    if (b.view_space != XR_NULL_HANDLE) {
        xrDestroySpace(b.view_space);
        b.view_space = XR_NULL_HANDLE;
    }
    if (b.local_space != XR_NULL_HANDLE) {
        xrDestroySpace(b.local_space);
        b.local_space = XR_NULL_HANDLE;
    }
    if (b.session != XR_NULL_HANDLE) {
        if (b.session_running) xrEndSession(b.session);
        xrDestroySession(b.session);
        b.session = XR_NULL_HANDLE;
    }
    if (b.instance != XR_NULL_HANDLE) {
        xrDestroyInstance(b.instance);
        b.instance = XR_NULL_HANDLE;
    }
}

float depth_step_meters(int depth_mode) {
    if (depth_mode == 1) return 0.06f;  // STOCK 3D default spacing (11 cm variant handled by caller)
    if (depth_mode == 2) return 0.065f;
    return 0.0f;
}

bool blit_rgba_to_swapchain(Backend& b, unsigned plane_idx, int iw, int ih,
                            const uint8_t* bottom_up_rgba, bool nearest_only) {
    if (plane_idx >= kMaskedQuadCount || b.screen_swapchains[plane_idx] == XR_NULL_HANDLE) {
        return false;
    }
    glBindTexture(GL_TEXTURE_2D, b.src_tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, iw, ih,
                    GL_RGBA, GL_UNSIGNED_BYTE, bottom_up_rgba);
    glBindTexture(GL_TEXTURE_2D, 0);

    uint32_t sc_idx = 0;
    XrSwapchainImageAcquireInfo acq{XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO};
    if (XR_FAILED(xrAcquireSwapchainImage(b.screen_swapchains[plane_idx], &acq, &sc_idx))) {
        return false;
    }
    XrSwapchainImageWaitInfo wait{XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO};
    wait.timeout = XR_INFINITE_DURATION;
    bool ok = false;
    if (XR_SUCCEEDED(xrWaitSwapchainImage(b.screen_swapchains[plane_idx], &wait))) {
        glDisable(GL_SCISSOR_TEST);
        glDisable(GL_BLEND);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, b.src_fbo);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, b.dst_fbo);
        glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D, b.screen_images[plane_idx][sc_idx].image, 0);
        const GLenum filter = (!nearest_only && b.linear_filter.load(std::memory_order_relaxed))
                                  ? GL_LINEAR
                                  : GL_NEAREST;
        GLint dx0 = 0, dy0 = 0, dx1 = static_cast<GLint>(b.swap_w), dy1 = static_cast<GLint>(b.swap_h);
        if (b.pad > 0 && iw == b.base_w && ih == b.base_h) {
            // Unpadded image (2D flat, overlays) into a padded swapchain: centre, clear margins.
            const GLint sc = static_cast<GLint>(b.swap_w) / (iw + 2 * b.pad);
            glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
            glClear(GL_COLOR_BUFFER_BIT);
            dx0 = b.pad * sc; dy0 = b.pad * sc; dx1 = dx0 + iw * sc; dy1 = dy0 + ih * sc;
        }
        glBlitFramebuffer(0, 0, iw, ih, dx0, dy0, dx1, dy1, GL_COLOR_BUFFER_BIT, filter);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
        ok = true;
    }
    XrSwapchainImageReleaseInfo rel{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
    xrReleaseSwapchainImage(b.screen_swapchains[plane_idx], &rel);
    return ok;
}

// Battle backgrounds: redraw the swirling BG layers at swapchain resolution
// (gbarecomp/src/runtime/hd_text_bg.h) and upload 1:1. `native_topdown` is the
// plane as staged, `b.rgba_fb` must already hold its graded, flipped copy.
// Returns false (nothing uploaded) when this plane has no battle background.
bool blit_hd_text_bg(Backend& b, unsigned plane_idx, const uint8_t* native_topdown, int iw, int ih) {
    if (!b.xr_hdbg_ok || !b.xr_hdbg.active || iw <= 0 || ih <= 0) return false;
    if (plane_idx >= kMaskedQuadCount || b.screen_swapchains[plane_idx] == XR_NULL_HANDLE) return false;
    if (b.swap_w % static_cast<uint32_t>(iw) != 0 || b.swap_h != b.swap_w / iw * static_cast<uint32_t>(ih))
        return false;
    const int S = static_cast<int>(b.swap_w / iw);
    if (!b.hd_lut_ready) {
        b.hd_lut555.resize(32768);
        std::vector<uint8_t> rgb(32768 * 3), graded(32768 * 3);
        for (uint32_t c = 0; c < 32768; ++c) gba::color::to_rgb888(static_cast<uint16_t>(c), &rgb[c * 3]);
        if (b.color_lut && !b.color_lut->is_passthrough())
            b.color_lut->map_rgb888(rgb.data(), graded.data(), 32768, 1);
        else
            graded = rgb;
        for (uint32_t c = 0; c < 32768; ++c)
            b.hd_lut555[c] = graded[c * 3] | (graded[c * 3 + 1] << 8) | (graded[c * 3 + 2] << 16) | 0xFF000000u;
        b.hd_lut_ready = true;
    }
    const uint32_t W = b.swap_w, H = b.swap_h;
    b.hd_fb.resize(static_cast<size_t>(W) * H * 4);
    if (!b.hd_renderer.render(b.xr_hdbg, native_topdown, b.rgba_fb.data(), iw, ih, S, b.hd_lut555.data(),
                              b.hd_fb.data()))
        return false;
    if (!b.hd_tex) {
        glGenTextures(1, &b.hd_tex);
        glGenFramebuffers(1, &b.hd_fbo);
    }
    if (b.hd_tex_w != W || b.hd_tex_h != H) {
        glBindTexture(GL_TEXTURE_2D, b.hd_tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, b.swapchain_srgb ? GL_SRGB8_ALPHA8 : GL_RGBA8, static_cast<GLsizei>(W),
                     static_cast<GLsizei>(H), 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, b.hd_fbo);
        glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, b.hd_tex, 0);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
        b.hd_tex_w = W;
        b.hd_tex_h = H;
    }
    glBindTexture(GL_TEXTURE_2D, b.hd_tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, static_cast<GLsizei>(W), static_cast<GLsizei>(H), GL_RGBA,
                    GL_UNSIGNED_BYTE, b.hd_fb.data());
    glBindTexture(GL_TEXTURE_2D, 0);

    uint32_t sc_idx = 0;
    XrSwapchainImageAcquireInfo acq{XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO};
    if (XR_FAILED(xrAcquireSwapchainImage(b.screen_swapchains[plane_idx], &acq, &sc_idx))) return false;
    XrSwapchainImageWaitInfo wait{XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO};
    wait.timeout = XR_INFINITE_DURATION;
    bool ok = false;
    if (XR_SUCCEEDED(xrWaitSwapchainImage(b.screen_swapchains[plane_idx], &wait))) {
        glDisable(GL_SCISSOR_TEST);
        glDisable(GL_BLEND);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, b.hd_fbo);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, b.dst_fbo);
        glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                               b.screen_images[plane_idx][sc_idx].image, 0);
        glBlitFramebuffer(0, 0, static_cast<GLint>(W), static_cast<GLint>(H), 0, 0, static_cast<GLint>(W),
                          static_cast<GLint>(H), GL_COLOR_BUFFER_BIT, GL_NEAREST);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
        ok = true;
    }
    XrSwapchainImageReleaseInfo rel{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
    xrReleaseSwapchainImage(b.screen_swapchains[plane_idx], &rel);
    return ok;
}

void flip_and_grade_rgba_plane(Backend& b, const uint8_t* plane_rgba, int iw, int ih,
                               bool* out_has_opaque, bool premultiply = true) {
    const size_t px_count = static_cast<size_t>(iw) * ih;
    b.rgba_fb.resize(px_count * 4);
    const bool use_lut = (b.color_lut && !b.color_lut->is_passthrough());
    size_t vis_count = 0;
    if (use_lut) {
        b.color_lut_rgb.clear();
        for (size_t i = 0; i < px_count; ++i) {
            const uint8_t a = plane_rgba[i * 4 + 3];
            if (a == 0 && !(gba::g_layered_fog_alpha && (plane_rgba[i * 4] | plane_rgba[i * 4 + 1] | plane_rgba[i * 4 + 2])))
                continue;
            b.color_lut_rgb.push_back(plane_rgba[i * 4 + 0]);
            b.color_lut_rgb.push_back(plane_rgba[i * 4 + 1]);
            b.color_lut_rgb.push_back(plane_rgba[i * 4 + 2]);
            ++vis_count;
        }
        if (vis_count == 0) {
            if (out_has_opaque) *out_has_opaque = false;
            return;
        }
        b.graded_fb.resize(vis_count * 3);
        b.color_lut->map_rgb888(b.color_lut_rgb.data(), b.graded_fb.data(),
                                static_cast<int>(vis_count), 1);
    }

    const uint8_t* graded_ptr = use_lut ? b.graded_fb.data() : nullptr;
    size_t graded_idx = 0;
    bool any_visible = false;
    for (int y = 0; y < ih; ++y) {
        const size_t in_row_px = static_cast<size_t>(y) * iw;
        uint8_t* out_row = b.rgba_fb.data() +
                           static_cast<size_t>(ih - 1 - y) * iw * 4;
        for (int x = 0; x < iw; ++x) {
            const size_t p_off = (in_row_px + x) * 4;
            const uint8_t a = plane_rgba[p_off + 3];
            // Premultiplied planes (gba::g_layered_fog_alpha): alpha 0 with colour = additive glow.
            const bool premul_in = gba::g_layered_fog_alpha != 0;
            if (a == 0 && !(premul_in && (plane_rgba[p_off] | plane_rgba[p_off + 1] | plane_rgba[p_off + 2]))) {
                out_row[x * 4 + 0] = 0;
                out_row[x * 4 + 1] = 0;
                out_row[x * 4 + 2] = 0;
                out_row[x * 4 + 3] = 0;
                continue;
            }
            any_visible = true;
            uint8_t r = graded_ptr ? graded_ptr[graded_idx * 3 + 0] : plane_rgba[p_off + 0];
            uint8_t g = graded_ptr ? graded_ptr[graded_idx * 3 + 1] : plane_rgba[p_off + 1];
            uint8_t bl = graded_ptr ? graded_ptr[graded_idx * 3 + 2] : plane_rgba[p_off + 2];
            if (graded_ptr) ++graded_idx;
            if (premultiply && !premul_in && a < 255) {
                r = premul_channel(r, a, b.swapchain_srgb);
                g = premul_channel(g, a, b.swapchain_srgb);
                bl = premul_channel(bl, a, b.swapchain_srgb);
            }
            out_row[x * 4 + 0] = r;
            out_row[x * 4 + 1] = g;
            out_row[x * 4 + 2] = bl;
            out_row[x * 4 + 3] = a;
        }
    }
    if (out_has_opaque) *out_has_opaque = any_visible;
}

// Split Plane 3 (world sprites) into 3 pure-OpenXR parallel actor quads (P3a/P3b/P3c)
// anchored by each connected sprite silhouette's bottom-most foot_y coordinate relative to
// the active P1 floor vertical span, promoting any sprite that occludes P2 props to >= P3b.
[[maybe_unused]] void split_sprites_by_foot_y(Backend& b,
                             const uint8_t* p1_rgba,
                             const uint8_t* p2_rgba,
                             const uint8_t* p3_rgba,
                             int iw,
                             int ih) {
    b.p3_band_any[0] = b.p3_band_any[1] = b.p3_band_any[2] = false;
    if (!p3_rgba || iw <= 0 || ih <= 0) return;
    const size_t px_count = static_cast<size_t>(iw) * ih;
    bool any_p3 = false;
    for (size_t i = 0; i < px_count; ++i) {
        if (p3_rgba[i * 4 + 3] != 0) {
            any_p3 = true;
            break;
        }
    }
    if (!any_p3) return;

    const size_t plane_bytes = px_count * 4;
    for (int k = 0; k < 3; ++k) {
        b.p3_bands[k].assign(plane_bytes, 0);
    }
    b.p3_vis.assign(px_count, 0);

    // Measure active walkable stage (P1) vertical bounds so indoor rooms surrounded by
    // black void (e.g. Map 121 Yado Inn where floor spans y=0..99) split across the room's
    // actual floor height rather than the raw 180px framebuffer height.
    int floor_y_min = ih;
    int floor_y_max = -1;
    if (p1_rgba) {
        const int min_row_px = std::max(16, iw / 4);
        for (int y = 0; y < ih; ++y) {
            const uint8_t* row = p1_rgba + static_cast<size_t>(y) * iw * 4;
            int cnt = 0;
            for (int x = 0; x < iw; ++x) {
                if (row[x * 4 + 3] != 0) ++cnt;
            }
            if (cnt >= min_row_px) {
                if (y < floor_y_min) floor_y_min = y;
                if (y > floor_y_max) floor_y_max = y;
            }
        }
    }
    if (floor_y_max <= floor_y_min + 24) {
        floor_y_min = 0;
        floor_y_max = ih - 1;
    }
    const int floor_span = std::max(1, floor_y_max - floor_y_min);
    const int y_back_max = floor_y_min + (floor_span * 36 + 50) / 100;
    const int y_mid_max  = floor_y_min + (floor_span * 68 + 50) / 100;

    for (int y = 0; y < ih; ++y) {
        for (int x = 0; x < iw; ++x) {
            const size_t start = static_cast<size_t>(y) * iw + x;
            if (p3_rgba[start * 4 + 3] == 0 || b.p3_vis[start]) continue;
            b.p3_vis[start] = 1;
            b.p3_bfs_q.clear();
            b.p3_comp.clear();
            b.p3_bfs_q.push_back(static_cast<int>(start));
            size_t head = 0;
            int max_y = y;
            bool overlaps_p2 = false;
            while (head < b.p3_bfs_q.size()) {
                const int idx = b.p3_bfs_q[head++];
                b.p3_comp.push_back(idx);
                const int cy = idx / iw;
                const int cx = idx - cy * iw;
                if (cy > max_y) max_y = cy;
                if (p2_rgba && p2_rgba[static_cast<size_t>(idx) * 4 + 3] != 0) {
                    overlaps_p2 = true;
                }
                for (int dy = -2; dy <= 2; ++dy) {
                    const int ny = cy + dy;
                    if (ny < 0 || ny >= ih) continue;
                    for (int dx = -2; dx <= 2; ++dx) {
                        const int nx = cx + dx;
                        if (nx < 0 || nx >= iw) continue;
                        const size_t nidx = static_cast<size_t>(ny) * iw + nx;
                        if (!b.p3_vis[nidx] && p3_rgba[nidx * 4 + 3] != 0) {
                            b.p3_vis[nidx] = 1;
                            b.p3_bfs_q.push_back(static_cast<int>(nidx));
                        }
                    }
                }
            }
            int band = (max_y <= y_back_max) ? 0 : ((max_y <= y_mid_max) ? 1 : 2);
            // If an upper-band sprite occludes P2 furniture/props in front of it, promote to P3b
            // so it renders in front of Quad 3 (P2). Sprites behind P2 counters stay in P3a (Quad 2).
            if (band == 0 && overlaps_p2) band = 1;
            uint8_t* dst = b.p3_bands[band].data();
            for (int pi : b.p3_comp) {
                std::memcpy(&dst[static_cast<size_t>(pi) * 4], &p3_rgba[static_cast<size_t>(pi) * 4], 4);
            }
            b.p3_band_any[band] = true;
        }
    }
}

void poll_xr_actions(Backend& b) {
    if (!b.actions_attached || b.session_state != XR_SESSION_STATE_FOCUSED) {
        b.latest_keyinput.store(0x03FFu, std::memory_order_release);
        b.latest_gyro_rate_z.store(0.0f, std::memory_order_release);
        return;
    }

    XrActiveActionSet active{b.action_set, XR_NULL_PATH};
    XrActionsSyncInfo sync{XR_TYPE_ACTIONS_SYNC_INFO};
    sync.countActiveActionSets = 1;
    sync.activeActionSets = &active;
    if (XR_FAILED(xrSyncActions(b.session, &sync))) {
        return;
    }

    auto read_bool = [&](XrAction action, XrPath sub = XR_NULL_PATH) -> bool {
        XrActionStateGetInfo gi{XR_TYPE_ACTION_STATE_GET_INFO};
        gi.action = action;
        gi.subactionPath = sub;
        XrActionStateBoolean st{XR_TYPE_ACTION_STATE_BOOLEAN};
        if (XR_SUCCEEDED(xrGetActionStateBoolean(b.session, &gi, &st)) &&
            st.isActive && st.currentState) {
            return true;
        }
        return false;
    };
    auto read_float = [&](XrAction action, XrPath sub) -> float {
        XrActionStateGetInfo gi{XR_TYPE_ACTION_STATE_GET_INFO};
        gi.action = action;
        gi.subactionPath = sub;
        XrActionStateFloat st{XR_TYPE_ACTION_STATE_FLOAT};
        if (XR_SUCCEEDED(xrGetActionStateFloat(b.session, &gi, &st)) && st.isActive) {
            return st.currentState;
        }
        return 0.0f;
    };
    auto read_vec2 = [&](XrAction action, XrPath sub) -> XrVector2f {
        XrActionStateGetInfo gi{XR_TYPE_ACTION_STATE_GET_INFO};
        gi.action = action;
        gi.subactionPath = sub;
        XrActionStateVector2f st{XR_TYPE_ACTION_STATE_VECTOR2F};
        if (XR_SUCCEEDED(xrGetActionStateVector2f(b.session, &gi, &st)) && st.isActive) {
            return st.currentState;
        }
        return {0.0f, 0.0f};
    };

    const XrVector2f lstick = read_vec2(b.act_thumbstick, b.hand_paths[0]);
    const XrVector2f rstick = read_vec2(b.act_thumbstick, b.hand_paths[1]);
    const float ltrig = read_float(b.act_trigger, b.hand_paths[0]);
    const float rtrig = read_float(b.act_trigger, b.hand_paths[1]);
    const float lgrip = read_float(b.act_squeeze, b.hand_paths[0]);
    const float rgrip = read_float(b.act_squeeze, b.hand_paths[1]);
    const bool btn_a = read_bool(b.act_a);
    const bool btn_b = read_bool(b.act_b);
    const bool btn_x = read_bool(b.act_x);
    const bool btn_y = read_bool(b.act_y);
    const bool l_click = read_bool(b.act_thumb_click, b.hand_paths[0]);
    const bool r_click = read_bool(b.act_thumb_click, b.hand_paths[1]);
    const bool l_menu = read_bool(b.act_menu);

    const bool both_grips = (lgrip > kGripThreshold) && (rgrip > kGripThreshold);
    const XrTime pose_time = b.last_predicted_time > 0 ? b.last_predicted_time : 1;
    const float max_stick_y = (std::abs(lstick.y) > std::abs(rstick.y)) ? lstick.y : rstick.y;
    update_two_grip_window_move(b, pose_time, both_grips, max_stick_y);

    // GBA KEYINPUT active-low: 0=A 1=B 2=Sel 3=Sta 4=Right 5=Left 6=Up 7=Down 8=R 9=L
    uint16_t keys = 0x03FFu;
    if (btn_a || rtrig > kTriggerThreshold)               keys &= ~(1u << 0);
    if (btn_b || (!both_grips && rgrip > kGripThreshold)) keys &= ~(1u << 1);
    if (btn_y)                                            keys &= ~(1u << 2);
    if (btn_x)                                            keys &= ~(1u << 3);
    if (!both_grips) {
        if (lstick.x > kThumbstickDeadzone || rstick.x > kThumbstickDeadzone)   keys &= ~(1u << 4);
        if (lstick.x < -kThumbstickDeadzone || rstick.x < -kThumbstickDeadzone) keys &= ~(1u << 5);
        if (lstick.y > kThumbstickDeadzone || rstick.y > kThumbstickDeadzone)   keys &= ~(1u << 6);
        if (lstick.y < -kThumbstickDeadzone || rstick.y < -kThumbstickDeadzone) keys &= ~(1u << 7);
        if (lgrip > kGripThreshold)                                              keys &= ~(1u << 8);
    }
    if (ltrig > kTriggerThreshold)                                               keys &= ~(1u << 9);

    // VR options menu: the Menu button opens/closes it (and pauses/resumes the game); while it
    // is open the stick moves the selection / changes values, A selects, B closes, and the game
    // gets no input.
    if (b.menu_enabled) {
        if (l_menu && !b.prev_l_menu_click) {
            b.menu_open = !b.menu_open;
            b.menu_status.clear();
            b.menu_dirty = true;
            b.pending_pause_toggles.fetch_add(1, std::memory_order_acq_rel);
            trigger_haptic_ns(b, 0, 20000000LL, 0.35f);
            b.menu_prev_a = btn_a || rtrig > kTriggerThreshold;
            b.menu_prev_b = btn_b;
        }
        if (b.menu_open) {
            const float sy = std::abs(lstick.y) > std::abs(rstick.y) ? lstick.y : rstick.y;
            const float sx = std::abs(lstick.x) > std::abs(rstick.x) ? lstick.x : rstick.x;
            const int dir_y = sy > 0.6f ? 1 : (sy < -0.6f ? -1 : 0);
            const int dir_x = sx > 0.6f ? 1 : (sx < -0.6f ? -1 : 0);
            const bool a_now = btn_a || rtrig > kTriggerThreshold;
            const bool b_now = btn_b;
            auto close_menu = [&]() {
                b.menu_open = false;
                b.pending_pause_toggles.fetch_add(1, std::memory_order_acq_rel);
            };
            if (dir_y != 0 && dir_y != b.menu_prev_dir_y) {
                do {
                    b.menu_sel = (b.menu_sel + (dir_y > 0 ? kMenuCount - 1 : 1)) % kMenuCount;
                } while (!menu_item_visible(b, b.menu_sel));
                b.menu_status.clear();
                b.menu_dirty = true;
                trigger_haptic_ns(b, 1, 8000000LL, 0.2f);
            }
            if (dir_x != 0 && dir_x != b.menu_prev_dir_x) {
                if (b.menu_sel == kMenuSave || b.menu_sel == kMenuLoad) {
                    b.menu_slot = (b.menu_slot - 1 + (dir_x > 0 ? 1 : 2)) % 3 + 1;
                } else if (b.menu_sel == kMenuGyro) {
                    b.gyro_sens_idx = std::clamp(b.gyro_sens_idx + (dir_x > 0 ? 1 : -1), 0, kGyroSensCount - 1);
                    save_xr_prefs(b);
                } else if (b.menu_sel == kMenuGyroHand) {
                    b.active_gyro_hand = 1 - b.active_gyro_hand;
                    trigger_haptic_ns(b, b.active_gyro_hand, 30000000LL, 0.5f);
                    save_xr_prefs(b);
                } else if (b.menu_sel == kMenuDepth) {
                    const int d = (menu_depth_index(b) + (dir_x > 0 ? 1 : 2)) % 3;
                    b.depth_mode = d == 2 ? 0 : 1;
                    b.stock_deep = d == 1;
                    b.hud_dirty = true;
                    b.force_reblit = true;
                } else if (b.menu_sel == kMenuAutoResume) {
                    b.auto_resume = !b.auto_resume;
                    save_xr_prefs(b);
                } else if (b.menu_sel == kMenuEdges) {
                    b.window_clip = !b.window_clip;
                    save_xr_prefs(b);
                } else if (b.menu_sel == kMenuBackground) {
                    if (b.passthrough_supported) {
                        b.hud_mode = b.hud_mode == 2 ? 1 : 2;  // ROOM <-> DARK
                        b.passthrough_active = (b.hud_mode != 2);
                        apply_passthrough_state(b);
                        save_xr_prefs(b);
                    }
                    b.hud_dirty = true;
                }
                b.menu_status.clear();
                b.menu_dirty = true;
                trigger_haptic_ns(b, 1, 8000000LL, 0.2f);
            }
            if (a_now && !b.menu_prev_a) {
                char msg[64] = {};
                switch (b.menu_sel) {
                    case kMenuResume: close_menu(); break;
                    case kMenuSave:
                        b.pending_save_slot.store(b.menu_slot, std::memory_order_release);
                        std::snprintf(msg, sizeof(msg), "SAVED TO SLOT %d", b.menu_slot);
                        break;
                    case kMenuLoad:
                        b.pending_load_slot.store(b.menu_slot, std::memory_order_release);
                        std::snprintf(msg, sizeof(msg), "LOADED SLOT %d", b.menu_slot);
                        close_menu();
                        break;
                    case kMenuDepth: {
                        const int d = (menu_depth_index(b) + 1) % 3;
                        b.depth_mode = d == 2 ? 0 : 1;
                        b.stock_deep = d == 1;
                        b.hud_dirty = true;
                        b.force_reblit = true;
                        break;
                    }
                    case kMenuBackground:
                        if (b.passthrough_supported) {
                            b.hud_mode = b.hud_mode == 2 ? 1 : 2;  // ROOM <-> DARK
                            b.passthrough_active = (b.hud_mode != 2);
                            apply_passthrough_state(b);
                            save_xr_prefs(b);
                        }
                        b.hud_dirty = true;
                        break;
                    case kMenuAutoResume:
                        b.auto_resume = !b.auto_resume;
                        save_xr_prefs(b);
                        break;
                    case kMenuEdges:
                        b.window_clip = !b.window_clip;
                        save_xr_prefs(b);
                        break;
                    case kMenuGyro:
                        b.gyro_sens_idx = (b.gyro_sens_idx + 1) % kGyroSensCount;
                        save_xr_prefs(b);
                        break;
                    case kMenuGyroHand:
                        b.active_gyro_hand = 1 - b.active_gyro_hand;
                        trigger_haptic_ns(b, b.active_gyro_hand, 30000000LL, 0.5f);
                        save_xr_prefs(b);
                        break;
                    case kMenuRecenter:
                        b.recenter_after_time = 0;
                        recenter_screen_to_head(b, b.last_predicted_time > 0 ? b.last_predicted_time : 1);
                        std::snprintf(msg, sizeof(msg), "SCREEN RECENTERED");
                        break;
                    case kMenuResetView:
                        b.screen_distance = 1.65f;
                        b.scale = 3;
                        b.screen_width_m.store(0.85f + 0.19f * 3.0f, std::memory_order_release);
                        recenter_screen_to_head(b, b.last_predicted_time > 0 ? b.last_predicted_time : 1);
                        save_xr_view(b);
                        std::snprintf(msg, sizeof(msg), "SCREEN SIZE RESET");
                        break;
                    default: break;
                }
                b.menu_status = msg;
                b.menu_dirty = true;
                trigger_haptic_ns(b, 1, 20000000LL, 0.4f);
            }
            if (b_now && !b.menu_prev_b) close_menu();
            b.menu_prev_dir_y = dir_y;
            b.menu_prev_dir_x = dir_x;
            b.menu_prev_a = a_now;
            b.menu_prev_b = b_now;
            keys = 0x03FFu;
        }
    }
    b.latest_keyinput.store(keys, std::memory_order_release);

    // L-Stick Click (or L+R chord) -> Cycle [STOCK 3D 6 cm -> STOCK 3D 11 cm -> 2D FLAT]
    // R-Stick Click -> Recenter virtual screen to current head pose
    auto cycle_depth_mode = [&]() {
        // STOCK 3D 6 cm -> STOCK 3D 11 cm -> 2D FLAT -> ...
        if (b.depth_mode == 1 && !b.stock_deep) {
            b.stock_deep = true;
        } else if (b.depth_mode == 1) {
            b.depth_mode = 0;
        } else {
            b.depth_mode = 1;
            b.stock_deep = false;
        }
        const char* label = (b.depth_mode == 1) ? (b.stock_deep ? "STOCK 3D 11CM" : "STOCK 3D 6CM") : "2D FLAT";
        b.hud_dirty = true;
        b.force_reblit = true;
        trigger_haptic_ns(b, 0, 25000000LL, 0.45f);
        trigger_haptic_ns(b, 1, 25000000LL, 0.45f);
        XR_LOGI("3D Layered Depth mode -> %d (%s)", b.depth_mode, label);
    };
    // Hold R-Stick Click for 0.8 s -> capture the current frame's layers for offline review
    // (files/captures/<time>/; pulled with tools/quest_pull_captures.sh). Releasing after a
    // capture does not recenter.
    {
        const auto now = std::chrono::steady_clock::now();
        if (r_click && !b.prev_r_stick_click) b.r_click_down_at = now;
        if (r_click && !l_click && !b.stick_chord_consumed &&
            now - b.r_click_down_at > std::chrono::milliseconds(800)) {
            b.capture_requested.store(true, std::memory_order_release);
            b.stick_chord_consumed = true;
            trigger_haptic_ns(b, 0, 60000000LL, 0.7f);
            trigger_haptic_ns(b, 1, 60000000LL, 0.7f);
            XR_LOGI("capture requested (R-stick hold)");
        }
    }
    const bool both_clicks = l_click && r_click;
    if (both_clicks && !b.prev_both_stick_clicks) {
        if (!b.menu_enabled) cycle_depth_mode();  // the options menu owns the 3D mode
        b.stick_chord_consumed = true;
    }
    b.prev_both_stick_clicks = both_clicks;

    if (!l_click && !r_click) {
        if (!b.stick_chord_consumed) {
            if (b.prev_r_stick_click) {
                recenter_screen_to_head(b, pose_time);
                trigger_haptic_ns(b, 1, 20000000LL, 0.35f);
            } else if (b.prev_l_stick_click && !b.menu_enabled) {
                cycle_depth_mode();
            }
        }
        b.stick_chord_consumed = false;
    }
    b.prev_r_stick_click = r_click;
    b.prev_l_stick_click = l_click;

    // Left Menu -> Cycle [MR+HUD -> MR Clean -> Dark Void Clean] (or HUD on/off if no MR)
    if (l_menu && !b.prev_l_menu_click && !b.menu_enabled) {
        if (b.passthrough_supported) {
            b.hud_mode = (b.hud_mode + 1) % 3;
            b.passthrough_active = (b.hud_mode != 2);
            apply_passthrough_state(b);
        } else {
            b.hud_mode = (b.hud_mode == 0) ? 1 : 0;
        }
        b.hud_dirty = true;
        trigger_haptic_ns(b, 0, 20000000LL, 0.35f);
    }
    b.prev_l_menu_click = l_menu;

    if (b.gyro_enabled) {
        XrSpaceVelocity vel{XR_TYPE_SPACE_VELOCITY};
        XrSpaceLocation hand_loc{XR_TYPE_SPACE_LOCATION, &vel};
        const XrSpace active_space = b.aim_spaces[b.active_gyro_hand];
        if (!both_grips && !b.menu_open && active_space != XR_NULL_HANDLE &&
            XR_SUCCEEDED(xrLocateSpace(active_space, b.local_space, pose_time, &hand_loc)) &&
            (hand_loc.locationFlags & XR_SPACE_LOCATION_ORIENTATION_VALID_BIT) &&
            (vel.velocityFlags & XR_SPACE_VELOCITY_ANGULAR_VALID_BIT)) {
            // Forearm pronation/supination axis in controller aim space (tilted slightly toward
            // the handle so twisting works pointing at the screen, lying down or holding the
            // controller upright) + screen-normal projection for flat steering twists.
            const XrVector3f wrist_axis = v3_norm(
                rotate_vec(hand_loc.pose.orientation, 0.0f, -0.26f, 0.965f), {0.0f, 0.0f, 1.0f});
            const XrVector3f screen_normal = v3_norm(
                rotate_vec(b.screen_pose.orientation, 0.0f, 0.0f, 1.0f), wrist_axis);
            const float rate_wrist = v3_dot(vel.angularVelocity, wrist_axis);
            const float rate_screen = v3_dot(vel.angularVelocity, screen_normal);
            const float rate_z = (std::abs(rate_wrist) >= std::abs(rate_screen) * 0.75f)
                                     ? rate_wrist : rate_screen;
            const float filtered = (std::abs(rate_z) >= kGyroDriftDeadzoneRadPerSec) ? rate_z : 0.0f;
            b.latest_gyro_rate_z.store(filtered * kQuestGyroRateScale * kGyroSensSteps[b.gyro_sens_idx],
                                       std::memory_order_release);
            if (filtered != 0.0f && !b.gyro_motion_logged) {
                XR_LOGI("Quest Touch gyro active: hand=%s rate_z=%.3f rad/s",
                        b.active_gyro_hand == 1 ? "RIGHT" : "LEFT", static_cast<double>(rate_z));
                b.gyro_motion_logged = true;
            }
        } else {
            b.latest_gyro_rate_z.store(0.0f, std::memory_order_release);
        }
    }
}

void process_pending_haptics(Backend& b) {
    bool pulse = false;
    XrDuration pulse_ns = 0;
    float pulse_amp = 0.0f;
    bool rumble = false;
    {
        std::lock_guard<std::mutex> lk(b.haptic_mtx);
        if (b.haptic_pulse_queued) {
            pulse = true;
            pulse_ns = b.queued_haptic_ns;
            pulse_amp = b.queued_haptic_amp;
            b.haptic_pulse_queued = false;
        }
        rumble = b.desired_rumble;
    }
    if (pulse) {
        trigger_haptic_ns(b, 0, pulse_ns, pulse_amp);
        trigger_haptic_ns(b, 1, pulse_ns, pulse_amp);
    }
    const auto now = std::chrono::steady_clock::now();
    if (rumble) {
        if (!b.rumble_active ||
            (now - b.last_rumble_refresh) >= std::chrono::milliseconds(22)) {
            trigger_haptic_ns(b, 0, 40000000LL, 0.70f);
            trigger_haptic_ns(b, 1, 40000000LL, 0.70f);
            b.last_rumble_refresh = now;
        }
        b.rumble_active = true;
    } else if (b.rumble_active) {
        stop_haptic(b, 0);
        stop_haptic(b, 1);
        b.rumble_active = false;
    }
}

// ---------------------------------------------------------------- POP-UP 3D (projection layer)
bool ensure_popup_ready(Backend& b) {
    if (b.popup_init_failed) return false;
    if (!b.popup.ready() && !b.popup.init()) {
        XR_LOGE("POP-UP 3D: shader/GL init failed; falling back to flat");
        b.popup_init_failed = true;
        return false;
    }
    if (b.proj_swapchains[0] != XR_NULL_HANDLE) return true;
    uint32_t view_count = 0;
    xrEnumerateViewConfigurationViews(b.instance, b.system_id, XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO,
                                      0, &view_count, nullptr);
    if (view_count < 2) {
        b.popup_init_failed = true;
        return false;
    }
    std::vector<XrViewConfigurationView> views(view_count, {XR_TYPE_VIEW_CONFIGURATION_VIEW});
    xrEnumerateViewConfigurationViews(b.instance, b.system_id, XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO,
                                      view_count, &view_count, views.data());
    // Render above the recommended resolution (which is below the Quest 3 panel's density);
    // quads are sampled by the compositor at display density, so a 1.0x projection layer
    // looks visibly softer than STOCK 3D. 1.5x, capped at the runtime maximum.
    constexpr float kPopupRenderScale = 1.5f;
    b.proj_w = std::min(views[0].maxImageRectWidth,
                        static_cast<uint32_t>(views[0].recommendedImageRectWidth * kPopupRenderScale));
    b.proj_h = std::min(views[0].maxImageRectHeight,
                        static_cast<uint32_t>(views[0].recommendedImageRectHeight * kPopupRenderScale));
    for (int i = 0; i < 2; ++i) {
        if (!create_swapchain_with_images(b, b.proj_w, b.proj_h, b.proj_swapchains[i], b.proj_images[i])) {
            XR_LOGE("POP-UP 3D: projection swapchain creation failed");
            b.popup_init_failed = true;
            return false;
        }
    }
    XR_LOGI("POP-UP 3D ready: %ux%u per eye (recommended %ux%u), tilt %.1f deg, sharpening=%s", b.proj_w,
            b.proj_h, views[0].recommendedImageRectWidth, views[0].recommendedImageRectHeight, b.popup_tilt_deg,
            b.layer_settings_ext ? "yes" : "no");
    return true;
}

bool render_popup_projection(Backend& b, XrTime display_time, float screen_w, float screen_h,
                             XrCompositionLayerProjection& layer, XrCompositionLayerProjectionView pv[2]) {
    if (!ensure_popup_ready(b)) return false;
    XrViewLocateInfo li{XR_TYPE_VIEW_LOCATE_INFO};
    li.viewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
    li.displayTime = display_time;
    li.space = b.local_space;
    XrViewState vs{XR_TYPE_VIEW_STATE};
    XrView views[2] = {{XR_TYPE_VIEW}, {XR_TYPE_VIEW}};
    uint32_t n = 0;
    if (XR_FAILED(xrLocateViews(b.session, &li, &vs, 2, &n, views)) || n < 2 ||
        !(vs.viewStateFlags & XR_VIEW_STATE_POSITION_VALID_BIT) ||
        !(vs.viewStateFlags & XR_VIEW_STATE_ORIENTATION_VALID_BIT)) {
        return false;
    }
    const float eye_z = std::max(0.5f, b.screen_distance);
    b.popup.prepare(eye_z, screen_w, screen_h, b.popup_tilt_deg * 3.14159265f / 180.0f);

    float model[16];
    popup::Renderer::pose_matrix(model, b.screen_pose.orientation.x, b.screen_pose.orientation.y,
                                 b.screen_pose.orientation.z, b.screen_pose.orientation.w,
                                 b.screen_pose.position.x, b.screen_pose.position.y, b.screen_pose.position.z);
    for (int i = 0; i < 2; ++i) {
        uint32_t idx = 0;
        XrSwapchainImageAcquireInfo acq{XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO};
        if (XR_FAILED(xrAcquireSwapchainImage(b.proj_swapchains[i], &acq, &idx))) return false;
        XrSwapchainImageWaitInfo wi{XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO};
        wi.timeout = XR_INFINITE_DURATION;
        if (XR_SUCCEEDED(xrWaitSwapchainImage(b.proj_swapchains[i], &wi))) {
            float proj[16], eye[16], view[16], vm[16], mvp[16];
            const XrFovf& f = views[i].fov;
            popup::Renderer::projection_from_fov(proj, std::tan(f.angleLeft), std::tan(f.angleRight),
                                                 std::tan(f.angleUp), std::tan(f.angleDown), 0.05f, 100.0f);
            const XrPosef& p = views[i].pose;
            popup::Renderer::pose_matrix(eye, p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w,
                                         p.position.x, p.position.y, p.position.z);
            popup::Renderer::rigid_inverse(view, eye);
            popup::Renderer::mul(vm, view, model);
            popup::Renderer::mul(mvp, proj, vm);
            b.popup.draw_eye(b.proj_images[i][idx].image, static_cast<int>(b.proj_w),
                             static_cast<int>(b.proj_h), mvp);
        }
        XrSwapchainImageReleaseInfo rel{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
        xrReleaseSwapchainImage(b.proj_swapchains[i], &rel);

        pv[i] = {XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW};
        pv[i].pose = views[i].pose;
        pv[i].fov = views[i].fov;
        pv[i].subImage.swapchain = b.proj_swapchains[i];
        pv[i].subImage.imageRect = {{0, 0}, {static_cast<int32_t>(b.proj_w), static_cast<int32_t>(b.proj_h)}};
        pv[i].subImage.imageArrayIndex = 0;
    }
    layer = {XR_TYPE_COMPOSITION_LAYER_PROJECTION};
    layer.layerFlags = XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT |
                       XR_COMPOSITION_LAYER_CORRECT_CHROMATIC_ABERRATION_BIT;
    layer.space = b.local_space;
    layer.viewCount = 2;
    layer.views = pv;
    if (b.layer_settings_ext) {
        b.popup_layer_settings = {XR_TYPE_COMPOSITION_LAYER_SETTINGS_FB};
        b.popup_layer_settings.layerFlags = XR_COMPOSITION_LAYER_SETTINGS_QUALITY_SHARPENING_BIT_FB;
        layer.next = &b.popup_layer_settings;
    }
    return true;
}

// Far planes through the window (GBARECOMP_XR_WINDOW): per-eye projection layer.
bool render_window_projection(Backend& b, XrTime display_time, float screen_w, float aspect, float step_m,
                              const int* slots, const float* zs, int n,
                              XrCompositionLayerProjection& layer, XrCompositionLayerProjectionView pv[2]) {
    if (b.window_failed) return false;
    if (!ensure_popup_ready(b)) { b.window_failed = true; return false; }
    if (!b.winr.ready() && !b.winr.init()) {
        XR_LOGE("window renderer init failed; far planes stay quads");
        b.window_failed = true;
        return false;
    }
    XrViewLocateInfo li{XR_TYPE_VIEW_LOCATE_INFO};
    li.viewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
    li.displayTime = display_time;
    li.space = b.local_space;
    XrViewState vs{XR_TYPE_VIEW_STATE};
    XrView views[2] = {{XR_TYPE_VIEW}, {XR_TYPE_VIEW}};
    uint32_t nv = 0;
    if (XR_FAILED(xrLocateViews(b.session, &li, &vs, 2, &nv, views)) || nv < 2 ||
        !(vs.viewStateFlags & XR_VIEW_STATE_POSITION_VALID_BIT) ||
        !(vs.viewStateFlags & XR_VIEW_STATE_ORIENTATION_VALID_BIT)) {
        return false;
    }
    const float d_ref = std::max(0.5f, b.screen_distance);
    const float pad_w = b.pad > 0 ? static_cast<float>(b.base_w + 2 * b.pad) / b.base_w : 1.0f;
    const float pad_h = b.pad > 0 ? static_cast<float>(b.base_h + 2 * b.pad) / b.base_h : 1.0f;
    WindowRenderer::Plane planes[WindowRenderer::kSlots];
    int count = 0;
    for (int k = 0; k < n && count < WindowRenderer::kSlots; ++k) {
        const int p = slots[k];
        const float z = zs[k];
        const float sc = std::max(0.25f, d_ref - z) / d_ref;
        planes[count++] = {p, z, 0.5f * screen_w * sc * pad_w, 0.5f * screen_w * aspect * sc * pad_h};
    }
    const float z_win = 2.0f * step_m;  // the front plane: the window's opening
    const float s_win = std::max(0.25f, d_ref - z_win) / d_ref;
    const float win[3] = {0.5f * screen_w * s_win, 0.5f * screen_w * aspect * s_win, z_win};
    float model[16], model_inv[16];
    popup::Renderer::pose_matrix(model, b.screen_pose.orientation.x, b.screen_pose.orientation.y,
                                 b.screen_pose.orientation.z, b.screen_pose.orientation.w,
                                 b.screen_pose.position.x, b.screen_pose.position.y, b.screen_pose.position.z);
    popup::Renderer::rigid_inverse(model_inv, model);
    for (int i = 0; i < 2; ++i) {
        uint32_t idx = 0;
        XrSwapchainImageAcquireInfo acq{XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO};
        if (XR_FAILED(xrAcquireSwapchainImage(b.proj_swapchains[i], &acq, &idx))) return false;
        XrSwapchainImageWaitInfo wi{XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO};
        wi.timeout = XR_INFINITE_DURATION;
        if (XR_SUCCEEDED(xrWaitSwapchainImage(b.proj_swapchains[i], &wi))) {
            float proj[16], eye[16], view[16], vm[16], mvp[16];
            const XrFovf& f = views[i].fov;
            popup::Renderer::projection_from_fov(proj, std::tan(f.angleLeft), std::tan(f.angleRight),
                                                 std::tan(f.angleUp), std::tan(f.angleDown), 0.05f, 100.0f);
            const XrPosef& p = views[i].pose;
            popup::Renderer::pose_matrix(eye, p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w,
                                         p.position.x, p.position.y, p.position.z);
            popup::Renderer::rigid_inverse(view, eye);
            popup::Renderer::mul(vm, view, model);
            popup::Renderer::mul(mvp, proj, vm);
            const float ex = p.position.x, ey = p.position.y, ez = p.position.z;
            const float eye_local[3] = {
                model_inv[0] * ex + model_inv[4] * ey + model_inv[8] * ez + model_inv[12],
                model_inv[1] * ex + model_inv[5] * ey + model_inv[9] * ez + model_inv[13],
                model_inv[2] * ex + model_inv[6] * ey + model_inv[10] * ez + model_inv[14]};
            b.winr.draw_eye(b.proj_images[i][idx].image, static_cast<int>(b.proj_w), static_cast<int>(b.proj_h),
                            mvp, eye_local, win, planes, count);
        }
        XrSwapchainImageReleaseInfo rel{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
        xrReleaseSwapchainImage(b.proj_swapchains[i], &rel);
        pv[i] = {XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW};
        pv[i].pose = views[i].pose;
        pv[i].fov = views[i].fov;
        pv[i].subImage.swapchain = b.proj_swapchains[i];
        pv[i].subImage.imageRect = {{0, 0}, {static_cast<int32_t>(b.proj_w), static_cast<int32_t>(b.proj_h)}};
        pv[i].subImage.imageArrayIndex = 0;
    }
    layer = {XR_TYPE_COMPOSITION_LAYER_PROJECTION};
    layer.layerFlags = XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT |
                       XR_COMPOSITION_LAYER_CORRECT_CHROMATIC_ABERRATION_BIT;
    layer.space = b.local_space;
    layer.viewCount = 2;
    layer.views = pv;
    return true;
}

void render_xr_frame(Backend& b) {
    if (b.session == XR_NULL_HANDLE || !b.session_running.load(std::memory_order_acquire)) return;

    XrFrameWaitInfo wait_info{XR_TYPE_FRAME_WAIT_INFO};
    XrFrameState frame_state{XR_TYPE_FRAME_STATE};
    if (XR_FAILED(xrWaitFrame(b.session, &wait_info, &frame_state))) return;
    b.last_predicted_time = frame_state.predictedDisplayTime;

    if (b.session_state == XR_SESSION_STATE_FOCUSED &&
        (!b.pose_initialized || b.recenter_frames_remaining > 0) &&
        frame_state.predictedDisplayTime >= b.recenter_after_time) {
        recenter_screen_to_head(b, frame_state.predictedDisplayTime);
        if (b.pose_initialized && b.recenter_frames_remaining > 0) {
            --b.recenter_frames_remaining;
        }
    }

    XrFrameBeginInfo begin_info{XR_TYPE_FRAME_BEGIN_INFO};
    if (XR_FAILED(xrBeginFrame(b.session, &begin_info))) return;

    poll_xr_actions(b);

    std::vector<const XrCompositionLayerBaseHeader*> layers;
    XrCompositionLayerPassthroughFB pt_layer{XR_TYPE_COMPOSITION_LAYER_PASSTHROUGH_FB};
    XrCompositionLayerQuad screen_quads[kMaskedQuadCount]{};
    for (auto& sq : screen_quads) sq.type = XR_TYPE_COMPOSITION_LAYER_QUAD;
    XrCompositionLayerQuad hud_quad{XR_TYPE_COMPOSITION_LAYER_QUAD};
    XrCompositionLayerQuad frame_quads[4];
    XrCompositionLayerProjection popup_layer{XR_TYPE_COMPOSITION_LAYER_PROJECTION};
    XrCompositionLayerProjectionView popup_views[2] = {{XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW},
                                                       {XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW}};

    if (frame_state.shouldRender) {
        bool do_blit = false;
        {
            std::lock_guard<std::mutex> lk(b.frame_mtx);
            if (b.pending_seq != b.uploaded_seq && !b.staging_pixels.empty()) {
                b.xr_pixels.swap(b.staging_pixels);
                if (b.staging_has_masked) {
                    b.xr_masked_pixels.swap(b.staging_masked_pixels);
                } else {
                    b.xr_masked_pixels.clear();
                }
                b.xr_margin.swap(b.staging_margin);
                b.xr_w = b.staging_w;
                b.xr_h = b.staging_h;
                b.xr_planes = b.staging_planes;
                b.xr_is_rgba = b.staging_is_rgba;
                b.xr_has_masked = b.staging_has_masked;
                b.xr_masked_scene = b.staging_masked_scene;
                std::swap(b.xr_hdbg, b.staging_hdbg);
                b.xr_hdbg_ok = b.staging_hdbg_ok;
                b.uploaded_seq = b.pending_seq;
                do_blit = true;
            }
        }
        if (b.force_reblit && !b.xr_pixels.empty()) {
            do_blit = true;
            b.force_reblit = false;
        }

        if (do_blit && b.xr_w > 0 && b.xr_h > 0) {
            const int iw = static_cast<int>(b.xr_w);
            const int ih = static_cast<int>(b.xr_h);
            if (iw != b.base_w || ih != b.base_h) {
                b.base_w = iw;
                b.base_h = ih;
                allocate_source_texture(b, iw + 2 * b.pad, ih + 2 * b.pad);
                b.rgba_fb.assign(static_cast<size_t>(iw) * ih * 4, 255);
                const uint32_t scale = target_swapchain_scale(iw);
                const uint32_t target_sw = static_cast<uint32_t>(iw + 2 * b.pad) * scale;
                const uint32_t target_sh = static_cast<uint32_t>(ih + 2 * b.pad) * scale;
                if (target_sw != b.swap_w || target_sh != b.swap_h) {
                    b.swap_w = target_sw;
                    b.swap_h = target_sh;
                    b.reported_swap_w.store(b.swap_w, std::memory_order_release);
                    b.reported_swap_h.store(b.swap_h, std::memory_order_release);
                    for (unsigned p = 0; p < kMaskedQuadCount; ++p) {
                        if (b.screen_swapchains[p] != XR_NULL_HANDLE) {
                            xrDestroySwapchain(b.screen_swapchains[p]);
                            b.screen_swapchains[p] = XR_NULL_HANDLE;
                        }
                        create_swapchain_with_images(b, b.swap_w, b.swap_h,
                                                     b.screen_swapchains[p], b.screen_images[p]);
                    }
                    XR_LOGI("Resized OpenXR screen swapchains -> %ux%u (%dx%d)",
                            b.swap_w, b.swap_h, iw, ih);
                }
                b.hud_dirty = true;
            }

            if (b.xr_is_rgba && b.xr_planes > 1) {
                // A sixth plane, when present, is the separate sprite layer (gba::g_layered_sprite_layer).
                const unsigned used_planes = std::min(b.xr_planes, kLayeredPlaneCount + 1);
                const bool has_sprite_layer = used_planes > kLayeredPlaneCount;
                const size_t plane_bytes = static_cast<size_t>(iw) * ih * 4;
                const uint8_t* masked_ptr = b.xr_has_masked ? b.xr_masked_pixels.data() : nullptr;
                // Debug: `adb shell run-as <pkg> touch files/dump_planes` dumps the current
                // frame's stock planes (stock_plane_N.rgba) and semantic planes (plane_N.rgba)
                // to files/plane_dump/ for offline analysis, then removes the trigger.
                {
                    static unsigned s_dump_poll = 0;
                    if ((++s_dump_poll % 30u) == 0u) {
                        const char* dir = SDL_AndroidGetInternalStoragePath();
                        if (dir) {
                            const std::string trig = std::string(dir) + "/dump_planes";
                            if (std::FILE* tf = std::fopen(trig.c_str(), "rb")) {
                                std::fclose(tf);
                                std::remove(trig.c_str());
                                const std::string out = std::string(dir) + "/plane_dump";
                                mkdir(out.c_str(), 0755);
                                auto dump = [&](const std::string& name, const uint8_t* data, size_t n) {
                                    if (std::FILE* f = std::fopen((out + "/" + name).c_str(), "wb")) {
                                        std::fwrite(data, 1, n, f);
                                        std::fclose(f);
                                    }
                                };
                                for (unsigned p = 0; p < used_planes; ++p)
                                    dump("stock_plane_" + std::to_string(p) + ".rgba",
                                         b.xr_pixels.data() + p * plane_bytes, plane_bytes);
                                if (masked_ptr)
                                    for (unsigned p = 0; p < kLayeredPlaneCount; ++p)
                                        dump("plane_" + std::to_string(p) + ".rgba", masked_ptr + p * plane_bytes,
                                             plane_bytes);
                                if (std::FILE* f = std::fopen((out + "/info.txt").c_str(), "w")) {
                                    std::fprintf(f, "w=%d h=%d stock_planes=%u masked=%d masked_scene=%d\n", iw, ih,
                                                 used_planes, masked_ptr ? 1 : 0, b.xr_masked_scene ? 1 : 0);
                                    std::fclose(f);
                                }
                                XR_LOGI("plane dump written to %s (%dx%d, masked=%d)", out.c_str(), iw, ih,
                                        masked_ptr ? 1 : 0);
                            }
                        }
                    }
                }
                const bool use_masked =
                    (b.depth_mode == 2 && masked_ptr != nullptr && b.xr_masked_scene &&
                     !b.popup_init_failed);
                if (b.depth_mode == 0 || (b.depth_mode == 2 && !use_masked)) {
                    b.masked_scene_active = false;
                    // 2D FLAT (or MASKED 3D non-field fallback): composite all planes back-to-front into swapchain 0.
                    b.rgba_composite_fb.assign(b.xr_pixels.data(), b.xr_pixels.data() + plane_bytes);
                    unsigned flat_order[kLayeredPlaneCount + 1];
                    unsigned flat_n = 0;
                    const int sprite_after = static_cast<int>(std::floor(gba::g_layered_sprite_depth_live));
                    for (unsigned p = 1; p < std::min(used_planes, kLayeredPlaneCount); ++p) {
                        flat_order[flat_n++] = p;
                        if (has_sprite_layer && static_cast<int>(p) == sprite_after) flat_order[flat_n++] = kLayeredPlaneCount;
                    }
                    if (has_sprite_layer && sprite_after >= static_cast<int>(kLayeredPlaneCount) - 1 &&
                        (flat_n == 0 || flat_order[flat_n - 1] != kLayeredPlaneCount))
                        flat_order[flat_n++] = kLayeredPlaneCount;
                    for (unsigned fi = 0; fi < flat_n; ++fi) {
                        const unsigned p = flat_order[fi];
                        const uint8_t* layer = b.xr_pixels.data() + p * plane_bytes;
                        const bool premul_planes = gba::g_layered_fog_alpha != 0;
                        for (size_t i = 0; i < plane_bytes; i += 4) {
                            const uint32_t a = layer[i + 3];
                            if (premul_planes && a < 255) {
                                // premultiplied: src + dst * (1 - a)
                                for (int k = 0; k < 3; ++k)
                                    b.rgba_composite_fb[i + k] = static_cast<uint8_t>(std::min<uint32_t>(
                                        255u, layer[i + k] + b.rgba_composite_fb[i + k] * (255u - a) / 255u));
                                continue;
                            }
                            if (a == 0) continue;
                            if (a == 255) {
                                b.rgba_composite_fb[i + 0] = layer[i + 0];
                                b.rgba_composite_fb[i + 1] = layer[i + 1];
                                b.rgba_composite_fb[i + 2] = layer[i + 2];
                                b.rgba_composite_fb[i + 3] = 255;
                            } else {
                                const uint32_t inv_a = 255u - a;
                                b.rgba_composite_fb[i + 0] = static_cast<uint8_t>(
                                    (layer[i + 0] * a + b.rgba_composite_fb[i + 0] * inv_a) / 255u);
                                b.rgba_composite_fb[i + 1] = static_cast<uint8_t>(
                                    (layer[i + 1] * a + b.rgba_composite_fb[i + 1] * inv_a) / 255u);
                                b.rgba_composite_fb[i + 2] = static_cast<uint8_t>(
                                    (layer[i + 2] * a + b.rgba_composite_fb[i + 2] * inv_a) / 255u);
                                b.rgba_composite_fb[i + 3] = 255;
                            }
                        }
                    }
                    flip_and_grade_rgba_plane(b, b.rgba_composite_fb.data(), iw, ih, nullptr, false);
                    if (blit_hd_text_bg(b, 0, b.rgba_composite_fb.data(), iw, ih) ||
                        blit_rgba_to_swapchain(b, 0, iw, ih, b.rgba_fb.data(), false)) {
                        glFlush();
                        b.active_planes[0] = true;
                        for (unsigned p = 1; p < kMaskedQuadCount; ++p) b.active_planes[p] = false;
                        b.last_frame_layered = false;
                        b.has_submitted_frame = true;
                    }
                } else if (use_masked) {
                    // POP-UP 3D: analyse the frame into grounded cards and upload textures;
                    // the projection layer itself is redrawn every display frame below.
                    b.masked_scene_active = true;
                    for (unsigned p = 0; p < kMaskedQuadCount; ++p) b.active_planes[p] = false;
                    b.popup_front_active = false;
                    if (ensure_popup_ready(b)) {
                        b.popup.upload(masked_ptr, iw, ih, b.color_lut.get(), b.swapchain_srgb);
                        // Front layer (room frame, overlays, HUD) stays a compositor quad: sharpest text.
                        if (b.popup.layout.front_any) {
                            bool has_visible = false;
                            flip_and_grade_rgba_plane(b, b.popup.layout.front_rgba.data(), iw, ih,
                                                      &has_visible, true);
                            if (has_visible &&
                                blit_rgba_to_swapchain(b, kLayeredPlaneCount, iw, ih, b.rgba_fb.data(), true)) {
                                b.popup_front_active = true;
                                b.active_planes[kLayeredPlaneCount] = true;
                            }
                        }
                        glFlush();
                        b.last_frame_layered = true;
                        b.has_submitted_frame = true;
                    } else {
                        b.masked_scene_active = false;
                        b.last_frame_layered = false;
                    }
                } else {
                    const uint8_t* plane_src = b.xr_pixels.data();
                    // STOCK 3D building fix: the stock planes follow GBA hardware layers, and
                    // Mother 3 paints one building across BG2 and BG3, so a roof sits on Plane 2
                    // while its walls lie on the floor plane (split roofs/houses). Where the
                    // semantic classifier says "building/prop" (Plane 2) but the stock pixel is on
                    // a floor plane (0/1), copy it up to Plane 2 so the whole building stands
                    // together. The lower copy is kept so head motion never reveals a hole.
                    if (b.depth_mode == 1 && masked_ptr != nullptr && b.xr_masked_scene && used_planes >= 3) {
                        plane_src = b.fix.run(plane_src, masked_ptr, iw, ih, used_planes);
                        static double s_fix_ms_sum = 0, s_fix_ms_max = 0;
                        static int s_fix_n = 0;
                        s_fix_ms_sum += b.fix.last_ms; s_fix_ms_max = std::max(s_fix_ms_max, b.fix.last_ms);
                        if (++s_fix_n == 300) {
                            XR_LOGI("stock3d-fix: avg %.2f ms max %.2f ms per game frame", s_fix_ms_sum / s_fix_n,
                                    s_fix_ms_max);
                            s_fix_ms_sum = s_fix_ms_max = 0; s_fix_n = 0;
                        }
                    }
                    // Controller capture (R-stick hold): stock planes, semantic planes and the
                    // planes as displayed after the building fix, into files/captures/<ms>/.
                    if (b.capture_requested.exchange(false, std::memory_order_acq_rel)) {
                        const char* dir = SDL_AndroidGetInternalStoragePath();
                        if (dir) {
                            const std::string base = std::string(dir) + "/captures";
                            mkdir(base.c_str(), 0755);
                            const long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                                     std::chrono::system_clock::now().time_since_epoch()).count();
                            const std::string out = base + "/" + std::to_string(ms);
                            mkdir(out.c_str(), 0755);
                            auto dump = [&](const std::string& name, const uint8_t* data, size_t n) {
                                if (std::FILE* f = std::fopen((out + "/" + name).c_str(), "wb")) {
                                    std::fwrite(data, 1, n, f);
                                    std::fclose(f);
                                }
                            };
                            for (unsigned p = 0; p < used_planes; ++p) {
                                dump("stock_plane_" + std::to_string(p) + ".rgba", b.xr_pixels.data() + p * plane_bytes, plane_bytes);
                                dump("final_plane_" + std::to_string(p) + ".rgba", plane_src + p * plane_bytes, plane_bytes);
                            }
                            if (masked_ptr)
                                for (unsigned p = 0; p < kLayeredPlaneCount; ++p)
                                    dump("plane_" + std::to_string(p) + ".rgba", masked_ptr + p * plane_bytes, plane_bytes);
                            if (std::FILE* f = std::fopen((out + "/info.txt").c_str(), "w")) {
                                std::fprintf(f, "w=%d h=%d stock_planes=%u masked=%d masked_scene=%d depth_mode=%d deep=%d\n",
                                             iw, ih, used_planes, masked_ptr ? 1 : 0, b.xr_masked_scene ? 1 : 0,
                                             b.depth_mode, b.stock_deep ? 1 : 0);
                                std::fclose(f);
                            }
                            // raw machine memory too (offline: map id, whole-map layers, camera);
                            // copied from the running GBA thread, so it may tear slightly
                            if (gba::GbaBus* bus = gbarecomp::active_bus()) {
                                dump("ewram.bin", bus->ewram_ptr(), 0x40000);
                                dump("vram.bin", bus->vram_ptr(), 0x18000);
                                dump("pal.bin", bus->pal_ptr(), 0x400);
                                dump("io.bin", bus->io().raw(), 0x400);
                            }
                            XR_LOGI("capture saved to %s", out.c_str());
                        }
                    }
                    b.masked_scene_active = false;
                    bool any_blitted = false;
                    for (unsigned p = 0; p < kLayeredPlaneCount + 1; ++p) {
                        if (p >= used_planes) {
                            b.active_planes[p] = false;
                            continue;
                        }
                        bool has_visible = false;
                        flip_and_grade_rgba_plane(b, plane_src + p * plane_bytes, iw, ih,
                                                  &has_visible, p > 0);
                        if (p == 0) has_visible = true; // Backdrop / Void plane always submitted
                        if (!has_visible) {
                            b.active_planes[p] = false;
                            continue;
                        }
                        bool blitted = false;
                        const int mw = b.wide_margin;
                        const bool real_margin = b.pad > 0 && mw == b.pad && p < kLayeredPlaneCount &&
                                                 b.xr_margin.size() >= static_cast<size_t>(p + 1) * ih * 2 * mw * 4;
                        if (real_margin) {
                            // Real level columns left/right (rendered wide), mirrored rows top/bottom.
                            const int ww = iw + 2 * mw;
                            b.wide_plane.resize(static_cast<size_t>(ww) * ih * 4);
                            const uint8_t* srcp = plane_src + p * plane_bytes;
                            for (int y = 0; y < ih; ++y) {
                                uint8_t* d = b.wide_plane.data() + static_cast<size_t>(y) * ww * 4;
                                const uint8_t* mrow = b.xr_margin.data() + (static_cast<size_t>(p) * ih + y) * 2 * mw * 4;
                                std::memcpy(d, mrow, mw * 4);
                                std::memcpy(d + mw * 4, srcp + static_cast<size_t>(y) * iw * 4, static_cast<size_t>(iw) * 4);
                                std::memcpy(d + (mw + iw) * 4, mrow + mw * 4, mw * 4);
                            }
                            bool vis2 = false;
                            flip_and_grade_rgba_plane(b, b.wide_plane.data(), ww, ih, &vis2, p > 0);
                            const int pw = ww, ph = ih + 2 * b.pad;
                            b.pad_fb.assign(static_cast<size_t>(pw) * ph * 4, 0);
                            for (int y = 0; y < ph; ++y) {
                                int sy = y - b.pad;  // mirror at the top and bottom edge
                                if (sy < 0) sy = -1 - sy;
                                if (sy >= ih) sy = 2 * ih - 1 - sy;
                                sy = std::clamp(sy, 0, ih - 1);
                                std::memcpy(b.pad_fb.data() + static_cast<size_t>(y) * pw * 4,
                                            b.rgba_fb.data() + static_cast<size_t>(sy) * ww * 4, static_cast<size_t>(pw) * 4);
                            }
                            blitted = blit_rgba_to_swapchain(b, p, pw, ph, b.pad_fb.data(), true);
                            if (b.window_clip && b.winr.ready())
                                b.winr.upload(static_cast<int>(p), b.pad_fb.data(), pw, ph, b.swapchain_srgb);
                        } else if (b.pad > 0) {
                            // Edge-extend the GBA planes into the margin (the frame hides the
                            // seam); the separate sprite layer (index 5) gets a clear margin.
                            const int pw = iw + 2 * b.pad, ph = ih + 2 * b.pad;
                            b.pad_fb.assign(static_cast<size_t>(pw) * ph * 4, 0);
                            const bool extend = p < kLayeredPlaneCount;
                            for (int y = 0; y < ph; ++y) {
                                int sy = y - b.pad;
                                if (sy < 0 || sy >= ih) { if (!extend) continue; sy = std::clamp(sy, 0, ih - 1); }
                                uint8_t* drow = b.pad_fb.data() + static_cast<size_t>(y) * pw * 4;
                                const uint8_t* srow = b.rgba_fb.data() + static_cast<size_t>(sy) * iw * 4;
                                std::memcpy(drow + b.pad * 4, srow, static_cast<size_t>(iw) * 4);
                                if (extend) {
                                    for (int x = 0; x < b.pad; ++x) {
                                        std::memcpy(drow + x * 4, srow, 4);
                                        std::memcpy(drow + (b.pad + iw + x) * 4, srow + (iw - 1) * 4, 4);
                                    }
                                }
                            }
                            blitted = blit_rgba_to_swapchain(b, p, pw, ph, b.pad_fb.data(), true);
                            if (b.window_clip && b.winr.ready())
                                b.winr.upload(static_cast<int>(p), b.pad_fb.data(), pw, ph, b.swapchain_srgb);
                        } else if (b.window_clip && b.winr.ready()) {
                            b.winr.upload(static_cast<int>(p), b.rgba_fb.data(), iw, ih, b.swapchain_srgb);
                        }
                        if (blitted || blit_hd_text_bg(b, p, plane_src + p * plane_bytes, iw, ih) ||
                            blit_rgba_to_swapchain(b, p, iw, ih, b.rgba_fb.data(), true)) {
                            b.active_planes[p] = true;
                            any_blitted = true;
                        } else {
                            b.active_planes[p] = false;
                        }
                    }
                    for (unsigned p = kLayeredPlaneCount + 1; p < kMaskedQuadCount; ++p) {
                        b.active_planes[p] = false;
                    }
                    if (any_blitted) {
                        glFlush();
                        b.last_frame_layered = true;
                        b.has_submitted_frame = true;
                    }
                }

                static int s_last_mask = -1;
                static unsigned s_last_w = 0, s_last_h = 0;
                static int s_last_depth = -1;
                static int s_last_masked_scene = -1;
                int mask = 0;
                for (unsigned p = 0; p < kMaskedQuadCount; ++p) {
                    if (b.active_planes[p]) mask |= (1 << p);
                }
                const int scene_flag = b.masked_scene_active ? 1 : 0;
                if (mask != s_last_mask || b.xr_w != s_last_w || b.xr_h != s_last_h ||
                    b.depth_mode != s_last_depth || scene_flag != s_last_masked_scene) {
                    s_last_mask = mask;
                    s_last_w = b.xr_w;
                    s_last_h = b.xr_h;
                    s_last_depth = b.depth_mode;
                    s_last_masked_scene = scene_flag;
                    XR_LOGI("present_layered_3d %ux%u active_mask=0x%02x layered=%d depth_mode=%d masked_scene=%d",
                            b.xr_w, b.xr_h, mask, b.last_frame_layered ? 1 : 0, b.depth_mode,
                            scene_flag);
                }
            } else {
                const uint8_t* graded = b.xr_pixels.data();
                if (b.color_lut && !b.color_lut->is_passthrough()) {
                    b.graded_fb.resize(static_cast<size_t>(iw) * ih * 3);
                    b.color_lut->map_rgb888(b.xr_pixels.data(), b.graded_fb.data(), iw, ih);
                    graded = b.graded_fb.data();
                }
                for (int y = 0; y < ih; ++y) {
                    const uint8_t* in_row = graded + static_cast<size_t>(y) * iw * 3;
                    uint8_t* out_row = b.rgba_fb.data() +
                                       static_cast<size_t>(ih - 1 - y) * iw * 4;
                    for (int x = 0; x < iw; ++x) {
                        out_row[x * 4 + 0] = in_row[x * 3 + 0];
                        out_row[x * 4 + 1] = in_row[x * 3 + 1];
                        out_row[x * 4 + 2] = in_row[x * 3 + 2];
                        out_row[x * 4 + 3] = 255;
                    }
                }
                if (blit_rgba_to_swapchain(b, 0, iw, ih, b.rgba_fb.data(), false)) {
                    glFlush();
                    b.masked_scene_active = false;
                    b.active_planes[0] = true;
                    for (unsigned p = 1; p < kMaskedQuadCount; ++p) b.active_planes[p] = false;
                    b.last_frame_layered = false;
                    b.has_submitted_frame = true;
                }
            }
        }

        if (b.passthrough_supported && b.passthrough_active) {
            pt_layer.layerHandle = b.passthrough_layer;
            pt_layer.flags = XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT;
            pt_layer.space = XR_NULL_HANDLE;
            layers.push_back(reinterpret_cast<const XrCompositionLayerBaseHeader*>(&pt_layer));
        }

        if (b.has_submitted_frame) {
            const float aspect = static_cast<float>(b.base_h) /
                                 static_cast<float>(std::max(1, b.base_w));
            // Preserve vertical comfort when Mother 3's 16:9 widescreen expands width.
            const float width_m = b.screen_width_m.load(std::memory_order_relaxed);
            const float screen_w = (b.base_w > 240)
                                       ? (width_m * 1.18f)
                                       : width_m;
            const float d_ref = std::max(0.5f, b.screen_distance);
            auto push_screen_quad = [&](unsigned p, float z_local) {
                // Perspective-match size so planes align pixel-for-pixel from head origin.
                const float d_p = std::max(0.25f, d_ref - z_local);
                const float scale_p = d_p / d_ref;
                const XrVector3f z_off = rotate_vec(b.screen_pose.orientation, 0.0f, 0.0f, z_local);
                XrCompositionLayerQuad& q = screen_quads[p];
                q.layerFlags = (p == 0)
                                   ? XR_COMPOSITION_LAYER_CORRECT_CHROMATIC_ABERRATION_BIT
                                   : (XR_COMPOSITION_LAYER_CORRECT_CHROMATIC_ABERRATION_BIT |
                                      XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT);
                q.space = b.local_space;
                q.eyeVisibility = XR_EYE_VISIBILITY_BOTH;
                q.subImage.swapchain = b.screen_swapchains[p];
                q.subImage.imageRect = {{0, 0},
                                        {static_cast<int32_t>(b.swap_w),
                                         static_cast<int32_t>(b.swap_h)}};
                q.subImage.imageArrayIndex = 0;
                q.pose.orientation = b.screen_pose.orientation;
                q.pose.position = {
                    b.screen_pose.position.x + z_off.x,
                    b.screen_pose.position.y + z_off.y,
                    b.screen_pose.position.z + z_off.z,
                };
                float pad_w = b.pad > 0 ? static_cast<float>(b.base_w + 2 * b.pad) / b.base_w : 1.0f;
                float pad_h = b.pad > 0 ? static_cast<float>(b.base_h + 2 * b.pad) / b.base_h : 1.0f;
                if (b.window_clip && b.pad > 0) {
                    // Window mode: near planes show only the real screen, no stretched margin.
                    const int32_t sc = static_cast<int32_t>(b.swap_w) / (b.base_w + 2 * b.pad);
                    q.subImage.imageRect = {{b.pad * sc, b.pad * sc}, {b.base_w * sc, b.base_h * sc}};
                    pad_w = pad_h = 1.0f;
                }
                q.size = {screen_w * scale_p * pad_w, screen_w * aspect * scale_p * pad_h};
                layers.push_back(reinterpret_cast<const XrCompositionLayerBaseHeader*>(&q));
            };
            // Shadow box frame: four dark bars just in front of the front plane, their inner
            // edges on the screen edge, wide enough to cover the layers' margins when leaning.
            auto push_frame = [&](float z_local, bool upload_only = false) {
                if (b.frame_swapchain == XR_NULL_HANDLE) return;
                if (!b.frame_uploaded) {
                    uint32_t fi = 0;
                    XrSwapchainImageAcquireInfo acq{XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO};
                    if (XR_SUCCEEDED(xrAcquireSwapchainImage(b.frame_swapchain, &acq, &fi))) {
                        XrSwapchainImageWaitInfo wait{XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO};
                        wait.timeout = XR_INFINITE_DURATION;
                        if (XR_SUCCEEDED(xrWaitSwapchainImage(b.frame_swapchain, &wait))) {
                            std::vector<uint8_t> px(16 * 16 * 4);
                            for (size_t i = 0; i < px.size(); i += 4) { px[i] = 0; px[i + 1] = 0; px[i + 2] = 0; px[i + 3] = 255; }  // = the dark void
                            glBindTexture(GL_TEXTURE_2D, b.frame_images[fi].image);
                            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 16, 16, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
                            glBindTexture(GL_TEXTURE_2D, 0);
                        }
                        XrSwapchainImageReleaseInfo rel{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
                        xrReleaseSwapchainImage(b.frame_swapchain, &rel);
                        b.frame_uploaded = true;
                    }
                }
                if (upload_only) return;
                const float d_p = std::max(0.25f, d_ref - z_local);
                const float sc = d_p / d_ref;
                const float w = screen_w * sc, h = screen_w * aspect * sc, bw = 0.45f * sc;  // original shadow box
                const float cx[4] = {0.0f, 0.0f, -(w + bw) * 0.5f, (w + bw) * 0.5f};
                const float cy[4] = {(h + bw) * 0.5f, -(h + bw) * 0.5f, 0.0f, 0.0f};
                const float sx[4] = {w + 2.0f * bw, w + 2.0f * bw, bw, bw};
                const float sy[4] = {bw, bw, h + 2.0f * bw, h + 2.0f * bw};  // sides overlap the corners: no seams
                for (int i = 0; i < 4; ++i) {
                    XrCompositionLayerQuad& f = frame_quads[i];
                    f = XrCompositionLayerQuad{XR_TYPE_COMPOSITION_LAYER_QUAD};
                    f.layerFlags = XR_COMPOSITION_LAYER_CORRECT_CHROMATIC_ABERRATION_BIT;
                    f.space = b.local_space;
                    f.eyeVisibility = XR_EYE_VISIBILITY_BOTH;
                    f.subImage.swapchain = b.frame_swapchain;
                    f.subImage.imageRect = {{0, 0}, {16, 16}};
                    f.pose.orientation = b.screen_pose.orientation;
                    const XrVector3f off = rotate_vec(b.screen_pose.orientation, cx[i], cy[i], z_local);
                    f.pose.position = {b.screen_pose.position.x + off.x, b.screen_pose.position.y + off.y,
                                       b.screen_pose.position.z + off.z};
                    f.size = {sx[i], sy[i]};
                    layers.push_back(reinterpret_cast<const XrCompositionLayerBaseHeader*>(&f));
                }
            };
            // Diorama walls (GBARECOMP_XR_BOXWALLS=1): four dark walls following the view frustum
            // from the play layers back to the farthest layer, so leaning shows the inside of a
            // box instead of the far layers' edges. Edge-on (invisible) from the default view.
            // Compositor layers have no depth test, so the walls are submitted between the far
            // planes and the near ones (near planes keep their edge margins).
            auto push_walls = [&](float z_front, float z_back) {
                if (!b.box_walls || b.frame_swapchain == XR_NULL_HANDLE || z_back >= z_front - 0.01f) return;
                if (!b.frame_uploaded) push_frame(0.0f, true);  // uploads the wall texture only
                const float W0 = screen_w, H0 = screen_w * aspect;
                auto half = [&](float z, float extent) { return 0.5f * extent * std::max(0.25f, d_ref - z) / d_ref; };
                const float hwf = half(z_front, W0), hwb = half(z_back, W0), hhf = half(z_front, H0), hhb = half(z_back, H0);
                struct V3 { float x, y, z; };
                auto sub = [](V3 a, V3 c) { return V3{a.x - c.x, a.y - c.y, a.z - c.z}; };
                auto cross = [](V3 a, V3 c) { return V3{a.y * c.z - a.z * c.y, a.z * c.x - a.x * c.z, a.x * c.y - a.y * c.x}; };
                auto norm = [](V3 a) { const float l = std::sqrt(a.x * a.x + a.y * a.y + a.z * a.z); return V3{a.x / l, a.y / l, a.z / l}; };
                const V3 F[4] = {{-hwf, 0, z_front}, {hwf, 0, z_front}, {0, hhf, z_front}, {0, -hhf, z_front}};
                const V3 B[4] = {{-hwb, 0, z_back}, {hwb, 0, z_back}, {0, hhb, z_back}, {0, -hhb, z_back}};
                for (int i = 0; i < 4; ++i) {
                    const V3 d = sub(B[i], F[i]);
                    const float L = std::sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
                    const V3 dir = norm(d);
                    V3 xa, ya; float sx, sy;
                    if (i < 2) { xa = dir; ya = V3{0, 1, 0}; sx = L; sy = 2.0f * hhb; }
                    else       { xa = V3{1, 0, 0}; ya = dir; sx = 2.0f * hwb; sy = L; }
                    const V3 za = norm(cross(xa, ya));
                    const V3 c{(F[i].x + B[i].x) * 0.5f, (F[i].y + B[i].y) * 0.5f, (F[i].z + B[i].z) * 0.5f};
                    const XrQuaternionf& o = b.screen_pose.orientation;
                    const XrVector3f xw = rotate_vec(o, xa.x, xa.y, xa.z), yw = rotate_vec(o, ya.x, ya.y, ya.z),
                                     zw = rotate_vec(o, za.x, za.y, za.z), cw = rotate_vec(o, c.x, c.y, c.z);
                    XrCompositionLayerQuad& f = frame_quads[i];
                    f = XrCompositionLayerQuad{XR_TYPE_COMPOSITION_LAYER_QUAD};
                    f.layerFlags = XR_COMPOSITION_LAYER_CORRECT_CHROMATIC_ABERRATION_BIT;
                    f.space = b.local_space;
                    f.eyeVisibility = XR_EYE_VISIBILITY_BOTH;
                    f.subImage.swapchain = b.frame_swapchain;
                    f.subImage.imageRect = {{0, 0}, {16, 16}};
                    f.pose.orientation = quat_from_basis(xw, yw, zw);
                    f.pose.position = {b.screen_pose.position.x + cw.x, b.screen_pose.position.y + cw.y,
                                       b.screen_pose.position.z + cw.z};
                    f.size = {sx, sy};
                    layers.push_back(reinterpret_cast<const XrCompositionLayerBaseHeader*>(&f));
                }
            };
            if (b.last_frame_layered && b.depth_mode == 2 && b.masked_scene_active) {
                // POP-UP 3D: tilted floor + grounded cards rendered per eye, then the front
                // layer (room frame / overlays / HUD) as a sharp compositor quad.
                if (render_popup_projection(b, frame_state.predictedDisplayTime, screen_w,
                                            screen_w * aspect, popup_layer, popup_views)) {
                    layers.push_back(reinterpret_cast<const XrCompositionLayerBaseHeader*>(&popup_layer));
                }
                if (b.popup_front_active) push_screen_quad(kLayeredPlaneCount, 0.165f);
            } else if (b.last_frame_layered && b.depth_mode == 1) {
                // STOCK 3D: GBA hardware layers as parallel quads.
                const float step_m = (b.depth_mode == 1 && b.stock_deep) ? 0.11f : depth_step_meters(b.depth_mode);
                // Back to front by depth: the separate sprite layer (quad 5) sits between planes.
                const float sprite_z = (gba::g_layered_sprite_depth_live - 2.0f) * step_m;
                bool sprite_pushed = !b.active_planes[kLayeredPlaneCount];
                float far_z = 0.0f;
                for (unsigned p = 0; p < 2; ++p)
                    if (b.active_planes[p]) far_z = std::min(far_z, (static_cast<float>(p) - 2.0f) * step_m - gba::g_layered_plane_depth_bias[p]);
                bool in_window = false;
                if (b.window_clip) {
                    // Window mode: every plane drawn per eye, clipped to the front plane's opening.
                    int slots[kLayeredPlaneCount + 1];
                    float zs[kLayeredPlaneCount + 1];
                    int n = 0;
                    bool sp = false;
                    for (unsigned p = 0; p < kLayeredPlaneCount; ++p) {
                        const float z = (static_cast<float>(p) - 2.0f) * step_m - gba::g_layered_plane_depth_bias[p];
                        if (!sp && z > sprite_z) {
                            if (b.active_planes[kLayeredPlaneCount]) { slots[n] = kLayeredPlaneCount; zs[n++] = sprite_z; }
                            sp = true;
                        }
                        if (!b.active_planes[p]) continue;
                        slots[n] = static_cast<int>(p);
                        zs[n++] = z;
                    }
                    if (!sp && b.active_planes[kLayeredPlaneCount]) { slots[n] = kLayeredPlaneCount; zs[n++] = sprite_z; }
                    in_window = render_window_projection(b, frame_state.predictedDisplayTime, screen_w, aspect,
                                                         step_m, slots, zs, n, popup_layer, popup_views);
                    if (in_window) layers.push_back(reinterpret_cast<const XrCompositionLayerBaseHeader*>(&popup_layer));
                }
                if (in_window) sprite_pushed = true;
                for (unsigned p = 0; p < kLayeredPlaneCount && !in_window; ++p) {
                    const float z = (static_cast<float>(p) - 2.0f) * step_m - gba::g_layered_plane_depth_bias[p];
                    if (p == 2) push_walls(z, far_z - 0.02f);
                    if (!sprite_pushed && z > sprite_z) { push_screen_quad(kLayeredPlaneCount, sprite_z); sprite_pushed = true; }
                    if (!b.active_planes[p]) continue;
                    push_screen_quad(p, z);
                }
                if (!sprite_pushed) push_screen_quad(kLayeredPlaneCount, sprite_z);
                if (!b.box_walls && !b.window_clip && b.pad > 0) push_frame(2.0f * step_m + 0.03f);
            } else {
                XrCompositionLayerQuad& q = screen_quads[0];
                q.layerFlags = XR_COMPOSITION_LAYER_CORRECT_CHROMATIC_ABERRATION_BIT;
                q.space = b.local_space;
                q.eyeVisibility = XR_EYE_VISIBILITY_BOTH;
                q.subImage.swapchain = b.screen_swapchains[0];
                q.subImage.imageRect = {{0, 0},
                                        {static_cast<int32_t>(b.swap_w),
                                         static_cast<int32_t>(b.swap_h)}};
                if (b.pad > 0) {
                    const int32_t sc = static_cast<int32_t>(b.swap_w) / (b.base_w + 2 * b.pad);
                    q.subImage.imageRect = {{b.pad * sc, b.pad * sc}, {b.base_w * sc, b.base_h * sc}};
                }
                q.subImage.imageArrayIndex = 0;
                q.pose = b.screen_pose;
                q.size = {screen_w, screen_w * aspect};
                layers.push_back(reinterpret_cast<const XrCompositionLayerBaseHeader*>(&q));
            }

            if (b.hud_swapchain != XR_NULL_HANDLE && b.hud_mode == 0 && !b.menu_enabled) {
                upload_hud_image(b);
                const float screen_h = screen_w * aspect;
                const float hud_w = 1.24f;
                const float hud_h = hud_w * (static_cast<float>(kHudHeight) / kHudWidth);
                // Place HUD bar neatly centered below the virtual screen stack.
                const XrVector3f up_off = rotate_vec(b.screen_pose.orientation, 0.0f,
                                                     -(screen_h * 0.5f + hud_h * 0.5f + 0.06f),
                                                     0.05f);
                hud_quad.layerFlags = XR_COMPOSITION_LAYER_CORRECT_CHROMATIC_ABERRATION_BIT |
                                      XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT;
                hud_quad.space = b.local_space;
                hud_quad.eyeVisibility = XR_EYE_VISIBILITY_BOTH;
                hud_quad.subImage.swapchain = b.hud_swapchain;
                hud_quad.subImage.imageRect = {{0, 0}, {kHudWidth, kHudHeight}};
                hud_quad.subImage.imageArrayIndex = 0;
                hud_quad.pose.orientation = b.screen_pose.orientation;
                hud_quad.pose.position = {
                    b.screen_pose.position.x + up_off.x,
                    b.screen_pose.position.y + up_off.y,
                    b.screen_pose.position.z + up_off.z,
                };
                hud_quad.size = {hud_w, hud_h};
                layers.push_back(reinterpret_cast<const XrCompositionLayerBaseHeader*>(&hud_quad));
            }
        }
    }

    XrCompositionLayerQuad menu_quad{XR_TYPE_COMPOSITION_LAYER_QUAD};
    if (b.menu_enabled && b.menu_open && b.menu_swapchain != XR_NULL_HANDLE && b.session_running &&
        frame_state.shouldRender) {
        upload_menu_image(b);
        const float menu_w = 0.95f;
        const XrVector3f fwd_off = rotate_vec(b.screen_pose.orientation, 0.0f, 0.0f, 0.35f);
        menu_quad.layerFlags = XR_COMPOSITION_LAYER_CORRECT_CHROMATIC_ABERRATION_BIT |
                               XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT;
        menu_quad.space = b.local_space;
        menu_quad.eyeVisibility = XR_EYE_VISIBILITY_BOTH;
        menu_quad.subImage.swapchain = b.menu_swapchain;
        menu_quad.subImage.imageRect = {{0, 0}, {kMenuWidth, kMenuHeight}};
        menu_quad.pose.orientation = b.screen_pose.orientation;
        menu_quad.pose.position = {b.screen_pose.position.x + fwd_off.x, b.screen_pose.position.y + fwd_off.y,
                                   b.screen_pose.position.z + fwd_off.z};
        menu_quad.size = {menu_w, menu_w * static_cast<float>(kMenuHeight) / kMenuWidth};
        layers.push_back(reinterpret_cast<const XrCompositionLayerBaseHeader*>(&menu_quad));
    }

    XrFrameEndInfo end_info{XR_TYPE_FRAME_END_INFO};
    end_info.displayTime = frame_state.predictedDisplayTime;
    end_info.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
    end_info.layerCount = static_cast<uint32_t>(layers.size());
    end_info.layers = layers.empty() ? nullptr : layers.data();
    const XrResult eres = xrEndFrame(b.session, &end_info);
    if (XR_FAILED(eres)) {
        static bool s_logged = false;
        if (!s_logged) {
            XR_LOGW("xrEndFrame failed: %d (layers=%u)", eres, end_info.layerCount);
            s_logged = true;
        }
    }
}

void xr_thread_main(Backend& b) {
    // Ensure this thread is attached to JavaVM before OpenXR loader/session calls.
    (void)SDL_AndroidGetJNIEnv();

    const bool ok = init_egl(b) && init_openxr(b);
    if (ok) {
        b.reported_swap_w.store(b.swap_w, std::memory_order_release);
        b.reported_swap_h.store(b.swap_h, std::memory_order_release);
    }
    {
        std::lock_guard<std::mutex> lk(b.xr_init_mtx);
        b.xr_init_ok = ok;
        b.xr_init_done = true;
    }
    b.xr_init_cv.notify_all();

    if (!ok) {
        shutdown_openxr(b);
        shutdown_egl(b);
        return;
    }

    if (b.egl_display != EGL_NO_DISPLAY) {
        eglMakeCurrent(b.egl_display, b.egl_surface, b.egl_surface, b.egl_context);
    }

    while (!b.stop_xr_thread.load(std::memory_order_acquire)) {
        handle_xr_events(b);
        process_pending_haptics(b);
        if (b.session_running.load(std::memory_order_acquire) &&
            !b.exit_requested.load(std::memory_order_acquire)) {
            render_xr_frame(b);
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(8));
        }
    }

    shutdown_openxr(b);
    shutdown_egl(b);
}

void quest_audio_callback(void* userdata, Uint8* stream, int len) {
    auto* b = static_cast<Backend*>(userdata);
    const int frames = len / static_cast<int>(sizeof(int16_t));
    if (b && b->audio_enabled && b->bridge_ready && b->audio_mtx) {
        auto* out = reinterpret_cast<int16_t*>(stream);
        const auto now = std::chrono::steady_clock::now();
        SDL_LockMutex(b->audio_mtx);
        if (b->last_cb_tp.time_since_epoch().count() != 0) {
            const double gap_ms = std::chrono::duration<double, std::milli>(now - b->last_cb_tp).count();
            if (gap_ms > b->max_cb_gap_ms) b->max_cb_gap_ms = gap_ms;
        }
        b->last_cb_tp = now;
        ++b->audio_cb_count;
        if (frames > b->max_cb_frames) b->max_cb_frames = frames;
        const bool was_conceal = (b->bridge.concealing != 0);
        rab_pull(&b->bridge, out, frames);
        const bool now_conceal = (b->bridge.concealing != 0);
        if (frames > 0) {
            const int32_t bjump = std::abs(static_cast<int32_t>(out[0]) - static_cast<int32_t>(b->prev_out_sample));
            if (bjump > b->out_boundary_max_jump) b->out_boundary_max_jump = bjump;
            int32_t maxj = bjump;
            for (int i = 1; i < frames; ++i) {
                const int32_t j = std::abs(static_cast<int32_t>(out[i]) - static_cast<int32_t>(out[i - 1]));
                if (j > maxj) maxj = j;
            }
            if (maxj > b->out_max_jump) b->out_max_jump = maxj;
            b->prev_out_sample = out[frames - 1];
            if (was_conceal && !now_conceal) {
                ++b->conceal_exit_jumps;
                if (maxj > b->conceal_exit_max_jump) b->conceal_exit_max_jump = maxj;
            }
        }
        b->prev_concealing = now_conceal;
        SDL_UnlockMutex(b->audio_mtx);
    } else {
        std::memset(stream, 0, static_cast<size_t>(len));
    }
}

bool init_rab_bridge(Backend& b, double source_rate, double host_rate) {
    rab_config cfg{};
    rab_config_defaults(&cfg);
    cfg.channels = 1;
    cfg.source_rate = source_rate;
    cfg.host_rate = host_rate;
    cfg.target_ms = 90.0;
    cfg.ring_ms = 350.0;
    cfg.em_high_ms = 280.0;
    cfg.preroll_ms = 110.0;
    if (rab_init(&b.bridge, &cfg) != 0) {
        b.bridge_ready = false;
        return false;
    }
    b.bridge_ready = true;
    return true;
}

void reset_rab_bridge(Backend& b) {
    if (!b.audio_mtx || !b.bridge_ready) return;
    SDL_LockMutex(b.audio_mtx);
    const rab_config cfg = b.bridge.cfg;
    rab_free(&b.bridge);
    b.bridge_ready = (rab_init(&b.bridge, &cfg) == 0);
    SDL_UnlockMutex(b.audio_mtx);
}

void init_sdl_audio(Backend& b) {
    if (SDL_WasInit(SDL_INIT_AUDIO) == 0) {
        if (SDL_InitSubSystem(SDL_INIT_AUDIO) != 0) {
            XR_LOGW("SDL_InitSubSystem(AUDIO) failed: %s", SDL_GetError());
            return;
        }
        b.sdl_audio_init = true;
    }
    SDL_AudioSpec want{};
    want.freq = 48000;
    want.format = AUDIO_S16SYS;
    want.channels = 1;
    want.samples = 2048;
    want.callback = quest_audio_callback;
    want.userdata = &b;
    SDL_AudioSpec got{};
    b.audio_dev = SDL_OpenAudioDevice(nullptr, 0, &want, &got, SDL_AUDIO_ALLOW_FREQUENCY_CHANGE);
    if (b.audio_dev != 0) {
        const double host_rate = static_cast<double>(got.freq > 0 ? got.freq : 48000);
        b.audio_mtx = SDL_CreateMutex();
        if (b.audio_mtx && init_rab_bridge(b, 65536.0, host_rate)) {
            SDL_PauseAudioDevice(b.audio_dev, 0);
            XR_LOGI("SDL audio opened (RAB): driver=%s rate=%dHz samples=%u target=75ms",
                    SDL_GetCurrentAudioDriver() ? SDL_GetCurrentAudioDriver() : "?",
                    got.freq, static_cast<unsigned>(got.samples));
        } else {
            XR_LOGW("RAB audio bridge initialization failed");
        }
    } else {
        XR_LOGW("SDL_OpenAudioDevice failed: %s", SDL_GetError());
    }
}

void push_audio_to_bridge(Backend& b, const int16_t* samples, std::size_t count, uint32_t rate) {
    if (!b.audio_enabled || b.audio_dev == 0 || !b.bridge_ready || !b.audio_mtx ||
        !samples || count == 0 || rate == 0) {
        return;
    }

    const int16_t* src = samples;
    if (b.volume != 100) {
        b.volume_buf.resize(count);
        for (std::size_t i = 0; i < count; ++i) {
            b.volume_buf[i] = static_cast<int16_t>(
                (static_cast<int32_t>(samples[i]) * b.volume) / 100);
        }
        src = b.volume_buf.data();
    }

    if (rate != static_cast<uint32_t>(b.bridge.cfg.source_rate)) {
        const bool was_down = b.bridge.cfg.source_rate > b.bridge.cfg.host_rate;
        const bool now_down = static_cast<double>(rate) > b.bridge.cfg.host_rate;
        SDL_LockMutex(b.audio_mtx);
        if (was_down || now_down) {
            rab_config cfg = b.bridge.cfg;
            cfg.source_rate = static_cast<double>(rate);
            rab_free(&b.bridge);
            b.bridge_ready = (rab_init(&b.bridge, &cfg) == 0);
        } else {
            b.bridge.cfg.source_rate = static_cast<double>(rate);
        }
        SDL_UnlockMutex(b.audio_mtx);
        if (!b.bridge_ready) return;
    }

    const auto now = std::chrono::steady_clock::now();
    if (b.last_push_tp.time_since_epoch().count() != 0) {
        const double gap_ms = std::chrono::duration<double, std::milli>(now - b.last_push_tp).count();
        if (gap_ms > b.max_push_gap_ms) b.max_push_gap_ms = gap_ms;
    }
    b.last_push_tp = now;
    {
        const int32_t bjump = std::abs(static_cast<int32_t>(src[0]) - static_cast<int32_t>(b.prev_in_sample));
        if (bjump > b.in_boundary_max_jump) b.in_boundary_max_jump = bjump;
        int32_t maxj = bjump;
        for (std::size_t i = 1; i < count; ++i) {
            const int32_t j = std::abs(static_cast<int32_t>(src[i]) - static_cast<int32_t>(src[i - 1]));
            if (j > maxj) maxj = j;
        }
        if (maxj > b.in_max_jump) b.in_max_jump = maxj;
        b.prev_in_sample = src[count - 1];
    }

    SDL_LockMutex(b.audio_mtx);
    rab_push(&b.bridge, src, static_cast<int>(count));
    ++b.audio_push_count;
    b.audio_push_samples += count;
    if ((b.audio_push_count % 120ULL) == 0ULL) {
        rab_stats st{};
        rab_get_stats(&b.bridge, &st);
        const double fill_ms = rab_fill_ms(&b.bridge);
        const double secs = static_cast<double>(b.audio_push_samples) / 65536.0;
        const double stretch_ms = static_cast<double>(st.stretch_frames) * 1000.0 /
                                  std::max(1.0, b.bridge.cfg.host_rate);
        const uint64_t cb_cnt = b.audio_cb_count;
        const int cb_max_f = b.max_cb_frames;
        const double cb_gap = b.max_cb_gap_ms;
        const int32_t out_bj = b.out_boundary_max_jump;
        const int32_t out_mj = b.out_max_jump;
        const uint64_t ce_cnt = b.conceal_exit_jumps;
        const int32_t ce_mj = b.conceal_exit_max_jump;
        b.max_cb_gap_ms = 0.0;
        b.out_boundary_max_jump = 0;
        b.out_max_jump = 0;
        b.conceal_exit_max_jump = 0;
        SDL_UnlockMutex(b.audio_mtx);
        XR_LOGI("audio-probe t=%.1fs pushes=%llu cb=%llu(maxF=%d,maxGap=%.1fms) "
                "pushGap=%.1fms fill=%.1fms corr=%+.3f%% underrun=%llu stretch=%.0fms(ev=%llu,ce=%llu,ceJump=%d) "
                "ovf=%llu inJump=(b=%d,m=%d) outJump=(b=%d,m=%d) rate=%u",
                secs,
                static_cast<unsigned long long>(b.audio_push_count),
                static_cast<unsigned long long>(cb_cnt), cb_max_f, cb_gap,
                b.max_push_gap_ms, fill_ms, st.last_correction * 100.0,
                static_cast<unsigned long long>(st.underrun_events),
                stretch_ms, static_cast<unsigned long long>(st.stretch_events),
                static_cast<unsigned long long>(ce_cnt), ce_mj,
                static_cast<unsigned long long>(st.overflow_drops),
                b.in_boundary_max_jump, b.in_max_jump, out_bj, out_mj, rate);
        b.max_push_gap_ms = 0.0;
        b.in_boundary_max_jump = 0;
        b.in_max_jump = 0;
    } else {
        SDL_UnlockMutex(b.audio_mtx);
    }
}

void stage_frame(Backend& b, const uint8_t* src_pixels, unsigned src_w, unsigned src_h,
                 unsigned plane_count = 1, bool is_rgba_planes = false,
                 const uint8_t* masked_planes_rgba = nullptr, bool masked_scene = false) {
    if (!src_pixels || src_w == 0 || src_h == 0) return;
    std::lock_guard<std::mutex> lk(b.frame_mtx);
    const unsigned m = static_cast<unsigned>(b.wide_margin);
    if (m > 0 && is_rgba_planes && src_w > 2 * m &&
        src_w - 2 * m == static_cast<unsigned>(b.requested_base_w.load(std::memory_order_acquire))) {
        // Wide render: keep the 240 px screen as the frame, the side columns as margin data.
        const unsigned nw = src_w - 2 * m, planes = std::max(1u, plane_count);
        const uint8_t* margin_src = (masked_planes_rgba && masked_scene) ? masked_planes_rgba : src_pixels;
        b.staging_pixels.resize(static_cast<size_t>(nw) * src_h * 4 * planes);
        b.staging_masked_pixels.resize(masked_planes_rgba && masked_scene ? b.staging_pixels.size() : 0);
        b.staging_margin.resize(static_cast<size_t>(2 * m) * src_h * 4 * planes);
        for (unsigned p = 0; p < planes; ++p) {
            for (unsigned y = 0; y < src_h; ++y) {
                const size_t srow = (static_cast<size_t>(p) * src_h + y) * src_w * 4;
                const size_t drow = (static_cast<size_t>(p) * src_h + y) * nw * 4;
                std::memcpy(b.staging_pixels.data() + drow, src_pixels + srow + m * 4, nw * 4);
                if (!b.staging_masked_pixels.empty())
                    std::memcpy(b.staging_masked_pixels.data() + drow, masked_planes_rgba + srow + m * 4, nw * 4);
                uint8_t* mrow = b.staging_margin.data() + (static_cast<size_t>(p) * src_h + y) * 2 * m * 4;
                std::memcpy(mrow, margin_src + srow, m * 4);
                std::memcpy(mrow + m * 4, margin_src + srow + (m + nw) * 4, m * 4);
            }
        }
        b.staging_has_masked = !b.staging_masked_pixels.empty();
        b.staging_hdbg_ok = gbarecomp::hdbg::snapshot(b.staging_hdbg);
        b.staging_w = nw;
        b.staging_h = src_h;
        b.staging_planes = plane_count;
        b.staging_is_rgba = true;
        b.staging_masked_scene = masked_scene;
        ++b.pending_seq;
        return;
    }
    b.staging_margin.clear();
    const size_t plane_bytes = static_cast<size_t>(src_w) * src_h * (is_rgba_planes ? 4u : 3u);
    const size_t total_bytes = plane_bytes * std::max(1u, plane_count);
    b.staging_pixels.assign(src_pixels, src_pixels + total_bytes);
    if (is_rgba_planes && masked_planes_rgba && masked_scene) {
        b.staging_masked_pixels.assign(masked_planes_rgba, masked_planes_rgba + total_bytes);
        b.staging_has_masked = true;
    } else {
        b.staging_masked_pixels.clear();
        b.staging_has_masked = false;
    }
    b.staging_hdbg_ok = is_rgba_planes && gbarecomp::hdbg::snapshot(b.staging_hdbg);
    b.staging_w = src_w;
    b.staging_h = src_h;
    b.staging_planes = plane_count;
    b.staging_is_rgba = is_rgba_planes;
    b.staging_masked_scene = masked_scene;
    ++b.pending_seq;
}

}  // namespace

HostWindow::HostWindow() = default;
HostWindow::~HostWindow() { close(); }

bool HostWindow::is_available() { return true; }

bool HostWindow::open(int scale, int base_w, int base_h, const char* title,
                      const char* screen, bool linear_filter, bool /*sharp_filter*/,
                      bool /*resize_driven_view*/, bool /*freely_resizable_window*/) {
    close();
    auto b = std::make_unique<Backend>();
    b->scale = std::clamp(scale, 1, 8);
    b->base_w = std::clamp(base_w, 1, static_cast<int>(gba::GbaPpu::kMaxRenderWidth));
    b->base_h = std::clamp(base_h, 1, static_cast<int>(gba::GbaPpu::kMaxRenderHeight));
    b->requested_base_w.store(b->base_w, std::memory_order_release);
    b->requested_base_h.store(b->base_h, std::memory_order_release);
    if (title && *title) b->title = title;
    b->linear_filter.store(linear_filter, std::memory_order_release);
    b->color_lut = std::make_unique<runtime::ColorLut>(parse_color_settings(screen));
    load_xr_view(*b);
    if (const char* e = std::getenv("GBARECOMP_XR_MENU")) b->menu_enabled = (*e == '1');
    if (const char* e = std::getenv("GBARECOMP_XR_GYRO")) b->gyro_enabled = (*e == '1');
    if (b->menu_enabled) b->hud_mode = 1;  // controls are in the menu: no help bar
    if (const char* e = std::getenv("GBARECOMP_XR_SHADOWBOX")) b->pad = std::clamp(std::atoi(e), 0, 64);
    if (const char* e = std::getenv("GBARECOMP_XR_BOXWALLS")) b->box_walls = (*e == '1');
    if (const char* e = std::getenv("GBARECOMP_XR_WINDOW")) b->window_clip = (*e == '1');
    if (const char* e = std::getenv("GBARECOMP_XR_DARK")) b->start_dark = (*e == '1');
    if (const char* e = std::getenv("GBARECOMP_XR_MARGIN_RENDER"); e && *e == '1' && b->pad > 0 &&
        b->base_w > 2 * b->pad) {
        // The runtime renders 240 + 2*pad wide; the screen itself stays 240 px.
        b->wide_margin = b->pad;
        b->base_w -= 2 * b->pad;
        b->requested_base_w.store(b->base_w, std::memory_order_release);
    }
    load_xr_prefs(*b);
    setenv("GBARECOMP_AUTO_RESUME", b->auto_resume ? "1" : "0", 1);
    if (const char* e = std::getenv("GBARECOMP_XR_TITLE"); e && *e) b->hud_title = e;

    JNIEnv* env = static_cast<JNIEnv*>(SDL_AndroidGetJNIEnv());
    jobject local_activity = static_cast<jobject>(SDL_AndroidGetActivity());
    JavaVM* vm = nullptr;
    if (env) env->GetJavaVM(&vm);
    b->vm = vm;
    if (env && local_activity) {
        b->activity_global_ref = env->NewGlobalRef(local_activity);
        env->DeleteLocalRef(local_activity);
    }

    b->xr_thread = std::thread([ptr = b.get()]() { xr_thread_main(*ptr); });
    {
        std::unique_lock<std::mutex> lk(b->xr_init_mtx);
        b->xr_init_cv.wait(lk, [&]() { return b->xr_init_done; });
    }
    if (env && b->activity_global_ref) {
        env->DeleteGlobalRef(b->activity_global_ref);
        b->activity_global_ref = nullptr;
    }
    if (!b->xr_init_ok) {
        b->stop_xr_thread.store(true, std::memory_order_release);
        if (b->xr_thread.joinable()) b->xr_thread.join();
        return false;
    }
    init_sdl_audio(*b);

    impl_ = b.release();
    open_ = true;
    g_quest_window = this;
    return true;
}

void HostWindow::close() {
    if (g_quest_window == this) g_quest_window = nullptr;
    if (!impl_) {
        open_ = false;
        return;
    }
    auto* b = as_backend(impl_);
    b->stop_xr_thread.store(true, std::memory_order_release);
    if (b->xr_thread.joinable()) {
        b->xr_thread.join();
    }
    if (b->audio_dev != 0) {
        SDL_CloseAudioDevice(b->audio_dev);
        b->audio_dev = 0;
    }
    if (b->bridge_ready) {
        if (b->audio_mtx) SDL_LockMutex(b->audio_mtx);
        rab_free(&b->bridge);
        b->bridge_ready = false;
        if (b->audio_mtx) SDL_UnlockMutex(b->audio_mtx);
    }
    if (b->audio_mtx) {
        SDL_DestroyMutex(b->audio_mtx);
        b->audio_mtx = nullptr;
    }
    if (b->sdl_audio_init) {
        SDL_QuitSubSystem(SDL_INIT_AUDIO);
        b->sdl_audio_init = false;
    }
    delete b;
    impl_ = nullptr;
    open_ = false;
}

void HostWindow::set_title(const char* title) {
    if (impl_ && title) as_backend(impl_)->title = title;
}

bool HostWindow::set_surface_size(int base_w, int base_h) {
    if (!impl_ || base_w <= 0 || base_h <= 0 ||
        base_w > static_cast<int>(gba::GbaPpu::kMaxRenderWidth) ||
        base_h > static_cast<int>(gba::GbaPpu::kMaxRenderHeight)) {
        return false;
    }
    auto* b = as_backend(impl_);
    b->requested_base_w.store(base_w, std::memory_order_release);
    b->requested_base_h.store(base_h, std::memory_order_release);
    return true;
}

bool HostWindow::drawable_size(int* width, int* height) const {
    if (!impl_ || !width || !height) return false;
    const auto* b = as_backend(impl_);
    *width = static_cast<int>(b->reported_swap_w.load(std::memory_order_acquire));
    *height = static_cast<int>(b->reported_swap_h.load(std::memory_order_acquire));
    return true;
}

void HostWindow::load_input_config(const char* /*dir*/, bool assist_tools_default,
                                   int fast_forward_multiplier_default) {
    if (!impl_) return;
    auto* b = as_backend(impl_);
    b->assist_tools = assist_tools_default;
    b->fast_forward_mult = std::clamp(fast_forward_multiplier_default, 2, 16);
}

void HostWindow::set_fullscreen(int /*mode*/) {}
int HostWindow::fullscreen() const { return 2; }
void HostWindow::adjust_scale(int delta) {
    if (!impl_) return;
    auto* b = as_backend(impl_);
    b->scale = std::clamp(b->scale + delta, 1, 8);
    b->screen_width_m.store(0.85f + 0.19f * static_cast<float>(b->scale), std::memory_order_release);
    save_xr_view(*b);
}
int HostWindow::window_scale() const { return impl_ ? as_backend(impl_)->scale : 3; }
void HostWindow::set_volume(int pct) {
    if (impl_) as_backend(impl_)->volume = std::clamp(pct, 0, 100);
}
int HostWindow::volume() const { return impl_ ? as_backend(impl_)->volume : 100; }
void HostWindow::set_linear_filter(bool enabled) {
    if (impl_) as_backend(impl_)->linear_filter.store(enabled, std::memory_order_release);
}
bool HostWindow::linear_filter() const {
    return impl_ ? as_backend(impl_)->linear_filter.load(std::memory_order_acquire) : false;
}
void HostWindow::set_audio_enabled(bool enabled) {
    if (!impl_) return;
    auto* b = as_backend(impl_);
    b->audio_enabled = enabled;
    if (b->audio_dev != 0) {
        if (!enabled) {
            SDL_PauseAudioDevice(b->audio_dev, 1);
            reset_rab_bridge(*b);
        } else {
            SDL_PauseAudioDevice(b->audio_dev, 0);
        }
    }
}
bool HostWindow::audio_enabled() const {
    return impl_ ? as_backend(impl_)->audio_enabled : true;
}
void HostWindow::set_resize_driven_view(bool /*enabled*/) {}
void HostWindow::set_fill_display(bool /*enabled*/) {}
bool HostWindow::fill_display() const { return false; }
void HostWindow::set_integer_scale(bool /*enabled*/, bool /*persist*/) {}
bool HostWindow::integer_scale() const { return true; }
bool HostWindow::saved_audio_enhanced(bool& /*enabled*/) const { return false; }
void HostWindow::save_audio_enhanced(bool /*enabled*/) {}
void HostWindow::set_menu_scancode(int /*scancode*/) {}

#if defined(GBARECOMP_RUNTIME_UI)
void HostWindow::set_runtime_ui(RecompRuntimeUi* /*ui*/) {}
bool HostWindow::set_runtime_ui_font(const char* /*path*/) { return false; }
#endif

void HostWindow::configure_touch(bool /*policy*/, std::uint32_t /*claims*/,
                                 int /*pad_default*/, bool /*emulate_touch*/) {}
void HostWindow::set_touch_config_dir(const char* /*dir*/) {}
bool HostWindow::touch_pad_visible() const { return false; }
void HostWindow::set_touch_pad_visible(bool /*visible*/) {}
void HostWindow::set_view_margins(std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t) {}
void HostWindow::set_host_overlay(void (*)(HostOverlay*)) {}
float HostWindow::px_per_mm() const { return 96.0f / 25.4f; }
void HostWindow::set_presentation_anchor_top(bool /*anchor_top*/) {}

bool HostWindow::wait_for_foreground() {
    if (!open_ || !impl_) return false;
    auto* b = as_backend(impl_);
    if (b->audio_dev != 0) {
        SDL_PauseAudioDevice(b->audio_dev, 1);
        reset_rab_bridge(*b);
    }
    while (!b->exit_requested.load(std::memory_order_acquire) &&
           !b->session_running.load(std::memory_order_acquire)) {
        SDL_Event sev;
        while (SDL_PollEvent(&sev)) {
            if (sev.type == SDL_QUIT || sev.type == SDL_APP_TERMINATING) {
                b->exit_requested.store(true, std::memory_order_release);
            }
        }
        if (b->exit_requested.load(std::memory_order_acquire)) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(15));
    }
    if (b->audio_dev != 0 && b->audio_enabled) {
        SDL_PauseAudioDevice(b->audio_dev, 0);
    }
    return !b->exit_requested.load(std::memory_order_acquire);
}

void HostWindow::haptic_pulse(int duration_ms, float strength) {
    if (!open_ || !impl_ || duration_ms <= 0 || strength <= 0.0f) return;
    auto* b = as_backend(impl_);
    const XrDuration ns = static_cast<XrDuration>(std::clamp(duration_ms, 1, 2000)) * 1000000LL;
    std::lock_guard<std::mutex> lk(b->haptic_mtx);
    b->haptic_pulse_queued = true;
    b->queued_haptic_ns = ns;
    b->queued_haptic_amp = strength;
}

void HostWindow::set_rumble(bool on) {
    if (!open_ || !impl_) return;
    auto* b = as_backend(impl_);
    std::lock_guard<std::mutex> lk(b->haptic_mtx);
    b->desired_rumble = on;
}

void HostWindow::set_fps_readout(bool on) {
    if (impl_) as_backend(impl_)->fps_readout = on;
}
bool HostWindow::fps_readout() const {
    return impl_ ? as_backend(impl_)->fps_readout : false;
}
bool HostWindow::assist_tools_enabled() const {
    return impl_ ? as_backend(impl_)->assist_tools : false;
}
int HostWindow::fast_forward_multiplier() const {
    return impl_ ? as_backend(impl_)->fast_forward_mult : 4;
}

bool HostWindow::display_timing(std::chrono::nanoseconds* /*period*/,
                                std::chrono::steady_clock::time_point* /*vblank*/) const {
    // Dedicated OpenXR compositor thread runs 72 Hz independently; let GBA FramePacer hold 59.73 Hz.
    return false;
}

void HostWindow::present(const uint8_t* rgb888) {
    if (!open_ || !impl_) return;
    auto* b = as_backend(impl_);
    stage_frame(*b, rgb888,
                static_cast<unsigned>(b->requested_base_w.load(std::memory_order_acquire)),
                static_cast<unsigned>(b->requested_base_h.load(std::memory_order_acquire)));
}

void HostWindow::present(const uint8_t* rgb888, unsigned sampled_w, unsigned sampled_h) {
    if (!open_ || !impl_) return;
    auto* b = as_backend(impl_);
    stage_frame(*b, rgb888, sampled_w, sampled_h, 1, false);
}

void HostWindow::present_scaled(const uint8_t* rgb888, int factor) {
    if (!open_ || !impl_ || factor <= 0) return;
    auto* b = as_backend(impl_);
    stage_frame(*b, rgb888,
                static_cast<unsigned>(b->requested_base_w.load(std::memory_order_acquire) * factor),
                static_cast<unsigned>(b->requested_base_h.load(std::memory_order_acquire) * factor),
                1, false);
}

void HostWindow::present_layered_3d(const uint8_t* planes_rgba, unsigned width,
                                      unsigned height, unsigned plane_count,
                                      const uint8_t* masked_planes_rgba, bool masked_scene) {
    if (!open_ || !impl_) return;
    auto* b = as_backend(impl_);
    stage_frame(*b, planes_rgba, width, height, plane_count, true,
                masked_planes_rgba, masked_scene);
}

void HostWindow::push_audio_samples(const int16_t* samples, std::size_t count) {
    if (!open_ || !impl_ || !samples || count == 0) return;
    push_audio_to_bridge(*as_backend(impl_), samples, count, 65536u);
}

void HostWindow::push_audio_block(const int16_t* samples, std::size_t count, uint32_t rate) {
    if (!open_ || !impl_ || !samples || count == 0 || rate == 0) return;
    push_audio_to_bridge(*as_backend(impl_), samples, count, rate);
}

void HostWindow::service_events() {}

HostWindow::Events HostWindow::pump() {
    Events ev;
    if (!open_ || !impl_) {
        ev.quit = true;
        return ev;
    }
    auto* b = as_backend(impl_);

    if (b->background_pending.exchange(false, std::memory_order_acq_rel)) {
        ev.enter_background = true;
    }

    SDL_Event sev;
    while (SDL_PollEvent(&sev)) {
        if (sev.type == SDL_QUIT) {
            b->exit_requested.store(true, std::memory_order_release);
        } else if (sev.type == SDL_APP_WILLENTERBACKGROUND ||
                   sev.type == SDL_APP_DIDENTERBACKGROUND) {
            ev.enter_background = true;
        } else if (sev.type == SDL_APP_TERMINATING) {
            ev.terminating = true;
            b->exit_requested.store(true, std::memory_order_release);
        }
    }

    if (b->exit_requested.load(std::memory_order_acquire)) {
        ev.quit = true;
        return ev;
    }

    ev.keyinput = b->latest_keyinput.load(std::memory_order_acquire);
    if (b->gyro_enabled) ev.gyro_rate_z = b->latest_gyro_rate_z.load(std::memory_order_acquire);
    ev.save_slot = b->pending_save_slot.exchange(0, std::memory_order_acq_rel);
    ev.load_slot = b->pending_load_slot.exchange(0, std::memory_order_acq_rel);
    if (b->pending_pause_toggles.load(std::memory_order_acquire) > 0) {
        b->pending_pause_toggles.fetch_sub(1, std::memory_order_acq_rel);
        ev.toggle_pause = true;
    }
    return ev;
}

void host_haptic_pulse(int duration_ms, float strength) {
    if (g_quest_window) g_quest_window->haptic_pulse(duration_ms, strength);
}

void host_request_settings_menu() {}

}  // namespace gbarecomp

extern "C" {

JNIEXPORT void JNICALL
Java_org_gbarecomp_GbaNative_setSafeInsets(JNIEnv*, jclass, jint, jint, jint, jint) {}

JNIEXPORT void JNICALL
Java_org_gbarecomp_GbaNative_setDisplayDpi(JNIEnv*, jclass, jfloat, jfloat) {}

}  // extern "C"

#endif  // __ANDROID__ && GBARECOMP_QUEST_XR
