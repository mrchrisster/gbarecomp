// host_window_quest.cpp — Meta Quest OpenXR HostWindow backend.
//
// Renders the GBA framebuffer onto a crisp 4x integer-upscaled OpenXR quad
// compositor layer (with optional XR_FB_passthrough mixed reality + a tiny
// toggleable control-legend HUD quad below the screen) and translates Quest
// Touch Plus motion, buttons, thumbsticks, and haptics directly into GBA inputs
// and WarioWare: Twisted! cartridge gyro + rumble feedback.

#if defined(__ANDROID__) && defined(GBARECOMP_QUEST_XR)

#include "host_window.h"

#include "color_lut.h"
#include "gba_ppu.h"

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
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace gbarecomp {

namespace {

constexpr const char* kLogTag = "WarioWareXR";

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

constexpr uint32_t kUpscaleFactor = 4;
constexpr unsigned kLayeredPlaneCount = 5;
constexpr int kHudWidth = 864;
constexpr int kHudHeight = 84;
constexpr float kThumbstickDeadzone = 0.35f;
constexpr float kTriggerThreshold = 0.45f;
constexpr float kGripThreshold = 0.55f;
constexpr float kGyroDriftDeadzoneRadPerSec = 0.025f;
// Real GBA SP twisted in-hand at 1 rad/s produces ~49 ADC LSB vs runtime.cpp's
// baseline 128 * 0.25 * 0.75 = 24 LSB/rad/s, so 2.0x maps Touch Plus wrist
// twist ~1:1 to physical cartridge turn angles.
constexpr float kQuestGyroRateScale = 2.0f;

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

struct Backend {
    // GBA logical framebuffer size
    int base_w = 240;
    int base_h = 160;
    uint32_t swap_w = 960;
    uint32_t swap_h = 640;
    std::string title = "WarioWare: Twisted! VR";

    // Color LUT + CPU conversion staging buffers
    std::unique_ptr<runtime::ColorLut> color_lut;
    std::vector<uint8_t> graded_fb;
    std::vector<uint8_t> color_lut_rgb;
    std::vector<uint8_t> rgba_composite_fb;
    std::vector<uint8_t> rgba_fb;

    // EGL state
    EGLDisplay egl_display = EGL_NO_DISPLAY;
    EGLConfig egl_config = nullptr;
    EGLContext egl_context = EGL_NO_CONTEXT;
    EGLSurface egl_surface = EGL_NO_SURFACE;
    GLuint src_tex = 0;
    GLuint src_fbo = 0;
    GLuint dst_fbo = 0;

    // OpenXR core objects
    XrInstance instance = XR_NULL_HANDLE;
    XrSystemId system_id = XR_NULL_SYSTEM_ID;
    XrSession session = XR_NULL_HANDLE;
    XrSpace local_space = XR_NULL_HANDLE;
    XrSpace view_space = XR_NULL_HANDLE;
    XrSessionState session_state = XR_SESSION_STATE_UNKNOWN;
    bool session_running = false;
    bool exit_requested = false;
    bool background_pending = false;

    // Layered GBA swapchains & images (0 = Backdrop/2D composite, 1..4 = Priorities 3..0)
    XrSwapchain screen_swapchains[kLayeredPlaneCount] = {};
    std::vector<XrSwapchainImageOpenGLESKHR> screen_images[kLayeredPlaneCount];
    bool active_planes[kLayeredPlaneCount] = {true, false, false, false, false};
    bool last_frame_layered = false;
    int depth_mode = 1; // 0 = 2D FLAT, 1 = 3D (6CM), 2 = 3D DEEP (11CM)
    bool has_submitted_frame = false;
    XrTime last_predicted_time = 0;

    // HUD swapchain & state
    XrSwapchain hud_swapchain = XR_NULL_HANDLE;
    std::vector<XrSwapchainImageOpenGLESKHR> hud_images;
    bool hud_dirty = true;
    int hud_mode = 0; // 0 = MR/Void + HUD, 1 = MR/Void without HUD, 2 = Dark Void only

    // Optional XR_FB_passthrough
    bool passthrough_ext = false;
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
    XrPosef screen_pose = {{0.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, -1.45f}};
    bool pose_initialized = false;
    float screen_distance = 1.45f;
    float screen_width_m = 1.26f;

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

    int active_gyro_hand = 1; // 1 = right controller (default), 0 = left controller
    bool prev_r_stick_click = false;
    bool prev_l_stick_click = false;
    bool prev_both_stick_clicks = false;
    bool stick_chord_consumed = false;
    bool prev_l_menu_click = false;
    bool gyro_motion_logged = false;

    // Cartridge rumble state
    bool rumble_active = false;
    std::chrono::steady_clock::time_point last_rumble_refresh{};

    // SDL audio (OpenSL ES / AAudio)
    SDL_AudioDeviceID audio_dev = 0;
    bool sdl_audio_init = false;
    bool audio_enabled = true;
    int volume = 100;
    bool audio_direct_started = false;
    std::vector<int16_t> volume_buf;

    // Host flags queried by runtime.cpp
    int scale = 3;
    bool linear_filter = false;
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
        }
        b.src_tex = b.src_fbo = b.dst_fbo = 0;
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
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
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
    if (!found_rgba8 && found_srgb) chosen = GL_SRGB8_ALPHA8;
    else if (!found_rgba8 && !formats.empty()) chosen = formats[0];

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
                  "WarioWare Twisted Controls");
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
    const XrVector3f fwd = rotate_vec(loc.pose.orientation, 0.0f, 0.0f, -1.0f);
    float fx = fwd.x;
    float fz = fwd.z;
    const float len = std::sqrt(fx * fx + fz * fz);
    if (len < 1e-4f) {
        fx = 0.0f;
        fz = -1.0f;
    } else {
        fx /= len;
        fz /= len;
    }
    b.screen_pose.position = {
        loc.pose.position.x + fx * b.screen_distance,
        loc.pose.position.y - 0.04f,
        loc.pose.position.z + fz * b.screen_distance,
    };
    const float yaw = std::atan2(-fx, -fz);
    b.screen_pose.orientation = {0.0f, std::sin(yaw * 0.5f), 0.0f, std::cos(yaw * 0.5f)};
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
                rgba[idx + 3] = 220;
            } else {
                rgba[idx + 0] = 18;
                rgba[idx + 1] = 22;
                rgba[idx + 2] = 32;
                rgba[idx + 3] = 215;
            }
        }
    }

    const char* depth_name = "3D (6CM)";
    if (b.depth_mode == 0) depth_name = "2D FLAT";
    else if (b.depth_mode == 2) depth_name = "3D DEEP";

    char line1[128];
    std::snprintf(line1, sizeof(line1),
                  "WARIOWARE TWISTED VR | GYRO:%s | DEPTH:%s | MR:%s",
                  b.active_gyro_hand == 1 ? "RIGHT" : "LEFT",
                  depth_name,
                  (b.passthrough_supported && b.passthrough_active) ? "ON" : "OFF");
    draw_hud_text(rgba, kHudWidth, kHudHeight, 14, 10, 2, 0xFAD23CFFu, line1);
    draw_hud_text(rgba, kHudWidth, kHudHeight, 14, 34, 2, 0xFFFFFFFFu,
                  "TWIST: GYRO  A/RT: A  B: B  GRIPS/LT: R/L  L+R-CLK: 3D DEPTH");
    draw_hud_text(rgba, kHudWidth, kHudHeight, 14, 58, 2, 0xB8C4E0FFu,
                  "STICK: DPAD  X/Y: START/SEL  R-CLK: CENTER  L-CLK: HAND  MENU: MR");

    glBindTexture(GL_TEXTURE_2D, b.hud_images[img_idx].image);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, kHudWidth, kHudHeight,
                    GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());
    glBindTexture(GL_TEXTURE_2D, 0);

    XrSwapchainImageReleaseInfo rel{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
    xrReleaseSwapchainImage(b.hud_swapchain, &rel);
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
            case XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED: {
                const auto* ssc = reinterpret_cast<const XrEventDataSessionStateChanged*>(&ev);
                b.session_state = ssc->state;
                XR_LOGI("OpenXR session state -> %d", static_cast<int>(ssc->state));
                if (ssc->state == XR_SESSION_STATE_READY) {
                    XrSessionBeginInfo sbi{XR_TYPE_SESSION_BEGIN_INFO};
                    sbi.primaryViewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
                    if (XR_SUCCEEDED(xrBeginSession(b.session, &sbi))) {
                        b.session_running = true;
                        apply_passthrough_state(b);
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
    jobject activity = static_cast<jobject>(SDL_AndroidGetActivity());
    JavaVM* vm = nullptr;
    if (env) env->GetJavaVM(&vm);
    if (!vm || !activity) {
        XR_LOGE("Missing JavaVM (%p) or Activity (%p)", static_cast<void*>(vm),
                static_cast<void*>(activity));
        if (env && activity) env->DeleteLocalRef(activity);
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
            env->DeleteLocalRef(activity);
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
    env->DeleteLocalRef(activity);
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
                        b.passthrough_active = true; // MR on by default for safe wrist twisting
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

    b.swap_w = static_cast<uint32_t>(b.base_w) * kUpscaleFactor;
    b.swap_h = static_cast<uint32_t>(b.base_h) * kUpscaleFactor;
    for (unsigned p = 0; p < kLayeredPlaneCount; ++p) {
        if (!create_swapchain_with_images(b, b.swap_w, b.swap_h,
                                          b.screen_swapchains[p], b.screen_images[p])) {
            return false;
        }
    }
    if (!create_swapchain_with_images(b, kHudWidth, kHudHeight, b.hud_swapchain, b.hud_images)) {
        XR_LOGW("HUD swapchain creation failed; continuing without HUD quad");
    }
    allocate_source_texture(b, b.base_w, b.base_h);
    b.rgba_fb.assign(static_cast<size_t>(b.base_w) * b.base_h * 4, 255);

    if (!init_actions(b)) return false;

    XR_LOGI("OpenXR initialized (swapchain=%ux%u planes=%u passthrough=%s)",
            b.swap_w, b.swap_h, kLayeredPlaneCount,
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
    if (b.hud_swapchain != XR_NULL_HANDLE) {
        xrDestroySwapchain(b.hud_swapchain);
        b.hud_swapchain = XR_NULL_HANDLE;
    }
    for (unsigned p = 0; p < kLayeredPlaneCount; ++p) {
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
    if (depth_mode == 1) return 0.06f;
    if (depth_mode == 2) return 0.11f;
    return 0.0f;
}

bool blit_rgba_to_swapchain(Backend& b, unsigned plane_idx, int iw, int ih,
                            const uint8_t* bottom_up_rgba, bool nearest_only) {
    if (plane_idx >= kLayeredPlaneCount || b.screen_swapchains[plane_idx] == XR_NULL_HANDLE) {
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
        const GLenum filter = (!nearest_only && b.linear_filter) ? GL_LINEAR : GL_NEAREST;
        glBlitFramebuffer(0, 0, iw, ih,
                          0, 0, static_cast<GLint>(b.swap_w), static_cast<GLint>(b.swap_h),
                          GL_COLOR_BUFFER_BIT, filter);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
        ok = true;
    }
    XrSwapchainImageReleaseInfo rel{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
    xrReleaseSwapchainImage(b.screen_swapchains[plane_idx], &rel);
    return ok;
}

void flip_and_grade_rgba_plane(Backend& b, const uint8_t* plane_rgba, int iw, int ih,
                               bool* out_has_opaque) {
    const size_t px_count = static_cast<size_t>(iw) * ih;
    const bool use_lut = (b.color_lut && !b.color_lut->is_passthrough());
    const uint8_t* rgb_src = nullptr;
    if (use_lut) {
        b.color_lut_rgb.resize(px_count * 3);
        b.graded_fb.resize(px_count * 3);
        for (size_t i = 0; i < px_count; ++i) {
            b.color_lut_rgb[i * 3 + 0] = plane_rgba[i * 4 + 0];
            b.color_lut_rgb[i * 3 + 1] = plane_rgba[i * 4 + 1];
            b.color_lut_rgb[i * 3 + 2] = plane_rgba[i * 4 + 2];
        }
        b.color_lut->map_rgb888(b.color_lut_rgb.data(), b.graded_fb.data(), iw, ih);
        rgb_src = b.graded_fb.data();
    }

    bool any_visible = false;
    for (int y = 0; y < ih; ++y) {
        const size_t in_row_px = static_cast<size_t>(y) * iw;
        uint8_t* out_row = b.rgba_fb.data() +
                           static_cast<size_t>(ih - 1 - y) * iw * 4;
        for (int x = 0; x < iw; ++x) {
            const size_t p_off = (in_row_px + x) * 4;
            const uint8_t a = plane_rgba[p_off + 3];
            if (a != 0) any_visible = true;
            if (rgb_src) {
                const size_t g_off = (in_row_px + x) * 3;
                out_row[x * 4 + 0] = rgb_src[g_off + 0];
                out_row[x * 4 + 1] = rgb_src[g_off + 1];
                out_row[x * 4 + 2] = rgb_src[g_off + 2];
            } else {
                out_row[x * 4 + 0] = plane_rgba[p_off + 0];
                out_row[x * 4 + 1] = plane_rgba[p_off + 1];
                out_row[x * 4 + 2] = plane_rgba[p_off + 2];
            }
            out_row[x * 4 + 3] = a;
        }
    }
    if (out_has_opaque) *out_has_opaque = any_visible;
}

void submit_frame(Backend& b, const uint8_t* src_pixels, unsigned src_w, unsigned src_h,
                  unsigned plane_count = 1, bool is_rgba_planes = false) {
    if (b.session == XR_NULL_HANDLE || !b.session_running) return;
    if (b.egl_display != EGL_NO_DISPLAY) {
        eglMakeCurrent(b.egl_display, b.egl_surface, b.egl_surface, b.egl_context);
    }

    XrFrameWaitInfo wait_info{XR_TYPE_FRAME_WAIT_INFO};
    XrFrameState frame_state{XR_TYPE_FRAME_STATE};
    if (XR_FAILED(xrWaitFrame(b.session, &wait_info, &frame_state))) return;
    b.last_predicted_time = frame_state.predictedDisplayTime;

    if (!b.pose_initialized && b.session_state >= XR_SESSION_STATE_SYNCHRONIZED) {
        recenter_screen_to_head(b, frame_state.predictedDisplayTime);
    }

    XrFrameBeginInfo begin_info{XR_TYPE_FRAME_BEGIN_INFO};
    if (XR_FAILED(xrBeginFrame(b.session, &begin_info))) return;

    std::vector<const XrCompositionLayerBaseHeader*> layers;
    XrCompositionLayerPassthroughFB pt_layer{XR_TYPE_COMPOSITION_LAYER_PASSTHROUGH_FB};
    XrCompositionLayerQuad screen_quads[kLayeredPlaneCount]{};
    for (auto& sq : screen_quads) sq.type = XR_TYPE_COMPOSITION_LAYER_QUAD;
    XrCompositionLayerQuad hud_quad{XR_TYPE_COMPOSITION_LAYER_QUAD};

    if (frame_state.shouldRender) {
        if (src_pixels && src_w > 0 && src_h > 0) {
            const int iw = static_cast<int>(src_w);
            const int ih = static_cast<int>(src_h);
            if (iw != b.base_w || ih != b.base_h) {
                b.base_w = iw;
                b.base_h = ih;
                allocate_source_texture(b, iw, ih);
                b.rgba_fb.assign(static_cast<size_t>(iw) * ih * 4, 255);
            }

            if (is_rgba_planes && plane_count > 1) {
                const unsigned used_planes = std::min(plane_count, kLayeredPlaneCount);
                const size_t plane_bytes = static_cast<size_t>(iw) * ih * 4;
                if (b.depth_mode == 0) {
                    // 2D FLAT mode: composite all planes back-to-front on CPU into swapchain 0.
                    b.rgba_composite_fb.assign(src_pixels, src_pixels + plane_bytes);
                    for (unsigned p = 1; p < used_planes; ++p) {
                        const uint8_t* layer = src_pixels + p * plane_bytes;
                        for (size_t i = 0; i < plane_bytes; i += 4) {
                            const uint32_t a = layer[i + 3];
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
                    flip_and_grade_rgba_plane(b, b.rgba_composite_fb.data(), iw, ih, nullptr);
                    if (blit_rgba_to_swapchain(b, 0, iw, ih, b.rgba_fb.data(), false)) {
                        glFlush();
                        b.active_planes[0] = true;
                        for (unsigned p = 1; p < kLayeredPlaneCount; ++p) b.active_planes[p] = false;
                        b.last_frame_layered = false;
                        b.has_submitted_frame = true;
                    }
                } else {
                    bool any_blitted = false;
                    for (unsigned p = 0; p < kLayeredPlaneCount; ++p) {
                        if (p >= used_planes) {
                            b.active_planes[p] = false;
                            continue;
                        }
                        bool has_visible = false;
                        flip_and_grade_rgba_plane(b, src_pixels + p * plane_bytes, iw, ih,
                                                  &has_visible);
                        if (p == 0) has_visible = true; // Backdrop plane always submitted
                        if (!has_visible) {
                            b.active_planes[p] = false;
                            continue;
                        }
                        if (blit_rgba_to_swapchain(b, p, iw, ih, b.rgba_fb.data(), true)) {
                            b.active_planes[p] = true;
                            any_blitted = true;
                        } else {
                            b.active_planes[p] = false;
                        }
                    }
                    if (any_blitted) {
                        glFlush();
                        b.last_frame_layered = true;
                        b.has_submitted_frame = true;
                    }
                }
            } else {
                const uint8_t* graded = src_pixels;
                if (b.color_lut && !b.color_lut->is_passthrough()) {
                    b.graded_fb.resize(static_cast<size_t>(iw) * ih * 3);
                    b.color_lut->map_rgb888(src_pixels, b.graded_fb.data(), iw, ih);
                    graded = b.graded_fb.data();
                }
                // Write top-down GBA rows bottom-to-top into GL texture so OpenXR
                // quad UVs display upright without shader passes.
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
                    b.active_planes[0] = true;
                    for (unsigned p = 1; p < kLayeredPlaneCount; ++p) b.active_planes[p] = false;
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
            if (b.last_frame_layered && b.depth_mode != 0) {
                const float step_m = depth_step_meters(b.depth_mode);
                const float d_ref = std::max(0.5f, b.screen_distance);
                for (unsigned p = 0; p < kLayeredPlaneCount; ++p) {
                    if (!b.active_planes[p]) continue;
                    // Plane 0 (Backdrop) farthest (-2*step), Plane 4 (Prio 0) closest (+2*step).
                    // Perspective-match size so planes align pixel-for-pixel from head origin.
                    const float z_local = (static_cast<float>(p) - 2.0f) * step_m;
                    const float d_p = std::max(0.25f, d_ref - z_local);
                    const float scale_p = d_p / d_ref;
                    const XrVector3f z_off = rotate_vec(b.screen_pose.orientation, 0.0f, 0.0f, z_local);
                    XrCompositionLayerQuad& q = screen_quads[p];
                    q.layerFlags = (p == 0)
                                       ? 0
                                       : (XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT |
                                          XR_COMPOSITION_LAYER_UNPREMULTIPLIED_ALPHA_BIT);
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
                    q.size = {b.screen_width_m * scale_p, b.screen_width_m * aspect * scale_p};
                    layers.push_back(reinterpret_cast<const XrCompositionLayerBaseHeader*>(&q));
                }
            } else {
                XrCompositionLayerQuad& q = screen_quads[0];
                q.layerFlags = 0;
                q.space = b.local_space;
                q.eyeVisibility = XR_EYE_VISIBILITY_BOTH;
                q.subImage.swapchain = b.screen_swapchains[0];
                q.subImage.imageRect = {{0, 0},
                                        {static_cast<int32_t>(b.swap_w),
                                         static_cast<int32_t>(b.swap_h)}};
                q.subImage.imageArrayIndex = 0;
                q.pose = b.screen_pose;
                q.size = {b.screen_width_m, b.screen_width_m * aspect};
                layers.push_back(reinterpret_cast<const XrCompositionLayerBaseHeader*>(&q));
            }

            if (b.hud_swapchain != XR_NULL_HANDLE && b.hud_mode == 0) {
                upload_hud_image(b);
                const float screen_h = b.screen_width_m * aspect;
                const float hud_w = 1.24f;
                const float hud_h = hud_w * (static_cast<float>(kHudHeight) / kHudWidth);
                // Place HUD bar neatly centered below the virtual screen stack.
                const XrVector3f up_off = rotate_vec(b.screen_pose.orientation, 0.0f,
                                                     -(screen_h * 0.5f + hud_h * 0.5f + 0.06f),
                                                     0.05f);
                hud_quad.layerFlags = XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT |
                                      XR_COMPOSITION_LAYER_UNPREMULTIPLIED_ALPHA_BIT;
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

void init_sdl_audio(Backend& b) {
    if (SDL_WasInit(SDL_INIT_AUDIO) == 0) {
        if (SDL_InitSubSystem(SDL_INIT_AUDIO) != 0) {
            XR_LOGW("SDL_InitSubSystem(AUDIO) failed: %s", SDL_GetError());
            return;
        }
        b.sdl_audio_init = true;
    }
    SDL_AudioSpec want{};
    want.freq = 65536;
    want.format = AUDIO_S16SYS;
    want.channels = 1;
    want.samples = 1024;
    SDL_AudioSpec got{};
    b.audio_dev = SDL_OpenAudioDevice(nullptr, 0, &want, &got, 0);
    if (b.audio_dev != 0) {
        XR_LOGI("SDL audio opened: driver=%s rate=%dHz samples=%u",
                SDL_GetCurrentAudioDriver() ? SDL_GetCurrentAudioDriver() : "?",
                got.freq, static_cast<unsigned>(got.samples));
    } else {
        XR_LOGW("SDL_OpenAudioDevice failed: %s", SDL_GetError());
    }
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
    if (title && *title) b->title = title;
    b->linear_filter = linear_filter;
    b->color_lut = std::make_unique<runtime::ColorLut>(parse_color_settings(screen));

    if (!init_egl(*b) || !init_openxr(*b)) {
        shutdown_openxr(*b);
        shutdown_egl(*b);
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
    if (b->audio_dev != 0) {
        SDL_CloseAudioDevice(b->audio_dev);
        b->audio_dev = 0;
    }
    if (b->sdl_audio_init) {
        SDL_QuitSubSystem(SDL_INIT_AUDIO);
        b->sdl_audio_init = false;
    }
    shutdown_openxr(*b);
    shutdown_egl(*b);
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
    b->base_w = base_w;
    b->base_h = base_h;
    return true;
}

bool HostWindow::drawable_size(int* width, int* height) const {
    if (!impl_ || !width || !height) return false;
    const auto* b = as_backend(impl_);
    *width = static_cast<int>(b->swap_w);
    *height = static_cast<int>(b->swap_h);
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
    b->screen_width_m = 0.75f + 0.17f * static_cast<float>(b->scale);
}
int HostWindow::window_scale() const { return impl_ ? as_backend(impl_)->scale : 3; }
void HostWindow::set_volume(int pct) {
    if (impl_) as_backend(impl_)->volume = std::clamp(pct, 0, 100);
}
int HostWindow::volume() const { return impl_ ? as_backend(impl_)->volume : 100; }
void HostWindow::set_linear_filter(bool enabled) {
    if (impl_) as_backend(impl_)->linear_filter = enabled;
}
bool HostWindow::linear_filter() const {
    return impl_ ? as_backend(impl_)->linear_filter : false;
}
void HostWindow::set_audio_enabled(bool enabled) {
    if (!impl_) return;
    auto* b = as_backend(impl_);
    b->audio_enabled = enabled;
    if (b->audio_dev != 0 && !enabled) {
        SDL_ClearQueuedAudio(b->audio_dev);
        SDL_PauseAudioDevice(b->audio_dev, 1);
        b->audio_direct_started = false;
    }
}
bool HostWindow::audio_enabled() const {
    return impl_ ? as_backend(impl_)->audio_enabled : true;
}
void HostWindow::set_resize_driven_view(bool /*enabled*/) {}

#if defined(GBARECOMP_RUNTIME_UI)
void HostWindow::set_runtime_ui(RecompRuntimeUi* /*ui*/) {}
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
        SDL_ClearQueuedAudio(b->audio_dev);
        b->audio_direct_started = false;
    }
    while (!b->exit_requested && !b->session_running) {
        handle_xr_events(*b);
        SDL_Event sev;
        while (SDL_PollEvent(&sev)) {
            if (sev.type == SDL_QUIT || sev.type == SDL_APP_TERMINATING) {
                b->exit_requested = true;
            }
        }
        if (b->exit_requested) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(15));
    }
    return !b->exit_requested;
}

void HostWindow::haptic_pulse(int duration_ms, float strength) {
    if (!open_ || !impl_ || duration_ms <= 0 || strength <= 0.0f) return;
    auto* b = as_backend(impl_);
    const XrDuration ns = static_cast<XrDuration>(std::clamp(duration_ms, 1, 2000)) * 1000000LL;
    trigger_haptic_ns(*b, b->active_gyro_hand, ns, strength);
}

void HostWindow::set_rumble(bool on) {
    if (!open_ || !impl_) return;
    auto* b = as_backend(impl_);
    const auto now = std::chrono::steady_clock::now();
    if (on) {
        if (!b->rumble_active ||
            (now - b->last_rumble_refresh) >= std::chrono::milliseconds(22)) {
            trigger_haptic_ns(*b, b->active_gyro_hand, 40000000LL, 0.70f);
            b->last_rumble_refresh = now;
        }
        b->rumble_active = true;
    } else if (b->rumble_active) {
        stop_haptic(*b, b->active_gyro_hand);
        b->rumble_active = false;
    }
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

void HostWindow::present(const uint8_t* rgb888) {
    if (!open_ || !impl_) return;
    auto* b = as_backend(impl_);
    submit_frame(*b, rgb888, static_cast<unsigned>(b->base_w), static_cast<unsigned>(b->base_h));
}

void HostWindow::present(const uint8_t* rgb888, unsigned sampled_w, unsigned sampled_h) {
    if (!open_ || !impl_) return;
    auto* b = as_backend(impl_);
    submit_frame(*b, rgb888, sampled_w, sampled_h, 1, false);
}

void HostWindow::present_layered_3d(const uint8_t* planes_rgba, unsigned width,
                                      unsigned height, unsigned plane_count) {
    if (!open_ || !impl_) return;
    auto* b = as_backend(impl_);
    submit_frame(*b, planes_rgba, width, height, plane_count, true);
}

void HostWindow::push_audio_samples(const int16_t* samples, std::size_t count) {
    if (!open_ || !impl_ || !samples || count == 0) return;
    auto* b = as_backend(impl_);
    if (!b->audio_enabled || b->audio_dev == 0) return;
    if (b->volume != 100) {
        b->volume_buf.resize(count);
        for (std::size_t i = 0; i < count; ++i) {
            b->volume_buf[i] = static_cast<int16_t>(
                (static_cast<int32_t>(samples[i]) * b->volume) / 100);
        }
        samples = b->volume_buf.data();
    }
    const Uint32 queued = SDL_GetQueuedAudioSize(b->audio_dev);
    if (b->audio_direct_started && queued == 0) {
        b->audio_direct_started = false;
        SDL_PauseAudioDevice(b->audio_dev, 1);
    }
    // Cap latency at ~180 ms in case the device stalled briefly
    if (queued < 24576) {
        SDL_QueueAudio(b->audio_dev, samples, static_cast<Uint32>(count * sizeof(int16_t)));
    }
    if (!b->audio_direct_started && SDL_GetQueuedAudioSize(b->audio_dev) >= 16384) {
        b->audio_direct_started = true;
        SDL_PauseAudioDevice(b->audio_dev, 0);
    }
}

void HostWindow::service_events() {
    if (!open_ || !impl_) return;
    handle_xr_events(*as_backend(impl_));
}

HostWindow::Events HostWindow::pump() {
    Events ev;
    if (!open_ || !impl_) {
        ev.quit = true;
        return ev;
    }
    auto* b = as_backend(impl_);
    handle_xr_events(*b);

    if (b->background_pending) {
        ev.enter_background = true;
        b->background_pending = false;
    }

    SDL_Event sev;
    while (SDL_PollEvent(&sev)) {
        if (sev.type == SDL_QUIT) {
            b->exit_requested = true;
        } else if (sev.type == SDL_APP_WILLENTERBACKGROUND ||
                   sev.type == SDL_APP_DIDENTERBACKGROUND) {
            ev.enter_background = true;
        } else if (sev.type == SDL_APP_TERMINATING) {
            ev.terminating = true;
            b->exit_requested = true;
        }
    }

    // Wait briefly until OpenXR enters READY/running at startup so we can bind
    // controllers and start receiving frames cleanly without spinning blind.
    if (!b->session_running && !b->exit_requested &&
        b->session_state != XR_SESSION_STATE_STOPPING) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        handle_xr_events(*b);
    }

    if (b->exit_requested) {
        ev.quit = true;
        return ev;
    }

    // Advance XR session frame if running and we haven't had a present() yet
    // (e.g. during initial boot/menus or runtime pauses) so compositor & pose
    // timestamps stay fresh.
    if (b->session_running && !b->has_submitted_frame) {
        submit_frame(*b, nullptr, 0, 0);
    }

    if (!b->actions_attached || b->session_state != XR_SESSION_STATE_FOCUSED) {
        return ev;
    }

    XrActiveActionSet active{b->action_set, XR_NULL_PATH};
    XrActionsSyncInfo sync{XR_TYPE_ACTIONS_SYNC_INFO};
    sync.countActiveActionSets = 1;
    sync.activeActionSets = &active;
    if (XR_FAILED(xrSyncActions(b->session, &sync))) {
        return ev;
    }

    auto read_bool = [&](XrAction action, XrPath sub = XR_NULL_PATH) -> bool {
        XrActionStateGetInfo gi{XR_TYPE_ACTION_STATE_GET_INFO};
        gi.action = action;
        gi.subactionPath = sub;
        XrActionStateBoolean st{XR_TYPE_ACTION_STATE_BOOLEAN};
        if (XR_SUCCEEDED(xrGetActionStateBoolean(b->session, &gi, &st)) &&
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
        if (XR_SUCCEEDED(xrGetActionStateFloat(b->session, &gi, &st)) && st.isActive) {
            return st.currentState;
        }
        return 0.0f;
    };
    auto read_vec2 = [&](XrAction action, XrPath sub) -> XrVector2f {
        XrActionStateGetInfo gi{XR_TYPE_ACTION_STATE_GET_INFO};
        gi.action = action;
        gi.subactionPath = sub;
        XrActionStateVector2f st{XR_TYPE_ACTION_STATE_VECTOR2F};
        if (XR_SUCCEEDED(xrGetActionStateVector2f(b->session, &gi, &st)) && st.isActive) {
            return st.currentState;
        }
        return {0.0f, 0.0f};
    };

    const XrVector2f lstick = read_vec2(b->act_thumbstick, b->hand_paths[0]);
    const XrVector2f rstick = read_vec2(b->act_thumbstick, b->hand_paths[1]);
    const float ltrig = read_float(b->act_trigger, b->hand_paths[0]);
    const float rtrig = read_float(b->act_trigger, b->hand_paths[1]);
    const float lgrip = read_float(b->act_squeeze, b->hand_paths[0]);
    const float rgrip = read_float(b->act_squeeze, b->hand_paths[1]);
    const bool btn_a = read_bool(b->act_a);
    const bool btn_b = read_bool(b->act_b);
    const bool btn_x = read_bool(b->act_x);
    const bool btn_y = read_bool(b->act_y);
    const bool l_click = read_bool(b->act_thumb_click, b->hand_paths[0]);
    const bool r_click = read_bool(b->act_thumb_click, b->hand_paths[1]);
    const bool l_menu = read_bool(b->act_menu);

    // GBA KEYINPUT active-low: 0=A 1=B 2=Sel 3=Sta 4=Right 5=Left 6=Up 7=Down 8=R 9=L
    uint16_t keys = 0x03FFu;
    if (btn_a || rtrig > kTriggerThreshold) keys &= ~(1u << 0);
    if (btn_b)                              keys &= ~(1u << 1);
    if (btn_y)                              keys &= ~(1u << 2);
    if (btn_x)                              keys &= ~(1u << 3);
    if (lstick.x > kThumbstickDeadzone || rstick.x > kThumbstickDeadzone)   keys &= ~(1u << 4);
    if (lstick.x < -kThumbstickDeadzone || rstick.x < -kThumbstickDeadzone) keys &= ~(1u << 5);
    if (lstick.y > kThumbstickDeadzone || rstick.y > kThumbstickDeadzone)   keys &= ~(1u << 6);
    if (lstick.y < -kThumbstickDeadzone || rstick.y < -kThumbstickDeadzone) keys &= ~(1u << 7);
    if (rgrip > kGripThreshold)                                              keys &= ~(1u << 8);
    if (ltrig > kTriggerThreshold || lgrip > kGripThreshold)                 keys &= ~(1u << 9);
    ev.keyinput = keys;

    const XrTime pose_time = b->last_predicted_time > 0 ? b->last_predicted_time : 1;

    // L+R Thumbstick Click -> Cycle 3D Layered Depth [3D (6CM) -> 3D DEEP -> 2D FLAT]
    // Single R-CLICK (on release) -> Recenter screen
    // Single L-CLICK (on release) -> Toggle active gyro hand
    const bool both_clicks = l_click && r_click;
    if (both_clicks && !b->prev_both_stick_clicks) {
        b->depth_mode = (b->depth_mode + 1) % 3;
        b->stick_chord_consumed = true;
        b->hud_dirty = true;
        trigger_haptic_ns(*b, 0, 25000000LL, 0.45f);
        trigger_haptic_ns(*b, 1, 25000000LL, 0.45f);
        XR_LOGI("3D Layered Depth mode -> %d (step=%.2fm)",
                b->depth_mode, static_cast<double>(depth_step_meters(b->depth_mode)));
    }
    b->prev_both_stick_clicks = both_clicks;

    if (!l_click && !r_click) {
        if (!b->stick_chord_consumed) {
            if (b->prev_r_stick_click) {
                recenter_screen_to_head(*b, pose_time);
                trigger_haptic_ns(*b, 1, 20000000LL, 0.35f);
            } else if (b->prev_l_stick_click) {
                b->active_gyro_hand = 1 - b->active_gyro_hand;
                b->hud_dirty = true;
                trigger_haptic_ns(*b, b->active_gyro_hand, 30000000LL, 0.5f);
                XR_LOGI("Active gyro controller -> %s",
                        b->active_gyro_hand == 1 ? "RIGHT" : "LEFT");
            }
        }
        b->stick_chord_consumed = false;
    }
    b->prev_r_stick_click = r_click;
    b->prev_l_stick_click = l_click;

    // Left Menu -> Cycle [MR+HUD -> MR Clean -> Dark Void Clean] (or HUD on/off if no MR)
    if (l_menu && !b->prev_l_menu_click) {
        if (b->passthrough_supported) {
            b->hud_mode = (b->hud_mode + 1) % 3;
            b->passthrough_active = (b->hud_mode != 2);
            apply_passthrough_state(*b);
        } else {
            b->hud_mode = (b->hud_mode == 0) ? 1 : 0;
        }
        b->hud_dirty = true;
        trigger_haptic_ns(*b, 0, 20000000LL, 0.35f);
    }
    b->prev_l_menu_click = l_menu;

    // Touch Plus controller angular velocity around its aim face-normal (+Z) -> GBA gyro
    XrSpaceVelocity vel{XR_TYPE_SPACE_VELOCITY};
    XrSpaceLocation hand_loc{XR_TYPE_SPACE_LOCATION, &vel};
    const XrSpace active_space = b->aim_spaces[b->active_gyro_hand];
    if (active_space != XR_NULL_HANDLE &&
        XR_SUCCEEDED(xrLocateSpace(active_space, b->local_space, pose_time, &hand_loc)) &&
        (hand_loc.locationFlags & XR_SPACE_LOCATION_ORIENTATION_VALID_BIT) &&
        (vel.velocityFlags & XR_SPACE_VELOCITY_ANGULAR_VALID_BIT)) {
        const XrVector3f z_axis = rotate_vec(hand_loc.pose.orientation, 0.0f, 0.0f, 1.0f);
        const float rate_z = vel.angularVelocity.x * z_axis.x +
                             vel.angularVelocity.y * z_axis.y +
                             vel.angularVelocity.z * z_axis.z;
        const float filtered = (std::abs(rate_z) >= kGyroDriftDeadzoneRadPerSec) ? rate_z : 0.0f;
        ev.gyro_rate_z = filtered * kQuestGyroRateScale;
        if (filtered != 0.0f && !b->gyro_motion_logged) {
            XR_LOGI("Quest Touch Plus gyro active: hand=%s rate_z=%.3f rad/s",
                    b->active_gyro_hand == 1 ? "RIGHT" : "LEFT",
                    static_cast<double>(rate_z));
            b->gyro_motion_logged = true;
        }
    }

    return ev;
}

void host_haptic_pulse(int duration_ms, float strength) {
    if (g_quest_window) g_quest_window->haptic_pulse(duration_ms, strength);
}

void host_request_settings_menu() {}

}  // namespace gbarecomp

#endif  // __ANDROID__ && GBARECOMP_QUEST_XR
