// quest_popup_renderer.h — GLES3 renderer for the Quest "POP-UP 3D" mode.
//
// Geometry lives in SCREEN-LOCAL meters: the virtual screen is the plane z = 0 (centred,
// width sw, height sh), the default eye E0 sits at (0, 0, eye_z) looking down -z.
//   - P0 backdrop : screen-facing plane at z = -0.25 m
//   - P1 floor    : plane through the origin tilted back by `tilt` (top of the image farther away)
//   - cards       : one screen-facing quad per distinct base row b, at the depth where the
//                   default-eye ray through row b meets the floor (props and actors)
// Every fragment looks up its colour by projecting its position back through E0 onto the
// screen plane, so from the default seat the image is the stock frame pixel-for-pixel and
// depth only shows up as stereo / head-motion parallax.
// Sampling is "sharp bilinear": hard pixel edges without shimmer when the head moves.
// The front layer (room frame, overlays, HUD) is NOT drawn here; the host submits it as
// a compositor quad so text stays on the sharpest path.
#pragma once

#include <GLES3/gl3.h>
#include <android/log.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "color_lut.h"
#include "quest_popup_layout.h"

namespace gbarecomp {
namespace popup {

class Renderer {
public:
    Layout layout;

    bool init() {
        static const char* kVs =
            "#version 300 es\n"
            "layout(location = 0) in vec3 a_pos;\n"
            "uniform mat4 u_mvp;\n"
            "out vec3 v_pos;\n"
            "void main() { v_pos = a_pos; gl_Position = u_mvp * vec4(a_pos, 1.0); }\n";
        static const char* kFs =
            "#version 300 es\n"
            "precision highp float;\n"
            "precision highp int;\n"
            "precision highp usampler2D;\n"
            "in vec3 v_pos;\n"
            "uniform sampler2D u_color;\n"
            "uniform usampler2D u_base;\n"
            "uniform int u_use_base;\n"
            "uniform uint u_base_val;\n"
            "uniform float u_eye_z;\n"
            "uniform vec2 u_screen;\n"
            "uniform vec2 u_img;\n"
            "out vec4 o_color;\n"
            "void main() {\n"
            "  float s = u_eye_z / (u_eye_z - v_pos.z);\n"
            "  vec2 q = v_pos.xy * s;\n"
            "  vec2 p = vec2((q.x / u_screen.x + 0.5) * u_img.x, (0.5 - q.y / u_screen.y) * u_img.y);\n"
            "  vec2 fw = max(fwidth(p), vec2(1e-4));\n"
            "  vec2 fl = floor(p - 0.5) + 0.5;\n"
            "  vec2 t = clamp((p - fl - 0.5) / fw + 0.5, 0.0, 1.0);\n"
            "  vec4 c = texture(u_color, (fl + t) / u_img);\n"
            "  if (p.x < 0.0 || p.y < 0.0 || p.x >= u_img.x || p.y >= u_img.y) discard;\n"
            "  if (u_use_base != 0) {\n"
            "    uint bv = texelFetch(u_base, ivec2(floor(p)), 0).r;\n"
            "    if (bv != u_base_val) discard;\n"
            "  }\n"
            "  if (c.a < 0.5) discard;\n"
            "  o_color = vec4(c.rgb, 1.0);\n"
            "}\n";
        prog_ = link(kVs, kFs);
        if (!prog_) return false;
        u_mvp_ = glGetUniformLocation(prog_, "u_mvp");
        u_color_ = glGetUniformLocation(prog_, "u_color");
        u_base_ = glGetUniformLocation(prog_, "u_base");
        u_use_base_ = glGetUniformLocation(prog_, "u_use_base");
        u_base_val_ = glGetUniformLocation(prog_, "u_base_val");
        u_eye_z_ = glGetUniformLocation(prog_, "u_eye_z");
        u_screen_ = glGetUniformLocation(prog_, "u_screen");
        u_img_ = glGetUniformLocation(prog_, "u_img");
        glGenVertexArrays(1, &vao_);
        glGenBuffers(1, &vbo_);
        glGenTextures(kColorTexCount, color_tex_);
        glGenTextures(2, base_tex_);
        glGenFramebuffers(1, &fbo_);
        glGenRenderbuffers(1, &depth_rb_);
        ready_ = true;
        return true;
    }

    void shutdown() {
        if (!ready_) return;
        glDeleteProgram(prog_);
        glDeleteVertexArrays(1, &vao_);
        glDeleteBuffers(1, &vbo_);
        glDeleteTextures(kColorTexCount, color_tex_);
        glDeleteTextures(2, base_tex_);
        glDeleteFramebuffers(1, &fbo_);
        glDeleteRenderbuffers(1, &depth_rb_);
        prog_ = vao_ = vbo_ = fbo_ = depth_rb_ = 0;
        tex_w_ = tex_h_ = depth_w_ = depth_h_ = 0;
        ready_ = false;
    }

    bool ready() const { return ready_; }

    // Analyse a new game frame (5 masked RGBA planes) and upload all textures.
    void upload(const uint8_t* planes, int w, int h, const runtime::ColorLut* lut, bool srgb) {
        if (!ready_ || !planes || w <= 0 || h <= 0) return;
        build_layout(planes, w, h, layout);
        const size_t pb = static_cast<size_t>(w) * h * 4;
        if (w != tex_w_ || h != tex_h_ || srgb != srgb_) allocate(w, h, srgb);

        // P0: fill the parts hidden behind P1 with the colour above/below them in the same
        // column, so no black slivers show when the head looks around a cliff or wall edge.
        backdrop_.assign(planes, planes + pb);
        fill_backdrop(planes + 1 * pb, w, h);
        upload_color(kTexBackdrop, backdrop_.data(), w, h, lut);
        upload_color(kTexFloor, planes + 1 * pb, w, h, lut);
        upload_color(kTexProps, layout.props_rgba.data(), w, h, lut);
        upload_color(kTexActors, layout.actors_rgba.data(), w, h, lut);
        glBindTexture(GL_TEXTURE_2D, base_tex_[0]);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RED_INTEGER, GL_UNSIGNED_SHORT, layout.prop_base.data());
        glBindTexture(GL_TEXTURE_2D, base_tex_[1]);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RED_INTEGER, GL_UNSIGNED_SHORT, layout.actor_base.data());
        glBindTexture(GL_TEXTURE_2D, 0);
        geometry_dirty_ = true;
    }

    // Build the shared vertex buffer for the current layout (both eyes use it).
    void prepare(float eye_z, float sw, float sh, float tilt_rad) {
        if (!ready_ || tex_w_ == 0) return;
        if (!geometry_dirty_ && eye_z == g_eye_z_ && sw == g_sw_ && sh == g_sh_ && tilt_rad == g_tilt_) return;
        g_eye_z_ = eye_z; g_sw_ = sw; g_sh_ = sh; g_tilt_ = tilt_rad;
        geometry_dirty_ = false;
        verts_.clear();
        draws_.clear();
        const float W = static_cast<float>(tex_w_), H = static_cast<float>(tex_h_);
        const float ca = std::cos(tilt_rad), sa = std::sin(tilt_rad);
        auto screen_x = [&](float px) { return (px / W - 0.5f) * sw; };
        auto screen_y = [&](float py) { return (0.5f - py / H) * sh; };
        // screen-facing quad: stock-pixel rect projected from E0 onto plane z
        auto facing = [&](int tex, int base_idx, uint16_t base_val, float x0, float x1, float y0, float y1, float z) {
            const float k = (eye_z - z) / eye_z;
            Draw d{tex, base_idx, base_val, static_cast<int>(verts_.size() / 3)};
            const float xs[2] = {screen_x(x0) * k, screen_x(x1) * k};
            const float ys[2] = {screen_y(y1) * k, screen_y(y0) * k};
            push(xs[0], ys[0], z); push(xs[1], ys[0], z); push(xs[0], ys[1], z); push(xs[1], ys[1], z);
            draws_.push_back(d);
        };
        auto floor_pt = [&](float px, float py) {
            const float X = screen_x(px), Y = screen_y(py);
            const float s = eye_z * ca / (eye_z * ca - Y * sa);
            push(X * s, Y * s, eye_z * (1.0f - s));
        };
        auto card_z = [&](float row) {
            const float Y = screen_y(row);
            const float s = eye_z * ca / (eye_z * ca - Y * sa);
            return eye_z * (1.0f - s);
        };
        // backdrop
        facing(kTexBackdrop, -1, 0, 0, W, 0, H, -0.25f);
        // floor (tilted plane through the screen centre)
        {
            Draw d{kTexFloor, -1, 0, static_cast<int>(verts_.size() / 3)};
            floor_pt(0, H); floor_pt(W, H); floor_pt(0, 0); floor_pt(W, 0);
            draws_.push_back(d);
        }
        for (const Group& g : layout.prop_groups)
            facing(kTexProps, 0, g.base, g.x0 - 1.f, g.x1 + 1.f, g.y0 - 1.f, g.y1 + 1.f, card_z(g.base));
        for (const Group& g : layout.actor_groups)
            facing(kTexActors, 1, g.base, g.x0 - 1.f, g.x1 + 1.f, g.y0 - 1.f, g.y1 + 1.f, card_z(g.base));
        glBindBuffer(GL_ARRAY_BUFFER, vbo_);
        glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(verts_.size() * sizeof(float)), verts_.data(),
                     GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    // Render one eye into `color_image` (a swapchain GL texture). mvp maps screen-local
    // meters to clip space for this eye. Clears to transparent so passthrough shows around it.
    void draw_eye(GLuint color_image, int vw, int vh, const float mvp[16]) {
        if (!ready_ || tex_w_ == 0 || draws_.empty()) return;
        if (vw != depth_w_ || vh != depth_h_) {
            glBindRenderbuffer(GL_RENDERBUFFER, depth_rb_);
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, vw, vh);
            glBindRenderbuffer(GL_RENDERBUFFER, 0);
            depth_w_ = vw; depth_h_ = vh;
        }
        glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color_image, 0);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_rb_);
        glViewport(0, 0, vw, vh);
        glDisable(GL_SCISSOR_TEST);
        glDisable(GL_BLEND);
        glDisable(GL_CULL_FACE);
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);
        glDepthMask(GL_TRUE);
        glClearColor(0.f, 0.f, 0.f, 0.f);
        glClearDepthf(1.f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(prog_);
        glUniformMatrix4fv(u_mvp_, 1, GL_FALSE, mvp);
        glUniform1f(u_eye_z_, g_eye_z_);
        glUniform2f(u_screen_, g_sw_, g_sh_);
        glUniform2f(u_img_, static_cast<float>(tex_w_), static_cast<float>(tex_h_));
        glUniform1i(u_color_, 0);
        glUniform1i(u_base_, 1);
        glBindVertexArray(vao_);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), nullptr);
        int bound_tex = -1, bound_base = -2;
        for (const Draw& d : draws_) {
            if (d.tex != bound_tex) {
                glActiveTexture(GL_TEXTURE0);
                glBindTexture(GL_TEXTURE_2D, color_tex_[d.tex]);
                bound_tex = d.tex;
            }
            if (d.base_idx != bound_base) {
                glActiveTexture(GL_TEXTURE1);
                glBindTexture(GL_TEXTURE_2D, d.base_idx >= 0 ? base_tex_[d.base_idx] : base_tex_[0]);
                glUniform1i(u_use_base_, d.base_idx >= 0 ? 1 : 0);
                bound_base = d.base_idx;
            }
            if (d.base_idx >= 0) glUniform1ui(u_base_val_, d.base_val);
            glDrawArrays(GL_TRIANGLE_STRIP, d.first, 4);
        }
        glBindVertexArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, 0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, 0);
        glUseProgram(0);
        glDisable(GL_DEPTH_TEST);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    // ---- small matrix helpers (column-major, GLES clip space)
    static void projection_from_fov(float out[16], float tan_l, float tan_r, float tan_u, float tan_d,
                                    float n, float f) {
        const float w = tan_r - tan_l, h = tan_u - tan_d;
        std::memset(out, 0, 16 * sizeof(float));
        out[0] = 2.f / w;
        out[5] = 2.f / h;
        out[8] = (tan_r + tan_l) / w;
        out[9] = (tan_u + tan_d) / h;
        out[10] = -(f + n) / (f - n);
        out[11] = -1.f;
        out[14] = -(2.f * f * n) / (f - n);
    }
    // rigid transform from quaternion (x,y,z,w) + position
    static void pose_matrix(float out[16], float qx, float qy, float qz, float qw, float px, float py, float pz) {
        const float xx = qx * qx, yy = qy * qy, zz = qz * qz;
        const float xy = qx * qy, xz = qx * qz, yz = qy * qz, wx = qw * qx, wy = qw * qy, wz = qw * qz;
        out[0] = 1 - 2 * (yy + zz); out[1] = 2 * (xy + wz);     out[2] = 2 * (xz - wy);     out[3] = 0;
        out[4] = 2 * (xy - wz);     out[5] = 1 - 2 * (xx + zz); out[6] = 2 * (yz + wx);     out[7] = 0;
        out[8] = 2 * (xz + wy);     out[9] = 2 * (yz - wx);     out[10] = 1 - 2 * (xx + yy); out[11] = 0;
        out[12] = px; out[13] = py; out[14] = pz; out[15] = 1;
    }
    static void rigid_inverse(float out[16], const float m[16]) {
        // R^T, -R^T t
        out[0] = m[0]; out[1] = m[4]; out[2] = m[8];  out[3] = 0;
        out[4] = m[1]; out[5] = m[5]; out[6] = m[9];  out[7] = 0;
        out[8] = m[2]; out[9] = m[6]; out[10] = m[10]; out[11] = 0;
        out[12] = -(m[0] * m[12] + m[1] * m[13] + m[2] * m[14]);
        out[13] = -(m[4] * m[12] + m[5] * m[13] + m[6] * m[14]);
        out[14] = -(m[8] * m[12] + m[9] * m[13] + m[10] * m[14]);
        out[15] = 1;
    }
    static void mul(float out[16], const float a[16], const float b[16]) {
        float r[16];
        for (int c = 0; c < 4; ++c)
            for (int rr = 0; rr < 4; ++rr) {
                float s = 0;
                for (int k = 0; k < 4; ++k) s += a[k * 4 + rr] * b[c * 4 + k];
                r[c * 4 + rr] = s;
            }
        std::memcpy(out, r, sizeof(r));
    }

private:
    enum { kTexBackdrop = 0, kTexFloor = 1, kTexProps = 2, kTexActors = 3, kColorTexCount = 4 };
    struct Draw {
        int tex;
        int base_idx;  // -1 = no base test, 0 = props, 1 = actors
        uint16_t base_val;
        int first;
    };

    GLuint prog_ = 0, vao_ = 0, vbo_ = 0, fbo_ = 0, depth_rb_ = 0;
    GLuint color_tex_[kColorTexCount] = {};
    GLuint base_tex_[2] = {};
    GLint u_mvp_ = -1, u_color_ = -1, u_base_ = -1, u_use_base_ = -1, u_base_val_ = -1, u_eye_z_ = -1,
          u_screen_ = -1, u_img_ = -1;
    int tex_w_ = 0, tex_h_ = 0, depth_w_ = 0, depth_h_ = 0;
    bool srgb_ = false, ready_ = false, geometry_dirty_ = true;
    float g_eye_z_ = 0, g_sw_ = 0, g_sh_ = 0, g_tilt_ = 0;
    std::vector<float> verts_;
    std::vector<Draw> draws_;
    std::vector<uint8_t> backdrop_, staged_, rgb_in_, rgb_out_;

    void push(float x, float y, float z) { verts_.push_back(x); verts_.push_back(y); verts_.push_back(z); }

    static GLuint compile(GLenum type, const char* src) {
        GLuint s = glCreateShader(type);
        glShaderSource(s, 1, &src, nullptr);
        glCompileShader(s);
        GLint ok = 0;
        glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
        if (!ok) {
            char log[1024] = {};
            glGetShaderInfoLog(s, sizeof(log) - 1, nullptr, log);
            __android_log_print(ANDROID_LOG_ERROR, "Mother3XR", "POP-UP 3D: shader compile failed: %s", log);
            glDeleteShader(s);
            return 0;
        }
        return s;
    }
    static GLuint link(const char* vs, const char* fs) {
        GLuint v = compile(GL_VERTEX_SHADER, vs), f = compile(GL_FRAGMENT_SHADER, fs);
        if (!v || !f) { if (v) glDeleteShader(v); if (f) glDeleteShader(f); return 0; }
        GLuint p = glCreateProgram();
        glAttachShader(p, v);
        glAttachShader(p, f);
        glLinkProgram(p);
        glDeleteShader(v);
        glDeleteShader(f);
        GLint ok = 0;
        glGetProgramiv(p, GL_LINK_STATUS, &ok);
        if (!ok) {
            char log[1024] = {};
            glGetProgramInfoLog(p, sizeof(log) - 1, nullptr, log);
            __android_log_print(ANDROID_LOG_ERROR, "Mother3XR", "POP-UP 3D: program link failed: %s", log);
            glDeleteProgram(p);
            return 0;
        }
        return p;
    }

    void allocate(int w, int h, bool srgb) {
        tex_w_ = w; tex_h_ = h; srgb_ = srgb;
        for (int i = 0; i < kColorTexCount; ++i) {
            glBindTexture(GL_TEXTURE_2D, color_tex_[i]);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexImage2D(GL_TEXTURE_2D, 0, srgb ? GL_SRGB8_ALPHA8 : GL_RGBA8, w, h, 0, GL_RGBA,
                         GL_UNSIGNED_BYTE, nullptr);
        }
        for (int i = 0; i < 2; ++i) {
            glBindTexture(GL_TEXTURE_2D, base_tex_[i]);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_R16UI, w, h, 0, GL_RED_INTEGER, GL_UNSIGNED_SHORT, nullptr);
        }
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    void fill_backdrop(const uint8_t* p1, int w, int h) {
        for (int x = 0; x < w; ++x) {
            // top-down carry, then bottom-up for anything still unfilled at the top
            int carry = -1;
            for (int y = 0; y < h; ++y) {
                const size_t i = (static_cast<size_t>(y) * w + x) * 4;
                if (p1[i + 3] == 0) { carry = static_cast<int>(i); continue; }
                if (carry >= 0) std::memcpy(&backdrop_[i], &backdrop_[carry], 4);
            }
            carry = -1;
            for (int y = h - 1; y >= 0; --y) {
                const size_t i = (static_cast<size_t>(y) * w + x) * 4;
                if (p1[i + 3] == 0) { carry = static_cast<int>(i); break; }
            }
            for (int y = 0; y < h; ++y) {
                const size_t i = (static_cast<size_t>(y) * w + x) * 4;
                if (p1[i + 3] == 0) break;
                if (carry >= 0) std::memcpy(&backdrop_[i], &backdrop_[carry], 4);
            }
        }
    }

    // Grade with the colour LUT and bleed edge colours into transparent texels so the
    // sharp-bilinear filter never mixes in black at sprite edges.
    void upload_color(int tex, const uint8_t* rgba, int w, int h, const runtime::ColorLut* lut) {
        const size_t n = static_cast<size_t>(w) * h;
        staged_.assign(rgba, rgba + n * 4);
        if (lut && !lut->is_passthrough()) {
            rgb_in_.resize(n * 3);
            rgb_out_.resize(n * 3);
            for (size_t i = 0; i < n; ++i) std::memcpy(&rgb_in_[i * 3], &staged_[i * 4], 3);
            lut->map_rgb888(rgb_in_.data(), rgb_out_.data(), w, h);
            for (size_t i = 0; i < n; ++i)
                if (staged_[i * 4 + 3]) std::memcpy(&staged_[i * 4], &rgb_out_[i * 3], 3);
        }
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                uint8_t* p = &staged_[(static_cast<size_t>(y) * w + x) * 4];
                if (p[3]) continue;
                static const int dx[4] = {1, -1, 0, 0}, dy[4] = {0, 0, 1, -1};
                for (int k = 0; k < 4; ++k) {
                    const int xx = x + dx[k], yy = y + dy[k];
                    if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
                    const uint8_t* q = &rgba[(static_cast<size_t>(yy) * w + xx) * 4];
                    if (q[3]) {
                        const uint8_t* g = &staged_[(static_cast<size_t>(yy) * w + xx) * 4];
                        p[0] = g[0]; p[1] = g[1]; p[2] = g[2];
                        break;
                    }
                }
            }
        glBindTexture(GL_TEXTURE_2D, color_tex_[tex]);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, staged_.data());
    }
};

}  // namespace popup
}  // namespace gbarecomp
