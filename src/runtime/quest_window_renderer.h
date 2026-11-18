// quest_window_renderer.h - per-eye "window" rendering of far STOCK 3D planes.
//
// GBARECOMP_XR_WINDOW=1: the far planes (parallax-pushed sky / back wall) are not submitted as
// compositor quads but drawn per eye into a projection layer, each fragment kept only if the ray
// from that eye through it passes the window opening (the screen edge at the front plane's
// depth). Leaning then shows exactly what a real window would, never a layer's border.
#pragma once

#include <GLES3/gl3.h>
#include <android/log.h>

#include <cstdint>
#include <cstring>

namespace gbarecomp {

class WindowRenderer {
public:
    static constexpr int kSlots = 6;
    struct Plane {
        int slot;
        float z, hw, hh;  // screen-local: centre at (0,0,z), half extents (incl. margin)
    };

    bool ready() const { return prog_ != 0; }

    bool init() {
        static const char* vs =
            "#version 300 es\n"
            "layout(location=0) in vec3 a_pos;\n"
            "layout(location=1) in vec2 a_uv;\n"
            "uniform mat4 u_mvp;\n"
            "out vec3 v_pos; out vec2 v_uv;\n"
            "void main(){ v_pos=a_pos; v_uv=a_uv; gl_Position=u_mvp*vec4(a_pos,1.0); }\n";
        static const char* fs =
            "#version 300 es\n"
            "precision highp float;\n"
            "in vec3 v_pos; in vec2 v_uv;\n"
            "uniform sampler2D u_tex; uniform vec3 u_eye; uniform vec3 u_win;\n"
            "out vec4 o;\n"
            "void main(){\n"
            "  vec3 d = v_pos - u_eye;\n"
            "  float t = (u_win.z - u_eye.z) / d.z;\n"
            "  vec2 q = u_eye.xy + t * d.xy;\n"
            "  if (abs(q.x) > u_win.x || abs(q.y) > u_win.y) discard;\n"
            "  o = texture(u_tex, v_uv);\n"
            "}\n";
        const GLuint v = compile(GL_VERTEX_SHADER, vs), f = compile(GL_FRAGMENT_SHADER, fs);
        if (!v || !f) return false;
        prog_ = glCreateProgram();
        glAttachShader(prog_, v);
        glAttachShader(prog_, f);
        glLinkProgram(prog_);
        glDeleteShader(v);
        glDeleteShader(f);
        GLint ok = 0;
        glGetProgramiv(prog_, GL_LINK_STATUS, &ok);
        if (!ok) { glDeleteProgram(prog_); prog_ = 0; return false; }
        u_mvp_ = glGetUniformLocation(prog_, "u_mvp");
        u_tex_ = glGetUniformLocation(prog_, "u_tex");
        u_eye_ = glGetUniformLocation(prog_, "u_eye");
        u_win_ = glGetUniformLocation(prog_, "u_win");
        glGenVertexArrays(1, &vao_);
        glGenBuffers(1, &vbo_);
        glGenFramebuffers(1, &fbo_);
        glGenTextures(kSlots, tex_);
        for (int i = 0; i < kSlots; ++i) {
            glBindTexture(GL_TEXTURE_2D, tex_[i]);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        }
        glBindTexture(GL_TEXTURE_2D, 0);
        return true;
    }

    // rgba: bottom-up rows, premultiplied (as the quads get it).
    void upload(int slot, const uint8_t* rgba, int w, int h, bool srgb) {
        if (!ready() || slot < 0 || slot >= kSlots) return;
        glBindTexture(GL_TEXTURE_2D, tex_[slot]);
        if (w != tw_[slot] || h != th_[slot]) {
            glTexImage2D(GL_TEXTURE_2D, 0, srgb ? GL_SRGB8_ALPHA8 : GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba);
            tw_[slot] = w;
            th_[slot] = h;
        } else {
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, rgba);
        }
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    // planes back to front; eye[3] and win[3] (half w, half h, z) in screen-local metres.
    void draw_eye(GLuint color_image, int vw, int vh, const float mvp[16], const float eye[3],
                  const float win[3], const Plane* planes, int count) {
        glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color_image, 0);
        glViewport(0, 0, vw, vh);
        glDisable(GL_SCISSOR_TEST);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_CULL_FACE);
        glClearColor(0.f, 0.f, 0.f, 0.f);
        glClear(GL_COLOR_BUFFER_BIT);
        glEnable(GL_BLEND);
        glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
        glUseProgram(prog_);
        glUniformMatrix4fv(u_mvp_, 1, GL_FALSE, mvp);
        glUniform3f(u_eye_, eye[0], eye[1], eye[2]);
        glUniform3f(u_win_, win[0], win[1], win[2]);
        glUniform1i(u_tex_, 0);
        glActiveTexture(GL_TEXTURE0);
        glBindVertexArray(vao_);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_);
        for (int i = 0; i < count; ++i) {
            const Plane& p = planes[i];
            if (p.slot < 0 || p.slot >= kSlots || !tw_[p.slot]) continue;
            const float v[] = {-p.hw, -p.hh, p.z, 0, 0,  p.hw, -p.hh, p.z, 1, 0,
                               -p.hw,  p.hh, p.z, 0, 1,  p.hw,  p.hh, p.z, 1, 1};
            glBufferData(GL_ARRAY_BUFFER, sizeof(v), v, GL_STREAM_DRAW);
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(0));
            glEnableVertexAttribArray(1);
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(3 * sizeof(float)));
            glBindTexture(GL_TEXTURE_2D, tex_[p.slot]);
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        }
        glBindTexture(GL_TEXTURE_2D, 0);
        glBindVertexArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glUseProgram(0);
        glDisable(GL_BLEND);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

private:
    static GLuint compile(GLenum type, const char* src) {
        const GLuint s = glCreateShader(type);
        glShaderSource(s, 1, &src, nullptr);
        glCompileShader(s);
        GLint ok = 0;
        glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
        if (!ok) {
            char log[512] = {};
            glGetShaderInfoLog(s, sizeof(log), nullptr, log);
            __android_log_print(ANDROID_LOG_ERROR, "Mother3XR", "window shader: %s", log);
            glDeleteShader(s);
            return 0;
        }
        return s;
    }
    GLuint prog_ = 0, vao_ = 0, vbo_ = 0, fbo_ = 0;
    GLuint tex_[kSlots] = {};
    int tw_[kSlots] = {}, th_[kSlots] = {};
    GLint u_mvp_ = -1, u_tex_ = -1, u_eye_ = -1, u_win_ = -1;
};

}  // namespace gbarecomp
