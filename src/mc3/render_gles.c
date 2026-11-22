/* mc3 — GLES 3.2 render backend for Midnight Club: L.A. Remix (USA).
 *
 * Implements the psp_render_backend interface from psprecomp using SDL2 +
 * EGL + OpenGL ES 3.2.
 *
 * Threading & presentation architecture:
 * - The main OS thread owns the SDL2 window, the EGL/GLES 3.2 context, and
 *   runs `gles32_present_loop()` at ~60 FPS (`SDL_PollEvent`, `glTexSubImage2D`,
 *   `glDrawArrays`, `SDL_GL_SwapWindow`). The GL context NEVER leaves the
 *   main thread.
 * - Guest threads (`user_main`, etc.) rasterize into guest VRAM via
 *   `psp_render_software` and snapshot the completed frontbuffer (`0x04000000`)
 *   into `g_front_buf` inside `gles_present()`.
 *
 * This guarantees:
 * 1. Zero flicker (only completed frontbuffer snapshots from `disp_addr` are
 *    uploaded to the GLES texture, identical to `-r sdl2`).
 * 2. Zero EGL thread-affinity hangs or black screens (all GLES and SDL window
 *    calls happen exclusively on the main thread).
 * 3. Full compatibility with PortMaster's GL-only SDL2 shim on the R36S
 *    (Mali-G31 / RK3326), where `SDL_Renderer` is unavailable.
 */

#include "render_gles.h"

#if defined(PSPRECOMP_HAVE_GLES)

#include <psprecomp/mem.h>
#include <psprecomp/hle.h>

#include <SDL2/SDL.h>
#include <EGL/egl.h>
#include <GLES3/gl3.h>

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* ---- window / EGL / input state ----------------------------------------- */

static SDL_Window *g_window = NULL;
static SDL_GLContext g_glcontext = NULL;
static EGLContext g_egl_context = EGL_NO_CONTEXT;
static EGLSurface g_egl_surface = EGL_NO_SURFACE;
static SDL_GameController *g_controller = NULL;

/* ---- shader / program / quad VBO / framebuffer texture ------------------ */

static GLuint g_vertex_shader = 0;
static GLuint g_fragment_shader = 0;
static GLuint g_program = 0;
static GLint g_loc_pos = -1;
static GLint g_loc_uv = -1;
static GLint g_u_tex = -1;
static GLuint g_vbo = 0;
static GLuint g_fb_tex = 0;

/* ---- double-buffered frontbuffer snapshot ------------------------------- */

static pthread_mutex_t g_vram_mutex = PTHREAD_MUTEX_INITIALIZER;
static uint32_t g_fb_addr = 0;
static uint32_t g_fb_stride = 512;
static int      g_fb_fmt = 3;
static uint32_t g_present_stage[480 * 272];
static uint32_t g_front_buf[480 * 272];
static uint32_t g_upload_buf[480 * 272];
static int      g_front_ready = 0;
static int      g_front_dirty = 0;
static int      g_vsync_ok = 0;

/* ---- window / letterbox geometry ---------------------------------------- */

static int g_window_w = 640;
static int g_window_h = 480;
static int g_viewport_x = 0;
static int g_viewport_y = 58;
static int g_viewport_w = 640;
static int g_viewport_h = 363;

/* ---- shaders ------------------------------------------------------------ */

static const char *vertex_shader_src =
    "#version 300 es\n"
    "in vec2 a_pos;\n"
    "in vec2 a_uv;\n"
    "out vec2 v_uv;\n"
    "void main() {\n"
    "    gl_Position = vec4(a_pos, 0.0, 1.0);\n"
    "    v_uv = a_uv;\n"
    "}\n";

static const char *fragment_shader_src =
    "#version 300 es\n"
    "precision mediump float;\n"
    "in vec2 v_uv;\n"
    "uniform sampler2D u_tex;\n"
    "out vec4 frag_color;\n"
    "void main() {\n"
    "    frag_color = vec4(texture(u_tex, v_uv).rgb, 1.0);\n"
    "}\n";

static GLuint compile_shader(GLenum type, const char *src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetShaderInfoLog(s, sizeof(log), NULL, log);
        fprintf(stderr, "gles32: shader compile failed:\n%s\n", log);
    }
    return s;
}

static void update_viewport(int w, int h) {
    if (w <= 0 || h <= 0) return;
    g_window_w = w;
    g_window_h = h;
    float content_aspect = 480.0f / 272.0f;
    float window_aspect = (float)w / (float)h;
    if (window_aspect > content_aspect) {
        g_viewport_h = h;
        g_viewport_w = (int)(g_viewport_h * content_aspect);
        g_viewport_x = (w - g_viewport_w) / 2;
        g_viewport_y = 0;
    } else {
        g_viewport_w = w;
        g_viewport_h = (int)(g_viewport_w / content_aspect);
        g_viewport_x = 0;
        g_viewport_y = (h - g_viewport_h) / 2;
    }
}

/* ---- backend entry points ----------------------------------------------- */

static void gles_shutdown(void);

static int gles_init(int w, int h) {
    g_window_w = w;
    g_window_h = h;

    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMECONTROLLER) < 0) {
        if (SDL_Init(SDL_INIT_VIDEO) < 0) {
            fprintf(stderr, "gles32: SDL_Init failed: %s\n", SDL_GetError());
            return -1;
        }
    }
    for (int i = 0; i < SDL_NumJoysticks(); i++) {
        if (SDL_IsGameController(i)) {
            g_controller = SDL_GameControllerOpen(i);
            if (g_controller) {
                fprintf(stderr, "gles32: opened game controller: %s\n",
                        SDL_GameControllerName(g_controller));
                break;
            }
        }
    }

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_ES);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

    g_window = SDL_CreateWindow("MIDNIGHT CLUB: L.A. REMIX - PSP Static Recomp (GLES 3.2)",
                                SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                w, h,
                                SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);
    if (!g_window) {
        fprintf(stderr, "gles32: SDL_CreateWindow failed: %s\n", SDL_GetError());
        gles_shutdown();
        return -1;
    }

    SDL_RaiseWindow(g_window);
    SDL_SetWindowInputFocus(g_window);
    SDL_ShowWindow(g_window);

    g_glcontext = SDL_GL_CreateContext(g_window);
    if (!g_glcontext) {
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
        g_glcontext = SDL_GL_CreateContext(g_window);
        if (!g_glcontext) {
            fprintf(stderr, "gles32: SDL_GL_CreateContext failed: %s\n", SDL_GetError());
            gles_shutdown();
            return -1;
        }
    }

    SDL_GL_MakeCurrent(g_window, g_glcontext);
    g_vsync_ok = (SDL_GL_SetSwapInterval(1) == 0);

    g_egl_context = eglGetCurrentContext();
    g_egl_surface = eglGetCurrentSurface(EGL_DRAW);
    if (g_egl_context == EGL_NO_CONTEXT || g_egl_surface == EGL_NO_SURFACE) {
        fprintf(stderr, "gles32: warning: eglGetCurrentContext/Surface returned NO_CONTEXT\n");
    }

    g_vertex_shader = compile_shader(GL_VERTEX_SHADER, vertex_shader_src);
    g_fragment_shader = compile_shader(GL_FRAGMENT_SHADER, fragment_shader_src);

    g_program = glCreateProgram();
    glAttachShader(g_program, g_vertex_shader);
    glAttachShader(g_program, g_fragment_shader);
    glLinkProgram(g_program);
    GLint link_ok = 0;
    glGetProgramiv(g_program, GL_LINK_STATUS, &link_ok);
    if (!link_ok) {
        char log[1024];
        glGetProgramInfoLog(g_program, sizeof(log), NULL, log);
        fprintf(stderr, "gles32: program link failed:\n%s\n", log);
        gles_shutdown();
        return -1;
    }

    g_loc_pos = glGetAttribLocation(g_program, "a_pos");
    g_loc_uv  = glGetAttribLocation(g_program, "a_uv");
    g_u_tex   = glGetUniformLocation(g_program, "u_tex");

    /* Fullscreen quad: (x, y, u, v) with top-left UV (0,0) at NDC (-1, +1) */
    static const float quad_verts[6 * 4] = {
        -1.0f,  1.0f,  0.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 0.0f,
        -1.0f, -1.0f,  0.0f, 1.0f,
         1.0f,  1.0f,  1.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 1.0f,
        -1.0f, -1.0f,  0.0f, 1.0f,
    };
    glGenBuffers(1, &g_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, g_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad_verts), quad_verts, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    /* 480x272 streaming texture initialized with test gradient */
    for (int y = 0; y < 272; y++) {
        for (int x = 0; x < 480; x++) {
            uint32_t r = (uint32_t)(x * 255 / 480);
            uint32_t g = (uint32_t)(y * 255 / 272);
            uint32_t b = 128u;
            g_upload_buf[y * 480 + x] = r | (g << 8) | (b << 16) | 0xFF000000u;
        }
    }

    glGenTextures(1, &g_fb_tex);
    glBindTexture(GL_TEXTURE_2D, g_fb_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 480, 272, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, g_upload_buf);

    update_viewport(w, h);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    SDL_GL_SwapWindow(g_window);

    fprintf(stderr,
            "gles32: initialized %dx%d window, content %dx%d at (%d,%d), GLES %s\n",
            w, h, g_viewport_w, g_viewport_h, g_viewport_x, g_viewport_y,
            glGetString(GL_VERSION));
    return 0;
}

static void gles_shutdown(void) {
    if (g_fb_tex) { glDeleteTextures(1, &g_fb_tex); g_fb_tex = 0; }
    if (g_vbo) { glDeleteBuffers(1, &g_vbo); g_vbo = 0; }
    if (g_program) { glDeleteProgram(g_program); g_program = 0; }
    if (g_fragment_shader) { glDeleteShader(g_fragment_shader); g_fragment_shader = 0; }
    if (g_vertex_shader) { glDeleteShader(g_vertex_shader); g_vertex_shader = 0; }
    if (g_controller) { SDL_GameControllerClose(g_controller); g_controller = NULL; }
    if (g_glcontext) { SDL_GL_DeleteContext(g_glcontext); g_glcontext = NULL; }
    if (g_window) { SDL_DestroyWindow(g_window); g_window = NULL; }
    SDL_QuitSubSystem(SDL_INIT_VIDEO);
}

static void gles_set_target(uint32_t addr, uint32_t stride, int fmt) {
    pthread_mutex_lock(&g_vram_mutex);
    g_fb_addr = addr;
    g_fb_stride = stride ? stride : 512;
    g_fb_fmt = fmt;
    psp_render_software.set_target(addr, stride, fmt);
    pthread_mutex_unlock(&g_vram_mutex);
}

static void gles_draw(int prim, const psp_vertex *v, int count, const psp_draw_state *st) {
    psp_render_software.draw(prim, v, count, st);
}

static void gles_finish(void) {
}

static void gles_present(void) {
    uint32_t disp_addr = psp_display_framebuffer();
    uint32_t fb_addr = disp_addr ? disp_addr : g_fb_addr;
    if (!fb_addr) return;
    uint32_t fb_stride = (disp_addr && g_fb_stride < 480) ? 512u : (g_fb_stride ? g_fb_stride : 512u);
    int fb_fmt = g_fb_fmt;

    if (fb_fmt == PSP_TPSM_8888) {
        for (int y = 0; y < 272; y++) {
            psp_mem_read_block(&g_present_stage[y * 480],
                               fb_addr + (uint32_t)y * fb_stride * 4,
                               480 * 4);
        }
    } else {
        for (int y = 0; y < 272; y++) {
            uint32_t row_addr = fb_addr + (uint32_t)y * fb_stride * 2;
            uint32_t *row_dst = &g_present_stage[y * 480];
            for (int x = 0; x < 480; x++) {
                row_dst[x] = psp_color_expand(psp_read16(row_addr + (uint32_t)x * 2), fb_fmt);
            }
        }
    }

    pthread_mutex_lock(&g_vram_mutex);
    memcpy(g_front_buf, g_present_stage, sizeof(g_front_buf));
    g_front_ready = 1;
    g_front_dirty = 1;
    pthread_mutex_unlock(&g_vram_mutex);
}

void gles32_present_loop(void) {
    if (!g_window || !g_glcontext) return;

    fprintf(stderr, "gles32: present loop started (~60 FPS)\n");
    fflush(stderr);
    uint32_t kbd_buttons = 0;

    while (!psp_exit_requested()) {
        SDL_Event ev;
        while (SDL_PollEvent(&ev)) {
            if (ev.type == SDL_QUIT) {
                psp_exit_requested_set(1);
                return;
            } else if (ev.type == SDL_WINDOWEVENT &&
                       ev.window.event == SDL_WINDOWEVENT_SIZE_CHANGED) {
                update_viewport(ev.window.data1, ev.window.data2);
            } else if (ev.type == SDL_KEYDOWN || ev.type == SDL_KEYUP) {
                uint32_t mask = 0;
                switch (ev.key.keysym.sym) {
                case SDLK_UP:     mask = 0x0010; break;
                case SDLK_RIGHT:  mask = 0x0020; break;
                case SDLK_DOWN:   mask = 0x0040; break;
                case SDLK_LEFT:   mask = 0x0080; break;
                case SDLK_q:      mask = 0x0100; break; /* L trigger */
                case SDLK_w:      mask = 0x0200; break; /* R trigger */
                case SDLK_t:      mask = 0x1000; break; /* Triangle */
                case SDLK_c:
                case SDLK_BACKSPACE: mask = 0x2000; break; /* Circle */
                case SDLK_x:
                case SDLK_z:      mask = 0x4000; break; /* Cross (X / OK) */
                case SDLK_s:
                case SDLK_a:      mask = 0x8000; break; /* Square */
                case SDLK_RETURN: mask = 0x0008; break; /* Start */
                case SDLK_SPACE:  mask = 0x0001; break; /* Select */
                case SDLK_ESCAPE:
                    if (ev.type == SDL_KEYDOWN) {
                        psp_exit_requested_set(1);
                        return;
                    }
                    break;
                default: break;
                }
                if (mask) {
                    if (ev.type == SDL_KEYDOWN) kbd_buttons |= mask;
                    else kbd_buttons &= ~mask;
                }
            }
        }

        uint32_t pad_buttons = 0;
        uint8_t lx = 128, ly = 128;
        if (g_controller) {
            if (SDL_GameControllerGetButton(g_controller, SDL_CONTROLLER_BUTTON_DPAD_UP))    pad_buttons |= 0x0010;
            if (SDL_GameControllerGetButton(g_controller, SDL_CONTROLLER_BUTTON_DPAD_RIGHT)) pad_buttons |= 0x0020;
            if (SDL_GameControllerGetButton(g_controller, SDL_CONTROLLER_BUTTON_DPAD_DOWN))  pad_buttons |= 0x0040;
            if (SDL_GameControllerGetButton(g_controller, SDL_CONTROLLER_BUTTON_DPAD_LEFT))  pad_buttons |= 0x0080;
            if (SDL_GameControllerGetButton(g_controller, SDL_CONTROLLER_BUTTON_A))          pad_buttons |= 0x4000; /* Cross */
            if (SDL_GameControllerGetButton(g_controller, SDL_CONTROLLER_BUTTON_B))          pad_buttons |= 0x2000; /* Circle */
            if (SDL_GameControllerGetButton(g_controller, SDL_CONTROLLER_BUTTON_X))          pad_buttons |= 0x8000; /* Square */
            if (SDL_GameControllerGetButton(g_controller, SDL_CONTROLLER_BUTTON_Y))          pad_buttons |= 0x1000; /* Triangle */
            if (SDL_GameControllerGetButton(g_controller, SDL_CONTROLLER_BUTTON_START))      pad_buttons |= 0x0008;
            if (SDL_GameControllerGetButton(g_controller, SDL_CONTROLLER_BUTTON_BACK))       pad_buttons |= 0x0001;
            if (SDL_GameControllerGetButton(g_controller, SDL_CONTROLLER_BUTTON_LEFTSHOULDER))  pad_buttons |= 0x0100;
            if (SDL_GameControllerGetButton(g_controller, SDL_CONTROLLER_BUTTON_RIGHTSHOULDER)) pad_buttons |= 0x0200;

            int raw_x = (int)SDL_GameControllerGetAxis(g_controller, SDL_CONTROLLER_AXIS_LEFTX);
            int raw_y = (int)SDL_GameControllerGetAxis(g_controller, SDL_CONTROLLER_AXIS_LEFTY);
            int vx = (raw_x + 32768) / 256;
            int vy = (raw_y + 32768) / 256;
            if (vx < 0) vx = 0; else if (vx > 255) vx = 255;
            if (vy < 0) vy = 0; else if (vy > 255) vy = 255;
            lx = (uint8_t)vx;
            ly = (uint8_t)vy;
        }
        psp_ctrl_set(kbd_buttons | pad_buttons, lx, ly);

        int ready = 0;
        pthread_mutex_lock(&g_vram_mutex);
        if (g_front_dirty != 0) {
            memcpy(g_upload_buf, g_front_buf, sizeof(g_upload_buf));
            ready = 1;
            g_front_dirty = 0;
        }
        pthread_mutex_unlock(&g_vram_mutex);

        glBindTexture(GL_TEXTURE_2D, g_fb_tex);
        if (ready) {
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 480, 272,
                            GL_RGBA, GL_UNSIGNED_BYTE, g_upload_buf);
        }

        glViewport(0, 0, g_window_w, g_window_h);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glViewport(g_viewport_x, g_viewport_y, g_viewport_w, g_viewport_h);
        glUseProgram(g_program);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, g_fb_tex);
        glUniform1i(g_u_tex, 0);

        glBindBuffer(GL_ARRAY_BUFFER, g_vbo);
        glEnableVertexAttribArray((GLuint)g_loc_pos);
        glEnableVertexAttribArray((GLuint)g_loc_uv);
        glVertexAttribPointer((GLuint)g_loc_pos, 2, GL_FLOAT, GL_FALSE,
                              4 * sizeof(float), (void *)0);
        glVertexAttribPointer((GLuint)g_loc_uv, 2, GL_FLOAT, GL_FALSE,
                              4 * sizeof(float), (void *)(2 * sizeof(float)));

        glDrawArrays(GL_TRIANGLES, 0, 6);

        glDisableVertexAttribArray((GLuint)g_loc_pos);
        glDisableVertexAttribArray((GLuint)g_loc_uv);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glUseProgram(0);

        SDL_GL_SwapWindow(g_window);
        if (!g_vsync_ok) usleep(16667);
    }

    fprintf(stderr, "gles32: present loop exiting\n");
    fflush(stderr);
}

const psp_render_backend psp_render_gles32 = {
    "gles32",
    gles_init,
    gles_shutdown,
    gles_set_target,
    gles_draw,
    gles_finish,
    gles_present,
};

#else /* !PSPRECOMP_HAVE_GLES */

/* Stub backend when SDL2/EGL/GLESv2 are not available. Init always fails so
 * the game falls back to the software backend. */
#include <psprecomp/render.h>
#include <stdio.h>

void gles32_present_loop(void) { }

static int gles_stub_init(int w, int h) { (void)w; (void)h; return -1; }
static void gles_stub_shutdown(void) { }
static void gles_stub_set_target(uint32_t a, uint32_t s, int f) { (void)a; (void)s; (void)f; }
static void gles_stub_draw(int p, const psp_vertex *v, int n, const psp_draw_state *st) {
    (void)p; (void)v; (void)n; (void)st;
}
static void gles_stub_finish(void) { }
static void gles_stub_present(void) { }

const psp_render_backend psp_render_gles32 = {
    "gles32",
    gles_stub_init,
    gles_stub_shutdown,
    gles_stub_set_target,
    gles_stub_draw,
    gles_stub_finish,
    gles_stub_present,
};

#endif /* PSPRECOMP_HAVE_GLES */
