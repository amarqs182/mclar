/* mc3 — GLES 3.2 render backend for Midnight Club: L.A. Remix (USA).
 *
 * Implements the psp_render_backend interface from psprecomp using SDL2 +
 * EGL + OpenGL ES 3.2. Primitives arrive assembled (screen-space position,
 * depth, colour, and texel coordinates) and are drawn directly with GLES.
 * Targets GLES 3.2 (Mali-G31 on the RK3326 / R36S and desktop EGL/GLES).
 *
 * Important threading detail: psprecomp with PSPRECOMP_REAL_THREADS=ON runs
 * each guest thread on its own host pthread (user_main calls sceGeListEnQueue
 * from a worker pthread, while gles_init ran on the main thread). Because an
 * OpenGL ES context can only be current on one thread at a time, we ensure
 * the GL context is bound to the calling thread before issuing GL calls,
 * serialized by a mutex.
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

/* ---- window / EGL / thread / input state -------------------------------- */

static SDL_Window *g_window = NULL;
static SDL_GLContext g_glcontext = NULL;
static EGLContext g_egl_context = EGL_NO_CONTEXT;
static EGLSurface g_egl_surface = EGL_NO_SURFACE;
static SDL_GameController *g_controller = NULL;
static uint32_t g_kbd_buttons = 0;

static pthread_mutex_t g_gl_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_t g_gl_owner_tid;
static int g_gl_has_owner = 0;

static void gl_lock_and_make_current(void) {
    pthread_mutex_lock(&g_gl_mutex);
    pthread_t self = pthread_self();
    if (!g_gl_has_owner || !pthread_equal(g_gl_owner_tid, self)) {
        if (g_window && g_glcontext) {
            SDL_GL_MakeCurrent(g_window, g_glcontext);
            g_gl_owner_tid = self;
            g_gl_has_owner = 1;
        }
    }
}

static void gl_unlock(void) {
    pthread_mutex_unlock(&g_gl_mutex);
}

/* ---- shader / program --------------------------------------------------- */

static GLuint g_vertex_shader = 0;
static GLuint g_fragment_shader = 0;
static GLuint g_program = 0;
static GLint g_loc_pos = -1;
static GLint g_loc_color = -1;
static GLint g_loc_uv = -1;

static GLint g_u_tex = -1;
static GLint g_u_tex_enable = -1;
static GLint g_u_tex_func = -1;
static GLint g_u_tex_alpha = -1;
static GLint g_u_tex_double = -1;
static GLint g_u_tex_env = -1;
static GLint g_u_atest = -1;
static GLint g_u_atest_func = -1;
static GLint g_u_atest_ref = -1;
static GLint g_u_atest_mask = -1;

/* ---- VBO ---------------------------------------------------------------- */

static GLuint g_vbo = 0;

/* ---- framebuffer target (GE_FBP/GE_FBW) --------------------------------- */

static uint32_t g_fb_addr = 0;
static uint32_t g_fb_stride = 512;
static int g_fb_fmt = 3;

/* ---- window / content geometry ------------------------------------------ */

static int g_window_w = 640;
static int g_window_h = 480;

/* Letterbox: PSP 480x272 -> content 640x363 within 640x480. */
static int g_viewport_x = 0;
static int g_viewport_y = 58;
static int g_viewport_w = 640;
static int g_viewport_h = 363;

/* ---- GPU texture cache (128-slot LRU) ----------------------------------- */

#define TEX_CACHE_SIZE 128

typedef struct {
    int      used;
    uint32_t tex_addr;
    int      tex_w;
    int      tex_h;
    int      tex_bw;
    int      tex_fmt;
    int      tex_swizzle;
    int      clut_fmt;
    int      clut_shift;
    int      clut_mask;
    int      clut_start;
    uint32_t clut_hash;
    uint32_t data_hash;
    uint64_t last_used;
    GLuint   gl_tex;
} tex_cache_entry;

static tex_cache_entry g_tex_cache[TEX_CACHE_SIZE];
static uint64_t g_tex_clock = 0;
static GLuint g_dummy_tex = 0;
static uint32_t g_tex_upload_buf[512 * 512];

static uint32_t fnv1a_bytes(const uint8_t *data, size_t len) {
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < len; i++) {
        h ^= data[i];
        h *= 16777619u;
    }
    return h;
}

static uint32_t compute_clut_hash(const psp_draw_state *st) {
    if (st->tex_fmt < PSP_TPSM_T4 || st->tex_fmt > PSP_TPSM_T32 || !st->clut)
        return 0;
    size_t bytes = (st->tex_fmt == PSP_TPSM_T4) ? 64u : 1024u;
    size_t off = (size_t)st->clut_start * 16u * ((st->clut_fmt == 3) ? 4u : 2u);
    if (off >= 1024u) off = 0;
    if (off + bytes > 1024u) bytes = 1024u - off;
    return fnv1a_bytes(st->clut + off, bytes);
}

static uint32_t compute_data_hash(const psp_draw_state *st) {
    uint32_t h = 2166136261u;
    uint32_t base = st->tex_addr;
    uint32_t span = (uint32_t)(st->tex_bw > 0 ? st->tex_bw : st->tex_w) *
                    (uint32_t)(st->tex_h > 0 ? st->tex_h : 1);
    if (span < 64u) span = 64u;
    uint32_t step = (span / 16u) & ~3u;
    if (step < 4u) step = 4u;
    for (uint32_t i = 0; i < 16u; i++) {
        uint32_t w = psp_read32(base + i * step);
        h ^= w;
        h *= 16777619u;
    }
    return h;
}

static GLuint bind_cached_texture(const psp_draw_state *st) {
    if (!st->tex_enable || !st->tex_addr ||
        st->tex_w <= 0 || st->tex_h <= 0 ||
        st->tex_w > 512 || st->tex_h > 512 ||
        st->tex_fmt > PSP_TPSM_DXT5) {
        return g_dummy_tex;
    }

    uint32_t clut_hash = compute_clut_hash(st);
    uint32_t data_hash = compute_data_hash(st);
    g_tex_clock++;

    int lru_idx = 0;
    uint64_t oldest = UINT64_MAX;

    for (int i = 0; i < TEX_CACHE_SIZE; i++) {
        tex_cache_entry *e = &g_tex_cache[i];
        if (!e->used) {
            lru_idx = i;
            oldest = 0;
            continue;
        }
        if (e->tex_addr == st->tex_addr &&
            e->tex_w == st->tex_w &&
            e->tex_h == st->tex_h &&
            e->tex_bw == st->tex_bw &&
            e->tex_fmt == st->tex_fmt &&
            e->tex_swizzle == st->tex_swizzle &&
            e->clut_fmt == st->clut_fmt &&
            e->clut_shift == st->clut_shift &&
            e->clut_mask == st->clut_mask &&
            e->clut_start == st->clut_start &&
            e->clut_hash == clut_hash &&
            e->data_hash == data_hash) {
            e->last_used = g_tex_clock;
            glBindTexture(GL_TEXTURE_2D, e->gl_tex);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,
                            st->tex_clamp_u ? GL_CLAMP_TO_EDGE : GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,
                            st->tex_clamp_v ? GL_CLAMP_TO_EDGE : GL_REPEAT);
            return e->gl_tex;
        }
        if (e->last_used < oldest) {
            oldest = e->last_used;
            lru_idx = i;
        }
    }

    /* Cache miss: decode texels once via psp_tex_fetch and upload to GL */
    tex_cache_entry *e = &g_tex_cache[lru_idx];
    if (!e->used) {
        glGenTextures(1, &e->gl_tex);
        e->used = 1;
    }

    psp_draw_state fetch_st = *st;
    fetch_st.tex_clamp_u = 1;
    fetch_st.tex_clamp_v = 1;

    int w = st->tex_w;
    int h = st->tex_h;
    for (int y = 0; y < h; y++) {
        uint32_t *row = &g_tex_upload_buf[y * w];
        for (int x = 0; x < w; x++) {
            row[x] = psp_tex_fetch(&fetch_st, x, y);
        }
    }

    glBindTexture(GL_TEXTURE_2D, e->gl_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,
                    st->tex_clamp_u ? GL_CLAMP_TO_EDGE : GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,
                    st->tex_clamp_v ? GL_CLAMP_TO_EDGE : GL_REPEAT);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, g_tex_upload_buf);

    e->tex_addr = st->tex_addr;
    e->tex_w = st->tex_w;
    e->tex_h = st->tex_h;
    e->tex_bw = st->tex_bw;
    e->tex_fmt = st->tex_fmt;
    e->tex_swizzle = st->tex_swizzle;
    e->clut_fmt = st->clut_fmt;
    e->clut_shift = st->clut_shift;
    e->clut_mask = st->clut_mask;
    e->clut_start = st->clut_start;
    e->clut_hash = clut_hash;
    e->data_hash = data_hash;
    e->last_used = g_tex_clock;
    return e->gl_tex;
}

/* ---- shaders ------------------------------------------------------------ */

static const char *vertex_shader_src =
    "#version 300 es\n"
    "in vec3 a_pos;\n"
    "in vec4 a_color;\n"
    "in vec2 a_uv;\n"
    "out vec4 v_color;\n"
    "out vec2 v_uv;\n"
    "void main() {\n"
    "    gl_Position = vec4(a_pos, 1.0);\n"
    "    v_color = a_color;\n"
    "    v_uv = a_uv;\n"
    "}\n";

static const char *fragment_shader_src =
    "#version 300 es\n"
    "precision mediump float;\n"
    "in vec4 v_color;\n"
    "in vec2 v_uv;\n"
    "uniform sampler2D u_tex;\n"
    "uniform int u_tex_enable;\n"
    "uniform int u_tex_func;\n"
    "uniform int u_tex_alpha;\n"
    "uniform int u_tex_double;\n"
    "uniform vec3 u_tex_env;\n"
    "uniform int u_atest;\n"
    "uniform int u_atest_func;\n"
    "uniform int u_atest_ref;\n"
    "uniform int u_atest_mask;\n"
    "out vec4 frag_color;\n"
    "void main() {\n"
    "    vec4 c = v_color;\n"
    "    if (u_tex_enable != 0) {\n"
    "        vec4 t = texture(u_tex, v_uv);\n"
    "        vec3 rgb;\n"
    "        float a;\n"
    "        if (u_tex_func == 0) {\n"
    "            rgb = t.rgb * c.rgb;\n"
    "            a = (u_tex_alpha != 0) ? (t.a * c.a) : c.a;\n"
    "        } else if (u_tex_func == 1) {\n"
    "            rgb = (u_tex_alpha != 0) ? mix(c.rgb, t.rgb, t.a) : t.rgb;\n"
    "            a = c.a;\n"
    "        } else if (u_tex_func == 2) {\n"
    "            rgb = c.rgb * (1.0 - t.rgb) + u_tex_env * t.rgb;\n"
    "            a = (u_tex_alpha != 0) ? (t.a * c.a) : c.a;\n"
    "        } else if (u_tex_func == 3) {\n"
    "            rgb = t.rgb;\n"
    "            a = (u_tex_alpha != 0) ? t.a : c.a;\n"
    "        } else {\n"
    "            rgb = min(t.rgb + c.rgb, vec3(1.0));\n"
    "            a = (u_tex_alpha != 0) ? (t.a * c.a) : c.a;\n"
    "        }\n"
    "        if (u_tex_double != 0) rgb = min(rgb * 2.0, vec3(1.0));\n"
    "        c = vec4(rgb, clamp(a, 0.0, 1.0));\n"
    "    }\n"
    "    if (u_atest != 0) {\n"
    "        int ia = int(c.a * 255.0 + 0.5) & u_atest_mask;\n"
    "        int ref = u_atest_ref & u_atest_mask;\n"
    "        bool pass = true;\n"
    "        if (u_atest_func == 0) pass = false;\n"
    "        else if (u_atest_func == 1) pass = true;\n"
    "        else if (u_atest_func == 2) pass = (ia == ref);\n"
    "        else if (u_atest_func == 3) pass = (ia != ref);\n"
    "        else if (u_atest_func == 4) pass = (ia < ref);\n"
    "        else if (u_atest_func == 5) pass = (ia <= ref);\n"
    "        else if (u_atest_func == 6) pass = (ia > ref);\n"
    "        else pass = (ia >= ref);\n"
    "        if (!pass) discard;\n"
    "    }\n"
    "    frag_color = c;\n"
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
    glViewport(g_viewport_x, g_viewport_y, g_viewport_w, g_viewport_h);
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

    g_window = SDL_CreateWindow("Midnight Club: L.A. Remix",
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
        /* Fallback to GLES 3.0 if 3.2 request is rejected by the driver */
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
    g_gl_owner_tid = pthread_self();
    g_gl_has_owner = 1;

    g_egl_context = eglGetCurrentContext();
    g_egl_surface = eglGetCurrentSurface(EGL_DRAW);
    if (g_egl_context == EGL_NO_CONTEXT || g_egl_surface == EGL_NO_SURFACE) {
        fprintf(stderr, "gles32: warning: eglGetCurrentContext/Surface returned NO_CONTEXT (continuing with SDL GL context)\n");
    }

    /* Shaders */
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
    g_loc_color = glGetAttribLocation(g_program, "a_color");
    g_loc_uv = glGetAttribLocation(g_program, "a_uv");

    g_u_tex = glGetUniformLocation(g_program, "u_tex");
    g_u_tex_enable = glGetUniformLocation(g_program, "u_tex_enable");
    g_u_tex_func = glGetUniformLocation(g_program, "u_tex_func");
    g_u_tex_alpha = glGetUniformLocation(g_program, "u_tex_alpha");
    g_u_tex_double = glGetUniformLocation(g_program, "u_tex_double");
    g_u_tex_env = glGetUniformLocation(g_program, "u_tex_env");
    g_u_atest = glGetUniformLocation(g_program, "u_atest");
    g_u_atest_func = glGetUniformLocation(g_program, "u_atest_func");
    g_u_atest_ref = glGetUniformLocation(g_program, "u_atest_ref");
    g_u_atest_mask = glGetUniformLocation(g_program, "u_atest_mask");

    glGenBuffers(1, &g_vbo);

    /* 1x1 white dummy texture */
    uint32_t white = 0xFFFFFFFFu;
    glGenTextures(1, &g_dummy_tex);
    glBindTexture(GL_TEXTURE_2D, g_dummy_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, &white);

    memset(g_tex_cache, 0, sizeof(g_tex_cache));
    g_tex_clock = 0;

    update_viewport(w, h);
    glClearColor(0.05f, 0.05f, 0.1f, 1.0f);
    glClearDepthf(1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    SDL_GL_SwapWindow(g_window);

    /* Release the GL context from the init thread so the first guest thread
     * that calls gles_draw() or gles_present() can claim it cleanly via
     * SDL_GL_MakeCurrent(). EGL forbids making a context current on thread B
     * while it is still current on thread A! */
    SDL_GL_MakeCurrent(g_window, NULL);
    g_gl_has_owner = 0;

    fprintf(stderr,
            "gles32: initialized %dx%d window, content %dx%d at (%d,%d)\n",
            w, h, g_viewport_w, g_viewport_h, g_viewport_x, g_viewport_y);
    return 0;
}

static void gles_shutdown(void) {
    gl_lock_and_make_current();
    for (int i = 0; i < TEX_CACHE_SIZE; i++) {
        if (g_tex_cache[i].used && g_tex_cache[i].gl_tex) {
            glDeleteTextures(1, &g_tex_cache[i].gl_tex);
        }
    }
    memset(g_tex_cache, 0, sizeof(g_tex_cache));
    if (g_dummy_tex) { glDeleteTextures(1, &g_dummy_tex); g_dummy_tex = 0; }
    if (g_vbo) { glDeleteBuffers(1, &g_vbo); g_vbo = 0; }
    if (g_program) { glDeleteProgram(g_program); g_program = 0; }
    if (g_fragment_shader) { glDeleteShader(g_fragment_shader); g_fragment_shader = 0; }
    if (g_vertex_shader) { glDeleteShader(g_vertex_shader); g_vertex_shader = 0; }
    if (g_controller) { SDL_GameControllerClose(g_controller); g_controller = NULL; }
    if (g_glcontext) {
        SDL_GL_MakeCurrent(g_window, NULL);
        SDL_GL_DeleteContext(g_glcontext);
        g_glcontext = NULL;
    }
    g_gl_has_owner = 0;
    if (g_window) { SDL_DestroyWindow(g_window); g_window = NULL; }
    gl_unlock();
    SDL_QuitSubSystem(SDL_INIT_VIDEO);
}

/* Map one PSP screen-space vertex to NDC.
 * X: [0,480]->[-1,1], Y: [0,272]->[1,-1], Z: [0,65535]->[-1,1]. */
static void map_vertex(int x, int y, float z, float *ox, float *oy, float *oz) {
    *ox = (x / 480.0f) * 2.0f - 1.0f;
    *oy = -((y / 272.0f) * 2.0f - 1.0f);
    float zc = z < 0.0f ? 0.0f : (z > 65535.0f ? 65535.0f : z);
    *oz = (zc / 32767.5f) - 1.0f;
}

static void rgba_to_float(uint32_t rgba, float c[4]) {
    c[0] = ( rgba        & 0xFF) / 255.0f; /* R */
    c[1] = ((rgba >> 8)  & 0xFF) / 255.0f; /* G */
    c[2] = ((rgba >> 16) & 0xFF) / 255.0f; /* B */
    c[3] = ((rgba >> 24) & 0xFF) / 255.0f; /* A */
}

static void apply_blend_state(const psp_draw_state *st) {
    if (!st->blend || st->clear) {
        glDisable(GL_BLEND);
        return;
    }
    glEnable(GL_BLEND);

    GLenum eq = GL_FUNC_ADD;
    switch (st->blend_op) {
    case 0: eq = GL_FUNC_ADD; break;
    case 1: eq = GL_FUNC_SUBTRACT; break;
    case 2: eq = GL_FUNC_REVERSE_SUBTRACT; break;
    case 3: eq = GL_MIN; break;
    case 4: eq = GL_MAX; break;
    default: eq = GL_FUNC_ADD; break;
    }
    glBlendEquation(eq);

    GLenum src = GL_ONE;
    switch (st->blend_src) {
    case 0: src = GL_DST_COLOR; break;
    case 1: src = GL_ONE_MINUS_DST_COLOR; break;
    case 2: src = GL_SRC_ALPHA; break;
    case 3: src = GL_ONE_MINUS_SRC_ALPHA; break;
    case 4: src = GL_DST_ALPHA; break;
    case 5: src = GL_ONE_MINUS_DST_ALPHA; break;
    case 6: src = GL_SRC_ALPHA; break;
    case 7: src = GL_ONE_MINUS_SRC_ALPHA; break;
    case 8: src = GL_DST_ALPHA; break;
    case 9: src = GL_ONE_MINUS_DST_ALPHA; break;
    default: {
        uint32_t f = st->blend_sfix & 0x00FFFFFFu;
        if (f == 0) src = GL_ZERO;
        else if (f == 0x00FFFFFFu) src = GL_ONE;
        else {
            float c[4];
            rgba_to_float(st->blend_sfix | 0xFF000000u, c);
            glBlendColor(c[0], c[1], c[2], c[3]);
            src = GL_CONSTANT_COLOR;
        }
        break;
    }
    }

    GLenum dst = GL_ZERO;
    switch (st->blend_dst) {
    case 0: dst = GL_SRC_COLOR; break;
    case 1: dst = GL_ONE_MINUS_SRC_COLOR; break;
    case 2: dst = GL_SRC_ALPHA; break;
    case 3: dst = GL_ONE_MINUS_SRC_ALPHA; break;
    case 4: dst = GL_DST_ALPHA; break;
    case 5: dst = GL_ONE_MINUS_DST_ALPHA; break;
    case 6: dst = GL_SRC_ALPHA; break;
    case 7: dst = GL_ONE_MINUS_SRC_ALPHA; break;
    case 8: dst = GL_DST_ALPHA; break;
    case 9: dst = GL_ONE_MINUS_DST_ALPHA; break;
    default: {
        uint32_t f = st->blend_dfix & 0x00FFFFFFu;
        if (f == 0) dst = GL_ZERO;
        else if (f == 0x00FFFFFFu) dst = GL_ONE;
        else {
            float c[4];
            rgba_to_float(st->blend_dfix | 0xFF000000u, c);
            glBlendColor(c[0], c[1], c[2], c[3]);
            dst = GL_CONSTANT_COLOR;
        }
        break;
    }
    }

    glBlendFuncSeparate(src, dst, GL_ONE, GL_ZERO);
}

static void apply_depth_and_mask_state(const psp_draw_state *st) {
    if (st->clear) {
        GLboolean cm = st->clear_color ? GL_TRUE : GL_FALSE;
        GLboolean am = st->clear_alpha ? GL_TRUE : GL_FALSE;
        glColorMask(cm, cm, cm, am);
        if (st->clear_depth) {
            glEnable(GL_DEPTH_TEST);
            glDepthFunc(GL_ALWAYS);
            glDepthMask(GL_TRUE);
        } else {
            glDisable(GL_DEPTH_TEST);
            glDepthMask(GL_FALSE);
        }
        return;
    }

    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    int have_z = st->zaddr && st->zbw > 0;
    if (st->ztest && have_z) {
        glEnable(GL_DEPTH_TEST);
        GLenum zf = GL_ALWAYS;
        switch (st->zfunc) {
        case 0: zf = GL_NEVER; break;
        case 1: zf = GL_ALWAYS; break;
        case 2: zf = GL_EQUAL; break;
        case 3: zf = GL_NOTEQUAL; break;
        case 4: zf = GL_LESS; break;
        case 5: zf = GL_LEQUAL; break;
        case 6: zf = GL_GREATER; break;
        default: zf = GL_GEQUAL; break;
        }
        glDepthFunc(zf);
        glDepthMask(st->zwrite ? GL_TRUE : GL_FALSE);
    } else {
        glDisable(GL_DEPTH_TEST);
        glDepthMask(GL_FALSE);
    }
}

static void apply_scissor_state(const psp_draw_state *st) {
    int x0 = st->sc_x0 < 0 ? 0 : (st->sc_x0 > 479 ? 479 : st->sc_x0);
    int y0 = st->sc_y0 < 0 ? 0 : (st->sc_y0 > 271 ? 271 : st->sc_y0);
    int x1 = st->sc_x1 < 0 ? 0 : (st->sc_x1 > 479 ? 479 : st->sc_x1);
    int y1 = st->sc_y1 < 0 ? 0 : (st->sc_y1 > 271 ? 271 : st->sc_y1);
    if (x0 <= 0 && y0 <= 0 && x1 >= 479 && y1 >= 271) {
        glDisable(GL_SCISSOR_TEST);
        return;
    }
    if (x1 < x0 || y1 < y0) {
        glEnable(GL_SCISSOR_TEST);
        glScissor(0, 0, 0, 0);
        return;
    }
    float sx = (float)g_viewport_w / 480.0f;
    float sy = (float)g_viewport_h / 272.0f;
    int gx = g_viewport_x + (int)(x0 * sx);
    int gw = (int)((x1 - x0 + 1) * sx + 0.5f);
    int gy = g_viewport_y + (int)((272 - 1 - y1) * sy);
    int gh = (int)((y1 - y0 + 1) * sy + 0.5f);
    glEnable(GL_SCISSOR_TEST);
    glScissor(gx, gy, gw, gh);
}

/* Max vertices emitted per draw batch */
#define MAX_DRAW_VERT 65536

static void emit_gl_vertex(const psp_vertex *v, uint32_t override_rgba,
                           float inv_tw, float inv_th,
                           float *pos, float *col, float *uv, int idx) {
    map_vertex(v->x, v->y, v->z, &pos[idx * 3 + 0], &pos[idx * 3 + 1], &pos[idx * 3 + 2]);
    rgba_to_float(override_rgba, &col[idx * 4]);
    uv[idx * 2 + 0] = v->u * inv_tw;
    uv[idx * 2 + 1] = v->v * inv_th;
}

static void gles_draw(int prim, const psp_vertex *v, int count, const psp_draw_state *st) {
    if (!v || count < 1 || !st) return;

    /* Only draw to the GL screen buffer when targeting an actual 480-wide display buffer */
    if (g_fb_stride < 480) return;

    static float pos[MAX_DRAW_VERT * 3];
    static float col[MAX_DRAW_VERT * 4];
    static float uv[MAX_DRAW_VERT * 2];
    int n = 0;

    float inv_tw = (st->tex_enable && st->tex_w > 0) ? (1.0f / (float)st->tex_w) : 1.0f;
    float inv_th = (st->tex_enable && st->tex_h > 0) ? (1.0f / (float)st->tex_h) : 1.0f;
    GLenum mode = GL_TRIANGLES;

    switch (prim) {
    case PSP_PRIM_SPRITES:
        for (int i = 0; i + 1 < count && n + 6 <= MAX_DRAW_VERT; i += 2) {
            const psp_vertex *a = &v[i];
            const psp_vertex *b = &v[i + 1];
            if (a->behind || b->behind) continue;
            if (a->x == b->x || a->y == b->y) continue;

            psp_vertex c00 = *a;
            c00.x = a->x; c00.y = a->y; c00.z = b->z; c00.u = a->u; c00.v = a->v;
            psp_vertex c10 = *a;
            c10.x = b->x; c10.y = a->y; c10.z = b->z; c10.u = b->u; c10.v = a->v;
            psp_vertex c01 = *a;
            c01.x = a->x; c01.y = b->y; c01.z = b->z; c01.u = a->u; c01.v = b->v;
            psp_vertex c11 = *b;
            c11.x = b->x; c11.y = b->y; c11.z = b->z; c11.u = b->u; c11.v = b->v;

            emit_gl_vertex(&c00, b->rgba, inv_tw, inv_th, pos, col, uv, n++);
            emit_gl_vertex(&c10, b->rgba, inv_tw, inv_th, pos, col, uv, n++);
            emit_gl_vertex(&c01, b->rgba, inv_tw, inv_th, pos, col, uv, n++);
            emit_gl_vertex(&c10, b->rgba, inv_tw, inv_th, pos, col, uv, n++);
            emit_gl_vertex(&c11, b->rgba, inv_tw, inv_th, pos, col, uv, n++);
            emit_gl_vertex(&c01, b->rgba, inv_tw, inv_th, pos, col, uv, n++);
        }
        mode = GL_TRIANGLES;
        break;

    case PSP_PRIM_TRIANGLES:
        for (int i = 0; i + 2 < count && n + 3 <= MAX_DRAW_VERT; i += 3) {
            const psp_vertex *a = &v[i], *b = &v[i + 1], *c = &v[i + 2];
            if (a->behind || b->behind || c->behind) continue;
            uint32_t ca = st->gouraud ? a->rgba : c->rgba;
            uint32_t cb = st->gouraud ? b->rgba : c->rgba;
            emit_gl_vertex(a, ca, inv_tw, inv_th, pos, col, uv, n++);
            emit_gl_vertex(b, cb, inv_tw, inv_th, pos, col, uv, n++);
            emit_gl_vertex(c, c->rgba, inv_tw, inv_th, pos, col, uv, n++);
        }
        mode = GL_TRIANGLES;
        break;

    case PSP_PRIM_TRIANGLE_STRIP:
        for (int i = 0; i + 2 < count && n + 3 <= MAX_DRAW_VERT; i++) {
            const psp_vertex *a = &v[i], *b = &v[i + 1], *c = &v[i + 2];
            if (a->behind || b->behind || c->behind) continue;
            uint32_t ca = st->gouraud ? a->rgba : c->rgba;
            uint32_t cb = st->gouraud ? b->rgba : c->rgba;
            emit_gl_vertex(a, ca, inv_tw, inv_th, pos, col, uv, n++);
            emit_gl_vertex(b, cb, inv_tw, inv_th, pos, col, uv, n++);
            emit_gl_vertex(c, c->rgba, inv_tw, inv_th, pos, col, uv, n++);
        }
        mode = GL_TRIANGLES;
        break;

    case PSP_PRIM_TRIANGLE_FAN:
        for (int i = 1; i + 1 < count && n + 3 <= MAX_DRAW_VERT; i++) {
            const psp_vertex *a = &v[0], *b = &v[i], *c = &v[i + 1];
            if (a->behind || b->behind || c->behind) continue;
            uint32_t ca = st->gouraud ? a->rgba : c->rgba;
            uint32_t cb = st->gouraud ? b->rgba : c->rgba;
            emit_gl_vertex(a, ca, inv_tw, inv_th, pos, col, uv, n++);
            emit_gl_vertex(b, cb, inv_tw, inv_th, pos, col, uv, n++);
            emit_gl_vertex(c, c->rgba, inv_tw, inv_th, pos, col, uv, n++);
        }
        mode = GL_TRIANGLES;
        break;

    case PSP_PRIM_POINTS:
    case PSP_PRIM_LINES:
    case PSP_PRIM_LINE_STRIP:
        for (int i = 0; i < count && n < MAX_DRAW_VERT; i++) {
            if (v[i].behind) continue;
            emit_gl_vertex(&v[i], v[i].rgba, inv_tw, inv_th, pos, col, uv, n++);
        }
        mode = (prim == PSP_PRIM_POINTS) ? GL_POINTS :
               (prim == PSP_PRIM_LINES)  ? GL_LINES  : GL_LINE_STRIP;
        break;

    default:
        return;
    }

    if (n == 0) return;

    gl_lock_and_make_current();

    glUseProgram(g_program);
    glDisable(GL_CULL_FACE);

    apply_scissor_state(st);
    apply_depth_and_mask_state(st);
    apply_blend_state(st);

    int use_tex = (!st->clear && st->tex_enable && st->tex_addr && st->tex_w > 0 && st->tex_h > 0);
    glActiveTexture(GL_TEXTURE0);
    if (use_tex) {
        bind_cached_texture(st);
    } else {
        glBindTexture(GL_TEXTURE_2D, g_dummy_tex);
    }

    glUniform1i(g_u_tex, 0);
    glUniform1i(g_u_tex_enable, use_tex ? 1 : 0);
    glUniform1i(g_u_tex_func, st->tex_func);
    glUniform1i(g_u_tex_alpha, st->tex_alpha);
    glUniform1i(g_u_tex_double, st->tex_double);
    float env_c[4];
    rgba_to_float(st->tex_env, env_c);
    glUniform3f(g_u_tex_env, env_c[0], env_c[1], env_c[2]);

    glUniform1i(g_u_atest, (!st->clear && st->atest) ? 1 : 0);
    glUniform1i(g_u_atest_func, st->atest_func);
    glUniform1i(g_u_atest_ref, st->atest_ref);
    glUniform1i(g_u_atest_mask, st->atest_mask);

    size_t pos_bytes = (size_t)n * 3 * sizeof(float);
    size_t col_bytes = (size_t)n * 4 * sizeof(float);
    size_t uv_bytes  = (size_t)n * 2 * sizeof(float);

    glBindBuffer(GL_ARRAY_BUFFER, g_vbo);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(pos_bytes + col_bytes + uv_bytes),
                 NULL, GL_STREAM_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, 0, (GLsizeiptr)pos_bytes, pos);
    glBufferSubData(GL_ARRAY_BUFFER, (GLintptr)pos_bytes, (GLsizeiptr)col_bytes, col);
    glBufferSubData(GL_ARRAY_BUFFER, (GLintptr)(pos_bytes + col_bytes), (GLsizeiptr)uv_bytes, uv);

    glEnableVertexAttribArray((GLuint)g_loc_pos);
    glEnableVertexAttribArray((GLuint)g_loc_color);
    glEnableVertexAttribArray((GLuint)g_loc_uv);

    glVertexAttribPointer((GLuint)g_loc_pos, 3, GL_FLOAT, GL_FALSE, 0, (void *)0);
    glVertexAttribPointer((GLuint)g_loc_color, 4, GL_FLOAT, GL_FALSE, 0, (void *)pos_bytes);
    glVertexAttribPointer((GLuint)g_loc_uv, 2, GL_FLOAT, GL_FALSE, 0, (void *)(pos_bytes + col_bytes));

    glDrawArrays(mode, 0, n);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDisableVertexAttribArray((GLuint)g_loc_pos);
    glDisableVertexAttribArray((GLuint)g_loc_color);
    glDisableVertexAttribArray((GLuint)g_loc_uv);
    glUseProgram(0);

    gl_unlock();
}

static void gles_finish(void) {
    gl_lock_and_make_current();
    glFlush();
    gl_unlock();
}

static void poll_sdl_input_locked(void) {
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
            case SDLK_x:
            case SDLK_z:      mask = 0x1000; break;
            case SDLK_c:      mask = 0x0800; break;
            case SDLK_s:      mask = 0x2000; break;
            case SDLK_t:      mask = 0x0400; break;
            case SDLK_RETURN: mask = 0x0008; break;
            case SDLK_SPACE:  mask = 0x0001; break;
            case SDLK_ESCAPE:
                if (ev.type == SDL_KEYDOWN) {
                    psp_exit_requested_set(1);
                    return;
                }
                break;
            default: break;
            }
            if (mask) {
                if (ev.type == SDL_KEYDOWN) g_kbd_buttons |= mask;
                else g_kbd_buttons &= ~mask;
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
        if (SDL_GameControllerGetButton(g_controller, SDL_CONTROLLER_BUTTON_A))          pad_buttons |= 0x1000;
        if (SDL_GameControllerGetButton(g_controller, SDL_CONTROLLER_BUTTON_B))          pad_buttons |= 0x0800;
        if (SDL_GameControllerGetButton(g_controller, SDL_CONTROLLER_BUTTON_X))          pad_buttons |= 0x2000;
        if (SDL_GameControllerGetButton(g_controller, SDL_CONTROLLER_BUTTON_Y))          pad_buttons |= 0x0400;
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
    psp_ctrl_set(g_kbd_buttons | pad_buttons, lx, ly);
}

static void gles_present(void) {
    gl_lock_and_make_current();
    poll_sdl_input_locked();
    if (g_window) {
        glDisable(GL_SCISSOR_TEST);
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
        glDepthMask(GL_TRUE);
        SDL_GL_SwapWindow(g_window);
    }
    gl_unlock();
}

static void gles_set_target(uint32_t addr, uint32_t stride, int fmt) {
    g_fb_addr = addr;
    g_fb_stride = stride ? stride : 512;
    g_fb_fmt = fmt;
    psp_render_software.set_target(addr, stride, fmt);
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
