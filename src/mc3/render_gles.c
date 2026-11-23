/* mc3 — GLES 3.2 render backend for Midnight Club: L.A. Remix (USA).
 *
 * Implements the psp_render_backend interface from psprecomp using SDL2 +
 * EGL + OpenGL ES 3.2.
 *
 * Threading & presentation architecture:
 * - The main OS thread owns the SDL2 window, the EGL/GLES 3.2 context, and
 *   runs `gles32_present_loop()` (`SDL_PollEvent`, GPU command execution into
 *   a 480x272 FBO, fullscreen letterboxed blit, `SDL_GL_SwapWindow`).
 *   The GL context NEVER leaves the main thread.
 * - Guest threads (`user_main`, `MC3Loading`, etc.) record GE draw commands
 *   and vertices into a staging buffer inside `gles_draw()` / `gles_set_target()`
 *   with zero CPU rasterization. On `gles_present()`, the completed frame's
 *   command list (or VRAM snapshot when no GE draws occurred, e.g. FMVs) is
 *   atomically handed to the main thread.
 * - A 128-slot LRU GPU texture cache decodes PSP textures via `psp_tex_fetch()`
 *   only on cache misses.
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

/* ---- quad blit shader + FBO (480x272) ----------------------------------- */

static GLuint g_quad_prog = 0;
static GLint  g_quad_loc_pos = -1;
static GLint  g_quad_loc_uv = -1;
static GLint  g_quad_u_tex = -1;
static GLuint g_quad_vbo = 0;
static GLuint g_fb_tex = 0;

#define FBO_SLOTS 4

typedef struct {
    int      valid;
    uint32_t target_off;
    GLuint   fbo;
    GLuint   tex;
    GLuint   depth;
} gles_fbo_slot;

static gles_fbo_slot g_fbos[FBO_SLOTS];
static GLuint g_last_fbo_tex = 0;

/* ---- GE draw shader + dynamic VBO --------------------------------------- */

static GLuint g_ge_prog = 0;
static GLint  g_ge_loc_pos = -1;
static GLint  g_ge_loc_col = -1;
static GLint  g_ge_loc_uv  = -1;
static GLint  g_ge_u_tex = -1;
static GLint  g_ge_u_tex_enable = -1;
static GLint  g_ge_u_tex_func = -1;
static GLint  g_ge_u_tex_alpha = -1;
static GLint  g_ge_u_tex_double = -1;
static GLint  g_ge_u_tex_env = -1;
static GLint  g_ge_u_atest = -1;
static GLint  g_ge_u_atest_func = -1;
static GLint  g_ge_u_atest_ref = -1;
static GLint  g_ge_u_atest_mask = -1;
static GLint  g_ge_u_clear = -1;
static GLint  g_ge_u_fbo_uv_scale = -1;
static GLuint g_ge_vbo = 0;

/* ---- GPU vertex format -------------------------------------------------- */

typedef struct {
    float x, y, z;
    float r, g, b, a;
    float u, v;
} gles_vert;

typedef struct {
    uint32_t       target_addr;
    GLenum         gl_prim;
    uint32_t       first_vert;
    uint32_t       vert_count;
    psp_draw_state st;
    uint32_t       clut_offset; /* byte offset in clut_buf, UINT32_MAX if none */
    uint32_t       clut_len;
    uint32_t       clut_hash;
    uint32_t       data_hash;
} gles_cmd;

typedef struct {
    gles_cmd  *cmds;
    uint32_t   cmd_count;
    uint32_t   cmd_cap;

    gles_vert *verts;
    uint32_t   vert_count;
    uint32_t   vert_cap;

    uint8_t   *cluts;
    uint32_t   clut_bytes;
    uint32_t   clut_cap;

    uint32_t   disp_addr;
} gles_frame_list;

static gles_frame_list g_stage = {0};
static gles_frame_list g_front = {0};
static gles_frame_list g_exec  = {0};

static pthread_mutex_t g_vram_mutex = PTHREAD_MUTEX_INITIALIZER;
static uint32_t g_fb_addr = 0;
static uint32_t g_fb_stride = 512;
static int      g_fb_fmt = 3;
static uint32_t g_present_stage[480 * 272];
static uint32_t g_front_buf[480 * 272];
static uint32_t g_upload_buf[480 * 272];
static int      g_front_ready = 0;
static int      g_front_dirty = 0;
static int      g_front_has_cmds = 0;
static int      g_ever_had_ge_draws = 0;

/* ---- 128-slot LRU texture cache ----------------------------------------- */

#define TEX_CACHE_SLOTS 128

typedef struct {
    int      valid;
    GLuint   tex;
    uint64_t last_used;
    uint32_t tex_addr;
    int      tex_w, tex_h, tex_bw, tex_fmt, tex_swizzle;
    uint32_t clut_hash;
    uint32_t data_hash;
    int      clut_fmt, clut_shift, clut_mask, clut_start;
} tex_cache_entry;

static tex_cache_entry g_tex_cache[TEX_CACHE_SLOTS];
static uint64_t        g_tex_lru_tick = 0;

/* ---- window / letterbox geometry ---------------------------------------- */

static int g_window_w = 640;
static int g_window_h = 480;
static int g_viewport_x = 0;
static int g_viewport_y = 58;
static int g_viewport_w = 640;
static int g_viewport_h = 363;

/* ---- shaders ------------------------------------------------------------ */

static const char *quad_vs_src =
    "#version 300 es\n"
    "in vec2 a_pos;\n"
    "in vec2 a_uv;\n"
    "out vec2 v_uv;\n"
    "void main() {\n"
    "    gl_Position = vec4(a_pos, 0.0, 1.0);\n"
    "    v_uv = a_uv;\n"
    "}\n";

static const char *quad_fs_src =
    "#version 300 es\n"
    "precision mediump float;\n"
    "in vec2 v_uv;\n"
    "uniform sampler2D u_tex;\n"
    "out vec4 frag_color;\n"
    "void main() {\n"
    "    frag_color = vec4(texture(u_tex, v_uv).rgb, 1.0);\n"
    "}\n";

static const char *ge_vs_src =
    "#version 300 es\n"
    "in vec3 a_pos;\n"
    "in vec4 a_col;\n"
    "in vec2 a_uv;\n"
    "out vec4 v_col;\n"
    "out vec2 v_uv;\n"
    "void main() {\n"
    "    gl_Position = vec4(a_pos, 1.0);\n"
    "    v_col = a_col;\n"
    "    v_uv = a_uv;\n"
    "}\n";

static const char *ge_fs_src =
    "#version 300 es\n"
    "precision mediump float;\n"
    "in vec4 v_col;\n"
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
    "uniform int u_clear;\n"
    "uniform vec2 u_fbo_uv_scale;\n"
    "out vec4 frag_color;\n"
    "void main() {\n"
    "    vec4 col = v_col;\n"
    "    if (u_clear == 0 && u_tex_enable != 0) {\n"
    "        vec4 t = texture(u_tex, v_uv * u_fbo_uv_scale);\n"
    "        if (u_tex_alpha == 0) t.a = 1.0;\n"
    "        if (u_tex_func == 0) { /* MODULATE */\n"
    "            col.rgb = col.rgb * t.rgb;\n"
    "            col.a = (u_tex_alpha != 0) ? (col.a * t.a) : col.a;\n"
    "        } else if (u_tex_func == 1) { /* DECAL */\n"
    "            col.rgb = mix(col.rgb, t.rgb, t.a);\n"
    "        } else if (u_tex_func == 2) { /* BLEND */\n"
    "            col.rgb = mix(col.rgb, u_tex_env, t.rgb);\n"
    "            col.a = (u_tex_alpha != 0) ? (col.a * t.a) : col.a;\n"
    "        } else if (u_tex_func == 3) { /* REPLACE */\n"
    "            col.rgb = t.rgb;\n"
    "            col.a = (u_tex_alpha != 0) ? t.a : col.a;\n"
    "        } else if (u_tex_func == 4) { /* ADD */\n"
    "            col.rgb = min(col.rgb + t.rgb, vec3(1.0));\n"
    "            col.a = (u_tex_alpha != 0) ? (col.a * t.a) : col.a;\n"
    "        }\n"
    "        if (u_tex_double != 0) {\n"
    "            col.rgb = min(col.rgb * 2.0, vec3(1.0));\n"
    "        }\n"
    "    }\n"
    "    if (u_clear == 0 && u_atest != 0) {\n"
    "        int a = (int(clamp(col.a, 0.0, 1.0) * 255.0 + 0.5)) & u_atest_mask;\n"
    "        int r = u_atest_ref & u_atest_mask;\n"
    "        bool pass = true;\n"
    "        if (u_atest_func == 0) pass = false;\n"
    "        else if (u_atest_func == 1) pass = true;\n"
    "        else if (u_atest_func == 2) pass = (a == r);\n"
    "        else if (u_atest_func == 3) pass = (a != r);\n"
    "        else if (u_atest_func == 4) pass = (a < r);\n"
    "        else if (u_atest_func == 5) pass = (a <= r);\n"
    "        else if (u_atest_func == 6) pass = (a > r);\n"
    "        else if (u_atest_func == 7) pass = (a >= r);\n"
    "        if (!pass) discard;\n"
    "    }\n"
    "    frag_color = col;\n"
    "}\n";

/* ---- helpers ------------------------------------------------------------ */

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

static GLuint link_program(const char *vs_src, const char *fs_src) {
    GLuint vs = compile_shader(GL_VERTEX_SHADER, vs_src);
    GLuint fs = compile_shader(GL_FRAGMENT_SHADER, fs_src);
    GLuint p = glCreateProgram();
    glAttachShader(p, vs);
    glAttachShader(p, fs);
    glLinkProgram(p);
    glDeleteShader(vs);
    glDeleteShader(fs);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetProgramInfoLog(p, sizeof(log), NULL, log);
        fprintf(stderr, "gles32: program link failed:\n%s\n", log);
        glDeleteProgram(p);
        return 0;
    }
    return p;
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

static uint32_t fnv1a_32(const uint8_t *data, uint32_t len) {
    uint32_t h = 2166136261u;
    for (uint32_t i = 0; i < len; i++) {
        h ^= data[i];
        h *= 16777619u;
    }
    return h;
}

static void free_frame_list(gles_frame_list *fl) {
    free(fl->cmds);
    free(fl->verts);
    free(fl->cluts);
    memset(fl, 0, sizeof(*fl));
}

static void swap_frame_lists(gles_frame_list *a, gles_frame_list *b) {
    gles_frame_list tmp = *a;
    *a = *b;
    *b = tmp;
}

static GLenum map_depth_func(int zfunc) {
    switch (zfunc) {
    case 0: return GL_NEVER;
    case 1: return GL_ALWAYS;
    case 2: return GL_EQUAL;
    case 3: return GL_NOTEQUAL;
    case 4: return GL_LESS;
    case 5: return GL_LEQUAL;
    case 6: return GL_GREATER;
    case 7: return GL_GEQUAL;
    default: return GL_ALWAYS;
    }
}

static GLenum map_blend_eq(int op) {
    switch (op) {
    case 0: return GL_FUNC_ADD;
    case 1: return GL_FUNC_SUBTRACT;
    case 2: return GL_FUNC_REVERSE_SUBTRACT;
    case 3: return GL_MIN;
    case 4: return GL_MAX;
    default: return GL_FUNC_ADD;
    }
}

static GLenum map_blend_factor(int f, int is_src) {
    switch (f) {
    case 0: return is_src ? GL_DST_COLOR : GL_SRC_COLOR;
    case 1: return is_src ? GL_ONE_MINUS_DST_COLOR : GL_ONE_MINUS_SRC_COLOR;
    case 2: return GL_SRC_ALPHA;
    case 3: return GL_ONE_MINUS_SRC_ALPHA;
    case 4: return GL_DST_ALPHA;
    case 5: return GL_ONE_MINUS_DST_ALPHA;
    case 6: return GL_SRC_ALPHA;
    case 7: return GL_ONE_MINUS_SRC_ALPHA;
    case 8: return GL_DST_ALPHA;
    case 9: return GL_ONE_MINUS_DST_ALPHA;
    case 10: return GL_CONSTANT_COLOR;
    default: return is_src ? GL_ONE : GL_ZERO;
    }
}

static gles_fbo_slot *get_fbo_for_target(uint32_t target_off, int create) {
    for (int i = 0; i < FBO_SLOTS; i++) {
        if (g_fbos[i].valid && g_fbos[i].target_off == target_off)
            return &g_fbos[i];
    }
    if (!create) return NULL;

    int slot_idx = 0;
    for (int i = 0; i < FBO_SLOTS; i++) {
        if (!g_fbos[i].valid) { slot_idx = i; break; }
    }
    gles_fbo_slot *s = &g_fbos[slot_idx];
    if (!s->fbo) {
        glGenTextures(1, &s->tex);
        glBindTexture(GL_TEXTURE_2D, s->tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 480, 272, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, NULL);

        glGenRenderbuffers(1, &s->depth);
        glBindRenderbuffer(GL_RENDERBUFFER, s->depth);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, 480, 272);
        glBindRenderbuffer(GL_RENDERBUFFER, 0);

        glGenFramebuffers(1, &s->fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, s->fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, s->tex, 0);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, s->depth);
        GLenum st = glCheckFramebufferStatus(GL_FRAMEBUFFER);
        if (st != GL_FRAMEBUFFER_COMPLETE) {
            fprintf(stderr, "gles32: FBO incomplete: 0x%x\n", (unsigned)st);
        }
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
    s->valid = 1;
    s->target_off = target_off;
    return s;
}

static uint32_t hash_tex_sample(uint32_t tex_addr, int tex_w, int tex_h, int tex_bw, int tex_fmt) {
    if (!tex_addr || tex_w <= 0 || tex_h <= 0) return 0;
    uint32_t bpp_bits = 16u;
    switch (tex_fmt) {
        case PSP_TPSM_T4:
        case 8:
            bpp_bits = 4u; break;
        case PSP_TPSM_T8:
        case 9:
        case 10:
            bpp_bits = 8u; break;
        case 3:
        case PSP_TPSM_T32:
            bpp_bits = 32u; break;
        default:
            bpp_bits = 16u; break;
    }
    uint32_t stride = (uint32_t)(tex_bw > 0 ? tex_bw : tex_w);
    uint32_t bytes = (uint32_t)(((uint64_t)stride * (uint64_t)tex_h * bpp_bits) >> 3);
    if (bytes < 4u) bytes = 4u;
    const uint8_t *p = (const uint8_t *)psp_mem_ptr(tex_addr, bytes);
    if (!p) {
        bytes = 256u;
        p = (const uint8_t *)psp_mem_ptr(tex_addr, bytes);
        if (!p) return 0;
    }
    uint32_t nwords = bytes >> 2;
    if (nwords == 0) return 0;
    uint32_t step = nwords / 64u;
    if (step == 0) step = 1u;
    uint32_t h = 2166136261u;
    uint32_t samples = 0;
    for (uint32_t i = 0; i < nwords && samples < 64u; i += step, samples++) {
        uint32_t off = i << 2;
        uint32_t w = (uint32_t)p[off] | ((uint32_t)p[off + 1] << 8) | ((uint32_t)p[off + 2] << 16) | ((uint32_t)p[off + 3] << 24);
        h ^= w;
        h *= 16777619u;
    }
    return h ? h : 1u;
}

static GLuint get_cached_texture(const gles_cmd *cmd, const uint8_t *clut_data) {
    const psp_draw_state *st = &cmd->st;
    int w = st->tex_w > 0 ? st->tex_w : 1;
    int h = st->tex_h > 0 ? st->tex_h : 1;
    if (w > 512) w = 512;
    if (h > 512) h = 512;

    g_tex_lru_tick++;

    int lru_idx = 0;
    uint64_t lru_min = UINT64_MAX;
    for (int i = 0; i < TEX_CACHE_SLOTS; i++) {
        tex_cache_entry *e = &g_tex_cache[i];
        if (!e->valid) {
            lru_idx = i;
            break;
        }
        if (e->tex_addr == st->tex_addr &&
            e->tex_w == w && e->tex_h == h &&
            e->tex_bw == st->tex_bw && e->tex_fmt == st->tex_fmt &&
            e->tex_swizzle == st->tex_swizzle &&
            e->clut_hash == cmd->clut_hash &&
            e->data_hash == cmd->data_hash &&
            e->clut_fmt == st->clut_fmt &&
            e->clut_shift == st->clut_shift &&
            e->clut_mask == st->clut_mask &&
            e->clut_start == st->clut_start) {
            e->last_used = g_tex_lru_tick;
            return e->tex;
        }
        if (e->last_used < lru_min) {
            lru_min = e->last_used;
            lru_idx = i;
        }
    }

    tex_cache_entry *slot = &g_tex_cache[lru_idx];
    if (!slot->tex) glGenTextures(1, &slot->tex);
    slot->valid = 1;
    slot->last_used = g_tex_lru_tick;
    slot->tex_addr = st->tex_addr;
    slot->tex_w = w;
    slot->tex_h = h;
    slot->tex_bw = st->tex_bw;
    slot->tex_fmt = st->tex_fmt;
    slot->tex_swizzle = st->tex_swizzle;
    slot->clut_hash = cmd->clut_hash;
    slot->data_hash = cmd->data_hash;
    slot->clut_fmt = st->clut_fmt;
    slot->clut_shift = st->clut_shift;
    slot->clut_mask = st->clut_mask;
    slot->clut_start = st->clut_start;

    psp_draw_state local_st = *st;
    local_st.clut = clut_data;

    static uint32_t tex_decode[512 * 512];
    for (int y = 0; y < h; y++) {
        uint32_t *row = &tex_decode[y * w];
        for (int x = 0; x < w; x++) {
            row[x] = psp_tex_fetch(&local_st, x, y);
        }
    }

    glBindTexture(GL_TEXTURE_2D, slot->tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, tex_decode);
    return slot->tex;
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
    SDL_GL_SetSwapInterval(1);

    g_egl_context = eglGetCurrentContext();
    g_egl_surface = eglGetCurrentSurface(EGL_DRAW);
    if (g_egl_context == EGL_NO_CONTEXT || g_egl_surface == EGL_NO_SURFACE) {
        fprintf(stderr, "gles32: warning: eglGetCurrentContext/Surface returned NO_CONTEXT\n");
    }

    g_quad_prog = link_program(quad_vs_src, quad_fs_src);
    g_ge_prog   = link_program(ge_vs_src, ge_fs_src);
    if (!g_quad_prog || !g_ge_prog) {
        gles_shutdown();
        return -1;
    }

    g_quad_loc_pos = glGetAttribLocation(g_quad_prog, "a_pos");
    g_quad_loc_uv  = glGetAttribLocation(g_quad_prog, "a_uv");
    g_quad_u_tex   = glGetUniformLocation(g_quad_prog, "u_tex");

    g_ge_loc_pos      = glGetAttribLocation(g_ge_prog, "a_pos");
    g_ge_loc_col      = glGetAttribLocation(g_ge_prog, "a_col");
    g_ge_loc_uv       = glGetAttribLocation(g_ge_prog, "a_uv");
    g_ge_u_tex        = glGetUniformLocation(g_ge_prog, "u_tex");
    g_ge_u_tex_enable = glGetUniformLocation(g_ge_prog, "u_tex_enable");
    g_ge_u_tex_func   = glGetUniformLocation(g_ge_prog, "u_tex_func");
    g_ge_u_tex_alpha  = glGetUniformLocation(g_ge_prog, "u_tex_alpha");
    g_ge_u_tex_double = glGetUniformLocation(g_ge_prog, "u_tex_double");
    g_ge_u_tex_env    = glGetUniformLocation(g_ge_prog, "u_tex_env");
    g_ge_u_atest      = glGetUniformLocation(g_ge_prog, "u_atest");
    g_ge_u_atest_func = glGetUniformLocation(g_ge_prog, "u_atest_func");
    g_ge_u_atest_ref  = glGetUniformLocation(g_ge_prog, "u_atest_ref");
    g_ge_u_atest_mask = glGetUniformLocation(g_ge_prog, "u_atest_mask");
    g_ge_u_clear      = glGetUniformLocation(g_ge_prog, "u_clear");
    g_ge_u_fbo_uv_scale = glGetUniformLocation(g_ge_prog, "u_fbo_uv_scale");

    /* Fullscreen quad: (x, y, u, v) with top-left UV (0,0) at NDC (-1, +1) */
    static const float quad_verts[6 * 4] = {
        -1.0f,  1.0f,  0.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 0.0f,
        -1.0f, -1.0f,  0.0f, 1.0f,
         1.0f,  1.0f,  1.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 1.0f,
        -1.0f, -1.0f,  0.0f, 1.0f,
    };
    glGenBuffers(1, &g_quad_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, g_quad_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad_verts), quad_verts, GL_STATIC_DRAW);

    glGenBuffers(1, &g_ge_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    /* 480x272 streaming texture (for VRAM fallback / FMVs) */
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

    /* Pre-create default FBO slot 0 */
    gles_fbo_slot *s0 = get_fbo_for_target(0u, 1);
    if (s0) g_last_fbo_tex = s0->tex;

    update_viewport(w, h);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    SDL_GL_SwapWindow(g_window);

    fprintf(stderr,
            "gles32: initialized %dx%d window, content %dx%d at (%d,%d), GLES %s (GPU raster)\n",
            w, h, g_viewport_w, g_viewport_h, g_viewport_x, g_viewport_y,
            glGetString(GL_VERSION));
    return 0;
}

static void gles_shutdown(void) {
    for (int i = 0; i < TEX_CACHE_SLOTS; i++) {
        if (g_tex_cache[i].tex) {
            glDeleteTextures(1, &g_tex_cache[i].tex);
            g_tex_cache[i].tex = 0;
        }
        g_tex_cache[i].valid = 0;
    }
    for (int i = 0; i < FBO_SLOTS; i++) {
        if (g_fbos[i].fbo) { glDeleteFramebuffers(1, &g_fbos[i].fbo); g_fbos[i].fbo = 0; }
        if (g_fbos[i].depth) { glDeleteRenderbuffers(1, &g_fbos[i].depth); g_fbos[i].depth = 0; }
        if (g_fbos[i].tex) { glDeleteTextures(1, &g_fbos[i].tex); g_fbos[i].tex = 0; }
        g_fbos[i].valid = 0;
    }
    g_last_fbo_tex = 0;
    if (g_fb_tex) { glDeleteTextures(1, &g_fb_tex); g_fb_tex = 0; }
    if (g_ge_vbo) { glDeleteBuffers(1, &g_ge_vbo); g_ge_vbo = 0; }
    if (g_quad_vbo) { glDeleteBuffers(1, &g_quad_vbo); g_quad_vbo = 0; }
    if (g_ge_prog) { glDeleteProgram(g_ge_prog); g_ge_prog = 0; }
    if (g_quad_prog) { glDeleteProgram(g_quad_prog); g_quad_prog = 0; }
    if (g_controller) { SDL_GameControllerClose(g_controller); g_controller = NULL; }
    if (g_glcontext) { SDL_GL_DeleteContext(g_glcontext); g_glcontext = NULL; }
    if (g_window) { SDL_DestroyWindow(g_window); g_window = NULL; }
    free_frame_list(&g_stage);
    free_frame_list(&g_front);
    free_frame_list(&g_exec);
    SDL_QuitSubSystem(SDL_INIT_VIDEO);
}

static void gles_set_target(uint32_t addr, uint32_t stride, int fmt) {
    g_fb_addr = addr;
    g_fb_stride = stride ? stride : 512;
    g_fb_fmt = fmt;
}

static gles_vert make_vert(const psp_vertex *pv, float inv_tw, float inv_th) {
    gles_vert gv;
    gv.x = ((float)pv->x / 240.0f) - 1.0f;
    gv.y = ((float)pv->y / 136.0f) - 1.0f;
    float zc = pv->z < 0.0f ? 0.0f : (pv->z > 65535.0f ? 65535.0f : pv->z);
    gv.z = (zc / 32767.5f) - 1.0f;
    if (gv.z < -0.99999f) gv.z = -0.99999f;
    if (gv.z >  0.99999f) gv.z =  0.99999f;
    gv.r = (float)(pv->rgba & 0xFFu) / 255.0f;
    gv.g = (float)((pv->rgba >> 8) & 0xFFu) / 255.0f;
    gv.b = (float)((pv->rgba >> 16) & 0xFFu) / 255.0f;
    gv.a = (float)((pv->rgba >> 24) & 0xFFu) / 255.0f;
    gv.u = pv->u * inv_tw;
    gv.v = pv->v * inv_th;
    return gv;
}

static int ensure_verts(gles_frame_list *fl, uint32_t add) {
    uint32_t need = fl->vert_count + add;
    if (need <= fl->vert_cap) return 1;
    uint32_t ncap = fl->vert_cap ? fl->vert_cap * 2u : 8192u;
    while (ncap < need) ncap *= 2u;
    gles_vert *p = (gles_vert *)realloc(fl->verts, (size_t)ncap * sizeof(*p));
    if (!p) return 0;
    fl->verts = p;
    fl->vert_cap = ncap;
    return 1;
}

static void push_tri(const psp_vertex *a, const psp_vertex *b, const psp_vertex *c,
                     const psp_draw_state *st, float inv_tw, float inv_th) {
    if (a->behind || b->behind || c->behind) return;
    if (!ensure_verts(&g_stage, 3)) return;
    g_stage.verts[g_stage.vert_count++] = make_vert(a, inv_tw, inv_th);
    if (!st->gouraud) {
        psp_vertex b_flat = *b; b_flat.rgba = c->rgba;
        psp_vertex a_flat = *a; a_flat.rgba = c->rgba;
        g_stage.verts[g_stage.vert_count - 1] = make_vert(&a_flat, inv_tw, inv_th);
        g_stage.verts[g_stage.vert_count++] = make_vert(&b_flat, inv_tw, inv_th);
    } else {
        g_stage.verts[g_stage.vert_count++] = make_vert(b, inv_tw, inv_th);
    }
    g_stage.verts[g_stage.vert_count++] = make_vert(c, inv_tw, inv_th);
}

static void gles_draw(int prim, const psp_vertex *v, int count, const psp_draw_state *st) {
    if (!v || count <= 0 || !st) return;

    float inv_tw = (st->tex_enable && st->tex_w > 0) ? (1.0f / (float)st->tex_w) : 1.0f;
    float inv_th = (st->tex_enable && st->tex_h > 0) ? (1.0f / (float)st->tex_h) : 1.0f;

    uint32_t first_vert = g_stage.vert_count;
    GLenum gl_prim = GL_TRIANGLES;

    switch (prim) {
    case PSP_PRIM_POINTS:
        gl_prim = GL_POINTS;
        if (ensure_verts(&g_stage, (uint32_t)count)) {
            for (int i = 0; i < count; i++) {
                if (!v[i].behind)
                    g_stage.verts[g_stage.vert_count++] = make_vert(&v[i], inv_tw, inv_th);
            }
        }
        break;
    case PSP_PRIM_LINES:
        gl_prim = GL_LINES;
        if (ensure_verts(&g_stage, (uint32_t)count)) {
            for (int i = 0; i + 1 < count; i += 2) {
                if (!v[i].behind && !v[i + 1].behind) {
                    g_stage.verts[g_stage.vert_count++] = make_vert(&v[i], inv_tw, inv_th);
                    g_stage.verts[g_stage.vert_count++] = make_vert(&v[i + 1], inv_tw, inv_th);
                }
            }
        }
        break;
    case PSP_PRIM_LINE_STRIP:
        gl_prim = GL_LINES;
        if (ensure_verts(&g_stage, (uint32_t)(count > 1 ? (count - 1) * 2 : 0))) {
            for (int i = 0; i + 1 < count; i++) {
                if (!v[i].behind && !v[i + 1].behind) {
                    g_stage.verts[g_stage.vert_count++] = make_vert(&v[i], inv_tw, inv_th);
                    g_stage.verts[g_stage.vert_count++] = make_vert(&v[i + 1], inv_tw, inv_th);
                }
            }
        }
        break;
    case PSP_PRIM_TRIANGLES:
        for (int i = 0; i + 2 < count; i += 3)
            push_tri(&v[i], &v[i + 1], &v[i + 2], st, inv_tw, inv_th);
        break;
    case PSP_PRIM_TRIANGLE_STRIP:
        for (int i = 0; i + 2 < count; i++) {
            if (i & 1) push_tri(&v[i + 1], &v[i], &v[i + 2], st, inv_tw, inv_th);
            else       push_tri(&v[i], &v[i + 1], &v[i + 2], st, inv_tw, inv_th);
        }
        break;
    case PSP_PRIM_TRIANGLE_FAN:
        for (int i = 1; i + 1 < count; i++)
            push_tri(&v[0], &v[i], &v[i + 1], st, inv_tw, inv_th);
        break;
    case PSP_PRIM_SPRITES:
        for (int i = 0; i + 1 < count; i += 2) {
            if (v[i].behind || v[i + 1].behind) continue;
            psp_vertex tl = v[i], br = v[i + 1];
            psp_vertex tr = br, bl = br;
            tl.z = br.z;
            tl.rgba = br.rgba;
            tr.x = br.x; tr.y = tl.y; tr.u = br.u; tr.v = tl.v;
            bl.x = tl.x; bl.y = br.y; bl.u = tl.u; bl.v = br.v;
            push_tri(&tl, &tr, &br, st, inv_tw, inv_th);
            push_tri(&tl, &br, &bl, st, inv_tw, inv_th);
        }
        break;
    default:
        return;
    }

    uint32_t vcount = g_stage.vert_count - first_vert;
    if (vcount == 0) return;

    if (g_stage.cmd_count >= g_stage.cmd_cap) {
        uint32_t ncap = g_stage.cmd_cap ? g_stage.cmd_cap * 2u : 1024u;
        gles_cmd *p = (gles_cmd *)realloc(g_stage.cmds, (size_t)ncap * sizeof(*p));
        if (!p) { g_stage.vert_count = first_vert; return; }
        g_stage.cmds = p;
        g_stage.cmd_cap = ncap;
    }

    gles_cmd *cmd = &g_stage.cmds[g_stage.cmd_count++];
    cmd->target_addr = g_fb_addr;
    cmd->gl_prim = gl_prim;
    cmd->first_vert = first_vert;
    cmd->vert_count = vcount;
    cmd->st = *st;
    cmd->st.clut = NULL;
    cmd->clut_offset = UINT32_MAX;
    cmd->clut_len = 0;
    cmd->clut_hash = 0;
    cmd->data_hash = (!st->clear && st->tex_enable && st->tex_addr) ? hash_tex_sample(st->tex_addr, st->tex_w, st->tex_h, st->tex_bw, st->tex_fmt) : 0;

    if (st->tex_enable && st->clut &&
        (st->tex_fmt == PSP_TPSM_T4 || st->tex_fmt == PSP_TPSM_T8 ||
         st->tex_fmt == PSP_TPSM_T16 || st->tex_fmt == PSP_TPSM_T32)) {
        uint32_t clut_sz = 1024u;
        if (g_stage.clut_bytes + clut_sz > g_stage.clut_cap) {
            uint32_t ncap = g_stage.clut_cap ? g_stage.clut_cap * 2u : 65536u;
            while (ncap < g_stage.clut_bytes + clut_sz) ncap *= 2u;
            uint8_t *cp = (uint8_t *)realloc(g_stage.cluts, ncap);
            if (cp) {
                g_stage.cluts = cp;
                g_stage.clut_cap = ncap;
            }
        }
        if (g_stage.clut_bytes + clut_sz <= g_stage.clut_cap) {
            memcpy(g_stage.cluts + g_stage.clut_bytes, st->clut, clut_sz);
            cmd->clut_offset = g_stage.clut_bytes;
            cmd->clut_len = clut_sz;
            cmd->clut_hash = fnv1a_32(st->clut, clut_sz);
            g_stage.clut_bytes += clut_sz;
        }
    }
}

static void gles_finish(void) {
}

static void gles_present(void) {
    uint32_t disp_addr = psp_display_framebuffer();
    uint32_t fb_addr = disp_addr ? disp_addr : g_fb_addr;

    if (g_stage.cmd_count > 0) {
        g_ever_had_ge_draws = 1;
        g_stage.disp_addr = fb_addr;
        pthread_mutex_lock(&g_vram_mutex);
        swap_frame_lists(&g_stage, &g_front);
        g_front_has_cmds = 1;
        g_front_ready = 1;
        g_front_dirty = 1;
        pthread_mutex_unlock(&g_vram_mutex);
        g_stage.cmd_count = 0;
        g_stage.vert_count = 0;
        g_stage.clut_bytes = 0;
        return;
    }

    if (g_ever_had_ge_draws) {
        /* 30 FPS / repeated VBlank present with no new GE draws: keep GPU FBOs,
         * do not overwrite with empty guest VRAM. */
        pthread_mutex_lock(&g_vram_mutex);
        g_front.disp_addr = fb_addr;
        g_front_has_cmds = 1;
        g_front_dirty = 1;
        pthread_mutex_unlock(&g_vram_mutex);
        return;
    }

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
    g_front_has_cmds = 0;
    g_front_ready = 1;
    g_front_dirty = 1;
    pthread_mutex_unlock(&g_vram_mutex);
}

static void execute_ge_frame(const gles_frame_list *fl) {
    glViewport(0, 0, 480, 272);
    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_CULL_FACE);

    glUseProgram(g_ge_prog);
    glBindBuffer(GL_ARRAY_BUFFER, g_ge_vbo);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(fl->vert_count * sizeof(gles_vert)),
                 fl->verts, GL_STREAM_DRAW);

    glEnableVertexAttribArray((GLuint)g_ge_loc_pos);
    glEnableVertexAttribArray((GLuint)g_ge_loc_col);
    glEnableVertexAttribArray((GLuint)g_ge_loc_uv);
    glVertexAttribPointer((GLuint)g_ge_loc_pos, 3, GL_FLOAT, GL_FALSE,
                          sizeof(gles_vert), (void *)0);
    glVertexAttribPointer((GLuint)g_ge_loc_col, 4, GL_FLOAT, GL_FALSE,
                          sizeof(gles_vert), (void *)(3 * sizeof(float)));
    glVertexAttribPointer((GLuint)g_ge_loc_uv, 2, GL_FLOAT, GL_FALSE,
                          sizeof(gles_vert), (void *)(7 * sizeof(float)));

    static int dbg_init = 0, dbg_on = 0;
    static uint32_t dbg_frames = 0;
    if (!dbg_init) {
        const char *e = getenv("MC3_DEBUG_GE");
        dbg_on = (e && *e && *e != '0') ? 1 : 0;
        dbg_init = 1;
    }
    int log_this = (dbg_on && fl->cmd_count > 20 && (dbg_frames % 180 == 0));
    dbg_frames++;
    if (log_this) {
        fprintf(stderr, "gles32: frame cmds=%u verts=%u disp=0x%08x\n",
                fl->cmd_count, fl->vert_count, fl->disp_addr);
    }

    uint32_t cur_off = UINT32_MAX;
    gles_fbo_slot *cur_fbo = NULL;
    for (uint32_t i = 0; i < fl->cmd_count; i++) {
        const gles_cmd *cmd = &fl->cmds[i];
        if (log_this && i < 20) {
            const gles_vert *v0 = &fl->verts[cmd->first_vert];
            fprintf(stderr, "  cmd[%u] prim=%u vc=%u tgt=0x%08x clr=%d(%d,%d,%d) z=%d/%d/%d tex=%d(fmt=%d,fn=%d) bl=%d v0=(%.2f,%.2f,%.3f,rgba=%.2f,%.2f,%.2f,%.2f)\n",
                    i, (unsigned)cmd->gl_prim, cmd->vert_count, cmd->target_addr,
                    cmd->st.clear, cmd->st.clear_color, cmd->st.clear_alpha, cmd->st.clear_depth,
                    cmd->st.ztest, cmd->st.zfunc, cmd->st.zwrite,
                    cmd->st.tex_enable, cmd->st.tex_fmt, cmd->st.tex_func, cmd->st.blend,
                    v0->x, v0->y, v0->z, v0->r, v0->g, v0->b, v0->a);
        }
        uint32_t target_off = cmd->target_addr ? (cmd->target_addr & 0x001FFFFFu) : 0u;
        if (target_off != cur_off || !cur_fbo) {
            cur_fbo = get_fbo_for_target(target_off, 1);
            if (!cur_fbo) continue;
            cur_off = target_off;
            glBindFramebuffer(GL_FRAMEBUFFER, cur_fbo->fbo);
            g_last_fbo_tex = cur_fbo->tex;
        }
        const psp_draw_state *st = &cmd->st;

        /* Scissor (PSP inclusive coords) */
        int sx0 = st->sc_x0 < 0 ? 0 : st->sc_x0;
        int sy0 = st->sc_y0 < 0 ? 0 : st->sc_y0;
        int sx1 = st->sc_x1 > 479 ? 479 : st->sc_x1;
        int sy1 = st->sc_y1 > 271 ? 271 : st->sc_y1;
        if (sx1 < sx0 || sy1 < sy0) continue;
        glEnable(GL_SCISSOR_TEST);
        glScissor(sx0, sy0, sx1 - sx0 + 1, sy1 - sy0 + 1);

        if (st->clear) {
            glDisable(GL_BLEND);
            glColorMask(st->clear_color ? GL_TRUE : GL_FALSE,
                        st->clear_color ? GL_TRUE : GL_FALSE,
                        st->clear_color ? GL_TRUE : GL_FALSE,
                        st->clear_alpha ? GL_TRUE : GL_FALSE);
            if (st->clear_depth) {
                glEnable(GL_DEPTH_TEST);
                glDepthFunc(GL_ALWAYS);
                glDepthMask(GL_TRUE);
            } else {
                glDisable(GL_DEPTH_TEST);
                glDepthMask(GL_FALSE);
            }
            glUniform1i(g_ge_u_clear, 1);
            glUniform1i(g_ge_u_tex_enable, 0);
            glUniform1i(g_ge_u_atest, 0);
            glUniform2f(g_ge_u_fbo_uv_scale, 1.0f, 1.0f);
        } else {
      #ifdef PSP_DRAW_STATE_HAS_MASK_CULL
            glColorMask((st->mask_rgb & 0xFFu) != 0xFFu, ((st->mask_rgb >> 8) & 0xFFu) != 0xFFu, ((st->mask_rgb >> 16) & 0xFFu) != 0xFFu, (st->mask_alpha & 0xFFu) != 0xFFu);
            if (st->cull_enable) {
                glEnable(GL_CULL_FACE);
                /* gv.y is Y-down, which flips winding vs GL_CCW */
                glCullFace(st->cull_dir ? GL_BACK : GL_FRONT);
            } else {
                glDisable(GL_CULL_FACE);
            }
            #else
            glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
            #endif
            if (st->ztest) {
                glEnable(GL_DEPTH_TEST);
                glDepthFunc(map_depth_func(st->zfunc));
            } else {
                glDisable(GL_DEPTH_TEST);
            }
            glDepthMask(st->zwrite ? GL_TRUE : GL_FALSE);

            if (st->blend) {
                GLenum src_f = map_blend_factor(st->blend_src, 1);
                GLenum dst_f = map_blend_factor(st->blend_dst, 0);
                glEnable(GL_BLEND);
                glBlendEquation(map_blend_eq(st->blend_op));
                if (st->blend_src == 10 && st->blend_dst == 10 && (st->blend_dfix & 0x00FFFFFFu) == (~st->blend_sfix & 0x00FFFFFFu)) {
                    src_f = GL_CONSTANT_COLOR;
                    dst_f = GL_ONE_MINUS_CONSTANT_COLOR;
                }
                glBlendFuncSeparate(src_f, dst_f, GL_ONE, GL_ZERO);
                if (st->blend_src == 10 || st->blend_dst == 10) {
                    uint32_t fix = (st->blend_src == 10) ? st->blend_sfix : st->blend_dfix;
                    glBlendColor((float)(fix & 0xFFu) / 255.0f,
                                 (float)((fix >> 8) & 0xFFu) / 255.0f,
                                 (float)((fix >> 16) & 0xFFu) / 255.0f,
                                 (float)((fix >> 24) & 0xFFu) / 255.0f);
                }
            } else {
                glDisable(GL_BLEND);
            }

            glUniform1i(g_ge_u_clear, 0);
            glUniform1i(g_ge_u_atest, st->atest ? 1 : 0);
            glUniform1i(g_ge_u_atest_func, st->atest_func);
            glUniform1i(g_ge_u_atest_ref, st->atest_ref);
            glUniform1i(g_ge_u_atest_mask, st->atest_mask);

            if (st->tex_enable) {
                gles_fbo_slot *fbo_src = NULL;
                uint32_t tex_phys = st->tex_addr & 0x1FFFFFFFu;
                if (tex_phys >= 0x04000000u && tex_phys < 0x04200000u) {
                    fbo_src = get_fbo_for_target(tex_phys & 0x001FFFFFu, 0);
                }
                glActiveTexture(GL_TEXTURE0);
                if (fbo_src && fbo_src != cur_fbo) {
                    glBindTexture(GL_TEXTURE_2D, fbo_src->tex);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
                    float sx = (st->tex_w > 0) ? ((float)st->tex_w / 480.0f) : 1.0f;
                    float sy = (st->tex_h > 0) ? ((float)st->tex_h / 272.0f) : 1.0f;
                    glUniform2f(g_ge_u_fbo_uv_scale, sx, sy);
                } else {
                    const uint8_t *clut_ptr = (cmd->clut_offset != UINT32_MAX)
                        ? (fl->cluts + cmd->clut_offset) : NULL;
                    GLuint tex = get_cached_texture(cmd, clut_ptr);
                    glBindTexture(GL_TEXTURE_2D, tex);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,
                                    st->tex_clamp_u ? GL_CLAMP_TO_EDGE : GL_REPEAT);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,
                                    st->tex_clamp_v ? GL_CLAMP_TO_EDGE : GL_REPEAT);
                    glUniform2f(g_ge_u_fbo_uv_scale, 1.0f, 1.0f);
                }
                glUniform1i(g_ge_u_tex, 0);
                glUniform1i(g_ge_u_tex_enable, 1);
                glUniform1i(g_ge_u_tex_func, st->tex_func);
                glUniform1i(g_ge_u_tex_alpha, st->tex_alpha ? 1 : 0);
                glUniform1i(g_ge_u_tex_double, st->tex_double ? 1 : 0);
                glUniform3f(g_ge_u_tex_env,
                            (float)(st->tex_env & 0xFFu) / 255.0f,
                            (float)((st->tex_env >> 8) & 0xFFu) / 255.0f,
                            (float)((st->tex_env >> 16) & 0xFFu) / 255.0f);
            } else {
                glUniform1i(g_ge_u_tex_enable, 0);
            }
        }

        glDrawArrays(cmd->gl_prim, (GLint)cmd->first_vert, (GLsizei)cmd->vert_count);
    }

    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glDepthMask(GL_TRUE);
    glDisableVertexAttribArray((GLuint)g_ge_loc_pos);
    glDisableVertexAttribArray((GLuint)g_ge_loc_col);
    glDisableVertexAttribArray((GLuint)g_ge_loc_uv);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void gles32_present_loop(void) {
    if (!g_window || !g_glcontext) return;

    int has_vsync = (SDL_GL_GetSwapInterval() != 0);
    int use_fbo_tex = 0;
    GLuint cur_fbo_tex = 0;

    fprintf(stderr, "gles32: present loop started (vsync=%d, GPU raster)\n", has_vsync);
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

        int dirty = 0;
        int has_cmds = 0;
        pthread_mutex_lock(&g_vram_mutex);
        if (g_front_ready && g_front_dirty) {
            dirty = 1;
            has_cmds = g_front_has_cmds;
            if (has_cmds) {
                swap_frame_lists(&g_front, &g_exec);
                g_front.cmd_count = 0;
                g_front.vert_count = 0;
                g_front.clut_bytes = 0;
            } else {
                memcpy(g_upload_buf, g_front_buf, sizeof(g_upload_buf));
            }
            g_front_dirty = 0;
        }
        pthread_mutex_unlock(&g_vram_mutex);

        if (dirty) {
            if (has_cmds) {
                if (g_exec.cmd_count > 0) {
                    execute_ge_frame(&g_exec);
                }
                uint32_t disp_off = g_exec.disp_addr ? (g_exec.disp_addr & 0x001FFFFFu) : 0u;
                gles_fbo_slot *disp_slot = get_fbo_for_target(disp_off, 0);
                if (disp_slot) cur_fbo_tex = disp_slot->tex;
                else if (g_last_fbo_tex) cur_fbo_tex = g_last_fbo_tex;
                use_fbo_tex = 1;
            } else {
                glBindTexture(GL_TEXTURE_2D, g_fb_tex);
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 480, 272,
                                GL_RGBA, GL_UNSIGNED_BYTE, g_upload_buf);
                use_fbo_tex = 0;
            }
        }

        glViewport(0, 0, g_window_w, g_window_h);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glViewport(g_viewport_x, g_viewport_y, g_viewport_w, g_viewport_h);
        glUseProgram(g_quad_prog);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, (use_fbo_tex && cur_fbo_tex) ? cur_fbo_tex : g_fb_tex);
        glUniform1i(g_quad_u_tex, 0);

        glBindBuffer(GL_ARRAY_BUFFER, g_quad_vbo);
        glEnableVertexAttribArray((GLuint)g_quad_loc_pos);
        glEnableVertexAttribArray((GLuint)g_quad_loc_uv);
        glVertexAttribPointer((GLuint)g_quad_loc_pos, 2, GL_FLOAT, GL_FALSE,
                              4 * sizeof(float), (void *)0);
        glVertexAttribPointer((GLuint)g_quad_loc_uv, 2, GL_FLOAT, GL_FALSE,
                              4 * sizeof(float), (void *)(2 * sizeof(float)));

        glDrawArrays(GL_TRIANGLES, 0, 6);

        glDisableVertexAttribArray((GLuint)g_quad_loc_pos);
        glDisableVertexAttribArray((GLuint)g_quad_loc_uv);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glUseProgram(0);

        SDL_GL_SwapWindow(g_window);
        if (!has_vsync) usleep(16667);
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
