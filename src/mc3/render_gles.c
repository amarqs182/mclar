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
#include "mc3_telemetry.h"

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
#include <time.h>
#include <unistd.h>

static uint64_t gles_now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)ts.tv_nsec / 1000ULL;
}

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

#define FBO_SLOTS 16

typedef struct {
    int      valid;
    int      has_ge_draws;
    uint32_t target_off;
    GLuint   fbo;
    GLuint   tex;
    GLuint   depth_tex;
    uint32_t last_vram_hash;
} gles_fbo_slot;

static gles_fbo_slot g_fbos[FBO_SLOTS];
static GLuint g_last_fbo_tex = 0;
static GLuint g_fbo_scratch_tex = 0;
static GLuint g_depth_clut_tex = 0;

/* ---- GE draw shader + dynamic VBO --------------------------------------- */

static GLuint g_ge_prog = 0;
static GLint  g_ge_loc_pos = -1;
static GLint  g_ge_loc_col = -1;
static GLint  g_ge_loc_uv  = -1;
static GLint  g_ge_loc_fog = -1;
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
static GLint  g_ge_u_fbo_uv_offset = -1;
static GLint  g_ge_u_depth_clut_mode = -1;
static GLint  g_ge_u_depth_tex = -1;
static GLint  g_ge_u_clut_tex = -1;
static GLint  g_ge_u_clut_shift = -1;
static GLint  g_ge_u_clut_mask = -1;
static GLint  g_ge_u_clut_start = -1;
#ifdef PSP_DRAW_STATE_HAS_STENCIL_FOG
static GLint  g_ge_u_fog_enable = -1;
static GLint  g_ge_u_fog_far = -1;
static GLint  g_ge_u_fog_dist = -1;
static GLint  g_ge_u_fog_color = -1;
#endif
static GLuint g_ge_vbo = 0;

/* ---- GPU vertex format -------------------------------------------------- */

typedef struct {
    float    x, y, z;
    uint32_t rgba;
    float    u, v, q;
    float    fog;
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

/* ---- 1024-slot O(1) hash + LRU texture cache --------------------------- */

#define TEX_CACHE_SLOTS 1024
#define TEX_HASH_BUCKETS 2048

typedef struct {
    int      valid;
    GLuint   tex;
    uint64_t last_used;
    uint32_t tex_addr;
    int      tex_w, tex_h, tex_bw, tex_fmt, tex_swizzle;
    uint32_t clut_hash;
    uint32_t data_hash;
    int      clut_fmt, clut_shift, clut_mask, clut_start;
    uint32_t bucket;
    int      hash_next; /* 0 = end, else (slot_idx + 1) */
} tex_cache_entry;

static tex_cache_entry g_tex_cache[TEX_CACHE_SLOTS];
static int             g_tex_buckets[TEX_HASH_BUCKETS]; /* 0 = empty, else (slot_idx + 1) */
static uint64_t        g_tex_lru_tick = 0;
static uint32_t        g_tex_slots_used = 0;
static uint32_t        g_exec_tex_hits = 0;
static uint32_t        g_exec_tex_misses = 0;
static uint64_t        g_exec_tex_up_us = 0;

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
    "in vec3 a_uv;\n"
    "in float a_fog;\n"
    "out vec4 v_col;\n"
    "out vec3 v_uvq;\n"
    "out float v_fog;\n"
    "void main() {\n"
    "    gl_Position = vec4(a_pos, 1.0);\n"
    "    v_col = a_col;\n"
    "    v_uvq = a_uv;\n"
    "    v_fog = a_fog;\n"
    "}\n";

static const char *ge_fs_src =
    "#version 300 es\n"
    "precision mediump float;\n"
    "in vec4 v_col;\n"
    "in vec3 v_uvq;\n"
    "in float v_fog;\n"
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
    "uniform vec2 u_fbo_uv_offset;\n"
    "uniform int u_depth_clut_mode;\n"
    "uniform highp sampler2D u_depth_tex;\n"
    "uniform sampler2D u_clut_tex;\n"
    "uniform int u_clut_shift;\n"
    "uniform int u_clut_mask;\n"
    "uniform int u_clut_start;\n"
    "uniform int u_fog_enable;\n"
    "uniform highp float u_fog_far;\n"
    "uniform highp float u_fog_dist;\n"
    "uniform vec3 u_fog_color;\n"
    "out vec4 frag_color;\n"
    "void main() {\n"
    "    vec4 col = v_col;\n"
    "    if (u_clear == 0 && u_tex_enable != 0) {\n"
    "        vec2 uv = (v_uvq.z != 0.0) ? (v_uvq.xy / v_uvq.z) : v_uvq.xy;\n"
    "        vec2 suv = uv * u_fbo_uv_scale + u_fbo_uv_offset;\n"
    "        vec4 t;\n"
    "        if (u_depth_clut_mode != 0) {\n"
    "            highp float d = texture(u_depth_tex, suv).r;\n"
    "            highp int z16 = clamp(int(d * 65535.0 + 0.5), 0, 65535);\n"
    "            int idx = (((z16 >> u_clut_shift) & u_clut_mask) + (u_clut_start << 4)) & 511;\n"
    "            t = texelFetch(u_clut_tex, ivec2(idx, 0), 0);\n"
    "        } else {\n"
    "            t = texture(u_tex, suv);\n"
    "        }\n"
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
    "    if (u_clear == 0 && u_fog_enable != 0) {\n"
    "        col.rgb = mix(u_fog_color, col.rgb, clamp(v_fog, 0.0, 1.0));\n"
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

        glGenTextures(1, &s->depth_tex);
        glBindTexture(GL_TEXTURE_2D, s->depth_tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH24_STENCIL8, 480, 272, 0,
                     GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8, NULL);
        glBindTexture(GL_TEXTURE_2D, 0);

        glGenFramebuffers(1, &s->fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, s->fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, s->tex, 0);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D, s->depth_tex, 0);
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
        case PSP_TPSM_T16:
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
    uint32_t step = nwords / 32u;
    if (step == 0) step = 1u;
    uint32_t h = 2166136261u;
    uint32_t samples = 0;
    for (uint32_t i = 0; i < nwords && samples < 32u; i += step, samples++) {
        uint32_t off = i << 2;
        uint32_t w = (uint32_t)p[off] | ((uint32_t)p[off + 1] << 8) | ((uint32_t)p[off + 2] << 16) | ((uint32_t)p[off + 3] << 24);
        h ^= w;
        h *= 16777619u;
    }
    return h ? h : 1u;
}

static inline uint32_t tex_cache_bucket(uint32_t addr, int w, int h, int fmt, uint32_t chash, uint32_t dhash) {
    uint32_t k = addr ^ ((uint32_t)w << 12) ^ ((uint32_t)h << 20) ^ ((uint32_t)fmt << 8) ^ chash ^ dhash;
    k ^= k >> 16;
    k *= 0x85ebca6bu;
    k ^= k >> 13;
    return k & (TEX_HASH_BUCKETS - 1u);
}

static GLuint get_cached_texture(const gles_cmd *cmd, const uint8_t *clut_data) {
    const psp_draw_state *st = &cmd->st;
    int w = st->tex_w > 0 ? st->tex_w : 1;
    int h = st->tex_h > 0 ? st->tex_h : 1;
    if (w > 1024) w = 1024;
    if (h > 1024) h = 1024;

    g_tex_lru_tick++;

    uint32_t b = tex_cache_bucket(st->tex_addr, w, h, st->tex_fmt, cmd->clut_hash, cmd->data_hash);
    int cur = g_tex_buckets[b];
    while (cur > 0) {
        tex_cache_entry *e = &g_tex_cache[cur - 1];
        if (e->valid &&
            e->tex_addr == st->tex_addr &&
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
            g_exec_tex_hits++;
            return e->tex;
        }
        cur = e->hash_next;
    }

    static int free_hint = 0;
    int lru_idx = 0;
    if (free_hint < TEX_CACHE_SLOTS && !g_tex_cache[free_hint].valid) {
        lru_idx = free_hint++;
    } else {
        uint64_t lru_min = UINT64_MAX;
        for (int i = 0; i < TEX_CACHE_SLOTS; i++) {
            if (!g_tex_cache[i].valid) { lru_idx = i; break; }
            if (g_tex_cache[i].last_used < lru_min) {
                lru_min = g_tex_cache[i].last_used;
                lru_idx = i;
            }
        }
    }

    tex_cache_entry *slot = &g_tex_cache[lru_idx];
    if (slot->valid) {
        int *pp = &g_tex_buckets[slot->bucket];
        while (*pp > 0) {
            if (*pp == lru_idx + 1) {
                *pp = slot->hash_next;
                break;
            }
            pp = &g_tex_cache[*pp - 1].hash_next;
        }
    }
    if (!slot->valid) g_tex_slots_used++;
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
    slot->bucket = b;
    slot->hash_next = g_tex_buckets[b];
    g_tex_buckets[b] = lru_idx + 1;

    uint64_t t_up0 = gles_now_us();
    g_exec_tex_misses++;

    psp_draw_state local_st = *st;
    local_st.clut = clut_data;

    static uint32_t tex_decode[1024 * 1024];
    for (int y = 0; y < h; y++) {
        uint32_t *row = &tex_decode[y * w];
        for (int x = 0; x < w; x++) {
            row[x] = psp_tex_fetch(&local_st, x, y);
        }
    }

    glBindTexture(GL_TEXTURE_2D, slot->tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, tex_decode);
    if ((w & (w - 1)) == 0 && (h & (h - 1)) == 0 && w >= 16 && h >= 16) {
        glGenerateMipmap(GL_TEXTURE_2D);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    } else {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    }
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    g_exec_tex_up_us += gles_now_us() - t_up0;
    return slot->tex;
}

/* ---- backend entry points ----------------------------------------------- */

static SDL_AudioDeviceID g_audio_dev = 0;

#define AUDIO_RING_FRAMES 16384
#define AUDIO_MAX_STREAMS 9
static pthread_mutex_t g_audio_mtx = PTHREAD_MUTEX_INITIALIZER;
static uint64_t g_audio_seq = 0;
static struct {
    int used;
    int primed;
    uint64_t rpos;
    uint64_t wpos;
    uint64_t last_seq;
    uint32_t idle_cnt;
    int16_t  last_l, last_r;
    int      fade_in_rem;
    int16_t  buf[AUDIO_RING_FRAMES * 2];
} g_audio_streams[AUDIO_MAX_STREAMS];

static uint32_t gles_audio_buffered_frames(int slot) {
    if (!g_audio_dev || slot < 0 || slot >= AUDIO_MAX_STREAMS) return 0;
    pthread_mutex_lock(&g_audio_mtx);
    uint64_t avail = g_audio_streams[slot].used
        ? (g_audio_streams[slot].wpos - g_audio_streams[slot].rpos)
        : 0;
    pthread_mutex_unlock(&g_audio_mtx);
    return (avail > UINT32_MAX) ? UINT32_MAX : (uint32_t)avail;
}

static void SDLCALL gles_audio_callback(void *userdata, Uint8 *stream, int len) {
    (void)userdata;
    int16_t *out = (int16_t *)stream;
    int frames = len / 4;
    int32_t mix[4096 * 2];
    if (frames > 4096) frames = 4096;
    memset(mix, 0, (size_t)frames * 2u * sizeof(int32_t));

    static uint64_t last_cb_us = 0;
    uint64_t now_us = gles_now_us();
    double cb_gap_ms = last_cb_us ? ((double)(now_us - last_cb_us) / 1000.0) : 0.0;
    last_cb_us = now_us;

    int active_chs = 0;
    uint32_t min_buf = UINT32_MAX, max_buf = 0;
    int underrun_ch = -1;
    uint32_t underrun_avail = 0, underrun_need = (uint32_t)frames;

    pthread_mutex_lock(&g_audio_mtx);
    for (int s = 0; s < AUDIO_MAX_STREAMS; s++) {
        if (!g_audio_streams[s].used) continue;
        uint64_t avail = g_audio_streams[s].wpos - g_audio_streams[s].rpos;
        if (!g_audio_streams[s].primed) {
            if (avail < 1024u && !(avail > 0u && g_audio_streams[s].idle_cnt >= 2u)) {
                if (++g_audio_streams[s].idle_cnt > 200u)
                    g_audio_streams[s].used = 0;
                continue;
            }
            g_audio_streams[s].primed = 1;
        }
        active_chs++;
        uint32_t b32 = (avail > UINT32_MAX) ? UINT32_MAX : (uint32_t)avail;
        if (b32 < min_buf) min_buf = b32;
        if (b32 > max_buf) max_buf = b32;

        if (avail < (uint64_t)frames && underrun_ch < 0) {
            underrun_ch = s;
            underrun_avail = b32;
        }
        if (avail == 0) {
            if (g_audio_streams[s].last_l || g_audio_streams[s].last_r) {
                int ramp = (frames < 64) ? frames : 64;
                for (int i = 0; i < ramp; i++) {
                    int32_t w = ramp - i;
                    mix[i * 2 + 0] += (int32_t)g_audio_streams[s].last_l * w / ramp;
                    mix[i * 2 + 1] += (int32_t)g_audio_streams[s].last_r * w / ramp;
                }
                g_audio_streams[s].last_l = 0;
                g_audio_streams[s].last_r = 0;
                g_audio_streams[s].fade_in_rem = 64;
            }
            if (++g_audio_streams[s].idle_cnt > 200u) {
                g_audio_streams[s].used = 0;
                g_audio_streams[s].primed = 0;
            }
            continue;
        }
        g_audio_streams[s].idle_cnt = 0;
        int n = (avail < (uint64_t)frames) ? (int)avail : frames;
        for (int i = 0; i < n; i++) {
            uint32_t idx = (uint32_t)((g_audio_streams[s].rpos + (uint64_t)i) % AUDIO_RING_FRAMES) * 2u;
            int16_t sl = g_audio_streams[s].buf[idx + 0];
            int16_t sr = g_audio_streams[s].buf[idx + 1];
            if (g_audio_streams[s].fade_in_rem > 0) {
                int32_t w = 65 - g_audio_streams[s].fade_in_rem;
                sl = (int16_t)(((int32_t)sl * w) / 64);
                sr = (int16_t)(((int32_t)sr * w) / 64);
                g_audio_streams[s].fade_in_rem--;
            }
            mix[i * 2 + 0] += sl;
            mix[i * 2 + 1] += sr;
            g_audio_streams[s].last_l = sl;
            g_audio_streams[s].last_r = sr;
        }
        if (n < frames) {
            if (g_audio_streams[s].last_l || g_audio_streams[s].last_r) {
                int ramp = frames - n;
                if (ramp > 64) ramp = 64;
                for (int i = 0; i < ramp; i++) {
                    int32_t w = ramp - i;
                    mix[(n + i) * 2 + 0] += (int32_t)g_audio_streams[s].last_l * w / ramp;
                    mix[(n + i) * 2 + 1] += (int32_t)g_audio_streams[s].last_r * w / ramp;
                }
                g_audio_streams[s].last_l = 0;
                g_audio_streams[s].last_r = 0;
            }
            g_audio_streams[s].fade_in_rem = 64;
        }
        g_audio_streams[s].rpos += (uint64_t)n;
    }
    pthread_mutex_unlock(&g_audio_mtx);
    mc3_telemetry_on_audio_cb(cb_gap_ms, active_chs,
                              (active_chs > 0) ? min_buf : 0u,
                              max_buf, underrun_ch, underrun_avail, underrun_need);

    for (int i = 0; i < frames * 2; i++) {
        int32_t v = mix[i];
        if (v > 32767) v = 32767; else if (v < -32768) v = -32768;
        out[i] = (int16_t)v;
    }
}

static void gles_audio_sink(const int16_t *pcm, int samples, int freq, int channels) {
    if (!g_audio_dev || !pcm || samples <= 0) return;
    if (samples > AUDIO_RING_FRAMES) samples = AUDIO_RING_FRAMES;
    int slot = psp_audio_current_channel();
    if (slot < 0 || slot >= AUDIO_MAX_STREAMS) slot = 0;

    pthread_mutex_lock(&g_audio_mtx);
    if (!g_audio_streams[slot].used) {
        g_audio_streams[slot].used = 1;
        g_audio_streams[slot].primed = 0;
        g_audio_streams[slot].rpos = 0;
        g_audio_streams[slot].wpos = 0;
        g_audio_streams[slot].idle_cnt = 0;
        g_audio_streams[slot].last_l = 0;
        g_audio_streams[slot].last_r = 0;
        g_audio_streams[slot].fade_in_rem = 0;
    }
    g_audio_streams[slot].last_seq = ++g_audio_seq;
    g_audio_streams[slot].idle_cnt = 0;
    uint64_t avail = g_audio_streams[slot].wpos - g_audio_streams[slot].rpos;
    int dropped = 0;
    if (avail + (uint64_t)samples > AUDIO_RING_FRAMES) {
        dropped = (int)(avail + (uint64_t)samples - AUDIO_RING_FRAMES);
        g_audio_streams[slot].rpos = g_audio_streams[slot].wpos + (uint64_t)samples - AUDIO_RING_FRAMES;
        g_audio_streams[slot].fade_in_rem = 64;
    }
    uint64_t wp = g_audio_streams[slot].wpos;
    for (int i = 0; i < samples; i++) {
        uint32_t idx = (uint32_t)((wp + (uint64_t)i) % AUDIO_RING_FRAMES) * 2u;
        g_audio_streams[slot].buf[idx + 0] = (channels == 1) ? pcm[i] : pcm[i * 2 + 0];
        g_audio_streams[slot].buf[idx + 1] = (channels == 1) ? pcm[i] : pcm[i * 2 + 1];
    }
    g_audio_streams[slot].wpos = wp + (uint64_t)samples;
    uint32_t buf_after = (uint32_t)(g_audio_streams[slot].wpos - g_audio_streams[slot].rpos);
    pthread_mutex_unlock(&g_audio_mtx);
    mc3_telemetry_on_audio_sink(slot, samples, freq, channels, buf_after, dropped);
}

static GLuint g_hud_prog = 0;
static GLuint g_hud_vbo = 0;
static GLuint g_hud_font_tex = 0;
static void init_hud(void);

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
    if (SDL_InitSubSystem(SDL_INIT_AUDIO) == 0) {
        SDL_AudioSpec want, have;
        memset(&want, 0, sizeof(want));
        memset(&have, 0, sizeof(have));
        want.freq = 44100;
        want.format = AUDIO_S16SYS;
        want.channels = 2;
        want.samples = 512;
        want.callback = gles_audio_callback;
        g_audio_dev = SDL_OpenAudioDevice(NULL, 0, &want, &have, 0);
        if (g_audio_dev) {
            mc3_telemetry_log_audio_spec(want.freq, want.samples, want.channels,
                                         have.freq, have.samples, have.channels, have.size);
            SDL_PauseAudioDevice(g_audio_dev, 0);
            psp_audio_set_sink(gles_audio_sink);
            psp_audio_set_buffered_fn(gles_audio_buffered_frames);
        } else {
            fprintf(stderr, "gles32: SDL_OpenAudioDevice failed: %s\n", SDL_GetError());
        }
    } else {
        fprintf(stderr, "gles32: SDL_InitSubSystem(SDL_INIT_AUDIO) failed: %s\n", SDL_GetError());
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
    SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);

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
    g_ge_loc_fog      = glGetAttribLocation(g_ge_prog, "a_fog");
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
    g_ge_u_fbo_uv_scale    = glGetUniformLocation(g_ge_prog, "u_fbo_uv_scale");
    g_ge_u_fbo_uv_offset   = glGetUniformLocation(g_ge_prog, "u_fbo_uv_offset");
    g_ge_u_depth_clut_mode = glGetUniformLocation(g_ge_prog, "u_depth_clut_mode");
    g_ge_u_depth_tex       = glGetUniformLocation(g_ge_prog, "u_depth_tex");
    g_ge_u_clut_tex        = glGetUniformLocation(g_ge_prog, "u_clut_tex");
    g_ge_u_clut_shift      = glGetUniformLocation(g_ge_prog, "u_clut_shift");
    g_ge_u_clut_mask       = glGetUniformLocation(g_ge_prog, "u_clut_mask");
    g_ge_u_clut_start      = glGetUniformLocation(g_ge_prog, "u_clut_start");
#ifdef PSP_DRAW_STATE_HAS_STENCIL_FOG
    g_ge_u_fog_enable = glGetUniformLocation(g_ge_prog, "u_fog_enable");
    g_ge_u_fog_far    = glGetUniformLocation(g_ge_prog, "u_fog_far");
    g_ge_u_fog_dist   = glGetUniformLocation(g_ge_prog, "u_fog_dist");
    g_ge_u_fog_color  = glGetUniformLocation(g_ge_prog, "u_fog_color");
#endif

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

    mc3_telemetry_log_video(w, h, g_viewport_w, g_viewport_h,
                            SDL_GL_GetSwapInterval() != 0,
                            (const char *)glGetString(GL_RENDERER),
                            (const char *)glGetString(GL_VERSION));

    init_hud();

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
        g_tex_cache[i].hash_next = 0;
    }
    memset(g_tex_buckets, 0, sizeof(g_tex_buckets));
    g_tex_slots_used = 0;
    if (g_hud_font_tex) { glDeleteTextures(1, &g_hud_font_tex); g_hud_font_tex = 0; }
    if (g_hud_vbo) { glDeleteBuffers(1, &g_hud_vbo); g_hud_vbo = 0; }
    if (g_hud_prog) { glDeleteProgram(g_hud_prog); g_hud_prog = 0; }
    for (int i = 0; i < FBO_SLOTS; i++) {
        if (g_fbos[i].fbo) { glDeleteFramebuffers(1, &g_fbos[i].fbo); g_fbos[i].fbo = 0; }
        if (g_fbos[i].depth_tex) { glDeleteTextures(1, &g_fbos[i].depth_tex); g_fbos[i].depth_tex = 0; }
        if (g_fbos[i].tex) { glDeleteTextures(1, &g_fbos[i].tex); g_fbos[i].tex = 0; }
        g_fbos[i].valid = 0;
    }
    g_last_fbo_tex = 0;
    if (g_fbo_scratch_tex) { glDeleteTextures(1, &g_fbo_scratch_tex); g_fbo_scratch_tex = 0; }
    if (g_depth_clut_tex) { glDeleteTextures(1, &g_depth_clut_tex); g_depth_clut_tex = 0; }
    if (g_fb_tex) { glDeleteTextures(1, &g_fb_tex); g_fb_tex = 0; }
    if (g_ge_vbo) { glDeleteBuffers(1, &g_ge_vbo); g_ge_vbo = 0; }
    if (g_quad_vbo) { glDeleteBuffers(1, &g_quad_vbo); g_quad_vbo = 0; }
    if (g_ge_prog) { glDeleteProgram(g_ge_prog); g_ge_prog = 0; }
    if (g_quad_prog) { glDeleteProgram(g_quad_prog); g_quad_prog = 0; }
    psp_audio_set_sink(NULL);
    if (g_audio_dev) { SDL_CloseAudioDevice(g_audio_dev); g_audio_dev = 0; }
    SDL_QuitSubSystem(SDL_INIT_AUDIO);
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
#ifdef PSP_VERTEX_HAS_SUBPIXEL_FOG
    gv.x = (pv->fx / 240.0f) - 1.0f;
    gv.y = (pv->fy / 136.0f) - 1.0f;
    gv.fog = pv->fog;
#else
    gv.x = ((float)pv->x / 240.0f) - 1.0f;
    gv.y = ((float)pv->y / 136.0f) - 1.0f;
    gv.fog = 1.0f;
#endif
    float zc = pv->z < 0.0f ? 0.0f : (pv->z > 65535.0f ? 65535.0f : pv->z);
    gv.z = (zc / 32767.5f) - 1.0f;
    if (gv.z < -1.0f) gv.z = -1.0f;
    if (gv.z >  1.0f) gv.z =  1.0f;
    gv.rgba = pv->rgba;
#ifdef PSP_VERTEX_HAS_W
    float q = (pv->w != 0.0f) ? pv->w : 1.0f;
#else
    float q = 1.0f;
#endif
    gv.u = (pv->u * inv_tw) * q;
    gv.v = (pv->v * inv_th) * q;
    gv.q = q;
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

static inline void emit_tri_verts(const gles_vert *a, const gles_vert *b, const gles_vert *c, int gouraud) {
    gles_vert *dst = &g_stage.verts[g_stage.vert_count];
    dst[0] = *a;
    dst[1] = *b;
    dst[2] = *c;
    if (!gouraud) {
        dst[0].rgba = dst[1].rgba = c->rgba;
    }
    g_stage.vert_count += 3;
}

static void gles_draw(int prim, const psp_vertex *v, int count, const psp_draw_state *st) {
    if (!v || count <= 0 || !st) return;
    if (count > 65536) count = 65536;

    float inv_tw = (st->tex_enable && st->tex_w > 0) ? (1.0f / (float)st->tex_w) : 1.0f;
    float inv_th = (st->tex_enable && st->tex_h > 0) ? (1.0f / (float)st->tex_h) : 1.0f;

    uint32_t max_out = 0;
    GLenum gl_prim = GL_TRIANGLES;
    switch (prim) {
    case PSP_PRIM_POINTS:         gl_prim = GL_POINTS; max_out = (uint32_t)count; break;
    case PSP_PRIM_LINES:          gl_prim = GL_LINES;  max_out = (uint32_t)count; break;
    case PSP_PRIM_LINE_STRIP:     gl_prim = GL_LINES;  max_out = (uint32_t)(count > 1 ? (count - 1) * 2 : 0); break;
    case PSP_PRIM_TRIANGLES:      max_out = (uint32_t)((count / 3) * 3); break;
    case PSP_PRIM_TRIANGLE_STRIP: max_out = (uint32_t)(count >= 3 ? (count - 2) * 3 : 0); break;
    case PSP_PRIM_TRIANGLE_FAN:   max_out = (uint32_t)(count >= 3 ? (count - 2) * 3 : 0); break;
    case PSP_PRIM_SPRITES:        max_out = (uint32_t)((count / 2) * 6); break;
    default: return;
    }
    if (max_out == 0 || !ensure_verts(&g_stage, max_out)) return;

    uint32_t first_vert = g_stage.vert_count;

    if (prim != PSP_PRIM_SPRITES) {
        static gles_vert cv[65536];
        for (int i = 0; i < count; i++) {
            if (!v[i].behind) cv[i] = make_vert(&v[i], inv_tw, inv_th);
        }
        switch (prim) {
        case PSP_PRIM_POINTS:
            for (int i = 0; i < count; i++) {
                if (!v[i].behind) g_stage.verts[g_stage.vert_count++] = cv[i];
            }
            break;
        case PSP_PRIM_LINES:
            for (int i = 0; i + 1 < count; i += 2) {
                if (!v[i].behind && !v[i + 1].behind) {
                    g_stage.verts[g_stage.vert_count++] = cv[i];
                    g_stage.verts[g_stage.vert_count++] = cv[i + 1];
                }
            }
            break;
        case PSP_PRIM_LINE_STRIP:
            for (int i = 0; i + 1 < count; i++) {
                if (!v[i].behind && !v[i + 1].behind) {
                    g_stage.verts[g_stage.vert_count++] = cv[i];
                    g_stage.verts[g_stage.vert_count++] = cv[i + 1];
                }
            }
            break;
        case PSP_PRIM_TRIANGLES:
            for (int i = 0; i + 2 < count; i += 3) {
                if (!v[i].behind && !v[i + 1].behind && !v[i + 2].behind)
                    emit_tri_verts(&cv[i], &cv[i + 1], &cv[i + 2], st->gouraud);
            }
            break;
        case PSP_PRIM_TRIANGLE_STRIP:
            for (int i = 0; i + 2 < count; i++) {
                if (!v[i].behind && !v[i + 1].behind && !v[i + 2].behind) {
                    if (i & 1) emit_tri_verts(&cv[i + 1], &cv[i], &cv[i + 2], st->gouraud);
                    else       emit_tri_verts(&cv[i], &cv[i + 1], &cv[i + 2], st->gouraud);
                }
            }
            break;
        case PSP_PRIM_TRIANGLE_FAN:
            if (!v[0].behind) {
                for (int i = 1; i + 1 < count; i++) {
                    if (!v[i].behind && !v[i + 1].behind)
                        emit_tri_verts(&cv[0], &cv[i], &cv[i + 1], st->gouraud);
                }
            }
            break;
        }
    } else {
        for (int i = 0; i + 1 < count; i += 2) {
            if (v[i].behind || v[i + 1].behind) continue;
            psp_vertex tl = v[i], br = v[i + 1];
            psp_vertex tr = br, bl = br;
            tl.z = br.z;
            tl.rgba = br.rgba;
            tr.x = br.x; tr.y = tl.y; tr.u = br.u; tr.v = tl.v;
            bl.x = tl.x; bl.y = br.y; bl.u = tl.u; bl.v = br.v;
#ifdef PSP_VERTEX_HAS_SUBPIXEL_FOG
            tr.fx = br.fx; tr.fy = tl.fy;
            bl.fx = tl.fx; bl.fy = br.fy;
#endif
            gles_vert gtl = make_vert(&tl, inv_tw, inv_th);
            gles_vert gtr = make_vert(&tr, inv_tw, inv_th);
            gles_vert gbr = make_vert(&br, inv_tw, inv_th);
            gles_vert gbl = make_vert(&bl, inv_tw, inv_th);
            emit_tri_verts(&gtl, &gtr, &gbr, st->gouraud);
            emit_tri_verts(&gtl, &gbr, &gbl, st->gouraud);
        }
    }

    uint32_t vcount = g_stage.vert_count - first_vert;
    if (vcount == 0) return;

    static uint32_t stage_tex_epoch = 1;
    static struct {
        uint32_t epoch;
        uint32_t tex_addr;
        int      tex_w, tex_h, tex_bw, tex_fmt;
        uint32_t hash;
    } stage_tex_cache[256];
    static uint32_t last_clut_epoch = 0, last_clut_gen = 0, last_clut_offset = UINT32_MAX, last_clut_hash = 0;
    if (g_stage.cmd_count == 0) stage_tex_epoch++;

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
    if (prim == PSP_PRIM_SPRITES || prim < PSP_PRIM_TRIANGLES) {
#ifdef PSP_DRAW_STATE_HAS_MASK_CULL
        cmd->st.cull_enable = 0;
#endif
    }
    cmd->st.clut = NULL;
    cmd->clut_offset = UINT32_MAX;
    cmd->clut_len = 0;
    cmd->clut_hash = 0;
    if (!st->clear && st->tex_enable && st->tex_addr) {
        uint32_t slot = (st->tex_addr ^ ((uint32_t)st->tex_w << 7) ^ ((uint32_t)st->tex_h << 13) ^ (uint32_t)st->tex_fmt) & 255u;
        if (stage_tex_cache[slot].epoch == stage_tex_epoch &&
            stage_tex_cache[slot].tex_addr == st->tex_addr &&
            stage_tex_cache[slot].tex_w == st->tex_w &&
            stage_tex_cache[slot].tex_h == st->tex_h &&
            stage_tex_cache[slot].tex_bw == st->tex_bw &&
            stage_tex_cache[slot].tex_fmt == st->tex_fmt) {
            cmd->data_hash = stage_tex_cache[slot].hash;
        } else {
            cmd->data_hash = hash_tex_sample(st->tex_addr, st->tex_w, st->tex_h, st->tex_bw, st->tex_fmt);
            stage_tex_cache[slot].epoch = stage_tex_epoch;
            stage_tex_cache[slot].tex_addr = st->tex_addr;
            stage_tex_cache[slot].tex_w = st->tex_w;
            stage_tex_cache[slot].tex_h = st->tex_h;
            stage_tex_cache[slot].tex_bw = st->tex_bw;
            stage_tex_cache[slot].tex_fmt = st->tex_fmt;
            stage_tex_cache[slot].hash = cmd->data_hash;
        }
    } else {
        cmd->data_hash = 0;
    }

    if (st->tex_enable && st->clut &&
        (st->tex_fmt == PSP_TPSM_T4 || st->tex_fmt == PSP_TPSM_T8 ||
         st->tex_fmt == PSP_TPSM_T16 || st->tex_fmt == PSP_TPSM_T32)) {
        if (last_clut_epoch == stage_tex_epoch && last_clut_gen == st->clut_gen &&
            last_clut_offset != UINT32_MAX) {
            cmd->clut_offset = last_clut_offset;
            cmd->clut_len = 1024u;
            cmd->clut_hash = last_clut_hash;
        } else {
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
                last_clut_epoch = stage_tex_epoch;
                last_clut_gen = st->clut_gen;
                last_clut_offset = cmd->clut_offset;
                last_clut_hash = cmd->clut_hash;
            }
        }
    }
}

static void gles_finish(void) {
}

static void gles_present(void) {
    uint32_t disp_addr = psp_display_framebuffer();
    uint32_t fb_addr = disp_addr ? disp_addr : g_fb_addr;
    uint32_t frame_draws = g_stage.cmd_count;
    uint32_t frame_verts = g_stage.vert_count;
    int had_draws = 0;

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
        had_draws = 1;
    }

    uint64_t vb = psp_display_vblanks();
    static uint64_t last_tel_vb = UINT64_MAX;
    int is_new_vb = (vb != last_tel_vb);
    if (is_new_vb) last_tel_vb = vb;

    double pace_wait_ms = 0.0;
    static int s_unpaced = -1;
    if (s_unpaced < 0) s_unpaced = (getenv("MC3_UNPACED") != NULL);
    if (!s_unpaced) {
        static uint64_t next_present_us = 0;
        static uint64_t last_paced_vblank = UINT64_MAX;
        static uint64_t last_draw_vb = UINT64_MAX;
        int should_pace = 0;
        if (is_new_vb && vb != last_paced_vblank) {
            should_pace = 1;
            last_paced_vblank = vb;
        } else if (had_draws) {
            if (vb == last_draw_vb) {
                should_pace = 1;
            }
            last_draw_vb = vb;
        }
        if (should_pace) {
            uint64_t now_us = gles_now_us();
            if (next_present_us == 0) {
                next_present_us = now_us + 16667ULL;
            } else {
                uint64_t w0 = now_us;
                int yielded = 0;
                while (now_us < next_present_us && !psp_exit_requested()) {
                    psp_threadman_yield();
                    yielded = 1;
                    now_us = gles_now_us();
                    if (now_us < next_present_us && (next_present_us - now_us) > 1500ULL) {
                        usleep(500);
                        now_us = gles_now_us();
                    }
                }
                if (!yielded) psp_threadman_yield();
                pace_wait_ms = (double)(now_us - w0) / 1000.0;
                next_present_us += 16667ULL;
                if (next_present_us + 50000ULL < now_us) {
                    next_present_us = now_us + 16667ULL;
                }
            }
        }
    }

    if (had_draws || (is_new_vb && !g_ever_had_ge_draws)) {
        static uint64_t prev_pres_us = 0;
        static uint64_t prev_ge_us = 0, prev_lists = 0, prev_bb_t = 0, prev_bb_c = 0;
        uint64_t now_p = gles_now_us();
        double ft_ms = prev_pres_us ? ((double)(now_p - prev_pres_us) / 1000.0) : 16.667;
        prev_pres_us = now_p;
        uint64_t cur_ge_us = psp_ge_run_list_us();
        uint64_t cur_lists = psp_ge_lists();
        uint64_t cur_bb_t = psp_ge_bbox_tested();
        uint64_t cur_bb_c = psp_ge_bbox_culled();
        mc3_telemetry_on_guest_frame(ft_ms,
                                     (double)(cur_ge_us - prev_ge_us) / 1000.0,
                                     pace_wait_ms,
                                     frame_draws, frame_verts,
                                     cur_lists - prev_lists,
                                     cur_bb_t - prev_bb_t,
                                     cur_bb_c - prev_bb_c);
        prev_ge_us = cur_ge_us;
        prev_lists = cur_lists;
        prev_bb_t = cur_bb_t;
        prev_bb_c = cur_bb_c;
    }

    if (had_draws) return;

    uint32_t disp_w = (uint32_t)psp_display_fb_width();
    int disp_fmt = psp_display_fb_format();
    uint32_t fb_stride = disp_w ? disp_w : ((disp_addr && g_fb_stride < 480) ? 512u : (g_fb_stride ? g_fb_stride : 512u));
    int fb_fmt = (disp_w && disp_fmt >= 0) ? disp_fmt : g_fb_fmt;

    uint32_t cpu_fb = fb_addr & 0x1FFFFFFFu;
    gles_fbo_slot *disp_slot = fb_addr ? get_fbo_for_target(fb_addr & 0x001FFFFFu, 0) : NULL;
    if (disp_slot && disp_slot->has_ge_draws) {
        uint32_t bpp = (fb_fmt == PSP_TPSM_8888) ? 4u : 2u;
        uint32_t total_bytes = fb_stride * 272u * bpp;
        const uint8_t *vp = cpu_fb ? (const uint8_t *)psp_mem_ptr(cpu_fb, total_bytes) : NULL;
        uint32_t vh = 2166136261u;
        if (vp) {
            uint32_t nwords = total_bytes >> 2;
            uint32_t step = nwords / 64u;
            if (step == 0) step = 1u;
            for (uint32_t i = 0; i < nwords; i += step) {
                uint32_t off = i << 2;
                uint32_t w = (uint32_t)vp[off] | ((uint32_t)vp[off + 1] << 8) | ((uint32_t)vp[off + 2] << 16) | ((uint32_t)vp[off + 3] << 24);
                vh ^= w;
                vh *= 16777619u;
            }
        }
        if (vh != disp_slot->last_vram_hash) {
            disp_slot->last_vram_hash = vh;
            disp_slot->has_ge_draws = 0;
        } else {
            /* 30 FPS / repeated VBlank present with no new GE draws: keep GPU FBOs,
             * do not overwrite with empty guest VRAM. */
            pthread_mutex_lock(&g_vram_mutex);
            g_front.disp_addr = fb_addr;
            g_front_has_cmds = 1;
            g_front_dirty = 1;
            pthread_mutex_unlock(&g_vram_mutex);
            return;
        }
    }

    if (!cpu_fb) return;

    if (fb_fmt == PSP_TPSM_8888) {
        for (int y = 0; y < 272; y++) {
            psp_mem_read_block(&g_present_stage[y * 480],
                               cpu_fb + (uint32_t)y * fb_stride * 4,
                               480 * 4);
        }
    } else {
        for (int y = 0; y < 272; y++) {
            uint32_t row_addr = cpu_fb + (uint32_t)y * fb_stride * 2;
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
    glVertexAttribPointer((GLuint)g_ge_loc_col, 4, GL_UNSIGNED_BYTE, GL_TRUE,
                          sizeof(gles_vert), (void *)(3 * sizeof(float)));
    glVertexAttribPointer((GLuint)g_ge_loc_uv, 3, GL_FLOAT, GL_FALSE,
                          sizeof(gles_vert), (void *)(4 * sizeof(float)));
    if (g_ge_loc_fog >= 0) {
        glEnableVertexAttribArray((GLuint)g_ge_loc_fog);
        glVertexAttribPointer((GLuint)g_ge_loc_fog, 1, GL_FLOAT, GL_FALSE,
                              sizeof(gles_vert), (void *)(7 * sizeof(float)));
    }

    static int dbg_init = 0, dbg_on = 0;
    static uint32_t dbg_frames = 0;
    if (!dbg_init) {
        const char *e = getenv("MC3_DEBUG_GE");
        dbg_on = (e && *e && *e != '0') ? 1 : 0;
        dbg_init = 1;
    }
    int log_this = (dbg_on && (dbg_frames % 180) == 0);
    dbg_frames++;
    if (log_this) {
        fprintf(stderr, "gles32: frame cmds=%u verts=%u disp=0x%08x\n",
                fl->cmd_count, fl->vert_count, fl->disp_addr);
    }

    uint32_t cur_off = UINT32_MAX;
    gles_fbo_slot *cur_fbo = NULL;
    if (log_this) {
        uint32_t msk_ff = 0, st_on = 0, atest_on = 0, z_off = 0, cull_on = 0, blend_on = 0, tgt_other = 0, clear_cnt = 0;
        uint32_t disp_off = fl->disp_addr & 0x001FFFFFu;
        for (uint32_t j = 0; j < fl->cmd_count; j++) {
            const psp_draw_state *s = &fl->cmds[j].st;
            if ((s->mask_rgb & 0xFFFFFFu) == 0xFFFFFFu) msk_ff++;
            if (s->stencil_enable) st_on++;
            if (s->atest) atest_on++;
            if (!s->ztest) z_off++;
            if (s->cull_enable) cull_on++;
            if (s->blend) blend_on++;
            if (s->clear) clear_cnt++;
            if ((fl->cmds[j].target_addr & 0x001FFFFFu) != disp_off) tgt_other++;
        }
        fprintf(stderr, "gles32_ge_summary: cmds=%u msk_ff=%u st_on=%u atest_on=%u z_off=%u cull_on=%u blend_on=%u clr=%u tgt_other=%u\n",
                fl->cmd_count, msk_ff, st_on, atest_on, z_off, cull_on, blend_on, clear_cnt, tgt_other);
    }

    for (uint32_t i = 0; i < fl->cmd_count; i++) {
        const gles_cmd *cmd = &fl->cmds[i];
        if (log_this && (i < 10 || (i >= 50 && (i % 40) == 0) || i + 2 >= fl->cmd_count || cmd->st.clear)) {
            const gles_vert *v0 = &fl->verts[cmd->first_vert];
            fprintf(stderr, "  cmd[%u] prim=%u vc=%u tgt=0x%08x clr=%d z=%d/%d/%d tex=%d(0x%08x,%dx%d,fmt=%d,fn=%d) bl=%d(%d,%d) cull=%d/%d st=%d msk=%06x fog=%d(%.2f,%.5f,%06x) v0=(%.2f,%.2f,%.3f,uv=%.3f,%.3f)\n",
                    i, (unsigned)cmd->gl_prim, cmd->vert_count, cmd->target_addr,
                    cmd->st.clear, cmd->st.ztest, cmd->st.zfunc, cmd->st.zwrite,
                    cmd->st.tex_enable, cmd->st.tex_addr, cmd->st.tex_w, cmd->st.tex_h,
                    cmd->st.tex_fmt, cmd->st.tex_func, cmd->st.blend, cmd->st.blend_src, cmd->st.blend_dst,
                    cmd->st.cull_enable, cmd->st.cull_dir, cmd->st.stencil_enable, cmd->st.mask_rgb,
                    cmd->st.fog_enable, cmd->st.fog_far, cmd->st.fog_dist, cmd->st.fog_color & 0xFFFFFFu,
                    v0->x, v0->y, v0->z, v0->u, v0->v);
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
        static uint32_t g_last_z_off = UINT32_MAX;
        static GLuint   g_last_z_tex = 0;

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
            glDisable(GL_CULL_FACE);
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
            glUniform2f(g_ge_u_fbo_uv_offset, 0.0f, 0.0f);
            glUniform1i(g_ge_u_depth_clut_mode, 0);
#ifdef PSP_DRAW_STATE_HAS_STENCIL_FOG
            glDisable(GL_STENCIL_TEST);
            if (st->clear_alpha) {
                glStencilMask(0xFF);
                glClearStencil(0);
                glClear(GL_STENCIL_BUFFER_BIT);
            }
            glUniform1i(g_ge_u_fog_enable, 0);
#endif
        } else {
      #ifdef PSP_DRAW_STATE_HAS_MASK_CULL
            glColorMask((st->mask_rgb & 0xFFu) != 0xFFu, ((st->mask_rgb >> 8) & 0xFFu) != 0xFFu, ((st->mask_rgb >> 16) & 0xFFu) != 0xFFu, (st->mask_alpha & 0xFFu) != 0xFFu);
            if (st->cull_enable) {
                glEnable(GL_CULL_FACE);
                /* PSP GE 0x9B: 0 = cull CW (GL_FRONT under Y-down), 1 = cull CCW (GL_BACK under Y-down) */
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
#ifdef PSP_DRAW_STATE_HAS_STENCIL_FOG
            if (st->stencil_enable) {
                static const GLenum s_ops[6] = {
                    GL_KEEP, GL_ZERO, GL_REPLACE, GL_INVERT, GL_INCR, GL_DECR
                };
                glEnable(GL_STENCIL_TEST);
                glStencilMask(~st->mask_alpha & 0xFFu);
                glStencilFunc(map_depth_func(st->stencil_func), (GLint)st->stencil_ref, (GLuint)st->stencil_mask);
                glStencilOp(s_ops[st->stencil_sfail < 6 ? st->stencil_sfail : 0],
                            s_ops[st->stencil_zfail < 6 ? st->stencil_zfail : 0],
                            s_ops[st->stencil_zpass < 6 ? st->stencil_zpass : 0]);
            } else {
                glDisable(GL_STENCIL_TEST);
                glStencilMask(0);
            }
            /* Per-vertex PSP GE fog factor (v_fog) computed in view space. */
            glUniform1i(g_ge_u_fog_enable, st->fog_enable ? 1 : 0);
            glUniform1f(g_ge_u_fog_far, st->fog_far);
            glUniform1f(g_ge_u_fog_dist, st->fog_dist);
            glUniform3f(g_ge_u_fog_color,
                        (float)(st->fog_color & 0xFFu) / 255.0f,
                        (float)((st->fog_color >> 8) & 0xFFu) / 255.0f,
                        (float)((st->fog_color >> 16) & 0xFFu) / 255.0f);
#endif

            if (st->blend) {
                GLenum src_f = map_blend_factor(st->blend_src, 1);
                GLenum dst_f = map_blend_factor(st->blend_dst, 0);
                glEnable(GL_BLEND);
                glBlendEquationSeparate(map_blend_eq(st->blend_op), GL_FUNC_ADD);
                uint32_t sfix = st->blend_sfix & 0x00FFFFFFu;
                uint32_t dfix = st->blend_dfix & 0x00FFFFFFu;
                int has_const = 0;
                uint32_t const_col = 0;
                if (st->blend_src == 10) {
                    if (sfix == 0x00FFFFFFu) {
                        src_f = GL_ONE;
                    } else if (sfix == 0u) {
                        src_f = GL_ZERO;
                    } else {
                        src_f = GL_CONSTANT_COLOR;
                        has_const = 1;
                        const_col = st->blend_sfix;
                    }
                }
                if (st->blend_dst == 10) {
                    if (dfix == 0x00FFFFFFu) {
                        dst_f = GL_ONE;
                    } else if (dfix == 0u) {
                        dst_f = GL_ZERO;
                    } else if (has_const && dfix == (~sfix & 0x00FFFFFFu)) {
                        dst_f = GL_ONE_MINUS_CONSTANT_COLOR;
                    } else if (!has_const) {
                        dst_f = GL_CONSTANT_COLOR;
                        has_const = 1;
                        const_col = st->blend_dfix;
                    } else {
                        dst_f = GL_CONSTANT_COLOR;
                    }
                }
                glBlendFuncSeparate(src_f, dst_f, GL_ONE, GL_ZERO);
                if (has_const) {
                    glBlendColor((float)(const_col & 0xFFu) / 255.0f,
                                 (float)((const_col >> 8) & 0xFFu) / 255.0f,
                                 (float)((const_col >> 16) & 0xFFu) / 255.0f,
                                 (float)((const_col >> 24) & 0xFFu) / 255.0f);
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
                const uint8_t *clut_ptr = (cmd->clut_offset != UINT32_MAX)
                    ? (fl->cluts + cmd->clut_offset) : NULL;
                gles_fbo_slot *fbo_src = NULL;
                float uv_off_x = 0.0f, uv_off_y = 0.0f;
                int is_depth_clut = 0;
                uint32_t tex_phys = st->tex_addr & 0x1FFFFFFFu;
                if (tex_phys >= 0x04000000u && tex_phys < 0x04800000u) {
                    uint32_t tex_off = tex_phys & 0x001FFFFFu;
                    if (tex_off == g_last_z_off && g_last_z_tex != 0 &&
                        g_last_z_tex != cur_fbo->depth_tex &&
                        (st->tex_fmt == PSP_TPSM_T16 || st->tex_fmt == PSP_TPSM_T8 || st->tex_fmt == PSP_TPSM_T4)) {
                        is_depth_clut = 1;
                    } else {
                        fbo_src = get_fbo_for_target(tex_off, 0);
                        if (!fbo_src) {
                            uint32_t bpp = (st->tex_fmt == PSP_TPSM_8888) ? 4u : 2u;
                            uint32_t bw = (st->tex_bw > 0) ? (uint32_t)st->tex_bw : 512u;
                            uint32_t span = bw * 272u * bpp;
                            for (int sidx = 0; sidx < FBO_SLOTS; sidx++) {
                                gles_fbo_slot *cand = &g_fbos[sidx];
                                if (cand->valid && cand->has_ge_draws &&
                                    tex_off > cand->target_off && tex_off < cand->target_off + span) {
                                    uint32_t boff = tex_off - cand->target_off;
                                    uint32_t px_off = boff / bpp;
                                    uv_off_x = (float)(px_off % bw) / 480.0f;
                                    uv_off_y = (float)(px_off / bw) / 272.0f;
                                    fbo_src = cand;
                                    break;
                                }
                            }
                        }
                    }
                }
                if (is_depth_clut) {
                    uint32_t clut_rgba[512];
                    memset(clut_rgba, 0, sizeof(clut_rgba));
                    if (clut_ptr && cmd->clut_len > 0u) {
                        if (st->clut_fmt == 3) {
                            uint32_t n = cmd->clut_len / 4u;
                            if (n > 256u) n = 256u;
                            for (uint32_t k = 0; k < n; k++) {
                                uint32_t v;
                                memcpy(&v, clut_ptr + k * 4u, 4u);
                                clut_rgba[k] = v;
                            }
                        } else {
                            uint32_t n = cmd->clut_len / 2u;
                            if (n > 512u) n = 512u;
                            for (uint32_t k = 0; k < n; k++) {
                                uint16_t v;
                                memcpy(&v, clut_ptr + k * 2u, 2u);
                                clut_rgba[k] = psp_color_expand(v, st->clut_fmt);
                            }
                        }
                    }
                    if (!g_depth_clut_tex) {
                        glGenTextures(1, &g_depth_clut_tex);
                        glActiveTexture(GL_TEXTURE2);
                        glBindTexture(GL_TEXTURE_2D, g_depth_clut_tex);
                        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
                        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
                        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
                        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 512, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, clut_rgba);
                    } else {
                        glActiveTexture(GL_TEXTURE2);
                        glBindTexture(GL_TEXTURE_2D, g_depth_clut_tex);
                        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 512, 1, GL_RGBA, GL_UNSIGNED_BYTE, clut_rgba);
                    }
                    glActiveTexture(GL_TEXTURE1);
                    glBindTexture(GL_TEXTURE_2D, g_last_z_tex);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
                    glActiveTexture(GL_TEXTURE0);
                    float sx = (st->tex_w > 0) ? ((float)st->tex_w / 480.0f) : 1.0f;
                    float sy = (st->tex_h > 0) ? ((float)st->tex_h / 272.0f) : 1.0f;
                    glUniform2f(g_ge_u_fbo_uv_scale, sx, sy);
                    glUniform2f(g_ge_u_fbo_uv_offset, 0.0f, 0.0f);
                    glUniform1i(g_ge_u_depth_clut_mode, 1);
                    glUniform1i(g_ge_u_depth_tex, 1);
                    glUniform1i(g_ge_u_clut_tex, 2);
                    glUniform1i(g_ge_u_clut_shift, st->clut_shift);
                    glUniform1i(g_ge_u_clut_mask, st->clut_mask);
                    glUniform1i(g_ge_u_clut_start, st->clut_start);
                } else if (fbo_src && fbo_src->has_ge_draws) {
                    glActiveTexture(GL_TEXTURE0);
                    if (fbo_src == cur_fbo) {
                        if (!g_fbo_scratch_tex) {
                            glGenTextures(1, &g_fbo_scratch_tex);
                            glBindTexture(GL_TEXTURE_2D, g_fbo_scratch_tex);
                            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 480, 272, 0,
                                         GL_RGBA, GL_UNSIGNED_BYTE, NULL);
                        } else {
                            glBindTexture(GL_TEXTURE_2D, g_fbo_scratch_tex);
                        }
                        glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, 480, 272);
                    } else {
                        glBindTexture(GL_TEXTURE_2D, fbo_src->tex);
                    }
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
                    float sx = (st->tex_w > 0) ? ((float)st->tex_w / 480.0f) : 1.0f;
                    float sy = (st->tex_h > 0) ? ((float)st->tex_h / 272.0f) : 1.0f;
                    glUniform2f(g_ge_u_fbo_uv_scale, sx, sy);
                    glUniform2f(g_ge_u_fbo_uv_offset, uv_off_x, uv_off_y);
                    glUniform1i(g_ge_u_depth_clut_mode, 0);
                } else {
                    glActiveTexture(GL_TEXTURE0);
                    GLuint tex = get_cached_texture(cmd, clut_ptr);
                    glBindTexture(GL_TEXTURE_2D, tex);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,
                                    st->tex_clamp_u ? GL_CLAMP_TO_EDGE : GL_REPEAT);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,
                                    st->tex_clamp_v ? GL_CLAMP_TO_EDGE : GL_REPEAT);
                    glUniform2f(g_ge_u_fbo_uv_scale, 1.0f, 1.0f);
                    glUniform2f(g_ge_u_fbo_uv_offset, 0.0f, 0.0f);
                    glUniform1i(g_ge_u_depth_clut_mode, 0);
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
                glUniform1i(g_ge_u_depth_clut_mode, 0);
            }
        }

        if (((st->clear && st->clear_depth) || (!st->clear && st->ztest && st->zwrite)) && st->zaddr != 0u) {
            g_last_z_off = st->zaddr & 0x001FFFFFu;
            g_last_z_tex = cur_fbo->depth_tex;
        }
        cur_fbo->has_ge_draws = 1;
        glDrawArrays(cmd->gl_prim, (GLint)cmd->first_vert, (GLsizei)cmd->vert_count);
    }

    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_STENCIL_TEST);
    glDisable(GL_BLEND);
    glDisable(GL_CULL_FACE);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glDepthMask(GL_TRUE);
    glDisableVertexAttribArray((GLuint)g_ge_loc_pos);
    glDisableVertexAttribArray((GLuint)g_ge_loc_col);
    glDisableVertexAttribArray((GLuint)g_ge_loc_uv);
    if (g_ge_loc_fog >= 0) glDisableVertexAttribArray((GLuint)g_ge_loc_fog);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

static GLint  g_hud_loc_pos = -1;
static GLint  g_hud_loc_col = -1;
static GLint  g_hud_loc_uv  = -1;
static GLint  g_hud_u_tex   = -1;
static int    g_hud_enabled = 1;

typedef struct {
    float x, y;
    float r, g, b, a;
    float u, v;
} hud_vert;

static const uint8_t HUD_FONT_5X7[95][5] = {
    {0x00,0x00,0x00,0x00,0x00}, /* ' ' */
    {0x00,0x00,0x5F,0x00,0x00}, /* '!' */
    {0x00,0x07,0x00,0x07,0x00}, /* '"' */
    {0x14,0x7F,0x14,0x7F,0x14}, /* '#' */
    {0x24,0x2A,0x7F,0x2A,0x12}, /* '$' */
    {0x23,0x13,0x08,0x64,0x62}, /* '%' */
    {0x36,0x49,0x55,0x22,0x50}, /* '&' */
    {0x00,0x05,0x03,0x00,0x00}, /* ''' */
    {0x00,0x1C,0x22,0x41,0x00}, /* '(' */
    {0x00,0x41,0x22,0x1C,0x00}, /* ')' */
    {0x14,0x08,0x3E,0x08,0x14}, /* '*' */
    {0x08,0x08,0x3E,0x08,0x08}, /* '+' */
    {0x00,0x50,0x30,0x00,0x00}, /* ',' */
    {0x08,0x08,0x08,0x08,0x08}, /* '-' */
    {0x00,0x60,0x60,0x00,0x00}, /* '.' */
    {0x20,0x10,0x08,0x04,0x02}, /* '/' */
    {0x3E,0x51,0x49,0x45,0x3E}, /* '0' */
    {0x00,0x42,0x7F,0x40,0x00}, /* '1' */
    {0x42,0x61,0x51,0x49,0x46}, /* '2' */
    {0x21,0x41,0x45,0x4B,0x31}, /* '3' */
    {0x18,0x14,0x12,0x7F,0x10}, /* '4' */
    {0x27,0x45,0x45,0x45,0x39}, /* '5' */
    {0x3C,0x4A,0x49,0x49,0x30}, /* '6' */
    {0x01,0x71,0x09,0x05,0x03}, /* '7' */
    {0x36,0x49,0x49,0x49,0x36}, /* '8' */
    {0x06,0x49,0x49,0x29,0x1E}, /* '9' */
    {0x00,0x36,0x36,0x00,0x00}, /* ':' */
    {0x00,0x56,0x36,0x00,0x00}, /* ';' */
    {0x08,0x14,0x22,0x41,0x00}, /* '<' */
    {0x14,0x14,0x14,0x14,0x14}, /* '=' */
    {0x00,0x41,0x22,0x14,0x08}, /* '>' */
    {0x02,0x01,0x51,0x09,0x06}, /* '?' */
    {0x32,0x49,0x79,0x41,0x3E}, /* '@' */
    {0x7E,0x11,0x11,0x11,0x7E}, /* 'A' */
    {0x7F,0x49,0x49,0x49,0x36}, /* 'B' */
    {0x3E,0x41,0x41,0x41,0x22}, /* 'C' */
    {0x7F,0x41,0x41,0x22,0x1C}, /* 'D' */
    {0x7F,0x49,0x49,0x49,0x41}, /* 'E' */
    {0x7F,0x09,0x09,0x09,0x01}, /* 'F' */
    {0x3E,0x41,0x49,0x49,0x7A}, /* 'G' */
    {0x7F,0x08,0x08,0x08,0x7F}, /* 'H' */
    {0x00,0x41,0x7F,0x41,0x00}, /* 'I' */
    {0x20,0x40,0x41,0x3F,0x01}, /* 'J' */
    {0x7F,0x08,0x14,0x22,0x41}, /* 'K' */
    {0x7F,0x40,0x40,0x40,0x40}, /* 'L' */
    {0x7F,0x02,0x0C,0x02,0x7F}, /* 'M' */
    {0x7F,0x04,0x08,0x10,0x7F}, /* 'N' */
    {0x3E,0x41,0x41,0x41,0x3E}, /* 'O' */
    {0x7F,0x09,0x09,0x09,0x06}, /* 'P' */
    {0x3E,0x41,0x51,0x21,0x5E}, /* 'Q' */
    {0x7F,0x09,0x19,0x29,0x46}, /* 'R' */
    {0x46,0x49,0x49,0x49,0x31}, /* 'S' */
    {0x01,0x01,0x7F,0x01,0x01}, /* 'T' */
    {0x3F,0x40,0x40,0x40,0x3F}, /* 'U' */
    {0x1F,0x20,0x40,0x20,0x1F}, /* 'V' */
    {0x3F,0x40,0x38,0x40,0x3F}, /* 'W' */
    {0x63,0x14,0x08,0x14,0x63}, /* 'X' */
    {0x07,0x08,0x70,0x08,0x07}, /* 'Y' */
    {0x61,0x51,0x49,0x45,0x43}, /* 'Z' */
    {0x00,0x7F,0x41,0x41,0x00}, /* '[' */
    {0x02,0x04,0x08,0x10,0x20}, /* '\' */
    {0x00,0x41,0x41,0x7F,0x00}, /* ']' */
    {0x04,0x02,0x01,0x02,0x04}, /* '^' */
    {0x40,0x40,0x40,0x40,0x40}, /* '_' */
    {0x00,0x01,0x02,0x04,0x00}, /* '`' */
    {0x20,0x54,0x54,0x54,0x78}, /* 'a' */
    {0x7F,0x48,0x44,0x44,0x38}, /* 'b' */
    {0x38,0x44,0x44,0x44,0x20}, /* 'c' */
    {0x38,0x44,0x44,0x48,0x7F}, /* 'd' */
    {0x38,0x54,0x54,0x54,0x18}, /* 'e' */
    {0x08,0x7E,0x09,0x01,0x02}, /* 'f' */
    {0x0C,0x52,0x52,0x52,0x3E}, /* 'g' */
    {0x7F,0x08,0x04,0x04,0x78}, /* 'h' */
    {0x00,0x44,0x7D,0x40,0x00}, /* 'i' */
    {0x20,0x40,0x44,0x3D,0x00}, /* 'j' */
    {0x7F,0x10,0x28,0x44,0x00}, /* 'k' */
    {0x00,0x41,0x7F,0x40,0x00}, /* 'l' */
    {0x7C,0x04,0x18,0x04,0x78}, /* 'm' */
    {0x7C,0x08,0x04,0x04,0x78}, /* 'n' */
    {0x38,0x44,0x44,0x44,0x38}, /* 'o' */
    {0x7C,0x14,0x14,0x14,0x08}, /* 'p' */
    {0x08,0x14,0x14,0x18,0x7C}, /* 'q' */
    {0x7C,0x08,0x04,0x04,0x08}, /* 'r' */
    {0x48,0x54,0x54,0x54,0x20}, /* 's' */
    {0x04,0x3F,0x44,0x40,0x20}, /* 't' */
    {0x3C,0x40,0x40,0x20,0x7C}, /* 'u' */
    {0x1C,0x20,0x40,0x20,0x1C}, /* 'v' */
    {0x3C,0x40,0x30,0x40,0x3C}, /* 'w' */
    {0x44,0x28,0x10,0x28,0x44}, /* 'x' */
    {0x0C,0x50,0x50,0x50,0x3C}, /* 'y' */
    {0x44,0x64,0x54,0x4C,0x44}, /* 'z' */
    {0x00,0x08,0x36,0x41,0x00}, /* '{' */
    {0x00,0x00,0x7F,0x00,0x00}, /* '|' */
    {0x00,0x41,0x36,0x08,0x00}, /* '}' */
    {0x10,0x08,0x08,0x10,0x08}  /* '~' */
};

static const char *hud_vs_src =
    "#version 300 es\n"
    "in vec2 a_pos;\n"
    "in vec4 a_col;\n"
    "in vec2 a_uv;\n"
    "out vec4 v_col;\n"
    "out vec2 v_uv;\n"
    "void main() {\n"
    "    gl_Position = vec4(a_pos, 0.0, 1.0);\n"
    "    v_col = a_col;\n"
    "    v_uv = a_uv;\n"
    "}\n";

static const char *hud_fs_src =
    "#version 300 es\n"
    "precision mediump float;\n"
    "in vec4 v_col;\n"
    "in vec2 v_uv;\n"
    "uniform sampler2D u_tex;\n"
    "out vec4 frag_color;\n"
    "void main() {\n"
    "    if (v_uv.x < 0.0) frag_color = v_col;\n"
    "    else frag_color = vec4(v_col.rgb, v_col.a * texture(u_tex, v_uv).a);\n"
    "}\n";

static void init_hud(void) {
    const char *e = getenv("MC3_HUD");
    g_hud_enabled = (e && strcmp(e, "0") == 0) ? 0 : 1;

    g_hud_prog = link_program(hud_vs_src, hud_fs_src);
    if (!g_hud_prog) return;
    g_hud_loc_pos = glGetAttribLocation(g_hud_prog, "a_pos");
    g_hud_loc_col = glGetAttribLocation(g_hud_prog, "a_col");
    g_hud_loc_uv  = glGetAttribLocation(g_hud_prog, "a_uv");
    g_hud_u_tex   = glGetUniformLocation(g_hud_prog, "u_tex");
    glGenBuffers(1, &g_hud_vbo);

    static uint32_t font_rgba[1024 * 8];
    memset(font_rgba, 0, sizeof(font_rgba));
    for (int g = 0; g < 95; g++) {
        for (int col = 0; col < 5; col++) {
            uint8_t bits = HUD_FONT_5X7[g][col];
            int gx = g * 6 + col;
            for (int row = 0; row < 7; row++) {
                if (bits & (1u << row)) {
                    font_rgba[row * 1024 + gx] = 0xFFFFFFFFu;
                }
            }
        }
    }
    glGenTextures(1, &g_hud_font_tex);
    glBindTexture(GL_TEXTURE_2D, g_hud_font_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1024, 8, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, font_rgba);
}

#define MAX_HUD_VERTS 8192

static void hud_push_quad(hud_vert *buf, int *cnt,
                          float x0, float y0, float w, float h,
                          float u0, float v0, float u1, float v1,
                          float r, float g, float b, float a) {
    if (*cnt + 6 > MAX_HUD_VERTS) return;
    float nx0 = (2.0f * x0 / (float)g_window_w) - 1.0f;
    float nx1 = (2.0f * (x0 + w) / (float)g_window_w) - 1.0f;
    float ny0 = 1.0f - (2.0f * y0 / (float)g_window_h);
    float ny1 = 1.0f - (2.0f * (y0 + h) / (float)g_window_h);
    hud_vert v[6] = {
        {nx0, ny0, r, g, b, a, u0, v0},
        {nx1, ny0, r, g, b, a, u1, v0},
        {nx1, ny1, r, g, b, a, u1, v1},
        {nx0, ny0, r, g, b, a, u0, v0},
        {nx1, ny1, r, g, b, a, u1, v1},
        {nx0, ny1, r, g, b, a, u0, v1},
    };
    memcpy(&buf[*cnt], v, sizeof(v));
    *cnt += 6;
}

static void hud_push_rect(hud_vert *buf, int *cnt, float x, float y, float w, float h,
                          float r, float g, float b, float a) {
    hud_push_quad(buf, cnt, x, y, w, h, -1.0f, 0.0f, -1.0f, 0.0f, r, g, b, a);
}

static void hud_push_text(hud_vert *buf, int *cnt, float x, float y, float scale,
                          float r, float g, float b, float a, const char *s) {
    float cw = 5.0f * scale;
    float ch = 7.0f * scale;
    float adv = 6.0f * scale;
    for (; *s; s++, x += adv) {
        int c = (unsigned char)*s;
        if (c < 32 || c > 126) c = '?';
        int idx = c - 32;
        if (idx == 0) continue;
        float u0 = (float)(idx * 6) / 1024.0f;
        float u1 = (float)(idx * 6 + 5) / 1024.0f;
        hud_push_quad(buf, cnt, x, y, cw, ch, u0, 0.0f, u1, 7.0f / 8.0f, r, g, b, a);
    }
}

static void render_hud(void) {
    if (!g_hud_enabled || !g_hud_prog || !g_hud_font_tex || g_window_w <= 0 || g_window_h <= 0)
        return;

    mc3_tel_snapshot snap;
    mc3_telemetry_get_snapshot(&snap);

    static hud_vert verts[MAX_HUD_VERTS];
    int vcnt = 0;

    float sc = (g_window_h >= 720) ? 2.0f : 1.0f;
    float px = 8.0f * sc;
    float py = 8.0f * sc;
    float lh = 10.0f * sc;
    float pw = 356.0f * sc;
    float gh = 34.0f * sc;
    float ph = 6.0f * sc + 5.0f * lh + gh + 10.0f * sc;

    /* Background + border */
    hud_push_rect(verts, &vcnt, px, py, pw, ph, 0.04f, 0.06f, 0.09f, 0.78f);
    hud_push_rect(verts, &vcnt, px, py, pw, 1.0f * sc, 0.25f, 0.65f, 0.90f, 0.80f);

    float tx = px + 6.0f * sc;
    float ty = py + 5.0f * sc;
    char line[128];

    /* Line 1: FPS & Frame Timing */
    float fr = 0.35f, fg = 0.95f, fb = 0.45f;
    if (snap.ft_avg_ms > 45.0) { fr = 1.0f; fg = 0.30f; fb = 0.30f; }
    else if (snap.ft_avg_ms > 22.0) { fr = 1.0f; fg = 0.85f; fb = 0.25f; }
    snprintf(line, sizeof(line), "FPS G:%4.1f H:%4.1f  FT:%5.1fms (P95:%4.1f MAX:%5.1f)",
             snap.guest_fps, snap.host_fps, snap.ft_avg_ms, snap.ft_p95_ms, snap.ft_max_ms);
    hud_push_text(verts, &vcnt, tx, ty, sc, fr, fg, fb, 1.0f, line);
    ty += lh;

    /* Line 2: Timing breakdown */
    snprintf(line, sizeof(line), "TIME GE:%4.1fms GPU:%4.1fms TEX:%4.1fms WAIT:%4.1fms",
             snap.ge_cpu_ms, snap.gpu_exec_ms, snap.tex_up_ms, snap.pace_wait_ms);
    hud_push_text(verts, &vcnt, tx, ty, sc, 0.85f, 0.92f, 1.0f, 1.0f, line);
    ty += lh;

    /* Line 3: Draw calls, vertices, BBOX */
    snprintf(line, sizeof(line), "DRAW %4u cmds  %6.1fk verts  BBOX:%llu/%llu",
             snap.last_draws, (double)snap.last_verts / 1000.0,
             (unsigned long long)snap.bbox_culled_sec,
             (unsigned long long)snap.bbox_tested_sec);
    hud_push_text(verts, &vcnt, tx, ty, sc, 0.92f, 0.92f, 0.92f, 1.0f, line);
    ty += lh;

    /* Line 4: Texture cache */
    uint32_t tex_tot = snap.last_tex_hits + snap.last_tex_misses;
    double hit_pct = tex_tot ? (100.0 * (double)snap.last_tex_hits / (double)tex_tot) : 100.0;
    snprintf(line, sizeof(line), "TEX  HIT:%5.1f%%  MISS:%3u/f  SLOTS:%u/%u",
             hit_pct, snap.last_tex_misses, snap.tex_slots_used, snap.tex_slots_cap);
    hud_push_text(verts, &vcnt, tx, ty, sc, 0.78f, 0.90f, 1.0f, 1.0f, line);
    ty += lh;

    /* Line 5: Audio health */
    float ar = 0.45f, ag = 0.95f, ab = 0.55f;
    if (snap.aud_underruns_recent > 0) { ar = 1.0f; ag = 0.30f; ab = 0.30f; }
    else if (snap.aud_buf_min_ms < 25.0 && snap.aud_active_chs > 0) { ar = 1.0f; ag = 0.85f; ab = 0.25f; }
    snprintf(line, sizeof(line), "AUD  CH:%d BUF:%3.0fms UR:%llu OV:%llu GAP:%3.0fms  [F3 HUD]",
             snap.aud_active_chs, snap.aud_buf_ms,
             (unsigned long long)snap.aud_underruns_total,
             (unsigned long long)snap.aud_overruns_total,
             snap.aud_max_gap_ms);
    hud_push_text(verts, &vcnt, tx, ty, sc, ar, ag, ab, 1.0f, line);
    ty += lh + 2.0f * sc;

    /* Sparkline graph (120 bars, capped at 80ms) */
    float gx = tx;
    float gw = pw - 12.0f * sc;
    hud_push_rect(verts, &vcnt, gx, ty, gw, gh, 0.02f, 0.03f, 0.05f, 0.85f);
    float y_16 = ty + gh * (1.0f - 16.667f / 80.0f);
    float y_33 = ty + gh * (1.0f - 33.333f / 80.0f);
    hud_push_rect(verts, &vcnt, gx, y_16, gw, 1.0f, 0.30f, 0.70f, 0.40f, 0.45f);
    hud_push_rect(verts, &vcnt, gx, y_33, gw, 1.0f, 0.85f, 0.75f, 0.25f, 0.45f);

    float bw = gw / (float)MC3_TEL_HISTORY;
    for (int i = 0; i < MC3_TEL_HISTORY; i++) {
        int idx = (snap.ft_hist_head + i) % MC3_TEL_HISTORY;
        float ms = snap.ft_hist_ms[idx];
        if (ms <= 0.0f) continue;
        float frac = ms / 80.0f;
        if (frac > 1.0f) frac = 1.0f;
        float bh = frac * gh;
        if (bh < 1.0f) bh = 1.0f;
        float br = 0.25f, bg = 0.90f, bb = 0.40f;
        if (ms > 50.0f) { br = 0.95f; bg = 0.25f; bb = 0.25f; }
        else if (ms > 33.5f) { br = 0.95f; bg = 0.80f; bb = 0.20f; }
        hud_push_rect(verts, &vcnt, gx + (float)i * bw, ty + gh - bh,
                      bw > 1.2f ? (bw - 0.3f) : bw, bh, br, bg, bb, 0.90f);
    }

    if (vcnt == 0) return;

    glViewport(0, 0, g_window_w, g_window_h);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_CULL_FACE);
    glDisable(GL_STENCIL_TEST);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glUseProgram(g_hud_prog);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, g_hud_font_tex);
    glUniform1i(g_hud_u_tex, 0);

    glBindBuffer(GL_ARRAY_BUFFER, g_hud_vbo);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)((size_t)vcnt * sizeof(hud_vert)), verts, GL_STREAM_DRAW);
    glEnableVertexAttribArray((GLuint)g_hud_loc_pos);
    glEnableVertexAttribArray((GLuint)g_hud_loc_col);
    glEnableVertexAttribArray((GLuint)g_hud_loc_uv);
    glVertexAttribPointer((GLuint)g_hud_loc_pos, 2, GL_FLOAT, GL_FALSE, sizeof(hud_vert), (void *)0);
    glVertexAttribPointer((GLuint)g_hud_loc_col, 4, GL_FLOAT, GL_FALSE, sizeof(hud_vert), (void *)(2 * sizeof(float)));
    glVertexAttribPointer((GLuint)g_hud_loc_uv, 2, GL_FLOAT, GL_FALSE, sizeof(hud_vert), (void *)(6 * sizeof(float)));

    glDrawArrays(GL_TRIANGLES, 0, vcnt);

    glDisableVertexAttribArray((GLuint)g_hud_loc_pos);
    glDisableVertexAttribArray((GLuint)g_hud_loc_col);
    glDisableVertexAttribArray((GLuint)g_hud_loc_uv);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDisable(GL_BLEND);
    glUseProgram(0);
}

void gles32_present_loop(void) {
    glDisable(GL_CULL_FACE);
    if (!g_window || !g_glcontext) return;

    int has_vsync = (SDL_GL_GetSwapInterval() != 0);
    int use_fbo_tex = 0;
    GLuint cur_fbo_tex = 0;

    fprintf(stderr, "gles32: present loop started (vsync=%d, GPU raster)\n", has_vsync);
    fflush(stderr);
    uint32_t kbd_buttons = 0;
    int kbd_steer_left = 0, kbd_steer_right = 0;

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
                case SDLK_UP:     mask = 0x0010; break; /* D-Pad up (menus) */
                case SDLK_DOWN:   mask = 0x0040; break; /* D-Pad down (menus) */
                case SDLK_RIGHT:  mask = 0x0020; kbd_steer_right = (ev.type == SDL_KEYDOWN); break;
                case SDLK_LEFT:   mask = 0x0080; kbd_steer_left  = (ev.type == SDL_KEYDOWN); break;
                case SDLK_q:      mask = 0x0100; break; /* L trigger */
                case SDLK_e:      mask = 0x0200; break; /* R trigger */
                case SDLK_t:      mask = 0x1000; break; /* Triangle */
                case SDLK_c:
                case SDLK_BACKSPACE: mask = 0x2000; break; /* Circle */
                case SDLK_w:
                case SDLK_x:
                case SDLK_z:      mask = 0x4000; break; /* Cross: accelerate / OK */
                case SDLK_s:      mask = 0x8000; break; /* Square: brake / reverse */
                case SDLK_a:      kbd_steer_left  = (ev.type == SDL_KEYDOWN); break;
                case SDLK_d:      kbd_steer_right = (ev.type == SDL_KEYDOWN); break;
                case SDLK_RETURN: mask = 0x0008; break; /* Start */
                case SDLK_SPACE:  mask = 0x0001; break; /* Select */
                case SDLK_F3:
                    if (ev.type == SDL_KEYDOWN) g_hud_enabled = !g_hud_enabled;
                    break;
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
        if (kbd_steer_left && !kbd_steer_right) lx = 0;
        else if (kbd_steer_right && !kbd_steer_left) lx = 255;
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

        uint64_t t_host0 = gles_now_us();
        static uint64_t prev_host_us = 0;
        double host_ft_ms = prev_host_us ? ((double)(t_host0 - prev_host_us) / 1000.0) : 16.667;
        prev_host_us = t_host0;
        double gpu_exec_ms = 0.0, tex_up_ms = 0.0;
        uint32_t frame_tex_hits = 0, frame_tex_misses = 0;

        if (dirty) {
            if (has_cmds) {
                if (g_exec.cmd_count > 0) {
                    g_exec_tex_hits = 0;
                    g_exec_tex_misses = 0;
                    g_exec_tex_up_us = 0;
                    uint64_t t_gpu0 = gles_now_us();
                    execute_ge_frame(&g_exec);
                    gpu_exec_ms = (double)(gles_now_us() - t_gpu0) / 1000.0;
                    tex_up_ms = (double)g_exec_tex_up_us / 1000.0;
                    frame_tex_hits = g_exec_tex_hits;
                    frame_tex_misses = g_exec_tex_misses;
                }
                uint32_t disp_off = g_exec.disp_addr ? (g_exec.disp_addr & 0x001FFFFFu) : 0u;
                gles_fbo_slot *disp_slot = get_fbo_for_target(disp_off, 0);
                if (disp_slot && disp_slot->has_ge_draws) cur_fbo_tex = disp_slot->tex;
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

        int dump_hud = (getenv("MC3_DUMP_HUD") != NULL);
        if (dump_hud) render_hud();

        if (dirty) {
            const char *ppm_dir = getenv("MC3_DUMP_PPM");
            if (ppm_dir) {
                static uint32_t s_ppm_frame = 0;
                if ((s_ppm_frame % 30u) == 0u && g_window_w > 0 && g_window_h > 0) {
                    uint8_t *rgba = (uint8_t *)malloc((size_t)g_window_w * (size_t)g_window_h * 4u);
                    if (rgba) {
                        glReadPixels(0, 0, g_window_w, g_window_h, GL_RGBA, GL_UNSIGNED_BYTE, rgba);
                        char path[512];
                        snprintf(path, sizeof(path), "%s/frame_%04u.ppm", ppm_dir, s_ppm_frame);
                        FILE *fp = fopen(path, "wb");
                        if (fp) {
                            fprintf(fp, "P6\n%d %d\n255\n", g_window_w, g_window_h);
                            for (int y = g_window_h - 1; y >= 0; y--) {
                                const uint8_t *row = rgba + (size_t)y * (size_t)g_window_w * 4u;
                                for (int x = 0; x < g_window_w; x++) {
                                    fwrite(row + x * 4, 1, 3, fp);
                                }
                            }
                            fclose(fp);
                        }
                        free(rgba);
                    }
                }
                s_ppm_frame++;
            }
        }

        if (!dump_hud) render_hud();

        uint64_t t_sw0 = gles_now_us();
        SDL_GL_SwapWindow(g_window);
        double swap_ms = (double)(gles_now_us() - t_sw0) / 1000.0;
        mc3_telemetry_on_gpu_frame(host_ft_ms, gpu_exec_ms, tex_up_ms, swap_ms,
                                   frame_tex_hits, frame_tex_misses,
                                   g_tex_slots_used, TEX_CACHE_SLOTS);
        if (!has_vsync) usleep(16667);
        else if (!dirty && swap_ms < 2.0) usleep(1000);
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

void gles32_present_loop(void) {
    glDisable(GL_CULL_FACE); }

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
