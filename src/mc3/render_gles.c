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

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
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
#if defined(__linux__) && defined(__aarch64__)
#include <sched.h>
#endif

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
static SDL_Joystick *g_joystick = NULL;

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

#ifndef GL_TEXTURE_MAX_ANISOTROPY_EXT
#define GL_TEXTURE_MAX_ANISOTROPY_EXT     0x84FE
#endif
#ifndef GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT
#define GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT 0x84FF
#endif

static int g_fbo_w = 480;
static int g_fbo_h = 272;
static float g_max_aniso = 1.0f;

static gles_fbo_slot g_fbos[FBO_SLOTS];
static GLuint g_last_fbo_tex = 0;
static GLuint g_fbo_scratch_tex = 0;
static GLuint g_depth_clut_tex = 0;

/* ---- GE draw shader + dynamic VBO --------------------------------------- */

#define GE_MT_MAX 14

typedef struct {
    GLuint prog;
    GLint  loc_pos, loc_col, loc_uv, loc_fog, loc_nrm;
    GLint  u_tex[GE_MT_MAX], u_tex_enable, u_tex_func, u_tex_alpha, u_tex_double, u_tex_env;
    GLint  u_atest, u_atest_func, u_atest_ref, u_atest_mask, u_clear;
    GLint  u_fbo_uv_scale, u_fbo_uv_offset;
    GLint  u_depth_clut_mode, u_depth_tex, u_clut_tex, u_clut_shift, u_clut_mask, u_clut_start;
    GLint  u_fog_enable, u_fog_far, u_fog_dist, u_fog_color;
    GLint  u_raw_mode, u_mvp, u_wv_z, u_uv_xform, u_pos_scale, u_nrm_scale;
    GLint  u_tex_map_mode, u_tex_proj_mode, u_world, u_world_t, u_tmat, u_env_lu, u_env_lv;
    GLint  u_cmat, u_rnorm, u_spow;
    GLint  u_mat_emis, u_mat_amb, u_mat_diff, u_mat_spec, u_lt_global_amb;
    GLint  u_lt_on, u_lt_type, u_lt_comp, u_lt_pos, u_lt_dir, u_lt_att;
    GLint  u_lt_spot_exp, u_lt_spot_cut, u_lt_amb, u_lt_diff, u_lt_spec;
} ge_prog_locs;

static ge_prog_locs g_ge_unlit;
static ge_prog_locs g_ge_lit;
static GLuint g_ge_prog = 0;
static GLuint g_ge_lit_prog = 0;
#define GE_VBO_RING 3
static GLuint g_ge_vbo[GE_VBO_RING] = {0};
static int    g_ge_vbo_idx = 0;
static GLuint g_ge_raw_vbo = 0;
static GLuint g_ge_raw_cbo = 0;
static GLuint g_ge_raw_ibo = 0;

/* ---- GPU vertex format -------------------------------------------------- */

typedef struct {
    float    x, y, z;
    uint32_t rgba;
    float    u, v, q;
    float    fog;
} gles_vert;

typedef struct {
    uint32_t tex_addr;
    int      tex_w, tex_h, tex_bw, tex_fmt, tex_swizzle;
    int      tex_clamp_u, tex_clamp_v;
    uint32_t clut_offset, clut_len, clut_hash, data_hash;
    int      clut_fmt, clut_shift, clut_mask, clut_start;
} gles_mt_slot;

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
    uint16_t       mt_start;    /* index into fl->mt_slots when mt_count > 1 */
    uint8_t        mt_count;    /* 0 = single-tex (use st), 2..8 = multi-tex */
#ifdef PSP_RENDER_HAS_DRAW_RAW
    int            is_raw;
    uint32_t       raw_voff;
    uint32_t       raw_ioff;
    uint32_t       raw_col_off;
    int            raw_stride;
    uint32_t       raw_vtype;
    int            raw_isz;
    uint32_t       xform_idx;
#endif
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

    gles_mt_slot *mt_slots;
    uint32_t      mt_count;
    uint32_t      mt_cap;
#ifdef PSP_RENDER_HAS_DRAW_RAW
    uint8_t         *raw_vbuf;
    uint32_t         raw_vbytes;
    uint32_t         raw_vcap;
    uint8_t         *raw_cbuf;   /* unpacked 16-bit colours mirrored at vertex byte offsets */
    uint32_t         raw_cbytes;
    uint32_t         raw_ccap;
    uint8_t         *raw_ibuf;
    uint32_t         raw_ibytes;
    uint32_t         raw_icap;
    psp_xform_state *xforms;
    uint32_t         xform_count;
    uint32_t         xform_cap;
#endif

    uint32_t   disp_addr;

    uint32_t  *fmv_buf;
    uint32_t   fmv_tex_addr;
    int        fmv_valid;
} gles_frame_list;

static GLuint g_fmv_tex = 0;

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
    int8_t   wrap_s, wrap_t;
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
    "in vec3 a_nrm;\n"
    "uniform int u_raw_mode;\n"
    "uniform highp mat4 u_mvp;\n"
    "uniform highp vec4 u_wv_z;\n"
    "uniform highp vec4 u_uv_xform;\n"
    "uniform highp float u_pos_scale;\n"
    "uniform highp float u_nrm_scale;\n"
    "uniform mediump int u_fog_enable;\n"
    "uniform highp float u_fog_far;\n"
    "uniform highp float u_fog_dist;\n"
    "uniform int u_tex_map_mode;\n"
    "uniform int u_tex_proj_mode;\n"
    "uniform highp mat3 u_world;\n"
    "uniform highp vec4 u_tmat[3];\n"
    "uniform highp vec3 u_env_lu;\n"
    "uniform highp vec3 u_env_lv;\n"
    "out vec4 v_col;\n"
    "out vec3 v_uvq;\n"
    "out float v_fog;\n"
    "flat out float v_tex_slot;\n"
    "void main() {\n"
    "    if (u_raw_mode == 0) {\n"
    "        gl_Position = vec4(a_pos, 1.0);\n"
    "        v_col = a_col;\n"
    "        v_uvq = a_uv;\n"
    "        float slot = floor((a_fog + 0.5) * 0.25);\n"
    "        v_tex_slot = slot;\n"
    "        v_fog = a_fog - slot * 4.0;\n"
    "        return;\n"
    "    }\n"
    "    v_tex_slot = 0.0;\n"
    "    gl_Position = u_mvp * vec4(a_pos, 1.0);\n"
    "    v_fog = (u_fog_enable != 0 && u_fog_dist != 0.0)\n"
    "          ? clamp((dot(u_wv_z, vec4(a_pos, 1.0)) + u_fog_far) * u_fog_dist, 0.0, 1.0)\n"
    "          : 1.0;\n"
    "    vec2 uv;\n"
    "    if (u_tex_map_mode == 2) {\n"
    "        vec3 wn = u_world * (a_nrm * u_nrm_scale);\n"
    "        float wl = length(wn);\n"
    "        if (wl > 0.0) wn /= wl;\n"
    "        uv = vec2(0.5 * (1.0 + dot(wn, u_env_lu)), 0.5 * (1.0 + dot(wn, u_env_lv)));\n"
    "    } else if (u_tex_map_mode == 1) {\n"
    "        vec3 ti;\n"
    "        if (u_tex_proj_mode == 0) ti = a_pos * u_pos_scale;\n"
    "        else if (u_tex_proj_mode == 1) ti = vec3(a_uv.xy * u_uv_xform.xz + u_uv_xform.yw, 1.0);\n"
    "        else if (u_tex_proj_mode == 2) { vec3 n = a_nrm * u_nrm_scale; float nl = length(n); ti = (nl > 0.0) ? (n / nl) : n; }\n"
    "        else ti = a_nrm * u_nrm_scale;\n"
    "        vec4 ti4 = vec4(ti, 1.0);\n"
    "        uv = vec2(dot(u_tmat[0], ti4), dot(u_tmat[1], ti4));\n"
    "    } else {\n"
    "        uv = a_uv.xy * u_uv_xform.xz + u_uv_xform.yw;\n"
    "    }\n"
    "    v_uvq = vec3(uv, 1.0);\n"
    "    v_col = a_col;\n"
    "}\n";

static const char *ge_lit_vs_src =
    "#version 300 es\n"
    "in vec3 a_pos;\n"
    "in vec4 a_col;\n"
    "in vec3 a_uv;\n"
    "in vec3 a_nrm;\n"
    "uniform highp mat4 u_mvp;\n"
    "uniform highp vec4 u_wv_z;\n"
    "uniform highp vec4 u_uv_xform;\n"
    "uniform highp float u_pos_scale;\n"
    "uniform highp float u_nrm_scale;\n"
    "uniform mediump int u_fog_enable;\n"
    "uniform highp float u_fog_far;\n"
    "uniform highp float u_fog_dist;\n"
    "uniform int u_tex_map_mode;\n"
    "uniform int u_tex_proj_mode;\n"
    "uniform highp mat3 u_world;\n"
    "uniform highp vec3 u_world_t;\n"
    "uniform highp vec4 u_tmat[3];\n"
    "uniform highp vec3 u_env_lu;\n"
    "uniform highp vec3 u_env_lv;\n"
    "uniform int u_cmat;\n"
    "uniform int u_rnorm;\n"
    "uniform highp float u_spow;\n"
    "uniform vec4 u_mat_emis;\n"
    "uniform vec4 u_mat_amb;\n"
    "uniform vec4 u_mat_diff;\n"
    "uniform vec4 u_mat_spec;\n"
    "uniform vec3 u_lt_global_amb;\n"
    "uniform ivec4 u_lt_on;\n"
    "uniform ivec4 u_lt_type;\n"
    "uniform ivec4 u_lt_comp;\n"
    "uniform highp vec3 u_lt_pos[4];\n"
    "uniform highp vec3 u_lt_dir[4];\n"
    "uniform highp vec3 u_lt_att[4];\n"
    "uniform highp vec4 u_lt_spot_exp;\n"
    "uniform highp vec4 u_lt_spot_cut;\n"
    "uniform vec3 u_lt_amb[4];\n"
    "uniform vec3 u_lt_diff[4];\n"
    "uniform vec3 u_lt_spec[4];\n"
    "out vec4 v_col;\n"
    "out vec3 v_uvq;\n"
    "out float v_fog;\n"
    "flat out float v_tex_slot;\n"
    "void main() {\n"
    "    v_tex_slot = 0.0;\n"
    "    gl_Position = u_mvp * vec4(a_pos, 1.0);\n"
    "    v_fog = (u_fog_enable != 0 && u_fog_dist != 0.0)\n"
    "          ? clamp((dot(u_wv_z, vec4(a_pos, 1.0)) + u_fog_far) * u_fog_dist, 0.0, 1.0)\n"
    "          : 1.0;\n"
    "    vec3 nrm = a_nrm * u_nrm_scale;\n"
    "    vec3 wn = u_world * nrm;\n"
    "    if (u_rnorm != 0 || u_tex_map_mode == 2) {\n"
    "        float wl = length(wn);\n"
    "        if (wl > 0.0) wn /= wl;\n"
    "    }\n"
    "    vec2 uv;\n"
    "    if (u_tex_map_mode == 2) {\n"
    "        uv = vec2(0.5 * (1.0 + dot(wn, u_env_lu)), 0.5 * (1.0 + dot(wn, u_env_lv)));\n"
    "    } else if (u_tex_map_mode == 1) {\n"
    "        vec3 ti;\n"
    "        if (u_tex_proj_mode == 0) ti = a_pos * u_pos_scale;\n"
    "        else if (u_tex_proj_mode == 1) ti = vec3(a_uv.xy * u_uv_xform.xz + u_uv_xform.yw, 1.0);\n"
    "        else if (u_tex_proj_mode == 2) { float nl = length(nrm); ti = (nl > 0.0) ? (nrm / nl) : nrm; }\n"
    "        else ti = nrm;\n"
    "        vec4 ti4 = vec4(ti, 1.0);\n"
    "        uv = vec2(dot(u_tmat[0], ti4), dot(u_tmat[1], ti4));\n"
    "    } else {\n"
    "        uv = a_uv.xy * u_uv_xform.xz + u_uv_xform.yw;\n"
    "    }\n"
    "    v_uvq = vec3(uv, 1.0);\n"
    "    vec4 mat_a = ((u_cmat & 1) != 0) ? a_col : u_mat_amb;\n"
    "    vec3 mat_d = ((u_cmat & 2) != 0) ? a_col.rgb : u_mat_diff.rgb;\n"
    "    vec3 mat_s = ((u_cmat & 4) != 0) ? a_col.rgb : u_mat_spec.rgb;\n"
    "    vec3 rgb = u_mat_emis.rgb + mat_a.rgb * u_lt_global_amb;\n"
    "    vec3 wp = u_world * (a_pos * u_pos_scale) + u_world_t;\n"
    "    for (int i = 0; i < 4; i++) {\n"
    "        if (u_lt_on[i] == 0) continue;\n"
    "        vec3 L;\n"
    "        float att = 1.0;\n"
    "        if (u_lt_type[i] == 0) {\n"
    "            float ll = length(u_lt_pos[i]);\n"
    "            L = (ll > 0.0) ? (u_lt_pos[i] / ll) : vec3(0.0);\n"
    "        } else {\n"
    "            vec3 d = u_lt_pos[i] - wp;\n"
    "            float dist = length(d);\n"
    "            L = (dist > 0.0) ? (d / dist) : vec3(0.0);\n"
    "            att = 1.0 / max(1e-6, u_lt_att[i].x + u_lt_att[i].y * dist + u_lt_att[i].z * dist * dist);\n"
    "            if (u_lt_type[i] == 2) {\n"
    "                float dl = length(u_lt_dir[i]);\n"
    "                vec3 sd_v = (dl > 0.0) ? (u_lt_dir[i] / dl) : vec3(0.0);\n"
    "                float sd = dot(-L, sd_v);\n"
    "                if (sd < u_lt_spot_cut[i]) att = 0.0;\n"
    "                else att *= pow(max(sd, 0.0), u_lt_spot_exp[i]);\n"
    "            }\n"
    "        }\n"
    "        rgb += att * (mat_a.rgb * u_lt_amb[i]);\n"
    "        float ndotl = max(dot(wn, L), 0.0);\n"
    "        if (ndotl > 0.0) {\n"
    "            rgb += att * ndotl * mat_d * u_lt_diff[i];\n"
    "            if (u_lt_comp[i] == 1 && u_spow > 0.0) {\n"
    "                vec3 H = normalize(L + vec3(0.0, 0.0, 1.0));\n"
    "                float ndoth = max(dot(wn, H), 0.0);\n"
    "                if (ndoth > 0.0) {\n"
    "                    rgb += att * pow(ndoth, u_spow) * mat_s * u_lt_spec[i];\n"
    "                }\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "    v_col = vec4(clamp(rgb, 0.0, 1.0), clamp(mat_a.a, 0.0, 1.0));\n"
    "}\n";

static const char *ge_fs_src =
    "#version 300 es\n"
    "precision mediump float;\n"
    "in vec4 v_col;\n"
    "in vec3 v_uvq;\n"
    "in float v_fog;\n"
    "flat in float v_tex_slot;\n"
    "uniform sampler2D u_tex;\n"
    "uniform sampler2D u_tex1;\n"
    "uniform sampler2D u_tex2;\n"
    "uniform sampler2D u_tex3;\n"
    "uniform sampler2D u_tex4;\n"
    "uniform sampler2D u_tex5;\n"
    "uniform sampler2D u_tex6;\n"
    "uniform sampler2D u_tex7;\n"
    "uniform sampler2D u_tex8;\n"
    "uniform sampler2D u_tex9;\n"
    "uniform sampler2D u_tex10;\n"
    "uniform sampler2D u_tex11;\n"
    "uniform sampler2D u_tex12;\n"
    "uniform sampler2D u_tex13;\n"
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
    "uniform mediump int u_fog_enable;\n"
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
    "            int s = int(v_tex_slot + 0.5);\n"
    "            if (s < 8) {\n"
    "                if (s < 4) {\n"
    "                    if (s < 2) { t = (s == 0) ? texture(u_tex, suv, -0.75) : texture(u_tex1, suv, -0.75); }\n"
    "                    else       { t = (s == 2) ? texture(u_tex2, suv, -0.75) : texture(u_tex3, suv, -0.75); }\n"
    "                } else {\n"
    "                    if (s < 6) { t = (s == 4) ? texture(u_tex4, suv, -0.75) : texture(u_tex5, suv, -0.75); }\n"
    "                    else       { t = (s == 6) ? texture(u_tex6, suv, -0.75) : texture(u_tex7, suv, -0.75); }\n"
    "                }\n"
    "            } else {\n"
    "                if (s < 12) {\n"
    "                    if (s < 10) { t = (s == 8) ? texture(u_tex8, suv, -0.75) : texture(u_tex9, suv, -0.75); }\n"
    "                    else        { t = (s == 10) ? texture(u_tex10, suv, -0.75) : texture(u_tex11, suv, -0.75); }\n"
    "                } else {\n"
    "                    t = (s == 12) ? texture(u_tex12, suv, -0.75) : texture(u_tex13, suv, -0.75);\n"
    "                }\n"
    "            }\n"
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

static void query_ge_prog_locs(GLuint prog, ge_prog_locs *L) {
    memset(L, 0xFF, sizeof(*L));
    L->prog            = prog;
    L->loc_pos         = glGetAttribLocation(prog, "a_pos");
    L->loc_col         = glGetAttribLocation(prog, "a_col");
    L->loc_uv          = glGetAttribLocation(prog, "a_uv");
    L->loc_fog         = glGetAttribLocation(prog, "a_fog");
    L->loc_nrm         = glGetAttribLocation(prog, "a_nrm");
    L->u_tex[0]        = glGetUniformLocation(prog, "u_tex");
    L->u_tex[1]        = glGetUniformLocation(prog, "u_tex1");
    L->u_tex[2]        = glGetUniformLocation(prog, "u_tex2");
    L->u_tex[3]        = glGetUniformLocation(prog, "u_tex3");
    L->u_tex[4]        = glGetUniformLocation(prog, "u_tex4");
    L->u_tex[5]        = glGetUniformLocation(prog, "u_tex5");
    L->u_tex[6]        = glGetUniformLocation(prog, "u_tex6");
    L->u_tex[7]        = glGetUniformLocation(prog, "u_tex7");
    L->u_tex[8]        = glGetUniformLocation(prog, "u_tex8");
    L->u_tex[9]        = glGetUniformLocation(prog, "u_tex9");
    L->u_tex[10]       = glGetUniformLocation(prog, "u_tex10");
    L->u_tex[11]       = glGetUniformLocation(prog, "u_tex11");
    L->u_tex[12]       = glGetUniformLocation(prog, "u_tex12");
    L->u_tex[13]       = glGetUniformLocation(prog, "u_tex13");
    L->u_tex_enable    = glGetUniformLocation(prog, "u_tex_enable");
    L->u_tex_func      = glGetUniformLocation(prog, "u_tex_func");
    L->u_tex_alpha     = glGetUniformLocation(prog, "u_tex_alpha");
    L->u_tex_double    = glGetUniformLocation(prog, "u_tex_double");
    L->u_tex_env       = glGetUniformLocation(prog, "u_tex_env");
    L->u_atest         = glGetUniformLocation(prog, "u_atest");
    L->u_atest_func    = glGetUniformLocation(prog, "u_atest_func");
    L->u_atest_ref     = glGetUniformLocation(prog, "u_atest_ref");
    L->u_atest_mask    = glGetUniformLocation(prog, "u_atest_mask");
    L->u_clear         = glGetUniformLocation(prog, "u_clear");
    L->u_fbo_uv_scale  = glGetUniformLocation(prog, "u_fbo_uv_scale");
    L->u_fbo_uv_offset = glGetUniformLocation(prog, "u_fbo_uv_offset");
    L->u_depth_clut_mode = glGetUniformLocation(prog, "u_depth_clut_mode");
    L->u_depth_tex     = glGetUniformLocation(prog, "u_depth_tex");
    L->u_clut_tex      = glGetUniformLocation(prog, "u_clut_tex");
    L->u_clut_shift    = glGetUniformLocation(prog, "u_clut_shift");
    L->u_clut_mask     = glGetUniformLocation(prog, "u_clut_mask");
    L->u_clut_start    = glGetUniformLocation(prog, "u_clut_start");
    L->u_fog_enable    = glGetUniformLocation(prog, "u_fog_enable");
    L->u_fog_far       = glGetUniformLocation(prog, "u_fog_far");
    L->u_fog_dist      = glGetUniformLocation(prog, "u_fog_dist");
    L->u_fog_color     = glGetUniformLocation(prog, "u_fog_color");
    L->u_raw_mode      = glGetUniformLocation(prog, "u_raw_mode");
    L->u_mvp           = glGetUniformLocation(prog, "u_mvp");
    L->u_wv_z          = glGetUniformLocation(prog, "u_wv_z");
    L->u_uv_xform      = glGetUniformLocation(prog, "u_uv_xform");
    L->u_pos_scale     = glGetUniformLocation(prog, "u_pos_scale");
    L->u_nrm_scale     = glGetUniformLocation(prog, "u_nrm_scale");
    L->u_tex_map_mode  = glGetUniformLocation(prog, "u_tex_map_mode");
    L->u_tex_proj_mode = glGetUniformLocation(prog, "u_tex_proj_mode");
    L->u_world         = glGetUniformLocation(prog, "u_world");
    L->u_world_t       = glGetUniformLocation(prog, "u_world_t");
    L->u_tmat          = glGetUniformLocation(prog, "u_tmat");
    L->u_env_lu        = glGetUniformLocation(prog, "u_env_lu");
    L->u_env_lv        = glGetUniformLocation(prog, "u_env_lv");
    L->u_cmat          = glGetUniformLocation(prog, "u_cmat");
    L->u_rnorm         = glGetUniformLocation(prog, "u_rnorm");
    L->u_spow          = glGetUniformLocation(prog, "u_spow");
    L->u_mat_emis      = glGetUniformLocation(prog, "u_mat_emis");
    L->u_mat_amb       = glGetUniformLocation(prog, "u_mat_amb");
    L->u_mat_diff      = glGetUniformLocation(prog, "u_mat_diff");
    L->u_mat_spec      = glGetUniformLocation(prog, "u_mat_spec");
    L->u_lt_global_amb = glGetUniformLocation(prog, "u_lt_global_amb");
    L->u_lt_on         = glGetUniformLocation(prog, "u_lt_on");
    L->u_lt_type       = glGetUniformLocation(prog, "u_lt_type");
    L->u_lt_comp       = glGetUniformLocation(prog, "u_lt_comp");
    L->u_lt_pos        = glGetUniformLocation(prog, "u_lt_pos");
    L->u_lt_dir        = glGetUniformLocation(prog, "u_lt_dir");
    L->u_lt_att        = glGetUniformLocation(prog, "u_lt_att");
    L->u_lt_spot_exp   = glGetUniformLocation(prog, "u_lt_spot_exp");
    L->u_lt_spot_cut   = glGetUniformLocation(prog, "u_lt_spot_cut");
    L->u_lt_amb        = glGetUniformLocation(prog, "u_lt_amb");
    L->u_lt_diff       = glGetUniformLocation(prog, "u_lt_diff");
    L->u_lt_spec       = glGetUniformLocation(prog, "u_lt_spec");
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
    const char *asp = getenv("MC3_ASPECT");
    if ((w * 272 < h * 480 && !getenv("MC3_LETTERBOX")) ||
        (asp && strcmp(asp, "4:3") == 0)) {
        g_viewport_x = 0;
        g_viewport_y = 0;
        g_viewport_w = w;
        g_viewport_h = h;
    } else {
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
#ifdef PSP_GE_HAS_SET_ASPECT
    psp_ge_set_aspect((float)g_viewport_w / (float)g_viewport_h);
#endif
}

static uint32_t fnv1a_32(const uint8_t *data, uint32_t len) {
    uint32_t h = 2166136261u;
    if ((((uintptr_t)data | len) & 3u) == 0u) {
        const uint32_t *w = (const uint32_t *)data;
        uint32_t n = len >> 2;
        for (uint32_t i = 0; i < n; i++) {
            h ^= w[i];
            h *= 16777619u;
        }
        return h;
    }
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
    free(fl->mt_slots);
#ifdef PSP_RENDER_HAS_DRAW_RAW
    free(fl->raw_vbuf);
    free(fl->raw_ibuf);
    free(fl->raw_cbuf);
    free(fl->xforms);
#endif
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
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, g_fbo_w, g_fbo_h, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, NULL);

        glGenTextures(1, &s->depth_tex);
        glBindTexture(GL_TEXTURE_2D, s->depth_tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH24_STENCIL8, g_fbo_w, g_fbo_h, 0,
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
        uint32_t w;
        memcpy(&w, p + (i << 2), 4);
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

static tex_cache_entry *get_cached_texture_slot(const gles_mt_slot *ms, const uint8_t *clut_data,
                                                GLuint *cur_bound_tex) {
    int w = ms->tex_w > 0 ? ms->tex_w : 1;
    int h = ms->tex_h > 0 ? ms->tex_h : 1;
    if (w > 1024) w = 1024;
    if (h > 1024) h = 1024;

    g_tex_lru_tick++;

    uint32_t b = tex_cache_bucket(ms->tex_addr, w, h, ms->tex_fmt, ms->clut_hash, ms->data_hash);
    int cur = g_tex_buckets[b];
    while (cur > 0) {
        tex_cache_entry *e = &g_tex_cache[cur - 1];
        if (e->valid &&
            e->tex_addr == ms->tex_addr &&
            e->tex_w == w && e->tex_h == h &&
            e->tex_bw == ms->tex_bw && e->tex_fmt == ms->tex_fmt &&
            e->tex_swizzle == ms->tex_swizzle &&
            e->clut_hash == ms->clut_hash &&
            e->data_hash == ms->data_hash &&
            e->clut_fmt == ms->clut_fmt &&
            e->clut_shift == ms->clut_shift &&
            e->clut_mask == ms->clut_mask &&
            e->clut_start == ms->clut_start) {
            e->last_used = g_tex_lru_tick;
            g_exec_tex_hits++;
            return e;
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
    slot->tex_addr = ms->tex_addr;
    slot->tex_w = w;
    slot->tex_h = h;
    slot->tex_bw = ms->tex_bw;
    slot->tex_fmt = ms->tex_fmt;
    slot->tex_swizzle = ms->tex_swizzle;
    slot->clut_hash = ms->clut_hash;
    slot->data_hash = ms->data_hash;
    slot->clut_fmt = ms->clut_fmt;
    slot->clut_shift = ms->clut_shift;
    slot->clut_mask = ms->clut_mask;
    slot->clut_start = ms->clut_start;
    slot->wrap_s = -1;
    slot->wrap_t = -1;
    slot->bucket = b;
    slot->hash_next = g_tex_buckets[b];
    g_tex_buckets[b] = lru_idx + 1;

    uint64_t t_up0 = gles_now_us();
    g_exec_tex_misses++;

    psp_draw_state local_st;
    memset(&local_st, 0, sizeof(local_st));
    local_st.tex_addr = ms->tex_addr;
    local_st.tex_w = ms->tex_w;
    local_st.tex_h = ms->tex_h;
    local_st.tex_bw = ms->tex_bw;
    local_st.tex_fmt = ms->tex_fmt;
    local_st.tex_swizzle = ms->tex_swizzle;
    local_st.clut_fmt = ms->clut_fmt;
    local_st.clut_shift = ms->clut_shift;
    local_st.clut_mask = ms->clut_mask;
    local_st.clut_start = ms->clut_start;
    local_st.clut = clut_data;

    static uint32_t tex_decode[1024 * 1024];
    for (int y = 0; y < h; y++) {
        uint32_t *row = &tex_decode[y * w];
        for (int x = 0; x < w; x++) {
            row[x] = psp_tex_fetch(&local_st, x, y);
        }
    }

    glBindTexture(GL_TEXTURE_2D, slot->tex);
    if (cur_bound_tex) *cur_bound_tex = slot->tex;
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, tex_decode);
    if ((w & (w - 1)) == 0 && (h & (h - 1)) == 0 && w >= 16 && h >= 16) {
        glGenerateMipmap(GL_TEXTURE_2D);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST);
        if (g_max_aniso > 1.0f) {
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, g_max_aniso);
        }
    } else {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    }
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    g_exec_tex_up_us += gles_now_us() - t_up0;
    return slot;
}

/* ---- backend entry points ----------------------------------------------- */

static SDL_AudioDeviceID g_audio_dev = 0;
static int g_audio_have_freq = 44100;

#define AUDIO_RING_FRAMES 32768
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
    uint32_t frac_pos;
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
    int total_frames = len / 4;
    int frames = total_frames;
    int32_t mix[4096 * 2];
    if (frames > 4096) {
        memset(out + 4096 * 2, 0, (size_t)(total_frames - 4096) * 4u);
        frames = 4096;
    }
    memset(mix, 0, (size_t)frames * 2u * sizeof(int32_t));

    static uint64_t last_cb_us = 0;
    uint64_t now_us = gles_now_us();
    double cb_gap_ms = last_cb_us ? ((double)(now_us - last_cb_us) / 1000.0) : 0.0;
    last_cb_us = now_us;

    int active_chs = 0;
    uint32_t min_buf = UINT32_MAX, max_buf = 0;
    int underrun_ch = -1;
    uint32_t underrun_avail = 0, underrun_need = (uint32_t)frames;

    uint32_t step_fp = (uint32_t)(((uint64_t)44100u << 16) /
                                  (uint64_t)(g_audio_have_freq > 0 ? g_audio_have_freq : 44100));
    uint64_t need_in = ((uint64_t)frames * step_fp + 65535u) >> 16;

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

        if (avail < need_in && underrun_ch < 0) {
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
        int n = 0;
        if (step_fp == 65536u) {
            n = (avail < (uint64_t)frames) ? (int)avail : frames;
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
            g_audio_streams[s].rpos += (uint64_t)n;
        } else {
            for (int i = 0; i < frames; i++) {
                uint64_t rem = g_audio_streams[s].wpos - g_audio_streams[s].rpos;
                if (rem < 2u) break;
                uint32_t i0 = (uint32_t)(g_audio_streams[s].rpos % AUDIO_RING_FRAMES) * 2u;
                uint32_t i1 = (uint32_t)((g_audio_streams[s].rpos + 1u) % AUDIO_RING_FRAMES) * 2u;
                int32_t l0 = g_audio_streams[s].buf[i0 + 0], r0 = g_audio_streams[s].buf[i0 + 1];
                int32_t l1 = g_audio_streams[s].buf[i1 + 0], r1 = g_audio_streams[s].buf[i1 + 1];
                int32_t frac = (int32_t)g_audio_streams[s].frac_pos;
                int16_t sl = (int16_t)(l0 + (((l1 - l0) * frac) >> 16));
                int16_t sr = (int16_t)(r0 + (((r1 - r0) * frac) >> 16));
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
                uint32_t nf = g_audio_streams[s].frac_pos + step_fp;
                g_audio_streams[s].rpos += (uint64_t)(nf >> 16);
                g_audio_streams[s].frac_pos = nf & 0xFFFFu;
                n++;
            }
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
        g_audio_streams[slot].frac_pos = 0;
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
static void stage_worker_start(void);
static void stage_worker_stop(void);

#ifdef PSP_RENDER_HAS_DRAW_RAW
extern psp_render_backend g_gles_be;
#endif
static int gles_init(int w, int h) {
#ifdef PSP_RENDER_HAS_DRAW_RAW
    if (!getenv("MC3_GPU_TL")) g_gles_be.draw_raw = NULL;
#endif
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
        want.samples = 1024;
        want.callback = gles_audio_callback;
        g_audio_dev = SDL_OpenAudioDevice(NULL, 0, &want, &have, SDL_AUDIO_ALLOW_FREQUENCY_CHANGE);
        if (g_audio_dev) {
            g_audio_have_freq = (have.freq > 0) ? have.freq : 44100;
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
                g_joystick = SDL_GameControllerGetJoystick(g_controller);
                break;
            }
        }
    }
    if (!g_joystick && SDL_NumJoysticks() > 0) {
        g_joystick = SDL_JoystickOpen(0);
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

    SDL_DisableScreenSaver();
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
    update_viewport(w, h);
    {
        const char *rs = getenv("MC3_RES_SCALE");
        if (rs && *rs) {
            float sc = strtof(rs, NULL);
            if (sc < 1.0f) sc = 1.0f;
            if (sc > 4.0f) sc = 4.0f;
            g_fbo_w = (int)(480.0f * sc + 0.5f);
            g_fbo_h = (int)(272.0f * sc + 0.5f);
        } else {
            g_fbo_w = (g_viewport_w >= 480) ? (g_viewport_w <= 1920 ? g_viewport_w : 1920) : 480;
            g_fbo_h = (g_viewport_h >= 272) ? (g_viewport_h <= 1088 ? g_viewport_h : 1088) : 272;
        }
    }
    if (SDL_GL_ExtensionSupported("GL_EXT_texture_filter_anisotropic")) {
        float max_a = 1.0f;
        glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &max_a);
        float want_a = 4.0f;
        const char *ae = getenv("MC3_ANISO");
        if (ae && *ae) {
            float v = strtof(ae, NULL);
            if (v >= 1.0f) want_a = v;
        }
        g_max_aniso = (max_a < want_a) ? max_a : want_a;
    }

    g_egl_context = eglGetCurrentContext();
    g_egl_surface = eglGetCurrentSurface(EGL_DRAW);
    if (g_egl_context == EGL_NO_CONTEXT || g_egl_surface == EGL_NO_SURFACE) {
        fprintf(stderr, "gles32: warning: eglGetCurrentContext/Surface returned NO_CONTEXT\n");
    }

    g_quad_prog = link_program(quad_vs_src, quad_fs_src);
    g_ge_prog   = link_program(ge_vs_src, ge_fs_src);
    g_ge_lit_prog = link_program(ge_lit_vs_src, ge_fs_src);
    if (!g_quad_prog || !g_ge_prog || !g_ge_lit_prog) {
        gles_shutdown();
        return -1;
    }

    g_quad_loc_pos = glGetAttribLocation(g_quad_prog, "a_pos");
    g_quad_loc_uv  = glGetAttribLocation(g_quad_prog, "a_uv");
    g_quad_u_tex   = glGetUniformLocation(g_quad_prog, "u_tex");

    query_ge_prog_locs(g_ge_prog, &g_ge_unlit);
    query_ge_prog_locs(g_ge_lit_prog, &g_ge_lit);

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

    glGenBuffers(GE_VBO_RING, g_ge_vbo);
    glGenBuffers(1, &g_ge_raw_vbo);
    glGenBuffers(1, &g_ge_raw_cbo);
    glGenBuffers(1, &g_ge_raw_ibo);
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
    stage_worker_start();

    fprintf(stderr,
            "gles32: initialized %dx%d window, content %dx%d at (%d,%d), FBO %dx%d, aniso %.1fx, GLES %s (GPU raster)\n",
            w, h, g_viewport_w, g_viewport_h, g_viewport_x, g_viewport_y,
            g_fbo_w, g_fbo_h, g_max_aniso, glGetString(GL_VERSION));
    return 0;
}

static void gles_shutdown(void) {
    stage_worker_stop();
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
    if (g_fmv_tex) { glDeleteTextures(1, &g_fmv_tex); g_fmv_tex = 0; }
    if (g_fb_tex) { glDeleteTextures(1, &g_fb_tex); g_fb_tex = 0; }
    free(g_stage.fmv_buf); g_stage.fmv_buf = NULL;
    free(g_front.fmv_buf); g_front.fmv_buf = NULL;
    free(g_exec.fmv_buf);  g_exec.fmv_buf  = NULL;
    for (int i = 0; i < GE_VBO_RING; i++) {
        if (g_ge_vbo[i]) { glDeleteBuffers(1, &g_ge_vbo[i]); g_ge_vbo[i] = 0; }
    }
    if (g_ge_raw_vbo) { glDeleteBuffers(1, &g_ge_raw_vbo); g_ge_raw_vbo = 0; }
    if (g_ge_raw_cbo) { glDeleteBuffers(1, &g_ge_raw_cbo); g_ge_raw_cbo = 0; }
    if (g_ge_raw_ibo) { glDeleteBuffers(1, &g_ge_raw_ibo); g_ge_raw_ibo = 0; }
    if (g_quad_vbo) { glDeleteBuffers(1, &g_quad_vbo); g_quad_vbo = 0; }
    if (g_ge_prog) { glDeleteProgram(g_ge_prog); g_ge_prog = 0; }
    if (g_ge_lit_prog) { glDeleteProgram(g_ge_lit_prog); g_ge_lit_prog = 0; }
    if (g_quad_prog) { glDeleteProgram(g_quad_prog); g_quad_prog = 0; }
    psp_audio_set_sink(NULL);
    if (g_audio_dev) { SDL_CloseAudioDevice(g_audio_dev); g_audio_dev = 0; }
    SDL_QuitSubSystem(SDL_INIT_AUDIO);
    if (g_controller) { SDL_GameControllerClose(g_controller); g_controller = NULL; g_joystick = NULL; }
    else if (g_joystick) { SDL_JoystickClose(g_joystick); g_joystick = NULL; }
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

#if defined(__GNUC__) || defined(__clang__)
#define MC3_NOINLINE __attribute__((noinline))
#else
#define MC3_NOINLINE
#endif

static inline void write_vert(gles_vert * __restrict__ out,
                              const psp_vertex * __restrict__ pv,
                              float inv_tw, float inv_th, int tex_en) {
#ifdef PSP_VERTEX_HAS_SUBPIXEL_FOG
    out->x = (pv->fx * (1.0f / 240.0f)) - 1.0f;
    out->y = (pv->fy * (1.0f / 136.0f)) - 1.0f;
    float fg = pv->fog;
    out->fog = (fg < 0.0f) ? 0.0f : ((fg > 1.0f) ? 1.0f : fg);
#else
    out->x = ((float)pv->x * (1.0f / 240.0f)) - 1.0f;
    out->y = ((float)pv->y * (1.0f / 136.0f)) - 1.0f;
    out->fog = 1.0f;
#endif
    float zn = (pv->z * (1.0f / 32767.5f)) - 1.0f;
    out->z = (zn < -1.0f) ? -1.0f : ((zn > 1.0f) ? 1.0f : zn);
    out->rgba = pv->rgba;
    if (tex_en) {
#ifdef PSP_VERTEX_HAS_W
        float q = (pv->w != 0.0f) ? pv->w : 1.0f;
#else
        float q = 1.0f;
#endif
        out->u = (pv->u * inv_tw) * q;
        out->v = (pv->v * inv_th) * q;
        out->q = q;
    } else {
        out->u = 0.0f;
        out->v = 0.0f;
        out->q = 1.0f;
    }
}

static inline gles_vert make_vert(const psp_vertex *pv, float inv_tw, float inv_th) {
    gles_vert gv;
    write_vert(&gv, pv, inv_tw, inv_th, 1);
    return gv;
}

static MC3_NOINLINE int ensure_verts_grow(gles_frame_list *fl, uint32_t need) {
    uint32_t ncap = fl->vert_cap ? fl->vert_cap * 2u : 8192u;
    while (ncap < need) ncap *= 2u;
    gles_vert *p = (gles_vert *)realloc(fl->verts, (size_t)ncap * sizeof(*p));
    if (!p) return 0;
    fl->verts = p;
    fl->vert_cap = ncap;
    return 1;
}

static inline int ensure_verts(gles_frame_list *fl, uint32_t add) {
    uint32_t need = fl->vert_count + add;
    if (need <= fl->vert_cap) return 1;
    return ensure_verts_grow(fl, need);
}

static inline int tri_culled_2d(const psp_vertex *a, const psp_vertex *b, const psp_vertex *c, int cull_mode) {
#ifdef PSP_VERTEX_HAS_SUBPIXEL_FOG
    float x0 = a->fx, x1 = b->fx, x2 = c->fx;
    float y0 = a->fy, y1 = b->fy, y2 = c->fy;
    if (cull_mode) {
        float area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0);
        if (cull_mode > 0) { if (area <= 0.0f) return 1; }
        else               { if (area >= 0.0f) return 1; }
    }
    if ((x0 < 0.0f && x1 < 0.0f && x2 < 0.0f) ||
        (x0 > 480.0f && x1 > 480.0f && x2 > 480.0f) ||
        (y0 < 0.0f && y1 < 0.0f && y2 < 0.0f) ||
        (y0 > 272.0f && y1 > 272.0f && y2 > 272.0f)) return 1;
#else
    (void)a; (void)b; (void)c; (void)cull_mode;
#endif
    return 0;
}

static inline gles_vert *emit_tri_direct(gles_vert *dst, const gles_vert *a, const gles_vert *b, const gles_vert *c, int gouraud) {
    dst[0] = *a;
    dst[1] = *b;
    dst[2] = *c;
    if (!gouraud) {
        dst[0].rgba = dst[1].rgba = c->rgba;
    }
    return dst + 3;
}

static inline gles_vert *emit_tri_to(gles_vert *dst, const gles_vert *a, const gles_vert *b, const gles_vert *c, int gouraud) {
    if ((a->x < -1.0f && b->x < -1.0f && c->x < -1.0f) ||
        (a->x >  1.0f && b->x >  1.0f && c->x >  1.0f) ||
        (a->y < -1.0f && b->y < -1.0f && c->y < -1.0f) ||
        (a->y >  1.0f && b->y >  1.0f && c->y >  1.0f)) {
        return dst;
    }
    return emit_tri_direct(dst, a, b, c, gouraud);
}

static inline int cmd_non_tex_eq(const psp_draw_state *a, const psp_draw_state *b);
static inline void fill_mt_slot_st(gles_mt_slot *s, const psp_draw_state *st,
                                   uint32_t clut_offset, uint32_t clut_len,
                                   uint32_t clut_hash, uint32_t data_hash) {
    s->tex_addr = st->tex_addr;
    s->tex_w = st->tex_w;
    s->tex_h = st->tex_h;
    s->tex_bw = st->tex_bw;
    s->tex_fmt = st->tex_fmt;
    s->tex_swizzle = st->tex_swizzle;
    s->tex_clamp_u = st->tex_clamp_u;
    s->tex_clamp_v = st->tex_clamp_v;
    s->clut_offset = clut_offset;
    s->clut_len = clut_len;
    s->clut_hash = clut_hash;
    s->data_hash = data_hash;
    s->clut_fmt = st->clut_fmt;
    s->clut_shift = st->clut_shift;
    s->clut_mask = st->clut_mask;
    s->clut_start = st->clut_start;
}
static inline int mt_slot_eq(const gles_mt_slot *a, const gles_mt_slot *b);

static MC3_NOINLINE gles_cmd *stage_push_cmd(GLenum gl_prim, uint32_t first_vert, uint32_t vcount,
                                const psp_draw_state *st, int prim, int is_raw,
                                uint32_t raw_voff, uint32_t raw_ioff, uint32_t raw_coff,
                                int raw_stride, uint32_t raw_vtype, int raw_isz, uint32_t xform_idx,
                                uint32_t fb_addr);

static void gles_draw_exec(int prim, const psp_vertex *v, int count, const psp_draw_state *st, uint32_t fb_addr) {
    if (!v || count <= 0 || !st) return;
    if (count > 65536) count = 65536;

    const int tex_en = (st->tex_enable && st->tex_w > 0 && st->tex_h > 0);
    float inv_tw = tex_en ? (1.0f / (float)st->tex_w) : 1.0f;
    float inv_th = tex_en ? (1.0f / (float)st->tex_h) : 1.0f;

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
    gles_vert *dst_base = &g_stage.verts[first_vert];
    gles_vert *dst = dst_base;
    const int gouraud = st->gouraud;
    const int cull_mode = (!st->clear && st->cull_enable) ? (st->cull_dir ? 1 : -1) : 0;

    if (prim == PSP_PRIM_TRIANGLES) {
        for (int i = 0; i + 2 < count; i += 3) {
            if (!(v[i].behind | v[i + 1].behind | v[i + 2].behind)) {
                if (tri_culled_2d(&v[i], &v[i + 1], &v[i + 2], cull_mode)) continue;
                write_vert(&dst[0], &v[i], inv_tw, inv_th, tex_en);
                write_vert(&dst[1], &v[i + 1], inv_tw, inv_th, tex_en);
                write_vert(&dst[2], &v[i + 2], inv_tw, inv_th, tex_en);
                if (!gouraud) dst[0].rgba = dst[1].rgba = dst[2].rgba;
                dst += 3;
            }
        }
    } else if (prim == PSP_PRIM_TRIANGLE_STRIP) {
        /* Keep last 2 converted vertices in registers; culled triangles skip write_vert(). */
        gles_vert prev0, prev1;
        int prev_valid = 0;
        for (int i = 0; i + 2 < count; i++) {
            int cm = (i & 1) ? -cull_mode : cull_mode;
            if ((v[i].behind | v[i + 1].behind | v[i + 2].behind) ||
                tri_culled_2d(&v[i], &v[i + 1], &v[i + 2], cm)) {
                prev_valid = 0;
                continue;
            }
            gles_vert a, b, c;
            if (prev_valid) {
                a = prev0;
                b = prev1;
            } else {
                write_vert(&a, &v[i], inv_tw, inv_th, tex_en);
                write_vert(&b, &v[i + 1], inv_tw, inv_th, tex_en);
                prev_valid = 1;
            }
            write_vert(&c, &v[i + 2], inv_tw, inv_th, tex_en);
            prev0 = b;
            prev1 = c;
            if (!gouraud) { a.rgba = c.rgba; b.rgba = c.rgba; }
            dst[0] = (i & 1) ? b : a;
            dst[1] = (i & 1) ? a : b;
            dst[2] = c;
#ifdef PSP_VERTEX_HAS_SUBPIXEL_FOG
            dst += 3;
#else
            dst = emit_tri_to(dst, &dst[0], &dst[1], &dst[2], gouraud);
#endif
        }
    } else if (prim == PSP_PRIM_TRIANGLE_FAN) {
        if (!v[0].behind) {
            gles_vert gv0, prev_gv;
            int v0_built = 0, prev_idx = -1;
            for (int i = 1; i + 1 < count; i++) {
                if (v[i].behind || v[i + 1].behind) continue;
                if (tri_culled_2d(&v[0], &v[i], &v[i + 1], cull_mode)) continue;
                if (!v0_built) {
                    write_vert(&gv0, &v[0], inv_tw, inv_th, tex_en);
                    v0_built = 1;
                }
                dst[0] = gv0;
                if (prev_idx == i) dst[1] = prev_gv;
                else               write_vert(&dst[1], &v[i], inv_tw, inv_th, tex_en);
                write_vert(&dst[2], &v[i + 1], inv_tw, inv_th, tex_en);
                prev_gv = dst[2];
                prev_idx = i + 1;
#ifdef PSP_VERTEX_HAS_SUBPIXEL_FOG
                if (!gouraud) dst[0].rgba = dst[1].rgba = dst[2].rgba;
                dst += 3;
#else
                dst = emit_tri_to(dst, &dst[0], &dst[1], &dst[2], gouraud);
#endif
            }
        }
    } else if (prim == PSP_PRIM_POINTS) {
        for (int i = 0; i < count; i++) {
            if (!v[i].behind) write_vert(dst++, &v[i], inv_tw, inv_th, tex_en);
        }
    } else if (prim == PSP_PRIM_LINES) {
        for (int i = 0; i + 1 < count; i += 2) {
            if (!v[i].behind && !v[i + 1].behind) {
                write_vert(dst++, &v[i], inv_tw, inv_th, tex_en);
                write_vert(dst++, &v[i + 1], inv_tw, inv_th, tex_en);
            }
        }
    } else if (prim == PSP_PRIM_LINE_STRIP) {
        gles_vert prev_gv;
        int prev_idx = -1;
        for (int i = 0; i + 1 < count; i++) {
            if (!v[i].behind && !v[i + 1].behind) {
                if (prev_idx == i) *dst++ = prev_gv;
                else               write_vert(dst++, &v[i], inv_tw, inv_th, tex_en);
                write_vert(dst, &v[i + 1], inv_tw, inv_th, tex_en);
                prev_gv = *dst++;
                prev_idx = i + 1;
            }
        }
    } else {
        for (int i = 0; i + 1 < count; i += 2) {
            if (v[i].behind || v[i + 1].behind) continue;
#ifdef PSP_VERTEX_HAS_SUBPIXEL_FOG
            if ((v[i].fx < 0.0f && v[i + 1].fx < 0.0f) ||
                (v[i].fx > 480.0f && v[i + 1].fx > 480.0f) ||
                (v[i].fy < 0.0f && v[i + 1].fy < 0.0f) ||
                (v[i].fy > 272.0f && v[i + 1].fy > 272.0f)) continue;
#endif
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
            dst = emit_tri_to(dst, &gtl, &gtr, &gbr, gouraud);
            dst = emit_tri_to(dst, &gtl, &gbr, &gbl, gouraud);
        }
    }

    uint32_t vcount = (uint32_t)(dst - dst_base);
    g_stage.vert_count = first_vert + vcount;
    if (vcount == 0) return;

    const psp_draw_state *push_st = st;
    psp_draw_state ui_st;
    if (!st->clear && g_fbo_w * 272 < g_fbo_h * 480 &&
        !(st->tex_enable && (st->tex_addr & 0x1FFFFFFFu) < 0x08000000u) &&
        (prim == PSP_PRIM_SPRITES || (!st->ztest && v[0].w == 1.0f && v[count - 1].w == 1.0f))) {
        int is_fade = 0;
        if (!st->tex_enable) {
            float min_x = dst_base[0].x, max_x = dst_base[0].x;
            float min_y = dst_base[0].y, max_y = dst_base[0].y;
            for (uint32_t k = 1; k < vcount; k++) {
                float x = dst_base[k].x, y = dst_base[k].y;
                if (x < min_x) min_x = x; if (x > max_x) max_x = x;
                if (y < min_y) min_y = y; if (y > max_y) max_y = y;
            }
            if ((max_x - min_x) >= (470.0f / 240.0f) && (max_y - min_y) >= (268.0f / 136.0f))
                is_fade = 1;
        }
        if (!is_fade) {
            float ay = ((float)g_fbo_w / (float)g_fbo_h) / (480.0f / 272.0f);
            for (uint32_t k = 0; k < vcount; k++) dst_base[k].y *= ay;
            ui_st = *st;
            ui_st.sc_y0 = (int)(136.0f + ((float)ui_st.sc_y0 - 136.0f) * ay);
            ui_st.sc_y1 = (int)(136.0f + ((float)(ui_st.sc_y1 + 1) - 136.0f) * ay) - 1;
            if (ui_st.sc_y0 < 0) ui_st.sc_y0 = 0;
            if (ui_st.sc_y1 > 271) ui_st.sc_y1 = 271;
            push_st = &ui_st;
        }
    }

    stage_push_cmd(gl_prim, first_vert, vcount, push_st, prim, 0, 0, 0, 0, 0, 0, 0, 0, fb_addr);
}

/* --- Core 2 stage worker: overlaps gles_draw_exec() with Core 0 GE T&L --- */
#define MAX_STAGE_BATCHES 8192u
#define STAGE_VERT_CAP    (512u * 1024u)
#define STAGE_CLUT_SLOTS  256u

typedef struct {
    int            prim;
    uint32_t       v_off;
    uint32_t       count;
    uint32_t       fb_addr;
    psp_draw_state st;
} gles_stage_batch;

static gles_stage_batch *g_sw_batches = NULL;
static psp_vertex       *g_sw_verts = NULL;
static uint8_t          *g_sw_cluts = NULL;
static uint32_t          g_sw_vwrite = 0;
static uint32_t          g_sw_cwrite = 0;
static uint32_t          g_sw_last_clut_gen = 0;
static int               g_sw_last_clut_slot = -1;
static uint8_t           g_sw_last_clut[1024];

static volatile uint32_t g_sw_head = 0;
static volatile uint32_t g_sw_tail = 0;
static pthread_t         g_sw_thread;
static pthread_mutex_t   g_sw_mtx = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t    g_sw_work_cv = PTHREAD_COND_INITIALIZER;
static pthread_cond_t    g_sw_done_cv = PTHREAD_COND_INITIALIZER;
static int               g_sw_started = 0;
static int               g_sw_quit = 0;
static int               g_sw_busy = 0;

static void *stage_worker_main(void *arg) {
    (void)arg;
#if defined(__linux__) && defined(__aarch64__)
    if (!getenv("MC3_NO_PIN")) {
        cpu_set_t cs;
        CPU_ZERO(&cs);
        CPU_SET(3, &cs);
        sched_setaffinity(0, sizeof(cs), &cs);
    }
#endif
    /* Slightly lower CFS priority than gles32_present_loop so GPU submission isn't starved. */
    int nice_rc = nice(2);
    (void)nice_rc;
    for (;;) {
        uint32_t h = g_sw_head;
        uint32_t t = __atomic_load_n(&g_sw_tail, __ATOMIC_ACQUIRE);
        if (h != t) {
            while (h != t) {
                const gles_stage_batch *b = &g_sw_batches[h & (MAX_STAGE_BATCHES - 1u)];
                gles_draw_exec(b->prim, &g_sw_verts[b->v_off], (int)b->count, &b->st, b->fb_addr);
                h++;
                __atomic_store_n(&g_sw_head, h, __ATOMIC_RELEASE);
                t = __atomic_load_n(&g_sw_tail, __ATOMIC_ACQUIRE);
            }
            continue;
        }
        pthread_mutex_lock(&g_sw_mtx);
        g_sw_busy = 0;
        pthread_cond_broadcast(&g_sw_done_cv);
        while (__atomic_load_n(&g_sw_tail, __ATOMIC_ACQUIRE) == g_sw_head && !g_sw_quit) {
            pthread_cond_wait(&g_sw_work_cv, &g_sw_mtx);
        }
        if (g_sw_quit && __atomic_load_n(&g_sw_tail, __ATOMIC_ACQUIRE) == g_sw_head) {
            pthread_mutex_unlock(&g_sw_mtx);
            break;
        }
        g_sw_busy = 1;
        pthread_mutex_unlock(&g_sw_mtx);
    }
    return NULL;
}

static void stage_worker_start(void) {
    if (g_sw_started || getenv("MC3_SYNC_STAGE")) return;
    g_sw_batches = (gles_stage_batch *)malloc(MAX_STAGE_BATCHES * sizeof(*g_sw_batches));
    g_sw_verts   = (psp_vertex *)malloc(STAGE_VERT_CAP * sizeof(*g_sw_verts));
    g_sw_cluts   = (uint8_t *)malloc(STAGE_CLUT_SLOTS * 1024u);
    if (!g_sw_batches || !g_sw_verts || !g_sw_cluts) {
        free(g_sw_batches); g_sw_batches = NULL;
        free(g_sw_verts);   g_sw_verts = NULL;
        free(g_sw_cluts);   g_sw_cluts = NULL;
        return;
    }
    g_sw_head = g_sw_tail = 0;
    g_sw_vwrite = g_sw_cwrite = 0;
    g_sw_last_clut_slot = -1;
    g_sw_quit = 0;
    g_sw_busy = 0;
    if (pthread_create(&g_sw_thread, NULL, stage_worker_main, NULL) == 0) {
        g_sw_started = 1;
    } else {
        free(g_sw_batches); g_sw_batches = NULL;
        free(g_sw_verts);   g_sw_verts = NULL;
        free(g_sw_cluts);   g_sw_cluts = NULL;
    }
}

static void stage_worker_drain(void) {
    if (!g_sw_started) return;
    pthread_mutex_lock(&g_sw_mtx);
    if (__atomic_load_n(&g_sw_head, __ATOMIC_ACQUIRE) != g_sw_tail) {
        pthread_cond_signal(&g_sw_work_cv);
    }
    while (__atomic_load_n(&g_sw_head, __ATOMIC_ACQUIRE) != g_sw_tail || g_sw_busy) {
        pthread_cond_wait(&g_sw_done_cv, &g_sw_mtx);
    }
    pthread_mutex_unlock(&g_sw_mtx);
    g_sw_vwrite = 0;
    g_sw_cwrite = 0;
    g_sw_last_clut_slot = -1;
}

static void stage_worker_stop(void) {
    if (!g_sw_started) return;
    stage_worker_drain();
    pthread_mutex_lock(&g_sw_mtx);
    g_sw_quit = 1;
    pthread_cond_signal(&g_sw_work_cv);
    pthread_mutex_unlock(&g_sw_mtx);
    pthread_join(g_sw_thread, NULL);
    g_sw_started = 0;
    free(g_sw_batches); g_sw_batches = NULL;
    free(g_sw_verts);   g_sw_verts = NULL;
    free(g_sw_cluts);   g_sw_cluts = NULL;
}

static void gles_draw(int prim, const psp_vertex *v, int count, const psp_draw_state *st) {
    if (!v || count <= 0 || !st) return;
    if (count > 65536) count = 65536;
    if (!g_sw_started) {
        gles_draw_exec(prim, v, count, st, g_fb_addr);
        return;
    }
    uint32_t t = g_sw_tail;
    uint32_t h = __atomic_load_n(&g_sw_head, __ATOMIC_ACQUIRE);
    if ((t - h) >= MAX_STAGE_BATCHES - 1u || g_sw_vwrite + (uint32_t)count > STAGE_VERT_CAP) {
        stage_worker_drain();
        t = g_sw_tail;
    }
    uint32_t voff = g_sw_vwrite;
    memcpy(&g_sw_verts[voff], v, (size_t)count * sizeof(psp_vertex));
    g_sw_vwrite = voff + (uint32_t)count;

    gles_stage_batch *b = &g_sw_batches[t & (MAX_STAGE_BATCHES - 1u)];
    b->prim = prim;
    b->v_off = voff;
    b->count = (uint32_t)count;
    b->fb_addr = g_fb_addr;
    b->st = *st;
    if (st->tex_enable && st->clut &&
        st->tex_fmt >= PSP_TPSM_T4 && st->tex_fmt <= PSP_TPSM_T32) {
        if (g_sw_last_clut_slot >= 0 && g_sw_last_clut_gen == st->clut_gen &&
            memcmp(g_sw_last_clut, st->clut, 1024u) == 0) {
            b->st.clut = &g_sw_cluts[(uint32_t)g_sw_last_clut_slot * 1024u];
        } else {
            if (g_sw_cwrite >= STAGE_CLUT_SLOTS) {
                stage_worker_drain();
                /* g_sw_verts/g_sw_vwrite were reset by drain; re-copy vertices at 0 */
                voff = 0;
                memcpy(&g_sw_verts[0], v, (size_t)count * sizeof(psp_vertex));
                g_sw_vwrite = (uint32_t)count;
                t = g_sw_tail;
                b = &g_sw_batches[t & (MAX_STAGE_BATCHES - 1u)];
                b->prim = prim;
                b->v_off = 0;
                b->count = (uint32_t)count;
                b->fb_addr = g_fb_addr;
                b->st = *st;
            }
            uint32_t cslot = g_sw_cwrite++;
            uint8_t *cdst = &g_sw_cluts[cslot * 1024u];
            memcpy(cdst, st->clut, 1024u);
            memcpy(g_sw_last_clut, st->clut, 1024u);
            g_sw_last_clut_gen = st->clut_gen;
            g_sw_last_clut_slot = (int)cslot;
            b->st.clut = cdst;
        }
    } else {
        b->st.clut = NULL;
    }

    uint32_t nt = t + 1u;
    __atomic_store_n(&g_sw_tail, nt, __ATOMIC_RELEASE);
    if ((nt & 15u) == 0u) {
        pthread_mutex_lock(&g_sw_mtx);
        pthread_cond_signal(&g_sw_work_cv);
        pthread_mutex_unlock(&g_sw_mtx);
    }
}

#ifdef PSP_RENDER_HAS_DRAW_RAW
static void raw_vertex_offsets(uint32_t vtype, int *tex_off, int *col_off, int *nrm_off, int *pos_off) {
    static const int tex_sz[4]  = { 0, 1, 2, 4 };
    static const int col_sz[8]  = { 0, 0, 0, 0, 2, 2, 2, 4 };
    static const int norm_sz[4] = { 0, 1, 2, 4 };
    static const int pos_sz[4]  = { 0, 1, 2, 4 };
    int off = 0;
    int ts = tex_sz[vtype & 3u];
    if (ts) { off = (off + ts - 1) & ~(ts - 1); *tex_off = off; off += ts * 2; }
    else *tex_off = -1;
    int cs = col_sz[(vtype >> 2) & 7u];
    if (cs) { off = (off + cs - 1) & ~(cs - 1); *col_off = off; off += cs; }
    else *col_off = -1;
    int ns = norm_sz[(vtype >> 5) & 3u];
    if (ns) { off = (off + ns - 1) & ~(ns - 1); *nrm_off = off; off += ns * 3; }
    else *nrm_off = -1;
    int ps = pos_sz[(vtype >> 7) & 3u];
    off = (off + ps - 1) & ~(ps - 1);
    *pos_off = off;
}

static int ensure_raw_vbuf(uint32_t add) {
    uint32_t need = (g_stage.raw_vbytes + 3u) & ~3u;
    if (need + add <= g_stage.raw_vcap) { g_stage.raw_vbytes = need; return 1; }
    uint32_t ncap = g_stage.raw_vcap ? g_stage.raw_vcap * 2u : 1048576u;
    while (ncap < need + add) ncap *= 2u;
    uint8_t *p = (uint8_t *)realloc(g_stage.raw_vbuf, ncap);
    if (!p) return 0;
    g_stage.raw_vbuf = p;
    g_stage.raw_vcap = ncap;
    g_stage.raw_vbytes = need;
    return 1;
}

static int ensure_raw_cbuf(uint32_t need) {
    if (need <= g_stage.raw_ccap) return 1;
    uint32_t ncap = g_stage.raw_ccap ? g_stage.raw_ccap * 2u : 1048576u;
    while (ncap < need) ncap *= 2u;
    uint8_t *p = (uint8_t *)realloc(g_stage.raw_cbuf, ncap);
    if (!p) return 0;
    g_stage.raw_cbuf = p;
    g_stage.raw_ccap = ncap;
    return 1;
}

static int ensure_raw_ibuf(uint32_t add) {
    uint32_t need = (g_stage.raw_ibytes + 3u) & ~3u;
    if (need + add <= g_stage.raw_icap) { g_stage.raw_ibytes = need; return 1; }
    uint32_t ncap = g_stage.raw_icap ? g_stage.raw_icap * 2u : 262144u;
    while (ncap < need + add) ncap *= 2u;
    uint8_t *p = (uint8_t *)realloc(g_stage.raw_ibuf, ncap);
    if (!p) return 0;
    g_stage.raw_ibuf = p;
    g_stage.raw_icap = ncap;
    g_stage.raw_ibytes = need;
    return 1;
}

static inline uint32_t read_raw_idx(const void *idata, int isz, uint32_t k) {
    if (isz == 0) return k;
    if (isz == 1) return ((const uint8_t *)idata)[k];
    if (isz == 2) return ((const uint16_t *)idata)[k];
    return ((const uint32_t *)idata)[k];
}

/* Per-frame local-space AABB cache for raw vertex slices (reset on frame start). */
typedef struct {
    const void *vdata;
    uint32_t    nverts, vtype;
    int         stride;
    float       mn[3], mx[3];
    uint32_t    gen;
    int         valid;
} raw_aabb_ent;
#define RAW_AABB_SLOTS 512
static raw_aabb_ent s_aabb[RAW_AABB_SLOTS];
static uint32_t g_aabb_gen = 1;

static int raw_get_aabb(const void *vdata, uint32_t nverts, uint32_t vtype, int stride,
                        float mn[3], float mx[3]) {
    int pfmt = (int)((vtype >> 7) & 3u);
    if (pfmt == 0 || (vtype & (1u << 23))) return 0;
    uintptr_t h = ((uintptr_t)vdata >> 2) * 2654435761u + nverts * 40503u + vtype;
    raw_aabb_ent *e = &s_aabb[h & (RAW_AABB_SLOTS - 1)];
    if (e->valid && e->gen == g_aabb_gen && e->vdata == vdata && e->nverts == nverts && e->vtype == vtype && e->stride == stride) {
        memcpy(mn, e->mn, sizeof(e->mn)); memcpy(mx, e->mx, sizeof(e->mx));
        return 1;
    }
    if (nverts > 65536u) return 0;
    int to, co, no, po;
    raw_vertex_offsets(vtype, &to, &co, &no, &po);
    const uint8_t *base = (const uint8_t *)vdata + po;
    float lo[3] = {  3.0e38f,  3.0e38f,  3.0e38f };
    float hi[3] = { -3.0e38f, -3.0e38f, -3.0e38f };
    for (uint32_t k = 0; k < nverts; k++) {
        const uint8_t *p = base + (size_t)k * (uint32_t)stride;
        float c[3];
        if (pfmt == 1)      { c[0] = (float)(int8_t)p[0]; c[1] = (float)(int8_t)p[1]; c[2] = (float)(int8_t)p[2]; }
        else if (pfmt == 2) { int16_t s[3]; memcpy(s, p, 6); c[0] = s[0]; c[1] = s[1]; c[2] = s[2]; }
        else                { memcpy(c, p, 12); }
        for (int a = 0; a < 3; a++) {
            if (!(c[a] == c[a]) || c[a] > 3.0e37f || c[a] < -3.0e37f) return 0;
            if (c[a] < lo[a]) lo[a] = c[a];
            if (c[a] > hi[a]) hi[a] = c[a];
        }
    }
    e->vdata = vdata; e->nverts = nverts; e->vtype = vtype; e->stride = stride; e->valid = 1; e->gen = g_aabb_gen;
    memcpy(e->mn, lo, sizeof(lo)); memcpy(e->mx, hi, sizeof(hi));
    memcpy(mn, lo, sizeof(lo)); memcpy(mx, hi, sizeof(hi));
    return 1;
}

/* Returns 1 if the whole AABB lies outside the clip-space x or y frustum planes. */
static int raw_aabb_outside(const float *m, const float mn[3], const float mx[3]) {
    float cen[3] = { 0.5f * (mn[0] + mx[0]), 0.5f * (mn[1] + mx[1]), 0.5f * (mn[2] + mx[2]) };
    float hf[3]  = { 0.5f * (mx[0] - mn[0]), 0.5f * (mx[1] - mn[1]), 0.5f * (mx[2] - mn[2]) };
    float c[4], e[4];
    for (int r = 0; r < 4; r++) {
        c[r] = m[r] * cen[0] + m[4 + r] * cen[1] + m[8 + r] * cen[2] + m[12 + r];
        e[r] = fabsf(m[r]) * hf[0] + fabsf(m[4 + r]) * hf[1] + fabsf(m[8 + r]) * hf[2];
    }
    /* rows: 0=x, 1=y, 3=w */
    if (c[3] + e[3] <= 0.0f) return 1;                     /* entirely behind the eye */
    if ((c[0] + c[3]) + (e[0] + e[3]) < 0.0f) return 1;   /* x < -w */
    if ((c[3] - c[0]) + (e[0] + e[3]) < 0.0f) return 1;   /* x >  w */
    if ((c[1] + c[3]) + (e[1] + e[3]) < 0.0f) return 1;   /* y < -w */
    if ((c[3] - c[1]) + (e[1] + e[3]) < 0.0f) return 1;   /* y >  w */
    return 0;
}

static void gles_draw_raw(int prim, const void *vdata, const void *idata,
                          uint32_t count, int stride, uint32_t vtype, int isz,
                          const psp_xform_state *xf, const psp_draw_state *st) {
    if (!vdata || count == 0 || stride <= 0 || !xf || !st) return;
    stage_worker_drain();

    GLenum gl_prim;
    int expand_tris = 0;
    switch (prim) {
    case PSP_PRIM_POINTS:         gl_prim = GL_POINTS; break;
    case PSP_PRIM_LINES:          gl_prim = GL_LINES; break;
    case PSP_PRIM_LINE_STRIP:     gl_prim = GL_LINE_STRIP; break;
    case PSP_PRIM_TRIANGLES:
        gl_prim = GL_TRIANGLES;
        if (isz == 0 || isz == 1) expand_tris = 1;
        break;
    case PSP_PRIM_TRIANGLE_STRIP:
    case PSP_PRIM_TRIANGLE_FAN:
        if (count < 3) return;
        gl_prim = GL_TRIANGLES;
        expand_tris = 1;
        break;
    default: return;
    }

    uint32_t nverts = count;
    uint32_t ioff = 0;
    int out_isz = isz;
    uint32_t out_count = count;

    if (isz > 0) {
        if (!idata) return;
        uint32_t max_idx = 0;
#ifdef PSP_XFORM_HAS_RAW_NVERTS
        if (xf->raw_nverts > 0u) {
            max_idx = xf->raw_nverts - 1u;
        } else
#endif
        if (isz == 1) {
            const uint8_t *ip = (const uint8_t *)idata;
            for (uint32_t k = 0; k < count; k++) if (ip[k] > max_idx) max_idx = ip[k];
        } else if (isz == 2) {
            const uint16_t *ip = (const uint16_t *)idata;
            for (uint32_t k = 0; k < count; k++) if (ip[k] > max_idx) max_idx = ip[k];
        } else {
            const uint32_t *ip = (const uint32_t *)idata;
            for (uint32_t k = 0; k < count; k++) if (ip[k] > max_idx) max_idx = ip[k];
        }
        nverts = max_idx + 1u;
    }

    if (!st->clear) {
        float mn[3], mx[3];
        if (raw_get_aabb(vdata, nverts, vtype, stride, mn, mx) &&
            raw_aabb_outside(xf->ndc_mvp, mn, mx)) {
            return;
        }
    }

    uint32_t xidx;
    if (g_stage.xform_count > 0 &&
        memcmp(&g_stage.xforms[g_stage.xform_count - 1u], xf, sizeof(*xf)) == 0) {
        xidx = g_stage.xform_count - 1u;
    } else {
        if (g_stage.xform_count >= g_stage.xform_cap) {
            uint32_t ncap = g_stage.xform_cap ? g_stage.xform_cap * 2u : 512u;
            psp_xform_state *xp = (psp_xform_state *)realloc(g_stage.xforms, (size_t)ncap * sizeof(*xp));
            if (!xp) return;
            g_stage.xforms = xp;
            g_stage.xform_cap = ncap;
        }
        xidx = g_stage.xform_count++;
        g_stage.xforms[xidx] = *xf;
    }

    uint32_t vbytes = nverts * (uint32_t)stride;
    int col_fmt = (int)((vtype >> 2) & 7u);
    int unpack_col = (col_fmt >= 4 && col_fmt <= 6);
    uint32_t cbytes = unpack_col ? (nverts * 4u) : 0u;

    const gles_cmd *prev_cmd = (g_stage.cmd_count > 0u) ? &g_stage.cmds[g_stage.cmd_count - 1u] : NULL;
    int can_rebase = (gl_prim == GL_TRIANGLES && nverts <= 65536u &&
                      prev_cmd && prev_cmd->is_raw && prev_cmd->gl_prim == GL_TRIANGLES &&
                      !prev_cmd->st.clear && prev_cmd->target_addr == g_fb_addr &&
                      prev_cmd->raw_vtype == vtype && prev_cmd->raw_stride == stride &&
                      prev_cmd->raw_isz == 2 &&
                      (unpack_col ? (prev_cmd->raw_col_off == prev_cmd->raw_voff)
                                  : (prev_cmd->raw_col_off == UINT32_MAX)) &&
                      prev_cmd->xform_idx == xidx &&
                      prev_cmd->raw_ioff + prev_cmd->vert_count * 2u == g_stage.raw_ibytes);

    #define RAW_VDATA_CACHE_SLOTS 16
    static struct {
        const void *vdata;
        uint32_t    vbytes, voff, coff;
        int         col_fmt, stride;
    } s_vcache[RAW_VDATA_CACHE_SLOTS];
    static int s_vcache_next = 0;
    if (g_stage.raw_vbytes == 0) {
        memset(s_vcache, 0, sizeof(s_vcache));
        s_vcache_next = 0;
    }

    uint32_t voff = 0, coff = UINT32_MAX;
    int hit = 0;
    for (int ci = 0; ci < RAW_VDATA_CACHE_SLOTS; ci++) {
        if (s_vcache[ci].vdata == vdata && s_vcache[ci].vbytes >= vbytes &&
            s_vcache[ci].col_fmt == col_fmt && s_vcache[ci].stride == stride) {
            voff = s_vcache[ci].voff;
            coff = s_vcache[ci].coff;
            hit = 1;
            break;
        }
    }
    if (!hit) {
        uint32_t pad = 0;
        if (can_rebase && g_stage.raw_vbytes >= prev_cmd->raw_voff) {
            uint32_t rem = (g_stage.raw_vbytes - prev_cmd->raw_voff) % (uint32_t)stride;
            if (rem) pad = (uint32_t)stride - rem;
            if (!ensure_raw_vbuf(pad + vbytes)) return;
            voff = g_stage.raw_vbytes + pad;
        } else {
            if (!ensure_raw_vbuf(vbytes + cbytes)) return;
            voff = g_stage.raw_vbytes;
        }
        memcpy(g_stage.raw_vbuf + voff, vdata, vbytes);
        g_stage.raw_vbytes = voff + vbytes;
        if (unpack_col) {
            int to, co, no, po;
            raw_vertex_offsets(vtype, &to, &co, &no, &po);
            if (!ensure_raw_cbuf(voff + vbytes)) return;
            const uint8_t *src = g_stage.raw_vbuf + voff + co;
            uint8_t *cdst = g_stage.raw_cbuf + voff;
            for (uint32_t k = 0; k < nverts; k++) {
                uint16_t px;
                memcpy(&px, src + k * (uint32_t)stride, 2);
                uint32_t c32 = psp_color_expand(px, col_fmt - 4);
                memcpy(cdst + k * (uint32_t)stride, &c32, 4);
            }
            if (voff + vbytes > g_stage.raw_cbytes) g_stage.raw_cbytes = voff + vbytes;
            coff = voff;
        }
        s_vcache[s_vcache_next].vdata = vdata;
        s_vcache[s_vcache_next].vbytes = vbytes;
        s_vcache[s_vcache_next].voff = voff;
        s_vcache[s_vcache_next].coff = coff;
        s_vcache[s_vcache_next].col_fmt = col_fmt;
        s_vcache[s_vcache_next].stride = stride;
        s_vcache_next = (s_vcache_next + 1) & (RAW_VDATA_CACHE_SLOTS - 1);
    }

    uint32_t base_vert = 0;
    if (can_rebase && voff >= prev_cmd->raw_voff &&
        ((voff - prev_cmd->raw_voff) % (uint32_t)stride) == 0u) {
        uint32_t bv = (voff - prev_cmd->raw_voff) / (uint32_t)stride;
        if (bv + nverts <= 65536u) {
            base_vert = bv;
            voff = prev_cmd->raw_voff;
            expand_tris = 1;
        } else {
            can_rebase = 0;
        }
    } else {
        can_rebase = 0;
    }

    if (unpack_col) coff = voff;

    if (expand_tris) {
        uint32_t max_tris = (prim == PSP_PRIM_TRIANGLES) ? (count / 3u) : (count - 2u);
        if (max_tris == 0) return;
        out_isz = (can_rebase || nverts <= 65536u) ? 2 : 4;
        uint32_t max_ibytes = max_tris * 3u * (uint32_t)out_isz;
        if (can_rebase) {
            uint32_t need = g_stage.raw_ibytes + max_ibytes;
            if (need > g_stage.raw_icap) {
                uint32_t ncap = g_stage.raw_icap ? g_stage.raw_icap * 2u : 262144u;
                while (ncap < need) ncap *= 2u;
                uint8_t *p = (uint8_t *)realloc(g_stage.raw_ibuf, ncap);
                if (!p) return;
                g_stage.raw_ibuf = p;
                g_stage.raw_icap = ncap;
            }
            ioff = g_stage.raw_ibytes;
        } else {
            if (!ensure_raw_ibuf(max_ibytes)) return;
            ioff = g_stage.raw_ibytes;
        }
        uint32_t n_out = 0;
        if (out_isz == 2) {
            uint16_t *dst = (uint16_t *)(g_stage.raw_ibuf + ioff);
            uint16_t bv = (uint16_t)base_vert;
            if (prim == PSP_PRIM_TRIANGLES) {
                for (uint32_t i = 0; i + 2u < count; i += 3u) {
                    uint32_t i0 = read_raw_idx(idata, isz, i);
                    uint32_t i1 = read_raw_idx(idata, isz, i + 1u);
                    uint32_t i2 = read_raw_idx(idata, isz, i + 2u);
                    if (i0 == i1 || i1 == i2 || i0 == i2) continue;
                    dst[n_out++] = (uint16_t)i0 + bv;
                    dst[n_out++] = (uint16_t)i1 + bv;
                    dst[n_out++] = (uint16_t)i2 + bv;
                }
            } else if (prim == PSP_PRIM_TRIANGLE_STRIP) {
                for (uint32_t i = 0; i + 2u < count; i++) {
                    uint32_t i0 = read_raw_idx(idata, isz, i);
                    uint32_t i1 = read_raw_idx(idata, isz, i + 1u);
                    uint32_t i2 = read_raw_idx(idata, isz, i + 2u);
                    if (i0 == i1 || i1 == i2 || i0 == i2) continue;
                    dst[n_out++] = (uint16_t)i0 + bv;
                    if (i & 1u) { dst[n_out++] = (uint16_t)i2 + bv; dst[n_out++] = (uint16_t)i1 + bv; }
                    else        { dst[n_out++] = (uint16_t)i1 + bv; dst[n_out++] = (uint16_t)i2 + bv; }
                }
            } else { /* PSP_PRIM_TRIANGLE_FAN */
                uint32_t i0 = read_raw_idx(idata, isz, 0);
                for (uint32_t i = 1u; i + 1u < count; i++) {
                    uint32_t i1 = read_raw_idx(idata, isz, i);
                    uint32_t i2 = read_raw_idx(idata, isz, i + 1u);
                    if (i0 == i1 || i1 == i2 || i0 == i2) continue;
                    dst[n_out++] = (uint16_t)i0 + bv;
                    dst[n_out++] = (uint16_t)i1 + bv;
                    dst[n_out++] = (uint16_t)i2 + bv;
                }
            }
        } else {
            uint32_t *dst = (uint32_t *)(g_stage.raw_ibuf + ioff);
            if (prim == PSP_PRIM_TRIANGLES) {
                for (uint32_t i = 0; i + 2u < count; i += 3u) {
                    uint32_t i0 = read_raw_idx(idata, isz, i);
                    uint32_t i1 = read_raw_idx(idata, isz, i + 1u);
                    uint32_t i2 = read_raw_idx(idata, isz, i + 2u);
                    if (i0 == i1 || i1 == i2 || i0 == i2) continue;
                    dst[n_out++] = i0; dst[n_out++] = i1; dst[n_out++] = i2;
                }
            } else if (prim == PSP_PRIM_TRIANGLE_STRIP) {
                for (uint32_t i = 0; i + 2u < count; i++) {
                    uint32_t i0 = read_raw_idx(idata, isz, i);
                    uint32_t i1 = read_raw_idx(idata, isz, i + 1u);
                    uint32_t i2 = read_raw_idx(idata, isz, i + 2u);
                    if (i0 == i1 || i1 == i2 || i0 == i2) continue;
                    dst[n_out++] = i0;
                    if (i & 1u) { dst[n_out++] = i2; dst[n_out++] = i1; }
                    else        { dst[n_out++] = i1; dst[n_out++] = i2; }
                }
            } else {
                uint32_t i0 = read_raw_idx(idata, isz, 0);
                for (uint32_t i = 1u; i + 1u < count; i++) {
                    uint32_t i1 = read_raw_idx(idata, isz, i);
                    uint32_t i2 = read_raw_idx(idata, isz, i + 1u);
                    if (i0 == i1 || i1 == i2 || i0 == i2) continue;
                    dst[n_out++] = i0; dst[n_out++] = i1; dst[n_out++] = i2;
                }
            }
        }
        if (n_out == 0) return;
        out_count = n_out;
        g_stage.raw_ibytes += n_out * (uint32_t)out_isz;
    } else if (isz > 0) {
        uint32_t ibytes = count * (uint32_t)isz;
        if (!ensure_raw_ibuf(ibytes)) return;
        ioff = g_stage.raw_ibytes;
        memcpy(g_stage.raw_ibuf + ioff, idata, ibytes);
        g_stage.raw_ibytes += ibytes;
    }

    stage_push_cmd(gl_prim, 0, out_count, st, prim, 1,
                   voff, ioff, coff, stride, vtype, out_isz, xidx, g_fb_addr);
}
#endif

static MC3_NOINLINE gles_cmd *stage_push_cmd(GLenum gl_prim, uint32_t first_vert, uint32_t vcount,
                                const psp_draw_state *st, int prim, int is_raw,
                                uint32_t raw_voff, uint32_t raw_ioff, uint32_t raw_coff,
                                int raw_stride, uint32_t raw_vtype, int raw_isz, uint32_t xform_idx,
                                uint32_t fb_addr) {
    static uint32_t stage_tex_epoch = 1;
    static struct {
        uint32_t epoch;
        uint32_t tex_addr;
        int      tex_w, tex_h, tex_bw, tex_fmt;
        uint32_t hash;
    } stage_tex_cache[4096];
    static uint32_t last_clut_epoch = 0, last_clut_gen = 0, last_clut_offset = UINT32_MAX, last_clut_hash = 0;
    if (g_stage.cmd_count == 0) stage_tex_epoch++;

    psp_draw_state new_st = *st;
    if (prim == PSP_PRIM_SPRITES || prim < PSP_PRIM_TRIANGLES) {
#ifdef PSP_DRAW_STATE_HAS_MASK_CULL
        new_st.cull_enable = 0;
#endif
    }
    new_st.clut = NULL;

    uint32_t cmd_data_hash = 0;
    int is_fmv_draw = 0;
    if (!new_st.clear && new_st.tex_enable && new_st.tex_addr) {
        if (new_st.tex_fmt == PSP_TPSM_8888 && !new_st.tex_swizzle &&
            new_st.tex_w == 512 && new_st.tex_h == 512 && new_st.tex_bw == 512 &&
            g_stage.cmd_count < 4u) {
            is_fmv_draw = 1;
            if (!g_stage.fmv_buf) {
                g_stage.fmv_buf = (uint32_t *)malloc(512u * 272u * 4u);
            }
            if (g_stage.fmv_buf) {
                psp_mem_read_block(g_stage.fmv_buf, new_st.tex_addr, 512u * 272u * 4u);
                g_stage.fmv_tex_addr = new_st.tex_addr;
                g_stage.fmv_valid = 1;
            }
        } else {
            static uint32_t last_tex_epoch = 0, last_tex_addr = 0, last_tex_hash = 0;
            static int last_tex_w = 0, last_tex_h = 0, last_tex_bw = 0, last_tex_fmt = -1;
            if (last_tex_epoch == stage_tex_epoch &&
                last_tex_addr == new_st.tex_addr &&
                last_tex_w == new_st.tex_w &&
                last_tex_h == new_st.tex_h &&
                last_tex_bw == new_st.tex_bw &&
                last_tex_fmt == new_st.tex_fmt) {
                cmd_data_hash = last_tex_hash;
            } else {
                uint32_t slot = (new_st.tex_addr ^ ((uint32_t)new_st.tex_w << 7) ^ ((uint32_t)new_st.tex_h << 13) ^ (uint32_t)new_st.tex_fmt) & 4095u;
                uint32_t tex_phys = new_st.tex_addr & 0x1FFFFFFFu;
                int is_ram = (tex_phys >= 0x08000000u);
                int fresh = (stage_tex_cache[slot].epoch != 0u) &&
                            (is_ram ? ((stage_tex_epoch - stage_tex_cache[slot].epoch) < 64u)
                                    : (stage_tex_cache[slot].epoch == stage_tex_epoch));
                if (fresh &&
                    stage_tex_cache[slot].tex_addr == new_st.tex_addr &&
                    stage_tex_cache[slot].tex_w == new_st.tex_w &&
                    stage_tex_cache[slot].tex_h == new_st.tex_h &&
                    stage_tex_cache[slot].tex_bw == new_st.tex_bw &&
                    stage_tex_cache[slot].tex_fmt == new_st.tex_fmt) {
                    cmd_data_hash = stage_tex_cache[slot].hash;
                } else {
                    cmd_data_hash = hash_tex_sample(new_st.tex_addr, new_st.tex_w, new_st.tex_h, new_st.tex_bw, new_st.tex_fmt);
                    stage_tex_cache[slot].epoch = stage_tex_epoch;
                    stage_tex_cache[slot].tex_addr = new_st.tex_addr;
                    stage_tex_cache[slot].tex_w = new_st.tex_w;
                    stage_tex_cache[slot].tex_h = new_st.tex_h;
                    stage_tex_cache[slot].tex_bw = new_st.tex_bw;
                    stage_tex_cache[slot].tex_fmt = new_st.tex_fmt;
                    stage_tex_cache[slot].hash = cmd_data_hash;
                }
                last_tex_epoch = stage_tex_epoch;
                last_tex_addr = new_st.tex_addr;
                last_tex_w = new_st.tex_w;
                last_tex_h = new_st.tex_h;
                last_tex_bw = new_st.tex_bw;
                last_tex_fmt = new_st.tex_fmt;
                last_tex_hash = cmd_data_hash;
            }
        }
    }

    uint32_t cmd_clut_offset = UINT32_MAX;
    uint32_t cmd_clut_len = 0;
    uint32_t cmd_clut_hash = 0;
    if (new_st.tex_enable && st->clut &&
        (new_st.tex_fmt == PSP_TPSM_T4 || new_st.tex_fmt == PSP_TPSM_T8 ||
         new_st.tex_fmt == PSP_TPSM_T16 || new_st.tex_fmt == PSP_TPSM_T32)) {
        if (last_clut_epoch == stage_tex_epoch && last_clut_gen == new_st.clut_gen &&
            last_clut_offset != UINT32_MAX) {
            cmd_clut_offset = last_clut_offset;
            cmd_clut_len = 1024u;
            cmd_clut_hash = last_clut_hash;
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
                cmd_clut_offset = g_stage.clut_bytes;
                cmd_clut_len = clut_sz;
                uint32_t esz = (new_st.clut_fmt == 3) ? 4u : 2u;
                uint32_t off = ((uint32_t)new_st.clut_start << 4) * esz;
                uint32_t hlen = (new_st.tex_fmt == PSP_TPSM_T4) ? (16u * esz) : (((uint32_t)new_st.clut_mask + 1u) * esz);
                if (off >= clut_sz) { off = 0u; hlen = clut_sz; }
                else if (off + hlen > clut_sz) { hlen = clut_sz - off; }
                cmd_clut_hash = fnv1a_32(st->clut + off, hlen);
                g_stage.clut_bytes += clut_sz;
                last_clut_epoch = stage_tex_epoch;
                last_clut_gen = new_st.clut_gen;
                last_clut_offset = cmd_clut_offset;
                last_clut_hash = cmd_clut_hash;
            }
        }
    }

    if (!is_fmv_draw && gl_prim == GL_TRIANGLES && !new_st.clear && g_stage.cmd_count > 0u) {
        gles_cmd *prev = &g_stage.cmds[g_stage.cmd_count - 1u];
        if (!is_raw && !prev->is_raw &&
            prev->gl_prim == GL_TRIANGLES &&
            !prev->st.clear &&
            prev->target_addr == fb_addr &&
            prev->first_vert + prev->vert_count == first_vert) {
            if (prev->data_hash == cmd_data_hash &&
                prev->clut_hash == cmd_clut_hash &&
                prev->clut_offset == cmd_clut_offset &&
                memcmp(&prev->st, &new_st, sizeof(psp_draw_state)) == 0) {
                prev->vert_count += vcount;
                return NULL;
            }
            static int mt_en = -1;
            if (mt_en < 0) mt_en = (getenv("MC3_NO_MULTI_TEX") == NULL);
            if (mt_en && new_st.tex_enable && new_st.tex_addr &&
                (new_st.tex_addr & 0x1FFFFFFFu) >= 0x08000000u &&
                prev->st.tex_enable && prev->st.tex_addr &&
                (prev->st.tex_addr & 0x1FFFFFFFu) >= 0x08000000u &&
                !(prev->st.tex_fmt == PSP_TPSM_8888 && !prev->st.tex_swizzle &&
                  prev->st.tex_w == 512 && prev->st.tex_h == 512 && prev->st.tex_bw == 512) &&
                cmd_non_tex_eq(&prev->st, &new_st)) {
                gles_mt_slot cs;
                fill_mt_slot_st(&cs, &new_st, cmd_clut_offset, cmd_clut_len, cmd_clut_hash, cmd_data_hash);
                int found = -1;
                if (prev->mt_count == 0) {
                    gles_mt_slot s0;
                    fill_mt_slot_st(&s0, &prev->st, prev->clut_offset, prev->clut_len, prev->clut_hash, prev->data_hash);
                    if (mt_slot_eq(&s0, &cs)) {
                        found = 0;
                    } else {
                        if (g_stage.mt_count + 2u > g_stage.mt_cap) {
                            uint32_t ncap = g_stage.mt_cap ? g_stage.mt_cap * 2u : 1024u;
                            while (ncap < g_stage.mt_count + 2u) ncap *= 2u;
                            gles_mt_slot *mp = (gles_mt_slot *)realloc(g_stage.mt_slots, (size_t)ncap * sizeof(*mp));
                            if (mp) { g_stage.mt_slots = mp; g_stage.mt_cap = ncap; }
                        }
                        if (g_stage.mt_count + 2u <= g_stage.mt_cap && g_stage.mt_count <= 65535u) {
                            prev->mt_start = (uint16_t)g_stage.mt_count;
                            prev->mt_count = 2;
                            g_stage.mt_slots[g_stage.mt_count++] = s0;
                            g_stage.mt_slots[g_stage.mt_count++] = cs;
                            found = 1;
                        }
                    }
                } else {
                    for (uint8_t s = 0; s < prev->mt_count; s++) {
                        if (mt_slot_eq(&g_stage.mt_slots[prev->mt_start + s], &cs)) { found = s; break; }
                    }
                    if (found < 0 && prev->mt_count < GE_MT_MAX &&
                        (uint32_t)prev->mt_start + prev->mt_count == g_stage.mt_count) {
                        if (g_stage.mt_count + 1u > g_stage.mt_cap) {
                            uint32_t ncap = g_stage.mt_cap ? g_stage.mt_cap * 2u : 1024u;
                            gles_mt_slot *mp = (gles_mt_slot *)realloc(g_stage.mt_slots, (size_t)ncap * sizeof(*mp));
                            if (mp) { g_stage.mt_slots = mp; g_stage.mt_cap = ncap; }
                        }
                        if (g_stage.mt_count + 1u <= g_stage.mt_cap) {
                            found = prev->mt_count++;
                            g_stage.mt_slots[g_stage.mt_count++] = cs;
                        }
                    }
                }
                if (found >= 0) {
                    if (found > 0) {
                        float bias = (float)found * 4.0f;
                        gles_vert *vp = &g_stage.verts[first_vert];
                        for (uint32_t v = 0; v < vcount; v++) vp[v].fog += bias;
                    }
                    prev->vert_count += vcount;
                    return NULL;
                }
            }
        }
#ifdef PSP_RENDER_HAS_DRAW_RAW
        if (is_raw && prev->is_raw &&
            prev->gl_prim == GL_TRIANGLES &&
            !prev->st.clear &&
            prev->target_addr == fb_addr &&
            prev->raw_voff == raw_voff &&
            prev->raw_vtype == raw_vtype &&
            prev->raw_stride == raw_stride &&
            prev->raw_isz == raw_isz &&
            prev->raw_col_off == raw_coff &&
            prev->xform_idx == xform_idx &&
            raw_isz > 0 &&
            prev->raw_ioff + prev->vert_count * (uint32_t)raw_isz == raw_ioff &&
            prev->data_hash == cmd_data_hash &&
            prev->clut_hash == cmd_clut_hash &&
            prev->clut_offset == cmd_clut_offset &&
            memcmp(&prev->st, &new_st, sizeof(psp_draw_state)) == 0) {
            prev->vert_count += vcount;
            return NULL;
        }
#endif
    }

    if (g_stage.cmd_count >= g_stage.cmd_cap) {
        uint32_t ncap = g_stage.cmd_cap ? g_stage.cmd_cap * 2u : 1024u;
        gles_cmd *p = (gles_cmd *)realloc(g_stage.cmds, (size_t)ncap * sizeof(*p));
        if (!p) return NULL;
        g_stage.cmds = p;
        g_stage.cmd_cap = ncap;
    }

    gles_cmd *cmd = &g_stage.cmds[g_stage.cmd_count++];
    memset(cmd, 0, sizeof(*cmd));
    cmd->target_addr = fb_addr;
    cmd->gl_prim = gl_prim;
    cmd->first_vert = first_vert;
    cmd->vert_count = vcount;
    cmd->st = new_st;
    cmd->clut_offset = cmd_clut_offset;
    cmd->clut_len = cmd_clut_len;
    cmd->clut_hash = cmd_clut_hash;
    cmd->data_hash = cmd_data_hash;
#ifdef PSP_RENDER_HAS_DRAW_RAW
    cmd->is_raw = is_raw;
    cmd->raw_voff = raw_voff;
    cmd->raw_ioff = raw_ioff;
    cmd->raw_col_off = raw_coff;
    cmd->raw_stride = raw_stride;
    cmd->raw_vtype = raw_vtype;
    cmd->raw_isz = raw_isz;
    cmd->xform_idx = xform_idx;
#else
    (void)raw_voff; (void)raw_ioff; (void)raw_coff;
    (void)raw_stride; (void)raw_vtype; (void)raw_isz; (void)xform_idx;
#endif
    return cmd;
}

static void gles_finish(void) {
    stage_worker_drain();
}

typedef struct {
    const gles_cmd *cmd;
    uint32_t        orig_idx;
} sort_cmd_ref;

static int sort_cmd_ref_cmp(const void *pa, const void *pb) {
    const sort_cmd_ref *ra = (const sort_cmd_ref *)pa;
    const sort_cmd_ref *rb = (const sort_cmd_ref *)pb;
    const gles_cmd *a = ra->cmd;
    const gles_cmd *b = rb->cmd;
    if (a->st.tex_enable != b->st.tex_enable) return (a->st.tex_enable < b->st.tex_enable) ? -1 : 1;
    if (a->st.tex_addr != b->st.tex_addr) return (a->st.tex_addr < b->st.tex_addr) ? -1 : 1;
    if (a->data_hash != b->data_hash) return (a->data_hash < b->data_hash) ? -1 : 1;
    if (a->clut_hash != b->clut_hash) return (a->clut_hash < b->clut_hash) ? -1 : 1;
    if (a->clut_offset != b->clut_offset) return (a->clut_offset < b->clut_offset) ? -1 : 1;
    if (a->st.tex_fmt != b->st.tex_fmt) return (a->st.tex_fmt < b->st.tex_fmt) ? -1 : 1;
    if (a->st.tex_w != b->st.tex_w) return (a->st.tex_w < b->st.tex_w) ? -1 : 1;
    if (a->st.tex_h != b->st.tex_h) return (a->st.tex_h < b->st.tex_h) ? -1 : 1;
    if (a->st.atest != b->st.atest) return (a->st.atest < b->st.atest) ? -1 : 1;
    if (a->st.cull_enable != b->st.cull_enable) return (a->st.cull_enable < b->st.cull_enable) ? -1 : 1;
    if (a->st.cull_dir != b->st.cull_dir) return (a->st.cull_dir < b->st.cull_dir) ? -1 : 1;
    if (a->st.fog_enable != b->st.fog_enable) return (a->st.fog_enable < b->st.fog_enable) ? -1 : 1;
    if (a->st.zfunc != b->st.zfunc) return (a->st.zfunc < b->st.zfunc) ? -1 : 1;
    int sc = memcmp(&a->st, &b->st, sizeof(psp_draw_state));
    if (sc != 0) return sc;
    return (ra->orig_idx < rb->orig_idx) ? -1 : (ra->orig_idx > rb->orig_idx);
}

static inline int cmd_is_opaque_tri(const gles_cmd *c) {
#ifdef PSP_RENDER_HAS_DRAW_RAW
    if (c->is_raw) return 0;
#endif
    return !c->st.clear && c->gl_prim == GL_TRIANGLES &&
           !c->st.blend && c->st.ztest && c->st.zwrite &&
           !c->st.stencil_enable;
}

static void stage_sort_batch_opaque(gles_frame_list *fl) {
    static int sort_en = -1;
    if (sort_en < 0) sort_en = (getenv("MC3_NO_SORT_BATCH") == NULL);
    if (!sort_en || fl->cmd_count < 4u) return;

    static sort_cmd_ref *s_refs = NULL;
    static gles_cmd     *s_cmds = NULL;
    static uint32_t      s_ref_cap = 0;
    static gles_vert    *s_verts = NULL;
    static uint32_t      s_vert_cap = 0;

    uint32_t n = fl->cmd_count;
    if (n > s_ref_cap) {
        uint32_t ncap = s_ref_cap ? s_ref_cap * 2u : 2048u;
        while (ncap < n) ncap *= 2u;
        sort_cmd_ref *rp = (sort_cmd_ref *)realloc(s_refs, (size_t)ncap * sizeof(*rp));
        gles_cmd *cp = (gles_cmd *)realloc(s_cmds, (size_t)ncap * sizeof(*cp));
        if (!rp || !cp) {
            if (rp) s_refs = rp;
            if (cp) s_cmds = cp;
            return;
        }
        s_refs = rp;
        s_cmds = cp;
        s_ref_cap = ncap;
    }

    uint32_t write_cmd = 0;
    uint32_t i = 0;
    while (i < n) {
        if (!cmd_is_opaque_tri(&fl->cmds[i])) {
            fl->cmds[write_cmd++] = fl->cmds[i++];
            continue;
        }
        uint32_t run_fb = fl->cmds[i].target_addr;
        int sx0 = fl->cmds[i].st.sc_x0, sy0 = fl->cmds[i].st.sc_y0;
        int sx1 = fl->cmds[i].st.sc_x1, sy1 = fl->cmds[i].st.sc_y1;
        uint32_t run_start = i;
        uint32_t run_first_vert = fl->cmds[i].first_vert;
        uint32_t run_verts = fl->cmds[i].vert_count;
        i++;
        while (i < n && cmd_is_opaque_tri(&fl->cmds[i])) {
            const gles_cmd *c = &fl->cmds[i];
            if (c->target_addr != run_fb ||
                c->st.sc_x0 != sx0 || c->st.sc_y0 != sy0 ||
                c->st.sc_x1 != sx1 || c->st.sc_y1 != sy1 ||
                c->first_vert != run_first_vert + run_verts) {
                break;
            }
            run_verts += c->vert_count;
            i++;
        }
        uint32_t run_len = i - run_start;
        if (run_len < 4u) {
            for (uint32_t k = run_start; k < i; k++) fl->cmds[write_cmd++] = fl->cmds[k];
            continue;
        }

        for (uint32_t k = 0; k < run_len; k++) {
            s_refs[k].cmd = &fl->cmds[run_start + k];
            s_refs[k].orig_idx = k;
        }
        qsort(s_refs, run_len, sizeof(s_refs[0]), sort_cmd_ref_cmp);

        int reordered = 0;
        for (uint32_t k = 0; k < run_len; k++) {
            if (s_refs[k].orig_idx != k) { reordered = 1; break; }
        }

        if (reordered) {
            if (run_verts > s_vert_cap) {
                uint32_t vcap = s_vert_cap ? s_vert_cap * 2u : 65536u;
                while (vcap < run_verts) vcap *= 2u;
                gles_vert *vp = (gles_vert *)realloc(s_verts, (size_t)vcap * sizeof(*vp));
                if (!vp) {
                    for (uint32_t k = run_start; k < i; k++) fl->cmds[write_cmd++] = fl->cmds[k];
                    continue;
                }
                s_verts = vp;
                s_vert_cap = vcap;
            }
            uint32_t voff = 0;
            for (uint32_t k = 0; k < run_len; k++) {
                s_cmds[k] = *s_refs[k].cmd;
                uint32_t vc = s_cmds[k].vert_count;
                memcpy(s_verts + voff, fl->verts + s_cmds[k].first_vert, (size_t)vc * sizeof(gles_vert));
                s_cmds[k].first_vert = run_first_vert + voff;
                voff += vc;
            }
            memcpy(fl->verts + run_first_vert, s_verts, (size_t)run_verts * sizeof(gles_vert));
        } else {
            memcpy(s_cmds, &fl->cmds[run_start], (size_t)run_len * sizeof(gles_cmd));
        }

        /* Merge adjacent commands with identical GL state. */
        fl->cmds[write_cmd++] = s_cmds[0];
        for (uint32_t k = 1; k < run_len; k++) {
            gles_cmd *prev = &fl->cmds[write_cmd - 1u];
            const gles_cmd *cur = &s_cmds[k];
            if (prev->first_vert + prev->vert_count == cur->first_vert &&
                prev->data_hash == cur->data_hash &&
                prev->clut_hash == cur->clut_hash &&
                prev->clut_offset == cur->clut_offset &&
                memcmp(&prev->st, &cur->st, sizeof(psp_draw_state)) == 0) {
                prev->vert_count += cur->vert_count;
            } else {
                fl->cmds[write_cmd++] = *cur;
            }
        }
    }
    fl->cmd_count = write_cmd;
}

static inline int cmd_can_multitex(const gles_cmd *c) {
#ifdef PSP_RENDER_HAS_DRAW_RAW
    if (c->is_raw) return 0;
#endif
    return !c->st.clear && c->gl_prim == GL_TRIANGLES && c->st.tex_enable && c->st.tex_addr &&
           (c->st.tex_addr & 0x1FFFFFFFu) >= 0x08000000u &&
           !(c->st.tex_fmt == PSP_TPSM_8888 && !c->st.tex_swizzle &&
             c->st.tex_w == 512 && c->st.tex_h == 512 && c->st.tex_bw == 512);
}

static inline int cmd_non_tex_eq(const psp_draw_state *a, const psp_draw_state *b) {
    return a->gouraud == b->gouraud &&
           a->tex_func == b->tex_func && a->tex_alpha == b->tex_alpha &&
           a->tex_double == b->tex_double && a->tex_env == b->tex_env &&
           a->blend == b->blend && a->blend_src == b->blend_src &&
           a->blend_dst == b->blend_dst && a->blend_op == b->blend_op &&
           a->blend_sfix == b->blend_sfix && a->blend_dfix == b->blend_dfix &&
           a->atest == b->atest && a->atest_func == b->atest_func &&
           a->atest_ref == b->atest_ref && a->atest_mask == b->atest_mask &&
           a->sc_x0 == b->sc_x0 && a->sc_y0 == b->sc_y0 &&
           a->sc_x1 == b->sc_x1 && a->sc_y1 == b->sc_y1 &&
           a->ztest == b->ztest && a->zfunc == b->zfunc && a->zwrite == b->zwrite &&
           a->mask_rgb == b->mask_rgb && a->mask_alpha == b->mask_alpha &&
           a->cull_enable == b->cull_enable && a->cull_dir == b->cull_dir &&
           a->stencil_enable == b->stencil_enable && a->stencil_func == b->stencil_func &&
           a->stencil_ref == b->stencil_ref && a->stencil_mask == b->stencil_mask &&
           a->stencil_sfail == b->stencil_sfail && a->stencil_zfail == b->stencil_zfail &&
           a->stencil_zpass == b->stencil_zpass &&
           a->fog_enable == b->fog_enable && a->fog_far == b->fog_far &&
           a->fog_dist == b->fog_dist && a->fog_color == b->fog_color;
}

static inline void fill_mt_slot(gles_mt_slot *s, const gles_cmd *c) {
    s->tex_addr = c->st.tex_addr;
    s->tex_w = c->st.tex_w;
    s->tex_h = c->st.tex_h;
    s->tex_bw = c->st.tex_bw;
    s->tex_fmt = c->st.tex_fmt;
    s->tex_swizzle = c->st.tex_swizzle;
    s->tex_clamp_u = c->st.tex_clamp_u;
    s->tex_clamp_v = c->st.tex_clamp_v;
    s->clut_offset = c->clut_offset;
    s->clut_len = c->clut_len;
    s->clut_hash = c->clut_hash;
    s->data_hash = c->data_hash;
    s->clut_fmt = c->st.clut_fmt;
    s->clut_shift = c->st.clut_shift;
    s->clut_mask = c->st.clut_mask;
    s->clut_start = c->st.clut_start;
}

static inline int mt_slot_eq(const gles_mt_slot *a, const gles_mt_slot *b) {
    return a->tex_addr == b->tex_addr && a->tex_w == b->tex_w && a->tex_h == b->tex_h &&
           a->tex_bw == b->tex_bw && a->tex_fmt == b->tex_fmt && a->tex_swizzle == b->tex_swizzle &&
           a->tex_clamp_u == b->tex_clamp_u && a->tex_clamp_v == b->tex_clamp_v &&
           a->clut_hash == b->clut_hash && a->data_hash == b->data_hash &&
           a->clut_fmt == b->clut_fmt && a->clut_shift == b->clut_shift &&
           a->clut_mask == b->clut_mask && a->clut_start == b->clut_start;
}

static void stage_batch_multitex(gles_frame_list *fl) {
    static int mt_en = -1;
    if (mt_en < 0) mt_en = (getenv("MC3_NO_MULTI_TEX") == NULL);
    if (!mt_en || fl->cmd_count < 2u) return;

    uint32_t n = fl->cmd_count;
    uint32_t write_cmd = 0;
    uint32_t i = 0;
    while (i < n) {
        gles_cmd base = fl->cmds[i];
        i++;
        if (!cmd_can_multitex(&base)) {
            fl->cmds[write_cmd++] = base;
            continue;
        }
        gles_mt_slot slots[8];
        fill_mt_slot(&slots[0], &base);
        uint8_t slot_cnt = 1;

        while (i < n) {
            const gles_cmd *cur = &fl->cmds[i];
            if (!cmd_can_multitex(cur) ||
                cur->target_addr != base.target_addr ||
                base.first_vert + base.vert_count != cur->first_vert ||
                !cmd_non_tex_eq(&base.st, &cur->st)) {
                break;
            }
            gles_mt_slot cs;
            fill_mt_slot(&cs, cur);
            int found = -1;
            for (int s = 0; s < slot_cnt; s++) {
                if (mt_slot_eq(&slots[s], &cs)) { found = s; break; }
            }
            if (found < 0) {
                if (slot_cnt >= 8) break;
                found = slot_cnt++;
                slots[found] = cs;
            }
            if (found > 0) {
                float bias = (float)found * 4.0f;
                gles_vert *vp = &fl->verts[cur->first_vert];
                uint32_t vc = cur->vert_count;
                for (uint32_t v = 0; v < vc; v++) vp[v].fog += bias;
            }
            base.vert_count += cur->vert_count;
            i++;
        }

        if (slot_cnt > 1) {
            if (fl->mt_count + slot_cnt > fl->mt_cap) {
                uint32_t ncap = fl->mt_cap ? fl->mt_cap * 2u : 1024u;
                while (ncap < fl->mt_count + slot_cnt) ncap *= 2u;
                gles_mt_slot *mp = (gles_mt_slot *)realloc(fl->mt_slots, (size_t)ncap * sizeof(*mp));
                if (mp) {
                    fl->mt_slots = mp;
                    fl->mt_cap = ncap;
                }
            }
            if (fl->mt_count + slot_cnt <= fl->mt_cap && fl->mt_count <= 65535u) {
                base.mt_start = (uint16_t)fl->mt_count;
                base.mt_count = slot_cnt;
                memcpy(&fl->mt_slots[fl->mt_count], slots, (size_t)slot_cnt * sizeof(gles_mt_slot));
                fl->mt_count += slot_cnt;
            }
        }
        fl->cmds[write_cmd++] = base;
    }
    fl->cmd_count = write_cmd;
}

static void gles_present(void) {
    stage_worker_drain();
    if (getenv("MC3_SORT_BATCH")) {
        stage_sort_batch_opaque(&g_stage);
        stage_batch_multitex(&g_stage);
    }
    g_aabb_gen++;
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
        g_stage.fmv_valid = 0;
        g_stage.fmv_tex_addr = 0;
        g_front_has_cmds = 1;
        g_front_ready = 1;
        g_front_dirty = 1;
        pthread_mutex_unlock(&g_vram_mutex);
        g_stage.cmd_count = 0;
        g_stage.vert_count = 0;
        g_stage.clut_bytes = 0;
        g_stage.mt_count = 0;
#ifdef PSP_RENDER_HAS_DRAW_RAW
        g_stage.raw_vbytes = 0;
        g_stage.raw_cbytes = 0;
        g_stage.raw_ibytes = 0;
        g_stage.xform_count = 0;
#endif
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
                    uint64_t rem = next_present_us - now_us;
                    if (rem > 2000ULL) {
                        psp_threadman_yield();
                    } else if (rem > 250ULL) {
                        psp_threadman_sleep_us(rem - 150ULL);
                    } else {
                        psp_threadman_yield_now();
                        break;
                    }
                    yielded = 1;
                    now_us = gles_now_us();
                }
                if (!yielded) psp_threadman_yield_now();
                pace_wait_ms = (double)(now_us - w0) / 1000.0;
                next_present_us += 16667ULL;
                if (next_present_us < now_us) {
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

static int g_gpu_prof_en = -1;
static uint64_t g_gp_vbo_us = 0, g_gp_draw_us = 0, g_gp_end_us = 0;

static void execute_ge_frame(const gles_frame_list *fl) {
    if (g_gpu_prof_en < 0) g_gpu_prof_en = (getenv("MC3_GPU_PROF") != NULL);
    uint64_t gp_t0 = g_gpu_prof_en ? gles_now_us() : 0;
    glViewport(0, 0, g_fbo_w, g_fbo_h);
    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_CULL_FACE);

    const ge_prog_locs *P = &g_ge_unlit;
    glUseProgram(P->prog);
    g_ge_vbo_idx = (g_ge_vbo_idx + 1) % GE_VBO_RING;
    GLuint cur_ge_vbo = g_ge_vbo[g_ge_vbo_idx];
    glBindBuffer(GL_ARRAY_BUFFER, cur_ge_vbo);
    GLsizeiptr vbo_bytes = (GLsizeiptr)(fl->vert_count * sizeof(gles_vert));
    glBufferData(GL_ARRAY_BUFFER, vbo_bytes, NULL, GL_STREAM_DRAW);
    if (vbo_bytes > 0) {
        glBufferSubData(GL_ARRAY_BUFFER, 0, vbo_bytes, fl->verts);
    }
#ifdef PSP_RENDER_HAS_DRAW_RAW
    if (fl->raw_vbytes > 0) {
        glBindBuffer(GL_ARRAY_BUFFER, g_ge_raw_vbo);
        glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)fl->raw_vbytes, fl->raw_vbuf, GL_STREAM_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, cur_ge_vbo);
    }
    if (fl->raw_cbytes > 0) {
        glBindBuffer(GL_ARRAY_BUFFER, g_ge_raw_cbo);
        glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)fl->raw_cbytes, fl->raw_cbuf, GL_STREAM_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, cur_ge_vbo);
    }
    if (fl->raw_ibytes > 0) {
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_ge_raw_ibo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, (GLsizeiptr)fl->raw_ibytes, fl->raw_ibuf, GL_STREAM_DRAW);
    }
    int cur_raw_mode = 0;
    uint32_t cur_xform_idx = UINT32_MAX;
    uint32_t cur_raw_voff = UINT32_MAX, cur_raw_vtype = 0, cur_raw_coff = UINT32_MAX, cur_raw_mat = 0;
    int cur_raw_stride = 0;
    if (P->u_raw_mode >= 0) glUniform1i(P->u_raw_mode, 0);
#endif

    glEnableVertexAttribArray((GLuint)P->loc_pos);
    glEnableVertexAttribArray((GLuint)P->loc_col);
    glEnableVertexAttribArray((GLuint)P->loc_uv);
    glVertexAttribPointer((GLuint)P->loc_pos, 3, GL_FLOAT, GL_FALSE,
                          sizeof(gles_vert), (void *)0);
    glVertexAttribPointer((GLuint)P->loc_col, 4, GL_UNSIGNED_BYTE, GL_TRUE,
                          sizeof(gles_vert), (void *)(3 * sizeof(float)));
    glVertexAttribPointer((GLuint)P->loc_uv, 3, GL_FLOAT, GL_FALSE,
                          sizeof(gles_vert), (void *)(4 * sizeof(float)));
    if (P->loc_fog >= 0) {
        glEnableVertexAttribArray((GLuint)P->loc_fog);
        glVertexAttribPointer((GLuint)P->loc_fog, 1, GL_FLOAT, GL_FALSE,
                              sizeof(gles_vert), (void *)(7 * sizeof(float)));
    }
    uint64_t gp_t1 = g_gpu_prof_en ? gles_now_us() : 0;

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
    int fmv_uploaded = 0;

    int c_sx0 = -1, c_sy0 = -1, c_sx1 = -1, c_sy1 = -1;
    int c_cm = -1, c_cull = -1, c_cull_dir = -1;
    int c_ztest = -1, c_zfunc = -1, c_zwrite = -1;
    int c_sten = -1, c_fog_en = -1, c_bl_en = -1;
    int c_clear_u = -1, c_atest = -1, c_tex_en = -1, c_dclut = -1;
    int c_tex_func = -1, c_tex_alpha = -1, c_tex_double = -1;
    uint32_t c_tex_env = UINT32_MAX;
    int c_atest_func = -1, c_atest_ref = -1, c_atest_mask = -1;
    float c_fog_far = -1e30f, c_fog_dist = -1e30f;
    uint32_t c_fog_color = UINT32_MAX;
    float c_fbo_sx = -1.0f, c_fbo_sy = -1.0f, c_fbo_ox = -1e30f, c_fbo_oy = -1e30f;
    static const GLint g_tex_units[GE_MT_MAX] = { 0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    GLuint c_tex[GE_MT_MAX];
    for (int u = 0; u < GE_MT_MAX; u++) c_tex[u] = UINT32_MAX;
    glEnable(GL_SCISSOR_TEST);
    for (int u = 0; u < GE_MT_MAX; u++) {
        if (P->u_tex[u] >= 0) glUniform1i(P->u_tex[u], g_tex_units[u]);
    }
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
#ifdef PSP_RENDER_HAS_DRAW_RAW
        const ge_prog_locs *want_P = (cmd->is_raw && !st->clear && fl->xforms[cmd->xform_idx].lt_enable)
                                   ? &g_ge_lit : &g_ge_unlit;
        if (want_P != P) {
            P = want_P;
            glUseProgram(P->prog);
            for (int u = 0; u < GE_MT_MAX; u++) {
                if (P->u_tex[u] >= 0) glUniform1i(P->u_tex[u], g_tex_units[u]);
            }
            c_clear_u = -1; c_atest = -1; c_tex_en = -1; c_dclut = -1; c_fog_en = -1;
            c_tex_func = -1; c_tex_alpha = -1; c_tex_double = -1; c_tex_env = UINT32_MAX;
            c_atest_func = -1; c_atest_ref = -1; c_atest_mask = -1;
            c_fog_far = -1e30f; c_fog_dist = -1e30f; c_fog_color = UINT32_MAX;
            c_fbo_sx = -1.0f; c_fbo_sy = -1.0f; c_fbo_ox = -1e30f; c_fbo_oy = -1e30f;
            cur_raw_mode = -1;
            cur_xform_idx = UINT32_MAX;
            cur_raw_voff = UINT32_MAX;
        }
#endif
        static uint32_t g_last_z_off = UINT32_MAX;
        static GLuint   g_last_z_tex = 0;

        /* Scissor (PSP inclusive coords) */
        int sx0 = st->sc_x0 < 0 ? 0 : st->sc_x0;
        int sy0 = st->sc_y0 < 0 ? 0 : st->sc_y0;
        int sx1 = st->sc_x1 > 479 ? 479 : st->sc_x1;
        int sy1 = st->sc_y1 > 271 ? 271 : st->sc_y1;
        if (sx1 < sx0 || sy1 < sy0) continue;
        if (sx0 != c_sx0 || sy0 != c_sy0 || sx1 != c_sx1 || sy1 != c_sy1) {
            int gx0 = (sx0 * g_fbo_w) / 480;
            int gy0 = (sy0 * g_fbo_h) / 272;
            int gx1 = ((sx1 + 1) * g_fbo_w + 479) / 480;
            int gy1 = ((sy1 + 1) * g_fbo_h + 271) / 272;
            if (gx1 > g_fbo_w) gx1 = g_fbo_w;
            if (gy1 > g_fbo_h) gy1 = g_fbo_h;
            glScissor(gx0, gy0, gx1 - gx0, gy1 - gy0);
            c_sx0 = sx0; c_sy0 = sy0; c_sx1 = sx1; c_sy1 = sy1;
        }

        if (st->clear) {
            if (c_bl_en != 0) { glDisable(GL_BLEND); c_bl_en = 0; }
            if (c_cull != 0) { glDisable(GL_CULL_FACE); c_cull = 0; }
            int cm_val = (st->clear_color ? 7 : 0) | (st->clear_alpha ? 8 : 0);
            if (cm_val != c_cm) {
                glColorMask(st->clear_color ? GL_TRUE : GL_FALSE,
                            st->clear_color ? GL_TRUE : GL_FALSE,
                            st->clear_color ? GL_TRUE : GL_FALSE,
                            st->clear_alpha ? GL_TRUE : GL_FALSE);
                c_cm = cm_val;
            }
            if (st->clear_depth) {
                if (c_ztest != 1) { glEnable(GL_DEPTH_TEST); c_ztest = 1; }
                if (c_zfunc != 1) { glDepthFunc(GL_ALWAYS); c_zfunc = 1; }
                if (c_zwrite != 1) { glDepthMask(GL_TRUE); c_zwrite = 1; }
            } else {
                if (c_ztest != 0) { glDisable(GL_DEPTH_TEST); c_ztest = 0; }
                if (c_zwrite != 0) { glDepthMask(GL_FALSE); c_zwrite = 0; }
            }
            if (c_clear_u != 1) { glUniform1i(P->u_clear, 1); c_clear_u = 1; }
            if (c_tex_en != 0) { glUniform1i(P->u_tex_enable, 0); c_tex_en = 0; }
            if (c_atest != 0) { glUniform1i(P->u_atest, 0); c_atest = 0; }
            if (c_fbo_sx != 1.0f || c_fbo_sy != 1.0f) {
                glUniform2f(P->u_fbo_uv_scale, 1.0f, 1.0f);
                c_fbo_sx = 1.0f; c_fbo_sy = 1.0f;
            }
            if (c_fbo_ox != 0.0f || c_fbo_oy != 0.0f) {
                glUniform2f(P->u_fbo_uv_offset, 0.0f, 0.0f);
                c_fbo_ox = 0.0f; c_fbo_oy = 0.0f;
            }
            if (c_dclut != 0) { glUniform1i(P->u_depth_clut_mode, 0); c_dclut = 0; }
#ifdef PSP_DRAW_STATE_HAS_STENCIL_FOG
            if (c_sten != 0) { glDisable(GL_STENCIL_TEST); c_sten = 0; }
            if (st->clear_alpha) {
                glStencilMask(0xFF);
                glClearStencil(0);
                glClear(GL_STENCIL_BUFFER_BIT);
            }
            if (c_fog_en != 0) { glUniform1i(P->u_fog_enable, 0); c_fog_en = 0; }
#endif
        } else {
      #ifdef PSP_DRAW_STATE_HAS_MASK_CULL
            int cm_val = ((st->mask_rgb & 0xFFu) != 0xFFu ? 1 : 0) |
                         (((st->mask_rgb >> 8) & 0xFFu) != 0xFFu ? 2 : 0) |
                         (((st->mask_rgb >> 16) & 0xFFu) != 0xFFu ? 4 : 0) |
                         ((st->mask_alpha & 0xFFu) != 0xFFu ? 8 : 0);
            if (cm_val != c_cm) {
                glColorMask((cm_val & 1) != 0, (cm_val & 2) != 0, (cm_val & 4) != 0, (cm_val & 8) != 0);
                c_cm = cm_val;
            }
            if (st->cull_enable) {
                if (c_cull != 1) { glEnable(GL_CULL_FACE); c_cull = 1; }
                int cd = st->cull_dir ? 1 : 0;
                if (cd != c_cull_dir) {
                    glCullFace(cd ? GL_BACK : GL_FRONT);
                    c_cull_dir = cd;
                }
            } else if (c_cull != 0) {
                glDisable(GL_CULL_FACE);
                c_cull = 0;
            }
            #else
            if (c_cm != 15) { glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE); c_cm = 15; }
            #endif
            if (st->ztest) {
                if (c_ztest != 1) { glEnable(GL_DEPTH_TEST); c_ztest = 1; }
                if (st->zfunc != c_zfunc) { glDepthFunc(map_depth_func(st->zfunc)); c_zfunc = st->zfunc; }
            } else if (c_ztest != 0) {
                glDisable(GL_DEPTH_TEST);
                c_ztest = 0;
            }
            int zw = st->zwrite ? 1 : 0;
            if (zw != c_zwrite) { glDepthMask(zw ? GL_TRUE : GL_FALSE); c_zwrite = zw; }
#ifdef PSP_DRAW_STATE_HAS_STENCIL_FOG
            if (st->stencil_enable) {
                static const GLenum s_ops[6] = {
                    GL_KEEP, GL_ZERO, GL_REPLACE, GL_INVERT, GL_INCR, GL_DECR
                };
                if (c_sten != 1) { glEnable(GL_STENCIL_TEST); c_sten = 1; }
                glStencilMask(~st->mask_alpha & 0xFFu);
                glStencilFunc(map_depth_func(st->stencil_func), (GLint)st->stencil_ref, (GLuint)st->stencil_mask);
                glStencilOp(s_ops[st->stencil_sfail < 6 ? st->stencil_sfail : 0],
                            s_ops[st->stencil_zfail < 6 ? st->stencil_zfail : 0],
                            s_ops[st->stencil_zpass < 6 ? st->stencil_zpass : 0]);
            } else if (c_sten != 0) {
                glDisable(GL_STENCIL_TEST);
                glStencilMask(0);
                c_sten = 0;
            }
            int fe = st->fog_enable ? 1 : 0;
            if (fe != c_fog_en) { glUniform1i(P->u_fog_enable, fe); c_fog_en = fe; }
            if (fe) {
                if (st->fog_far != c_fog_far) { glUniform1f(P->u_fog_far, st->fog_far); c_fog_far = st->fog_far; }
                if (st->fog_dist != c_fog_dist) { glUniform1f(P->u_fog_dist, st->fog_dist); c_fog_dist = st->fog_dist; }
                if (st->fog_color != c_fog_color) {
                    glUniform3f(P->u_fog_color,
                                (float)(st->fog_color & 0xFFu) / 255.0f,
                                (float)((st->fog_color >> 8) & 0xFFu) / 255.0f,
                                (float)((st->fog_color >> 16) & 0xFFu) / 255.0f);
                    c_fog_color = st->fog_color;
                }
            }
#endif

            if (st->blend) {
                GLenum src_f = map_blend_factor(st->blend_src, 1);
                GLenum dst_f = map_blend_factor(st->blend_dst, 0);
                if (c_bl_en != 1) { glEnable(GL_BLEND); c_bl_en = 1; }
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
            } else if (c_bl_en != 0) {
                glDisable(GL_BLEND);
                c_bl_en = 0;
            }

            if (c_clear_u != 0) { glUniform1i(P->u_clear, 0); c_clear_u = 0; }
            int ae = st->atest ? 1 : 0;
            if (ae != c_atest) { glUniform1i(P->u_atest, ae); c_atest = ae; }
            if (ae) {
                if (st->atest_func != c_atest_func) { glUniform1i(P->u_atest_func, st->atest_func); c_atest_func = st->atest_func; }
                if (st->atest_ref != c_atest_ref)   { glUniform1i(P->u_atest_ref, st->atest_ref); c_atest_ref = st->atest_ref; }
                if (st->atest_mask != c_atest_mask) { glUniform1i(P->u_atest_mask, st->atest_mask); c_atest_mask = st->atest_mask; }
            }

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
                    glUniform2f(P->u_fbo_uv_scale, sx, sy);
                    glUniform2f(P->u_fbo_uv_offset, 0.0f, 0.0f);
                    c_fbo_sx = sx; c_fbo_sy = sy; c_fbo_ox = 0.0f; c_fbo_oy = 0.0f;
                    if (c_dclut != 1) { glUniform1i(P->u_depth_clut_mode, 1); c_dclut = 1; }
                    glUniform1i(P->u_depth_tex, 1);
                    glUniform1i(P->u_clut_tex, 2);
                    glUniform1i(P->u_clut_shift, st->clut_shift);
                    glUniform1i(P->u_clut_mask, st->clut_mask);
                    glUniform1i(P->u_clut_start, st->clut_start);
                } else if (fbo_src && fbo_src->has_ge_draws) {
                    glActiveTexture(GL_TEXTURE0);
                    if (fbo_src == cur_fbo) {
                        if (!g_fbo_scratch_tex) {
                            glGenTextures(1, &g_fbo_scratch_tex);
                            glBindTexture(GL_TEXTURE_2D, g_fbo_scratch_tex);
                            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, g_fbo_w, g_fbo_h, 0,
                                         GL_RGBA, GL_UNSIGNED_BYTE, NULL);
                        } else {
                            glBindTexture(GL_TEXTURE_2D, g_fbo_scratch_tex);
                        }
                        c_tex[0] = g_fbo_scratch_tex;
                        glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, g_fbo_w, g_fbo_h);
                    } else {
                        glBindTexture(GL_TEXTURE_2D, fbo_src->tex);
                        c_tex[0] = fbo_src->tex;
                    }
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
                    float sx = (st->tex_w > 0) ? ((float)st->tex_w / 480.0f) : 1.0f;
                    float sy = (st->tex_h > 0) ? ((float)st->tex_h / 272.0f) : 1.0f;
                    if (sx != c_fbo_sx || sy != c_fbo_sy) {
                        glUniform2f(P->u_fbo_uv_scale, sx, sy);
                        c_fbo_sx = sx; c_fbo_sy = sy;
                    }
                    if (uv_off_x != c_fbo_ox || uv_off_y != c_fbo_oy) {
                        glUniform2f(P->u_fbo_uv_offset, uv_off_x, uv_off_y);
                        c_fbo_ox = uv_off_x; c_fbo_oy = uv_off_y;
                    }
                    if (c_dclut != 0) { glUniform1i(P->u_depth_clut_mode, 0); c_dclut = 0; }
                } else {
                    if (cmd->mt_count > 1) {
                        for (uint8_t s = 0; s < cmd->mt_count; s++) {
                            const gles_mt_slot *ms = &fl->mt_slots[cmd->mt_start + s];
                            const uint8_t *sc = (ms->clut_offset != UINT32_MAX && fl->cluts)
                                              ? (fl->cluts + ms->clut_offset) : NULL;
                            glActiveTexture(GL_TEXTURE0 + g_tex_units[s]);
                            tex_cache_entry *te = get_cached_texture_slot(ms, sc, &c_tex[s]);
                            if (te->tex != c_tex[s]) {
                                glBindTexture(GL_TEXTURE_2D, te->tex);
                                c_tex[s] = te->tex;
                            }
                            int8_t ws = ms->tex_clamp_u ? 1 : 0;
                            int8_t wt = ms->tex_clamp_v ? 1 : 0;
                            if (te->wrap_s != ws) {
                                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, ws ? GL_CLAMP_TO_EDGE : GL_REPEAT);
                                te->wrap_s = ws;
                            }
                            if (te->wrap_t != wt) {
                                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wt ? GL_CLAMP_TO_EDGE : GL_REPEAT);
                                te->wrap_t = wt;
                            }
                        }
                        glActiveTexture(GL_TEXTURE0);
                    } else {
                        glActiveTexture(GL_TEXTURE0);
                        if (fl->fmv_valid && fl->fmv_buf &&
                            st->tex_addr == fl->fmv_tex_addr &&
                            st->tex_fmt == PSP_TPSM_8888 &&
                            st->tex_w == 512 && st->tex_h == 512) {
                            if (!g_fmv_tex) {
                                glGenTextures(1, &g_fmv_tex);
                                glBindTexture(GL_TEXTURE_2D, g_fmv_tex);
                                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 512, 512, 0,
                                             GL_RGBA, GL_UNSIGNED_BYTE, NULL);
                            } else {
                                glBindTexture(GL_TEXTURE_2D, g_fmv_tex);
                            }
                            if (!fmv_uploaded) {
                                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 512, 272,
                                                GL_RGBA, GL_UNSIGNED_BYTE, fl->fmv_buf);
                                fmv_uploaded = 1;
                            }
                            c_tex[0] = g_fmv_tex;
                            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,
                                            st->tex_clamp_u ? GL_CLAMP_TO_EDGE : GL_REPEAT);
                            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,
                                            st->tex_clamp_v ? GL_CLAMP_TO_EDGE : GL_REPEAT);
                        } else {
                            gles_mt_slot single_ms;
                            fill_mt_slot(&single_ms, cmd);
                            tex_cache_entry *te = get_cached_texture_slot(&single_ms, clut_ptr, &c_tex[0]);
                            if (te->tex != c_tex[0]) {
                                glBindTexture(GL_TEXTURE_2D, te->tex);
                                c_tex[0] = te->tex;
                            }
                            int8_t ws = st->tex_clamp_u ? 1 : 0;
                            int8_t wt = st->tex_clamp_v ? 1 : 0;
                            if (te->wrap_s != ws) {
                                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, ws ? GL_CLAMP_TO_EDGE : GL_REPEAT);
                                te->wrap_s = ws;
                            }
                            if (te->wrap_t != wt) {
                                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wt ? GL_CLAMP_TO_EDGE : GL_REPEAT);
                                te->wrap_t = wt;
                            }
                        }
                    }
                    if (c_fbo_sx != 1.0f || c_fbo_sy != 1.0f) {
                        glUniform2f(P->u_fbo_uv_scale, 1.0f, 1.0f);
                        c_fbo_sx = 1.0f; c_fbo_sy = 1.0f;
                    }
                    if (c_fbo_ox != 0.0f || c_fbo_oy != 0.0f) {
                        glUniform2f(P->u_fbo_uv_offset, 0.0f, 0.0f);
                        c_fbo_ox = 0.0f; c_fbo_oy = 0.0f;
                    }
                    if (c_dclut != 0) { glUniform1i(P->u_depth_clut_mode, 0); c_dclut = 0; }
                }
                if (c_tex_en != 1) { glUniform1i(P->u_tex_enable, 1); c_tex_en = 1; }
                if (st->tex_func != c_tex_func) { glUniform1i(P->u_tex_func, st->tex_func); c_tex_func = st->tex_func; }
                int ta = st->tex_alpha ? 1 : 0;
                if (ta != c_tex_alpha) { glUniform1i(P->u_tex_alpha, ta); c_tex_alpha = ta; }
                int td = st->tex_double ? 1 : 0;
                if (td != c_tex_double) { glUniform1i(P->u_tex_double, td); c_tex_double = td; }
                if (st->tex_env != c_tex_env) {
                    glUniform3f(P->u_tex_env,
                                (float)(st->tex_env & 0xFFu) / 255.0f,
                                (float)((st->tex_env >> 8) & 0xFFu) / 255.0f,
                                (float)((st->tex_env >> 16) & 0xFFu) / 255.0f);
                    c_tex_env = st->tex_env;
                }
            } else {
                if (c_tex_en != 0) { glUniform1i(P->u_tex_enable, 0); c_tex_en = 0; }
                if (c_dclut != 0) { glUniform1i(P->u_depth_clut_mode, 0); c_dclut = 0; }
            }
        }

        if (((st->clear && st->clear_depth) || (!st->clear && st->ztest && st->zwrite)) && st->zaddr != 0u) {
            g_last_z_off = st->zaddr & 0x001FFFFFu;
            g_last_z_tex = cur_fbo->depth_tex;
        }
        cur_fbo->has_ge_draws = 1;
#ifdef PSP_RENDER_HAS_DRAW_RAW
        if (cmd->is_raw) {
            if (cur_raw_mode != 1) {
                if (P->u_raw_mode >= 0) glUniform1i(P->u_raw_mode, 1);
                cur_raw_mode = 1;
                glBindBuffer(GL_ARRAY_BUFFER, g_ge_raw_vbo);
                if (P->loc_fog >= 0) glDisableVertexAttribArray((GLuint)P->loc_fog);
                cur_raw_voff = UINT32_MAX;
            }
            const psp_xform_state *xf = &fl->xforms[cmd->xform_idx];
            uint32_t vt = cmd->raw_vtype;
            int pfmt = (int)((vt >> 7) & 3u);
            int cfmt = (int)((vt >> 2) & 7u);
            int nfmt = (int)((vt >> 5) & 3u);

            if (cmd->raw_voff != cur_raw_voff || vt != cur_raw_vtype ||
                cmd->raw_stride != cur_raw_stride || cmd->raw_col_off != cur_raw_coff) {
                int to, co, no, po;
                raw_vertex_offsets(vt, &to, &co, &no, &po);

                GLenum ptype = (pfmt == 1) ? GL_BYTE : (pfmt == 2) ? GL_SHORT : GL_FLOAT;
                glEnableVertexAttribArray((GLuint)P->loc_pos);
                glVertexAttribPointer((GLuint)P->loc_pos, 3, ptype, GL_FALSE, cmd->raw_stride,
                                      (const void *)(uintptr_t)(cmd->raw_voff + (uint32_t)po));

                if (cfmt == 7) {
                    glEnableVertexAttribArray((GLuint)P->loc_col);
                    glVertexAttribPointer((GLuint)P->loc_col, 4, GL_UNSIGNED_BYTE, GL_TRUE, cmd->raw_stride,
                                          (const void *)(uintptr_t)(cmd->raw_voff + (uint32_t)co));
                } else if (cfmt >= 4 && cfmt <= 6 && cmd->raw_col_off != UINT32_MAX) {
                    glEnableVertexAttribArray((GLuint)P->loc_col);
                    glBindBuffer(GL_ARRAY_BUFFER, g_ge_raw_cbo);
                    glVertexAttribPointer((GLuint)P->loc_col, 4, GL_UNSIGNED_BYTE, GL_TRUE, cmd->raw_stride,
                                          (const void *)(uintptr_t)cmd->raw_col_off);
                    glBindBuffer(GL_ARRAY_BUFFER, g_ge_raw_vbo);
                } else {
                    glDisableVertexAttribArray((GLuint)P->loc_col);
                    cur_raw_mat = ~xf->material;
                }

                int tfmt = (int)(vt & 3u);
                if (tfmt != 0) {
                    GLenum ttype = (tfmt == 1) ? GL_UNSIGNED_BYTE : (tfmt == 2) ? GL_UNSIGNED_SHORT : GL_FLOAT;
                    glEnableVertexAttribArray((GLuint)P->loc_uv);
                    glVertexAttribPointer((GLuint)P->loc_uv, 2, ttype, GL_FALSE, cmd->raw_stride,
                                          (const void *)(uintptr_t)(cmd->raw_voff + (uint32_t)to));
                } else {
                    glDisableVertexAttribArray((GLuint)P->loc_uv);
                    glVertexAttrib3f((GLuint)P->loc_uv, 0.0f, 0.0f, 1.0f);
                }

                if (nfmt != 0 && P->loc_nrm >= 0) {
                    GLenum ntype = (nfmt == 1) ? GL_BYTE : (nfmt == 2) ? GL_SHORT : GL_FLOAT;
                    glEnableVertexAttribArray((GLuint)P->loc_nrm);
                    glVertexAttribPointer((GLuint)P->loc_nrm, 3, ntype, GL_FALSE, cmd->raw_stride,
                                          (const void *)(uintptr_t)(cmd->raw_voff + (uint32_t)no));
                } else if (P->loc_nrm >= 0) {
                    glDisableVertexAttribArray((GLuint)P->loc_nrm);
                    glVertexAttrib3f((GLuint)P->loc_nrm, 0.0f, 0.0f, 1.0f);
                }

                cur_raw_voff = cmd->raw_voff;
                cur_raw_vtype = vt;
                cur_raw_stride = cmd->raw_stride;
                cur_raw_coff = cmd->raw_col_off;
            }
            if (cfmt == 0 && xf->material != cur_raw_mat) {
                glVertexAttrib4f((GLuint)P->loc_col,
                                 (float)(xf->material & 0xFFu) / 255.0f,
                                 (float)((xf->material >> 8) & 0xFFu) / 255.0f,
                                 (float)((xf->material >> 16) & 0xFFu) / 255.0f,
                                 (float)((xf->material >> 24) & 0xFFu) / 255.0f);
                cur_raw_mat = xf->material;
            }

            if (cmd->xform_idx != cur_xform_idx) {
                float pscale = (pfmt == 1) ? (1.0f / 128.0f) : (pfmt == 2) ? (1.0f / 32768.0f) : 1.0f;
                float nscale = (nfmt == 1) ? (1.0f / 128.0f) : (nfmt == 2) ? (1.0f / 32768.0f) : 1.0f;
                glUniformMatrix4fv(P->u_mvp, 1, GL_FALSE, xf->ndc_mvp);
                glUniform4fv(P->u_wv_z, 1, xf->wv_z);
                glUniform4f(P->u_uv_xform, xf->u_scale, xf->u_off, xf->v_scale, xf->v_off);
                if (P->u_pos_scale >= 0) glUniform1f(P->u_pos_scale, pscale);
                if (P->u_nrm_scale >= 0) glUniform1f(P->u_nrm_scale, nscale);
                glUniform1i(P->u_tex_map_mode, xf->tex_map_mode);
                glUniform1i(P->u_tex_proj_mode, xf->tex_proj_mode);
                if (xf->lt_enable || xf->tex_map_mode == 2) {
                    glUniformMatrix3fv(P->u_world, 1, GL_FALSE, xf->world);
                    if (P->u_world_t >= 0) glUniform3fv(P->u_world_t, 1, xf->world + 9);
                }
                if (xf->tex_map_mode == 1) {
                    const float *tm = xf->tmat;
                    float tm_rows[12] = {
                        tm[0], tm[3], tm[6], tm[9],
                        tm[1], tm[4], tm[7], tm[10],
                        tm[2], tm[5], tm[8], tm[11]
                    };
                    glUniform4fv(P->u_tmat, 3, tm_rows);
                } else if (xf->tex_map_mode == 2) {
                    glUniform3fv(P->u_env_lu, 1, xf->env_lu);
                    glUniform3fv(P->u_env_lv, 1, xf->env_lv);
                }
                if (xf->lt_enable) {
                    glUniform1i(P->u_cmat, xf->cmat);
                    glUniform1i(P->u_rnorm, xf->rnorm);
                    glUniform1f(P->u_spow, xf->spow);
                    glUniform4f(P->u_mat_emis,
                                (float)(xf->emissive & 0xFFu) / 255.0f,
                                (float)((xf->emissive >> 8) & 0xFFu) / 255.0f,
                                (float)((xf->emissive >> 16) & 0xFFu) / 255.0f,
                                (float)((xf->emissive >> 24) & 0xFFu) / 255.0f);
                    glUniform4f(P->u_mat_amb,
                                (float)(xf->material & 0xFFu) / 255.0f,
                                (float)((xf->material >> 8) & 0xFFu) / 255.0f,
                                (float)((xf->material >> 16) & 0xFFu) / 255.0f,
                                (float)((xf->material >> 24) & 0xFFu) / 255.0f);
                    glUniform4f(P->u_mat_diff,
                                (float)(xf->diffuse & 0xFFu) / 255.0f,
                                (float)((xf->diffuse >> 8) & 0xFFu) / 255.0f,
                                (float)((xf->diffuse >> 16) & 0xFFu) / 255.0f,
                                1.0f);
                    glUniform4f(P->u_mat_spec,
                                (float)(xf->specular & 0xFFu) / 255.0f,
                                (float)((xf->specular >> 8) & 0xFFu) / 255.0f,
                                (float)((xf->specular >> 16) & 0xFFu) / 255.0f,
                                1.0f);
                    glUniform3f(P->u_lt_global_amb,
                                (float)(xf->ambient & 0xFFu) / 255.0f,
                                (float)((xf->ambient >> 8) & 0xFFu) / 255.0f,
                                (float)((xf->ambient >> 16) & 0xFFu) / 255.0f);
                    glUniform4iv(P->u_lt_on, 1, xf->lt_on);
                    glUniform4iv(P->u_lt_type, 1, xf->lt_type);
                    glUniform4iv(P->u_lt_comp, 1, xf->lt_comp);
                    glUniform3fv(P->u_lt_pos, 4, &xf->lt_pos[0][0]);
                    glUniform3fv(P->u_lt_dir, 4, &xf->lt_dir[0][0]);
                    glUniform3fv(P->u_lt_att, 4, &xf->lt_att[0][0]);
                    glUniform4fv(P->u_lt_spot_exp, 1, xf->lt_spot_exp);
                    glUniform4fv(P->u_lt_spot_cut, 1, xf->lt_spot_cut);
                    float la[12], ld[12], ls[12];
                    for (int li = 0; li < 4; li++) {
                        for (int ch = 0; ch < 3; ch++) {
                            la[li * 3 + ch] = (float)((xf->lt_col[li][0] >> (ch * 8)) & 0xFFu) / 255.0f;
                            ld[li * 3 + ch] = (float)((xf->lt_col[li][1] >> (ch * 8)) & 0xFFu) / 255.0f;
                            ls[li * 3 + ch] = (float)((xf->lt_col[li][2] >> (ch * 8)) & 0xFFu) / 255.0f;
                        }
                    }
                    glUniform3fv(P->u_lt_amb, 4, la);
                    glUniform3fv(P->u_lt_diff, 4, ld);
                    glUniform3fv(P->u_lt_spec, 4, ls);
                }
                cur_xform_idx = cmd->xform_idx;
            }

            if (cmd->raw_isz == 0) {
                glDrawArrays(cmd->gl_prim, 0, (GLsizei)cmd->vert_count);
            } else {
                GLenum itype = (cmd->raw_isz == 1) ? GL_UNSIGNED_BYTE
                             : (cmd->raw_isz == 2) ? GL_UNSIGNED_SHORT
                                                   : GL_UNSIGNED_INT;
                glDrawElements(cmd->gl_prim, (GLsizei)cmd->vert_count, itype,
                               (const void *)(uintptr_t)cmd->raw_ioff);
            }
            continue;
        }
        if (cur_raw_mode != 0) {
            if (P->u_raw_mode >= 0) glUniform1i(P->u_raw_mode, 0);
            cur_raw_mode = 0;
            glBindBuffer(GL_ARRAY_BUFFER, cur_ge_vbo);
            glEnableVertexAttribArray((GLuint)P->loc_pos);
            glEnableVertexAttribArray((GLuint)P->loc_col);
            glEnableVertexAttribArray((GLuint)P->loc_uv);
            glVertexAttribPointer((GLuint)P->loc_pos, 3, GL_FLOAT, GL_FALSE,
                                  sizeof(gles_vert), (void *)0);
            glVertexAttribPointer((GLuint)P->loc_col, 4, GL_UNSIGNED_BYTE, GL_TRUE,
                                  sizeof(gles_vert), (void *)(3 * sizeof(float)));
            glVertexAttribPointer((GLuint)P->loc_uv, 3, GL_FLOAT, GL_FALSE,
                                  sizeof(gles_vert), (void *)(4 * sizeof(float)));
            if (P->loc_fog >= 0) {
                glEnableVertexAttribArray((GLuint)P->loc_fog);
                glVertexAttribPointer((GLuint)P->loc_fog, 1, GL_FLOAT, GL_FALSE,
                                      sizeof(gles_vert), (void *)(7 * sizeof(float)));
            }
            if (P->loc_nrm >= 0) glDisableVertexAttribArray((GLuint)P->loc_nrm);
        }
#endif
        glDrawArrays(cmd->gl_prim, (GLint)cmd->first_vert, (GLsizei)cmd->vert_count);
    }

    uint64_t gp_t2 = g_gpu_prof_en ? gles_now_us() : 0;
    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_STENCIL_TEST);
    glDisable(GL_BLEND);
    glDisable(GL_CULL_FACE);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glDepthMask(GL_TRUE);
    glDisableVertexAttribArray((GLuint)P->loc_pos);
    glDisableVertexAttribArray((GLuint)P->loc_col);
    glDisableVertexAttribArray((GLuint)P->loc_uv);
    if (P->loc_fog >= 0) glDisableVertexAttribArray((GLuint)P->loc_fog);
    if (P->loc_nrm >= 0) glDisableVertexAttribArray((GLuint)P->loc_nrm);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    if (g_gpu_prof_en) {
        uint64_t gp_t3 = gles_now_us();
        g_gp_vbo_us  = gp_t1 - gp_t0;
        g_gp_draw_us = (gp_t2 - gp_t1 > g_exec_tex_up_us) ? (gp_t2 - gp_t1 - g_exec_tex_up_us) : 0;
        g_gp_end_us  = gp_t3 - gp_t2;
    }
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
            } else if (ev.type == SDL_CONTROLLERBUTTONDOWN || ev.type == SDL_JOYBUTTONDOWN) {
                /* Diagnostic: record first 32 raw button events when MC3_DEBUG_BUTTONS is set. */
                static int btn_dbg = -1;
                static int btn_log_count = 0;
                if (btn_dbg < 0) btn_dbg = getenv("MC3_DEBUG_BUTTONS") != NULL;
                if (btn_dbg && btn_log_count < 32) {
                    btn_log_count++;
                    FILE *bf = fopen("mc3_buttons.log", "a");
                    if (bf) {
                        if (ev.type == SDL_CONTROLLERBUTTONDOWN)
                            fprintf(bf, "controller button %d (%s)\n", (int)ev.cbutton.button,
                                    SDL_GameControllerGetStringForButton((SDL_GameControllerButton)ev.cbutton.button));
                        else
                            fprintf(bf, "joystick button %d\n", (int)ev.jbutton.button);
                        fclose(bf);
                    }
                }
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
        if (g_joystick && SDL_JoystickNumButtons(g_joystick) >= 14) {
            if (SDL_JoystickGetButton(g_joystick, 12)) pad_buttons |= 0x0001; /* SELECT */
            if (SDL_JoystickGetButton(g_joystick, 13)) pad_buttons |= 0x0008; /* START */
        }
        if ((pad_buttons & 0x0009u) == 0x0009u) {
            psp_exit_requested_set(1);
            return;
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
                g_front.mt_count = 0;
#ifdef PSP_RENDER_HAS_DRAW_RAW
                g_front.raw_vbytes = 0;
                g_front.raw_cbytes = 0;
                g_front.raw_ibytes = 0;
                g_front.xform_count = 0;
#endif
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

        if (!use_fbo_tex && g_viewport_w * 272 < g_viewport_h * 480) {
            int qh = (g_viewport_w * 272) / 480;
            int qy = g_viewport_y + (g_viewport_h - qh) / 2;
            glViewport(g_viewport_x, qy, g_viewport_w, qh);
        } else {
            glViewport(g_viewport_x, g_viewport_y, g_viewport_w, g_viewport_h);
        }
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
        if (g_gpu_prof_en == 1 && dirty && has_cmds && g_exec.cmd_count > 0) {
            static uint64_t gp_last_print = 0, gp_vbo_sum = 0, gp_tex_sum = 0, gp_draw_sum = 0, gp_end_sum = 0;
            static double gp_swap_sum = 0.0;
            static uint64_t gp_cmds_sum = 0, gp_verts_sum = 0;
            static uint32_t gp_frames = 0;
            gp_vbo_sum += g_gp_vbo_us;
            gp_tex_sum += g_exec_tex_up_us;
            gp_draw_sum += g_gp_draw_us;
            gp_end_sum += g_gp_end_us;
            gp_swap_sum += swap_ms;
            gp_cmds_sum += g_exec.cmd_count;
            gp_verts_sum += g_exec.vert_count;
            gp_frames++;
            uint64_t gp_now = gles_now_us();
            if (!gp_last_print) gp_last_print = gp_now;
            else if (gp_now - gp_last_print >= 5000000ULL) {
                double inv_n = 1.0 / (double)gp_frames;
                fprintf(stderr,
                        "gpu_prof: frames=%u vbo=%.2fms tex=%.2fms draw=%.2fms end=%.2fms swap=%.2fms cmds=%u verts=%u\n",
                        gp_frames,
                        (double)gp_vbo_sum * inv_n / 1000.0,
                        (double)gp_tex_sum * inv_n / 1000.0,
                        (double)gp_draw_sum * inv_n / 1000.0,
                        (double)gp_end_sum * inv_n / 1000.0,
                        gp_swap_sum * inv_n,
                        (uint32_t)(gp_cmds_sum / gp_frames),
                        (uint32_t)(gp_verts_sum / gp_frames));
                fflush(stderr);
                gp_last_print = gp_now;
                gp_vbo_sum = gp_tex_sum = gp_draw_sum = gp_end_sum = 0;
                gp_swap_sum = 0.0;
                gp_cmds_sum = gp_verts_sum = 0;
                gp_frames = 0;
            }
        }
        mc3_telemetry_on_gpu_frame(host_ft_ms, gpu_exec_ms, tex_up_ms, swap_ms,
                                   frame_tex_hits, frame_tex_misses,
                                   g_tex_slots_used, TEX_CACHE_SLOTS);
        if (!has_vsync) usleep(16667);
        else if (!dirty && swap_ms < 2.0) usleep(1000);
    }

    fprintf(stderr, "gles32: present loop exiting\n");
    fflush(stderr);
}

/* Non-const so gles_init() can drop draw_raw when MC3_CPU_TL is set. */
psp_render_backend g_gles_be = {
    "gles32",
    gles_init,
    gles_shutdown,
    gles_set_target,
    gles_draw,
    gles_finish,
    gles_present,
#ifdef PSP_RENDER_HAS_DRAW_RAW
    gles_draw_raw,
#endif
};

extern const psp_render_backend psp_render_gles32 __attribute__((alias("g_gles_be")));

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
