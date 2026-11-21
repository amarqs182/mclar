/* mc3 — SDL2 software render backend for Midnight Club: L.A. Remix (USA).
 *
 * Implements the psp_render_backend interface using plain SDL2 (no EGL,
 * no GLES). The GE framebuffer lives in emulated PSP memory (guest VRAM).
 * Each frame we read it back and blit it to an SDL2 window, letterboxed
 * to preserve the PSP's 480x272 aspect ratio within the host window.
 *
 * Build: gcc ... -lSDL2
 */

#include "render_sdl.h"

#include <psprecomp/mem.h>
#include <psprecomp/hle.h>
#include <SDL2/SDL.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>

/* include the global framebuffer state */
#include <psprecomp/render.h>

/* ---- window state ------------------------------------------------- */

static SDL_Window *g_window = NULL;
static SDL_Renderer *g_renderer = NULL;
static SDL_Texture *g_tex = NULL;
static SDL_GameController *g_controller = NULL;
static pthread_mutex_t g_vram_mutex = PTHREAD_MUTEX_INITIALIZER;

/* Window size (host) */
static int g_window_w = 1280;
static int g_window_h = 960;

/* Letterbox: PSP 480x272 content centered in the host window */
static int g_viewport_x = 0;
static int g_viewport_y = 0;
static int g_viewport_w = 960;
static int g_viewport_h = 540;

/* ---- shader-less drawing: we just blit the framebuffer texture ------ */
static void gles_shutdown(void);

static void update_viewport(int w, int h) {
if (w <= 0 || h <= 0) return;
g_window_w = w;
g_window_h = h;
float window_aspect = (float)g_window_w / (float)g_window_h;
float content_aspect = 480.0f / 272.0f;
if (window_aspect > content_aspect) {
g_viewport_h = g_window_h;
g_viewport_w = (int)(g_viewport_h * content_aspect);
g_viewport_x = (g_window_w - g_viewport_w) / 2;
g_viewport_y = 0;
} else {
g_viewport_w = g_window_w;
g_viewport_h = (int)(g_viewport_w / content_aspect);
g_viewport_x = 0;
g_viewport_y = (g_window_h - g_viewport_h) / 2;
}
}

static int gles_init(int w, int h) {
    g_window_w = w;
    g_window_h = h;

    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMECONTROLLER) < 0) {
        if (SDL_Init(SDL_INIT_VIDEO) < 0) {
            fprintf(stderr, "sdl2: SDL_Init failed: %s\n", SDL_GetError());
            return -1;
        }
    }
    for (int i = 0; i < SDL_NumJoysticks(); i++) {
        if (SDL_IsGameController(i)) {
            g_controller = SDL_GameControllerOpen(i);
            if (g_controller) {
                fprintf(stderr, "sdl2: opened game controller: %s\n", SDL_GameControllerName(g_controller));
                break;
            }
        }
    }

    /* Give the compositor a moment to be ready before creating the window,
     * so it is mapped and visible. */
    usleep(500000);  /* 500 ms */

    g_window = SDL_CreateWindow(
        "MIDNIGHT CLUB: L.A. REMIX - PSP Static Recomp",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        g_window_w, g_window_h,
        SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);
    if (!g_window) {
        fprintf(stderr, "sdl2: SDL_CreateWindow failed: %s\n", SDL_GetError());
        fprintf(stderr, "sdl2: Try SDL_VIDEODRIVER=x11 if on Wayland\n");
        gles_shutdown(); return -1;
    }

    /* Force window to the front on Wayland/X11 */
    SDL_RaiseWindow(g_window);
    SDL_SetWindowInputFocus(g_window);
    SDL_ShowWindow(g_window);

    fprintf(stderr, "sdl2: window raised, focused and shown\n");

    g_renderer = SDL_CreateRenderer(g_window, -1,
        SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    if (!g_renderer) {
        g_renderer = SDL_CreateRenderer(g_window, -1, SDL_RENDERER_SOFTWARE);
        if (!g_renderer) {
            fprintf(stderr, "sdl2: SDL_CreateRenderer failed: %s\n", SDL_GetError());
            gles_shutdown(); return -1;
        }
    }

    /* Create texture matching PSP framebuffer: 480x272 RGBA8888 */
    g_tex = SDL_CreateTexture(g_renderer,
        SDL_PIXELFORMAT_ABGR8888,
        SDL_TEXTUREACCESS_STREAMING,
        480, 272);
    if (!g_tex) {
        fprintf(stderr, "sdl2: SDL_CreateTexture failed: %s\n", SDL_GetError());
        gles_shutdown(); return -1;
    }

    update_viewport(g_window_w, g_window_h);

    fprintf(stderr,
        "sdl2: initialized %dx%d window, content %dx%d at (%d,%d)\n",
        g_window_w, g_window_h, g_viewport_w, g_viewport_h,
        g_viewport_x, g_viewport_y);

    /* Draw an immediate test gradient so the user sees SOMETHING on screen
     * right away, before the game starts submitting GE commands. */
    {
        void *pixels = NULL;
        int pitch = 0;
        if (SDL_LockTexture(g_tex, NULL, &pixels, &pitch) == 0) {
            uint8_t *dst = (uint8_t *)pixels;
            for (int y = 0; y < 272; y++) {
                uint8_t *row_dst = dst + (size_t)y * (size_t)pitch;
                for (int x = 0; x < 480; x++) {
                    row_dst[x * 4 + 0] = (uint8_t)(x * 255 / 480);   /* R */
                    row_dst[x * 4 + 1] = (uint8_t)(y * 255 / 272);   /* G */
                    row_dst[x * 4 + 2] = 128;                          /* B */
                    row_dst[x * 4 + 3] = 255;                          /* A */
                }
            }
            SDL_UnlockTexture(g_tex);
            SDL_RenderClear(g_renderer);
            SDL_Rect dst_rect = { g_viewport_x, g_viewport_y, g_viewport_w, g_viewport_h };
            SDL_RenderCopy(g_renderer, g_tex, NULL, &dst_rect);
            SDL_RenderPresent(g_renderer);
            fprintf(stderr, "sdl2: test gradient drawn to window\n");
        }
    }
    return 0;
}

static void gles_shutdown(void) {
    if (g_tex) { SDL_DestroyTexture(g_tex); g_tex = NULL; }
    if (g_controller) { SDL_GameControllerClose(g_controller); g_controller = NULL; }
if (g_renderer) { SDL_DestroyRenderer(g_renderer); g_renderer = NULL; }
    if (g_window) { SDL_DestroyWindow(g_window); g_window = NULL; }
    SDL_QuitSubSystem(SDL_INIT_VIDEO);
}

static uint32_t g_fb_addr = 0;
static uint32_t g_fb_stride = 512;
static int      g_fb_fmt = 3;
static uint32_t g_present_stage[480 * 272];
static uint32_t g_front_buf[480 * 272];
static int g_front_ready = 0;

static void gles_set_target(uint32_t addr, uint32_t stride, int fmt) {
    pthread_mutex_lock(&g_vram_mutex);
    g_fb_addr = addr;
    g_fb_stride = stride ? stride : 512;
    g_fb_fmt = fmt;
    psp_render_software.set_target(addr, stride, fmt);
    pthread_mutex_unlock(&g_vram_mutex);
}

/* ---- draw: delegate to psp_render_software into guest VRAM --------- */

static void gles_draw(int prim, const psp_vertex *v, int count, const psp_draw_state *st) {
    psp_render_software.draw(prim, v, count, st);
}

/* Present loop: runs sdl2_draw_frame() at a fixed ~60 FPS until
 * psp_exit_requested() returns true. Runs on a dedicated thread so
 * presentation is independent of guest execution. */
void sdl2_present_loop(void) {
    if (!g_tex || !g_renderer)
        return;

    fprintf(stderr, "sdl2: present loop started (~60 FPS)\n");
    fflush(stderr);
    uint32_t kbd_buttons = 0;

    while (!psp_exit_requested()) {
        /* Process SDL events — detect window close (SDL_QUIT) so the game
         * can exit cleanly when the user closes the window. */
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

        /* Lock texture */
        void *pixels = NULL;
        int pitch = 0;
        if (SDL_LockTexture(g_tex, NULL, &pixels, &pitch) != 0) {
            usleep(16667);
            continue;
        }

       pthread_mutex_lock(&g_vram_mutex);
        int ready = g_front_ready;
        if (ready) {
            uint8_t *dst = (uint8_t *)pixels;
            for (int y = 0; y < 272; y++) {
                memcpy(dst + (size_t)y * (size_t)pitch, &g_front_buf[y * 480], 480 * 4);
            }
        }
        pthread_mutex_unlock(&g_vram_mutex);
        if (!ready) {
            /* Game hasn't set a framebuffer yet — draw a test gradient so the
             * user can see the window is working. */
            uint8_t *dst = (uint8_t *)pixels;
            for (int y = 0; y < 272; y++) {
                uint8_t *row_dst = dst + (size_t)y * (size_t)pitch;
                for (int x = 0; x < 480; x++) {
                    row_dst[x * 4 + 0] = (uint8_t)(x * 255 / 480);   /* R gradient */
                    row_dst[x * 4 + 1] = (uint8_t)(y * 255 / 272);   /* G gradient */
                    row_dst[x * 4 + 2] = 128;                          /* B constant */
                    row_dst[x * 4 + 3] = 255;                          /* A opaque */
                }
            }
        }

        SDL_UnlockTexture(g_tex);

        /* Render */
        SDL_RenderClear(g_renderer);
        SDL_Rect dst_rect = {
            g_viewport_x,
            g_viewport_y,
            g_viewport_w,
            g_viewport_h
        };
        SDL_RenderCopy(g_renderer, g_tex, NULL, &dst_rect);
        SDL_RenderPresent(g_renderer);

        /* ~60 FPS: ~16.6 ms per frame */
        usleep(16667);
    }

    fprintf(stderr, "sdl2: present loop exiting\n");
    fflush(stderr);
}

static void gles_finish(void) {
    /* Nothing to flush - we draw immediately */
}

static void gles_present(void) {
    uint32_t disp_addr = psp_display_framebuffer();
    uint32_t fb_addr = disp_addr ? disp_addr : g_fb_addr;
    if (!fb_addr) return;
    uint32_t fb_stride = (disp_addr && g_fb_stride < 480) ? 512u : (g_fb_stride ? g_fb_stride : 512u);
    int fb_fmt = g_fb_fmt;
    static int s_fb_log_count = 0;
    static uint32_t s_prev_fb = 0;
    static uint32_t s_prev_stride = 0;
    static int s_prev_fmt = -1;
    if (s_fb_log_count < 10 && (fb_addr != s_prev_fb || fb_stride != s_prev_stride || fb_fmt != s_prev_fmt)) {
        fprintf(stderr, "sdl2: gles_present disp_addr=0x%08x g_fb_addr=0x%08x fb_stride=%u fb_fmt=%d\n", (unsigned)disp_addr, (unsigned)g_fb_addr, (unsigned)fb_stride, fb_fmt);
        fflush(stderr);
        s_prev_fb = fb_addr;
        s_prev_stride = fb_stride;
        s_prev_fmt = fb_fmt;
        s_fb_log_count++;
    }

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
    pthread_mutex_unlock(&g_vram_mutex);
}

/* ---- backend registration --------------------------------------------- */

const psp_render_backend psp_render_sdl2 = {
    "sdl2",
    gles_init,
    gles_shutdown,
    gles_set_target,
    gles_draw,
    gles_finish,
    gles_present,
};
