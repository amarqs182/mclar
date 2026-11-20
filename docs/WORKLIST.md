# Midnight Club: L.A. Remix (USA) — Worklist & Regras

Inspirado em **N64Recomp**, **XenonRecomp** e **sa2-nx**:
- Configuração declarativa do jogo em `config/` (`mc3_seeds.txt`), mantendo o runtime `psprecomp` genérico.
- Progresso por fases com critério verificável no host (`build-host`) antes de ir para o R36S (`build-r36s`).

---

## Regras do Projeto

1. **Proibido mascarar sintomas:** Nunca alterar vértices, cores, alpha ou registradores em runtime para forçar algo na tela. Se vier zerado ou errado, corrigir a tradução MIPS→C, os seeds de função, o I/O ou o parser do GE.
2. **Debug sempre atrás de flag:** Traces pesados (`PSPRECOMP_TRACE`) ficam apenas no alvo `mc3_trace` ou atrás de variáveis de ambiente (`PSPRECOMP_TRACE_IO=1`).
3. **Host primeiro, R36S depois:** Só testar no R36S (`build-r36s`) o que já funciona e foi validado em `build-host`.
4. **Seeds em vez de stubs vazios:** Toda função alcançada por chamada indireta (`indirect call to 0x...`) deve ser adicionada a `config/mc3_seeds.txt` e recompilada com `--seeds`, nunca silenciada com stub vazio em `main.c`.
5. **Fonte única de verdade:** Estado técnico detalhado fica em `notes/MEMORY.md` (com data e comando); esta worklist acompanha as tarefas ativas.
6. **Repositório limpo:** Zero binários compilados, zero arquivos de jogo (`ISO`/`ELF`/`BIN`/`PCODE`) e zero diretórios de build no Git.

---

## Worklist

### Fase 1 — Limpeza e Base Limpa ✅
- [x] Remover diretórios de build redundantes (manter `build-host` e `build-r36s`), core dumps e backups.
- [x] Remover artefatos de jogo/scratch/binários do índice do Git e atualizar `.gitignore`.
- [x] Remover `PSPRECOMP_TRACE` do alvo principal `mc3` (corrigido `PSP_SP_RESET()` em `emit.c`).
- [x] Remover o hack do retângulo vermelho em `src/mc3/render_sdl.c`.
- [x] Validar compilação limpa em `build-host`.

### Fase 2 — Boot e Fluxo Real ✅
- [x] Remover hacks de `read_vertex()` em `src/psprecomp/src/hle/ge.c` e corrigir `GE_FBW` preservando `PSP_VRAM_BASE` (`0x04000000`).
- [x] Implementar `mtv` / `mfv` (`psp_mtv` / `psp_mfv`) em `src/psprecomp`.
- [x] Adicionar suporte a `--seeds <arquivo>` no `allegrexrecomp emit` e criar `config/mc3_seeds.txt` com os 12 alvos indiretos perdidos (incluindo `0x002CF4D0` `streamer`, `0x0037FB3C`, `0x003886A8`).
- [x] Reemitir, recompilar `mc3` e rodar no host: **0** chamadas indiretas perdidas, **0** traps VFPU (`mtv`/`mfv`), `fbp=0x04088000` preservado.
- [x] Remover criação manual da thread `0x000000D8` e os stubs vazios de `src/mc3/main.c` (`streamer` agora permanece vivo).
- [x] Corrigir herança de `$gp` (`0x0042CEB0`) em `main.c` e `threadman.c`, implementar `sceIoGetstat` (`0xACE946E8`), extrair `PSPASSET.DAT`/`PSPASSET2.DAT` e validar 300 frames estáveis no host (`switches=1199`, `0` misses, `0` bad memory accesses).

### Fase 3 — Renderização Real e Avanço Pós-Boot (GE + SDL2 → GLES 3.2) 🔄
- [x] Corrigir `sceUmdUser` (`umd.c`) para reportar disco montado (`0x32`), implementar `LwMutex` (`threadman.c`), `sceIoReadAsync`/`sceIoWaitAsyncCB` e leitura de setores via `umd0.iso` + LBA de `PSPASSET.DAT`.
- [x] Aplicar relocações PRX (`tools/prx_link.py` → `EBOOT_linked.ELF`) corrigindo 3.226 ponteiros `.bss`/`.sbss`/`.sdata` que apontavam para `.text`.
- [x] Validado no host: jogo lê `PCODE.TXT`, `PSPASSET.DAT` header e TOC (~272 KB) direto de `umd0.iso`.
- [x] Corrigir bug de pareamento `HI16`/`LO16` em `tools/prx_link.py`, `jalr` fall-through em `emit.c`, `vdiv`/`vrot`/`vcrsp` em VFPU, `floor.w.s`/`ceil.w.s`/`round.w.s` em FPU, espelho de EDRAM 8 MB em `mem.c`, e `sceKernelWaitSemaCB` em `threadman.c`.
- [x] Implementar HLE de `sceMpeg` (com parser `PSMF` + ringbuffer callback), `sceAtrac3plus`, `sceIoOpenAsync`/`CloseAsync`, `sceKernelVolatileMem*`, `sceKernelTerminateDeleteThread` e liberação de slots concluídos da fila GE (`ge.c`).
- [x] Validado no host (`build-host`, 900 frames, `EXIT=0`): passa pelos 3 vídeos de introdução, `MC3Loading` e alcança renderização contínua (`2.4M` comandos GE, `1.7M` vértices, `0` dispatch misses, `0` unimplemented calls, `0` bad memory accesses).
- [x] Inspecionar os comandos GE e formatos de vértice submetidos pela engine AGE (`0x80011E`/`0x800102`/`0x80011C` em modo `through`; `0x00019F`/`0x000111`/`0x00019E` transformados).
- [ ] Corrigir `VT_POS` (1/2/3 = 8/16/float + 3 coordenadas) em `ge.c`, implementar suporte a texturas (CLUT/swizzle/UV) e geometria transformada (matrizes GE/VFPU) no `psprecomp` (`ge.c` + `psp_render_backend` + `render_sdl.c`).
- [ ] Validar tela de título / menus visíveis no backend `sdl2` no host.
- [ ] Portar o suporte de texturas/transformação para `render_gles.c` (GLES 3.2).

### Fase 4 — Input, Áudio e PortMaster (R36S) ⏳
- [ ] Conectar input real (SDL2 keyboard/gamepad) a `psp_ctrl_set()` e remover auto-press de `CROSS`.
- [ ] Implementar saída de áudio (`sascore.c` → SDL2 audio).
- [ ] Compilar `build-r36s`, testar no R36S via `tools/deploy_r36s.sh` e medir RSS / frame times.
