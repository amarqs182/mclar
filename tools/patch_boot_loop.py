#!/usr/bin/env python3
"""Deprecated: startup-flow edits must be made in the recompiler or HLE.

Older revisions injected hard-coded register values and branch outcomes into
generated C. Those edits bypassed controller and file-I/O behavior and could
make the guest thread return before normal initialization. Regenerate with
``allegrexrecomp emit``; this compatibility script intentionally changes
nothing.
"""

print("patch_boot_loop.py is retired; generated code was left unchanged")
