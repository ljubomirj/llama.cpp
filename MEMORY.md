2026-05-07: Started debugging MTP forward-port graph reservation crash on LJ-Ling-2.6-flash-r2.
2026-05-07: Fixed MTP graph/context crash by keeping BAILING_HYBRID_MTP off hybrid memory and moved MTP streaming toward the GG-style speculative/server layer.
2026-05-07: Server MTP smoke now works with CPU/default MTP draft head; forced GPU MTP still Metal-OOMs but is guarded from cascading position errors.
2026-05-07: Reviewed latest PR #22673 mtp-clean head and captured lessons in docs/mtp/12-pr-22673-mtp-clean-latest.md.
