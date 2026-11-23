# NanoGPT Track 1 validation

These files record the original offline artifact run and dispatch benchmark.
The hashes use the source paths from that snapshot. The model now has its
own Lake target in VeriTileBench.

The migration check compiled all 238 Lean modules with Lean 4.29.0 and
VeriTile revision `7577e5edd6892a3ec424ad165eed994f74d50109`.
The Lean source binding accepted the default and packed reference
installations. The end-to-end equivalence reported an empty FP assumption
list.

- [Build, native path checks, and regression results](checks.log)
- [Full artifact replay](full-artifacts.log)
- [Recorded source and CUBIN hashes](sha256.txt)

The [Track 1 README](../../models/NanoGPT/Track1/README.md) gives the
source and replay commands.
