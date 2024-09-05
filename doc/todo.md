# ToDo

- Multiple sources
- Directional sources
- Optimal PPW (calculate phase error at fmax)
- Higher order cartesian stencil
- Simulation plugin

## Questions

- What are ABCs?
- The input signals are scaled multiple times + diff/integrate filter
- Why does diff/integrate help with float32?
- Scaling/Norm for both float32 & float64 is 4.0!?
- Nyquist mode backoff `*= 0.99` in SimConstants
- Absorption over 0.9512 Sabs
- Symmetric filters (LP&HP)
- Sources with only one voxel
- Multiprocess voxels disabled
- FCC Grid modes `1` & `2` (folded)
- 2D room modes don't match theoretical results
