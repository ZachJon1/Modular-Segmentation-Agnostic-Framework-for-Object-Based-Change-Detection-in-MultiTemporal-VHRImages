# Rosebud Pairwise Segmentation Metrics

This document explains the Region Correlation Matrix (RCM) statistics computed for the Rosebud site. The CSV `outputs/metrics/rosebud_pairwise.csv` contains pairwise comparisons between the segmentation algorithms:

| Algorithm | Description |
|-----------|-------------|
| `felzenszwalb` | Graph-based segments (skimage `felzenszwalb`). |
| `meanshift` | Colour-space mean shift with spatial radius (OpenCV). |
| `quickshift` | Quick shift superpixels (skimage). |
| `rezsd` | Provided REZsd segmentation (CSV masks). |
| `samgeo` | SAMGeo ViT-H segmentation with the added checkpoint. |

For each ordered pair `(algo_i, algo_j)` the CSV lists:

- **Overlap (`overlap_ab_t0`, `overlap_ab_t1`)** – fraction of mass in `algo_i` regions at time `t` explained by the best matching region in `algo_j`. Values near 1 indicate large segments align well.
- **Fragmentation (`fragmentation_ab_t0`, `fragmentation_ab_t1`)** – measures how many fragments `algo_i` regions break into when projected onto `algo_j`; 0 means perfectly intact, 1 indicates heavy fragmentation.
- **Composite (`composite_ab_t0`, `composite_ab_t1`)** – averaging fragmentation and (1 − overlap) with equal weights; lower is better (0 ideal, 1 worst).

The reciprocal entries (`overlap_ba_*`, etc.) capture the reverse direction to detect asymmetry (e.g., smaller segments may cover larger ones well, yet the reverse not).

## High-level observations

| Pair (algo_i → algo_j) | Overlap t0 | Fragment. t0 | Composite t0 | Overlap t1 | Fragment. t1 | Composite t1 | Notes |
|------------------------|-----------:|-------------:|-------------:|-----------:|-------------:|-------------:|-------|
| felzenszwalb → meanshift | 0.197 | 0.820 | 0.811 | 0.235 | 0.794 | 0.779 | Felzenszwalb segments split across many mean-shift regions; large fragmentation. |
| felzenszwalb → quickshift | 0.230 | 0.893 | 0.832 | 0.287 | 0.859 | 0.786 | Quickshift has many fine regions, so felzenszwalb coverage is fragmented. |
| felzenszwalb → rezsd | **0.897** | **0.177** | **0.140** | **0.900** | **0.139** | **0.119** | Strong alignment with REZsd; low fragmentation and composite. |
| felzenszwalb → samgeo | 0.742 | 0.297 | 0.277 | 0.790 | 0.242 | 0.226 | Good agreement—SAMGeo follows similar parcels with moderate fragmentation. |
| meanshift → quickshift | 0.086 | 0.932 | 0.923 | 0.082 | 0.942 | 0.930 | High fragmentation: mean-shift regions are coarse vs quickshift superpixels. |
| meanshift → rezsd | 0.869 | 0.141 | 0.136 | 0.827 | 0.173 | 0.173 | Similar to Felzenszwalb vs REZsd: mean-shift segments map well to REZsd. |
| meanshift → samgeo | 0.568 | 0.483 | 0.458 | 0.566 | 0.479 | 0.456 | Medium similarity; SAMGeo has higher detail than mean-shift. |
| quickshift → rezsd | 0.922 | 0.136 | 0.107 | 0.922 | 0.136 | 0.107 | Quickshift superpixels consistently nest inside the REZsd segments. |
| quickshift → samgeo | 0.913 | 0.284 | 0.209 | 0.908 | 0.269 | 0.205 | SAMGeo segments aggregate quickshift regions with moderate fragmentation. |
| rezsd → samgeo | 0.272 | 0.929 | 0.828 | 0.270 | 0.932 | 0.829 | In the reverse direction REZsd is very coarse relative to SAMGeo, causing high fragmentation. |

*Note:* The CSV includes all 20 ordered combinations; the table above highlights representative entries centred on felzenszwalb, mean shift, and quickshift relationships with REZsd and SAMGeo.

## Detailed interpretation

### Alignment with REZsd

- **Felzenszwalb ↔ REZsd**: Overlaps near 0.9 and fragmentation around 0.17 (forward) show Felzenszwalb segments are largely subsets of REZsd parcels. The reverse (REZsd → Felzenszwalb) yields lower overlap (~0.17) and high fragmentation (0.87), indicating REZsd’s five coarse regions span multiple Felzenszwalb segments.
- **Quickshift ↔ REZsd**: Quickshift’s 845/908 segments at t₀/t₁ strongly nest within the five REZsd regions. Overlap surpasses 0.92 forward, and fragmentation remains low (0.13–0.17). The reverse direction again reflects coarse vs fine segmentation.
- **Meanshift ↔ REZsd**: Similar to Felzenszwalb, mean-shift segments map cleanly onto REZsd (overlap ~0.87, fragmentation ~0.14), but REZsd → MeanShift shows high fragmentation, as expected when comparing five classes to >300 regions.

### SAMGeo comparisons

- **SAMGeo vs Felzenszwalb**: With ~193 regions, SAMGeo sits between Felzenszwalb (284/336) and REZsd (5/6). Forward overlap 0.74–0.79 and composite 0.23–0.28 indicate strong agreement. The reverse entries show SAMGeo captures additional structure beyond Felzenszwalb’s boundaries (overlap 0.35–0.34, fragmentation ~0.78).
- **SAMGeo vs Quickshift**: Quickshift offers fine-grained superpixels covering SAMGeo parcels (overlaps ~0.91 when quickshift → samgeo). Reverse direction shows moderate overlap (~0.34) and fragmentation (~0.84), again due to SAMGeo merging smaller regions.
- **SAMGeo vs MeanShift/REZsd**: SAMGeo retains more detail than mean-shift and REZsd. Forward overlaps from mean-shift → samgeo ~0.57 demonstrate many mean-shift regions align with SAMGeo, but the reverse direction reveals fragmentation because SAMGeo subdivisions do not sit cleanly inside coarse mean-shift segments. REZsd → SAMGeo is the most fragmented (fragmentation >0.93) because five REZsd classes explode into numerous SAMGeo parcels.

### Felzenszwalb vs Quickshift/MeanShift

- **Felzenszwalb ↔ Quickshift**: Forward overlap (Felz → Quick) 0.23–0.29 is moderate, and fragmentation 0.86–0.89 indicates quickshift breaks segments apart. Quickshift → Felz overlap 0.78–0.76 is high since the fine superpixels stay within Felzenszwalb regions. This asymmetry illustrates resolution differences: Quickshift is the most detailed.
- **Felzenszwalb ↔ MeanShift**: Both algorithms produce mid-scale segments (Felz 284/336 regions vs MeanShift 326/372). Overlaps remain modest (0.20–0.24) with high fragmentation (~0.75–0.82), signalling divergent boundaries; each method partitions the scene differently.

### Temporal symmetry

Across all pairs, the t₀ and t₁ statistics are similar—e.g., Felzenszwalb ↔ REZsd overlap remains ~0.90 at both times, implying the segmentation alignment patterns are consistent between 2022 and 2023. Differences emerge mainly in fragmentation/overlap directionality (coarse vs fine algorithms), not across time.

## Takeaways for Rosebud

1. **REZsd baseline** aligns best with Felzenszwalb and MeanShift when viewed forward (high overlap, low fragmentation). However, in the reverse direction REZsd looks overly coarse.
2. **Quickshift** offers the highest fidelity; its superpixels nest inside other algorithms’ regions (high overlap when Quickshift is the source) but fragment them when it is the target.
3. **SAMGeo** sits between coarse REZsd and fine Quickshift: it agrees well with Felzenszwalb (composite ≈ 0.23–0.28) and Quickshift (0.20–0.21 in forward direction), indicating it captures similar change-relevant boundaries.
4. **MeanShift** shows mixed behaviour—reasonably aligned with REZsd and SAMGeo, but divergent from Felzenszwalb/Quickshift.

When deciding which segmentation to pair with CVA for Rosebud, SAMGeo and Felzenszwalb provide a good balance between detail and stability, while Quickshift serves as the high-granularity “superpixel” reference and REZsd as the coarse baseline.
