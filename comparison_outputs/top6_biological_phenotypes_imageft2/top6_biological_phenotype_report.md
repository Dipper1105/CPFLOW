# Top6 image-derived biological phenotype metrics

This analysis computes lightweight biological phenotype proxies from the displayed RGB Cell Painting images.
The same metrics are computed for the real perturbation image and the best generated image for each drug.

Important interpretation note: the available files are RGB composites rather than separated Cell Painting
channels. Therefore mitochondrial brightness is reported as a red-channel/cytoplasmic signal proxy,
and nuclear/cytoplasmic regions are estimated by intensity-based masks rather than single-cell segmentation.

| Drug | Phenotype similarity | Biology/process context |
|---|---:|---|
| nystatin | 0.887 | Membrane sterol binding; membrane stress/permeability phenotype. |
| puromycin | 0.926 | Translation inhibition; proteotoxic stress and reduced growth/protein synthesis phenotype. |
| pd-98059 | 0.903 | MEK/ERK pathway inhibition; altered proliferation and cytoskeletal signalling. |
| hydroxyurea | 0.813 | Ribonucleotide reductase inhibition; S-phase arrest and DNA-replication stress. |
| quercetin | 0.809 | Polyphenol kinase/oxidative-stress modulator; mitochondrial/redox and cytoskeletal effects. |
| cyclophosphamide | 0.900 | Alkylating DNA-damage agent; genotoxic stress and growth arrest. |

Drug-process focused metrics:

| Drug | Process | Focused metrics | Mean focused agreement |
|---|---|---|---:|
| nystatin | membrane sterol binding / membrane stress | Mitochondrial brightness proxy; Cell area fraction; Cell clustering proxy | 0.867 |
| puromycin | translation inhibition / growth and proteotoxic stress | Cell count proxy; Mean cell area proxy; Edge/texture strength | 0.836 |
| pd-98059 | MEK-ERK inhibition / proliferation-cytoskeleton signalling | Cell count proxy; Cell area fraction; Edge/texture strength | 0.917 |
| hydroxyurea | S-phase arrest / DNA replication stress | N:C area ratio; Nuclear area fraction; Cell count proxy | 0.602 |
| quercetin | redox-mitochondrial and cytoskeletal stress | Mitochondrial brightness proxy; Mitochondrial enrichment proxy; Edge/texture strength | 0.818 |
| cyclophosphamide | alkylating DNA damage / growth arrest | N:C area ratio; Cell count proxy; Local contrast | 0.791 |

Generated files:

- `top6_biological_phenotype_metrics_long.csv`
- `top6_biological_phenotype_similarity.csv`
- `top6_drug_process_focused_metrics.csv`
- `figure_top6_phenotype_similarity.png/svg/pdf`
- `figure_top6_phenotype_metric_heatmap.png/svg/pdf`
- `figure_top6_biology_metrics_real_vs_generated.png/svg/pdf`
- `figure_top6_drug_process_focused_metrics.png/svg/pdf`

TIFF export can be enabled by rerunning the script with `--write-tiff`.

Metric definitions:

- Mitochondrial brightness proxy: mean red-channel intensity in the estimated cytoplasmic mask.
- N:C area ratio: estimated nuclear mask area divided by cytoplasmic mask area.
- Cytoplasm:nucleus intensity: estimated cytoplasmic RGB intensity divided by nuclear blue-channel intensity.
- Cell count proxy: connected nuclear-object count normalized to a 512 x 512 field.
- Clustering proxy: mean connected cell-field area divided by image area.
- Edge/texture strength: mean local absolute gradient of the RGB max-intensity image.
