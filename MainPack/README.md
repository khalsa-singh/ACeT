# Main endpoint workflows

The three directories contain the main viscosity, mouse-IV-exposure and clinical-outcome workflows. HIC is under `GenBench/hic_benchmark`.

From the root, `python acet.py train all --check-inputs` validates all four main endpoint inputs without importing TensorFlow. Use `python acet.py train <endpoint> --output-dir reruns/<name>` in the training environment for an actual fit. The root README provides all schemas, expected results, head selection, output paths and figure/table relationships.

`shared/minimal_baseline_patch` implements the matched Ridge/SVR/random-forest comparisons; `shared/pooling_ablation` implements the representation-aggregation experiments. Endpoint `results/` folders contain the reference predictions and diagnostic sources. New runs are always separate from these references.
