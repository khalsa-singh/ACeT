# Clearance demonstration implementation

`run_demo.py` calculates the primary mouse-exposure metrics and writes a parity/error-CDF SVG from the archived 11 predictions. `rasterize_demo.py` supplies a Pillow raster fallback for the graphic's elements. `expected_output/` contains the example outputs.

Use the repository-root command `python acet.py demo --formats svg,png,jpg`. See [Demo/README.md](../README.md) for installation, expected values, output units, raster-renderer choices and the transition to full neural training.
