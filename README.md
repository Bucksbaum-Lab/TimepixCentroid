# TimepixCentroid
TimePix3 fast processing with PyTorch

The best current description of our centroiding algorithm can be found in the
publication in Review of Scientific Instruments:
https://doi.org/10.1063/5.0332692

Related discussion can also be found in Appendix A of Ian's thesis, located at
https://stacks.stanford.edu/file/fp182mc4180/submission-12022-20251205-2984820-9klhp6-augmented.pdf

## Pipeline overview

Raw detector data goes through three stages before it's usable for analysis:

1. **`.tpx3` → `.txt`** (`TPX3_read_and_convert_files.cpp`): converts raw
   TimePix3 binary data into a plain-text stream of pixel/trigger events.
   Compile with a C++ compiler (e.g. `g++ TPX3_read_and_convert_files.cpp -o
   TPX3_read_and_convert_files`) and run on a `.tpx3` file. For continuous
   acquisition, we run this in a loop via a simple bash script that watches a
   folder for new `.tpx3` files and converts each one as it appears (not
   included in this repo but straightforward to write).

2. **`.txt` → centroided `.npy`** (`tpx3_centroiding.py`): groups raw pixel
   hits into individual particle centroids (x, y, ToT, ToF) using a GPU-
   accelerated array-based algorithm (see the paper above for the algorithm
   itself). The two main entry points:
   - `read_file_batched(filename, ...)`: centroids a single `.txt` file.
     Useful for testing on one file (see `timepix process single file.ipynb`
     for a minimal example).
   - `centroid_multi_scan(foldy, ...)`: our current workflow. Watches a data
     folder for new .txt files, centroids them, keeps trigger numbers
     consistent across the individual .txt files within a single run (for
     normalization, delay assignment, etc.), and checkpoints results to
     all_centroids_XXXX.npy files. Typically run in a loop, interleaved with
     the acquisition scripts, e.g.:
     ```python
     import tpx3_centroiding as tc
     while True:
         centroids = tc.centroid_multi_scan(foldy, batch_size=10, tottofcorr=True,
                                             checkpoint_interval=10, skip_last=True,
                                             max_gb=3, check_filesz=True)
         time.sleep(60)
     ```
     `skip_last=True` should be used during live acquisition (to avoid reading
     a `.txt` file that's still being written), and set to `False` for one
     final pass once acquisition is complete.

3. **Analysis**: downstream notebooks load the resulting `all_centroids_*.npy`
   files (columns: x, y, ToT, ToF, trigger, param[, filenum, delay]) for
   species separation, momentum calculation, etc. (see other lab repos/
   notebooks for this stage).

Acquisition itself (`ppscan_serval_bothstages.py`) drives the SERVAL server
and motorized stages for parameter scans; it's included here for reference
but is lab-hardware-specific and not part of the centroiding pipeline itself.

## Requirements

- Python 3.10+, NumPy, SciPy
- PyTorch with CUDA support
- An NVIDIA GPU (developed on a Titan RTX; lower-memory GPUs may need a
  smaller `batch_size`)

## Key parameters

- **`batch_size`** (`read_file_batched`, `centroid_multi_scan`): number of
  laser triggers processed in parallel on the GPU. Higher is faster but uses
  more GPU memory, so reduce if you hit CUDA out-of-memory errors, especially
  at high per-shot count rates.
- **`centroid_area_size`**: half-width (in pixels) of the neighborhood used
  to group pixels into a hit. Default 2 (5x5 pixel neighborhood). Adjust
  based on typical hit size for your MCP/phosphor voltages.
- **`centroid_time_size`**: ToF neighborhood half-width in seconds (default
  5e-7). Pixels from the same hit should light up within this window of each
  other; too large risks merging genuinely separate hits.
- **`min_size`**: minimum pixel count in a neighborhood for it to be
  considered a real hit (default 3). Filters out single stray pixels.
- **`tottofcorr`**: whether to apply the ToT-ToF timewalk correction
  (Sec. IV C of the paper). The correction constants are hardcoded in
  `read_file_batched` for our current detector conditions. This will need
  to be changed for each specific detector, and if VMI voltages or Timepix3
  thresholds change significantly, these need to be recalibrated.

## Important notes

- **The 260-pixel Y offset check** in `read_file_batched` (the "backwards
  compatibility" section) exists because the underlying chip readout logic
  supports quad-chip TimePix3 detectors, where each of the 4 chips occupies
  a different 256x256 quadrant of a larger sensor and offsets like this are
  used to stitch them into one coordinate system (see `TPX3_read_and_convert_
  files.cpp`, the `chipnr` switch statement). Our detector is a single chip,
  so this isn't needed, it's kept purely as a sanity check that the
  y-coordinate isn't unexpectedly shifted. If you ever integrate a quad-chip
  detector, this is the relevant mechanism to build on.
- **The `fix_y` pixel-range correction** in the ToT-ToF correction section
  (the `data_array[fix_y,0] -= 25*1e-9` line) is specific to a hardware
  quirk of our particular TPX3CAM unit (certain y-pixel rows reporting a
  fixed timing offset). If you're processing data from a different TimePix3
  camera, this correction likely needs to be re-derived or removed rather
  than assumed to carry over.
- The ToT tie-breaking offset in `read_file_batched` (small per-pixel offset
  added before finding local maxima, removed before computing centroid
  positions) is important: without it, pixels with exactly tied ToT values
  can be double-counted as two separate hits. This is on by default; don't
  remove it unless you know why.
- Trigger numbers reset to 0 at the start of each all_centroids_XXXX.npy
  file (they are only kept consistent within a chunk file, not across
  chunk boundaries). This was a deliberate choice to match the conventions
  expected by our existing downstream analysis pipelines. Anything downstream
  that relies on trigger number needs to account for this, by re-offsetting
  trigger numbers when combining multiple all_centroids files or something
  similar.

## Getting help

If you need help installing anything, get stuck, or find a bug, please
contact Eleanor, Ian, or Chuan.
