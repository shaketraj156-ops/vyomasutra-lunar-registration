# VyomaSutra Project Analysis

## 1. Executive Summary

VyomaSutra is a Python application for registering, or aligning, lunar images captured by different sensors. It provides an interactive Streamlit dashboard and a reusable backend registration pipeline.

The application accepts a moving/source image and a fixed/reference image, detects corresponding image features, estimates a geometric transformation, warps the source image onto the reference image, evaluates the result, and exports an aligned GeoTIFF.

The primary user interface is implemented in `app/streamlit_app.py`. The central processing workflow is implemented in `src/pipeline.py`.

The core SIFT-based workflow was tested locally with generated GeoTIFF data and completed successfully. The current environment does not have LightGlue available, so LightGlue and dual-matcher selections fall back to SIFT.

## 2. Repository Structure

```text
README.md
requirements.txt
PROJECT_ANALYSIS.md
app/
    streamlit_app.py
src/
    pipeline.py
    sensors.yaml
    acquisition/
        create_sample_data.py
        sanity_check.py
    config/
        sensors.yaml
    evaluation/
        comparison_table.py
        dual_track_validation.py
        limitations_report.py
        manual_point_picker.py
        metrics.py
    filtering/
        distribution_score.py
        ransac.py
    matching/
        classical_sift.py
        deep_lightglue.py
        dual_matcher.py
    preprocessing/
        clahe.py
        georef.py
        resample.py
    refinement/
        subpixel.py
    warp/
        transform.py
```

## 3. Application Entry Point

The application is started with:

```powershell
conda activate vyomasutra
streamlit run app/streamlit_app.py
```

The dashboard is normally available at:

```text
http://localhost:8501
```

The Streamlit application performs the following actions:

1. Configures the dashboard layout and styling.
2. Adds the `src` and source subdirectories to Python's import path.
3. Loads the sensor-pair and matcher selections.
4. Loads demo TIFFs if available, or accepts uploaded TIFF files.
5. Calls `run_registration_pipeline()` when the user presses the run button.
6. Displays registration metrics and visualizations.
7. Writes and exposes the aligned GeoTIFF for download.

## 4. Complete Execution Flow

```text
User starts Streamlit
        |
        v
Load or upload source/reference TIFFs
        |
        v
Read first raster band with Rasterio
        |
        v
run_registration_pipeline()
        |
        +--> Load src/config/sensors.yaml
        |
        +--> CLAHE preprocessing
        |
        +--> Normalize images to uint8
        |
        +--> Feature matching
        |       +--> SIFT
        |       +--> LightGlue/SuperPoint when available
        |       +--> Combined matcher when available
        |
        +--> RANSAC homography estimation
        |
        +--> Subpixel refinement with cornerSubPix
        |
        +--> Re-estimate homography
        |
        +--> OpenCV perspective warp
        |
        +--> Calculate metrics
        |
        +--> Generate limitations report
        |
        v
Display results and export GeoTIFF
```

## 5. Main Pipeline Details

### 5.1 Input validation

`src/pipeline.py` rejects missing or empty NumPy arrays before processing. At least four valid matches are required for homography estimation.

The Streamlit upload mode accepts `.tif` and `.tiff` files. Only the first raster band is read from each file.

### 5.2 Sensor configuration

The pipeline loads:

```text
src/config/sensors.yaml
```

The configuration controls CLAHE settings for OHRC, TMC, and IIRS sensor profiles.

For example:

- OHRC uses a CLAHE clip limit of `2.0`.
- TMC uses a CLAHE clip limit of `1.5`.
- IIRS uses a CLAHE clip limit of `1.0` and a smaller tile grid.

`src/sensors.yaml` contains a second configuration format, but the main pipeline does not load that file.

### 5.3 Preprocessing

`src/preprocessing/clahe.py` applies Contrast Limited Adaptive Histogram Equalization. This is intended to reduce the effect of changing lunar illumination and shadows.

After CLAHE, non-`uint8` images are normalized to the range 0-255 and converted to `uint8` for OpenCV feature processing.

### 5.4 Feature matching

The application exposes three matcher selections:

#### Classical SIFT

Implemented in:

```text
src/matching/classical_sift.py
```

The implementation:

- Detects SIFT keypoints.
- Computes descriptors.
- Uses a brute-force matcher.
- Applies Lowe's ratio test.
- Returns matches in the shared coordinate format.

#### SuperPoint and LightGlue

Implemented in:

```text
src/matching/deep_lightglue.py
```

The implementation uses PyTorch, SuperPoint, and LightGlue. Models are created lazily and cached globally. CUDA is used when available; otherwise CPU is used.

#### Dual matching

Implemented in:

```text
src/matching/dual_matcher.py
```

The dual mode combines SIFT and LightGlue matches and attaches a source tag to each match.

In the tested environment, LightGlue was not importable:

```text
LIGHTGLUE_AVAILABLE = False
```

Therefore, the current behavior is:

- `sift`: SIFT only.
- `lightglue`: falls back to SIFT.
- `dual`: runs SIFT because LightGlue is unavailable.

## 6. Geometric Registration

### 6.1 RANSAC

`src/filtering/ransac.py` uses `cv2.findHomography()` with the RANSAC method.

The default inlier threshold is 3 pixels, configurable from the Streamlit sidebar.

RANSAC removes matches that do not agree with a common projective transformation.

### 6.2 Subpixel refinement

`src/refinement/subpixel.py` uses OpenCV `cornerSubPix()` to refine both source and reference coordinates.

After refinement, the pipeline estimates the homography again using the refined points.

### 6.3 Warping

`src/warp/transform.py` applies the homography using `cv2.warpPerspective()`.

The output dimensions are taken from the processed reference image. The source image is therefore transformed into the reference image's pixel canvas.

## 7. Evaluation Metrics

The pipeline calculates the following metrics:

- Total number of feature matches.
- Number of RANSAC inliers.
- Inlier ratio.
- Reprojection RMSE.
- Structural Similarity Index, or SSIM.
- Weighted distribution score.

The distribution score measures both match coverage across an 8x8 image grid and average match confidence. It rewards matches that are spatially distributed rather than concentrated in one region.

The metric implementation is located in:

```text
src/evaluation/metrics.py
src/filtering/distribution_score.py
```

## 8. Limitations and Risk Reporting

`src/evaluation/limitations_report.py` generates a JSON report containing:

- Low-texture-region analysis.
- Mean patch variance.
- Polar-region warning when latitude is supplied.
- Match-quality warnings.
- Confidence classification.

The report is normally written to:

```text
outputs/reports/limitations_report.json
```

The Streamlit dashboard displays the warnings and confidence level in the limitations tab.

## 9. GeoTIFF Input and Output

Rasterio is used for reading and writing GeoTIFF files.

The export code:

- Writes a single-band GeoTIFF.
- Preserves the reference CRS when available.
- Preserves the reference affine transform when available.
- Uses a default lunar CRS when reference CRS metadata is missing.

Streamlit writes output files under:

```text
outputs/geotiff/
```

The generated file is also provided through a download button.

## 10. Synthetic Test Data

`src/acquisition/create_sample_data.py` generates two 512x512 synthetic lunar terrain images.

The generated source image includes:

- A five-degree rotation.
- Translation.
- Illumination and contrast variation.
- GeoTIFF CRS and affine-transform metadata.

To create demo TIFF files:

```powershell
conda activate vyomasutra
python src/acquisition/create_sample_data.py
```

The files are created at:

```text
data/samples/lunar_source_crater.tif
data/samples/lunar_reference_crater.tif
```

The current repository checkout did not initially contain these files, so the default demo mode requires generating them or uploading custom TIFFs.

## 11. Dependencies and Runtime Environment

The declared dependencies are listed in `requirements.txt`:

- `opencv-contrib-python`
- `numpy`
- `scikit-image`
- `streamlit`
- `pyyaml`
- `gdal`
- `rasterio`
- `torch`
- `torchvision`
- `kornia`
- LightGlue from GitHub

The tested local environment was:

```text
Python: 3.10.21
Streamlit: 1.63.0
PyTorch: 2.14.0+cpu
CUDA: unavailable
```

Conda is recommended on Windows because GDAL and Rasterio depend on native libraries. The setup documented in `README.md` is:

```powershell
conda create -n vyomasutra python=3.10 -y
conda activate vyomasutra
conda install -c conda-forge gdal rasterio -y
pip install -r requirements.txt
```

A GPU is optional. The code supports CPU execution, but LightGlue inference will generally be slower without CUDA.

## 12. Verification Performed

### Python compilation

All Python files compiled successfully with:

```powershell
python -m compileall -q app src
```

### Module import

The central pipeline imported successfully, including OpenCV, Rasterio, PyTorch, YAML, and internal modules.

### Streamlit server

The dashboard was launched with:

```powershell
conda run -n vyomasutra streamlit run app/streamlit_app.py --server.headless true --server.port 8501
```

The server listened on port `8501` and returned HTTP status `200`.

### End-to-end registration

A generated source/reference TIFF pair was processed using SIFT. The pipeline completed all stages:

```text
Preprocessing complete
Matching complete: 132 matches
RANSAC complete: 124 inliers
Subpixel refinement complete
Warping complete
Checkerboard complete
Metrics complete
GeoTIFF export complete
```

Measured results:

```text
Total matches: 132
RANSAC inliers: 124
Inlier ratio: 93.9%
Reprojection RMSE: 2.1477 px
SSIM: 0.9273
Distribution score: 0.7620
```

## 13. Known Limitations and Risks

### LightGlue is currently inactive

The installed environment reported that LightGlue was unavailable. The application handles this gracefully by falling back to SIFT, but the current test does not validate deep matching.

### Demo files are not committed

The dashboard expects files under `data/samples`, but that directory was absent in the original checkout.

### Real-data validation is pending

The project documentation describes synthetic rotated-data results. Real OHRC, TMC-2, IIRS, and LROC data are required for a true sensor-to-sensor validation.

### Benchmark comparison is not fully equivalent

The comparison table uses values reported in a research paper, while the current project results use synthetic test images. The values should not be treated as a direct apples-to-apples benchmark until the same real scenes are used.

### Duplicate configuration files exist

There are two sensor configuration files with different schemas. The pipeline uses `src/config/sensors.yaml`; `src/sensors.yaml` appears to be an older or alternate configuration.

### Auxiliary scripts use legacy paths

Some standalone test sections reference paths such as `data/raw/test_dummy.tif`, which are not part of the current repository structure. These paths do not prevent the Streamlit application from running.

### CPU-only deep-learning execution

The current PyTorch installation does not detect CUDA. This does not prevent execution, but it limits deep-matching performance.

## 14. Overall Assessment

The primary application is operational. The SIFT registration path has been verified from TIFF input through GeoTIFF output, including preprocessing, matching, RANSAC, subpixel refinement, warping, metric calculation, and report generation.

The project is best understood as a research prototype with a working interactive demonstration. Its main remaining validation requirement is testing LightGlue and the complete dual matcher on real multi-sensor lunar imagery.
