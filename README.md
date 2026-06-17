# About

This is a collaborative project for NIT/SENDAI International Exchange / Internship program.

Much like "AI generated closed captions" on videos for sub titles, this project aimed at dance notation
recognition & auto caption for traditional Japanese folk dances using raw data points in 3D space as the input feed.

## Development Environment

The project uses [Nix flakes](flake.nix) + `uv` for reproducible dependencies.

```bash
uv sync && uv venv
source .venv/bin/activate
jupyter notebook src/etl.ipynb
```

Key Python dependencies: `numpy`, `scipy`, `bvhio`, `plotly`, `matplotlib`.

## Goals

```
[/] Extraction
    [/] Custom Kinetics Library
        [/] Joint tree systems
        [/] Homogenous transformation
        [/] Joint modification system
    [/] Model Library
        [/] Model loading
            [/] BVH file
            [?] Live mocopi data stream
                Note: More research
        [/] Extract model to joint tree

[*] Transform
    [/] Normalization
        [/] Fix model to origin
        [/] Fix model rotation
            Note: Optimization (need more research)
        [/] Bone length normalization
    [/] Diffentiate Actions
        [/] Multidimensional DTW
        [/] Error function
    [*] Caption Mechanism
        [/] Segmentation
        [/] Reduction
        [ ] Enrollment mechanism

[ ] Load
    [ ] Export data as CSV
    [ ] Output as a file stream / socket for IPC
```
```
[/] = finished, [*] = in progress
[?] = may need a re-visit / more research
```
