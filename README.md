# About

This is a collaborative project for NIT/SENDAI International Exchange / Internship program.

Much like "AI generated closed captions" on videos for sub titles, this project aimed at dance notation
recognition & auto caption for traditional Japanese folk dances using raw data points in 3D space as the input feed.

# Development

## Recommended packages / Environment (not required)
```bash
direnv
nix
```

## Install python deps
```bash
uv sync
uv venv
```
## Activate venv
```bash
source .venv/bin/activate
```

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
    [?] Normalization
        [/] Fix model to origin
        [?] Fix model rotation
            Note: Optimization (need more research)
        [?] Bone length normalization
            Note: Current implimentation my be subject to chage
                due to quirks, and edge cases
    [*] Diffentiate Actions
        [*] Multidimensional DTW
        [ ] Error function
    [ ] Caption Mechanism
        [ ] Segmentation
        [ ] Enrollment mechanism
        [ ] Reduction

[ ] Load
    [ ] Export data as CSV
    [ ] Output as a file stream / socket for IPC
```
```
[/] = finished, [*] = in progress
[?] = may need a re-visit / more research
```
