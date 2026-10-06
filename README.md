# Ligand PathFinder (LPF)

Ligand PathFinder builds physically reasonable **ligand (un)binding pathways** for
protein–ligand complexes. Starting from a functionally relevant binding pose and a
rough direction of egress, it produces a smooth guide path and then a dense
trajectory of protein–ligand conformations along it, with a Rosetta binding-energy
estimate at every step. It is intended for studying binding/release routes,
gateway residues, and approximate energy profiles.

The pipeline has two stages, run in order:

| Stage | Script | Input | Output |
|-------|--------|-------|--------|
| **Coarse path** | `lpf/lpf_coarse_path.py` | start pose + two points (pose, exit direction) | a guess path (`coarsepath.txt`) |
| **Trajectory** | `lpf/lpf_trajectory.py` | start pose + a full guess path | per-window structures + `interface_scores.txt` |

The coarse stage is stochastic — run it several times and cluster the results, or
supply your own path (e.g. from CAVER / AQUA-DUCT) and skip straight to the
trajectory.

## Installation

```bash
conda env create -f environment.yml
conda activate ligpf
```

The Python side only needs `numpy` and `scipy` (see `environment.yml`).

**Rosetta is not included and must be installed separately.** LPF calls a
`rosetta_scripts` binary as a subprocess; download it from
<https://downloads.rosettacommons.org/software/academic/> and put its path in the
`"rosetta"` key of your `run.json`. An MPI launcher (e.g. `mpirun`) is only needed
if you set `rosetta_scripts.n_proc > 1`, in which case the binary must be an
`.mpi.` build.

> Rosetta has its own license, which you must agree to and comply with
> independently. The [LICENSE.md](LICENSE.md) in this repository covers **only the
> Ligand PathFinder code**.

## Running

Both scripts take one JSON config and, by convention, are run from a fresh
working directory:

```bash
# 1. coarse path
mkdir run_coarse && cd run_coarse
python /path/to/lpf/lpf_coarse_path.py ../run.json

# 2. trajectory (its `path` is the coarsepath.txt from step 1)
mkdir run_traj && cd run_traj
python /path/to/lpf/lpf_trajectory.py ../run.json
```

Any config value can be overridden on the command line:

```bash
python lpf/lpf_trajectory.py run.json --set parameters.step=0.5 --set rosetta_scripts.nstruct_main=20
```

`lpf_trajectory.py` also accepts `--fresh` to ignore an existing checkpoint.

## Configuration

Every `run.json` key — what it does, its units, and which stage uses it — is
documented in **[docs/config_reference.md](docs/config_reference.md)**.
Keys that a given stage ignores are listed in
[docs/unused_config_settings.md](docs/unused_config_settings.md).
`full_input.json` is a template listing every key both stages understand; its
paths are written relative to `rosettaXML_files/`, so copying that directory next
to your run as `files/` makes it work as-is.

## Examples and demo

* **[`examples/`](examples/)** — minimal working `run.json` for each stage
  (`examples/coarse_path/`, `examples/trajectory/`), showing the required keys.
* **[`demo/`](demo/)** — a full worked walkthrough
  ([`demo/tutorial.md`](demo/tutorial.md)) on a real system
  (butyrylcholinesterase + paraoxon), with every input file needed to reproduce
  it and the figures it produces.

> **Run outputs are not tracked in git.** `demo/*/run/` and `examples/*/run/`
> are where a run writes its results (hundreds of MB of decoy archives and
> trajectory PDBs); they are `.gitignore`d. All *inputs* are here, so both the
> demo and the examples can be reproduced from a fresh clone by following the
> tutorial.
>
> The reference runs are archived on Zenodo under
> [`10.5281/zenodo.23183178`](https://doi.org/10.5281/zenodo.23183178) (`LigPF2026.zip`, CC-BY-4.0).
> Inside it, `zenodo/demo/*/run/` mirrors this repository and unpacks straight
> into the matching folders; the same deposit also carries the full supporting
> dataset for the paper under `zenodo/paper/`.

## Repository layout

```
lpf/                 the two entry-point scripts + helpers (util.py, bintree.py)
rosettaXML_files/    stock RosettaScripts protocols and options — every protocol
                     referenced by full_input.json lives here, including
                     coarsepath.xml (coarse stage) and main_standard.xml,
                     zero_minimize.xml, zero_dock.xml, stats.xml (trajectory)
docs/                configuration reference
examples/            minimal configs, one per stage
demo/                worked tutorial (inputs + figures; run outputs untracked)
environment.yml      conda environment
full_input.json      config template covering every key
```

## License

MIT — see [LICENSE.md](LICENSE.md). This applies to the Ligand PathFinder code
only; Rosetta is licensed separately.
