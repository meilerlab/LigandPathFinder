# LIGAND PATH FINDER
# Trajectory script
# Given an optimized guide path, create a trajectory of conformations along it

import numpy as np
import os
import pickle as pkl
import argparse
import glob
import tempfile
from shutil import copy, copytree, rmtree
from datetime import timedelta
from time import time
import json

from util import (
    read_config,
    read_guide_path,
    check_protocol_chain,
    run_rosetta,
    run_tool,
    TrajVirt,
    GeometryUtils,
    PathBuilder,
    PDBHandler,
    ScoreParser,
    ConstraintBuilder,
    ConformerSampler,
)

_CHECKPOINT_VERSION = 1


def _project_onto_plane(point, plane_point, plane_normal):
    n = plane_normal / np.linalg.norm(plane_normal)
    return point - np.dot(point - plane_point, n) * n


def _path_max_turn_deg(guide_path):
    '''Return the maximum step-to-step turning angle (degrees) in a guide path array.'''
    turns = []
    for i in range(1, len(guide_path) - 1):
        v1 = guide_path[i] - guide_path[i - 1]
        v2 = guide_path[i + 1] - guide_path[i]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 > 0 and n2 > 0:
            cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
            turns.append(np.degrees(np.arccos(cos_a)))
    return max(turns) if turns else 0.0


def _coerce_value(value_str, original_value):
    '''
    Cast a CLI string to the same type as `original_value`.
    If original_value is None (new key), try int → float → bool → str.
    '''
    if isinstance(original_value, bool):
        return value_str.lower() in ('true', '1', 'yes')
    if isinstance(original_value, int):
        return int(value_str)
    if isinstance(original_value, float):
        return float(value_str)
    if original_value is None:
        for cast in (int, float):
            try:
                return cast(value_str)
            except ValueError:
                pass
        if value_str.lower() in ('true', 'false'):
            return value_str.lower() == 'true'
    return value_str


def apply_config_overrides(config, overrides):
    '''
    Apply a list of dot-notation "section.key=value" overrides to config in place.

    All keys except the last must already exist as dict sections in config.
    The leaf key may be new within an existing section.
    Type is inferred from the existing value; new leaf keys are auto-typed.
    '''
    for override in overrides:
        if '=' not in override:
            raise ValueError(
                f"Override must be in KEY=VALUE format (e.g. parameters.step=0.5), got: '{override}'"
            )
        key_path, value_str = override.split('=', 1)
        keys = key_path.split('.')

        d = config
        for k in keys[:-1]:
            if k not in d or not isinstance(d[k], dict):
                raise KeyError(
                    f"Override '{key_path}': section '{k}' not found in config"
                )
            d = d[k]

        final_key = keys[-1]
        original = d.get(final_key)   # None if new leaf key
        d[final_key] = _coerce_value(value_str, original)


_RAMDISK_CANDIDATES = ['/dev/shm', '/run/shm']


def _resolve_tmp_dir(config):
    '''
    Return the temporary working directory for window execution, or None.

    If output.use_tmp is False (default), returns None.
    If output.use_tmp is True and output.tmp_path is set, that path is used.
    If output.use_tmp is True and output.tmp_path is empty, well-known RAM-disc
    locations are tried in order; raises RuntimeError if none are available.
    '''
    output = config.get('output', {})
    if not output.get('use_tmp', False):
        return None

    tmp_path = str(output.get('tmp_path', '')).strip()
    if tmp_path:
        if not os.path.isdir(tmp_path):
            raise FileNotFoundError(
                f"output.tmp_path does not exist or is not a directory: {tmp_path}"
            )
        if not os.access(tmp_path, os.W_OK):
            raise PermissionError(f"output.tmp_path is not writable: {tmp_path}")
        return os.path.abspath(tmp_path)

    for candidate in _RAMDISK_CANDIDATES:
        if os.path.isdir(candidate) and os.access(candidate, os.W_OK):
            return os.path.abspath(candidate)

    raise RuntimeError(
        f"output.use_tmp is enabled but no RAM-disc location is available. "
        f"Tried: {_RAMDISK_CANDIDATES}. "
        f"Set output.tmp_path explicitly (e.g. /scratch) or disable output.use_tmp."
    )


_REQUIRED_CONFIG_KEYS = ('rosetta', 'general_files', 'extra_files', 'input',
                         'parameters', 'rosetta_scripts', 'output')
_REQUIRED_INPUT_KEYS = ('guide_path', 'ligand_files', 'ligand_chain')
_REQUIRED_PARAM_KEYS = ('step', 'smooth', 'conformer_subsampling', 'noise', 'skip',
                        'memory_mode', 'run_stats', 'select_best_by', 'select_next_by',
                        'outlier_rmsd', 'plane_constraint_point_extent')
_REQUIRED_RS_KEYS = ('options', 'protocol_zero', 'protocol_main', 'protocol_stats',
                     'nstruct_main', 'nstruct_stats', 'n_proc', 'mpi_launcher')


class TrajectoryRunner:

    def __init__(self, config_file, overrides=None):
        self.config = read_config(config_file)
        if overrides:
            apply_config_overrides(self.config, overrides)
        self._validate_config()

        self.calldir = os.getcwd()
        self.rootdir = os.path.abspath(self.config['output']['path'])
        self.tmp_dir = _resolve_tmp_dir(self.config)
        self.verbose = self.config['output'].get('verbose', True)

        # Ligand metadata
        self.ligand_info = {
            'chain': self.config['input']['ligand_chain'],
            'resnum': None,
            'atom_names': [],
            'nbr_atom_name': self._read_nbr_atom(),
        }
        if not self.ligand_info['nbr_atom_name']:
            raise ValueError(
                f"NBR_ATOM not found in "
                f"{self.config['input']['ligand_files']}/lig.params"
            )

        # Start window
        start_pdb = self.config['input'].get('start_pdb')
        if start_pdb:
            self.start_pdb_path = os.path.abspath(start_pdb)
            self._separate_start = False
        else:
            self._start_protein_pdb = os.path.abspath(self.config['input']['start_protein_pdb'])
            self._start_ligand_pdb  = os.path.abspath(self.config['input']['start_ligand_pdb'])
            self.start_pdb_path = os.path.join(self.rootdir, 'start_merged.pdb')
            self._separate_start = True
        pdb_for_meta = self._start_ligand_pdb if self._separate_start else self.start_pdb_path
        start_coords, start_nbr_index, names, resnum = PDBHandler.read_window(
            pdb_for_meta,
            self.ligand_info['chain'],
            self.ligand_info['nbr_atom_name'],
        )
        self.ligand_info['resnum'] = resnum
        self.ligand_info['atom_names'] = names
        self.start_nbr = start_coords[start_nbr_index]
        self._conformer_empty_warned = False

        # Raw guide path from file
        guide_path_file = self.config['input']['guide_path']
        self.input_path = read_guide_path(guide_path_file)
        if self.input_path.ndim != 2 or self.input_path.shape[1] != 3:
            raise ValueError(
                f"Guide path must be an (N, 3) array; got shape {self.input_path.shape}"
            )
        if len(self.input_path) < 2:
            raise ValueError(
                f"Guide path must have at least 2 points, got {len(self.input_path)}"
            )

        # These are populated by prepare_path()
        self.guide_path = None
        self.normals = None
        self.L = None
        self.dig = None
        self.end_nbr = None

        # Per-run state
        self.prev_path = self.start_pdb_path
        self.prev_point = None      # set by prepare_path()
        self.prev_prev_point = None # set during the main loop, once the ligand has moved
        self.trajectory = []
        self.total_scores = []
        self.interface_scores = []
        self.stats_total_scores = []
        self.stats_interface_scores = []
        self.processed_windows = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, fresh=False):
        '''
        Run the full trajectory.

        Checkpoint interval is read from config output.checkpoint_interval (default 0 = disabled).
        Unless fresh=True, an existing checkpoint in rootdir is detected automatically:
          - incomplete run  → resume from it
          - complete run    → print a message and return without doing anything
        '''
        checkpoint_interval = self.config.get('output', {}).get('checkpoint_interval', 10)
        if not isinstance(checkpoint_interval, int) or checkpoint_interval < 0:
            raise ValueError(
                f"output.checkpoint_interval must be a non-negative integer, got {checkpoint_interval!r}"
            )

        checkpoint_path = os.path.join(self.rootdir, 'checkpoint.pkl')
        resume_from = None

        if not fresh:
            status, _ = self._check_checkpoint_status()
            if status == 'complete':
                print("Run already complete — checkpoint indicates all windows were processed.")
                print("Use --fresh to force a new run from scratch.")
                return
            elif status == 'incomplete':
                resume_from = checkpoint_path

        if resume_from:
            self.load_checkpoint(resume_from)
            print(f"Resuming from checkpoint: {resume_from}")
            print(f"Skipping {len(self.processed_windows)} already-processed windows.\n")
        else:
            self.prepare_path()
            if self._separate_start:
                PDBHandler.merge_start(
                    self._start_protein_pdb, self._start_ligand_pdb, self.start_pdb_path,
                )

        print("\n------------------------------\n")
        print("LIGAND PATHFINDER : TRAJECTORY")
        print("\n------------------------------\n")
        print(f"The project will be contained here:\n{self.rootdir}\n")

        if not os.path.exists(self.rootdir):
            os.mkdir(self.rootdir)

        print(f"Script called from:\n{self.calldir}\n")
        print("------------------------------\n")

        if not resume_from:
            with open(f'{self.rootdir}/lig.resfile', 'w') as fh:
                fh.write(f"NATRO\nstart\n{self.ligand_info['resnum']} {self.ligand_info['chain']} NATAA\n")

        if self.tmp_dir:
            self.tmp_dir = tempfile.mkdtemp(dir=self.tmp_dir)
            if self.verbose:
                print(f"Temporary working directory:\n{self.tmp_dir}\n")

        processed_set = set(self.processed_windows)

        for i in range(self.L):
            if i in processed_set:
                continue

            window_name = f'window_{str(i).rjust(self.dig, "0")}'
            perm_dp = os.path.join(self.rootdir, window_name)
            work_dp = os.path.join(self.tmp_dir, window_name) if self.tmp_dir else perm_dp

            # Clean up directories left over from an interrupted prior run
            for stale in {work_dp, perm_dp}:
                if os.path.exists(stale):
                    rmtree(stale)

            start_time = time()

            self._setup_window(i, work_dp)
            skipped, plane = self._prepare_window(i, work_dp)

            if skipped:
                if os.path.exists(work_dp):
                    rmtree(work_dp)
                continue

            print(f"STARTED:  window {i}/{self.L-1}")
            if self.verbose and self.tmp_dir:
                print(f"  working in {work_dp}")

            for attempt in range(3):
                if attempt > 0:
                    scale = 0.2 ** attempt
                    if self.verbose:
                        print(f"  attempt {attempt + 1}/3: retrying with rmsd_constraint_sd * {scale}")
                    self._prepare_retry(i, work_dp, attempt)
                self._run_rosetta(i, work_dp)
                try:
                    d_oop, d_ip = self._process_results(i, work_dp, plane)
                    break
                except RuntimeError as exc:
                    if attempt == 2:
                        raise
                    if self.verbose:
                        print(f"  attempt {attempt + 1}/3 failed: {exc}")

            if self.config['parameters']['run_stats']:
                self._run_stats(i, work_dp)

            self._finalize_window(i, work_dp, start_time, d_oop, d_ip)
            self.processed_windows.append(i)
            processed_set.add(i)

            if self.tmp_dir:
                self._transfer_to_rootdir(work_dp, perm_dp)

            if checkpoint_interval > 0 and len(self.processed_windows) % checkpoint_interval == 0:
                self.save_checkpoint(checkpoint_path)
                if self.verbose:
                    print(f"Checkpoint saved ({len(self.processed_windows)}/{self.L} windows).\n")

        self.save_output()
        self.save_checkpoint(checkpoint_path, completed=True)

        if self.tmp_dir and os.path.isdir(self.tmp_dir):
            os.rmdir(self.tmp_dir)

    def prepare_path(self):
        '''Fit spline, sample equidistant guide path, save path files, and init traversal state.'''
        step = self.config['parameters']['step']
        if step <= 0:
            raise ValueError(f"prepare_path: step must be positive, got {step}")

        # Pin the first path point to the starting ligand position
        if (not self.config['parameters'].get('dock_zero', False)
                and np.linalg.norm(self.input_path[0] - self.start_nbr) > step / 5):
            self.input_path = np.vstack([[self.start_nbr], self.input_path[1:]])

        if not os.path.isdir(self.rootdir):
            os.makedirs(self.rootdir, exist_ok=True)

        np.savetxt(f'{self.rootdir}/input_path', self.input_path)

        spline = PathBuilder.spline(self.input_path, self.config['parameters']['smooth'])
        self.guide_path, self.normals = PathBuilder.equalize(spline, step)

        if len(self.guide_path) < 2:
            raise RuntimeError(
                "prepare_path: guide path has fewer than 2 windows after equalization — "
                "check that step size is smaller than the total path length"
            )

        np.savetxt(f'{self.rootdir}/spline', spline)
        np.savetxt(f'{self.rootdir}/guide_path', self.guide_path)
        np.savetxt(f'{self.rootdir}/normals', self.normals)
        PDBHandler.path_to_pdb(f'{self.rootdir}/guide_path.pdb', self.guide_path)

        self._warn_path_curvature()

        self.L = len(self.guide_path)
        self.dig = len(str(self.L))
        self.end_nbr = self.guide_path[-1]
        guide_step = self.guide_path[1] - self.guide_path[0]
        if self.config['parameters'].get('dock_zero', False):
            self.prev_point = self.guide_path[0] - guide_step
        else:
            self.prev_point = self.start_nbr - guide_step

    def _warn_path_curvature(self):
        '''Report max step-to-step turning angle; warn if > 25°; show 2x/5x smooth alternatives.'''
        smooth = self.config['parameters']['smooth']
        step   = self.config['parameters']['step']

        max_current = _path_max_turn_deg(self.guide_path)

        gp_2x = PathBuilder.equalize(PathBuilder.spline(self.input_path, smooth * 2), step)[0]
        gp_5x = PathBuilder.equalize(PathBuilder.spline(self.input_path, smooth * 5), step)[0]
        max_2x = _path_max_turn_deg(gp_2x)
        max_5x = _path_max_turn_deg(gp_5x)

        if self.verbose:
            print(f"Guide path curvature: max turn = {max_current:.1f}° (smooth={smooth})")
            print(f"  informational: 2x smooth ({smooth * 2}) → {max_2x:.1f}°,  "
                  f"5x smooth ({smooth * 5}) → {max_5x:.1f}°")

        if max_current > 25.0:
            flagged = []
            gp = self.guide_path
            for i in range(1, len(gp) - 1):
                v1 = gp[i] - gp[i - 1]
                v2 = gp[i + 1] - gp[i]
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if n1 > 0 and n2 > 0:
                    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
                    turn = np.degrees(np.arccos(cos_a))
                    if turn > 25.0:
                        flagged.append((i, turn))
            print(f"  WARNING: {len(flagged)} step(s) exceed 25°; "
                  f"the angle constraint may not enforce direction at these points:")
            for idx, deg in flagged:
                print(f"    gp[{idx}]: {deg:.1f}°")

    def save_checkpoint(self, path, completed=False):
        '''Pickle all mutable run state to `path` so the run can be resumed later.'''
        state = {
            'version':              _CHECKPOINT_VERSION,
            'completed':            completed,
            'rootdir':              self.rootdir,
            'ligand_info':          self.ligand_info,
            'prev_path':            self.prev_path,
            'prev_point':           self.prev_point,
            'prev_prev_point':      self.prev_prev_point,
            'trajectory':           self.trajectory,
            'total_scores':         self.total_scores,
            'interface_scores':     self.interface_scores,
            'stats_total_scores':   self.stats_total_scores,
            'stats_interface_scores': self.stats_interface_scores,
            'processed_windows':    self.processed_windows,
            'input_path':           self.input_path,
            'guide_path':           self.guide_path,
            'normals':              self.normals,
            'L':                    self.L,
            'dig':                  self.dig,
            'end_nbr':              self.end_nbr,
        }
        with open(path, 'wb') as fh:
            pkl.dump(state, fh)

        sidecar_path = os.path.splitext(path)[0] + '_meta.json'
        with open(sidecar_path, 'w') as fh:
            json.dump({
                'version':   _CHECKPOINT_VERSION,
                'completed': completed,
                'rootdir':   self.rootdir,
            }, fh)

    def load_checkpoint(self, path):
        '''Restore run state from a checkpoint file created by save_checkpoint.'''
        if not os.path.isfile(path):
            raise FileNotFoundError(f"load_checkpoint: checkpoint file not found: {path}")

        with open(path, 'rb') as fh:
            state = pkl.load(fh)

        if state.get('version') != _CHECKPOINT_VERSION:
            raise ValueError(
                f"load_checkpoint: checkpoint version mismatch "
                f"(file={state.get('version')}, expected={_CHECKPOINT_VERSION})"
            )
        if os.path.abspath(state['rootdir']) != self.rootdir:
            raise ValueError(
                f"load_checkpoint: checkpoint was created for a different output directory\n"
                f"  checkpoint: {state['rootdir']}\n"
                f"  current:    {self.rootdir}"
            )
        if not state['processed_windows']:
            raise ValueError("load_checkpoint: checkpoint contains no completed windows — nothing to resume")
        if not os.path.isfile(state['prev_path']):
            raise FileNotFoundError(
                f"load_checkpoint: prev_path from checkpoint no longer exists: {state['prev_path']}"
            )

        for key in ('prev_point', 'prev_prev_point'):
            val = state.get(key)
            assert val is None or isinstance(val, np.ndarray), (
                f"load_checkpoint: {key} must be ndarray or None, got {type(val)}"
            )

        self.ligand_info            = state['ligand_info']
        self.prev_path              = state['prev_path']
        self.prev_point             = state['prev_point']
        self.prev_prev_point        = state['prev_prev_point']
        self.trajectory             = state['trajectory']
        self.total_scores           = state['total_scores']
        self.interface_scores       = state['interface_scores']
        self.stats_total_scores     = state['stats_total_scores']
        self.stats_interface_scores = state['stats_interface_scores']
        self.processed_windows      = state['processed_windows']
        self.input_path             = state['input_path']
        self.guide_path             = state['guide_path']
        self.normals                = state['normals']
        self.L                      = state['L']
        self.dig                    = state['dig']
        self.end_nbr                = state['end_nbr']

    def _check_checkpoint_status(self):
        '''
        Inspect the checkpoint file in rootdir without fully restoring state.

        Returns (status, path) where status is one of:
          'complete'   — checkpoint exists and the run finished successfully
          'incomplete' — checkpoint exists but the run did not finish
          None         — no checkpoint file found
        Raises RuntimeError/ValueError if the checkpoint is unreadable or incompatible.
        '''
        checkpoint_path = os.path.join(self.rootdir, 'checkpoint.pkl')
        if not os.path.isfile(checkpoint_path):
            return None, checkpoint_path

        sidecar_path = os.path.join(self.rootdir, 'checkpoint_meta.json')
        if os.path.isfile(sidecar_path):
            with open(sidecar_path, 'r') as fh:
                meta = json.load(fh)
            version  = meta.get('version')
            rootdir  = meta.get('rootdir', '')
            completed = meta.get('completed', False)
        else:
            try:
                with open(checkpoint_path, 'rb') as fh:
                    state = pkl.load(fh)
            except Exception as e:
                raise RuntimeError(f"Could not read checkpoint {checkpoint_path}: {e}")
            version   = state.get('version')
            rootdir   = state.get('rootdir', '')
            completed = state.get('completed', False)

        if version != _CHECKPOINT_VERSION:
            raise ValueError(
                f"Checkpoint version mismatch "
                f"(file={version}, expected={_CHECKPOINT_VERSION}). "
                f"Delete {checkpoint_path} to start a fresh run."
            )
        if os.path.abspath(rootdir) != self.rootdir:
            raise ValueError(
                f"Checkpoint belongs to a different output directory:\n"
                f"  checkpoint: {rootdir}\n"
                f"  current:    {self.rootdir}"
            )

        status = 'complete' if completed else 'incomplete'
        return status, checkpoint_path

    def save_output(self):
        '''Write trajectory coordinates, score arrays, stats pickle, and trajectory PDB files.'''
        if not self.processed_windows:
            raise RuntimeError("save_output: no windows were processed — nothing to save")

        np.savetxt(f'{self.rootdir}/nbr_trajectory.txt', self.trajectory)
        np.savetxt(f'{self.rootdir}/total_scores.txt', self.total_scores)
        np.savetxt(f'{self.rootdir}/interface_scores.txt', self.interface_scores)

        stats = {
            'total_scores': self.stats_total_scores,
            'interface_scores': self.stats_interface_scores,
        }
        with open(f'{self.rootdir}/stats.pkl', 'wb') as fh:
            pkl.dump(stats, fh)

        lig_chain = self.ligand_info['chain']
        with open(f'{self.rootdir}/lig_trajectory.pdb', 'w') as lig_traj, \
             open(f'{self.rootdir}/full_trajectory.pdb', 'w') as full_traj:
            for i in self.processed_windows:
                wnd = f'window_{str(i).rjust(self.dig, "0")}'
                best_pdb = f'{self.rootdir}/{wnd}/best.pdb'
                if not os.path.isfile(best_pdb):
                    raise FileNotFoundError(
                        f"save_output: best.pdb missing for window {i}: {best_pdb}"
                    )
                with open(best_pdb, 'r') as final:
                    for line in final:
                        if line.startswith('ATOM') or line.startswith('HETATM'):
                            full_traj.write(line)
                            if line[21] == lig_chain:
                                lig_traj.write(line)
                lig_traj.write('ENDMDL\n')
                full_traj.write('ENDMDL\n')

                if self.config['output'].get('archive', False):
                    run_tool(['zip', '-rm', f'{wnd}.zip', wnd], cwd=self.rootdir)


    # ------------------------------------------------------------------
    # Private per-window steps
    # ------------------------------------------------------------------

    def _prepare_retry(self, i, work_dp, attempt):
        '''Clear Rosetta outputs and rewrite constraints with rmsd_constraint_sd * 0.2**attempt.'''
        score_sc = os.path.join(work_dp, 'score.sc')
        if os.path.isfile(score_sc):
            os.remove(score_sc)
        for f in glob.glob(os.path.join(work_dp, 'results', '*.pdb')):
            os.remove(f)

        params = dict(self.config['parameters'])
        if i == 0 and params.get('dock_zero', False):
            params['rmsd_constraint'] = False
        params['rmsd_constraint_sd'] = params['rmsd_constraint_sd'] * (0.2 ** attempt)
        cst_i_to_end = -1 if i == 0 else (self.L - i - 1)
        cst = ConstraintBuilder.create_constraints_trajectory(self.ligand_info, params, cst_i_to_end)
        with open(os.path.join(work_dp, 'constraints.cst'), 'w') as fh:
            fh.write(cst)

    def _transfer_to_rootdir(self, tmp_dp, perm_dp):
        '''
        Copy a completed window directory from tmp storage to the permanent rootdir,
        update prev_path if it pointed into the tmp location, then delete the tmp copy.
        '''
        if not os.path.isdir(tmp_dp):
            raise FileNotFoundError(
                f"_transfer_to_rootdir: tmp window directory not found: {tmp_dp}"
            )
        copytree(tmp_dp, perm_dp)
        if self.prev_path.startswith(tmp_dp + os.sep) or self.prev_path == tmp_dp:
            self.prev_path = perm_dp + self.prev_path[len(tmp_dp):]
        rmtree(tmp_dp)

    def _setup_window(self, i, work_dp):
        '''Create directory structure and copy static input files into work_dp.'''
        os.mkdir(work_dp)
        os.mkdir(f'{work_dp}/results')
        os.mkdir(f'{work_dp}/stats')
        os.mkdir(f'{work_dp}/grid')

        for src, dst_name in [
            (self.config['rosetta_scripts']['options'], 'options'),
            (self.config['rosetta_scripts']['protocol_stats'], 'stats.xml'),
            (f'{self.config["input"]["ligand_files"]}/lig.params', None),
            (f'{self.rootdir}/lig.resfile', None),
            (f'{self.config["general_files"]}/VRT1.params', None),
        ]:
            if not os.path.isfile(src):
                raise FileNotFoundError(f"_setup_window: required file not found: {src}")
            copy(src, f'{work_dp}/{dst_name}' if dst_name else work_dp)

        for fname in self.config['extra_files']:
            if not os.path.isfile(fname):
                raise FileNotFoundError(f"_setup_window: extra file not found: {fname}")
            copy(fname, work_dp)

    def _prepare_window(self, i, work_dp):
        '''
        Build task.pdb, task.xml, and constraints.cst for window i.

        Returns (skipped, plane_info) where:
          skipped    — True if the skip condition triggered (caller must clean up work_dp)
          plane_info — (plane_point, plane_normal) arrays for window i (i > 0), else None
        '''
        params = self.config['parameters']
        i_to_end = self.L - i - 1

        if i == 0:
            self._prepare_window_zero(work_dp)
            cst_i_to_end = -1
            plane_info = None
            if params.get('dock_zero', False):
                params = {**params, 'rmsd_constraint': False}
        else:
            plane_pt = self.guide_path[i]
            plane_n  = self.normals[i]

            if params['skip']:
                v_guide = self.guide_path[i] - self.guide_path[i - 1]
                prev_v = self.prev_point + v_guide
                proj = _project_onto_plane(self.prev_point, plane_pt, plane_n)
                angle = GeometryUtils.get_angle(prev_v, self.prev_point, proj)
                if angle > 90:
                    print(f"SKIPPED:  window {i}/{self.L-1}  "
                          f"(guide path turns back; plane angle = {angle:.1f}°)\n")
                    return True, None  # skipped; caller handles rmtree

            self._prepare_window_main(i, work_dp, i_to_end, plane_pt, plane_n)
            cst_i_to_end = i_to_end
            plane_info = (plane_pt, plane_n)

            if params['in_plane_dev_constraint']:
                end_padding_w = int(params['end_padding'] / params['step'])
                if i_to_end == 0 and end_padding_w > 0:
                    if self.verbose:
                        print(f"  end mode: harmonic snap to endpoint (ipdc_sd={params['ipdc_sd']:.3f})")
                elif 0 < i_to_end <= end_padding_w:
                    effective_d0 = params['ipdc_d0'] / end_padding_w * i_to_end
                    if self.verbose:
                        print(f"  end padding: {i_to_end} windows to end, effective ipdc_d0={effective_d0:.3f}")

        cst = ConstraintBuilder.create_constraints_trajectory(self.ligand_info, params, cst_i_to_end)
        with open(f'{work_dp}/constraints.cst', 'w') as fh:
            fh.write(cst)

        conformers_src = f'{os.path.abspath(self.config["input"]["ligand_files"])}/lig_conformers.pdb'
        if not os.path.isfile(conformers_src):
            raise FileNotFoundError(
                f"_prepare_window: conformers file not found: {conformers_src}"
            )
        use_subsampling = params['conformer_subsampling'] and not (i == 0 and params.get('dock_zero', False))
        subsampled = False
        if use_subsampling:
            subsampled = ConformerSampler.subsample_pdb(
                self.prev_path,
                conformers_src,
                f'{work_dp}/lig_conformers.pdb',
                params['conformer_rmsd_threshold'],
                self.ligand_info['chain'],
                self.ligand_info['nbr_atom_name'],
            )
            if not subsampled and not self._conformer_empty_warned:
                print("WARNING: lig_conformers.pdb contains no conformers; "
                      "using original file without subsampling (rigid/small ligand?)")
                self._conformer_empty_warned = True
        if not subsampled:
            copy(conformers_src, work_dp)

        return False, plane_info

    def _prepare_window_zero(self, work_dp):
        if self.config['parameters'].get('dock_zero', False):
            protocol = self.config['rosetta_scripts']['protocol_zero_dock']
            label = '_prepare_window_zero: protocol_zero_dock XML not found'
        else:
            protocol = self.config['rosetta_scripts']['protocol_zero']
            label = '_prepare_window_zero: protocol_zero XML not found'
        if not os.path.isfile(protocol):
            raise FileNotFoundError(f"{label}: {protocol}")
        if self.config['parameters'].get('dock_zero', False):
            xml = PDBHandler.make_xml_strings(protocol, [self.guide_path[0]])
            with open(f'{work_dp}/task.xml', 'w') as fh:
                fh.write(xml)
        else:
            copy(protocol, f'{work_dp}/task.xml')
        tri1, tri2, tri3 = GeometryUtils.equilateral_triangle_vertices(
            self.guide_path[0], self.normals[0],
            self.config['parameters']['plane_constraint_point_extent'],
        )
        V = TrajVirt
        virt_coords = [None] * len(V)
        virt_coords[V.TRI1]      = tri1
        virt_coords[V.TRI2]      = tri2
        virt_coords[V.TRI3]      = tri3
        virt_coords[V.PREV_PREV] = tri3                  # unused in start_mode; fill with neutral
        virt_coords[V.PREV]      = self.guide_path[0]
        virt_coords[V.CURR]      = self.guide_path[0]
        virt_coords[V.NEXT]      = self.guide_path[0]
        PDBHandler.prepare_start(
            self.prev_path, f'{work_dp}/task.pdb', 0,
            virt_coords, self.ligand_info, self.config['input'].get('header_pdb', False),
        )

    def _prepare_window_main(self, i, work_dp, i_to_end, plane_pt, plane_n):
        if self.prev_prev_point is None:
            # Use an extrapolated fallback on the first non-zero window before the ligand
            # has moved enough to set prev_prev_point via _process_results
            self.prev_prev_point = self.prev_point - (self.guide_path[1] - self.guide_path[0])

        params = self.config['parameters']
        protocol_main = self.config['rosetta_scripts']['protocol_main']
        if not os.path.isfile(protocol_main):
            raise FileNotFoundError(
                f"_prepare_window_main: protocol_main XML not found: {protocol_main}"
            )

        # Project previous point onto the current plane (with optional noise)
        random_shift = np.zeros(3)
        if params['noise']:
            random_shift += np.random.normal(0, params['step'] * 0.33, 3)
        noised_prev = self.prev_point + random_shift
        proj = _project_onto_plane(noised_prev, plane_pt, plane_n)

        new_start = proj

        xml = PDBHandler.make_xml_strings(protocol_main, [new_start])
        with open(f'{work_dp}/task.xml', 'w') as fh:
            fh.write(xml)

        displacement = new_start - self.prev_point
        tri1, tri2, tri3 = GeometryUtils.equilateral_triangle_vertices(
            proj, self.normals[i], params['plane_constraint_point_extent'],
        )

        V = TrajVirt
        virt_coords = [None] * len(V)
        virt_coords[V.TRI1]      = tri1
        virt_coords[V.TRI2]      = tri2
        virt_coords[V.TRI3]      = tri3
        virt_coords[V.PREV_PREV] = self.prev_prev_point
        virt_coords[V.CURR]      = self.guide_path[i]
        if i_to_end == 0:
            virt_coords[V.PREV] = self.guide_path[i]   # ligand snaps to guide at terminal window
            virt_coords[V.NEXT] = self.guide_path[i]
        else:
            virt_coords[V.PREV] = self.prev_point
            virt_coords[V.NEXT] = self.guide_path[i + 1]

        PDBHandler.prepare_start(
            self.prev_path, f'{work_dp}/task.pdb',
            displacement, virt_coords, self.ligand_info, self.config['input'].get('header_pdb', False),
        )

    def _build_mpi_prefix(self, nstruct):
        '''Return the MPI launcher prefix as a list, or [] for serial execution.'''
        n_proc = self.config['rosetta_scripts']['n_proc']
        effective = min(n_proc, nstruct)
        if effective > 1:
            launcher = self.config['rosetta_scripts']['mpi_launcher']
            return [launcher, '--oversubscribe', '-np', str(effective)]
        return []

    def _run_rosetta(self, i, work_dp):
        rs_bin = os.path.abspath(self.config['rosetta'])
        if i == 0:
            if self.config['parameters'].get('dock_zero', False):
                nstruct = self.config['rosetta_scripts']['nstruct_zero']
            else:
                nstruct = 1
        else:
            nstruct = self.config['rosetta_scripts']['nstruct_main']
        if nstruct < 1:
            raise ValueError(f"_run_rosetta: nstruct must be >= 1, got {nstruct}")
        prefix = self._build_mpi_prefix(nstruct)
        run_rosetta(
            prefix + [rs_bin, '@options', '-nstruct', str(nstruct),
                      '-out:path:pdb', './results', '-parser:protocol', 'task.xml',
                      '-in:file:s', 'task.pdb'],
            cwd=work_dp,
            log_path=f'{work_dp}/task.log', err_path=f'{work_dp}/task.err',
        )

    def _process_results(self, i, work_dp, plane_info):
        '''
        Pick best/next result, update tracking state, compress output PDBs.
        Returns (d_oop, d_ip) diagnostics for reporting.
        '''
        params = self.config['parameters']
        outlier_rmsd = float('inf') if (i == 0 and params.get('dock_zero', False)) else params['outlier_rmsd']

        best, i_score, t_score, cst_dict = ScoreParser.report_final(
            work_dp, self.ligand_info,
            by=params['select_best_by'], outlier_rmsd=outlier_rmsd,
        )
        self.total_scores.append(t_score)
        self.interface_scores.append(i_score)
        copy(best, f'{work_dp}/best.pdb')

        next_, _, _, _ = ScoreParser.report_final(
            work_dp, self.ligand_info,
            by=params['select_next_by'], outlier_rmsd=outlier_rmsd,
        )
        copy(next_, f'{work_dp}/next.pdb')

        total_cst = sum(cst_dict.values())
        if self.verbose:
            print(
                f"  cst:  angle={cst_dict['angle_constraint']:.2f}"
                f"  pair={cst_dict['atom_pair_constraint']:.2f}"
                f"  dihedral={cst_dict['dihedral_constraint']:.2f}"
                f"  total={total_cst:.2f}"
            )

        if params['memory_mode']:
            copy(f'{work_dp}/next.pdb', f'{work_dp}/final.pdb')
        else:
            PDBHandler.swap_lig_coords(
                f'{work_dp}/next.pdb', f'{work_dp}/task.pdb',
                f'{work_dp}/final.pdb', self.ligand_info,
            )

        self.prev_path = f'{work_dp}/final.pdb'
        prev_coords, nbr_index, names, _ = PDBHandler.read_window(
            self.prev_path, self.ligand_info['chain'], self.ligand_info['nbr_atom_name'],
        )

        if i > 0 and plane_info is not None:
            plane_pt, plane_n = plane_info
            proj = _project_onto_plane(prev_coords[nbr_index], plane_pt, plane_n)
            d_oop = np.linalg.norm(proj - prev_coords[nbr_index])
        else:
            d_oop = 0.0

        d_ip = np.linalg.norm(self.guide_path[i] - prev_coords[nbr_index])

        if np.linalg.norm(self.prev_point - prev_coords[nbr_index]) > 0.33 * params['step']:
            self.prev_prev_point = self.prev_point
        self.prev_point = prev_coords[nbr_index]

        if self.prev_prev_point is not None:
            guide_step = self.guide_path[i] - self.guide_path[i - 1]
            step_dir = self.prev_point - self.prev_prev_point
            if np.linalg.norm(step_dir) > 0:
                guide_step_hat = guide_step / np.linalg.norm(guide_step)
                backward = np.dot(step_dir, guide_step) < 0
                conflict = (
                    not backward
                    and i + 1 < self.L
                    and np.dot(step_dir, self.guide_path[i + 1] - self.prev_point) < 0
                )
                if backward or conflict:
                    self.prev_prev_point = self.prev_point - guide_step_hat * params['step']
        self.trajectory.append(self.prev_point)
        self.ligand_info['atom_names'] = names

        pdb_files = glob.glob(os.path.join(work_dp, 'results', '*.pdb'))
        if pdb_files:
            run_tool(['gzip'] + pdb_files)

        return d_oop, d_ip

    def _run_stats(self, i, work_dp):
        rs_bin = os.path.abspath(self.config['rosetta'])
        nstruct_stats = self.config['rosetta_scripts']['nstruct_stats']
        if nstruct_stats < 1:
            raise ValueError(f"_run_stats: nstruct_stats must be >= 1, got {nstruct_stats}")

        PDBHandler.swap_lig_coords(
            f'{work_dp}/best.pdb', f'{work_dp}/task.pdb',
            f'{work_dp}/stats.pdb', self.ligand_info,
        )
        prefix = self._build_mpi_prefix(nstruct_stats)
        run_rosetta(
            prefix + [rs_bin, '@options', '-in:file:s', 'stats.pdb',
                      '-out:path:pdb', './stats', '-out:file:scorefile', 'stats.sc',
                      '-parser:protocol', 'stats.xml', '-nstruct', str(nstruct_stats)],
            cwd=work_dp,
            log_path=f'{work_dp}/stats.log', err_path=f'{work_dp}/stats.err',
        )
        stats_i, stats_t = ScoreParser.report_stats(work_dp)
        self.stats_interface_scores.append(stats_i)
        self.stats_total_scores.append(stats_t)

        pdb_files = glob.glob(os.path.join(work_dp, 'stats', '*.pdb'))
        if pdb_files:
            run_tool(['gzip'] + pdb_files)

    def _finalize_window(self, i, work_dp, start_time, d_oop, d_ip):
        elapsed = time() - start_time
        print(f"FINISHED: window {i}/{self.L-1}")
        print(f"in {timedelta(seconds=elapsed)}")
        if self.verbose:
            print(f"d_to_plane {d_oop:.3f}")
            print(f"d_in_plane {d_ip:.3f}\n")
        rmtree(f'{work_dp}/grid')

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_nbr_atom(self):
        params_path = f'{self.config["input"]["ligand_files"]}/lig.params'
        if not os.path.isfile(params_path):
            raise FileNotFoundError(f"Ligand params file not found: {params_path}")
        with open(params_path, 'r') as fh:
            for line in fh:
                if line.startswith('NBR_ATOM'):
                    return line.strip().split()[1]
        return ''

    def _validate_config(self):
        missing = [k for k in _REQUIRED_CONFIG_KEYS if k not in self.config]
        if missing:
            raise KeyError(f"Config is missing required top-level keys: {missing}")

        missing_input = [k for k in _REQUIRED_INPUT_KEYS if k not in self.config['input']]
        if missing_input:
            raise KeyError(f"Config 'input' section is missing keys: {missing_input}")

        missing_params = [k for k in _REQUIRED_PARAM_KEYS if k not in self.config['parameters']]
        if missing_params:
            raise KeyError(f"Config 'parameters' section is missing keys: {missing_params}")

        missing_rs = [k for k in _REQUIRED_RS_KEYS if k not in self.config['rosetta_scripts']]
        if missing_rs:
            raise KeyError(f"Config 'rosetta_scripts' section is missing keys: {missing_rs}")

        rs_bin = os.path.abspath(self.config['rosetta'])
        if not os.path.isfile(rs_bin):
            raise FileNotFoundError(f"Rosetta binary not found: {rs_bin}")

        n_proc = self.config['rosetta_scripts']['n_proc']
        if not isinstance(n_proc, int) or n_proc < 1:
            raise ValueError(
                f"rosetta_scripts.n_proc must be a positive integer, got {n_proc!r}"
            )
        is_mpi = '.mpi.' in os.path.basename(rs_bin)
        if n_proc > 1 and not is_mpi:
            raise ValueError(
                f"rosetta_scripts.n_proc={n_proc} but the Rosetta binary does not appear "
                f"to be an MPI build (no '.mpi.' in filename): {rs_bin}\n"
                f"Use the MPI binary or set n_proc=1."
            )
        if n_proc > self.config['rosetta_scripts']['nstruct_main']:
            raise ValueError(
                f"rosetta_scripts.n_proc ({n_proc}) exceeds nstruct_main "
                f"({self.config['rosetta_scripts']['nstruct_main']}); extra ranks would be idle."
            )
        if self.config['parameters'].get('run_stats', False):
            if n_proc > self.config['rosetta_scripts']['nstruct_stats']:
                raise ValueError(
                    f"rosetta_scripts.n_proc ({n_proc}) exceeds nstruct_stats "
                    f"({self.config['rosetta_scripts']['nstruct_stats']}); extra ranks would be idle."
                )
        if self.config['parameters'].get('dock_zero', False):
            proto = self.config['rosetta_scripts'].get('protocol_zero_dock', '')
            if not proto or not os.path.isfile(proto):
                raise FileNotFoundError(
                    f"dock_zero is enabled but protocol_zero_dock not found: {proto!r}"
                )
            nstruct_zero = self.config['rosetta_scripts'].get('nstruct_zero')
            if not isinstance(nstruct_zero, int) or nstruct_zero < 1:
                raise ValueError(
                    f"rosetta_scripts.nstruct_zero must be a positive integer, got {nstruct_zero!r}"
                )
            if n_proc > nstruct_zero:
                raise ValueError(
                    f"rosetta_scripts.n_proc ({n_proc}) exceeds nstruct_zero "
                    f"({nstruct_zero}); extra ranks would be idle."
                )

        if not os.path.isdir(self.config['input']['ligand_files']):
            raise FileNotFoundError(
                f"Ligand files directory not found: {self.config['input']['ligand_files']}"
            )
        start_pdb         = self.config['input'].get('start_pdb')
        start_protein_pdb = self.config['input'].get('start_protein_pdb')
        start_ligand_pdb  = self.config['input'].get('start_ligand_pdb')
        if start_pdb:
            if not os.path.isfile(start_pdb):
                raise FileNotFoundError(f"Start PDB not found: {start_pdb}")
        elif start_protein_pdb and start_ligand_pdb:
            if not os.path.isfile(start_protein_pdb):
                raise FileNotFoundError(f"Start protein PDB not found: {start_protein_pdb}")
            if not os.path.isfile(start_ligand_pdb):
                raise FileNotFoundError(f"Start ligand PDB not found: {start_ligand_pdb}")
        else:
            raise KeyError(
                "Config 'input' must contain either 'start_pdb' or both "
                "'start_protein_pdb' and 'start_ligand_pdb'"
            )
        header = self.config['input'].get('header_pdb')
        if header and not os.path.isfile(header):
            raise FileNotFoundError(f"Header PDB not found: {header}")

        rs_files = {
            'options': self.config['rosetta_scripts']['options'],
            'protocol_zero': self.config['rosetta_scripts']['protocol_zero'],
            'protocol_main': self.config['rosetta_scripts']['protocol_main'],
            'protocol_stats': self.config['rosetta_scripts']['protocol_stats'],
        }
        for key, path in rs_files.items():
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"Config rosetta_scripts['{key}'] not found: {path}"
                )

        vrt_params = f'{self.config["general_files"]}/VRT1.params'
        if not os.path.isfile(vrt_params):
            raise FileNotFoundError(f"VRT1.params not found: {vrt_params}")

        check_protocol_chain(
            list(rs_files.values()) + [self.config['rosetta_scripts'].get('protocol_zero_dock')],
            self.config['input']['ligand_chain'],
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='LigPF — generate a ligand trajectory along a guide path',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Config overrides (--set) use dot-notation matching the JSON structure:\n'
            '  --set output.checkpoint_interval=10\n'
            '  --set parameters.step=0.5\n'
            '  --set rosetta_scripts.nstruct_main=20\n'
            '  --set output.path=/new/output/dir\n'
        ),
    )
    parser.add_argument('config', help='JSON run configuration file')
    parser.add_argument(
        '--set', metavar='KEY=VALUE', action='append', dest='overrides', default=[],
        help='Override any config value (repeatable); dot-notation, e.g. --set parameters.step=0.5',
    )
    parser.add_argument(
        '--fresh', action='store_true',
        help='Ignore any existing checkpoint and start from scratch',
    )
    args = parser.parse_args()

    runner = TrajectoryRunner(args.config, overrides=args.overrides)
    runner.run(fresh=args.fresh)
