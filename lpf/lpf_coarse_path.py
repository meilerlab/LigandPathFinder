# LIGAND PATH FINDER
# CoarsePath script
# Given a final point, create a guess ligand-specific path from the binding pose to it
# Is not guaranteed to converge — depends on the choice of the final point.

import numpy as np
import os
import argparse
import glob
from shutil import copy, rmtree
from datetime import timedelta
from time import time
from copy import copy as copyobj

from util import (
    read_config,
    read_guide_path,
    check_protocol_chain,
    run_rosetta,
    run_tool,
    CoarseVirt,
    GeometryUtils,
    PDBHandler,
    ScoreParser,
    ConstraintBuilder,
    ConformerSampler,
)
from bintree import BinTree, Node
from lpf_trajectory import apply_config_overrides

_REQUIRED_CONFIG_KEYS = ('rosetta', 'general_files', 'extra_files', 'input',
                         'parameters', 'rosetta_scripts', 'output')
_REQUIRED_INPUT_KEYS  = ('guide_path', 'ligand_files', 'ligand_chain')
_REQUIRED_PARAM_KEYS  = ('step', 'conformer_subsampling', 'memory_mode',
                         'rmsd_constraint_sd', 'select_best_by',
                         'in_plane_dev_constraint', 'ipdc_type',
                         'plane_constraint_point_extent',
                         'endpoint_jitter', 'endpoint_jitter_radius')
_REQUIRED_RS_KEYS     = ('options', 'protocol', 'nstruct', 'n_proc', 'mpi_launcher')


class CoarsePathRunner:

    def __init__(self, config_file, overrides=None):
        self.config = read_config(config_file)
        if overrides:
            apply_config_overrides(self.config, overrides)
        self._validate_config()

        self.rootdir = os.path.abspath(self.config['output']['path'])
        self.verbose = self.config['output'].get('verbose', True)

        # Ligand metadata
        self.ligand_info = {
            'chain':         self.config['input']['ligand_chain'],
            'resnum':        None,
            'atom_names':    [],
            'nbr_atom_name': self._read_nbr_atom(),
        }
        if not self.ligand_info['nbr_atom_name']:
            raise ValueError(
                f"NBR_ATOM not found in "
                f"{self.config['input']['ligand_files']}/lig.params"
            )

        # Start pose
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

        # Path endpoints
        guide_path_file = self.config['input']['guide_path']
        input_path = read_guide_path(guide_path_file)
        if input_path.ndim != 2 or input_path.shape[1] != 3:
            raise ValueError(
                f"Guide path must be an (N, 3) array; got shape {input_path.shape}"
            )

        leftmost  = input_path[0].copy()
        rightmost = input_path[-1].copy()

        if (not self.config['parameters'].get('dock_zero', False)
                and np.linalg.norm(leftmost - self.start_nbr) > self.config['parameters']['step'] / 5):
            leftmost = self.start_nbr.copy()

        # Optionally displace the endpoint off the straight axis for ensemble diversity:
        # each independent run lands at a slightly different endpoint, which improves
        # coverage when results from many runs are pooled and clustered.
        if self.config['parameters']['endpoint_jitter']:
            jitter_r = self.config['parameters']['endpoint_jitter_radius']
            lr = rightmost - leftmost
            normal = lr / np.linalg.norm(lr)
            rightmost = np.array(
                GeometryUtils.sample_points_in_disc(rightmost, normal, jitter_r, 1)[0]
            )

        self.leftmost  = leftmost
        self.rightmost = rightmost

        # Build initial binary tree with one root node at the midpoint
        N_left  = Node(xyz=leftmost,  origin='pseudo')
        N_right = Node(xyz=rightmost, origin='pseudo')
        start_point = np.mean([leftmost, rightmost], axis=0)
        self.tree = BinTree()
        self.tree.root = Node(
            init_xyz=start_point,
            xyz=None,
            Lm=N_left,
            Rm=N_right,
            init_pdb=self.start_pdb_path,
            id=0,
            origin='root',
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self):
        if self.verbose:
            print("\n------------------------------\n")
        print("LIGAND PATHFINDER : COARSEPATH")
        if self.verbose:
            print("\n------------------------------\n")
        print(f"The project will be contained here:\n{self.rootdir}\n")

        if not os.path.exists(self.rootdir):
            os.mkdir(self.rootdir)

        if self._separate_start:
            PDBHandler.merge_start(
                self._start_protein_pdb, self._start_ligand_pdb, self.start_pdb_path,
            )

        with open(f'{self.rootdir}/lig.resfile', 'w') as fh:
            fh.write(
                f"NATRO\nstart\n"
                f"{self.ligand_info['resnum']} {self.ligand_info['chain']} NATAA\n"
            )

        if self.verbose:
            print("------------------------------\n")

        if self.config['parameters'].get('dock_zero', False):
            docked_pdb = self._run_dock_zero()
            self.start_pdb_path     = docked_pdb
            self.tree.root.init_pdb = docked_pdb

        self._process_node(self.tree.root, 0)

        node_sequence = self.tree.unravel()
        self._save_output(node_sequence)

    # ------------------------------------------------------------------
    # Private per-node steps
    # ------------------------------------------------------------------

    def _run_dock_zero(self):
        '''
        Dock the ligand toward the guide path's first point (self.leftmost) before
        the bisection tree starts, for cases where the starting structure has the
        ligand present but not yet in position. Returns the path to final.pdb.
        '''
        if self.verbose:
            print("STARTED: dock zero")
        start_time = time()

        work_dp = os.path.join(self.rootdir, 'dock0')
        os.mkdir(work_dp)
        os.mkdir(f'{work_dp}/results')
        os.mkdir(f'{work_dp}/grid')

        for src, dst_name in [
            (self.config['rosetta_scripts']['options'],              'options'),
            (f'{self.config["input"]["ligand_files"]}/lig.params',   None),
            (f'{self.rootdir}/lig.resfile',                          None),
            (f'{self.config["general_files"]}/VRT1.params',          None),
        ]:
            if not os.path.isfile(src):
                raise FileNotFoundError(f"_run_dock_zero: required file not found: {src}")
            copy(src, f'{work_dp}/{dst_name}' if dst_name else work_dp)

        for fname in self.config['extra_files']:
            if not os.path.isfile(fname):
                raise FileNotFoundError(f"_run_dock_zero: extra file not found: {fname}")
            copy(fname, work_dp)

        step   = self.config['parameters']['step']
        normal = self.rightmost - self.leftmost
        normal /= np.linalg.norm(normal)
        tri1, tri2, tri3 = GeometryUtils.equilateral_triangle_vertices(
            self.leftmost, normal,
            self.config['parameters']['plane_constraint_point_extent'],
        )

        V = CoarseVirt
        virt_coords = [None] * len(V)
        virt_coords[V.TRI1] = tri1
        virt_coords[V.TRI2] = tri2
        virt_coords[V.TRI3] = tri3
        virt_coords[V.LM]   = self.leftmost
        virt_coords[V.RM]   = self.leftmost
        virt_coords[V.CURR] = self.leftmost
        virt_coords[V.LLM]  = self.leftmost
        virt_coords[V.RRM]  = self.leftmost
        PDBHandler.prepare_start(
            self.start_pdb_path, f'{work_dp}/task.pdb',
            0, virt_coords, self.ligand_info,
            self.config['input'].get('header_pdb', False),
        )

        # No RMSD-shape constraint or outlier filtering: the ligand's current pose
        # isn't trustworthy yet, so let it dock freely toward CURR under the
        # plane/in-plane-dev constraints only.
        dock_params = copyobj(self.config['parameters'])
        dock_params['rmsd_constraint'] = False
        dock_params['ipdc_d0'] = step * 0.5
        dock_params['ipdc_sd'] = step * 0.1
        # Same as _run_node: bounded ipdc only, no end padding.
        dock_params['ipdc_type']   = 'bounded'
        dock_params['end_padding'] = 0

        cst = ConstraintBuilder.create_constraints_coarse(
            self.ligand_info, dock_params, i_to_end=-1,
        )
        with open(f'{work_dp}/constraints.cst', 'w') as fh:
            fh.write(cst)

        xml = PDBHandler.make_xml_strings(
            self.config['rosetta_scripts']['protocol_zero_dock'], [self.leftmost],
        )
        with open(f'{work_dp}/task.xml', 'w') as fh:
            fh.write(xml)

        conformers_src = f"{self.config['input']['ligand_files']}/lig_conformers.pdb"
        if not os.path.isfile(conformers_src):
            raise FileNotFoundError(f"_run_dock_zero: conformers not found: {conformers_src}")
        copy(conformers_src, work_dp)

        rs_bin  = os.path.abspath(self.config['rosetta'])
        nstruct = self.config['rosetta_scripts']['nstruct_zero']
        prefix  = self._build_mpi_prefix(nstruct)
        run_rosetta(
            prefix + [rs_bin, '@options', '-nstruct', str(nstruct),
                      '-out:path:pdb', './results', '-parser:protocol', 'task.xml',
                      '-in:file:s', 'task.pdb'],
            cwd=work_dp,
            log_path=f'{work_dp}/task.log', err_path=f'{work_dp}/task.err',
        )

        params = self.config['parameters']
        best, _, _, _ = ScoreParser.report_final(
            work_dp, self.ligand_info,
            by=params['select_best_by'],
        )
        copy(best, f'{work_dp}/best.pdb')

        if params['memory_mode']:
            copy(f'{work_dp}/best.pdb', f'{work_dp}/final.pdb')
        else:
            PDBHandler.swap_lig_coords(
                f'{work_dp}/best.pdb', f'{work_dp}/task.pdb',
                f'{work_dp}/final.pdb', self.ligand_info,
            )

        pdb_files = glob.glob(os.path.join(work_dp, 'results', '*.pdb'))
        if pdb_files:
            run_tool(['gzip'] + pdb_files)
        rmtree(f'{work_dp}/grid')

        elapsed = time() - start_time
        if self.verbose:
            print("FINISHED: dock zero")
            print(f"in {timedelta(seconds=elapsed)}\n")

        return f'{work_dp}/final.pdb'

    def _run_node(self, node):
        '''
        Set up work directory, run Rosetta, pick best result.
        Returns the path to final.pdb.
        '''
        if self.verbose:
            print(f"STARTED: point {node.id}")
        start_time = time()

        work_dp = os.path.join(self.rootdir, str(node.id))
        os.mkdir(work_dp)
        os.mkdir(f'{work_dp}/results')
        os.mkdir(f'{work_dp}/grid')

        for src, dst_name in [
            (self.config['rosetta_scripts']['options'],              'options'),
            (f'{self.config["input"]["ligand_files"]}/lig.params',   None),
            (f'{self.rootdir}/lig.resfile',                          None),
            (f'{self.config["general_files"]}/VRT1.params',          None),
        ]:
            if not os.path.isfile(src):
                raise FileNotFoundError(f"_run_node: required file not found: {src}")
            copy(src, f'{work_dp}/{dst_name}' if dst_name else work_dp)

        for fname in self.config['extra_files']:
            if not os.path.isfile(fname):
                raise FileNotFoundError(f"_run_node: extra file not found: {fname}")
            copy(fname, work_dp)

        # Geometry for virtual coordinates and constraints
        new_start    = node.init_xyz
        displacement = new_start - node.Lm.xyz
        dL           = np.linalg.norm(displacement)
        normal       = node.Rm.xyz - node.Lm.xyz
        normal       /= np.linalg.norm(normal)
        tri1, tri2, tri3 = GeometryUtils.equilateral_triangle_vertices(
            new_start, normal,
            self.config['parameters']['plane_constraint_point_extent'],
        )

        has_second_left  = node.Lm.origin != 'pseudo'
        has_second_right = node.Rm.origin != 'pseudo'
        llm_xyz = node.Lm.Lm.xyz if has_second_left else node.Lm.xyz
        rrm_xyz = node.Rm.Rm.xyz if has_second_right else node.Rm.xyz

        V = CoarseVirt
        virt_coords = [None] * len(V)
        virt_coords[V.TRI1] = tri1
        virt_coords[V.TRI2] = tri2
        virt_coords[V.TRI3] = tri3
        virt_coords[V.LM]   = node.Lm.xyz
        virt_coords[V.RM]   = node.Rm.xyz
        virt_coords[V.CURR] = new_start
        virt_coords[V.LLM]  = llm_xyz
        virt_coords[V.RRM]  = rrm_xyz
        PDBHandler.prepare_start(
            node.init_pdb, f'{work_dp}/task.pdb',
            displacement, virt_coords, self.ligand_info,
            self.config['input'].get('header_pdb', False),
        )

        # Constraint file — scale rmsd_sd and ipdc by distance to handle variable spacing
        custom_params = copyobj(self.config['parameters'])
        step = self.config['parameters']['step']
        custom_params['rmsd_constraint_sd'] = (
            self.config['parameters']['rmsd_constraint_sd'] * np.sqrt(dL / step)
        )
        custom_params['ipdc_d0'] = dL * 0.5
        custom_params['ipdc_sd'] = dL * 0.1
        # CoarsePath always applies the in-plane-deviation constraint as 'bounded'
        # and never uses end padding (a trajectory-only feature). Force both here so
        # the config's ipdc_type / end_padding values cannot affect coarse behaviour.
        # See docs/unused_config_settings.md.
        custom_params['ipdc_type']   = 'bounded'
        custom_params['end_padding'] = 0

        cst = ConstraintBuilder.create_constraints_coarse(
            self.ligand_info, custom_params,
            has_second_left=has_second_left, has_second_right=has_second_right,
        )
        with open(f'{work_dp}/constraints.cst', 'w') as fh:
            fh.write(cst)

        # XML — scatter sample points around new_start as coordinate targets
        points = GeometryUtils.sample_points_in_disc(
            new_start, normal, custom_params['ipdc_d0'], 100
        )
        xml = PDBHandler.make_xml_strings(self.config['rosetta_scripts']['protocol'], points)
        with open(f'{work_dp}/task.xml', 'w') as fh:
            fh.write(xml)

        # Conformer subsampling
        if self.config['parameters']['conformer_subsampling']:
            conformers_src = (
                f'{os.path.abspath(self.config["input"]["ligand_files"])}/lig_conformers.pdb'
            )
            ConformerSampler.subsample_pdb(
                node.init_pdb, conformers_src,
                f'{work_dp}/lig_conformers.pdb',
                self.config['parameters']['conformer_rmsd_threshold'],
                self.ligand_info['chain'],
                self.ligand_info['nbr_atom_name'],
            )
        else:
            conformers_src = f"{self.config['input']['ligand_files']}/lig_conformers.pdb"
            if not os.path.isfile(conformers_src):
                raise FileNotFoundError(f"_run_node: conformers not found: {conformers_src}")
            copy(conformers_src, work_dp)

        # Rosetta run
        rs_bin  = os.path.abspath(self.config['rosetta'])
        nstruct = self.config['rosetta_scripts']['nstruct']
        prefix  = self._build_mpi_prefix(nstruct)
        run_rosetta(
            prefix + [rs_bin, '@options', '-nstruct', str(nstruct),
                      '-out:path:pdb', './results', '-parser:protocol', 'task.xml',
                      '-in:file:s', 'task.pdb'],
            cwd=work_dp,
            log_path=f'{work_dp}/task.log', err_path=f'{work_dp}/task.err',
        )

        # Pick best result
        params = self.config['parameters']
        best, _, _, _ = ScoreParser.report_final(
            work_dp, self.ligand_info,
            by=params['select_best_by'],
        )
        copy(best, f'{work_dp}/best.pdb')

        if params['memory_mode']:
            copy(f'{work_dp}/best.pdb', f'{work_dp}/final.pdb')
        else:
            PDBHandler.swap_lig_coords(
                f'{work_dp}/best.pdb', f'{work_dp}/task.pdb',
                f'{work_dp}/final.pdb', self.ligand_info,
            )

        pdb_files = glob.glob(os.path.join(work_dp, 'results', '*.pdb'))
        if pdb_files:
            run_tool(['gzip'] + pdb_files)
        rmtree(f'{work_dp}/grid')

        elapsed = time() - start_time
        if self.verbose:
            print(f"FINISHED: point {node.id}")
            print(f"in {timedelta(seconds=elapsed)}\n")

        return f'{work_dp}/final.pdb'

    def _process_node(self, node, max_id):
        '''
        Recursively run Rosetta on node and insert child nodes where gaps exceed step.
        Returns the highest node id assigned so far.
        '''
        if node.xyz is None:
            final_pdb_path = self._run_node(node)
            prev_coords, nbr_index, names, _ = PDBHandler.read_window(
                final_pdb_path,
                self.ligand_info['chain'],
                self.ligand_info['nbr_atom_name'],
            )
            node.xyz = prev_coords[nbr_index]
            self.ligand_info['atom_names'] = names

            step = self.config['parameters']['step']
            if 0.5 * (node.dleft() + node.dright()) > step:
                if node.dleft() > step:
                    new_crd = np.mean([node.Lm.xyz, node.xyz], axis=0)
                    max_id += 1
                    new_node = Node(
                        init_xyz=new_crd, xyz=None, id=max_id, origin='left',
                        Lm=node.Lm, Rm=node, init_pdb=node.init_pdb,
                    )
                    node.left       = new_node
                    new_node.parent = node
                    node.Lm.Rm      = new_node
                    node.Lm         = new_node
                    max_id = self._process_node(new_node, max_id)

                if node.dright() > step:
                    new_crd = np.mean([node.xyz, node.Rm.xyz], axis=0)
                    max_id += 1
                    new_node = Node(
                        init_xyz=new_crd, xyz=None, id=max_id, origin='right',
                        Lm=node, Rm=node.Rm, init_pdb=final_pdb_path,
                    )
                    node.right      = new_node
                    new_node.parent = node
                    node.Rm.Lm      = new_node
                    node.Rm         = new_node
                    max_id = self._process_node(new_node, max_id)

        return max_id

    def _save_output(self, node_sequence):
        '''Write coarse path coordinates and trajectory PDB files.'''
        point_sequence = (
            [self.leftmost]
            + [n.xyz for n in node_sequence]
            + [self.rightmost]
        )
        np.savetxt(f'{self.rootdir}/coarsepath.txt', point_sequence)
        PDBHandler.points_to_pdb(f'{self.rootdir}/trajectory.pdb', point_sequence)

        sources = (
            [self.start_pdb_path]
            + [os.path.join(self.rootdir, str(n.id), 'final.pdb') for n in node_sequence]
        )
        lig_chain = self.ligand_info['chain']

        with open(f'{self.rootdir}/lig_trajectory.pdb', 'w') as lig_traj, \
             open(f'{self.rootdir}/full_trajectory.pdb', 'w') as full_traj:
            for src in sources:
                with open(src, 'r') as fh:
                    for line in fh:
                        if line.startswith('ATOM') or line.startswith('HETATM'):
                            full_traj.write(line)
                            if line[21] == lig_chain:
                                lig_traj.write(line)
                lig_traj.write('ENDMDL\n')
                full_traj.write('ENDMDL\n')

        if self.config['output'].get('archive', False):
            for n in node_sequence:
                nid = str(n.id)
                run_tool(['zip', '-rm', f'{nid}.zip', nid], cwd=self.rootdir)

    # ------------------------------------------------------------------
    # Internal helpers (identical pattern to TrajectoryRunner)
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

    def _build_mpi_prefix(self, nstruct):
        '''Return the MPI launcher prefix as a list, or [] for serial execution.'''
        n_proc    = self.config['rosetta_scripts']['n_proc']
        effective = min(n_proc, nstruct)
        if effective > 1:
            launcher = self.config['rosetta_scripts']['mpi_launcher']
            return [launcher, '-np', str(effective)]
        return []

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
                f"rosetta_scripts.n_proc={n_proc} but the Rosetta binary is not an MPI build "
                f"(no '.mpi.' in filename): {rs_bin}\nUse the MPI binary or set n_proc=1."
            )
        if n_proc > self.config['rosetta_scripts']['nstruct']:
            raise ValueError(
                f"rosetta_scripts.n_proc ({n_proc}) exceeds nstruct "
                f"({self.config['rosetta_scripts']['nstruct']}); extra ranks would be idle."
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

        for key in ('options', 'protocol'):
            path = self.config['rosetta_scripts'][key]
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"Config rosetta_scripts['{key}'] not found: {path}"
                )

        vrt_params = f'{self.config["general_files"]}/VRT1.params'
        if not os.path.isfile(vrt_params):
            raise FileNotFoundError(f"VRT1.params not found: {vrt_params}")

        check_protocol_chain(
            [self.config['rosetta_scripts']['protocol'],
             self.config['rosetta_scripts'].get('protocol_zero_dock')],
            self.config['input']['ligand_chain'],
        )

        dead_ipdc_keys = [k for k in ('ipdc_d0', 'ipdc_sd') if k in self.config['parameters']]
        if dead_ipdc_keys:
            print(
                f"WARNING: parameters {dead_ipdc_keys} are set in the config but have no "
                f"effect — CoarsePathRunner always recomputes ipdc_d0/ipdc_sd per node from "
                f"dL (the node's distance to its target), overwriting any config value.\n"
            )

        ipdc_type = self.config['parameters'].get('ipdc_type')
        if ipdc_type is not None and str(ipdc_type).lower() != 'bounded':
            print(
                f"WARNING: parameters['ipdc_type'] = {ipdc_type!r} is ignored by "
                f"CoarsePathRunner — the in-plane-deviation constraint is always applied as "
                f"'bounded'. 'bounded' will be used. Set ipdc_type='bounded' to silence this.\n"
            )

        if 'end_padding' in self.config['parameters']:
            print(
                "WARNING: parameters['end_padding'] is set but has no effect in "
                "CoarsePathRunner — end padding is a trajectory-only feature and is always "
                "treated as 0 here.\n"
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='LigPF — generate a coarse ligand path via bisection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Config overrides (--set) use dot-notation matching the JSON structure:\n'
            '  --set parameters.step=0.5\n'
            '  --set rosetta_scripts.nstruct=20\n'
            '  --set output.path=/new/output/dir\n'
        ),
    )
    parser.add_argument('config', help='JSON run configuration file')
    parser.add_argument(
        '--set', metavar='KEY=VALUE', action='append', dest='overrides', default=[],
        help='Override any config value; dot-notation, e.g. --set parameters.step=0.5',
    )
    args = parser.parse_args()
    runner = CoarsePathRunner(args.config, overrides=args.overrides)
    runner.run()
