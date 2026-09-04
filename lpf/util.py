# LIGAND PATH FINDER
# Utility classes

import numpy as np
from scipy import interpolate
from enum import IntEnum
import json
import os
from copy import copy
from scipy.spatial import transform
from subprocess import run as sp_run


class TrajVirt(IntEnum):
    TRI1      = 0   # equilateral-triangle vertex (plane constraint)
    TRI2      = 1
    TRI3      = 2
    PREV_PREV = 3   # second-previous NBR position
    PREV      = 4   # previous NBR position
    CURR      = 5   # current guide point (in-plane dev target)
    NEXT      = 6   # next guide point (forward angle)


class CoarseVirt(IntEnum):
    TRI1  = 0   # equilateral-triangle vertex (plane constraint)
    TRI2  = 1
    TRI3  = 2
    LM    = 3   # left-neighbour node position
    RM    = 4   # right-neighbour node position
    CURR  = 5   # new_start (in-plane dev target)
    LLM   = 6   # second-order left (or LM fallback when neighbour is pseudo)
    RRM   = 7   # second-order right (or RM fallback when neighbour is pseudo)


def read_guide_path(filename):
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Guide path file not found: {filename}")
    if filename.lower().endswith('.pdb'):
        coords = []
        with open(filename, 'r') as fh:
            for line in fh:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        if not coords:
            raise ValueError(f"read_guide_path: no ATOM/HETATM records found in {filename}")
        return np.array(coords)
    return np.genfromtxt(filename)


def read_config(filename):
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Config file not found: {filename}")
    with open(filename, 'r') as fh:
        try:
            config = json.load(fh)
        except json.JSONDecodeError as e:
            raise ValueError(f"Config file is not valid JSON ({filename}): {e}")
    return config


# ----------------------------------------------------------------
# Subprocess helpers — never let a failed external tool pass silently
# ----------------------------------------------------------------

class RosettaError(Exception):
    """Raised when the rosetta_scripts subprocess exits with a non-zero status.

    Deliberately not a RuntimeError: TrajectoryRunner retries RuntimeError from
    result parsing with a tighter RMSD constraint, which would be pointless (and
    would mask the real error) for a Rosetta binary or protocol failure.
    """


def _tail(path, n_lines=20):
    """Last n_lines of a text file, for error messages. Never raises."""
    try:
        with open(path, 'r', errors='replace') as fh:
            lines = fh.readlines()
    except OSError:
        return ''
    return ''.join(lines[-n_lines:]).rstrip()


def run_rosetta(argv, cwd, log_path, err_path):
    """Run rosetta_scripts, raising RosettaError with context if it fails.

    stdout/stderr go to log_path/err_path; on failure the tail of both is
    included in the exception so the cause is visible without digging.
    """
    with open(log_path, 'w') as log_fh, open(err_path, 'w') as err_fh:
        proc = sp_run(argv, cwd=cwd, stdout=log_fh, stderr=err_fh)
    if proc.returncode != 0:
        msg = [
            f"Rosetta exited with status {proc.returncode}.",
            f"  command: {' '.join(argv)}",
            f"  workdir: {cwd}",
        ]
        err_tail = _tail(err_path)
        log_tail = _tail(log_path)
        if err_tail:
            msg.append(f"  last lines of {os.path.basename(err_path)}:\n{err_tail}")
        if log_tail:
            msg.append(f"  last lines of {os.path.basename(log_path)}:\n{log_tail}")
        if not err_tail and not log_tail:
            msg.append("  (both log files are empty — check that the binary is executable "
                       "and, for MPI builds, that the launcher is on PATH)")
        raise RosettaError('\n'.join(msg))
    return proc


def run_tool(argv, cwd=None):
    """Run a small helper tool (gzip/zip), raising RuntimeError on failure.

    `zip -rm` deletes its inputs, so a silently failed archive step could lose
    results; gzip failures would leave the output directory in a mixed state.
    """
    proc = sp_run(argv, cwd=cwd)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed with status {proc.returncode}: {' '.join(argv[:3])}"
            f"{' ...' if len(argv) > 3 else ''}"
            + (f" (in {cwd})" if cwd else "")
            + f"\nIs '{argv[0]}' installed and on PATH?"
        )
    return proc


def check_external_tools(archive_enabled):
    """Fail fast if a required command-line tool is missing.

    Both are shelled out to mid-run: gzip after every Rosetta call, zip only when
    output.archive is on. Checking here turns a failure 35 minutes into a run
    (with `zip -rm` potentially having already deleted its inputs) into an
    immediate, obvious error.
    """
    from shutil import which
    required = [('gzip', 'compressing decoy PDBs after every Rosetta call')]
    if archive_enabled:
        required.append(('zip', 'output.archive is enabled'))
    missing = [(tool, why) for tool, why in required if which(tool) is None]
    if missing:
        raise FileNotFoundError(
            "Required command-line tool(s) not found on PATH:\n"
            + '\n'.join(f"  {tool}  — needed for: {why}" for tool, why in missing)
            + "\nInstall them (they are in environment.yml) or disable output.archive."
        )


def check_protocol_chain(protocol_paths, ligand_chain):
    """Warn if a RosettaScripts protocol never references `ligand_chain`.

    The shipped XMLs hardcode chain="X" in LigandArea / SCORINGGRIDS / Transform /
    StartFrom / InterfaceScoreCalculator. Rosetta does not read input.ligand_chain,
    so a config that sets a different chain would silently produce nonsense rather
    than fail. This catches the mismatch at start-up.
    """
    import re
    pattern = re.compile(
        r'(?:ligand_)?chains?\s*=\s*"([^"]*)"'
    )
    offenders = []
    for path in protocol_paths:
        if not path or not os.path.isfile(path):
            continue
        try:
            with open(path, 'r', errors='replace') as fh:
                text = fh.read()
        except OSError:
            continue
        declared = {c for m in pattern.findall(text) for c in m}
        if declared and ligand_chain not in declared:
            offenders.append((path, ''.join(sorted(declared))))
    if offenders:
        print(
            f"WARNING: input.ligand_chain is '{ligand_chain}', but the following "
            f"RosettaScripts protocols never reference that chain. Rosetta does not "
            f"read ligand_chain from the config — edit the chain attributes in these "
            f"files to match, or the run will score the wrong chain:"
        )
        for path, declared in offenders:
            print(f"    {path}  (declares chain(s): {declared})")
        print()


# ----------------------------------------------------------------
# GeometryUtils — pure 3D math helpers
# ----------------------------------------------------------------

class GeometryUtils:

    @staticmethod
    def get_angle(p1, p2, p3):
        '''Angle in degrees defined by three points p1-p2-p3.'''
        p1 = np.array(p1, dtype=float)
        p2 = np.array(p2, dtype=float)
        p3 = np.array(p3, dtype=float)
        if p1.shape != (3,) or p2.shape != (3,) or p3.shape != (3,):
            raise ValueError("get_angle: all three points must be 3D vectors")
        ba = p1 - p2
        bc = p3 - p2
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        if norm_ba == 0 or norm_bc == 0:
            raise ValueError("get_angle: degenerate angle — two of the three points are identical")
        cosine_angle = np.clip(np.dot(ba, bc) / (norm_ba * norm_bc), -1.0, 1.0)
        return np.rad2deg(np.arccos(cosine_angle))

    @staticmethod
    def get_uv(n):
        '''Two orthonormal vectors both perpendicular to n (spanning the plane normal to n).'''
        n = np.array(n, dtype=float)
        if np.linalg.norm(n) == 0:
            raise ValueError("get_uv: normal vector must be non-zero")
        u = np.random.randn(3)
        u -= u.dot(n) * n
        if np.linalg.norm(u) < 1e-10:
            # Unlucky random draw — try again deterministically
            u = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            u -= u.dot(n) * n
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        return u / np.linalg.norm(u), v / np.linalg.norm(v)

    @staticmethod
    def sample_points_on_disc(center, normal, radius, num_points):
        '''Points uniformly sampled on the rim of a disc.'''
        if radius <= 0:
            raise ValueError(f"sample_points_on_disc: radius must be positive, got {radius}")
        if num_points < 1:
            raise ValueError(f"sample_points_on_disc: num_points must be >= 1, got {num_points}")
        points = []
        while len(points) < num_points:
            theta = np.random.rand() * 2 * np.pi
            u, v = GeometryUtils.get_uv(normal)
            p = np.cos(theta) * u + np.sin(theta) * v
            points.append(np.array(center) + p * radius)
        return points

    @staticmethod
    def sample_points_in_disc(center, normal, max_radius, num_points):
        '''Points uniformly sampled inside a disc.'''
        if max_radius <= 0:
            raise ValueError(f"sample_points_in_disc: max_radius must be positive, got {max_radius}")
        if num_points < 1:
            raise ValueError(f"sample_points_in_disc: num_points must be >= 1, got {num_points}")
        points = []
        while len(points) < num_points:
            theta = np.random.rand() * 2 * np.pi
            scale = np.random.rand() * max_radius
            u, v = GeometryUtils.get_uv(normal)
            p = np.cos(theta) * u + np.sin(theta) * v
            points.append(np.array(center) + p * scale)
        return points

    @staticmethod
    def two_points_in_plane(center, normal, radius):
        '''Two antipodal points lying in the plane defined by center and normal.'''
        if radius <= 0:
            raise ValueError(f"two_points_in_plane: radius must be positive, got {radius}")
        points = []
        u, v = GeometryUtils.get_uv(normal)
        for theta in [0, np.pi]:
            p = np.cos(theta) * u + np.sin(theta) * v
            points.append(np.array(center) + p * radius)
        return points

    @staticmethod
    def equilateral_triangle_vertices(O, normal, R):
        '''Vertices of an equilateral triangle of circumradius R centred at O in the plane normal to `normal`.'''
        O = np.asarray(O, dtype=float)
        n = np.asarray(normal, dtype=float)
        if O.shape != (3,):
            raise ValueError("equilateral_triangle_vertices: O must be a 3D point")
        if np.linalg.norm(n) == 0:
            raise ValueError("equilateral_triangle_vertices: normal vector must be non-zero")
        if R <= 0:
            raise ValueError(f"equilateral_triangle_vertices: R must be positive, got {R}")
        n = n / np.linalg.norm(n)

        tmp = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = np.cross(n, tmp)
        u /= np.linalg.norm(u)
        v = np.cross(n, u)

        angles = [0, 2 * np.pi / 3, 4 * np.pi / 3]
        vertices = [O + R * (np.cos(a) * u + np.sin(a) * v) for a in angles]
        return np.array(vertices)


# ----------------------------------------------------------------
# PathBuilder — spline fitting and equidistant path sampling
# ----------------------------------------------------------------

class PathBuilder:

    @staticmethod
    def spline(points, smoothness, umax_factor=1.0):
        '''Fit a smoothing spline through `points` and return 10 000 densely sampled positions.'''
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"spline: points must be an (N, 3) array, got shape {points.shape}")
        if len(points) < 4:
            raise ValueError(f"spline: need at least 4 points for cubic spline fitting, got {len(points)}")
        if smoothness < 0:
            raise ValueError(f"spline: smoothness must be >= 0, got {smoothness}")
        tck, u = interpolate.splprep([points[:, 0], points[:, 1], points[:, 2]], s=smoothness)
        umax = np.max(u)
        unew = np.linspace(0, umax * umax_factor, 10000)
        out = interpolate.splev(unew, tck)
        return np.column_stack(out)

    @staticmethod
    def equalize(curve, target_distance, target_length=-1):
        '''
        Sample equidistant points along `curve`.
        Returns (path, normals); target_length=-1 means no length cap.
        '''
        curve = np.asarray(curve, dtype=float)
        if len(curve) < 2:
            raise ValueError(f"equalize: curve must have at least 2 points, got {len(curve)}")
        if target_distance <= 0:
            raise ValueError(f"equalize: target_distance must be positive, got {target_distance}")
        if target_length == 0:
            raise ValueError("equalize: target_length must not be 0 (use -1 for no cap)")

        output = [curve[0]]
        normals = [curve[1] - curve[0]]
        prev_point = curve[0]
        cumulative_delta = 0
        pool = []
        tol = target_distance * 0.5
        n = 1

        for c in range(1, len(curve)):
            if (target_length > 0) and (n == target_length):
                break

            d = np.linalg.norm(curve[c] - prev_point)
            delta = np.abs(d - target_distance + cumulative_delta)

            if np.abs(d - target_distance) < tol:
                pool.append((c, d, delta))
            elif (d > target_distance + tol) or (c == len(curve) - 1):
                if pool:
                    best_point = sorted(pool, key=lambda x: x[2])[0]
                    bp_c = best_point[0]
                    output.append(curve[bp_c])
                    if c != len(curve) - 1:
                        normals.append(curve[bp_c + 1] - curve[bp_c - 1])
                    else:
                        normals.append(curve[bp_c] - curve[bp_c - 1])
                    n += 1
                    prev_point = curve[bp_c]
                    cumulative_delta += best_point[1] - target_distance
                    pool = []

        output = np.array(output)
        normals = np.array([nrm / np.linalg.norm(nrm) for nrm in normals])
        return output, normals

    @staticmethod
    def adjust_path(curve, initial_step, target_length, end, max_iter=60, tol=1e-9):
        '''
        Find a step size so equalize() returns exactly `target_length` points.

        Uses bisection on the monotonic relationship: smaller step → more points.
        Converges in at most max_iter iterations (typically ~20).
        '''
        if initial_step <= 0:
            raise ValueError(f"adjust_path: initial_step must be positive, got {initial_step}")
        if target_length < 2:
            raise ValueError(f"adjust_path: target_length must be >= 2, got {target_length}")

        # Find bracket: lo yields >= target_length points, hi yields <= target_length
        lo, hi = initial_step * 0.5, initial_step * 2.0
        for _ in range(40):
            if len(PathBuilder.equalize(curve, lo, target_length)[0]) >= target_length:
                break
            lo *= 0.5
        for _ in range(40):
            if len(PathBuilder.equalize(curve, hi, target_length)[0]) <= target_length:
                break
            hi *= 2.0

        for _ in range(max_iter):
            if hi - lo < tol:
                break
            mid = (lo + hi) / 2.0
            if len(PathBuilder.equalize(curve, mid, target_length)[0]) >= target_length:
                lo = mid
            else:
                hi = mid

        new_path, normals = PathBuilder.equalize(curve, lo, target_length)
        if len(new_path) != target_length:
            raise RuntimeError(
                f"adjust_path: could not converge to {target_length} points "
                f"(got {len(new_path)} at step={lo:.6f})"
            )
        return new_path, normals


# ----------------------------------------------------------------
# PDBHandler — reading/writing PDB files and Rosetta input prep
# ----------------------------------------------------------------

'''
ligand_info dict schema:
{
  "chain": "X",
  "resnum": 123,
  "nbr_atom_name": "C0",
  "atom_names": [...],
}
'''

class PDBHandler:

    @staticmethod
    def read_window(filename, ligand_chain, ligand_nbr_name):
        '''
        Extract ligand atom coordinates from a PDB file.
        Returns (coords_array, nbr_atom_index, atom_names, resnum).
        '''
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"read_window: PDB file not found: {filename}")
        if not ligand_chain:
            raise ValueError("read_window: ligand_chain must be a non-empty string")
        if not ligand_nbr_name:
            raise ValueError("read_window: ligand_nbr_name must be a non-empty string")

        coords = []
        names = []
        nbr_atom = None
        lig_resnum = None
        i = -1

        with open(filename, 'r') as fh:
            for line in fh:
                if line.startswith('HETATM') and line[21] == ligand_chain:
                    if lig_resnum is None:
                        lig_resnum = int(line[22:26])
                    i += 1
                    name = line[12:16].strip()
                    if name == ligand_nbr_name:
                        nbr_atom = i
                    coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                    names.append(name)

        if not coords:
            raise ValueError(
                f"read_window: no HETATM records found for chain '{ligand_chain}' in {filename}"
            )
        if nbr_atom is None:
            raise ValueError(
                f"read_window: NBR atom '{ligand_nbr_name}' not found in chain '{ligand_chain}' of {filename}"
            )

        return np.array(coords), nbr_atom, names, lig_resnum

    @staticmethod
    def read_names(paramfile):
        if not os.path.isfile(paramfile):
            raise FileNotFoundError(f"read_names: params file not found: {paramfile}")
        names = []
        with open(paramfile, 'r') as fh:
            for line in fh:
                if line.startswith('ATOM'):
                    names.append(line.strip().split()[1])
        if not names:
            raise ValueError(f"read_names: no ATOM records found in {paramfile}")
        return names

    @staticmethod
    def merge_start(protein_pdb, ligand_pdb, output_path):
        '''Write a merged PDB: protein ATOM records first, then ligand HETATM records.'''
        if not os.path.isfile(protein_pdb):
            raise FileNotFoundError(f"merge_start: protein PDB not found: {protein_pdb}")
        if not os.path.isfile(ligand_pdb):
            raise FileNotFoundError(f"merge_start: ligand PDB not found: {ligand_pdb}")
        outstr = ''
        with open(protein_pdb, 'r') as fh:
            for line in fh:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    outstr += line
        outstr += 'TER\n'
        with open(ligand_pdb, 'r') as fh:
            for line in fh:
                if line.startswith('HETATM'):
                    outstr += line
        outstr += 'TER\n'
        with open(output_path, 'w') as fh:
            fh.write(outstr)

    @staticmethod
    def swap_lig_coords(pdb_donor, pdb_acceptor, new_pdb, ligand_info):
        '''Replace ligand coordinates in pdb_acceptor with those from pdb_donor.'''
        if not os.path.isfile(pdb_donor):
            raise FileNotFoundError(f"swap_lig_coords: donor PDB not found: {pdb_donor}")
        if not os.path.isfile(pdb_acceptor):
            raise FileNotFoundError(f"swap_lig_coords: acceptor PDB not found: {pdb_acceptor}")
        if not ligand_info.get('atom_names'):
            raise ValueError("swap_lig_coords: ligand_info['atom_names'] is empty")

        lig_chain = ligand_info['chain']

        donor_xyz = {}
        with open(pdb_donor, 'r') as fh:
            for line in fh:
                if line[:6] in ('HETATM', 'ATOM  ') and len(line) > 54 and line[21] == lig_chain:
                    donor_xyz[line[12:16].strip()] = line[30:54]
        if not donor_xyz:
            raise ValueError(
                f"swap_lig_coords: no ligand atoms found in donor {pdb_donor} for chain {lig_chain}"
            )

        with open(pdb_acceptor, 'r') as fh_in, open(new_pdb, 'w') as fh_out:
            for line in fh_in:
                if line[:6] in ('HETATM', 'ATOM  ') and len(line) > 54 and line[21] == lig_chain:
                    name = line[12:16].strip()
                    if name not in donor_xyz:
                        raise ValueError(
                            f"swap_lig_coords: atom '{name}' in acceptor not found in donor {pdb_donor}"
                        )
                    line = line[:30] + donor_xyz[name] + line[54:]
                fh_out.write(line)

    @staticmethod
    def prepare_start(prev_final_path, output_path, displacement, virt_coordinates,
                      ligand_info, header=False, lig_coords_path=False):
        if not os.path.isfile(prev_final_path):
            raise FileNotFoundError(f"prepare_start: input PDB not found: {prev_final_path}")
        if header and not os.path.isfile(header):
            raise FileNotFoundError(f"prepare_start: header PDB not found: {header}")
        if lig_coords_path and not os.path.isfile(lig_coords_path):
            raise FileNotFoundError(f"prepare_start: lig_coords PDB not found: {lig_coords_path}")

        nbr_atom_name = ligand_info['nbr_atom_name']
        lig_resnum = ligand_info['resnum']
        chain = ligand_info['chain']

        lig_visited = False
        first_ter = False

        prev = PDBHandler.read_window(prev_final_path, chain, nbr_atom_name)
        ghost_coordinates = prev[0] + displacement

        outstr = ''
        lig_coords_file = lig_coords_path if lig_coords_path else prev_final_path

        ligstr = ''
        with open(lig_coords_file, 'r') as fh:
            for line in fh:
                if line.startswith('ATOM'):
                    outstr += line
                elif line.startswith('HETATM'):
                    if line[21] == chain:
                        if not lig_visited:
                            lig_visited = True
                        ligstr += line
                    elif 'VRT' not in line:
                        if not lig_visited:
                            outstr += line
                elif line.startswith('TER'):
                    if not first_ter:
                        outstr += line
                        first_ter = True
                    elif lig_visited:
                        ligstr += line

        if not lig_visited:
            raise ValueError(
                f"prepare_start: ligand chain '{chain}' not found in {lig_coords_file}"
            )

        for i, virt_coords in enumerate(virt_coordinates):
            virt_resnum = lig_resnum + 1 + i
            outstr += (f'HETATM    0 ORIG VRT Y {str(virt_resnum).rjust(3)}'
                       f'    {virt_coords[0]:8.3f}{virt_coords[1]:8.3f}{virt_coords[2]:8.3f}  1.00  0.00\n')
        outstr += 'TER\n'
        for i, c in enumerate(ghost_coordinates):
            virt_resnum = lig_resnum + len(virt_coordinates) + 1 + i
            outstr += (f'HETATM    0 ORIG VRT Z {str(virt_resnum).rjust(3)}'
                       f'    {c[0]:8.3f}{c[1]:8.3f}{c[2]:8.3f}  1.00  0.00\n')
        outstr += 'TER\n'

        with open(output_path, 'w') as fh:
            if header:
                with open(header, 'r') as hf:
                    fh.write(hf.read())
            fh.write(outstr)
            fh.write(ligstr)

    @staticmethod
    def prepare_stats(task_pdb, lig_coords_pdb, output_path, ligand_info):
        if not os.path.isfile(task_pdb):
            raise FileNotFoundError(f"prepare_stats: task PDB not found: {task_pdb}")
        if not os.path.isfile(lig_coords_pdb):
            raise FileNotFoundError(f"prepare_stats: lig_coords PDB not found: {lig_coords_pdb}")

        chain = ligand_info['chain']
        ligstring = ''
        with open(lig_coords_pdb, 'r') as fh:
            for line in fh:
                if line.startswith('HETATM') and line[21] == chain:
                    ligstring += line

        if not ligstring:
            raise ValueError(
                f"prepare_stats: no HETATM records for chain '{chain}' in {lig_coords_pdb}"
            )

        outstr = ''
        lig_filled = False
        with open(task_pdb, 'r') as fh:
            for line in fh:
                if line.startswith('HETATM') and line[21] == chain:
                    if not lig_filled:
                        outstr += ligstring
                        lig_filled = True
                else:
                    outstr += line

        with open(output_path, 'w') as fh:
            fh.write(outstr)

    @staticmethod
    def make_xml_strings(basepath, coords):
        '''Insert coordinate tags into a boilerplate RosettaScripts XML.'''
        if not os.path.isfile(basepath):
            raise FileNotFoundError(f"make_xml_strings: XML template not found: {basepath}")
        if not coords:
            raise ValueError("make_xml_strings: coords list is empty")

        with open(basepath, 'r') as fh:
            template = fh.read()
        if '$replace$' not in template:
            raise ValueError(
                f"make_xml_strings: placeholder '$replace$' not found in template {basepath}"
            )

        txt = ''.join(
            '\t' * 6 + f'<Coordinates x="{cc[0]:.3f}" y="{cc[1]:.3f}" z="{cc[2]:.3f}"/>\n'
            for cc in coords
        )
        return template.replace('$replace$', txt)

    @staticmethod
    def points_to_pdb(outfn, coords):
        if not coords:
            raise ValueError("points_to_pdb: coords list is empty")
        with open(outfn, 'w') as fh:
            for c_i, cc in enumerate(coords):
                fh.write(f'HETATM {str(c_i).rjust(4)} ORIG VRT A   1'
                         f'    {cc[0]:8.3f}{cc[1]:8.3f}{cc[2]:8.3f}  1.00  0.00\n')

    @staticmethod
    def path_to_pdb(outfn, points):
        '''Write ordered 3D points as a connected HETATM chain (PyMOL: show sticks, resn GP).'''
        with open(outfn, 'w') as fh:
            for i, (x, y, z) in enumerate(points):
                serial = i + 1
                fh.write(
                    f'HETATM{serial:5d}  C   GP  A{serial:4d}    '
                    f'{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n'
                )
            for i in range(len(points) - 1):
                fh.write(f'CONECT{i+1:5d}{i+2:5d}\n')


# ----------------------------------------------------------------
# ScoreParser — reading Rosetta score files
# ----------------------------------------------------------------

_CONSTRAINT_COLS    = ('angle_constraint', 'atom_pair_constraint', 'dihedral_constraint')
_IF_CONSTRAINT_COLS = tuple(f'if_X_{c}' for c in _CONSTRAINT_COLS)


class ScoreParser:

    @staticmethod
    def report_final(folder, ligand_info, by='total_score', outlier_rmsd=100):
        '''Return (best_pdb_path, interface_score, total_score, cst_dict) for the best result.

        interface_score and total_score are constraint-free values (trajectory constraint
        contributions subtracted).  cst_dict maps each constraint type to its raw energy
        for the selected decoy.
        '''
        score_path = f'{folder}/score.sc'
        if not os.path.isfile(score_path):
            raise FileNotFoundError(f"report_final: score file not found: {score_path}")
        if by not in ('total_score', 'interface_delta_X'):
            raise ValueError(f"report_final: 'by' must be 'total_score' or 'interface_delta_X', got '{by}'")

        with open(score_path, 'r') as fh:
            scorefl = fh.readlines()
        if len(scorefl) < 3:
            raise ValueError(f"report_final: score file has fewer than 2 data lines: {score_path}")

        tmp = np.array([xi.strip().split() for xi in scorefl[1:]])
        headers = tmp[0]
        data_rows = tmp[1:]
        n = len(data_rows)

        for col in ('interface_delta_X', 'total_score', 'description'):
            if col not in headers:
                raise ValueError(f"report_final: required column '{col}' not found in {score_path}")

        i_delta = int(np.argwhere(headers == 'interface_delta_X')[0][0])
        i_total = int(np.argwhere(headers == 'total_score')[0][0])
        i_descr = int(np.argwhere(headers == 'description')[0][0])

        cst_col_idx    = {c: int(np.argwhere(headers == c)[0][0])
                          for c in _CONSTRAINT_COLS    if c in headers}
        if_cst_col_idx = {c: int(np.argwhere(headers == c)[0][0])
                          for c in _IF_CONSTRAINT_COLS if c in headers}

        idelta_raw = data_rows[:, i_delta].astype(float)
        total_raw  = data_rows[:, i_total].astype(float)

        total_cst_arr = np.zeros(n)
        for j in cst_col_idx.values():
            total_cst_arr += data_rows[:, j].astype(float)

        if_cst_arr = np.zeros(n)
        for j in if_cst_col_idx.values():
            if_cst_arr += data_rows[:, j].astype(float)

        idelta_clean = idelta_raw - if_cst_arr
        total_clean  = total_raw  - total_cst_arr

        sort_key = idelta_clean if by == 'interface_delta_X' else total_clean
        sorted_idx = np.argsort(sort_key)

        for k in sorted_idx:
            fname = str(data_rows[k, i_descr])
            result_pdb = f'{folder}/results/{fname}.pdb'
            if not os.path.isfile(result_pdb):
                continue
            r = ConformerSampler.check_rmsd(
                result_pdb,
                f'{folder}/task.pdb',
                ligand_info['chain'],
                ligand_info['nbr_atom_name'],
            )
            if r < outlier_rmsd:
                cst_dict = {c: float(data_rows[k, j]) for c, j in cst_col_idx.items()}
                for c in _CONSTRAINT_COLS:
                    cst_dict.setdefault(c, 0.0)
                return result_pdb, float(idelta_clean[k]), float(total_clean[k]), cst_dict

        raise RuntimeError(
            f"report_final: no result in {folder} passed the RMSD filter (outlier_rmsd={outlier_rmsd})"
        )

    @staticmethod
    def report_stats(folder):
        '''Return (interface_scores, total_scores) arrays — constraint-free — for all stats.sc rows.'''
        score_path = f'{folder}/stats.sc'
        if not os.path.isfile(score_path):
            raise FileNotFoundError(f"report_stats: stats score file not found: {score_path}")

        with open(score_path, 'r') as fh:
            scorefl = fh.readlines()
        if len(scorefl) < 3:
            raise ValueError(f"report_stats: stats score file has fewer than 2 data lines: {score_path}")

        tmp = np.array([xi.strip().split() for xi in scorefl[1:]])
        headers = tmp[0]
        data_rows = tmp[1:]
        n = len(data_rows)

        for col in ('interface_delta_X', 'total_score'):
            if col not in headers:
                raise ValueError(f"report_stats: required column '{col}' not found in {score_path}")

        i_delta = int(np.argwhere(headers == 'interface_delta_X')[0][0])
        i_total = int(np.argwhere(headers == 'total_score')[0][0])

        cst_col_idx    = {c: int(np.argwhere(headers == c)[0][0])
                          for c in _CONSTRAINT_COLS    if c in headers}
        if_cst_col_idx = {c: int(np.argwhere(headers == c)[0][0])
                          for c in _IF_CONSTRAINT_COLS if c in headers}

        total_cst = np.zeros(n)
        for j in cst_col_idx.values():
            total_cst += data_rows[:, j].astype(float)

        if_cst = np.zeros(n)
        for j in if_cst_col_idx.values():
            if_cst += data_rows[:, j].astype(float)

        idelta_clean = data_rows[:, i_delta].astype(float) - if_cst
        total_clean  = data_rows[:, i_total].astype(float) - total_cst
        return idelta_clean, total_clean


# ----------------------------------------------------------------
# ConstraintBuilder — Rosetta constraint text generation
# ----------------------------------------------------------------

_REQUIRED_CONSTRAINT_PARAMS = (
    'plane_constraint_sd',
    'next_angle_constraint',
    'angle_constraint',
    'rmsd_constraint',
    'in_plane_dev_constraint',
    'ipdc_sd',
    'step',
)

class ConstraintBuilder:

    @staticmethod
    def create_RMSD_constraints(lig_info, sd, heavy_only=True, cst_type='harmonic', bound=0.0, n_virt=9):
        if sd <= 0:
            raise ValueError(f"create_RMSD_constraints: sd must be positive, got {sd}")
        names = lig_info.get('atom_names')
        if not names:
            raise ValueError("create_RMSD_constraints: ligand_info['atom_names'] is empty")

        lig_resnum = lig_info['resnum']
        lig_chain = lig_info['chain']

        filtered = [(c, name) for c, name in enumerate(names)
                    if not (heavy_only and name[0] == 'H')]
        if not filtered:
            raise ValueError("create_RMSD_constraints: no atoms remain after filtering hydrogens")

        N = len(filtered)
        weight = sd * np.sqrt(N)
        ghost_start = lig_resnum + n_virt + 1

        cst_type = cst_type.lower()
        s = ''
        for c, name in filtered:
            if cst_type == 'bounded':
                s += (f'AtomPair {name} {lig_resnum}{lig_chain} ORIG {ghost_start + c}Z'
                      f' BOUNDED 0 {bound:.3f} {weight:.5f} 0.5 tag\n')
            else:
                s += (f'AtomPair {name} {lig_resnum}{lig_chain} ORIG {ghost_start + c}Z'
                      f' HARMONIC 0 {weight:.5f}\n')
        return s

    @staticmethod
    def _constraint_preamble(lig_info, parameters, n_virt):
        '''Shared setup for both constraint functions. Returns (cst, nbr_s, virt_nums).'''
        missing = [k for k in _REQUIRED_CONSTRAINT_PARAMS if k not in parameters]
        if missing:
            raise KeyError(f"create_constraints: missing parameter keys: {missing}")
        lig_resnum = lig_info['resnum']
        lig_chain = lig_info['chain']
        nbr_s = f'{lig_info["nbr_atom_name"]} {lig_resnum}{lig_chain}'
        virt_nums = [lig_resnum + 1 + s for s in range(n_virt)]
        plane_sd = parameters['plane_constraint_sd']
        V = TrajVirt  # indices 0-2 are the same in both enums
        cst = 'MultiConstraint\n'
        cst += f'Dihedral ORIG {virt_nums[V.TRI3]}Y ORIG {virt_nums[V.TRI1]}Y ORIG {virt_nums[V.TRI2]}Y {nbr_s} HARMONIC 0 {plane_sd:.6f}\n'
        cst += f'Dihedral ORIG {virt_nums[V.TRI1]}Y ORIG {virt_nums[V.TRI2]}Y ORIG {virt_nums[V.TRI3]}Y {nbr_s} HARMONIC 0 {plane_sd:.6f}\n'
        cst += f'Dihedral ORIG {virt_nums[V.TRI2]}Y ORIG {virt_nums[V.TRI3]}Y ORIG {virt_nums[V.TRI1]}Y {nbr_s} HARMONIC 0 {plane_sd:.6f}\n'
        return cst, nbr_s, virt_nums

    @staticmethod
    def _append_rmsd_and_ipdc(cst, lig_info, parameters, i_to_end, end_mode, virt_curr, n_virt):
        '''Append RMSD and in-plane-dev constraints (shared logic).'''
        nbr_s = f'{lig_info["nbr_atom_name"]} {lig_info["resnum"]}{lig_info["chain"]}'
        if parameters['rmsd_constraint']:
            heavy_only = not parameters['rmsd_constraint_hydrogens']
            cst_type = parameters['rmsd_constraint_type']
            bound = parameters['rmsd_constraint_bound'] if cst_type.lower() == 'bounded' else 0.0
            cst += ConstraintBuilder.create_RMSD_constraints(
                lig_info, parameters['rmsd_constraint_sd'],
                heavy_only, cst_type, bound,
                n_virt=n_virt,
            )
        if parameters['in_plane_dev_constraint']:
            ipdc_type = parameters['ipdc_type'].lower()
            ipdc_d0 = parameters['ipdc_d0']
            ipdc_sd = parameters['ipdc_sd']
            end_padding = int(parameters['end_padding'] / parameters['step'])
            if end_mode and end_padding > 0:
                cst += f'AtomPair ORIG {virt_curr}Y {nbr_s} HARMONIC 0.0 {ipdc_sd:.3f}\n'
            else:
                if 0 < i_to_end <= end_padding:
                    ipdc_d0 = ipdc_d0 / end_padding * i_to_end
                if ipdc_type == 'bounded':
                    cst += f'AtomPair ORIG {virt_curr}Y {nbr_s} BOUNDED 0.0 {ipdc_d0:.3f} {ipdc_sd:.3f} 0.5 tag\n'
                elif ipdc_type == 'harmonic':
                    cst += f'AtomPair ORIG {virt_curr}Y {nbr_s} HARMONIC 0.0 {ipdc_sd:.3f}\n'
        return cst

    @staticmethod
    def create_constraints_trajectory(lig_info, parameters, i_to_end=float('inf')):
        '''
        Generate Rosetta constraint text for a trajectory window.

        Virtual atom layout (TrajVirt):
          TRI1-3:    equilateral triangle for plane constraints
          PREV_PREV: second-previous NBR position  (angle_constraint: prev_prev — prev — nbr)
          PREV:      previous NBR position          (next_angle_constraint: prev — nbr — next)
          CURR:      current guide point            (in-plane dev target)
          NEXT:      next guide point               (forward angle)
        '''
        n_virt = len(TrajVirt)
        start_mode = (i_to_end == -1)
        end_mode   = (i_to_end == 0)
        cst, nbr_s, virt_nums = ConstraintBuilder._constraint_preamble(lig_info, parameters, n_virt)
        V = TrajVirt

        if parameters['next_angle_constraint'] and not start_mode and not end_mode:
            angle_type = parameters['next_angle_constraint_type'].lower()
            angle_x    = parameters['next_angle_constraint_bound']
            angle_sd   = parameters['next_angle_constraint_sd']
            if angle_type == 'bounded':
                cst += (f'Angle ORIG {virt_nums[V.PREV]}Y {nbr_s} ORIG {virt_nums[V.NEXT]}Y'
                        f' BOUNDED {angle_x:.3f} 3.142 {angle_sd:.3f} 0.5 tag\n')
            elif angle_type == 'harmonic':
                cst += (f'Angle ORIG {virt_nums[V.PREV]}Y {nbr_s} ORIG {virt_nums[V.NEXT]}Y'
                        f' HARMONIC 3.1416 {angle_sd:.3f}\n')

        if parameters['angle_constraint'] and not start_mode:
            angle_type = parameters['angle_constraint_type'].lower()
            angle_x    = parameters['angle_constraint_bound']
            angle_sd   = parameters['angle_constraint_sd']
            if angle_type == 'bounded':
                cst += (f'Angle ORIG {virt_nums[V.PREV_PREV]}Y ORIG {virt_nums[V.PREV]}Y {nbr_s}'
                        f' BOUNDED {angle_x:.3f} 3.142 {angle_sd:.3f} 0.5 tag\n')
            elif angle_type == 'harmonic':
                cst += (f'Angle ORIG {virt_nums[V.PREV_PREV]}Y ORIG {virt_nums[V.PREV]}Y {nbr_s}'
                        f' HARMONIC 3.1416 {angle_sd:.3f}\n')

        cst = ConstraintBuilder._append_rmsd_and_ipdc(
            cst, lig_info, parameters, i_to_end, end_mode, virt_nums[V.CURR], n_virt
        )
        cst += 'END'
        return cst

    @staticmethod
    def create_constraints_coarse(lig_info, parameters, i_to_end=float('inf'),
                                  has_second_left=True, has_second_right=True):
        '''
        Generate Rosetta constraint text for a coarse-path node.

        Virtual atom layout (CoarseVirt):
          TRI1-3: equilateral triangle for plane constraints
          LM:     left-neighbour position
          RM:     right-neighbour position
          CURR:   new_start (in-plane dev target)
          LLM:    second-order left (or LM when left neighbour is pseudo)
          RRM:    second-order right (or RM when right neighbour is pseudo)

        angle_constraint:      Angle(LM — nbr — RM)        nbr is the vertex
        next_angle_constraint: Angle(LLM — LM — nbr) and
                               Angle(RRM — RM — nbr)       nbr is the endpoint

        has_second_left/has_second_right: pass False when the corresponding
        neighbour is a pseudo endpoint (so LLM/RRM would equal LM/RM, giving a
        zero-length-bond angle constraint) to skip just that one-sided line.
        '''
        n_virt = len(CoarseVirt)
        start_mode = (i_to_end == -1)
        end_mode   = (i_to_end == 0)
        cst, nbr_s, virt_nums = ConstraintBuilder._constraint_preamble(lig_info, parameters, n_virt)
        V = CoarseVirt

        if parameters['angle_constraint'] and not start_mode:
            angle_type = parameters['angle_constraint_type'].lower()
            angle_x    = parameters['angle_constraint_bound']
            angle_sd   = parameters['angle_constraint_sd']
            if angle_type == 'bounded':
                cst += (f'Angle ORIG {virt_nums[V.LM]}Y {nbr_s} ORIG {virt_nums[V.RM]}Y'
                        f' BOUNDED {angle_x:.3f} 3.142 {angle_sd:.3f} 0.5 tag\n')
            elif angle_type == 'harmonic':
                cst += (f'Angle ORIG {virt_nums[V.LM]}Y {nbr_s} ORIG {virt_nums[V.RM]}Y'
                        f' HARMONIC 3.1416 {angle_sd:.3f}\n')

        if parameters['next_angle_constraint'] and not start_mode:
            angle_type = parameters['next_angle_constraint_type'].lower()
            angle_x    = parameters['next_angle_constraint_bound']
            angle_sd   = parameters['next_angle_constraint_sd']
            if angle_type == 'bounded':
                if has_second_left:
                    cst += (f'Angle ORIG {virt_nums[V.LLM]}Y ORIG {virt_nums[V.LM]}Y {nbr_s}'
                            f' BOUNDED {angle_x:.3f} 3.142 {angle_sd:.3f} 0.5 tag\n')
                if has_second_right:
                    cst += (f'Angle ORIG {virt_nums[V.RRM]}Y ORIG {virt_nums[V.RM]}Y {nbr_s}'
                            f' BOUNDED {angle_x:.3f} 3.142 {angle_sd:.3f} 0.5 tag\n')
            elif angle_type == 'harmonic':
                if has_second_left:
                    cst += (f'Angle ORIG {virt_nums[V.LLM]}Y ORIG {virt_nums[V.LM]}Y {nbr_s}'
                            f' HARMONIC 3.1416 {angle_sd:.3f}\n')
                if has_second_right:
                    cst += (f'Angle ORIG {virt_nums[V.RRM]}Y ORIG {virt_nums[V.RM]}Y {nbr_s}'
                            f' HARMONIC 3.1416 {angle_sd:.3f}\n')

        cst = ConstraintBuilder._append_rmsd_and_ipdc(
            cst, lig_info, parameters, i_to_end, end_mode, virt_nums[V.CURR], n_virt
        )
        cst += 'END'
        return cst


# ----------------------------------------------------------------
# ConformerSampler — RMSD-based conformer filtering
# ----------------------------------------------------------------

class ConformerSampler:

    @staticmethod
    def rmsd(mol1, mol2):
        mol1 = np.array(mol1)
        mol2 = np.array(mol2)
        if mol1.shape != mol2.shape:
            raise ValueError(
                f"rmsd: mol1 and mol2 must have the same shape, got {mol1.shape} vs {mol2.shape}"
            )
        if len(mol1) == 0:
            raise ValueError("rmsd: cannot compute RMSD of empty molecules")
        return np.sqrt(np.sum(np.square(mol1 - mol2)) / len(mol1))

    @staticmethod
    def optimal_align_rmsd(query_mol, mobile_mol, nbr_atom_idx=0):
        query_mol = np.array(query_mol)
        mobile_mol = np.array(mobile_mol)
        if query_mol.shape != mobile_mol.shape:
            raise ValueError(
                f"optimal_align_rmsd: query and mobile must have the same shape, "
                f"got {query_mol.shape} vs {mobile_mol.shape}"
            )
        if nbr_atom_idx < 0 or nbr_atom_idx >= len(query_mol):
            raise IndexError(
                f"optimal_align_rmsd: nbr_atom_idx {nbr_atom_idx} out of range for molecule with {len(query_mol)} atoms"
            )

        query_nbr = copy(query_mol[nbr_atom_idx])
        mobile_nbr = copy(mobile_mol[nbr_atom_idx])
        query_mol -= query_nbr
        mobile_mol -= mobile_nbr
        rotation = transform.Rotation.align_vectors(query_mol, mobile_mol)[0]
        aligned_mobile = rotation.apply(mobile_mol)
        r = ConformerSampler.rmsd(aligned_mobile, query_mol)
        aligned_mobile += query_nbr
        return r, aligned_mobile

    @staticmethod
    def filter_coordset(input_coords, coordset, threshold, nbr_atom_idx=0):
        if threshold <= 0:
            raise ValueError(f"filter_coordset: threshold must be positive, got {threshold}")
        new_coordset = []
        for coords in coordset:
            r, updated = ConformerSampler.optimal_align_rmsd(input_coords, coords, nbr_atom_idx)
            if r < threshold:
                new_coordset.append(updated)
        return new_coordset

    @staticmethod
    def write_coordset(names, coordset, output_fn, chain='X'):
        if not names:
            raise ValueError("write_coordset: names list is empty")
        if not coordset:
            raise ValueError("write_coordset: coordset is empty")
        if not chain or len(chain) != 1:
            raise ValueError(f"write_coordset: chain must be a single character, got {chain!r}")
        with open(output_fn, 'w') as fh:
            for coords in coordset:
                if len(coords) != len(names):
                    raise ValueError(
                        f"write_coordset: coords length {len(coords)} does not match names length {len(names)}"
                    )
                for i, x in enumerate(coords):
                    fh.write(f'HETATM {str(i+1).rjust(4)}  {names[i].ljust(3)} lig {chain}   1'
                             f'    {x[0]:8.3f}{x[1]:8.3f}{x[2]:8.3f}  1.00  1.00\n')
                fh.write('TER\n')

    @staticmethod
    def subsample_pdb(query_pdb, input_conformers_pdb, output_conformers_pdb,
                      threshold, chain, nbr_atom_name):
        if not os.path.isfile(query_pdb):
            raise FileNotFoundError(f"subsample_pdb: query PDB not found: {query_pdb}")
        if not os.path.isfile(input_conformers_pdb):
            raise FileNotFoundError(f"subsample_pdb: conformers PDB not found: {input_conformers_pdb}")
        if threshold <= 0:
            raise ValueError(f"subsample_pdb: threshold must be positive, got {threshold}")

        query, nbr_atom_idx, names, resnum = PDBHandler.read_window(query_pdb, chain, nbr_atom_name)
        query = np.array(query)
        query -= query[nbr_atom_idx]

        input_coordset = []
        with open(input_conformers_pdb, 'r') as fh:
            coords = {}
            for line in fh:
                if line.startswith('TER'):
                    missing_atoms = [an for an in names if an not in coords]
                    if missing_atoms:
                        raise ValueError(
                            f"subsample_pdb: conformer in {input_conformers_pdb} is missing atoms: {missing_atoms}"
                        )
                    input_coordset.append([coords[an] for an in names])
                    coords = {}
                else:
                    atom_name = line[12:16].strip()
                    if atom_name:
                        coords[atom_name] = [float(line[30:38]), float(line[38:46]), float(line[46:54])]

        if not input_coordset:
            return False

        output_coordset = ConformerSampler.filter_coordset(query, input_coordset, threshold, nbr_atom_idx)
        output_coordset.append(query)
        ConformerSampler.write_coordset(names, output_coordset, output_conformers_pdb, chain)
        return True

    @staticmethod
    def check_rmsd(target_pdb, query_pdb, chain, nbr_atom_name):
        if not os.path.isfile(target_pdb):
            raise FileNotFoundError(f"check_rmsd: target PDB not found: {target_pdb}")
        if not os.path.isfile(query_pdb):
            raise FileNotFoundError(f"check_rmsd: query PDB not found: {query_pdb}")
        query, _, query_names, _ = PDBHandler.read_window(query_pdb, chain, nbr_atom_name)
        target, _, target_names, _ = PDBHandler.read_window(target_pdb, chain, nbr_atom_name)
        if len(query) != len(target):
            raise ValueError(
                f"check_rmsd: atom count mismatch between query ({len(query)}) and target ({len(target)})"
            )
        name_to_idx = {name: i for i, name in enumerate(target_names)}
        try:
            reordered = np.array([target[name_to_idx[n]] for n in query_names])
        except KeyError as e:
            raise ValueError(f"check_rmsd: atom {e} present in query but missing from target") from e
        return ConformerSampler.rmsd(query, reordered)
