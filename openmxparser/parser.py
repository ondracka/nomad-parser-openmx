#
# Copyright The NOMAD Authors.
#
# This file is part of NOMAD. See https://nomad-lab.eu for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import numpy as np
import re
from os import path

from nomad.datamodel import EntryArchive
from nomad.parsing import FairdiParser
from nomad.units import ureg as units
from nomad.datamodel.metainfo.public import section_run as Run
from nomad.datamodel.metainfo.public import section_scf_iteration as SCF
from nomad.datamodel.metainfo.public import section_system as System
from nomad.datamodel.metainfo.public import section_single_configuration_calculation as SCC
from nomad.datamodel.metainfo.public import section_method as Method
from nomad.datamodel.metainfo.public import section_sampling_method
from nomad.datamodel.metainfo.public import section_XC_functionals as xc_functionals
from nomad.parsing.file_parser import UnstructuredTextFileParser, Quantity

from .metainfo.openmx import OpenmxSCC  # pylint: disable=unused-import

'''
This is parser for OpenMX DFT code.
'''

A = (1 * units.angstrom).to_base_units().magnitude

scf_step_parser = UnstructuredTextFileParser(quantities=[
    Quantity('scf_step_number', r'   SCF=\s*(\d+)', repeats=False),
    Quantity('norm_rd', r'NormRD=\s*([\d\.]+)', repeats=False),
    Quantity('u_ele', r'Uele=\s*([-\d\.]+)', repeats=False)
])

md_step_parser = UnstructuredTextFileParser(quantities=[
    Quantity('scf_step', r'   (SCF=.+?Uele=\s*[-\d\.]+)', sub_parser=scf_step_parser, repeats=True),
    Quantity('Utot', r'Utot\.\s+(-?\d+\.\d+)', repeats=False)
])

input_atoms_parser = UnstructuredTextFileParser(quantities=[
    Quantity('atom', r'\s*\d+\s*([A-Za-z]{1,2})\s*([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)\s+[\d\.]+\s*[\d\.]+\s*',
             repeats=True)
])

mainfile_parser = UnstructuredTextFileParser(quantities=[
    Quantity('program_version', r'This calculation was performed by OpenMX Ver. ([\d\.]+)\s*', repeats=False),
    Quantity(
        'md_step', r'(SCF history at MD[\s\S]+?Chemical potential \(Hartree\)\s+[-\d\.]+)',
        sub_parser=md_step_parser,
        repeats=True),
    Quantity(
        'input_atoms', r'<Atoms.SpeciesAndCoordinates([\s\S]+)Atoms.SpeciesAndCoordinates>',
        sub_parser=input_atoms_parser,
        repeats=False),
    Quantity(
        'input_lattice_vectors', r'(?i)<Atoms.UnitVectors\s+((?:-?\d+\.\d+\s+)+)Atoms.UnitVectors>',
        repeats=False),
    Quantity('scf_XcType', r'scf.XcType\s+(\S+)', repeats=False),
    Quantity('scf_SpinPolarization', r'scf.SpinPolarization\s+(\S+)', repeats=False),
    Quantity('atoms_coordinates_units',
             r'(?i)Atoms.SpeciesAndCoordinates.Unit\s+([a-z]{2,4})', repeats=False),
    Quantity('lattice_vectors_units',
             r'(?i)Atoms.UnitVectors.Unit\s+([a-z]{2,3})', repeats=False),
    Quantity('scf_hubbard_u', r'(?i)scf.Hubbard.U\s+(on|off)', repeats=False),
    Quantity('md_type', r'(?i)MD\.Type\s+([a-z_\d]{3,6})', repeats=False),
    Quantity('md_opt_criterion', r'(?i)MD\.Opt\.criterion\s+([\d\.e-]+)', repeats=False),
])


def parse_md_file(md_file):
    result = []
    with open(md_file, 'r') as f:
        cell_vectors_re = re.compile(r'Cell_Vectors=((?:\s+-?\d+\.\d+)+)')
        temperature_re = re.compile(r'Temperature=\s+(\d+\.\d+)')
        for line in f:
            line_list = line.split()
            if len(line_list) == 1:
                step_header = True
                natoms = int(line_list[0])
                result.append({'species': [], 'positions': np.empty((natoms, 3))})
                atomindex = 0
            elif step_header:
                cell_vectors = cell_vectors_re.search(line)
                if cell_vectors is not None:
                    cell_vectors = [float(v) for v in cell_vectors.group(1).split()]
                    result[-1]['cell_vectors'] = np.array(cell_vectors).reshape(3, 3)
                temperature = temperature_re.search(line)
                if temperature is not None:
                    temperature = temperature.group(1)
                    result[-1]['temperature'] = temperature
                step_header = False
            else:
                result[-1]['positions'][atomindex][0:3] = [
                    float(val) for val in line_list[1:4]]
                result[-1]['species'].append(line_list[0])
                atomindex += 1
    f.close()
    return result


class OpenmxParser(FairdiParser):
    def __init__(self):
        super().__init__(
            name='parsers/openmx', code_name='OpenMX', code_homepage='http://www.openmx-square.org/',
            mainfile_mime_re=r'(text/.*)',
            mainfile_name_re=r'.*\.out$',
            mainfile_contents_re=(r'^\*{59}\s+\*{59}\s+This calculation was performed by OpenMX'),
        )

    def parse(self, mainfile: str, archive: EntryArchive, logger):

        # Use the previously defined parsers on the given mainfile
        mainfile_parser.mainfile = mainfile
        mainfile_parser.parse()
        del mainfile_parser._file_handler

        # Get system from the MD file
        md_file = path.splitext(mainfile)[0] + '.md'
        if path.isfile(md_file):
            mdfile_md_steps = parse_md_file(md_file)
        else:
            mdfile_md_steps = None

        # Some basic values
        run = archive.m_create(Run)
        run.program_name = 'OpenMX'
        run.program_version = str(mainfile_parser.get('program_version'))
        run.program_basis_set_type = "Numeric AOs"

        method = run.m_create(Method)
        method.electronic_structure_method = 'DFT'
        # FIXME: add some testcase for DFT+U
        scf_hubbard_u = mainfile_parser.get('scf_hubbard_u')
        if scf_hubbard_u is not None:
            if mainfile_parser.get('scf_hubbard_u').lower == 'on':
                method.electronic_structure_method = 'DFT+U'

        xc_functional_dictionary = {
            'GGA-PBE': ['GGA_C_PBE', 'GGA_X_PBE'],
            'LDA': ['LDA_X', 'LDA_C_PZ'],
            'LSDA-CA': ['LDA_X', 'LDA_C_PZ'],
            'LSDA-PW': ['LDA_X', 'LDA_C_PW'],
            None: ['LDA_X', 'LDA_C_PZ']
        }
        scf_XcType = mainfile_parser.get('scf_XcType')
        for xc in xc_functional_dictionary[scf_XcType]:
            method.m_create(xc_functionals).XC_functional_name = xc
        scf_XcType = mainfile_parser.get('scf_XcType')

        scf_SpinPolarizationType = mainfile_parser.get('scf_SpinPolarization')
        if scf_SpinPolarizationType.lower() == 'on':
            method.number_of_spin_channels = 2
        else:
            method.number_of_spin_channels = 1

        md_type = mainfile_parser.get('md_type')
        md_types_list = [
            # FIXME: handle the various OptCx methods with constraints
            ['OPT', 'geometry_optimization', 'steepest_descent'],
            ['DIIS', 'geometry_optimization', 'diis'],
            ['BFGS', 'geometry_optimization', 'bfgs'],
            # FIXME: not in https://gitlab.mpcdf.mpg.de/nomad-lab/nomad-meta-info/-/wikis/metainfo/geometry-optimization-method
            ['RF', 'geometry_optimization', 'rf'],
            # FIXME: not in https://gitlab.mpcdf.mpg.de/nomad-lab/nomad-meta-info/-/wikis/metainfo/geometry-optimization-method
            ['EF', 'geometry_optimization', 'ef'],
            ['NVE', 'molecular_dynamics', 'NVE'],
            ['NVT_VS', 'molecular_dynamics', 'NVT'],
            ['NVT_NH', 'molecular_dynamics', 'NVT'],
        ]
        if md_type is not None and 'nomd' not in md_type.lower():
            md_type = md_type.upper()
            sampling_method = run.m_create(section_sampling_method)
            for current_md_type in md_types_list:
                if current_md_type[0] in md_type:
                    sampling_method.sampling_method = current_md_type[1]
                    if current_md_type[1] == 'geometry_optimization':
                        sampling_method.geometry_optimization_method = current_md_type[2]
                        criterion = mainfile_parser.get('md_opt_criterion')
                        if criterion is not None:
                            sampling_method.geometry_optimization_threshold_force = (
                                criterion * units.hartree / units.bohr)
                        else:
                            sampling_method.geometry_optimization_threshold_force = (
                                1e-4 * units.hartree / units.bohr)
                    else:
                        sampling_method.ensemble_type = current_md_type[2]

        mainfile_md_steps = mainfile_parser.get('md_step')
        if mainfile_md_steps is not None:
            n_md_steps = len(mainfile_md_steps)
        if mdfile_md_steps is not None:
            n_mdfile_md_steps = len(mdfile_md_steps)
        # Do some consistency checks between the out and md file.
        ignore_md_file = False
        if n_mdfile_md_steps > n_md_steps:
            # This can happen when user runs two calculations in the same directory.
            # In that case the out file contains the latter calculation but the md file
            # would contain both calculations, so just take the corresponding number
            # of steps from the end of the file.
            mdfile_md_steps = mdfile_md_steps[-n_md_steps:]
        elif n_mdfile_md_steps < n_md_steps:
            # This is unlikely, but signals a problem with the md file, so just
            # ignore it.
            ignore_md_file = True
            logger.warning(".md file does not contain enough MD steps")

        if mainfile_md_steps is not None:
            for i, md_step in enumerate(mainfile_md_steps):
                system = run.m_create(System)
                if not ignore_md_file:
                    cell = mdfile_md_steps[i].get('cell_vectors')
                    system.lattice_vectors = cell * units.angstrom
                    system.configuration_periodic_dimensions = [True, True, True]
                    positions = mdfile_md_steps[i].get('positions')
                    system.atom_positions = positions * units.angstrom
                    system.atom_labels = mdfile_md_steps[i].get('species')
                if i == 0:
                    # Get the initial and final position from out file, it has better precision
                    # and we also have some fallback if the md file is missing.
                    atoms_units = mainfile_parser.get('atoms_coordinates_units')
                    lattice_vectors = mainfile_parser.get('input_lattice_vectors')
                    lattice_units = mainfile_parser.get('lattice_vectors_units')
                    atoms = mainfile_parser.get('input_atoms').get('atom')

                    if atoms is not None and lattice_vectors is not None:
                        lattice_vectors = np.array(lattice_vectors).reshape(3, 3)
                        if atoms_units is not None:
                            if lattice_units.lower() == 'au':
                                lattice_vectors = lattice_vectors * units.bohr
                                lattice_units = units.bohr
                            elif lattice_units.lower() == 'ang':
                                lattice_vectors = lattice_vectors * units.angstrom
                                lattice_units = units.angstrom
                        else:
                            lattice_vectors = lattice_vectors * units.angstrom
                        atom_positions = [a[1:4] for a in atoms]
                        if atoms_units is not None:
                            if atoms_units.lower() == 'au':
                                atom_positions = atom_positions * units.bohr
                            if atoms_units.lower() == 'ang':
                                atom_positions = atom_positions * units.angstrom
                            elif atoms_units.lower() == 'frac':
                                # There are some problems with pint here so simple matrix vector multiplications
                                # doesn't work.
                                atom_positions = np.array([np.array(pos).dot(lattice_vectors)
                                                          for pos in atom_positions]) * lattice_units
                        else:
                            # default unit is angstrom
                            atom_positions = atom_positions * units.bohr
                        system.atom_positions = atom_positions
                        system.lattice_vectors = lattice_vectors
                        system.configuration_periodic_dimensions = [True, True, True]
                        system.atom_labels = [a[0] for a in atoms]
                    else:
                        logger.warning('Failed to parse the input structure.')

                if i == n_md_steps - 1:
                    logger.warning("Not implemented")

                scc = run.m_create(SCC)
                scc.single_configuration_calculation_to_system_ref = system
                scc.single_configuration_to_calculation_method_ref = method
                scf_steps = md_step.get('scf_step')
                if scf_steps is not None:
                    for scf_step in scf_steps:
                        scf = scc.m_create(SCF)
                        u_ele = scf_step.get('u_ele')
                        if u_ele is not None:
                            scf.energy_sum_eigenvalues_scf_iteration = u_ele * units.hartree
                u_tot = md_step.get('Utot')
                if u_tot is not None:
                    scc.energy_total = u_tot * units.hartree
                temperature = mdfile_md_steps[i].get('temperature')
                if temperature is not None:
                    scc.temperature = temperature * units.kelvin
