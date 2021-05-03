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

import datetime
import numpy as np

from nomad.datamodel import EntryArchive
from nomad.parsing import FairdiParser
from nomad.units import ureg as units
from nomad.datamodel.metainfo.public import section_run as Run
from nomad.datamodel.metainfo.public import section_scf_iteration as SCF
from nomad.datamodel.metainfo.public import section_system as System
from nomad.datamodel.metainfo.public import section_single_configuration_calculation as SCC
from nomad.datamodel.metainfo.public import section_method as Method
from nomad.parsing.file_parser import UnstructuredTextFileParser, Quantity

from . import metainfo  # pylint: disable=unused-import

'''
This is a hello world style example for an example parser/converter.
'''

A = (1 * units.angstrom).to_base_units().magnitude


def str_to_sites(string):
    sym, pos = string.split('(')
    pos = np.array(pos.split(')')[0].split(',')[:3], dtype=float)
    return sym, pos


scf_step_parser = UnstructuredTextFileParser(quantities=[
    Quantity('scf_step_number', r'   SCF=\s*(\d+)', repeats=False),
    Quantity('norm_rd', r'NormRD=\s*([\d\.]+)', repeats=False),
    Quantity('ene', r'Uele=\s*([-\d\.]+)', repeats=False)
    ])

md_step_parser = UnstructuredTextFileParser(quantities=[
    Quantity('scf_step', r'   (SCF=.+?Uele=\s*[-\d\.]+)', sub_parser=scf_step_parser, repeats=True)
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
    Quantity('scf_XcType', r'scf.XcType                    (\S+)', repeats=False),
    Quantity('scf_SpinPolarization', r'scf.SpinPolarization          (\S+)', repeats=False)

    ])


class OpenmxParser(FairdiParser):
    def __init__(self):
        super().__init__(
            name='parsers/openmx', code_name='OpenMX', code_homepage='http://www.openmx-square.org/',
            mainfile_mime_re=r'(text/.*)',
            mainfile_contents_re=(r'^\s*#\s*This is example output'),
        )

    def run(self, mainfile: str, archive: EntryArchive, logger):

        # Use the previously defined parsers on the given mainfile
        mainfile_parser.mainfile = mainfile
        mainfile_parser.parse()

        # Output all parsed data into the given archive.
        run = archive.m_create(Run)
        run.program_name = 'OpenMX'
        run.program_version = str(mainfile_parser.get('program_version'))

        input_atoms = mainfile_parser.get('input_atoms')
        atoms = input_atoms.get('atom')
        print(atoms)
        system = run.m_create(System)
        system.atom_positions = [[a[1] * A, a[2] * A, a[3] * A] for a in atoms]

        scf_XcType = mainfile_parser.get('scf_XcType')
        print('\n\n\tscf.XcType is: ',scf_XcType)
        method = run.m_create(Method)
        scf_SpinPolarizationType = mainfile_parser.get('scf_SpinPolarization')
        print('\n\tscf.SpinPolarization Type is: ',scf_SpinPolarizationType)
        if scf_SpinPolarizationType.lower() == 'on':
            method.number_of_spin_channels = 2
        else :
            method.number_of_spin_channels = 1
             
        print ('\tnumber of spin channels is: ',method.number_of_spin_channels,'\n\n') 

        md_steps = mainfile_parser.get('md_step')
        if md_steps is not None:
            for md_step in md_steps:
                scc = run.m_create(SCC)
                scf_steps = md_step.get('scf_step')
                if scf_steps is not None:
                    for scf_step in scf_steps:
                        scf = scc.m_create(SCF)
                        ene = scf_step.get('ene')
                        if ene is not None:
                            # FIXME: double check that this is indeed the total energy
                            scf.energy_total_scf_iteration = ene * units.hartree
