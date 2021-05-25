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

import pytest
import logging

from nomad.datamodel import EntryArchive
from nomad.units import ureg as units

from openmxparser import OpenmxParser


@pytest.fixture
def parser():
    return OpenmxParser()


def A_to_m(value):
    return (value * units.angstrom).to_base_units().magnitude


# default pytest.approx settings are abs=1e-12, rel=1e-6 so it doesn't work for small numbers
# use the default just for comparison with zero
def approx(value):
    return pytest.approx(value, abs=0, rel=1e-6)


def test_HfO2(parser):
    '''
    Simple single point calculation monoclinic HfO2 test case.
    '''
    archive = EntryArchive()
    parser.parse('tests/data/HfO2_single_point/m-HfO2.out', archive, logging)

    run = archive.section_run[0]
    assert run.program_version == '3.9.2'
    scc = run.section_single_configuration_calculation
    assert len(scc) == 1
    scf = scc[0].section_scf_iteration
    assert len(scf) == 24
    scf[3].energy_total_scf_iteration == pytest.approx(-3.916702417016777e-16)
    method = run.section_method[0]
    section_XC_functionals1 = method.section_XC_functionals[0]
    section_XC_functionals2 = method.section_XC_functionals[1]
    assert method.number_of_spin_channels == 1
    assert method.electronic_structure_method == 'DFT'
    assert section_XC_functionals1.XC_functional_name == 'GGA_C_PBE'
    assert section_XC_functionals2.XC_functional_name == 'GGA_X_PBE'

    system = run.section_system[0]
    assert all([a == b for a, b in zip(system.configuration_periodic_dimensions,
                                       [True, True, True])])
    assert system.lattice_vectors[0][0].magnitude == approx(A_to_m(5.1156000))
    assert system.lattice_vectors[2][2].magnitude == approx(A_to_m(5.2269843))
    assert len(system.atom_positions) == 12
    assert system.atom_positions[5][0].magnitude == approx(A_to_m(-0.3293636))
    assert system.atom_positions[11][2].magnitude == approx(A_to_m(2.6762159))
    assert len(system.atom_labels) == 12
    assert system.atom_labels[9] == 'O'


def test_AlN(parser):
    '''
    Geometry optimization (atomic positions only) AlN test case.
    '''
    archive = EntryArchive()
    parser.parse('tests/data/AlN_ionic_optimization/AlN.out', archive, logging)

    run = archive.section_run[0]
    assert run.program_version == '3.9.2'
    scc = run.section_single_configuration_calculation
    assert len(scc) == 5
    scf = scc[0].section_scf_iteration
    assert len(scf) == 21
    scf[20].energy_total_scf_iteration == pytest.approx(-3.4038353611878345e-17)
    scf = scc[3].section_scf_iteration
    assert len(scf) == 6
    scf[5].energy_total_scf_iteration == pytest.approx(-3.4038520917173614e-17)
    method = run.section_method[0]
    section_XC_functionals1 = method.section_XC_functionals[0]
    section_XC_functionals2 = method.section_XC_functionals[1]
    assert method.number_of_spin_channels == 1
    assert method.electronic_structure_method == 'DFT'
    assert section_XC_functionals1.XC_functional_name == 'GGA_C_PBE'
    assert section_XC_functionals2.XC_functional_name == 'GGA_X_PBE'

    assert len(run.section_system) == 5
    system = run.section_system[0]
    assert all([a == b for a, b in zip(system.configuration_periodic_dimensions,
                                       [True, True, True])])
    assert system.lattice_vectors[0][0].magnitude == approx(A_to_m(3.109970))
    assert system.lattice_vectors[1][0].magnitude == approx(A_to_m(-1.5549850))
    assert len(system.atom_positions) == 4
    assert system.atom_positions[0][0].magnitude == approx(A_to_m(1.55498655))
    assert system.atom_positions[3][2].magnitude == approx(A_to_m(4.39210))
    assert len(system.atom_labels) == 4
    assert system.atom_labels[3] == 'N'
    system = run.section_system[3]
    assert system.lattice_vectors[1][1].magnitude == approx(A_to_m(2.69331))
    assert system.lattice_vectors[2][2].magnitude == approx(A_to_m(4.98010))
    assert len(system.atom_positions) == 4
    assert system.atom_positions[0][1].magnitude == approx(A_to_m(0.89807))
    assert system.atom_positions[2][2].magnitude == approx(A_to_m(1.90030))
    assert len(system.atom_labels) == 4
    assert system.atom_labels[0] == 'Al'
    system = run.section_system[4]
    assert system.lattice_vectors[0][0].magnitude == approx(A_to_m(3.10997))
    assert system.lattice_vectors[2][2].magnitude == approx(A_to_m(4.98010))
    assert len(system.atom_positions) == 4
    assert system.atom_positions[0][2].magnitude == approx(A_to_m(0.00253))
    assert system.atom_positions[3][2].magnitude == approx(A_to_m(4.39015))
    assert len(system.atom_labels) == 4
    assert system.atom_labels[1] == 'Al'
