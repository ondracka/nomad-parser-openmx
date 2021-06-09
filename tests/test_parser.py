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


def Ha_to_J(value):
    return (value * units.hartree).to_base_units().magnitude


def K_to_J(value):
    return (value * units.joules).to_base_units().magnitude


# default pytest.approx settings are abs=1e-12, rel=1e-6 so it doesn't work for small numbers
# use the default just for comparison with zero
def approx(value):
    return pytest.approx(value, abs=0, rel=1e-6)


def test_HfO2(parser):
    '''
    Simple single point calculation monoclinic HfO2 test case.
    '''
    archive = EntryArchive()
    parser.parse('tests/HfO2_single_point/m-HfO2.out', archive, logging)

    run = archive.section_run[0]
    assert run.program_version == '3.9.2'
    assert run.program_basis_set_type == 'Numeric AOs'
    scc = run.section_single_configuration_calculation
    assert len(scc) == 1
    assert scc[0].energy_total.magnitude == approx(Ha_to_J(-346.328738171942))
    scf = scc[0].section_scf_iteration
    assert len(scf) == 24
    scf[3].energy_sum_eigenvalues_scf_iteration == approx(-3.916702417016777e-16)

    method = run.section_method[0]
    section_XC_functionals1 = method.section_XC_functionals[0]
    section_XC_functionals2 = method.section_XC_functionals[1]
    assert method.number_of_spin_channels == 1
    assert method.electronic_structure_method == 'DFT'
    assert method.smearing_width == approx(K_to_J(300))
    assert section_XC_functionals1.XC_functional_name == 'GGA_C_PBE'
    assert section_XC_functionals2.XC_functional_name == 'GGA_X_PBE'

    assert run.section_sampling_method == []

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
    parser.parse('tests/AlN_ionic_optimization/AlN.out', archive, logging)

    run = archive.section_run[0]
    assert run.program_version == '3.9.2'
    assert run.program_basis_set_type == 'Numeric AOs'
    scc = run.section_single_configuration_calculation
    assert len(scc) == 5
    assert scc[0].energy_total.magnitude == approx(Ha_to_J(-25.194346653540))
    assert scc[4].energy_total.magnitude == approx(Ha_to_J(-25.194358042252))
    scf = scc[0].section_scf_iteration
    assert len(scf) == 21
    scf[20].energy_sum_eigenvalues_scf_iteration == approx(-3.4038353611878345e-17)
    scf = scc[3].section_scf_iteration
    assert len(scf) == 6
    scf[5].energy_sum_eigenvalues_scf_iteration == approx(-3.4038520917173614e-17)

    method = run.section_method[0]
    section_XC_functionals1 = method.section_XC_functionals[0]
    section_XC_functionals2 = method.section_XC_functionals[1]
    assert method.number_of_spin_channels == 1
    assert method.electronic_structure_method == 'DFT'
    assert method.smearing_width == approx(K_to_J(300))
    assert section_XC_functionals1.XC_functional_name == 'GGA_C_PBE'
    assert section_XC_functionals2.XC_functional_name == 'GGA_X_PBE'

    sampling_method = run.section_sampling_method
    assert len(sampling_method) == 1
    assert sampling_method[0].geometry_optimization_method == "steepest_descent"
    assert sampling_method[0].sampling_method == "geometry_optimization"
    assert sampling_method[0].geometry_optimization_threshold_force.magnitude == approx(
        (0.0003 * units.hartree / units.bohr).to_base_units().magnitude)

    assert len(run.section_system) == 5
    system = run.section_system[0]
    assert all([a == b for a, b in zip(system.configuration_periodic_dimensions,
                                       [True, True, True])])
    assert system.lattice_vectors[0][0].magnitude == approx(A_to_m(3.10997))
    assert system.lattice_vectors[1][0].magnitude == approx(A_to_m(-1.55499))
    assert len(system.atom_positions) == 4
    assert system.atom_positions[0][0].magnitude == approx(A_to_m(1.55499))
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


def test_C2N2(parser):
    '''
    Molecular dynamics using the Nose-Hover thermostat for simple N2H2 molecule
    '''
    archive = EntryArchive()
    parser.parse('tests/C2H2_molecular_dynamics/C2H2.out', archive, logging)

    run = archive.section_run[0]
    assert run.program_version == '3.9.2'
    assert run.program_basis_set_type == 'Numeric AOs'
    scc = run.section_single_configuration_calculation
    assert len(scc) == 100
    assert scc[0].temperature.magnitude == approx(300.0)
    assert scc[99].temperature.magnitude == approx(46.053)

    assert len(run.section_system) == 100

    method = run.section_method[0]
    assert method.number_of_spin_channels == 1
    assert method.electronic_structure_method == 'DFT'
    assert method.smearing_width == approx(K_to_J(500))
    assert method.section_XC_functionals[0].XC_functional_name == 'LDA_X'
    assert method.section_XC_functionals[1].XC_functional_name == 'LDA_C_PZ'

    sampling_method = run.section_sampling_method
    assert len(sampling_method) == 1
    assert sampling_method[0].sampling_method == "molecular_dynamics"
    assert sampling_method[0].ensemble_type == "NVT"
