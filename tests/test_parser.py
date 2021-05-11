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

from openmxparser import OpenmxParser


@pytest.fixture
def parser():
    return OpenmxParser()


def test_openmx(parser):
    archive = EntryArchive()
    parser.run('tests/data/HfO2_single_point/m-HfO2.out', archive, logging)

    run = archive.section_run[0]
    assert run.program_version == '3.9.2'
    scc = run.section_single_configuration_calculation
    assert len(scc) == 1
    scf = scc[0].section_scf_iteration
    assert len(scf) == 24
    scf[3].energy_total_scf_iteration == pytest.approx(-3.916702417016777e-16)
    method = run.section_method[0]
    assert method.number_of_spin_channels == 1

    archive = EntryArchive()
    parser.run('tests/data/AlN_ionic_optimization/AlN.out', archive, logging)

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
    assert method.number_of_spin_channels == 1
