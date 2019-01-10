"Create initial conditions"
from __future__ import print_function, division
import os
import numpy

from amuse.community.sse.interface import SSE
from amuse.ic.plummer import new_plummer_model
from amuse.ic.salpeter import new_salpeter_mass_distribution
from amuse.io import write_set_to_file
from amuse.units import units, nbody_system
from amuse.units.quantities import VectorQuantity


def create_ics():
    "Create ICs for a plummer sphere at different stellar ages"
    seed = 1
    numpy.random.seed(seed)
    number_of_particles = 40000
    mass_min = 0.1 | units.MSun
    mass_max = 100 | units.MSun
    metallicity = 0.002
    virial_ratio = 0.5
    stellar_masses = new_salpeter_mass_distribution(
        number_of_particles,
        mass_min=mass_min,
        mass_max=mass_max,
    )
    mass_scale = stellar_masses.sum()
    length_scale = 1 | units.parsec
    converter = nbody_system.nbody_to_si(mass_scale, length_scale)
    particles = new_plummer_model(number_of_particles, converter)
    particles.mass = stellar_masses
    # print(particles[0])
    # Make sure particles are in virial equilibrium
    particles.scale_to_standard(
        convert_nbody=converter,
        virial_ratio=virial_ratio,
    )

    stellar_evolution = SSE()
    stellar_evolution.parameters.metallicity = metallicity
    # print(stellar_evolution.parameters)
    timesteps = VectorQuantity(
        [3, 5, 10],
        units.Gyr,
    )
    stellar_evolution.particles.add_particles(particles)
    evo_to_model = stellar_evolution.particles.new_channel_to(
        particles,
    )
    # Copy only mass, not the other stellar parameters
    # evo_to_model.copy()
    evo_to_model.copy_attributes(["mass"])
    dataformats = {
        "amuse": "hdf5",
        "csv": "csv",
    }
    model_name = "Plummer_seed%i_N%i_salpeter" % (seed, number_of_particles)
    save_dir = "ICs/%s" % model_name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for dataformat, extension in dataformats.items():
        write_set_to_file(
            particles,
            "%s/zams.%s" % (save_dir, extension),
            dataformat,
        )
    for time in timesteps:
        stellar_evolution.evolve_model(time)
        evo_to_model.copy_attributes(["mass"])
        for dataformat, extension in dataformats.items():
            write_set_to_file(
                particles,
                "%s/%04.1fGyr.%s" % (
                    save_dir, time.value_in(units.Gyr), extension,
                ),
                dataformat,
            )

    stellar_evolution.stop()


if __name__ == "__main__":
    create_ics()
