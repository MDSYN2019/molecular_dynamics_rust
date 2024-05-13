//! ----------------------
//! Author: Sang Young Noh
//! ----------------------
//!
//! ------------------------
//! Last Updated: 26/04/2024
//! ------------------------
//!

/*

The HF-self_consistent_field is the standard first-principles
approach for computing approximate quantum mechanical eigenstates
of interacting fermion systems.

Such systems include electrons in atoms, molecules, and condensed matter. Protons
and neutrons in nuclei, and nuclear matter.
*/

#![allow(unused_variables)] // ensure that unused variables do not cause an error when compiling this program
                            // relax compiler warnings while working through ideas

use sang_md::lennard_jones_simulations;

fn main() {
    let mut lj_params_new = lennard_jones_simulations::LJParameters {
        n: 3,
        i: 0,
        j: 1,
        eps: 1.0,
        sigma: 4.0,
        sigma_sq: 0.0,
        pot: 0.0,
        rij_sq: 0.0,
        sr2: 0.0,
        sr6: 0.0,
        sr12: 0.0,
        epslj: 0.0,
        nsteps: 100,
        na: 2,
    };

    // Running a sample simulation - first generate the velocities and positions of the atoms
    let mut new_simulation_md =
        lennard_jones_simulations::create_atoms_with_set_positions_and_velocities(
            3, 10.0, 10.0, 10.0,
        );

    // running verlet update on the particles
    lennard_jones_simulations::run_verlet_update(
        new_simulation_md.expect("REASON"),
        (0.01, 0.01, 0.01),
        0.01,
    );
}
