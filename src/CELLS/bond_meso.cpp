/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "bond_meso.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "fix_bond_history.h"
#include "fix_store_local.h"
#include "force.h"
#include "memory.h"
#include "modify.h"
#include "update.h"

#include <vector>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

BondMESO::BondMESO(LAMMPS *_lmp) :
    Bond(_lmp), id_fix_dummy(nullptr), id_fix_dummy2(nullptr), id_fix_update(nullptr),
    id_fix_bond_history(nullptr), id_fix_store_local(nullptr), id_fix_prop_atom(nullptr),
    fix_store_local(nullptr), fix_bond_history(nullptr), fix_update_special_bonds(nullptr),
    pack_choice(nullptr), output_data(nullptr)
{
  overlay_flag = 0;
  prop_atom_flag = 0;
  nvalues = 0;

  r0_max_estimate = 0.0;

  // create dummy fix as placeholder for FixUpdateSpecialBonds & BondHistory
  // this is so final order of Modify:fix will conform to input script
  // BondHistory technically only needs this if updateflag = 1

  id_fix_dummy = utils::strdup("MESO_DUMMY");
  modify->add_fix(fmt::format("{} all DUMMY ", id_fix_dummy));

  id_fix_dummy2 = utils::strdup("MESO_DUMMY2");
  modify->add_fix(fmt::format("{} all DUMMY ", id_fix_dummy2));
}

/* ---------------------------------------------------------------------- */

BondMESO::~BondMESO()
{
  delete[] pack_choice;

  if (id_fix_dummy) modify->delete_fix(id_fix_dummy);
  if (id_fix_dummy2) modify->delete_fix(id_fix_dummy2);
  if (id_fix_update) modify->delete_fix(id_fix_update);
  if (id_fix_bond_history) modify->delete_fix(id_fix_bond_history);
  if (id_fix_store_local) modify->delete_fix(id_fix_store_local);
  if (id_fix_prop_atom) modify->delete_fix(id_fix_prop_atom);

  delete[] id_fix_dummy;
  delete[] id_fix_dummy2;
  delete[] id_fix_update;
  delete[] id_fix_bond_history;
  delete[] id_fix_store_local;
  delete[] id_fix_prop_atom;

  memory->destroy(output_data);
}

/* ---------------------------------------------------------------------- */

void BondMESO::init_style()
{
  if (id_fix_store_local) {
    auto ifix = modify->get_fix_by_id(id_fix_store_local);
    if (!ifix) error->all(FLERR, "Cannot find fix STORE/LOCAL id {}", id_fix_store_local);
    if (strcmp(ifix->style, "STORE/LOCAL") != 0)
      error->all(FLERR, "Incorrect fix style matched, not STORE/LOCAL: {}", ifix->style);
    fix_store_local = dynamic_cast<FixStoreLocal *>(ifix);
    fix_store_local->nvalues = nvalues;
  }

  if (force->special_lj[1] != 1.0)
    error->all(FLERR,
                "MESO bond styles require special_bonds weight of 1.0 for "
                "first neighbors");
  if (id_fix_update) {
    modify->delete_fix(id_fix_update);
    delete[] id_fix_update;
    id_fix_update = nullptr;
  }

  if (force->angle || force->dihedral || force->improper)
    error->all(FLERR, "Bond style meso cannot be used with 3,4-body interactions");
  if (atom->molecular == 2)
    error->all(FLERR, "Bond style meso cannot be used with atom style template");

  // special 1-3 and 1-4 weights must be 1 to prevent building 1-3 and 1-4 special bond lists
  if (force->special_lj[2] != 1.0 || force->special_lj[3] != 1.0 || force->special_coul[2] != 1.0 ||
      force->special_coul[3] != 1.0)
    error->all(FLERR, "Bond style meso requires 1-3 and 1-4 special weights of 1.0");
}

/* ----------------------------------------------------------------------
   global settings
   All args before store/local command are saved for potential args
     for specific bond MESO substyles
   All args after optional stode/local command are variables stored
     in the compute store/local
------------------------------------------------------------------------- */

// void BondMESO::settings(int narg, char **arg)
// {
//   leftover_iarg.clear();

// }

/* ----------------------------------------------------------------------
   used to check bond communiction cutoff - not perfect, estimates based on local-local only
------------------------------------------------------------------------- */

double BondMESO::equilibrium_distance(int /*i*/)
{
  // Ghost atoms may not yet be communicated, this may only be an estimate
  if (r0_max_estimate == 0) {
    if (!fix_bond_history->restart_reset) {
      int type, j;
      double delx, dely, delz, r;
      double **x = atom->x;
      for (int i = 0; i < atom->nlocal; i++) {
        for (int m = 0; m < atom->num_bond[i]; m++) {
          type = atom->bond_type[i][m];
          if (type == 0) continue;

          j = atom->map(atom->bond_atom[i][m]);
          if (j == -1) continue;

          delx = x[i][0] - x[j][0];
          dely = x[i][1] - x[j][1];
          delz = x[i][2] - x[j][2];
          domain->minimum_image(delx, dely, delz);

          r = sqrt(delx * delx + dely * dely + delz * delz);
          if (r > r0_max_estimate) r0_max_estimate = r;
        }
      }
    } else {
      int type, j;
      double r;
      for (int i = 0; i < atom->nlocal; i++) {
        for (int m = 0; m < atom->num_bond[i]; m++) {
          type = atom->bond_type[i][m];
          if (type == 0) continue;

          j = atom->map(atom->bond_atom[i][m]);
          if (j == -1) continue;

          // First value must always be reference length
          r = fix_bond_history->get_atom_value(i, m, 0);
          if (r > r0_max_estimate) r0_max_estimate = r;
        }
      }
    }

    double temp;
    MPI_Allreduce(&r0_max_estimate, &temp, 1, MPI_DOUBLE, MPI_MAX, world);
    r0_max_estimate = temp;
  }

  // Return maximum r0 value
  return r0_max_estimate;
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
 ------------------------------------------------------------------------- */

void BondMESO::write_restart(FILE *fp)
{
  fwrite(&overlay_flag, sizeof(int), 1, fp);
}

/* ----------------------------------------------------------------------
    proc 0 reads from restart file, bcasts
 ------------------------------------------------------------------------- */

void BondMESO::read_restart(FILE *fp)
{
  if (comm->me == 0)
    utils::sfread(FLERR, &overlay_flag, sizeof(int), 1, fp, nullptr, error);
  MPI_Bcast(&overlay_flag, 1, MPI_INT, 0, world);
}