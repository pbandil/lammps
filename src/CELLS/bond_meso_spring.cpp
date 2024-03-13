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

#include "bond_meso_spring.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "fix_bond_history.h"
#include "force.h"
#include "memory.h"
#include "modify.h"
#include "neighbor.h"

#define EPSILON 1e-10

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

BondMESOSpring::BondMESOSpring(LAMMPS *_lmp) :
    BondMESO(_lmp), Kstiff(nullptr)
{
  partial_flag = 1;
  single_extra = 1;
  svector = new double[1];
}

/* ---------------------------------------------------------------------- */

BondMESOSpring::~BondMESOSpring()
{
  delete[] svector;

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(Kstiff);
  }
}

/* ----------------------------------------------------------------------
  Store data for a single bond - if bond added after LAMMPS init (e.g. pour)
------------------------------------------------------------------------- */

double BondMESOSpring::store_bond(int n, int i, int j)
{
  double delx, dely, delz, r;
  double **x = atom->x;
  double **bondstore = fix_bond_history->bondstore;
  tagint *tag = atom->tag;

  delx = x[i][0] - x[j][0];
  dely = x[i][1] - x[j][1];
  delz = x[i][2] - x[j][2];

  // Bond stores initial length
  r = sqrt(delx * delx + dely * dely + delz * delz);
  bondstore[n][0] = r;

  if (i < atom->nlocal) {
    for (int m = 0; m < atom->num_bond[i]; m++) {
      if (atom->bond_atom[i][m] == tag[j]) { fix_bond_history->update_atom_value(i, m, 0, r); }
    }
  }

  if (j < atom->nlocal) {
    for (int m = 0; m < atom->num_bond[j]; m++) {
      if (atom->bond_atom[j][m] == tag[i]) { fix_bond_history->update_atom_value(j, m, 0, r); }
    }
  }

  return r;
}

/* ----------------------------------------------------------------------
  Store data for all bonds called once
------------------------------------------------------------------------- */

void BondMESOSpring::store_data()
{
  int i, j, m, type;
  double delx, dely, delz, r;
  double **x = atom->x;
  int **bond_type = atom->bond_type;

  for (i = 0; i < atom->nlocal; i++) {
    for (m = 0; m < atom->num_bond[i]; m++) {
      type = bond_type[i][m];

      //Skip if bond was turned off
      if (type < 0) continue;

      // map to find index n
      j = atom->map(atom->bond_atom[i][m]);
      if (j == -1) error->one(FLERR, "Atom missing in MESO bond");

      delx = x[i][0] - x[j][0];
      dely = x[i][1] - x[j][1];
      delz = x[i][2] - x[j][2];

      // Get closest image in case bonded with ghost
      domain->minimum_image(delx, dely, delz);
      r = sqrt(delx * delx + dely * dely + delz * delz);

      fix_bond_history->update_atom_value(i, m, 0, r);
    }
  }

  fix_bond_history->post_neighbor();
}

/* ---------------------------------------------------------------------- */

void BondMESOSpring::compute(int eflag, int vflag)
{

  if (!fix_bond_history->stored_flag) {
    fix_bond_history->stored_flag = true;
    store_data();
  }

  int i1, i2, itmp, n, type;
  double delx, dely, delz, delvx, delvy, delvz;
  double lam, rsq, r, r0, rinv, smooth, fbond, dot, k, N, ebond;
  double dr, rk;

  ebond = 0.0;
  ev_init(eflag, vflag);

  double **x = atom->x;
  double **f = atom->f;
  tagint *tag = atom->tag;
  int **bondlist = neighbor->bondlist;
  int nbondlist = neighbor->nbondlist;
  int nlocal = atom->nlocal;
  int newton_bond = force->newton_bond;

  double **bondstore = fix_bond_history->bondstore;

  for (n = 0; n < nbondlist; n++) {

    i1 = bondlist[n][0];
    i2 = bondlist[n][1];
    type = bondlist[n][2];
    r0 = bondstore[n][0];

    delx = x[i1][0] - x[i2][0];
    dely = x[i1][1] - x[i2][1];
    delz = x[i1][2] - x[i2][2];

    rsq = delx * delx + dely * dely + delz * delz;
    r   = sqrt(rsq);

    // If bond hasn't been set - should be initialized
    if (r0 < EPSILON || std::isnan(r0)) r0 = store_bond(n, i1, i2);

    dr = r - r0;
    rk = Kstiff[type] * dr;

    // force & energy

    if (r > 0.0)
      fbond = -2.0 * rk / r;
    else
      fbond = 0.0;

    if (eflag) ebond = rk * dr;

    // apply force to each of 2 atoms

    if (newton_bond || i1 < nlocal) {
      f[i1][0] += delx * fbond;
      f[i1][1] += dely * fbond;
      f[i1][2] += delz * fbond;
    }

    if (newton_bond || i2 < nlocal) {
      f[i2][0] -= delx * fbond;
      f[i2][1] -= dely * fbond;
      f[i2][2] -= delz * fbond;
    }

    if (evflag) ev_tally(i1, i2, nlocal, newton_bond, ebond, fbond, delx, dely, delz);
  }
}

/* ---------------------------------------------------------------------- */

void BondMESOSpring::allocate()
{
  allocated = 1;
  const int np1 = atom->nbondtypes + 1;

  memory->create(Kstiff, np1, "bond:Kstiff");
  memory->create(setflag, np1, "bond:setflag");
  for (int i = 1; i < np1; i++) setflag[i] = 0;
}

/* ----------------------------------------------------------------------
   set coeffs for one or more types
------------------------------------------------------------------------- */

void BondMESOSpring::coeff(int narg, char **arg)
{
  if (narg != 2) error->all(FLERR, "Incorrect args for bond coefficients");
  if (!allocated) allocate();

  int ilo, ihi;
  utils::bounds(FLERR, arg[0], 1, atom->nbondtypes, ilo, ihi, error);

  double Kstiff_one = utils::numeric(FLERR, arg[1], false, lmp);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    Kstiff[i] = Kstiff_one;
    setflag[i] = 1;
    count++;
  }

  if (count == 0) error->all(FLERR, "Incorrect args for bond coefficients");
}

/* ----------------------------------------------------------------------
   check for correct settings and create fix
------------------------------------------------------------------------- */

void BondMESOSpring::init_style()
{
  BondMESO::init_style();

  if (!id_fix_bond_history) {
    id_fix_bond_history = utils::strdup("HISTORY_MESO_SPRING");
    fix_bond_history = dynamic_cast<FixBondHistory *>(modify->replace_fix(
        id_fix_dummy2, fmt::format("{} all BOND_HISTORY 0 1", id_fix_bond_history), 1));
    delete[] id_fix_dummy2;
    id_fix_dummy2 = nullptr;
  }
}

/* ---------------------------------------------------------------------- */

// void BondMESOSpring::settings(int narg, char **arg)
// {
//   BondMESO::settings(narg, arg);

//   int iarg;
//   for (std::size_t i = 0; i < leftover_iarg.size(); i++) {
//     iarg = leftover_iarg[i];
//     if (strcmp(arg[iarg], "smooth") == 0) {
//       if (iarg + 1 > narg) error->all(FLERR, "Illegal bond meso command, missing option for smooth");
//       smooth_flag = utils::logical(FLERR, arg[iarg + 1], false, lmp);
//       i += 1;
//     } else {
//       error->all(FLERR, "Illegal bond meso command, invalid argument {}", arg[iarg]);
//     }
//   }
// }

/* ----------------------------------------------------------------------
   proc 0 writes out coeffs to restart file
------------------------------------------------------------------------- */

void BondMESOSpring::write_restart(FILE *fp)
{
  BondMESO::write_restart(fp);
  write_restart_settings(fp);

  fwrite(&Kstiff[1], sizeof(double), atom->nbondtypes, fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads coeffs from restart file, bcasts them
------------------------------------------------------------------------- */

void BondMESOSpring::read_restart(FILE *fp)
{
  BondMESO::read_restart(fp);
  read_restart_settings(fp);
  allocate();

  if (comm->me == 0) {
    utils::sfread(FLERR, &Kstiff[1], sizeof(double), atom->nbondtypes, fp, nullptr, error);
  }
  MPI_Bcast(&Kstiff[1], atom->nbondtypes, MPI_DOUBLE, 0, world);

  for (int i = 1; i <= atom->nbondtypes; i++) setflag[i] = 1;
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
 ------------------------------------------------------------------------- */

// void BondMESOSpring::write_restart_settings(FILE *fp)
// {
//   fwrite(&Kstiff[1], sizeof(double), atom->nbondtypes, fp);
// }

/* ----------------------------------------------------------------------
    proc 0 reads from restart file, bcasts
 ------------------------------------------------------------------------- */

// void BondMESOSpring::read_restart_settings(FILE *fp)
// {
//   if (comm->me == 0)
//     utils::sfread(FLERR, &smooth_flag, sizeof(int), 1, fp, nullptr, error);
//   MPI_Bcast(&smooth_flag, 1, MPI_INT, 0, world);
// }

/* ---------------------------------------------------------------------- */

double BondMESOSpring::single(int type, double rsq, int i, int j, double &fforce)
{

  double r0;
  for (int n = 0; n < atom->num_bond[i]; n++) {
    if (atom->bond_atom[i][n] == atom->tag[j]) r0 = fix_bond_history->get_atom_value(i, n, 0);
  }

  double r = sqrt(rsq);
  double dr = r - r0;
  double rk = Kstiff[type] * dr;
  fforce = 0;
  if (r > 0.0) fforce = -2.0 * rk / r;
  return rk * dr;

  svector[0] = r0;
  // fforce = -Kstiff[type]*sqrt(rsq);
  // return Kstiff[type]*rsq*0.5;
}
