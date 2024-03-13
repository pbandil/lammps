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

#include "bond_meso_polymer.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "fix_bond_history.h"
#include "force.h"
#include "memory.h"
#include "modify.h"
#include "neighbor.h"
#include "random_mars.h"

#define EPSILON 1e-10

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

BondMESOPolymer::BondMESOPolymer(LAMMPS *_lmp) :
    BondMESO(_lmp), b(nullptr), Nmean(nullptr), Nmin(nullptr), Nmax(nullptr), lamcrit(nullptr), gamma(nullptr)
{
  partial_flag = 1;
  smooth_flag = 1;

  single_extra = 1;
  svector = new double[1];

  // initialize Marsaglia RNG with processor-unique seed
  MPI_Comm_rank(world,&me);
  int seed = 12345;
  random = new RanMars(_lmp,seed + me);
}

/* ---------------------------------------------------------------------- */

BondMESOPolymer::~BondMESOPolymer()
{
  delete random;
  delete[] svector;

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(b);
    memory->destroy(Nmean);
    memory->destroy(Nmin);
    memory->destroy(Nmax);
    memory->destroy(lamcrit);
    memory->destroy(gamma);
  }
}

/* ----------------------------------------------------------------------
  Store data for a single bond - if bond added after LAMMPS init (e.g. pour)
------------------------------------------------------------------------- */

double BondMESOPolymer::store_bond(int n, int i, int j)
{
  int ii, jj, m, N, type;
  double **bondstore = fix_bond_history->bondstore;
  double Wsum,rnum;
  tagint *tag = atom->tag;

  // Determine bond type
  for (int m = 0; m < atom->num_bond[i]; m++) {
    if (atom->bond_atom[i][m] == tag[j]) { type = atom->bond_type[i][m]; }
  }

  // Initialize lengths and weights
  int NN;
  double WW;
  Wsum = 0.0;
  for (ii = 0; ii < (Nmax[type]-Nmin[type]); ii++) {
    NN = ii+Nmin[type];
    WW = pow(1.0+1.0/(Nmean[type]-Nmin[type]),Nmin[type]-NN)/(Nmean[type]-Nmin[type]+1);
    Wsum += WW;
  }

  // Determine bond length based on weights
  rnum = random->uniform();
  for (ii = 0; ii < (Nmax[type]-Nmin[type]); ii++) {
    NN = ii+Nmin[type];
    WW = pow(1.0+1.0/(Nmean[type]-Nmin[type]),Nmin[type]-NN)/(Nmean[type]-Nmin[type]+1);
    WW /= Wsum;
    if (rnum < WW) break;
    rnum -= WW;
  }

  N = NN;
  bondstore[n][0] = N;

  if (i < atom->nlocal) {
    for (int m = 0; m < atom->num_bond[i]; m++) {
      if (atom->bond_atom[i][m] == tag[j]) { fix_bond_history->update_atom_value(i, m, 0, N); }
    }
  }

  if (j < atom->nlocal) {
    for (int m = 0; m < atom->num_bond[j]; m++) {
      if (atom->bond_atom[j][m] == tag[i]) { fix_bond_history->update_atom_value(j, m, 0, N); }
    }
  }

  return N;
}

/* ----------------------------------------------------------------------
  Store data for all bonds called once
------------------------------------------------------------------------- */

void BondMESOPolymer::store_data()
{
  double Wsum, rnum, WW;
  int ii, jj, i, j, m, type, N, NN;
  int **bond_type = atom->bond_type;

  for (ii = 0; ii < atom->nlocal; ii++) {
    for (m = 0; m < atom->num_bond[ii]; m++) {
      type = bond_type[ii][m];

      //Skip if bond was turned off
      if (type < 0) continue;

      // map to find index n
      jj = atom->map(atom->bond_atom[ii][m]);
      if (jj == -1) error->one(FLERR, "Atom missing in MESO bond");

      // Initialize lengths and weights
      Wsum = 0.0;
      for (ii = 0; ii < (Nmax[type]-Nmin[type]); ii++) {
        NN = ii+Nmin[type];
        WW = pow(1.0+1.0/(Nmean[type]-Nmin[type]),Nmin[type]-NN)/(Nmean[type]-Nmin[type]+1);
        Wsum += WW;
      }

      // Determine bond length based on weights
      rnum = random->uniform();
      for (ii = 0; ii < (Nmax[type]-Nmin[type]); ii++) {
        NN = ii+Nmin[type];
        WW = pow(1.0+1.0/(Nmean[type]-Nmin[type]),Nmin[type]-NN)/(Nmean[type]-Nmin[type]+1);
        WW /= Wsum;
        if (rnum < WW) break;
        rnum -= WW;
      }

      N = NN;

      fix_bond_history->update_atom_value(ii, m, 0, N);
    }
  }

  fix_bond_history->post_neighbor();
}

/* ---------------------------------------------------------------------- */

void BondMESOPolymer::compute(int eflag, int vflag)
{

  if (!fix_bond_history->stored_flag) {
    fix_bond_history->stored_flag = true;
    store_data();
  }

  int i1, i2, itmp, n, type;
  double delx, dely, delz, delvx, delvy, delvz;
  double lam, rsq, r, r0, rinv, smooth, fbond, dot, k, N, ebond;

  ev_init(eflag, vflag);

  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  tagint *tag = atom->tag;
  int **bondlist = neighbor->bondlist;
  int nbondlist = neighbor->nbondlist;
  int nlocal = atom->nlocal;
  int newton_bond = force->newton_bond;

  double **bondstore = fix_bond_history->bondstore;

  for (n = 0; n < nbondlist; n++) {

    // skip bond if already broken
    if (bondlist[n][2] <= 0) continue;

    i1 = bondlist[n][0];
    i2 = bondlist[n][1];
    type = bondlist[n][2];
    N = bondstore[n][0];

    delx = x[i1][0] - x[i2][0];
    dely = x[i1][1] - x[i2][1];
    delz = x[i1][2] - x[i2][2];

    rsq = delx * delx + dely * dely + delz * delz;
    r   = sqrt(rsq);

    // If bond hasn't been set - should be initialized
    if (N < EPSILON || std::isnan(N)) N = store_bond(n, i1, i2);

    k  = 3.0/N/b[type]/b[type];
    lam = r/N/b[type];

    // Ensure pair is always ordered to ensure numerical operations
    // are identical to minimize the possibility that a bond straddling
    // an mpi grid (newton off) doesn't break on one proc but not the other
    if (tag[i2] < tag[i1]) {
      itmp = i1;
      i1 = i2;
      i2 = itmp;
    }

    // if (lam > lamcrit[type]) {
    //   bondlist[n][2] = 0;
    //   process_broken(i1, i2);
    //   continue;
    // }

    rinv = 1.0 / r;

    if (r > 0.0)
      fbond = -k*r;
    else
      fbond = 0.0;

    if (eflag) ebond = k*r*r*0.5;

    delvx = v[i1][0] - v[i2][0];
    delvy = v[i1][1] - v[i2][1];
    delvz = v[i1][2] - v[i2][2];
    dot = delx * delvx + dely * delvy + delz * delvz;
    // fbond -= gamma[type] * dot * rinv;
    // fbond *= rinv;

    // if (smooth_flag) {
    //   smooth = (lam) / (lamcrit[type]);
    //   smooth *= smooth;
    //   smooth *= smooth;
    //   smooth *= smooth;
    //   smooth = 1 - smooth;
    //   fbond *= smooth;
    // }

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

void BondMESOPolymer::allocate()
{
  allocated = 1;
  const int np1 = atom->nbondtypes + 1;

  memory->create(b, np1, "bond:b");
  memory->create(lamcrit, np1, "bond:lamcrit");
  memory->create(gamma, np1, "bond:gamma");
  memory->create(Nmean, np1, "bond:Nmean");
  memory->create(Nmin, np1, "bond:Nmin");
  memory->create(Nmax, np1, "bond:Nmax");

  memory->create(setflag, np1, "bond:setflag");
  for (int i = 1; i < np1; i++) setflag[i] = 0;
}

/* ----------------------------------------------------------------------
   set coeffs for one or more types
------------------------------------------------------------------------- */

void BondMESOPolymer::coeff(int narg, char **arg)
{
  if (narg != 7) error->all(FLERR, "Incorrect args for bond coefficients");
  if (!allocated) allocate();

  int ilo, ihi;
  utils::bounds(FLERR, arg[0], 1, atom->nbondtypes, ilo, ihi, error);

  double b_one = utils::numeric(FLERR, arg[1], false, lmp);
  int Nmean_one = utils::inumeric(FLERR, arg[2], false, lmp);
  int Nmin_one = utils::inumeric(FLERR, arg[3], false, lmp);
  int Nmax_one = utils::inumeric(FLERR, arg[4], false, lmp);
  double lamcrit_one = utils::numeric(FLERR, arg[5], false, lmp);
  double gamma_one = utils::numeric(FLERR, arg[6], false, lmp);

  double max_stretch = 0;
  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    b[i] = b_one;
    Nmean[i] = Nmean_one;
    Nmin[i] = Nmin_one;
    Nmax[i] = Nmax_one;
    lamcrit[i] = lamcrit_one;
    gamma[i] = gamma_one;
    setflag[i] = 1;
    count++;

    if (lamcrit[i] > max_stretch) max_stretch = lamcrit[i];
  }

  if (count == 0) error->all(FLERR, "Incorrect args for bond coefficients");
}

/* ----------------------------------------------------------------------
   check for correct settings and create fix
------------------------------------------------------------------------- */

void BondMESOPolymer::init_style()
{
  BondMESO::init_style();

  if (comm->ghost_velocity == 0)
    error->all(FLERR, "Bond meso/polymer requires ghost atoms store velocity");

  if (!id_fix_bond_history) {
    id_fix_bond_history = utils::strdup("HISTORY_MESO_POLYMER");
    fix_bond_history = dynamic_cast<FixBondHistory *>(modify->replace_fix(
        id_fix_dummy2, fmt::format("{} all BOND_HISTORY 0 1", id_fix_bond_history), 1));
    delete[] id_fix_dummy2;
    id_fix_dummy2 = nullptr;
  }
}

/* ---------------------------------------------------------------------- */

void BondMESOPolymer::settings(int narg, char **arg)
{
  BondMESO::settings(narg, arg);

  int iarg;
  for (std::size_t i = 0; i < leftover_iarg.size(); i++) {
    iarg = leftover_iarg[i];
    if (strcmp(arg[iarg], "smooth") == 0) {
      if (iarg + 1 > narg) error->all(FLERR, "Illegal bond meso command, missing option for smooth");
      smooth_flag = utils::logical(FLERR, arg[iarg + 1], false, lmp);
      i += 1;
    } else {
      error->all(FLERR, "Illegal bond meso command, invalid argument {}", arg[iarg]);
    }
  }
}

/* ----------------------------------------------------------------------
   proc 0 writes out coeffs to restart file
------------------------------------------------------------------------- */

void BondMESOPolymer::write_restart(FILE *fp)
{
  BondMESO::write_restart(fp);
  write_restart_settings(fp);

  fwrite(&b[1], sizeof(double), atom->nbondtypes, fp);
  fwrite(&Nmean[1], sizeof(int), atom->nbondtypes, fp);
  fwrite(&Nmin[1], sizeof(int), atom->nbondtypes, fp);
  fwrite(&Nmax[1], sizeof(int), atom->nbondtypes, fp);
  fwrite(&lamcrit[1], sizeof(double), atom->nbondtypes, fp);
  fwrite(&gamma[1], sizeof(double), atom->nbondtypes, fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads coeffs from restart file, bcasts them
------------------------------------------------------------------------- */

void BondMESOPolymer::read_restart(FILE *fp)
{
  BondMESO::read_restart(fp);
  read_restart_settings(fp);
  allocate();

  if (comm->me == 0) {
    utils::sfread(FLERR, &b[1], sizeof(double), atom->nbondtypes, fp, nullptr, error);
    utils::sfread(FLERR, &Nmean[1], sizeof(int), atom->nbondtypes, fp, nullptr, error);
    utils::sfread(FLERR, &Nmin[1], sizeof(int), atom->nbondtypes, fp, nullptr, error);
    utils::sfread(FLERR, &Nmax[1], sizeof(int), atom->nbondtypes, fp, nullptr, error);
    utils::sfread(FLERR, &lamcrit[1], sizeof(double), atom->nbondtypes, fp, nullptr, error);
    utils::sfread(FLERR, &gamma[1], sizeof(double), atom->nbondtypes, fp, nullptr, error);
  }
  MPI_Bcast(&b[1], atom->nbondtypes, MPI_DOUBLE, 0, world);
  MPI_Bcast(&Nmean[1], atom->nbondtypes, MPI_INT, 0, world);
  MPI_Bcast(&Nmin[1], atom->nbondtypes, MPI_INT, 0, world);
  MPI_Bcast(&Nmax[1], atom->nbondtypes, MPI_INT, 0, world);
  MPI_Bcast(&lamcrit[1], atom->nbondtypes, MPI_DOUBLE, 0, world);
  MPI_Bcast(&gamma[1], atom->nbondtypes, MPI_DOUBLE, 0, world);

  for (int i = 1; i <= atom->nbondtypes; i++) setflag[i] = 1;
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
 ------------------------------------------------------------------------- */

void BondMESOPolymer::write_restart_settings(FILE *fp)
{
  fwrite(&smooth_flag, sizeof(int), 1, fp);
}

/* ----------------------------------------------------------------------
    proc 0 reads from restart file, bcasts
 ------------------------------------------------------------------------- */

void BondMESOPolymer::read_restart_settings(FILE *fp)
{
  if (comm->me == 0)
    utils::sfread(FLERR, &smooth_flag, sizeof(int), 1, fp, nullptr, error);
  MPI_Bcast(&smooth_flag, 1, MPI_INT, 0, world);
}

/* ---------------------------------------------------------------------- */

double BondMESOPolymer::single(int type, double rsq, int i, int j, double &fforce)
{
  if (type <= 0) return 0.0;

  double N;
  for (int n = 0; n < atom->num_bond[i]; n++) {
    if (atom->bond_atom[i][n] == atom->tag[j]) N = fix_bond_history->get_atom_value(i, n, 0);
  }

  double r = sqrt(rsq);
  double rinv = 1.0 / r;

  double lam = r/N/b[type];
  double k   = 3.0/N/b[type]/b[type];

  // Force computation
  fforce = 0.0;
  if (r > 0.0) fforce = -k*r;

  double **x = atom->x;
  double **v = atom->v;
  double delx = x[i][0] - x[j][0];
  double dely = x[i][1] - x[j][1];
  double delz = x[i][2] - x[j][2];
  double delvx = v[i][0] - v[j][0];
  double delvy = v[i][1] - v[j][1];
  double delvz = v[i][2] - v[j][2];
  double dot = delx * delvx + dely * delvy + delz * delvz;
  // fforce -= gamma[type] * dot * rinv;
  // fforce *= rinv;

  // if (smooth_flag) {
  //   double smooth = (lam) / (lamcrit[type]);
  //   smooth *= smooth;
  //   smooth *= smooth;
  //   smooth *= smooth;
  //   smooth = 1 - smooth;
  //   fforce *= smooth;
  // }

  // set single_extra quantities

  svector[0] = N;

  return k*r*r*0.5;
}
