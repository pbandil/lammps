// clang-format off
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

#include "fix_bond_rupturess.h"

#include "atom.h"
#include "bond.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "memory.h"
#include "modify.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "pair.h"
#include "respa.h"
#include "update.h"

#include <cstring>
#include "math_const.h"

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixBondRupturess::FixBondRupturess(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg < 7) error->all(FLERR,"Illegal fix bond/rupturess command");

  MPI_Comm_rank(world,&me);
  MPI_Comm_size(world,&nprocs);

  nevery = utils::inumeric(FLERR,arg[3],false,lmp);
  if (nevery <= 0) error->all(FLERR,"Illegal fix bond/rupturess command");

  dynamic_group_allow = 1;
  force_reneighbor = 1;
  next_reneighbor = -1;
  vector_flag = 1;
  size_vector = 2;
  global_freq = 1;
  extvector = 0;

  iatomtype = utils::inumeric(FLERR,arg[4],false,lmp);
  btype = utils::inumeric(FLERR,arg[5],false,lmp);
  r_critical = utils::numeric(FLERR,arg[6],false,lmp);

  if (btype < 1 || btype > atom->nbondtypes)
    error->all(FLERR,"Invalid bond type in fix bond/rupturess command");
  if (r_critical < 0.0) error->all(FLERR,"Illegal fix bond/rupturess command");
  r2_critical = r_critical*r_critical;

  jatomtype = iatomtype;

  // optional keywords

  flag_mol = 0;
  flag_skip = 0;
  flag_fcrit = 0;
  f_critical = 0;
  skip = 0;
  maxbond = atom->bond_per_atom;
  int iarg = 7;

  while (iarg < narg) {
    if (strcmp(arg[iarg],"maxbond") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix bond/rupturess command");
      maxbond = utils::inumeric(FLERR,arg[iarg+1],false,lmp);
      if (maxbond < 0) error->all(FLERR,"Illegal fix bond/rupturess command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"mol") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix bond/rupturess command");
      flag_mol = utils::inumeric(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg],"skip") == 0) {
      if (iarg+3 > narg) error->all(FLERR,"Illegal fix bond/dynamic command");
      flag_skip = utils::inumeric(FLERR,arg[iarg+1],false,lmp);
      skip = utils::inumeric(FLERR,arg[iarg+2],false,lmp);
      iarg += 3;
        } else if (strcmp(arg[iarg],"fcrit") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix bond/dynamic command");
      flag_fcrit = 1;
      f_critical = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    } else error->all(FLERR,"Illegal fix bond/rupturess command");
  }

  // error checks

  if (atom->bond_per_atom < maxbond)
    error->all(FLERR,"Maxbond too large in fix bond/rupturess - increase bonds/per/atom");

  // allocate values local to this fix
  nmax = 0;
  countflag = 0;

  // set comm sizes needed by this fix
  // forward is big due to comm of broken bonds and 1-2 neighbors

  size_bonds_and_neighbors = MAX(2,1+atom->maxspecial);
  size_bond_lists = MAX(2,1+2*(atom->bond_per_atom));

  comm_forward = size_bonds_and_neighbors;
}

/* ---------------------------------------------------------------------- */

FixBondRupturess::~FixBondRupturess()
{

  // delete locally stored arrays
  if (new_fix_id && modify->nfix) modify->delete_fix(new_fix_id);
  delete [] new_fix_id;

}

/* ---------------------------------------------------------------------- */

int FixBondRupturess::setmask()
{
  int mask = 0;
  mask |= POST_INTEGRATE;
  mask |= POST_INTEGRATE_RESPA;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixBondRupturess::post_constructor()
{
  new_fix_id = utils::strdup(id + std::string("_FIX_PA"));
  modify->add_fix(fmt::format("{} {} property/atom i2_fbd_{} {} ghost yes",new_fix_id, group->names[igroup],id,std::to_string(maxbond)));

  int tmp1, tmp2;
  index = atom->find_custom(utils::strdup(std::string("fbd_")+id),tmp1,tmp2);

  nmax = atom->nmax;

  tagint **fbd = atom->iarray[index];
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int nall = nlocal + nghost;

  for (int i = 0; i < nall; i++) {
    for (int m = 0; m < maxbond; m++) {
      if (mask[i] & groupbit) {
        fbd[i][m] = 0;
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixBondRupturess::init() //POSSIBLY ADD NEWTON_BOND FLAG!!!!!
{
  if (utils::strmatch(update->integrate_style,"^respa"))
    nlevels_respa = ((Respa *) update->integrate)->nlevels;

  // check cutoff for iatomtype,jatomtype

  if (force->pair == nullptr || r2_critical > force->pair->cutsq[iatomtype][jatomtype])
    error->all(FLERR,"Fix bond/rupturess cutoff is longer than pairwise cutoff");

  // need a half neighbor list, built every Nevery steps

  neighbor->add_request(this, NeighConst::REQ_OCCASIONAL);
}

/* ---------------------------------------------------------------------- */

void FixBondRupturess::setup(int /*vflag*/)
{
  int i,j,m;

  // compute initial bond neighbors if this is first run
  // can't do this earlier, in constructor or init, b/c need ghost info

  if (countflag) return;
  countflag = 1;

  int tmp1, tmp2;
  index = atom->find_custom(utils::strdup(std::string("fbd_")+id),tmp1,tmp2);
  tagint **fbd = atom->iarray[index];

  int *num_bond = atom->num_bond;
  int **bond_type = atom->bond_type;
  tagint **bond_atom = atom->bond_atom;
  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int nall = nlocal + nghost;

  for (i = 0; i < nlocal; i++) {
    if (num_bond[i] == 0) continue;
    for (j = 0; j < num_bond[i]; j++) {
      if (bond_type[i][j] == btype) {
        fbd[i][j] = bond_atom[i][j]; // ADD OPTION FOR INCLUDING OUTSIDE ATOMS OR NOT DIRECTLY
      }
    }
  }

  // forward communication of fbd so ghost atoms have it
  commflag = 1;
  comm->forward_comm(this,maxbond);

}

/* ---------------------------------------------------------------------- */

void FixBondRupturess::post_integrate()
{
  int i,j,b,bb,n,k,ii;
  double xtmp,ytmp,ztmp,delx,dely,delz,rsq;
  tagint *slist, tagj, tagi;

  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int nall = nlocal + nghost;

  int tmp1, tmp2;
  index = atom->find_custom(utils::strdup(std::string("fbd_")+id),tmp1,tmp2);
  tagint **fbd = atom->iarray[index];

  int *num_bond = atom->num_bond;
  int **bond_type = atom->bond_type;
  tagint **bond_atom = atom->bond_atom;

  if ((update->ntimestep+1) % nevery) return;
  if (flag_skip) {
    if (!(update->ntimestep % skip)) return;
  }

  // acquire updated ghost atom positions
  // necessary b/c are calling this after integrate, but before Verlet comm

  comm->forward_comm();

  // basic atom information

  double **x = atom->x;
  tagint *tag = atom->tag;
  int *mask = atom->mask;
  int *type = atom->type;
  Bond *bond = force->bond;

  double fbond,engpot;

  /* BEGIN BREAKING PROCESS */
  // loop over local atoms
  // check for possible breaks

  for (i = 0; i < nlocal; i++) {

    if (!(mask[i] & groupbit)) continue;

    // Store coordinates of this atom //
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];

    for (b = 0; b < maxbond; b++) {
      tagj = fbd[i][b];
      
      if (tagj < 1) continue;

      j = atom->map(tagj);
      if (j < 0)
        error->one(FLERR,"Fix bond/rupturess needs ghost atoms "
                    "from further away");

      if (!(mask[j] & groupbit)) continue;
    //   if (!(type[j] == jatomtype)) continue;

        delx = xtmp - x[j][0];
        dely = ytmp - x[j][1];
        delz = ztmp - x[j][2];
        domain->minimum_image(delx, dely, delz);
        rsq = delx*delx + dely*dely + delz*delz;

        if (flag_fcrit) {
          engpot = bond->single(btype,rsq,i,j,fbond);
          if (abs(fbond) >= f_critical) {
            fbd[i][b] = -tagj;

            // find the entry of atom j and update its fbd as well
            // if j is a ghost atom, it will do this on its own processor
            for (bb = 0; bb < maxbond; bb++) {
                if (fbd[j][bb] == tag[i]) {
                    fbd[j][bb] = -tag[i];
                    break;
                }
            }
          }
          continue;
        }
        
        if (rsq >= r2_critical) {
            fbd[i][b] = -tagj;

            // find the entry of atom j and update its fbd as well
            // if j is a ghost atom, it will do this on its own processor
            for (bb = 0; bb < maxbond; bb++) {
                if (fbd[j][bb] == tag[i]) {
                    fbd[j][bb] = -tag[i];
                    break;
                }
            }
        }
    }
  }

  // forward communication of fbd so ghost atoms store their breaks
  commflag = 1;
  comm->forward_comm(this,maxbond);

  // Process breaking events
  for (i = 0; i < nlocal; i++) {
    for (b = 0; b < maxbond; b++) {

      // First, process broken bonds
      // check for negative fbd entry
      if (fbd[i][b] < 0) {
        tagj = -fbd[i][b];
        j = atom->map(tagj);

        if (j < 0)
          error->one(FLERR,"Fix bond/rupturess needs ghost atoms "
                      "from further away");

        // Update atom properties and fbd
        process_broken(i,j);
        fbd[i][b] = 0;
      }
    }
  }

  // forward communication of fbd
  commflag = 1;
  comm->forward_comm(this,maxbond);

  // forward communication of atom properties
  commflag = 5;
  comm->forward_comm(this,size_bond_lists);

  // trigger reneighboring
  next_reneighbor = update->ntimestep;
}

/* ---------------------------------------------------------------------- */

void FixBondRupturess::process_broken(int i, int j)
{

  // Manually search and remove from atom arrays
  // need to remove in case special bonds arrays rebuilt
  int m, n, k;
  int nlocal = atom->nlocal;

  tagint *tag = atom->tag;
  tagint **bond_atom = atom->bond_atom;
  int **bond_type = atom->bond_type;
  int *num_bond = atom->num_bond;

  if (i < nlocal) {
    n = num_bond[i];
    for (m = 0; m < n; m++) {
      if (bond_atom[i][m] == tag[j]) {
        for (k = m; k < n - 1; k++) {
          bond_type[i][k] = bond_type[i][k + 1];
          bond_atom[i][k] = bond_atom[i][k + 1];
        }
        num_bond[i]--;
        break;
      }
    }
  }

  // Update special neighbor list
  int n1, n3;

  tagint *slist;
  int **nspecial = atom->nspecial;
  tagint **special = atom->special;

  // remove j from special bond list for atom i
  if (i < nlocal) {
    slist = special[i];
    n1 = nspecial[i][0];
    for (m = 0; m < n1; m++)
        if (slist[m] == atom->tag[j]) break;
    n3 = nspecial[i][2];
    for (; m < n3-1; m++) slist[m] = slist[m+1];
    nspecial[i][0]--;
    nspecial[i][1]--;
    nspecial[i][2]--;
  }
}

/* ---------------------------------------------------------------------- */

void FixBondRupturess::post_integrate_respa(int ilevel, int /*iloop*/)
{
  if (ilevel == nlevels_respa-1) post_integrate();
}

/* ---------------------------------------------------------------------- */

int FixBondRupturess::pack_forward_comm(int n, int *list, double *buf,
                                    int /*pbc_flag*/, int * /*pbc*/)
{
  int i,j,k,m,ns;

  m = 0;

  if (commflag == 1) {
      int tmp1, tmp2;
      index = atom->find_custom(utils::strdup(std::string("fbd_") + id),tmp1,tmp2);
      tagint **fbd = atom->iarray[index];

      for (i = 0; i < n; i++) {
        j = list[i];
        for (k = 0; k < maxbond; k++) {
          buf[m++] = ubuf(fbd[j][k]).d;
        }
      }
      return m;
  }

  if (commflag == 5) {
      int *num_bond = atom->num_bond;
      int **bond_type = atom->bond_type;
      tagint **bond_atom = atom->bond_atom;

      for (i = 0; i < n; i++) {
        j = list[i];
        ns = num_bond[j];
        buf[m++] = ubuf(ns).d;
        for (k = 0; k < ns; k++) {
          buf[m++] = ubuf(bond_type[j][k]).d;
          buf[m++] = ubuf(bond_atom[j][k]).d;
        }
      }
      return m;
  }

  int **nspecial = atom->nspecial;
  tagint **special = atom->special;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    ns = nspecial[j][0];
    buf[m++] = ubuf(ns).d;
    for (k = 0; k < ns; k++)
      buf[m++] = ubuf(special[j][k]).d;
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void FixBondRupturess::unpack_forward_comm(int n, int first, double *buf)
{
  int i,j,m,ns,last;

  m = 0;
  last = first + n;

  if (commflag == 1) {
    int tmp1, tmp2;
    index = atom->find_custom(utils::strdup(std::string("fbd_") + id),tmp1,tmp2);
    tagint **fbd = atom->iarray[index];

    for (i = first; i < last; i++) {
        for (j = 0; j < maxbond; j++) {
          fbd[i][j] = (tagint) ubuf(buf[m++]).i;
        }
    }

  } else if (commflag == 5) {
    int *num_bond = atom->num_bond;
    int **bond_type = atom->bond_type;
    tagint **bond_atom = atom->bond_atom;

    for (i = first; i < last; i++) {
      ns = (int) ubuf(buf[m++]).i;
      num_bond[i] = ns;
        for (j = 0; j < ns; j++) {
          bond_type[i][j] = (int) ubuf(buf[m++]).i;
          bond_atom[i][j] = (tagint) ubuf(buf[m++]).i;
        }
    }

  } else {
    int **nspecial = atom->nspecial;
    tagint **special = atom->special;

    m = 0;
    last = first + n;
    for (i = first; i < last; i++) {
      ns = (int) ubuf(buf[m++]).i;
      nspecial[i][0] = ns;
      for (j = 0; j < ns; j++)
        special[i][j] = (tagint) ubuf(buf[m++]).i;
    }
  }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixBondRupturess::memory_usage()
{
  int nmax = atom->nmax;
  double bytes = maxbond*nmax * sizeof(tagint);
  return bytes;
}
