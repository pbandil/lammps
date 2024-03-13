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

#include "fix_bond_rupture.h"

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

FixBondRupture::FixBondRupture(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg),
  influenced(nullptr), list(nullptr),copy(nullptr)
{
  if (narg < 7) error->all(FLERR,"Illegal fix bond/rupture command");

  MPI_Comm_rank(world,&me);
  MPI_Comm_size(world,&nprocs);

  nevery = utils::inumeric(FLERR,arg[3],false,lmp);
  if (nevery <= 0) error->all(FLERR,"Illegal fix bond/rupture command");

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
    error->all(FLERR,"Invalid bond type in fix bond/rupture command");
  if (r_critical < 0.0) error->all(FLERR,"Illegal fix bond/rupture command");
  r2_critical = r_critical*r_critical;

  jatomtype = iatomtype;

  // optional keywords

  flag_mol = 0;
  flag_skip = 0;
  skip = 0;
  maxbond = atom->bond_per_atom;
  int iarg = 7;

  while (iarg < narg) {
    if (strcmp(arg[iarg],"maxbond") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix bond/rupture command");
      maxbond = utils::inumeric(FLERR,arg[iarg+1],false,lmp);
      if (maxbond < 0) error->all(FLERR,"Illegal fix bond/rupture command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"mol") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix bond/rupture command");
      flag_mol = utils::inumeric(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg],"skip") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix bond/rupture command");
      flag_skip = utils::inumeric(FLERR,arg[iarg+1],false,lmp);
      skip = 1;
      iarg += 2;
    } else error->all(FLERR,"Illegal fix bond/rupture command");
  }

  // error checks

  if (atom->bond_per_atom < maxbond)
    error->all(FLERR,"Maxbond too large in fix bond/rupture - increase bonds/per/atom");

  // allocate values local to this fix
  nmax = 0;
  countflag = 0;
  influenced = nullptr;

  // // copy = special list for one atom
  // // size = ms^2 + ms is sufficient
  // // b/c in rebuild_special_one() neighs of all 1-2s are added,
  // //   then a dedup(), then neighs of all 1-3s are added, then final dedup()
  // // this means intermediate size cannot exceed ms^2 + ms

  int maxspecial = atom->maxspecial;
  copy = new tagint[maxspecial*maxspecial + maxspecial];

  // set comm sizes needed by this fix
  // forward is big due to comm of broken bonds and 1-2 neighbors

  size_bonds_and_neighbors = MAX(2,1+atom->maxspecial);
  size_bond_lists = MAX(2,1+2*(atom->bond_per_atom));

  comm_forward = size_bonds_and_neighbors;
}

/* ---------------------------------------------------------------------- */

FixBondRupture::~FixBondRupture()
{

  // delete locally stored arrays

  memory->destroy(influenced);

  if (new_fix_id && modify->nfix) modify->delete_fix(new_fix_id);
  delete [] new_fix_id;
  delete [] copy;

}

/* ---------------------------------------------------------------------- */

int FixBondRupture::setmask()
{
  int mask = 0;
  mask |= POST_INTEGRATE;
  mask |= POST_INTEGRATE_RESPA;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixBondRupture::post_constructor()
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

void FixBondRupture::init() //POSSIBLY ADD NEWTON_BOND FLAG!!!!!
{
  if (utils::strmatch(update->integrate_style,"^respa"))
    nlevels_respa = ((Respa *) update->integrate)->nlevels;

  // check cutoff for iatomtype,jatomtype

  if (force->pair == nullptr || r2_critical > force->pair->cutsq[iatomtype][jatomtype])
    error->all(FLERR,"Fix bond/rupture cutoff is longer than pairwise cutoff");

  // need a half neighbor list, built every Nevery steps

  neighbor->add_request(this, NeighConst::REQ_OCCASIONAL);
}

/* ---------------------------------------------------------------------- */

void FixBondRupture::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

void FixBondRupture::setup(int /*vflag*/)
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

  // Create initial memory allocations
  memory->create(influenced,nmax,"bond/rupture:influenced");
}

/* ---------------------------------------------------------------------- */

void FixBondRupture::post_integrate()
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
    if (skip) {
      skip = 0;
      return;
    } else {
      skip = 1;
    }
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
        error->one(FLERR,"Fix bond/rupture needs ghost atoms "
                    "from further away");

      if (!(mask[j] & groupbit)) continue;
    //   if (!(type[j] == jatomtype)) continue;

      // Only consider each bond once - when my atom has the lower atom tag
      // if (tag[i] > tagj) continue;

        delx = xtmp - x[j][0];
        dely = ytmp - x[j][1];
        delz = ztmp - x[j][2];
        domain->minimum_image(delx, dely, delz);
        rsq = delx*delx + dely*dely + delz*delz;
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

  // Loop over ghost atoms, find corresponding entries of fdb and update
  // needed when a ghost atom bonded to an owned atom decides to break its bond
  // for (j = nlocal; j < nall; j++) {
  //   for (b = 0; b < maxbond; b++) {
  //     tagi = fbd[j][b];
      
  //     if (tagi > -1) continue;

  //     i = atom->map(-tagi);
  //     if (i < 0) continue;

  //     // find the entry of atom i and update its fbd as well
  //     for (bb = 0; bb < maxbond; bb++) {
  //       if (fbd[i][bb] == tag[j]) {
  //         fbd[i][bb] = -tag[j];
  //         break;
  //       }
  //     }
  //   }
  // }

  // Possibly resize the influenced array
  if (atom->nmax > nmax) {
    memory->destroy(influenced);
    nmax = atom->nmax;
    memory->create(influenced,nmax,"bond/rupture:influenced");
  }

  // Initialize influenced to zero
  for (i = 0; i < nall; i++) {
    influenced[i] = 0;
  }

  // Find influenced atoms (only care about local):
  //    yes if is one of 2 atoms in bond
  //    yes if both atom IDs appear in atom's special list
  //    else no
  // First, loop through explicit breaks
  // for (i = 0; i < nall; i++) {
  //   for (b = 0; b < maxbond; b++) {
  //     if (fbd[i][b] < 0) {
  //       tagj = -fbd[i][b];
  //       j = atom->map(tagj);

  //       // Add influenced if i or j are local
  //       if (i < nlocal) influenced[i] = 1;
  //       if (j < nlocal) influenced[j] = 1;
  //     }
  //   }
  // }

  // Next, find influenced atoms from special lists
  // int **nspecial = atom->nspecial;
  // tagint **special = atom->special;
  // int found;
  // for (i = 0; i < nlocal; i++) {

  //   // skip if already influenced
  //   if (influenced[i]) continue;

  //   n = nspecial[i][2];
  //   slist = special[i];
  //   found = 0;
  //   for (k = 0; k < n; k++)
  //       if (slist[k] == tag[i] || slist[k] == tag[j]) found++;
  //   if (found == 2) influenced[i] = 1;
  // }

  int **nspecial = atom->nspecial;
  tagint **special = atom->special;
  int found;
  // First, loop through explicit breaks
  for (i = 0; i < nlocal; i++) {
    for (b = 0; b < maxbond; b++) {
      if (fbd[i][b] < 0) {
        tagj = -fbd[i][b];
        j = atom->map(tagj);
        if (j < 0)
          error->one(FLERR,"Fix bond/rupture needs ghost atoms "
                      "from further away");

        // Next, find influenced atoms from special lists
        for (ii = 0; ii < nlocal; ii++) {
          if (ii == i || ii == j) continue;
          if (influenced[ii]) continue;

          n = nspecial[ii][2];
          slist = special[ii];
          found = 0;
          for (k = 0; k < n; k++)
              if (slist[k] == tag[i] || slist[k] == tag[j]) found++;
          if (found == 2) influenced[ii] = 1;
        }

        // Add influenced if i/j are local
        if (i < nlocal) influenced[i] = 1;
        if (j < nlocal) influenced[j] = 1;
      }
    }
  }

  // Process breaking events

  for (i = 0; i < nlocal; i++) {
    for (b = 0; b < maxbond; b++) {

      // First, process broken bonds
      // check for negative fbd entry
      if (fbd[i][b] < 0) {
        tagj = -fbd[i][b];
        j = atom->map(tagj);

        if (j < 0)
          error->one(FLERR,"Fix bond/rupture needs ghost atoms "
                      "from further away");

        // Update atom properties and fbd
        process_broken(i,j);
        fbd[i][b] = 0;

        // Only do this once if both local
        if (j < nlocal && i < j) {
          for (bb = 0; bb < maxbond; bb++) {
            if (fbd[j][bb] == -tag[i]) {
              fbd[j][bb] = 0;
              break;
            }
          }
        } 
      }
    }
  }

  // forward communication of fbd
  commflag = 1;
  comm->forward_comm(this,maxbond);

  // forward communication of atom properties
  commflag = 5;
  comm->forward_comm(this,size_bond_lists);

  // forward communication of special lists
  commflag = 6;
  comm->forward_comm(this);

  // update special neigh lists of all atoms affected by any created bond
  for (i = 0; i < nlocal; i++) {
    if (influenced[i]) rebuild_special_one(i);
  }

  // trigger reneighboring
  next_reneighbor = update->ntimestep;
}

/* ---------------------------------------------------------------------- */

void FixBondRupture::process_broken(int i, int j)
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

  if (j < nlocal) {
    n = num_bond[j];
    for (m = 0; m < n; m++) {
      if (bond_atom[j][m] == tag[i]) {
        for (k = m; k < n - 1; k++) {
          bond_type[j][k] = bond_type[j][k + 1];
          bond_atom[j][k] = bond_atom[j][k + 1];
        }
        num_bond[j]--;
        break;
      }
    }
  }

  // Update special neighbor list
  int n1, n3;

  tagint *slist;
  int **nspecial = atom->nspecial;
  tagint **special = atom->special;

  // remove j from special bond list for atom i and vice versa
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

  if (j < nlocal) {
    slist = special[j];
    n1 = nspecial[j][0];
    for (m = 0; m < n1; m++)
        if (slist[m] == atom->tag[i]) break;
    n3 = nspecial[j][2];
    for (; m < n3-1; m++) slist[m] = slist[m+1];
    nspecial[j][0]--;
    nspecial[j][1]--;
    nspecial[j][2]--;
  }

}

/* ----------------------------------------------------------------------
   re-build special list of atom M
   does not affect 1-2 neighs (already include effects of new bond)
   affects 1-3 and 1-4 neighs due to other atom's augmented 1-2 neighs
------------------------------------------------------------------------- */

void FixBondRupture::rebuild_special_one(int m)
{
  int i,j,n,n1,cn1,cn2,cn3;
  tagint *slist;

  tagint *tag = atom->tag;
  int **nspecial = atom->nspecial;
  tagint **special = atom->special;

  // existing 1-2 neighs of atom M

  slist = special[m];
  n1 = nspecial[m][0];
  cn1 = 0;
  for (i = 0; i < n1; i++)
    copy[cn1++] = slist[i];

  // new 1-3 neighs of atom M, based on 1-2 neighs of 1-2 neighs
  // exclude self
  // remove duplicates after adding all possible 1-3 neighs

  cn2 = cn1;
  for (i = 0; i < cn1; i++) {
    n = atom->map(copy[i]);
    if (n < 0)
      error->one(FLERR,"Fix bond/create needs ghost atoms from further away");
    slist = special[n];
    n1 = nspecial[n][0];
    for (j = 0; j < n1; j++)
      if (slist[j] != tag[m]) copy[cn2++] = slist[j];
  }

  cn2 = dedup(cn1,cn2,copy);
  if (cn2 > atom->maxspecial)
    error->one(FLERR,"Special list size exceeded in fix bond/create");

  // new 1-4 neighs of atom M, based on 1-2 neighs of 1-3 neighs
  // exclude self
  // remove duplicates after adding all possible 1-4 neighs

  cn3 = cn2;
  for (i = cn1; i < cn2; i++) {
    n = atom->map(copy[i]);
    if (n < 0)
      error->one(FLERR,"Fix bond/create needs ghost atoms from further away");
    slist = special[n];
    n1 = nspecial[n][0];
    for (j = 0; j < n1; j++)
      if (slist[j] != tag[m]) copy[cn3++] = slist[j];
  }

  cn3 = dedup(cn2,cn3,copy);
  if (cn3 > atom->maxspecial)
    error->one(FLERR,"Special list size exceeded in fix bond/create");

  // store new special list with atom M

  nspecial[m][0] = cn1;
  nspecial[m][1] = cn2;
  nspecial[m][2] = cn3;
  memcpy(special[m],copy,cn3*sizeof(int));
}

/* ----------------------------------------------------------------------
   remove all ID duplicates in copy from Nstart:Nstop-1
   compare to all previous values in copy
   return N decremented by any discarded duplicates
------------------------------------------------------------------------- */

int FixBondRupture::dedup(int nstart, int nstop, tagint *copy)
{
  int i;

  int m = nstart;
  while (m < nstop) {
    for (i = 0; i < m; i++)
      if (copy[i] == copy[m]) {
        copy[m] = copy[nstop-1];
        nstop--;
        break;
      }
    if (i == m) m++;
  }

  return nstop;
}

/* ---------------------------------------------------------------------- */

void FixBondRupture::post_integrate_respa(int ilevel, int /*iloop*/)
{
  if (ilevel == nlevels_respa-1) post_integrate();
}

/* ---------------------------------------------------------------------- */

int FixBondRupture::pack_forward_comm(int n, int *list, double *buf,
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

void FixBondRupture::unpack_forward_comm(int n, int first, double *buf)
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

double FixBondRupture::memory_usage()
{
  int nmax = atom->nmax;
  double bytes = (double)nmax * sizeof(int);
  bytes += maxbond*nmax * sizeof(tagint);
  return bytes;
}
