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

#include "fix_bond_dynamicss.h"

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
#include "random_mars.h"

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixBondDynamicss::FixBondDynamicss(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg),
  distsq(nullptr), probabilities(nullptr), list(nullptr),
  created(nullptr), broken(nullptr), copy(nullptr), random(nullptr), partners_possible_f(nullptr), partners_probs_f(nullptr),
  partners_possible(nullptr), partners_probs(nullptr), npos(nullptr), partners_success(nullptr)
{
  if (narg < 9) error->all(FLERR,"Illegal fix bond/dynamicss command");

  MPI_Comm_rank(world,&me);
  MPI_Comm_size(world,&nprocs);

  nevery = utils::inumeric(FLERR,arg[3],false,lmp);
  if (nevery <= 0) error->all(FLERR,"Illegal fix bond/dynamicss command");

  dynamic_group_allow = 1;
  force_reneighbor = 1;
  next_reneighbor = -1;
  vector_flag = 1;
  size_vector = 2;
  global_freq = 1;
  extvector = 0;

  iatomtype = utils::inumeric(FLERR,arg[4],false,lmp);
  btype = utils::inumeric(FLERR,arg[5],false,lmp);
  ka = utils::numeric(FLERR,arg[6],false,lmp);
  kd = utils::numeric(FLERR,arg[7],false,lmp);
  cutoff = utils::numeric(FLERR,arg[8],false,lmp);

  if (btype < 1 || btype > atom->nbondtypes)
    error->all(FLERR,"Invalid bond type in fix bond/dynamicss command");
  if (cutoff < 0.0) error->all(FLERR,"Illegal fix bond/dynamicss command");
  cutsq = cutoff*cutoff;

  // optional keywords

  jatomtype = iatomtype;
  flag_prob = 0;
  flag_bell = 0;
  flag_rouse = 0;
  flag_critical = 0;
  flag_mol = 0;
  flag_skip = 0;
  skip = 0;
  prob_attach = 0.0;
  prob_detach = 0.0;
  maxbond = atom->bond_per_atom;
  f0 = 0.0;
  b0 = 0.0;
  b2 = 0.0;
  r_critical = 0.0;
  r2_critical = 0.0;
  ka0 = ka;
  kd0 = kd;
  int seed = 12345;

  int iarg = 9;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"prob") == 0) {
      if (iarg+4 > narg) error->all(FLERR,"Illegal fix bond/dynamicss command");
      prob_attach = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      prob_detach = utils::numeric(FLERR,arg[iarg+2],false,lmp);
      seed = utils::inumeric(FLERR,arg[iarg+3],false,lmp);
      flag_prob = 1;
      if (prob_attach < 0.0 || prob_attach > 1.0 || prob_detach < 0.0 || prob_detach > 1.0)
        error->all(FLERR,"Illegal fix bond/dynamicss command");
      if (seed <= 0) error->all(FLERR,"Illegal fix bond/dynamicss command");
      iarg += 4;
    } else if (strcmp(arg[iarg],"maxbond") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix bond/dynamicss command");
      maxbond = utils::inumeric(FLERR,arg[iarg+1],false,lmp);
      if (maxbond < 0) error->all(FLERR,"Illegal fix bond/dynamicss command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"bell") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix bond/dynamicss command");
      f0 = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      flag_bell = 1;
      iarg += 2;
    } else if (strcmp(arg[iarg],"rouse") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix bond/dynamicss command");
      b0 = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      b2 = b0*b0;
      flag_rouse = 1;
      iarg += 2;
    } else if (strcmp(arg[iarg],"critical") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix bond/dynamicss command");
      r_critical = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      r2_critical = r_critical*r_critical;
      flag_critical = 1;
      iarg += 2;
    } else if (strcmp(arg[iarg],"jtype") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix bond/dynamicss command");
      jatomtype = utils::inumeric(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg],"mol") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix bond/dynamicss command");
      flag_mol = utils::inumeric(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg],"skip") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix bond/dynamicss command");
      flag_skip = utils::inumeric(FLERR,arg[iarg+1],false,lmp);
      skip = 1;
      iarg += 2;
    } else error->all(FLERR,"Illegal fix bond/dynamicss command");
  }

  // error checks

  if (flag_prob && flag_bell)
    error->all(FLERR,"Cannot use argument prob with argument bell");
  if (atom->molecular != Atom::MOLECULAR)
    error->all(FLERR,"Cannot use fix bond/dynamicss with non-molecular systems");
  if (atom->bond_per_atom < maxbond)
    error->all(FLERR,"Maxbond too large in fix bond/dynamicss - increase bonds/per/atom");

  // initialize Marsaglia RNG with processor-unique seed

  random = new RanMars(lmp,seed + me);

  // allocate values local to this fix

  nmax = 0;
  distsq = nullptr;
  created = nullptr;
  broken = nullptr;
  partners_possible = partners_possible_f = nullptr;
  partners_probs = partners_probs_f = nullptr;
  npos = nullptr;
  partners_success = nullptr;

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

  // zero out stats

  countflag = 0;

  breakcount = 0;
  breakcounttotal = 0;

  createcount = 0;
  createcounttotal = 0;
}

/* ---------------------------------------------------------------------- */

FixBondDynamicss::~FixBondDynamicss()
{
  delete random;

  // delete locally stored arrays

  memory->destroy(partners_possible);
  memory->destroy(partners_probs);
  memory->destroy(partners_possible_f);
  memory->destroy(partners_probs_f);
  memory->destroy(partners_success);
  memory->destroy(npos);

  if (new_fix_id && modify->nfix) modify->delete_fix(new_fix_id);
  delete [] new_fix_id;
  delete [] copy;

}

/* ---------------------------------------------------------------------- */

int FixBondDynamicss::setmask()
{
  int mask = 0;
  mask |= POST_INTEGRATE;
  mask |= POST_INTEGRATE_RESPA;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixBondDynamicss::post_constructor()
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

void FixBondDynamicss::init() //POSSIBLY ADD NEWTON_BOND FLAG!!!!!
{
  if (utils::strmatch(update->integrate_style,"^respa"))
    nlevels_respa = ((Respa *) update->integrate)->nlevels;

  // check cutoff for iatomtype,jatomtype

  if (force->pair == nullptr || cutsq > force->pair->cutsq[iatomtype][jatomtype])
    error->all(FLERR,"Fix bond/dynamicss cutoff is longer than pairwise cutoff");

  // need a half neighbor list, built every Nevery steps

  neighbor->add_request(this, NeighConst::REQ_OCCASIONAL);
}

/* ---------------------------------------------------------------------- */

void FixBondDynamicss::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

void FixBondDynamicss::setup(int /*vflag*/)
{
  int i,j,m;

  // compute initial bond neighbors if this is first run
  // can't do this earlier, in constructor or init, b/c need ghost info

  if (countflag) return;
  countflag = 1;

  // Initialize local atom array fbd
  // Custom list of dynamic bonds per atom
  //    0: open for bonding
  //   -1: to be broken this timestep
  //   -2: permanently broken
  //   >0: atom->tag of bonded atom

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
      // if (bond_type[i][j] == btype) {
        fbd[i][j] = bond_atom[i][j]; // ADD OPTION FOR INCLUDING OUTSIDE ATOMS OR NOT DIRECTLY
      // }
    }
  }

  // forward communication of fbd so ghost atoms have it
  commflag = 1;
  comm->forward_comm(this,maxbond);

  // Create initial memory allocations
  memory->create(partners_possible,nmax,maxbond,"bond/dynamicss:possible_partners");
  memory->create(partners_probs,nmax,maxbond,"bond/dynamicss:partners_probs");
  memory->create(partners_possible_f,nmax,maxbond,"bond/dynamicss:possible_partners_f");
  memory->create(partners_probs_f,nmax,maxbond,"bond/dynamicss:partners_probs_f");
  memory->create(partners_success,nmax,maxbond,"bond/dynamicss:partners_success");
  memory->create(npos,nmax,"bond/dynamicss:npos");
}

/* ---------------------------------------------------------------------- */

void FixBondDynamicss::post_integrate()
{
  int i,j,k,m,n,nn,b,bb,ii,jj,i1,i2,n1,n2,n3,type_tmp,inum,jnum,itype,jtype,possible;
  double xtmp,ytmp,ztmp,delx,dely,delz,rsq,prob_detach_f0,probability,done,nposj,fbond,engpot;
  int *ilist,*jlist,*numneigh,**firstneigh;
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

  int **nspecial = atom->nspecial;
  tagint **special = atom->special;

  if (update->ntimestep % nevery) return;

  // acquire updated ghost atom positions
  // necessary b/c are calling this after integrate, but before Verlet comm

  comm->forward_comm();

  // basic atom information

  double **x = atom->x;
  tagint *tag = atom->tag;
  int *mask = atom->mask;
  int *type = atom->type;
  Bond *bond = force->bond;
  DT_EQ = (update->dt)*nevery;
  // JTC: Probably not worth worrying about, but this definition of DT_EQ won't be
  // compatible with a variable timestep like that used in fix dt/reset.
  // Not sure there's a great solution (maybe incrementing?) or a good error check

  /* BEGIN BREAKING PROCESS */
  // loop over local atoms
  // check for possible breaks

  if (!flag_prob && !flag_bell) prob_detach = 1 - exp(-kd0*DT_EQ);
  for (i = 0; i < nlocal; i++) {

    if (!(mask[i] & groupbit)) continue;
    if (!(type[i] == iatomtype)) continue;

    if (flag_critical || flag_bell) {
      xtmp = x[i][0];
      ytmp = x[i][1];
      ztmp = x[i][2];
    }
    for (b = 0; b < maxbond; b++) {
      tagj = fbd[i][b];
      
      if (tagj < 1) continue;

      j = atom->map(tagj);
      if (j < 0)
        error->one(FLERR,"Fix bond/dynamicss needs ghost atoms "
                    "from further away");

      if (!(mask[j] & groupbit)) continue;
      if (!(type[j] == jatomtype)) continue;

      // Only consider each bond once - when my atom has the lower atom tag
      if (tag[i] > tagj) continue;

      probability = random->uniform();

      if (flag_bell) {
        delx = xtmp - x[j][0];
        dely = ytmp - x[j][1];
        delz = ztmp - x[j][2];
        domain->minimum_image(delx, dely, delz);
        rsq = delx*delx + dely*dely + delz*delz;
        engpot = bond->single(btype,rsq,i,j,fbond);
        kd = kd0*exp(abs(fbond)/f0);
        prob_detach = 1 - exp(-kd*DT_EQ);
      }
      if (flag_critical) {
        delx = xtmp - x[j][0];
        dely = ytmp - x[j][1];
        delz = ztmp - x[j][2];
        domain->minimum_image(delx, dely, delz);
        rsq = delx*delx + dely*dely + delz*delz;
        if (rsq >= r2_critical) prob_detach = 1.0;
      }

      // Apply probability constraint
      if (probability > prob_detach) continue;

      // if breaking was successful, update fbd to -tag
      fbd[i][b] *= -1;

      // find the entry of atom j and update its fbd as well
      // if j is a ghost atom, it will do this on its own processor
      // at the next step
      if (j < nlocal) {
        for (bb = 0; bb < maxbond; bb++) {
          if (fbd[j][bb] == tag[i]) {
            fbd[j][bb] *= -1;
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
  for (j = nlocal; j < nall; j++) {
    for (b = 0; b < maxbond; b++) {
      tagi = fbd[j][b];
      
      if (tagi > -1) continue;

      i = atom->map(-tagi);
      if (i < 0) continue;

      // find the entry of atom i and update its fbd as well
      for (bb = 0; bb < maxbond; bb++) {
        if (fbd[i][bb] == tag[j]) {
          fbd[i][bb] *= -1;
          break;
        }
      }
    }
  }

  /* BEGIN CREATION PROCESS */

  // Possibly resize the possible_partners array
  if (atom->nmax > nmax) {
    memory->destroy(partners_possible);
    memory->destroy(partners_probs);
    memory->destroy(partners_possible_f);
    memory->destroy(partners_probs_f);
    memory->destroy(partners_success);
    memory->destroy(npos);
    nmax = atom->nmax;
    memory->create(partners_possible,nmax,maxbond,"bond/dynamicss:possible_partners");
    memory->create(partners_probs,nmax,maxbond,"bond/dynamicss:partners_probs");
    memory->create(partners_possible_f,nmax,maxbond,"bond/dynamicss:possible_partners_f");
    memory->create(partners_probs_f,nmax,maxbond,"bond/dynamicss:partners_probs_f");
    memory->create(partners_success,nmax,maxbond,"bond/dynamicss:partners_success");
    memory->create(npos,nmax,"bond/dynamicss:npos");
  }

  // Initialize arrays to zero
  for (i = 0; i < nall; i++) {
    for (j = 0; j < maxbond; j++) {
      partners_possible[i][j] = 0;
      partners_possible_f[i][j] = 0;
      partners_success[i][j] = 0;
      partners_probs[i][j] = 1.0;
      partners_probs_f[i][j] = 1.0;
    }
    npos[i] = 0;
  }

  // Determine how many open slots each atom has
  for (i = 0; i < nlocal; i++) {
    for (b = 0; b < maxbond; b++) {
      if (fbd[i][b] == 0) npos[i]++;
    }
  }

  // Forward communication of open slots
  commflag = 2;
  comm->forward_comm(this,1);

  // build temporary neighbor list to determine closest images

  neighbor->build_one(list,1);
  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;
  tagint *molecule = atom->molecule;

  // find potential bonding partners

  if (!flag_prob && !flag_rouse) prob_attach = 1.0 - exp(-ka0*DT_EQ);
  for (i = 0; i < nlocal; i++) {

    if (!(mask[i] & groupbit)) continue;
    if (!(type[i] == iatomtype)) continue;
    if (npos[i] == 0) continue;

    // Store this atom's coordinates for reuse

    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;

      if (!(mask[j] & groupbit)) continue;
      if (!(type[j] == jatomtype)) continue;
      if (npos[j] == 0) continue;
      if (tag[i] == tag[j]) continue;

      if (flag_mol) {
        if (molecule[i] == molecule[j]) continue;
      }

      // do not allow a duplicate bond to be created
      // check fbd matrix of atom i
      // abs() in case this bond was just broken
      done = 0;
      for (b = 0; b < maxbond; b++) {
        if (abs(fbd[i][b]) == tag[j]) {
          done = 1;
          break;
        }
      }
      if (done) continue;

      // check duplicate via 1-2 neighbors
      for (b = 0; b < nspecial[i][0]; b++)
        if (special[i][b] == tag[j]) done = 1;
      if (done) continue;

      // check if this ghost atom was already seen
      done = 0;
      for (n = 0; n < maxbond; n++) {
        if (partners_possible[i][n] == tag[j]) {
          done = 1;
          break;
        }
      }
      if (done) continue;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      domain->minimum_image(delx, dely, delz);
      rsq = delx*delx + dely*dely + delz*delz;

      if (rsq > cutsq) continue;

      // Determine probability of attachment
      probability = random->uniform();

      if (flag_rouse) {
        ka = ka0*pow(b2/rsq,2);
        prob_attach = 1 - exp(-ka*DT_EQ);
      }

      // No reason to consider this if it will be unsuccessful
      if (probability > prob_attach) continue;

      // Next, check where to insert, if at all
      bb = 0;
      for (n = 0; n < maxbond; n++) {
        if (partners_probs[i][n] < probability) {
          bb++;
        } else {
          break;
        }
      }

      // continue if this is not possible
      if (bb > maxbond-1) continue; 

      // If bb is the last entry, no shifting required
      if (bb == maxbond-1) {
        partners_possible[i][bb] = tag[j];
        partners_probs[i][bb] = probability;
      } else {
        // Shift all elements to the right
        for (n = maxbond-2; n >= bb; n--) {
          partners_possible[i][n+1] = partners_possible[i][n];
          partners_probs[i][n+1] = partners_probs[i][n];
          if (bb == maxbond-2) break;
        }

        partners_possible[i][bb] = tag[j];
        partners_probs[i][bb] = probability;
      }
    }
  }

  // forward communication of partners arrays
  commflag = 3;
  comm->forward_comm(this,2*maxbond);

  // compile list of atoms j that see my owned atoms i
  // could be ghost or local
  double pmax;
  int imax;
  for (j = 0; j < nall; j++) {

    if (!(mask[j] & groupbit)) continue;
    if (!(type[j] == jatomtype)) continue;
    if (npos[j] == 0) continue;

    for (b = 0; b < maxbond; b++) {
      tagi = partners_possible[j][b];
      
      if (tagi < 1) continue;

      i = atom->map(tagi);
      if (i < 0 || i > nlocal) continue;

      // First case: atom i already has atom j as a final partner
      // check probabilities - update to more likely one if needed
      done = 0;
      for (bb = 0; bb < maxbond; bb++) {
        if (partners_possible_f[i][bb] == tag[j]) {
          done = 1;
          if (partners_probs[j][b] < partners_probs_f[i][bb]) partners_probs_f[i][bb] = partners_probs[j][b];
          break;
        }
      }
      if (done) continue;

      // Second case: atom i has open slots for final partners
      // just insert the new partner and probability in the first open slot
      done = 0;
      for (bb = 0; bb < maxbond; bb++) {
        if (partners_possible_f[i][bb] == 0) {
          partners_possible_f[i][bb] = tag[j];
          partners_probs_f[i][bb] = partners_probs[j][b];
          done = 1;
          break;
        }
      }
      if (done) continue;

      // Last case: atom i already has a full final partners matrix
      // find the least likely partner - check if this one is more likely and replace
      imax = 0;
      pmax = partners_probs_f[i][imax];
      for (bb = 0; bb < maxbond; bb++) {
        if (partners_probs_f[i][bb] > pmax) {
          pmax = partners_probs_f[i][bb];
          imax = bb;
        }
      }
      if (partners_probs[j][b] < pmax) {
        partners_probs_f[i][imax] = partners_probs[j][b];
        partners_possible_f[i][imax] = tag[j];
      }
    }
  }

  // Add lists together - no need to communicate partners_probs_f
  for (i = 0; i < nlocal; i++) {
    for (b = 0; b < maxbond; b++) {

      probability = partners_probs_f[i][b];
      tagj = partners_possible_f[i][b];

      // Check where to insert
      bb = 0;
      for (n = 0; n < maxbond; n++) {
        if (partners_probs[i][n] < probability) {
          bb++;
        } else {
          break;
        }
      }

      // continue if this is not possible
      if (bb > maxbond-1) continue;

      // If bb is the last entry, no shifting required
      if (bb == maxbond-1) {
        partners_possible[i][bb] = tagj;
        partners_probs[i][bb] = probability;
      } else {
        // Shift all elements to the right
        for (n = maxbond-2; n >= bb; n--) {
          partners_possible[i][n+1] = partners_possible[i][n];
          partners_probs[i][n+1] = partners_probs[i][n];
          if (bb == maxbond-2) break;
        }

        partners_possible[i][bb] = tagj;
        partners_probs[i][bb] = probability;
      }
    }
  }

  // forward communication of partners arrays
  commflag = 3;
  comm->forward_comm(this,2*maxbond);

  // At this point, the formation of bonds is completely determined
  // Just need to limit the formation to npos per atom and update arrays

  for (i = 0; i < nlocal; i++) {

    if (!(mask[i] & groupbit)) continue;
    if (!(type[i] == iatomtype)) continue;
    if (npos[i] == 0) continue;

    // Loop through possibilites

    for (n = 0; n < npos[i]; n++) {
      tagj = partners_possible[i][n];

      if (tagj < 1) continue;

      // Check local id of potential partner
      j = atom->map(tagj);
      if (j < 0)
        error->one(FLERR,"Fix bond/dynamicss needs ghost atoms "
                    "from further away");

      // find where this bond is in atom j's list
      // if its location is past npos[j], then this was unsuccessful
      // this will be consistant across processors as we are looping through npos[i]
      bb = 0;
      for (b = 0; b < maxbond; b++) {
        if (partners_possible[j][b] == 0) {
          continue;
        } else if (partners_possible[j][b] == tag[i]) {
          break;
        } else {
          bb++;
        }
      }
      if (bb > npos[j]-1) continue;

      // Success! A bond was created. Mark as successful
      // atom j will also do this, whatever proc it's on
      partners_success[i][n] = 1;
    }
  }

  // forward communication of success array
  commflag = 4;
  comm->forward_comm(this,maxbond);

  int flag_remove = 0;
  for (i = 0; i < nlocal; i++) {
    for (b = 0; b < maxbond; b++) {

      // First, process broken bonds
      // check for negative fbd entry
      if (fbd[i][b] < 0) {
        tagj = -fbd[i][b];
        j = atom->map(tagj);

        if (j < 0)
          error->one(FLERR,"Fix bond/dynamicss needs ghost atoms "
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

      // Next, process created bonds
      // check for positive partners_success entry
      if (!partners_success[i][b]) continue;

      tagj = partners_possible[i][b];
      j = atom->map(tagj);

      // Only do this once if both local
      if (j < nlocal && i > j) continue;

      if (j < 0)
        error->one(FLERR,"Fix bond/dynamicss needs ghost atoms "
                    "from further away");

      // do not allow a duplicate bond to be created
      // check fbd entry of atom i
      done = 0;
      for (k = 0; k < maxbond; k++) {
        if (fbd[i][k] == tag[j]) {
          done = 1;
          break;
        }
      }
      if (done) continue;

      // Update atom properties and fbd
      process_created(i,j);
      for (bb = 0; bb < maxbond; bb++) {
        if (fbd[i][bb] == 0) break;
      }
      fbd[i][bb] = tagj;

      if (j < nlocal) {
        for (bb = 0; bb < maxbond; bb++) {
          if (fbd[j][bb] == 0) break;
        }
        fbd[j][bb] = tag[i];
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

  // trigger reneighboring
  next_reneighbor = update->ntimestep;
}

/* ---------------------------------------------------------------------- */

void FixBondDynamicss::process_broken(int i, int j)
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

  // remove i from special bond list for atom j and vice versa
  if (i < nlocal) {
    slist = special[i];
    n1 = nspecial[i][0];
    for (m = 0; m < n1; m++) {
      if (slist[m] == atom->tag[j]) {
        for (k = m; k < n1; k++) {
          special[i][k] = special[i][k+1];
        }
        nspecial[i][0]--;
        break;
      }
    }
  }

  if (j < nlocal) {
    slist = special[j];
    n1 = nspecial[j][0];
    for (m = 0; m < n1; m++) {
      if (slist[m] == atom->tag[i]) {
        for (k = m; k < n1; k++) {
          special[j][k] = special[j][k+1];
        }
        nspecial[j][0]--;
        break;
      }
    }
  }

}

/* --------------------------------------------------------------------- */

void FixBondDynamicss::process_created(int i, int j)
{
  int n1,n2,n3,m,n;
  tagint id1,id2;
  tagint *slist;

  tagint *tag = atom->tag;
  int **nspecial = atom->nspecial;
  tagint **special = atom->special;

  tagint **bond_atom = atom->bond_atom;
  int **bond_type = atom->bond_type;
  int *num_bond = atom->num_bond;

  int newton_bond = force->newton_bond;
  int nlocal = atom->nlocal;

  // Add bonds to atom class for i and j
  if (i < nlocal) {
    if (num_bond[i] == atom->bond_per_atom)
      error->one(FLERR,"New bond exceeded bonds per atom in fix bond/create/dynamic");
    bond_type[i][num_bond[i]] = btype;
    bond_atom[i][num_bond[i]] = tag[j];
    num_bond[i]++;
  }

  if (j < nlocal) {
    if (num_bond[j] == atom->bond_per_atom)
      error->one(FLERR,"New bond exceeded bonds per atom in fix bond/create/dynamic");
    bond_type[j][num_bond[j]] = btype;
    bond_atom[j][num_bond[j]] = tag[i];
    num_bond[j]++;
  }

  // add a 1-2 neighbor to special bond list for atom I
  // atom J will also do this, whatever proc it is on
  // need to first remove tag[j] from later in list if it appears
  // prevents list from overflowing, will be rebuilt in rebuild_special_one()

  n1 = nspecial[i][0];
  n2 = nspecial[i][1];
  n3 = nspecial[i][2];
  for (m = n1; m < n3; m++)
    if (special[i][m] == tag[j]) break;
  if (m < n3) {
    for (n = m; n < n3-1; n++) special[i][n] = special[i][n+1];
    n3--;
    if (m < n2) n2--;
  }
  if (n3 == atom->maxspecial)
    error->one(FLERR,
                "New bond exceeded special list size in fix bond/create/dynamic");
  for (m = n3; m > n1; m--) special[i][m] = special[i][m-1];
  special[i][n1] = tag[j];
  nspecial[i][0] = n1+1;
  nspecial[i][1] = n2+1;
  nspecial[i][2] = n3+1;

  if (j < nlocal) {
    n1 = nspecial[j][0];
    n2 = nspecial[j][1];
    n3 = nspecial[j][2];
    for (m = n1; m < n3; m++)
      if (special[j][m] == tag[i]) break;
    if (m < n3) {
      for (n = m; n < n3-1; n++) special[j][n] = special[j][n+1];
      n3--;
      if (m < n2) n2--;
    }
    if (n3 == atom->maxspecial)
      error->one(FLERR,
                  "New bond exceeded special list size in fix bond/create/dynamic");
    for (m = n3; m > n1; m--) special[j][m] = special[j][m-1];
    special[j][n1] = tag[i];
    nspecial[j][0] = n1+1;
    nspecial[j][1] = n2+1;
    nspecial[j][2] = n3+1;
  }
}

/* ---------------------------------------------------------------------- */

void FixBondDynamicss::post_integrate_respa(int ilevel, int /*iloop*/)
{
  if (ilevel == nlevels_respa-1) post_integrate();
}

/* ---------------------------------------------------------------------- */

int FixBondDynamicss::pack_forward_comm(int n, int *list, double *buf,
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

  if (commflag == 2) {
      for (i = 0; i < n; i++) {
        j = list[i];
        buf[m++] = ubuf(npos[j]).d;
      }
      return m;
  }

  if (commflag == 3) {
      for (i = 0; i < n; i++) {
        j = list[i];
        for (k = 0; k < maxbond; k++) {
          buf[m++] = ubuf(partners_possible[j][k]).d;
          buf[m++] = partners_probs[j][k];
        }
      }
      return m;
  }

  if (commflag == 4) {
      for (i = 0; i < n; i++) {
        j = list[i];
        for (k = 0; k < maxbond; k++) {
          buf[m++] = ubuf(partners_success[j][k]).d;
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

void FixBondDynamicss::unpack_forward_comm(int n, int first, double *buf)
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

  } else if (commflag == 2) {
    for (i = first; i < last; i++) {
      npos[i] = (int) ubuf(buf[m++]).i;
    }

  } else if (commflag == 3) {
    for (i = first; i < last; i++) {
        for (j = 0; j < maxbond; j++) {
          partners_possible[i][j] = (tagint) ubuf(buf[m++]).i;
          partners_probs[i][j] = buf[m++];
        }
    }

  } else if (commflag == 4) {
    for (i = first; i < last; i++) {
        for (j = 0; j < maxbond; j++) {
          partners_success[i][j] = (int) ubuf(buf[m++]).i;
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

double FixBondDynamicss::memory_usage()
{
  int nmax = atom->nmax;
  double bytes = 3*nmax * sizeof(tagint);
  bytes += (double)nmax*3 * sizeof(double);
  bytes += (double)nmax*3 * sizeof(int);
  return bytes;
}
