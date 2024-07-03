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
#include "fix_tridynamic.h"

#include "arg_info.h"
#include "atom.h"
#include "atom_masks.h"
#include "atom_vec.h"
#include "cell.hh"
#include "comm.h"
#include "compute.h"
#include "domain.h"
#include "error.h"
#include "group.h"
#include "input.h"
#include "math_extra.h"
#include "memory.h"
#include "modify.h"
#include "random_mars.h"
#include "region.h"
#include "respa.h"
#include "update.h"
#include "variable.h"

#include <algorithm>
#include <bits/stdc++.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <math.h>
#include <random>
#include <set>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace std;
using namespace std::chrono;
using namespace voro;

enum { NONE, CONSTANT, EQUAL, ATOM };

/* ---------------------------------------------------------------------- */

FixTriDynamic::FixTriDynamic(LAMMPS *lmp, int narg, char **arg) :
    Fix(lmp, narg, arg), id_compute_voronoi(nullptr), cell_shape(nullptr), wgn(nullptr),
    idregion(nullptr), region(nullptr)
{
  if (narg < 13) error->all(FLERR, "Illegal fix TriDynamic command: not sufficient args");

  MPI_Comm_rank(world, &me);
  MPI_Comm_size(world, &nprocs);

  // initialize Marsaglia RNG with processor-unique seed (for adding noise)
  seed_wgn = 1;
  wgn = new RanMars(lmp, seed_wgn + comm->me);
  srand(10);

  dynamic_group_allow = 1;
  energy_peratom_flag = 1;
  virial_global_flag = virial_peratom_flag = 1;
  thermo_energy = thermo_virial = 1;

  respa_level_support = 1;
  ilevel_respa = 0;

  // Read the simulation parameters

  kp = utils::numeric(FLERR, arg[3], false, lmp);
  p0 = utils::numeric(FLERR, arg[4], false, lmp);
  id_compute_voronoi = utils::strdup(arg[5]);
  fa = utils::numeric(FLERR, arg[6], false, lmp);
  Js = utils::numeric(FLERR, arg[7], false, lmp);
  Jn = utils::numeric(FLERR, arg[8], false, lmp);
  Jv = utils::numeric(FLERR, arg[9], false, lmp);
  var = utils::numeric(FLERR, arg[10], false, lmp);
  neighs_MAX = utils::numeric(FLERR, arg[11], false, lmp);
  gamma_R = utils::numeric(FLERR, arg[12], false, lmp);

  /*This fix takes in input as per-atom array
  produced by compute voronoi*/

  vcompute = modify->get_compute_by_id(id_compute_voronoi);
  if (!vcompute)
    error->all(FLERR, "Could not find compute ID {} for voronoi compute", id_compute_voronoi);

  //parse values for optional arguments
  nevery = 1;    // Using default value for now

  if (narg > 13) {
    idregion = utils::strdup(arg[13]);
    region = domain->get_region_by_id(idregion);
  }

  // Initialize nmax and virial pointer
  nmax = atom->nmax;
  cell_shape = nullptr;

  // Specify attributes for dumping connectivity (neighs_array)
  // This fix generates a per-atom array with specified columns as output,
  // containing information for owned atoms (nlocal on each processor) (accessed from dump file)

  peratom_flag = 1;
  size_peratom_cols = neighs_MAX * 2;
  peratom_freq = 1;

  // perform initial allocation of atom-based arrays
  // register with Atom class
  if (peratom_flag) {
    FixTriDynamic::grow_arrays(atom->nmax);
    atom->add_callback(Atom::GROW);
  }
}

/* ---------------------------------------------------------------------- */

FixTriDynamic::~FixTriDynamic()
{
  delete[] id_compute_voronoi;
  delete[] idregion;
  delete wgn;

  memory->destroy(cell_shape);

  // unregister callbacks to this fix from atom class
  if (peratom_flag) { atom->delete_callback(id, Atom::GROW); }

  if (new_fix_id && modify->nfix) modify->delete_fix(new_fix_id);
  delete[] new_fix_id;

  // fclose(fp); //no writing of file is required
}

/* ---------------------------------------------------------------------- */
// returntype classname :: functidentifier(args)

int FixTriDynamic::setmask()
{
  datamask_read = datamask_modify = 0;

  int mask = 0;
  mask |= POST_FORCE;
  mask |= POST_FORCE_RESPA;
  mask |= MIN_POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixTriDynamic::post_constructor()
{
  // Create call to fix property/atom for storing neighs IDs

  new_fix_id = utils::strdup(
      id + std::string("_FIX_PA"));    // This is the name of the new fix property/atom
  modify->add_fix(fmt::format("{} {} property/atom i2_neighs {} ghost yes", new_fix_id,
                              group->names[igroup], std::to_string(neighs_MAX)));
  // the modify command is creating the fix proptery/atom call

  int tmp1, tmp2;    //these are flag variables that will be returned by find_custom
  index = atom->find_custom("neighs", tmp1, tmp2);
}

/* ---------------------------------------------------------------------- */

void FixTriDynamic::init()
{
  // set index and check validity of region
  if (idregion) {
    region = domain->get_region_by_id(idregion);
    if (!region) error->all(FLERR, "Region {} for fix tridynamic does not exist", idregion);
  }

  if (utils::strmatch(update->integrate_style, "^respa")) {
    ilevel_respa = (dynamic_cast<Respa *>(update->integrate))->nlevels - 1;
    if (respa_level >= 0) ilevel_respa = MIN(respa_level, ilevel_respa);
  }

  /*compute voronoi was giving weird results (not handling periodicity), hence moved to setup*/
}

/*-----------------------------------------------------------------------------------------*/

void FixTriDynamic::setup(int vflag)
{
  // Proc info
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int nall = nlocal + nghost;
  tagint *tag = atom->tag;

  // Invoke compute
  modify->clearstep_compute();
  vcompute = modify->get_compute_by_id(id_compute_voronoi);
  //  int lf = vcompute->local_flag;
  //  int pf = vcompute->peratom_flag;

  /*For peratom array*/

  //   if (!(vcompute->invoked_flag & Compute::INVOKED_PERATOM)) {
  //     vcompute->compute_peratom();
  //     vcompute->invoked_flag |= Compute::INVOKED_PERATOM;
  //   }

  /*We only need local array for our purpose*/
  if (!(vcompute->invoked_flag & Compute::INVOKED_LOCAL)) {
    vcompute->compute_local();
    vcompute->invoked_flag |= Compute::INVOKED_LOCAL;
  }

  int num_rows_loc = vcompute->size_local_rows;

  /*Create neighs array (local) from vcompute->local array*/

  //pointer to neighs
  tagint **neighs = atom->iarray[index];

  // Initialize entries to 0 (space allocation)
  for (int i = 0; i < nall; i++) {
    for (int j = 0; j < neighs_MAX; j++) {
      if (mask[i] & groupbit) { neighs[i][j] = 0; }
    }
  }

  // Populate neighs array with global ids of the voronoi neighs (initial)

  for (int n = 0; n < num_rows_loc; n++) {
    // skip rows with neighbor ID 0 as they denote z surfaces:
    if (int(vcompute->array_local[n][1]) == 0) { continue; }
    // get the local ID of atom
    int i = atom->map(int(vcompute->array_local[n][0]));
    if (i < 0) {
      error->one(FLERR, "Didn't find the atom");
      // Since the array is local, this should not get invoked
    }
    // Get the global ID of the neighbor of cell i
    int neigh_id = int(vcompute->array_local[n][1]);
    int m = 0;
    // Find the first empty space in the neighs array for atom i
    while (neighs[i][m] != 0) { m = m + 1; }
    if (mask[i] & groupbit) { neighs[i][m] = neigh_id; }
  }

  // communicate these initial values to neighboring processors to access ghost atom info
  commflag = 2;
  comm->forward_comm(this, neighs_MAX);    //here neighs_MAX  is the size/no of columns

  /*Create memeory allocations for other per atom info*/

  //cell_shape array does not need to persist accross time steps for atoms that have moved accross procs

  nmax = atom->nmax;
  memory->create(cell_shape, nmax, 3, "fix_tridynamic:cell_shape");

  /*Don't need this verlet or respa stuff (probably!) as long as things are going well*/

  /*Initializing the neighs_array before post force method is called inside setup*/

  // if (utils::strmatch(update->integrate_style, "^verlet")){
  //   post_force(vflag);
  // }
  // else {
  //   (dynamic_cast<Respa *>(update->integrate))->copy_flevel_f(ilevel_respa);
  //   post_force_respa(vflag, ilevel_respa, 0);
  //   (dynamic_cast<Respa *>(update->integrate))->copy_f_flevel(ilevel_respa);
  // }

  // run for zeroth step
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */
/*Not needed this as we are not doing any minimization and only doing dynamic runs*/

// void FixTriDynamic::min_setup(int vflag)
// {
//   post_force(vflag);
// }

/* --------------------------------------------------------------------------------- */

void FixTriDynamic::post_force(int vflag)
{
  /*Read in current data*/

  double **x = atom->x;
  double **f = atom->f;
  double **v = atom->v;
  int *mask = atom->mask;
  imageint *image = atom->image;
  tagint *tag = atom->tag;
  double dt = update->dt;
  int *sametag = atom->sametag;    //this returns the next local id of the atom having the same tag

  int natoms = atom->natoms;
  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int nall = nlocal + nghost;
  double *cut = comm->cutghost;

  if (update->ntimestep % nevery) return;

  // virial setup
  v_init(vflag);
  // update region if necessary
  if (region) region->prematch();

  int me = comm->me;    //current rank value
  int nprocs = comm->nprocs;

  /**** STEP 1: Get the neighs list and make it cyclic (CCW) ****/

  tagint **neighs = atom->iarray[index];

  // Determine the number of neighs for all atoms (owned+ghost)
  // This info is required frequently so better store it instead of recomputing every time
  // Doing it for nall right now and see later if you require for nall. if not just store for nlocal
  // We need num of neighs for ghost atoms in function get_non_bonded

  int num_neighs[nall] = {neighs_MAX};

  for (int i = 0; i < nall; i++) {
    for (int j = 0; j < neighs_MAX; j++) {
      if (neighs[i][j] == 0) {
        num_neighs[i] = j;
        break;
      }
    }
  }

  // Arrange neighs in cyclic manner (Just to make sure)
  // While arranging this will tell you whether a neighbor has moved out the ghost cutoff or not
  // If thrown an exception, increase the cutoff

  for (int i = 0; i < nlocal; i++) { arrange_cyclic(neighs[i], num_neighs[i], i); }

  // Communicate neighs (to have ordered neighs list of ghost atoms as well)
  commflag = 2;
  comm->forward_comm(this, neighs_MAX);

  /**** STEP 2: Get cell shape info (area, perimeter, energy) in the current topology ****/

  // Possibly resize arrays
  if (atom->nmax > nmax) {
    memory->destroy(cell_shape);
    nmax = atom->nmax;
    memory->create(cell_shape, nmax, 3, "fix_tridynamic:cell_shape");
  }

  // Initialize arrays to zero
  for (int i = 0; i < nall; i++) {
    for (int j = 0; j < 3; j++) { cell_shape[i][j] = 0.0; }
  }

  // Reset array-atom if outputting
  if (peratom_flag) {
    for (int i = 0; i < nlocal; i++) {
      for (int j = 0; j < neighs_MAX * 2; j++) { array_atom[i][j] = 0.0; }
    }
  }

  // Store topology for outputting at current step (this is before updating the connectivity)
  // we would still need to decide on what topology, the forces will be calculated
  // Whether it is on current one or the updated one

  // if (peratom_flag) {
  //   for (int i = 0; i < nlocal; i++) {
  //     for (int j = 0; j < neighs_MAX; j++) { array_atom[i][j] = neighs[i][j]; }
  //   }
  // }

  //Calculate areas, perimeters and energy of nlocal atoms and populate array_atom with vertices

  for (int i = 0; i < nlocal; i++) {
    get_cell_data(array_atom[i], cell_shape[i], neighs[i], num_neighs[i], i);
  }

  //forward communicate the cell shape data
  commflag = 1;
  comm->forward_comm(this, 3);

  /****STEP 4: Implement monte-carlo algorithm to update the network topology****/

  //First find the number of MC steps/iterations: N_mc = total neighs for all owned atoms accross procs/2

  int Num_neighs_this = 0;    //local value
  for (int i = 0; i < nlocal; i++) { Num_neighs_this += num_neighs[i]; }

  int Num_mc_steps = Num_neighs_this / 2;
  srand(time(NULL) + me);    //random seed for each rank

  //Start Monte-Carlo
  
  int num_swaps_rejected = 0;
  
  for (int n = 0; n < Num_mc_steps; n++) {

    //allocate memory for storing tags of cells in the Quad in this mc step and some other info
    tagint cell_tags[4] = {0};

    int cell_i = rand() % nlocal;    //return a random integer between [0, nlocal-1]

    //error check for num_neighs_celli-->behvaing a bit wierdly! when full
    if (num_neighs[cell_i] == 0) {
      print_neighs_list(neighs[cell_i], neighs_MAX, tag[cell_i]);
      error->one(FLERR, "no of neighs for atom is 0");
    }

    int rand_neigh_idx =
        rand() % num_neighs[cell_i];    //return a random neigh index [0, num_neighs_i - 1]
    int tmpj = atom->map(neighs[cell_i][rand_neigh_idx]);
    int cell_j = domain->closest_image(cell_i, tmpj);

    int comm_neigh[2] = {-1};    // array to store common pair atoms
    int num_comm_neighs = get_comm_neigh(comm_neigh, neighs[cell_i], neighs[cell_j],
                                         num_neighs[cell_i], num_neighs[cell_j]);

    // Error check for comm_neighs
    if (comm_neigh[0] == comm_neigh[1]) {
      printf("\n %d and %d\n", comm_neigh[0], comm_neigh[1]);
      print_neighs_list(neighs[cell_i], num_neighs[cell_i], tag[cell_i]);
      print_neighs_list(neighs[cell_j], num_neighs[cell_j], tag[cell_j]);
      error->one(FLERR, "both the comm neighs are same-->might be due some atom added twice");
    }

    //get the local ids using closest image
    int cell_k = domain->closest_image(cell_i, atom->map(comm_neigh[0]));
    int cell_l = domain->closest_image(cell_i, atom->map(comm_neigh[1]));

    cell_tags[0] = tag[cell_i];
    cell_tags[1] = tag[cell_j];
    cell_tags[2] = tag[cell_k];
    cell_tags[3] = tag[cell_l];

    //a flag variable that tells whether k and l bonded or not: 1 if unbonded and 0 if bonded
    int flag_unbonded = unbonded(neighs[cell_k], num_neighs[cell_k], tag[cell_l]);

    //Check for validity of bond between i and j to be swapped

    if (num_comm_neighs != 2 || num_neighs[cell_i] <= 4 || num_neighs[cell_j] <= 4) {
      continue;    //go to the next MC step and leave this bond
    }

    //Store the local indices of 'owned' atoms

    int owned_i = atom->map(cell_tags[0]);
    int owned_j = atom->map(cell_tags[1]);
    int owned_k = atom->map(cell_tags[2]);
    int owned_l = atom->map(cell_tags[3]);

    //store energy before swap
    double E_before = cell_shape[owned_i][2] + cell_shape[owned_j][2] + cell_shape[owned_k][2] +
        cell_shape[owned_l][2];

    //Attempt a bond swap--> Update neighs information first for all atoms (and their ghost images) involved

    //Delete bond
    for (int m = 0; m < 2; m++) {
      int owned_atom = atom->map(cell_tags[m]);
      int imj_atom = owned_atom;
      tagint neigh_tag;
      if (m == 0) {
        neigh_tag = cell_tags[1];
      } else if (m == 1) {
        neigh_tag = cell_tags[0];
      }
      while (imj_atom != -1) {
        remove_neigh(imj_atom, neigh_tag, neighs[imj_atom], num_neighs[imj_atom], neighs_MAX);
        num_neighs[imj_atom] -= 1;
        if (num_neighs[imj_atom] == 3) { error->one(FLERR, "number of neighs fell below 4"); }
        arrange_cyclic(neighs[imj_atom], num_neighs[imj_atom], imj_atom);
        imj_atom = sametag[imj_atom];
      }
    }

    //Create bond if not already bonded
    if (flag_unbonded == 1) {
      for (int m = 2; m < 4; m++) {
        int owned_atom = atom->map(cell_tags[m]);
        int imj_atom = owned_atom;
        tagint neigh_tag;
        if (m == 2) {
          neigh_tag = cell_tags[3];
        } else if (m == 3) {
          neigh_tag = cell_tags[2];
        }
        while (imj_atom != -1) {
          neighs[imj_atom][num_neighs[imj_atom]] = neigh_tag;
          num_neighs[imj_atom] += 1;
          if (num_neighs[imj_atom] == neighs_MAX) {
            printf("\n Limit reached for adding new neighs to atom %d\n", tag[imj_atom]);
            error->one(FLERR, "EXITED");
          }
          arrange_cyclic(neighs[imj_atom], num_neighs[imj_atom], imj_atom);
          imj_atom = sametag[imj_atom];
        }
      }
    }

    //Now get the cell_shape data for all atoms and compute energy after swap

    for (int m = 0; m < 4; m++) {
      int owned_atom = atom->map(cell_tags[m]);
      get_cell_data(array_atom[owned_atom], cell_shape[owned_atom], neighs[owned_atom],
                    num_neighs[owned_atom], owned_atom);
      int imj_atom = sametag[owned_atom];
      while (imj_atom != -1) {
        cell_shape[imj_atom][0] = cell_shape[owned_atom][0];
        cell_shape[imj_atom][1] = cell_shape[owned_atom][1];
        cell_shape[imj_atom][2] = cell_shape[owned_atom][2];
        imj_atom = sametag[imj_atom];
      }
    }

    double E_after = cell_shape[owned_i][2] + cell_shape[owned_j][2] + cell_shape[owned_k][2] +
        cell_shape[owned_l][2];

    double Delta_E = E_after - E_before;
    int flag_swapped = 1;

    //Metropolis scheme

    if (Delta_E < 0) {
      continue;
    } 
    else {
      double Prob = exp(-1.0 * Delta_E / KT);
      double rand_num = (double) rand() / RAND_MAX;    //generate a random no between 0 and 1
      if (Prob > rand_num) {
        continue;  //accept the swap with this probability
      } 
      else {
        flag_swapped = 0;    //swapping attempt failed as not energetically favoured
        num_swaps_rejected += 1;
      }
    }

    if (flag_swapped == 0) {

      cell_tags[0] = tag[cell_k];
      cell_tags[1] = tag[cell_l];
      cell_tags[2] = tag[cell_i];
      cell_tags[3] = tag[cell_j];

      //First delete bond between atom k and l (only if they were previously unbonded)
      if (flag_unbonded == 1) {
        for (int m = 0; m < 2; m++) {
          int owned_atom = atom->map(cell_tags[m]);
          int imj_atom = owned_atom;
          tagint neigh_tag;
          if (m == 0) {
            neigh_tag = cell_tags[1];
          } else if (m == 1) {
            neigh_tag = cell_tags[0];
          }
          while (imj_atom != -1) {
            remove_neigh(imj_atom, neigh_tag, neighs[imj_atom], num_neighs[imj_atom], neighs_MAX);
            num_neighs[imj_atom] -= 1;
            arrange_cyclic(neighs[imj_atom], num_neighs[imj_atom], imj_atom);
            imj_atom = sametag[imj_atom];
          }
        }
      }

      //Then Create bond between i and j

      for (int m = 2; m < 4; m++) {
        int owned_atom = atom->map(cell_tags[m]);
        int imj_atom = owned_atom;
        tagint neigh_tag;
        if (m == 2) {
          neigh_tag = cell_tags[3];
        } else if (m == 3) {
          neigh_tag = cell_tags[2];
        }
        while (imj_atom != -1) {
          neighs[imj_atom][num_neighs[imj_atom]] = neigh_tag;
          num_neighs[imj_atom] += 1;
          arrange_cyclic(neighs[imj_atom], num_neighs[imj_atom], imj_atom);
          imj_atom = sametag[imj_atom];
        }
      }

      // Again go back to original information
      for (int m = 0; m < 4; m++) {
        int owned_atom = atom->map(cell_tags[m]);
        get_cell_data(array_atom[owned_atom], cell_shape[owned_atom], neighs[owned_atom],
                      num_neighs[owned_atom], owned_atom);
        int imj_atom = sametag[owned_atom];
        while (imj_atom != -1) {
          cell_shape[imj_atom][0] = cell_shape[owned_atom][0];
          cell_shape[imj_atom][1] = cell_shape[owned_atom][1];
          cell_shape[imj_atom][2] = cell_shape[owned_atom][2];
          imj_atom = sametag[imj_atom];
        }
      }
    }
  }

  commflag = 2;
  comm->forward_comm(this, neighs_MAX);

  //END OF MONTE CARLO

  /****STEP 6: Again create polygonal tiling and store vertices with updated neighs list****/

  /****STEP 8: Do the force calculation based on updated toplogy****/

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      f[i][0] += 0.0;
      f[i][1] += 0.0;
      f[i][2] += 0.0;
    }
    //printf("force on cell %d is %f...%f...%f\n", tag[i], f[i][0], f[i][1], f[i][2]);
  }
}

/* ---------------------------------------------------------------------- */
/*<<<<<<<<<<<<<<<<<<<<<< HELPER FUNCTIONS (BEGIN) >>>>>>>>>>>>>>>>>>>>>>>>>*/
/* ---------------------------------------------------------------------- */

/*~~~~~~~~~~~~~~~~~~~ FUNCTION 1: Order the neighboring cells in CCW manner ~~~~~~~~~~~~~~~~*/

void FixTriDynamic::arrange_cyclic(tagint *celli_neighs, int num_faces, int icell)
{

  double **x = atom->x;
  tagint *tag = atom->tag;

  // Calculate angle for all points
  double theta[num_faces];
  int indices[num_faces];

  for (int j = 0; j < num_faces; j++) {

    // Local id of current face
    int ktmp = atom->map(celli_neighs[j]);

    //Error checks

    if (ktmp < 0) {
      //the neigh might not lie in [0,nall) and ghost cutoff might need to be increased
      //OR cell tag 0 was added as neighbor
      print_neighs_list(celli_neighs, num_faces, tag[icell]);
      printf("Problematic Neighbor: %d", celli_neighs[j]);
      error->one(FLERR, "\n local cell neighbor not found-->might have moved beyond ghost cutoff!");
    }

    if (tag[ktmp] == tag[icell]) {
      printf("\n Cell %d is a neighbor to itself \n", tag[icell]);
      error->one(FLERR, "EXITED");
    }

    //No errors detected--> move ahead

    int k = domain->closest_image(icell, ktmp);
    theta[j] = atan2(x[k][1] - x[icell][1], x[k][0] - x[icell][0]);

    if (isnan(theta[j])) { error->one(FLERR, "theta j in arrange cyclic function returned nan"); }

    // indices array for sorting
    indices[j] = j;
  }

  // Sort indices based on minimum angle
  std::sort(indices, indices + num_faces, [&](int j, int n) {
    return theta[j] < theta[n];
  });

  // Create dummy vector for sorting
  tagint celli_neighs_sorted[neighs_MAX];
  for (int j = 0; j < neighs_MAX; j++) { celli_neighs_sorted[j] = 0; }
  for (int j = 0; j < num_faces; j++) { celli_neighs_sorted[j] = celli_neighs[indices[j]]; }

  // Fill tag_vec with sorted values
  for (int j = 0; j < num_faces; j++) { celli_neighs[j] = celli_neighs_sorted[j]; }
}

/*~~~~~~~~~~~~~~~~~~~Function 2: Find cell polygon area, peri and energy~~~~~~~~~~~~~~~~*/

void FixTriDynamic::get_cell_data(double *celli_vertices, double *celli_geom, tagint *celli_neighs,
                                  int num_faces, int icell)
{

  // Pointer to atom positions
  double **x = atom->x;
  tagint *tag = atom->tag;

  // Coordinates of the vertices
  double vert[num_faces][2];

  // Find coordinates of each vertex
  for (int n = 0; n < num_faces; n++) {

    // Indices of current triangulation
    int mu1 = n;
    int mu2 = n + 1;

    // Wrap back to first vertex for final term
    if (mu2 == num_faces) { mu2 = 0; }

    // local id of mu1
    int j_mu1 = domain->closest_image(icell, atom->map(celli_neighs[mu1]));

    // local id of mu2
    int j_mu2 = domain->closest_image(icell, atom->map(celli_neighs[mu2]));

    // Coordinates of current triangulation
    double xn[3] = {x[icell][0], x[j_mu1][0], x[j_mu2][0]};
    double yn[3] = {x[icell][1], x[j_mu1][1], x[j_mu2][1]};

    // Find centroid
    vert[n][0] = (xn[0] + xn[1] + xn[2]) / 3.0;
    vert[n][1] = (yn[0] + yn[1] + yn[2]) / 3.0;

    celli_vertices[2 * n] = vert[n][0];
    celli_vertices[2 * n + 1] = vert[n][1];
  }

  double area = 0.0;
  double peri = 0.0;

  for (int n = 0; n < num_faces; n++) {

    // Indices of current and next vertex
    int mu1 = n;
    int mu2 = n + 1;

    // Wrap back to first vertex for final term
    if (mu2 == num_faces) { mu2 = 0; }

    // Sum the area contribution
    area += 0.5 * (vert[mu1][0] * vert[mu2][1] - vert[mu1][1] * vert[mu2][0]);
    peri += sqrt(pow(vert[mu2][0] - vert[mu1][0], 2.0) + pow(vert[mu2][1] - vert[mu1][1], 2.0));
  }

  celli_geom[0] = area;
  celli_geom[1] = peri;
  celli_geom[2] = 0.5 * pow((area - 1), 2.0) + 0.5 * kp * pow((peri - p0), 2.0);
}

/*~~~~~~~~~~~~~~~~~~~Function 3: Get common neighs for chosen bonded pair of atoms~~~~~~~~~~~~~~~~*/

int FixTriDynamic::get_comm_neigh(int *common_neighs, tagint *celli_neighs, tagint *cellj_neighs,
                                  int num_celli_neighs, int num_cellj_neighs)
{

  int num_comm = 0;

  for (int n = 0; n < num_celli_neighs; n++) {
    for (int m = 0; m < num_cellj_neighs; m++) {
      if (celli_neighs[n] == cellj_neighs[m]) {
        num_comm += 1;
        if (num_comm == 3) { return num_comm; }
        common_neighs[num_comm - 1] = celli_neighs[n];
      }
    }
  }

  return num_comm;
}

/*~~~~~~~~~~~~~~~~~~~ Function 4: see if cells k and l are already bonded or not ~~~~~~~~~~~~~~~~*/

int FixTriDynamic::unbonded(tagint *cellk_neighs, int num_cellk_neighs, tagint tag_cell_l)
{
  for (int n = 0; n < num_cellk_neighs; n++) {
    if (cellk_neighs[n] == tag_cell_l) { return 0; }
  }
  return 1;
}

/*~~~~~~~~~~~~~~~~~~~ Function 5: Add and remove neighbors ~~~~~~~~~~~~~~~~*/

void FixTriDynamic::remove_neigh(int celli, tagint tag_neigh, tagint *celli_neighs,
                                 int num_celli_neighs, int max_neighs)
{
  int pos_removed = -1;

  for (int n = 0; n < num_celli_neighs; n++) {
    if (celli_neighs[n] == tag_neigh) {
      pos_removed = n;
      break;
    }
  }

  //Now shift all the elements after pos_removed back
  for (int n = pos_removed; n < max_neighs; n++) {
    if (n < num_celli_neighs - 1) {
      celli_neighs[n] = celli_neighs[n + 1];
    } else {
      celli_neighs[n] = 0;
    }
  }
}

/*~~~~~~~~~~~~~~~~~~~ Function 6: Print neighs list as required ~~~~~~~~~~~~~~~~*/

void FixTriDynamic::print_neighs_list(tagint *cell_neighs, int num_cell_neighs, tagint cell)
{

  printf("\n%d--->", cell);
  for (int n = 0; n < num_cell_neighs; n++) { printf("%d,  ", cell_neighs[n]); }
  printf("\n");
}

/* ---------------------------------------------------------------------- */
/*<<<<<<<<<<<<<<<<<<<<<< HELPER FUNCTIONS (END) >>>>>>>>>>>>>>>>>>>>>>>>>*/
/* ---------------------------------------------------------------------- */

/* ---------------------------------------------------------------------- */

void FixTriDynamic::post_force_respa(int vflag, int ilevel, int /*iloop*/)
{
  if (ilevel == ilevel_respa) post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixTriDynamic::min_post_force(int vflag)
{
  post_force(vflag);
}

/*------------------------------------------------------------------------*/

int FixTriDynamic::pack_forward_comm(int n, int *list, double *buf, int /*pbc_flag*/, int * /*pbc*/)

{
  int i, j, k, m;

  m = 0;

  if (commflag == 1) {
    for (i = 0; i < n; i++) {
      j = list[i];
      for (k = 0; k < 3; k++) { buf[m++] = cell_shape[j][k]; }
    }
  } else if (commflag == 2) {
    int tmp1, tmp2;
    index = atom->find_custom("neighs", tmp1, tmp2);
    tagint **neighs = atom->iarray[index];
    for (i = 0; i < n; i++) {
      j = list[i];
      for (k = 0; k < neighs_MAX; k++) { buf[m++] = neighs[j][k]; }
    }
  }
  return m;
}

void FixTriDynamic::unpack_forward_comm(int n, int first, double *buf)
{
  int i, j, m, last;

  m = 0;
  last = first + n;

  if (commflag == 1) {
    for (i = first; i < last; i++) {
      for (j = 0; j < 3; j++) { cell_shape[i][j] = buf[m++]; }
    }
  } else if (commflag == 2) {
    int tmp1, tmp2;
    index = atom->find_custom("neighs", tmp1, tmp2);
    tagint **neighs = atom->iarray[index];
    for (i = first; i < last; i++) {
      for (j = 0; j < neighs_MAX; j++) { neighs[i][j] = buf[m++]; }
    }
  }
}

/* ----------------------------------------------------------------------
   pack values in local atom-based arrays for exchange with another proc
------------------------------------------------------------------------- */

int FixTriDynamic::pack_exchange(int i, double *buf)
{
  int n = 0;
  if (peratom_flag) {
    for (int m = 0; m < size_peratom_cols; m++) buf[n++] = array_atom[i][m];
  }
  return n;
}

/* ----------------------------------------------------------------------
   unpack values into local atom-based arrays after exchange
------------------------------------------------------------------------- */

int FixTriDynamic::unpack_exchange(int nlocal, double *buf)
{
  int n = 0;
  if (peratom_flag) {
    for (int m = 0; m < size_peratom_cols; m++) array_atom[nlocal][m] = buf[n++];
  }
  return n;
}

/* ----------------------------------------------------------------------
   allocate local atom-based arrays
------------------------------------------------------------------------- */

void FixTriDynamic::grow_arrays(int nmax)
{
  if (peratom_flag) {
    memory->grow(array_atom, nmax, size_peratom_cols, "fix_tridynamic:array_atom");
  }
}

/* ----------------------------------------------------------------------
   initialize one atom's array values, called when atom is created
------------------------------------------------------------------------- */

void FixTriDynamic::set_arrays(int i)
{
  if (peratom_flag) {
    for (int m = 0; m < size_peratom_cols; m++) array_atom[i][m] = 0;
  }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixTriDynamic::memory_usage()
{
  int maxatom = atom->nmax;
  double bytes = (double) maxatom * 4 * sizeof(double);
  return bytes;
}

/*--------------------------------------------------------------------
                        END OF MAIN CODE
---------------------------------------------------------------------*/
