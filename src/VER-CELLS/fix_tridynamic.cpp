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
  if (narg < 14) error->all(FLERR, "Illegal fix TriDynamic command: not sufficient args");

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
  KT = utils::numeric(FLERR, arg[13], false, lmp);

  /*This fix takes in input as per-atom array
  produced by compute voronoi*/

  vcompute = modify->get_compute_by_id(id_compute_voronoi);
  if (!vcompute)
    error->all(FLERR, "Could not find compute ID {} for voronoi compute", id_compute_voronoi);

  //parse values for optional arguments
  nevery = 1;    // Using default value for now

  if (narg > 14) {
    idregion = utils::strdup(arg[14]);
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

  // //DEBUGGER
  // for (int i = 0; i < nlocal; i++){
  //   printf("vertices for cell %d--> ", tag[i]);
  //   for (int j = 0; j < neighs_MAX*2; j++){
  //     printf("%f, ", array_atom[i][j]);
  //   }
  //   printf("\n\n");
  // }
  // //DEBUGGER

  //forward communicate the cell shape data
  commflag = 1;
  comm->forward_comm(this, 3);

  /****: Do the force calculation based on current toplogy****/

  double Jac = 1.0 / 3.0;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {

      //corrds of cell i
      double x0 = x[i][0];
      double y0 = x[i][1];

      double F_t1[2] = {0.0};
      double F_t2[2] = {0.0};

      //Force contri from neighs
      for (int j = 0; j < num_neighs[i]; j++) {

        int current_cell = domain->closest_image(i, atom->map(neighs[i][j]));   

        //coords of cell j
        double x1 = x[current_cell][0];
        double y1 = x[current_cell][1];

        int num_neighsj = num_neighs[current_cell];

        int cyclic_neighsj[num_neighsj+4];

        cyclic_neighsj[0] = neighs[current_cell][num_neighsj-2];
        cyclic_neighsj[1] = neighs[current_cell][num_neighsj-1];

        for (int n = 2; n < num_neighsj+2; n++){
          cyclic_neighsj[n] = neighs[current_cell][n-2];
        }

        cyclic_neighsj[num_neighsj+2] = neighs[current_cell][0];
        cyclic_neighsj[num_neighsj+3] = neighs[current_cell][1];

        // First term values needed
        double ai = cell_shape[current_cell][0];
        double elasticity_area = (1 / 2.0) * (ai - 1);

        // Second term values needed
        double pi = cell_shape[current_cell][1];
        double elasticity_peri = kp * (pi - p0);

        /*Now there are 2 vertices shared by cell i and j*/
        //Find those two vertices:

        int DT[2][3];    //each row is a triangle (local ids) and hence a vertex
        DT[0][0] = i;
        DT[1][0] = i;
        DT[0][1] = current_cell;
        DT[1][1] = current_cell;
        int tri_prev[2][3]; 
        int tri_next[2][3];

        for (int n = 2; n < num_neighsj + 2; n++) {
          if (cyclic_neighsj[n] == tag[i]) {
            DT[0][2] = domain->closest_image(current_cell, atom->map(cyclic_neighsj[n - 1]));
            tri_prev[0][0] = current_cell;
            tri_prev[0][1] = DT[0][2];
            tri_prev[0][2] = domain->closest_image(current_cell, atom->map(cyclic_neighsj[n - 2]));

            tri_next[0][0] = current_cell;
            tri_next[0][1] = i;
            tri_next[0][2] = domain->closest_image(current_cell, atom->map(cyclic_neighsj[n + 1]));

            DT[1][2] = domain->closest_image(current_cell, atom->map(cyclic_neighsj[n + 1]));
            tri_prev[1][0] = current_cell;
            tri_prev[1][1] = i;
            tri_prev[1][2] = domain->closest_image(current_cell, atom->map(cyclic_neighsj[n - 1]));

            tri_next[1][0] = current_cell;
            tri_next[1][1] = DT[1][2];
            tri_next[1][2] = domain->closest_image(current_cell, atom->map(cyclic_neighsj[n + 2]));
            break;
          }
        }

        double vertex_force_sum_t1[2] = {0.0};
        double vertex_force_sum_t2[2] = {0.0};

        //Now loop through both the vertices

        for (int nu = 0; nu < 2; nu++) {

          //Coords of cell k
          double x2 = x[DT[nu][2]][0];
          double y2 = x[DT[nu][2]][1];
          
          //Coords of the vertex
          double vert[2] = {0.0};
          vert[0] = (x0 + x1 + x2) / 3.0;
          vert[1] = (y0 + y1 + y2) / 3.0;

          /*Find coords of vert_next and vert_prev*/

          //Next and prev vertices for neighboring atom j
          double vert_prev[2] = {0.0};
          vert_prev[0] = (x[tri_prev[nu][0]][0] + x[tri_prev[nu][1]][0] + x[tri_prev[nu][2]][0]) / 3.0; 
          vert_prev[1] = (x[tri_prev[nu][0]][1] + x[tri_prev[nu][1]][1] + x[tri_prev[nu][2]][1]) / 3.0;

          double vert_next[2] = {0.0};
          vert_next[0] = (x[tri_next[nu][0]][0] + x[tri_next[nu][1]][0] + x[tri_next[nu][2]][0]) / 3.0;
          vert_next[1] = (x[tri_next[nu][0]][1] + x[tri_next[nu][1]][1] + x[tri_next[nu][2]][1]) / 3.0;

          //first term stuff

          double rprevnext[3] = {vert_next[0] - vert_prev[0], vert_next[1] - vert_prev[1], 0.0};
          double cp[3] = {0.0};
          double N[3] = {0, 0, 1};    // normal vector to the plane of cell layer (2D)
          getCP(cp, rprevnext, N);

          //second term stuff
          double rcurrprev[2] = {vert[0] - vert_prev[0], vert[1] - vert_prev[1]};
          double rnextcurr[2] = {vert_next[0] - vert[0], vert_next[1] - vert[1]};
          normalize(rcurrprev);
          normalize(rnextcurr);
          double rhatdiff_t2[2] = {rcurrprev[0] - rnextcurr[0], rcurrprev[1] - rnextcurr[1]};

          // Term 1 forces
          vertex_force_sum_t1[0] += cp[0] * Jac;
          vertex_force_sum_t1[1] += cp[1] * Jac;

          // Term 2 forces
          vertex_force_sum_t2[0] += rhatdiff_t2[0] * Jac;
          vertex_force_sum_t2[1] += rhatdiff_t2[1] * Jac;
        }

        F_t1[0] += elasticity_area * vertex_force_sum_t1[0];
        F_t1[1] += elasticity_area * vertex_force_sum_t1[1];
        F_t2[0] += elasticity_peri * vertex_force_sum_t2[0];
        F_t2[1] += elasticity_peri * vertex_force_sum_t2[1];
      }

      /*Force contribution from self*/

      double vertex_force_sum_t1[2] = {0.0};
      double vertex_force_sum_t2[2] = {0.0};

      // First term values needed
      double ai = cell_shape[i][0];
      double elasticity_area = (1 / 2.0) * (ai - 1);

      // Second term values needed
      double pi = cell_shape[i][1];
      double elasticity_peri = kp * (pi - p0);

      //Find vertices of cell i
      double vertices_i_x[num_neighs[i]] = {0.0};
      double vertices_i_y[num_neighs[i]] = {0.0};

      for (int n = 0; n < num_neighs[i]; n++) {
        vertices_i_x[n] = array_atom[i][2 * n];
        vertices_i_y[n] = array_atom[i][2 * n + 1];
      }

      //Loop through the vertices

      for (int nu = 0; nu < num_neighs[i]; nu++) {
        double vert[2] = {0.0};
        double vert_next[2] = {0.0};
        double vert_prev[2] = {0.0};

        vert[0] = array_atom[i][2 * nu];
        vert[1] = array_atom[i][2 * nu + 1];

        if (nu == 0) {
          vert_next[0] = array_atom[i][2 * (nu + 1)];
          vert_next[1] = array_atom[i][2 * (nu + 1) + 1];
          vert_prev[0] = array_atom[i][2 * (num_neighs[i] - 1)];
          vert_prev[1] = array_atom[i][2 * (num_neighs[i] - 1) + 1];
        } else if (nu == num_neighs[i] - 1) {
          vert_next[0] = array_atom[i][0];
          vert_next[1] = array_atom[i][1];
          vert_prev[0] = array_atom[i][2 * (nu - 1)];
          vert_prev[1] = array_atom[i][2 * (nu - 1) + 1];
        } else {
          vert_next[0] = array_atom[i][2 * (nu + 1)];
          vert_next[1] = array_atom[i][2 * (nu + 1) + 1];
          vert_prev[0] = array_atom[i][2 * (nu - 1)];
          vert_prev[1] = array_atom[i][2 * (nu - 1) + 1];
        }

        //first term stuff
        double rprevnext[3] = {vert_next[0] - vert_prev[0], vert_next[1] - vert_prev[1], 0.0};
        double cp[3] = {0.0};
        double N[3] = {0, 0, 1};    // normal vector to the plane of cell layer (2D)
        getCP(cp, rprevnext, N);

        //second term stuff
        double rcurrprev[2] = {vert[0] - vert_prev[0], vert[1] - vert_prev[1]};
        double rnextcurr[2] = {vert_next[0] - vert[0], vert_next[1] - vert[1]};
        normalize(rcurrprev);
        normalize(rnextcurr);
        double rhatdiff_t2[2] = {rcurrprev[0] - rnextcurr[0], rcurrprev[1] - rnextcurr[1]};

        // Term 1 forces
        vertex_force_sum_t1[0] += cp[0] * Jac;
        vertex_force_sum_t1[1] += cp[1] * Jac;

        // Term 2 forces
        vertex_force_sum_t2[0] += rhatdiff_t2[0] * Jac;
        vertex_force_sum_t2[1] += rhatdiff_t2[1] * Jac;
      }

      F_t1[0] += elasticity_area * vertex_force_sum_t1[0];
      F_t1[1] += elasticity_area * vertex_force_sum_t1[1];
      F_t2[0] += elasticity_peri * vertex_force_sum_t2[0];
      F_t2[1] += elasticity_peri * vertex_force_sum_t2[1];

      double fx = -F_t1[0] - F_t2[0];
      double fy = -F_t1[1] - F_t2[1];

      f[i][0] += fx;
      f[i][1] += fy;
      f[i][2] += 0.0;
    }
  }


  /**** Implement monte-carlo algorithm to update the network topology for next time step****/

  //First find the number of MC steps/iterations: N_mc = total neighs for all owned atoms accross procs/2

  int Num_neighs_this = 0;    //local value
  for (int i = 0; i < nlocal; i++) { Num_neighs_this += num_neighs[i]; }

  int Num_mc_steps = Num_neighs_this / 2;
  srand(time(NULL) + me);    //random seed for each rank

  //Start Monte-Carlo

  int num_swaps_rejected = 0;
  double dummy[neighs_MAX * 2];

  for (int n = 0; n < Num_mc_steps; n++) {

    //allocate memory for storing tags of cells in the Quad in this mc step and some other info
    tagint cell_tags[4] = {0};

    int cell_i = rand() % nlocal;    //return a random integer between [0, nlocal-1]

    //error check for num_neighs_celli-->behaving a bit wierdly! when full
    if (num_neighs[cell_i] == 0) {
      print_neighs_list(neighs[cell_i], neighs_MAX, tag[cell_i]);
      error->one(FLERR, "no of neighs for atom is 0");
    }

    int rand_neigh_idx =
        rand() % num_neighs[cell_i];    //return a random neigh index [0, num_neighs_i - 1]
    int tmpj = atom->map(neighs[cell_i][rand_neigh_idx]);
    int cell_j = domain->closest_image(cell_i, tmpj);

    //check 1
    if (num_neighs[cell_i] <= 4 || num_neighs[cell_j] <= 4) {
      continue;    //Not a valid candidate for swapping
    }

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

    //Check 2 (for num of common neighs)
    if (num_comm_neighs != 2) { continue; }

    //get the local ids using closest image
    int cell_k = domain->closest_image(cell_i, atom->map(comm_neigh[0]));
    int cell_l = domain->closest_image(cell_i, atom->map(comm_neigh[1]));

    //Check 3: See if the quadrialteral formed is convex or concave:
    //i->k->j->l is the order of points of quadralteral in a cyclic manner

    // //DEBUGGER
    // cell_i = 57;
    // cell_j = 48;
    // cell_k = 67;
    // cell_l = 36;

    // print_neighs_list(neighs[cell_i], num_neighs[cell_i], tag[cell_i]);
    // print_neighs_list(neighs[cell_j], num_neighs[cell_j], tag[cell_j]);
    // print_neighs_list(neighs[cell_k], num_neighs[cell_k], tag[cell_k]);
    // print_neighs_list(neighs[cell_l], num_neighs[cell_l], tag[cell_l]);
    // //DEBUGGER

    double p1[2] = {x[cell_i][0], x[cell_i][1]};
    double p2[2] = {x[cell_k][0], x[cell_k][1]};
    double p3[2] = {x[cell_j][0], x[cell_j][1]};
    double p4[2] = {x[cell_l][0], x[cell_l][1]};

    //If quadrilateral formed is concave then resultant tiling not valid
    if (isConcave(p1, p2, p3, p4)) {
      // //DEBUGGER
      // printf("\n quad formed is concave, hence rejected \n");
      // //DEBUGGER
      continue;
    }

    cell_tags[0] = tag[cell_i];
    cell_tags[1] = tag[cell_j];
    cell_tags[2] = tag[cell_k];
    cell_tags[3] = tag[cell_l];

    // //DEBUGGER
    // printf("\n Quad selected--> %d,  %d,  %d,  %d", cell_tags[0], cell_tags[1], cell_tags[2], cell_tags[3]);
    // //DEBUGGER

    //a flag variable that tells whether k and l bonded or not: 1 if unbonded and 0 if bonded
    int flag_unbonded = unbonded(neighs[cell_k], num_neighs[cell_k], tag[cell_l]);

    //Store the local indices of 'owned' atoms

    int owned_i = atom->map(cell_tags[0]);
    int owned_j = atom->map(cell_tags[1]);
    int owned_k = atom->map(cell_tags[2]);
    int owned_l = atom->map(cell_tags[3]);

    //store energy before swap
    double E_before = cell_shape[owned_i][2] + cell_shape[owned_j][2] + cell_shape[owned_k][2] +
        cell_shape[owned_l][2];

    // //Debugger
    // printf("\n Ebefore = %f", E_before);
    // //DEBUGGER

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
      // get_cell_data(array_atom[owned_atom], cell_shape[owned_atom], neighs[owned_atom],
      //               num_neighs[owned_atom], owned_atom);
      get_cell_data(dummy, cell_shape[owned_atom], neighs[owned_atom], num_neighs[owned_atom],
                    owned_atom);
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

    // //Debugger
    // printf("  Eafter \n = %f", E_after);
    // //DEBUGGER

    double Delta_E = E_after - E_before;
    int flag_swapped = 1;

    //Metropolis scheme

    if (Delta_E < 0) {
      continue;
    } else {
      double Prob = exp(-1.0 * Delta_E / KT);
      double rand_num = (double) rand() / RAND_MAX;    //generate a random no between 0 and 1
      if (Prob > rand_num) {
        continue;    //accept the swap with this probability
      } else {
        flag_swapped = 0;    //swapping attempt failed as not energetically favoured
        num_swaps_rejected += 1;
        //DEBUGGER
        //printf("\nswapping was rejected\n");
        //DEBUGGER
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
        // get_cell_data(array_atom[owned_atom], cell_shape[owned_atom], neighs[owned_atom],
        //               num_neighs[owned_atom], owned_atom);
        get_cell_data(dummy, cell_shape[owned_atom], neighs[owned_atom], num_neighs[owned_atom],
                      owned_atom);
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

  printf("\n Neighs List %d--->", cell);
  for (int n = 0; n < num_cell_neighs; n++) { printf("%d,  ", cell_neighs[n]); }
  printf("\n");
}

/*~~~~~~~~~~~~~~~~~~~ Function 7: check if the quad is convex or concave ~~~~~~~~~~~~~~~~*/

bool FixTriDynamic::isConcave(double *p1, double *p2, double *p3, double *p4)
{
  // Compute cross products for each consecutive triplet of points
  double cross1 = crossProduct(p1, p2, p3);
  double cross2 = crossProduct(p2, p3, p4);
  double cross3 = crossProduct(p3, p4, p1);
  double cross4 = crossProduct(p4, p1, p2);

  // Check if all cross products have the same sign
  if ((cross1 > 0 && cross2 > 0 && cross3 > 0 && cross4 > 0) ||
      (cross1 < 0 && cross2 < 0 && cross3 < 0 && cross4 < 0)) {
    return false;    // Convex
  }
  return true;    // Concave
}

/*~~~~~~~~~~~~~~~~~~~ Function 8: cross product ~~~~~~~~~~~~~~~~*/

double FixTriDynamic::crossProduct(double *p1, double *p2, double *p3)
{
  double cross = (p2[0] - p1[0]) * (p3[1] - p2[1]) - (p3[0] - p2[0]) * (p2[1] - p1[1]);
  return cross;
}

/*~~~~~~~~~~~~~~~~~~~~~~FUnction 9: get cross product~~~~~~~~~~~~~~~~*/

void FixTriDynamic::getCP(double *cp, double *v1, double *v2)
{
  cp[0] = v1[1] * v2[2] - v1[2] * v2[1];
  cp[1] = v1[2] * v2[0] - v1[0] * v2[2];
  cp[2] = v1[0] * v2[1] - v1[1] * v2[0];
}

/*~~~~~~~~~~~~~~~~~~~~~Function 10: normalize a vector*/

// helper function: normalizes a vector

void FixTriDynamic::normalize(double *v)
{
  double norm = pow(pow(v[0], 2) + pow(v[1], 2) + pow(v[2], 2), 0.5);
  v[0] = v[0] / norm;
  v[1] = v[1] / norm;
  v[2] = v[2] / norm;
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
