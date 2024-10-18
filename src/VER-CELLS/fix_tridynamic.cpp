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
    Fix(lmp, narg, arg), id_compute_voronoi(nullptr), cell_shape(nullptr), vertices(nullptr),
    x_updated(nullptr), wgn(nullptr), idregion(nullptr), region(nullptr)
{
  if (narg < 19) error->all(FLERR, "Illegal fix TriDynamic command: not sufficient args");

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
  eta_0 = utils::numeric(FLERR, arg[12], false, lmp);
  KT = utils::numeric(FLERR, arg[13], false, lmp);
  n_every_output = utils::numeric(FLERR, arg[14], false, lmp);
  Num_MC_input = utils::numeric(FLERR, arg[15], false, lmp);
  c1 = utils::numeric(FLERR, arg[16], false, lmp);
  c2 = utils::numeric(FLERR, arg[17], false, lmp);
  T1_threshold = utils::numeric(FLERR, arg[18], false, lmp);

  /*This fix takes in input a per-atom array
  produced by compute voronoi*/

  vcompute = modify->get_compute_by_id(id_compute_voronoi);
  if (!vcompute)
    error->all(FLERR, "Could not find compute ID {} for voronoi compute", id_compute_voronoi);

  //parse values for optional arguments
  nevery = 1;    // Using default value for now

  if (narg > 19) {
    idregion = utils::strdup(arg[19]);
    region = domain->get_region_by_id(idregion);
  }

  // Initialize nmax and virial pointer
  nmax = atom->nmax;
  cell_shape = nullptr;
  vertices = nullptr;
  x_updated = nullptr;

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
  memory->destroy(vertices);
  memory->destroy(x_updated);

  // unregister callbacks to this fix from atom class
  if (peratom_flag) { atom->delete_callback(id, Atom::GROW); }

  if (new_fix_id && modify->nfix) modify->delete_fix(new_fix_id);
  delete[] new_fix_id;

  //fclose(fp); //no writing of file is required
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

  // Populate neighs array with global ids of the voronoi neighs (initial configuration)

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
  //same for vertices array
  //allocate memory at once instead of every time step

  nmax = atom->nmax;
  memory->create(cell_shape, nmax, 3, "fix_tridynamic:cell_shape");
  memory->create(vertices, nmax, neighs_MAX * 2, "fix_tridynamic:vertices");
  memory->create(x_updated, nmax, 2, "fix_tridynamic:x_updated");

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

  double **x = atom->x;    //This is x_n
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

  tagint **neighs = atom->iarray[index];    //This is T_n

  // Determine the number of neighs for all atoms (owned + ghost)
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

  for (int i = 0; i < nlocal; i++) { arrange_cyclic(neighs[i], num_neighs[i], i, x); }

  // Communicate neighs (to have ordered neighs list of ghost atoms as well)
  commflag = 2;
  comm->forward_comm(this, neighs_MAX);

  // // DEBUGGER
  // std::string file0 = "Neighs_list.txt";          // + std::to_string(0) + ".txt";
  // std::string file1 = "Energy_montecarlo.txt";    // + std::to_string(0) + ".txt";
  // string filenames[2] = {file0, file1};
  // ofstream fp[2];

  // // To write output to files->first create them:
  // filenames[0] = {file0};
  // if (update->ntimestep % n_every_output == 0) {
  //   fp[0].open(filenames[0].c_str(), std::ios_base::app);
  //   fp[0] << "----------- timestep: " << update->ntimestep << "--------------" << endl;
  //   for (int i = 0; i < nlocal; i++) {
  //     fp[0] << tag[i] << "--->";
  //     for (int j = 0; j < num_neighs[i]; j++) { fp[0] << neighs[i][j] << ", "; }
  //     fp[0] << endl;
  //   }
  //   fp[0] << endl << endl;
  // }
  // //DEBUGGER

  /**** STEP 2: Get cell shape info (area, perimeter, energy) in the current topology ****/

  // Possibly resize arrays
  if (atom->nmax > nmax) {
    memory->destroy(cell_shape);
    memory->destroy(vertices);
    nmax = atom->nmax;
    memory->create(cell_shape, nmax, 3, "fix_tridynamic:cell_shape");
    memory->create(vertices, nmax, neighs_MAX * 2, "fix_tridynamic:vertices");
    memory->create(x_updated, nmax, 2, "fix_tridynamic:x_updated");
  }

  // Initialize arrays to zero
  for (int i = 0; i < nall; i++) {
    for (int j = 0; j < 3; j++) { cell_shape[i][j] = 0.0; }
  }

  // Reset array-atom if outputting
  // Initialize vertices info as well

  if (peratom_flag) {
    for (int i = 0; i < nlocal; i++) {
      for (int j = 0; j < neighs_MAX * 2; j++) {
        array_atom[i][j] = 0.0;
        vertices[i][j] = 0.0;
      }
    }
  }

  //Calculate areas, perimeters and energy of nlocal atoms and populate vertices array as well

  for (int i = 0; i < nlocal; i++) {
    get_cell_data(vertices[i], cell_shape[i], neighs[i], num_neighs[i],
                  i, x);    //changed from array_atom to vertices
  }

  //forward communicate the cell shape data
  commflag = 1;
  comm->forward_comm(this, 3);

  // Store topology (xn, Tn) for outputting at the current step (this is before updating the connectivity)

  // Write data for outputting and plotting
  if (peratom_flag) {
    for (int i = 0; i < nlocal; i++) {
      for (int j = 0; j < neighs_MAX * 2; j++) { array_atom[i][j] = vertices[i][j]; }
    }
  }

  /****** Find forces f_n based on x_n and T_n now *******/

  double Jac = 1.0 / 3.0;

  double eta[nlocal] = {eta_0}; 

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {

      //corrds of cell i
      double x0 = x[i][0];
      double y0 = x[i][1];

      double F_t1[2] = {0.0};
      double F_t2[2] = {0.0};

      //Force contri from neighs
      for (int j = 0; j < num_neighs[i]; j++) {

        int current_neigh = domain->closest_image(i, atom->map(neighs[i][j]));

        //coords of cell j
        double x1 = x[current_neigh][0];
        double y1 = x[current_neigh][1];

        int num_neighsj = num_neighs[current_neigh];

        int cyclic_neighsj[num_neighsj + 4];

        cyclic_neighsj[0] = neighs[current_neigh][num_neighsj - 2];
        cyclic_neighsj[1] = neighs[current_neigh][num_neighsj - 1];

        for (int n = 2; n < num_neighsj + 2; n++) {
          cyclic_neighsj[n] = neighs[current_neigh][n - 2];
        }

        cyclic_neighsj[num_neighsj + 2] = neighs[current_neigh][0];
        cyclic_neighsj[num_neighsj + 3] = neighs[current_neigh][1];

        // First term values needed
        double ai = cell_shape[current_neigh][0];
        double elasticity_area = (1.0 / 2.0) * (ai - 1);

        // Second term values needed
        double pi = cell_shape[current_neigh][1];
        double elasticity_peri = kp * (pi - p0);

        /*Now there are 2 vertices shared by cell i and j*/

        double nu[2][2] = {0.0};    //each row is a vertex and columns are nu_x and nu_y
        double nu_prev[2][2] = {0.0};
        double nu_next[2][2] = {0.0};

        int k = 2;
        while (cyclic_neighsj[k] != tag[i]) {
          k++;
          if (k == num_neighsj + 2) {
            error->one(FLERR,
                       " \n Inside force code: cell j did not find cell i as its neighbor \n");
          }
        }

        int cell_l = domain->closest_image(current_neigh, atom->map(cyclic_neighsj[k - 1]));
        int cell_k = domain->closest_image(current_neigh, atom->map(cyclic_neighsj[k + 1]));

        //error check
        int cell_l_from_i = domain->closest_image(i, atom->map(cyclic_neighsj[k - 1]));
        int cell_k_from_i = domain->closest_image(i, atom->map(cyclic_neighsj[k + 1]));

        if (cell_l != cell_l_from_i || cell_k != cell_k_from_i) {
          error->one(FLERR,
                     "Inside force code: nearest images of common neighbors for cell i and j do "
                     "not match");
        }

        /* for nu1 ---> (j,i,l): */
        nu[0][0] = (x[i][0] + x[current_neigh][0] + x[cell_l][0]) / 3.0;
        nu[0][1] = (x[i][1] + x[current_neigh][1] + x[cell_l][1]) / 3.0;

        //nu1_prev ---> (j,l,n)
        int cell_n = domain->closest_image(current_neigh, atom->map(cyclic_neighsj[k - 2]));
        nu_prev[0][0] = (x[current_neigh][0] + x[cell_l][0] + x[cell_n][0]) / 3.0;
        nu_prev[0][1] = (x[current_neigh][1] + x[cell_l][1] + x[cell_n][1]) / 3.0;

        //nu1_next ---> (j,i,k)
        nu_next[0][0] = (x[current_neigh][0] + x[i][0] + x[cell_k][0]) / 3.0;
        nu_next[0][1] = (x[current_neigh][1] + x[i][1] + x[cell_k][1]) / 3.0;

        /* for nu2 ---> (k,i,l) */
        nu[1][0] = (x[i][0] + x[current_neigh][0] + x[cell_k][0]) / 3.0;
        nu[1][1] = (x[i][1] + x[current_neigh][1] + x[cell_k][1]) / 3.0;

        //nu1_prev ---> (j,i,l)

        nu_prev[1][0] = (x[current_neigh][0] + x[i][0] + x[cell_l][0]) / 3.0;
        nu_prev[1][1] = (x[current_neigh][1] + x[i][1] + x[cell_l][1]) / 3.0;

        //nu1_next ---> (j,k,m)
        int cell_m = domain->closest_image(current_neigh, atom->map(cyclic_neighsj[k + 2]));
        nu_next[1][0] = (x[current_neigh][0] + x[cell_k][0] + x[cell_m][0]) / 3.0;
        nu_next[1][1] = (x[current_neigh][1] + x[cell_k][1] + x[cell_m][1]) / 3.0;

        double vertex_force_sum_t1[2] = {0.0};
        double vertex_force_sum_t2[2] = {0.0};

        //Now loop through both the vertices

        for (int n = 0; n < 2; n++) {

          //first term stuff

          double r_next_prev[3] = {nu_next[n][0] - nu_prev[n][0], nu_next[n][1] - nu_prev[n][1],
                                   0.0};
          double cp[3] = {0.0};
          double N[3] = {0, 0, 1};    // normal vector to the plane of cell layer (2D)
          getCP(cp, r_next_prev, N);

          //second term stuff
          double r_curr_prev[2] = {nu[n][0] - nu_prev[n][0], nu[n][1] - nu_prev[n][1]};
          double r_next_curr[2] = {nu_next[n][0] - nu[n][0], nu_next[n][1] - nu[n][1]};
          normalize(r_curr_prev);
          normalize(r_next_curr);
          double r_hatdiff_t2[2] = {r_curr_prev[0] - r_next_curr[0],
                                    r_curr_prev[1] - r_next_curr[1]};

          // Term 1 forces
          vertex_force_sum_t1[0] += cp[0] * Jac;
          vertex_force_sum_t1[1] += cp[1] * Jac;

          // Term 2 forces
          vertex_force_sum_t2[0] += r_hatdiff_t2[0] * Jac;
          vertex_force_sum_t2[1] += r_hatdiff_t2[1] * Jac;
        }

        F_t1[0] += elasticity_area * vertex_force_sum_t1[0];
        F_t1[1] += elasticity_area * vertex_force_sum_t1[1];
        F_t2[0] += elasticity_peri * vertex_force_sum_t2[0];
        F_t2[1] += elasticity_peri * vertex_force_sum_t2[1];
      }

      /*~~~~~~~~~~~~~~~~~ Force contribution from self ~~~~~~~~~~~~~~~~~*/

      double vertex_force_sum_t1[2] = {0.0};
      double vertex_force_sum_t2[2] = {0.0};

      // First term values needed
      double ai = cell_shape[i][0];
      double elasticity_area = (1.0 / 2.0) * (ai - 1); //ka/2

      // Second term values needed
      double pi = cell_shape[i][1];
      double elasticity_peri = kp * (pi - p0);

      //Find vertices of cell i
      double vertices_i_x[num_neighs[i]] = {0.0};
      double vertices_i_y[num_neighs[i]] = {0.0};

      //Also find the minimum distance of cell from its bonds
      double dist_bond;
      double min_dist_bond = INFINITY; //initialize to infinity

      int cell2, cell3;  //store the ids of neighs for which bond distance is minimum

      //Loop through the vertices
      for (int n = 0; n < num_neighs[i]; n++) {
        double vert[2] = {0.0};
        double vert_next[2] = {0.0};
        double vert_prev[2] = {0.0};

        vert[0] = vertices[i][2 * n];
        vert[1] = vertices[i][2 * n + 1];

        int neigh_j1 = domain->closest_image(i, atom->map(neighs[i][n])); //local id of the neigh
        int neigh_j2;

        if (n == 0) {
          vert_next[0] = vertices[i][2 * (n + 1)];
          vert_next[1] = vertices[i][2 * (n + 1) + 1];
          vert_prev[0] = vertices[i][2 * (num_neighs[i] - 1)];
          vert_prev[1] = vertices[i][2 * (num_neighs[i] - 1) + 1];
          neigh_j2 = domain->closest_image(i, atom->map(neighs[i][1]));
        } else if (n == num_neighs[i] - 1) {
          vert_next[0] = vertices[i][0];
          vert_next[1] = vertices[i][1];
          vert_prev[0] = vertices[i][2 * (n - 1)];
          vert_prev[1] = vertices[i][2 * (n - 1) + 1];
          neigh_j2 = domain->closest_image(i, atom->map(neighs[i][0]));
        } else {
          vert_next[0] = vertices[i][2 * (n + 1)];
          vert_next[1] = vertices[i][2 * (n + 1) + 1];
          vert_prev[0] = vertices[i][2 * (n - 1)];
          vert_prev[1] = vertices[i][2 * (n - 1) + 1];
          neigh_j2 = domain->closest_image(i, atom->map(neighs[i][n+1]));
        }

        //Viscosity stuff
        double xj1[2] = {x[neigh_j1][0], x[neigh_j1][1]};
        double xj2[2] = {x[neigh_j2][0], x[neigh_j2][1]};

        double A_line = xj1[1] - xj2[1];  //A = y1-y2
        double B_line = xj2[0] - xj1[0];  //B = x2-x1
        double C_line = xj1[0] * xj2[1] - xj2[0] * xj1[1];  //C = x1*y2 - x2*y1

        dist_bond = fabs(A_line * x[i][0] + B_line * x[i][1] + C_line) / sqrt(A_line*A_line + B_line*B_line);
        
        if (dist_bond < min_dist_bond){
          min_dist_bond = dist_bond;         //update the minimum value
          cell2 = atom->map(tag[neigh_j1]);  //store cell ids as well
          cell3 = atom->map(tag[neigh_j2]);
        }

        //first term stuff
        double r_next_prev[3] = {vert_next[0] - vert_prev[0], vert_next[1] - vert_prev[1], 0.0};
        double cp[3] = {0.0};
        double N[3] = {0, 0, 1};    // normal vector to the plane of cell layer (2D)
        getCP(cp, r_next_prev, N);

        //second term stuff
        double r_curr_prev[2] = {vert[0] - vert_prev[0], vert[1] - vert_prev[1]};
        double r_next_curr[2] = {vert_next[0] - vert[0], vert_next[1] - vert[1]};
        normalize(r_curr_prev);
        normalize(r_next_curr);
        double rhatdiff_t2[2] = {r_curr_prev[0] - r_next_curr[0], r_curr_prev[1] - r_next_curr[1]};

        // Term 1 forces
        vertex_force_sum_t1[0] += cp[0] * Jac;
        vertex_force_sum_t1[1] += cp[1] * Jac;

        // Term 2 forces
        vertex_force_sum_t2[0] += rhatdiff_t2[0] * Jac;
        vertex_force_sum_t2[1] += rhatdiff_t2[1] * Jac;
      }

      //Update the viscosities for celli and cell2 and cell3;
      //Always keep the maximum viscosity
      
      double eta_i = eta_0*(1 + c1*exp(-1.0 * c2 * min_dist_bond));
      
      if (eta_i > eta[i]){
        eta[i] = eta_i;
      }

      if (eta_i > eta[cell2]){
        eta[cell2] = eta_i;
      }

      if (eta_i > eta[cell3]){
        eta[cell3] = eta_i;
      }

      //vertex model forces

      F_t1[0] += elasticity_area * vertex_force_sum_t1[0];
      F_t1[1] += elasticity_area * vertex_force_sum_t1[1];
      F_t2[0] += elasticity_peri * vertex_force_sum_t2[0];
      F_t2[1] += elasticity_peri * vertex_force_sum_t2[1];

      /*self force contri end*/
      
      // Add all the force contributions
      double fx = -F_t1[0] - F_t2[0];
      double fy = -F_t1[1] - F_t2[1];

      f[i][0] += fx/eta[i];
      f[i][1] += fy/eta[i];
      f[i][2] = 0.0;
    }
  }


  //Initialize x_updated to x_n

  for (int i = 0; i < nall; i++) {
    x_updated[i][0] = x[i][0];
    x_updated[i][1] = x[i][1];
  }

  //Now find the updated positions x_n+1
  //here we update the positions to ensure that T_n+1 is compatible with x_n+1

  //The effect of viscosity has already been accounted for inside force
  //Fix brownian will now effectively update positions like this
  //This is more like the velocity now

  ///////////// COMMENTING OUT for debugging purposes only ////////////////

  for (int i = 0; i < nlocal; i++) {
    x_updated[i][0] = x[i][0] + f[i][0]  * dt;
    x_updated[i][1] = x[i][1] + f[i][1]  * dt;
    int imj_atom = sametag[i];
    while (imj_atom != -1) {
      x_updated[imj_atom][0] = x[imj_atom][0] + f[i][0]  * dt;
      x_updated[imj_atom][1] = x[imj_atom][1] + f[i][1]  * dt;
      imj_atom = sametag[imj_atom];
    }
  }

  ///////////// COMMENTING OUT for debugging purposes only ////////////////

  
  // Find the updated neighs list based on changed positions

  tagint neighs_rearr[nlocal][neighs_MAX];
  
  for (int i = 0; i < nlocal; i++) {
    for (int j = 0; j < neighs_MAX; j++) { neighs_rearr[i][j] = neighs[i][j]; }
    //arrange them cyclically based on x_updated
    arrange_cyclic(neighs_rearr[i], num_neighs[i], i, x_updated);
  }

  // Compare the original and rearranged neighs list to see if it is a cyclic permutation or not

  for (int i = 0; i < nlocal; i++) {
    if (is_cyclic_perm(neighs[i], neighs_rearr[i], num_neighs[i])) {
      continue;
    } else {
      print_neighs_list(neighs[i], num_neighs[i], i);
      print_neighs_list(neighs_rearr[i], num_neighs[i], i);
      error->one(FLERR, "\n !!!! The new neighs list (after cells positions have been updated) is not a cyclic permutation of the previous one !!!! \n");
    }
  }
  
  // Now that cells have moved, tiling has changed. Update cell-shape data based on x_updated

  for (int i = 0; i < nlocal; i++) {
    get_cell_data(vertices[i], cell_shape[i], neighs[i], num_neighs[i],
                  i, x_updated);    //changed from array_atom to vertices
  }

  // forward communicate the new cell shape data
  commflag = 1;
  comm->forward_comm(this, 3);

  // Update neighs list (now cyclically permuted based on x_updated)

  for (int i = 0; i < nlocal; i++){
    for (int j = 0; j < neighs_MAX; j++){
      neighs[i][j] = neighs_rearr[i][j];
    }
  }

  // Communicate neighs (to have ordered neighs list of ghost atoms as well)
  commflag = 2;
  comm->forward_comm(this, neighs_MAX);

 /////////////////////////////////////////////////////////////////////////////////////////////////////
 ////////////////////////////////////////////////////////////////////////////////////////////////////
  /*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~MONTE CARLO / Vertex model like T1 swap~~~~~~~~~~~~~~~~~~~~
  --> Implement monte-carlo algorithm to update the network topology for next time step
                             i.e.  obtain T_n+1 from x_n+1 and T_n
  --> We should think about incorporating timescale into it and relating it to physical params like p0
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~***/
 /////////////////////////////////////////////////////////////////////////////////////////////////////
 ////////////////////////////////////////////////////////////////////////////////////////////////////

   /*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
  /*We now need to ensure that x_n+1 and T_n+1 are compatible
  /*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/

  //First find the number of MC steps/iterations: N_mc = total neighs for all owned atoms accross procs/2

  int Num_neighs = 0;    //local value
  for (int i = 0; i < nlocal; i++) { Num_neighs += num_neighs[i]; }

  int Num_MC_bonds = Num_neighs /
      2;    //This says that the number of MC steps will be the same as number of bonds in the configuration.
  srand(time(NULL) + me);    //random seed for each rank

  //Start Monte-Carlo

  int num_swaps_rejected = 0;

  for (int n = 0; n < Num_MC_bonds; n++) {

    //allocate memory for storing tags of cells in the Quad in this mc step and some other info
    tagint cell_tags[4] = {0};

    int cell_i = rand() % nlocal;    //return a random integer between [0, nlocal-1]

    //error check for num_neighs_celli-->behaving a bit wierdly! when full
    if (num_neighs[cell_i] == 0) {
      //print_neighs_list(neighs[cell_i], neighs_MAX, tag[cell_i]);
      error->one(FLERR, "no of neighs for atom is 0");
    }

    //return a random neigh index [0, num_neighs_i - 1]
    int rand_neigh_idx = rand() % num_neighs[cell_i];
    int cell_j = atom->map(neighs[cell_i][rand_neigh_idx]);    //we need the owned atom id of cell_j

    //CHECK 1 ---> No of neighs for a cell should not fall below 4
    if (num_neighs[cell_i] <= 4 || num_neighs[cell_j] <= 4) {
      continue;    //Not a valid candidate for swapping
    }

    tagint comm_neigh_ij[2] = {-1};    // array to store common pair atoms
    int num_comm_neighs_ij = get_comm_neigh(comm_neigh_ij, neighs[cell_i], neighs[cell_j],
                                            num_neighs[cell_i], num_neighs[cell_j]);

    //CHECK 2 --> Two bonded atoms (i and j) should have exactly two distinct common neighs
    if (num_comm_neighs_ij != 2) {
      //print_neighs_list(neighs[cell_i], num_neighs[cell_i], tag[cell_i]);
      //print_neighs_list(neighs[cell_j], num_neighs[cell_j], tag[cell_j]);
      error->one(
          FLERR,
          " \n No of common neighs between bonded atoms (i and j) not exactly 2 --> probably some "
          "problem in swapping , This could also happen if there is a triangular cell involved\n");
    }

    if (comm_neigh_ij[0] == comm_neigh_ij[1]) {
      error->one(FLERR,
                 " \n Both the comm neighs of i and j are same --> might be due to some atom added "
                 "twice \n");
    }

    //If common neighs for cell i and j are exactly 2 and distinct then continue

    //this gives the ids of the owned images of the neighboring cells of cell i (and j)
    int cell_k = atom->map(comm_neigh_ij[0]);
    int cell_l = atom->map(comm_neigh_ij[1]);

    //all cell_i,j,k,l are 'local' ids at this point

    cell_tags[0] = tag[cell_i];
    cell_tags[1] = tag[cell_j];
    cell_tags[2] = tag[cell_k];
    cell_tags[3] = tag[cell_l];

    //Find the nearest images of cells from cell_i (now based on updated positions)

    int cell_j_nearest = nearest_image(cell_i, cell_j, x_updated);
    int cell_k_nearest = nearest_image(cell_i, cell_k, x_updated);
    int cell_l_nearest = nearest_image(cell_i, cell_l, x_updated);

    //CHECK 3 --> Get the correct images of cells and their neighbors

    int flag_valid_imj = 0;

    int closest_imj_of_l_from_k = nearest_image(cell_k, cell_l, x_updated);
    int closest_imj_of_k_from_l = nearest_image(cell_l, cell_k, x_updated);

    if (cell_k_nearest == cell_k &&
        cell_l_nearest == cell_l) {    // if k and l are owned then the nearest images must coincide
      if (cell_k_nearest == closest_imj_of_k_from_l && cell_l_nearest == closest_imj_of_l_from_k) {
        flag_valid_imj = 1;    //swapping allowed as nearest images coincide
      }
    } else if (cell_k_nearest != cell_k &&
               cell_l_nearest !=
                   cell_l) {    //if both k and l are ghost then nearest images must be different
      if (cell_k_nearest != closest_imj_of_k_from_l && cell_l_nearest != closest_imj_of_l_from_k) {
        flag_valid_imj = 1;    //swapping allowed as nearest images don't coincide
      }
    } else if (cell_k_nearest == cell_k &&
               cell_l_nearest != cell_l) {    //if k is owned and l is ghost
      if (cell_l_nearest == closest_imj_of_l_from_k && cell_k_nearest != closest_imj_of_k_from_l) {
        flag_valid_imj = 1;
      }
    } else if (cell_k_nearest != cell_k && cell_l_nearest == cell_l) {
      if (cell_k_nearest == closest_imj_of_k_from_l && cell_l_nearest != closest_imj_of_l_from_k) {
        flag_valid_imj = 1;
      }
    }

    if (flag_valid_imj == 0) {
      continue;    //not a valid swap
    }

    //CHECK 4 --> see if cell_k and cell_l are (1.) non-bonded and (2.) have exactly two common neighs

    int flag_unbonded = unbonded(neighs[cell_k], num_neighs[cell_k], tag[cell_l]);
    tagint comm_neigh_kl[2] = {-1};    // array to store common pair atoms
    int num_comm_neighs_kl = get_comm_neigh(comm_neigh_kl, neighs[cell_k], neighs[cell_l],
                                            num_neighs[cell_k], num_neighs[cell_l]);

    if (num_comm_neighs_kl != 2) {
      // error->one(FLERR,
      //            "\n No of common neighs between cells k and l not exactly 2 --> some problem in "
      //            "swapping \n");
      continue;
    }

    //CHECK 5: Edge length must be smaller than some threshold
    
    double nu1_x = (x_updated[cell_i][0] + x_updated[cell_j_nearest][0] + x_updated[cell_k_nearest][0]) / 3.0;
    double nu1_y = (x_updated[cell_i][1] + x_updated[cell_j_nearest][1] + x_updated[cell_k_nearest][1]) / 3.0;
    double nu2_x = (x_updated[cell_i][0] + x_updated[cell_j_nearest][0] + x_updated[cell_l_nearest][0]) / 3.0;
    double nu2_y = (x_updated[cell_i][1] + x_updated[cell_j_nearest][1] + x_updated[cell_l_nearest][1]) / 3.0; 

    double edge_length = sqrt((nu2_x - nu1_x)*(nu2_x - nu1_x) + (nu2_y - nu1_y)*(nu2_y - nu1_y));

    if (edge_length > T1_threshold){
      continue;
    }

    //CHECK 6 --> See if the quadrialteral formed is convex or concave:
    //i->k->j->l is the order of points of quadralteral in a cyclic manner

    double p1[2] = {x_updated[cell_i][0], x_updated[cell_i][1]};
    double p2[2] = {x_updated[cell_k_nearest][0], x_updated[cell_k_nearest][1]};
    double p3[2] = {x_updated[cell_j_nearest][0], x_updated[cell_j_nearest][1]};
    double p4[2] = {x_updated[cell_l_nearest][0], x_updated[cell_l_nearest][1]};

    //If quadrilateral formed is concave then resultant tiling not valid
    if (isConcave(p1, p2, p3, p4)) {
      continue;
    }

    /*********All check completed ---> Attempt the swapping***********/

    //store energy before swap
    double E_before = cell_shape[cell_i][2] + cell_shape[cell_j][2] + cell_shape[cell_k][2] +
        cell_shape[cell_l][2];

    //Attempt a bond swap--> Update neighs information first for all owned atoms (and their ghost images) involved

    //Delete bond
    for (int m = 0; m < 2; m++) {
      int owned_atom = atom->map(cell_tags[m]);
      tagint neigh_tag;
      if (m == 0) {
        neigh_tag = cell_tags[1];
      } else if (m == 1) {
        neigh_tag = cell_tags[0];
      }
      //First update info on owned atom
      remove_neigh(owned_atom, neigh_tag, neighs[owned_atom], num_neighs[owned_atom], neighs_MAX);
      num_neighs[owned_atom] -= 1;
      if (num_neighs[owned_atom] == 3) { error->one(FLERR, "number of neighs fell below 4"); }
      arrange_cyclic(neighs[owned_atom], num_neighs[owned_atom], owned_atom, x_updated);
      //Now update info on its ghost images
      int imj_atom = sametag[owned_atom];
      while (imj_atom != -1) {
        for (int ii = 0; ii < neighs_MAX; ii++) { neighs[imj_atom][ii] = neighs[owned_atom][ii]; }
        num_neighs[imj_atom] = num_neighs[owned_atom];
        imj_atom = sametag[imj_atom];
      }
    }

    //Create bond if not already bonded
    if (flag_unbonded == 1) {
      for (int m = 2; m < 4; m++) {
        int owned_atom = atom->map(cell_tags[m]);
        tagint neigh_tag;
        if (m == 2) {
          neigh_tag = cell_tags[3];
        } else if (m == 3) {
          neigh_tag = cell_tags[2];
        }
        //Update info of owned atom
        neighs[owned_atom][num_neighs[owned_atom]] = neigh_tag;
        num_neighs[owned_atom] += 1;
        if (num_neighs[owned_atom] == neighs_MAX) {
          printf("\n Limit reached for adding new neighs to atom %d\n", tag[owned_atom]);
          error->one(FLERR, "EXITED");
        }
        arrange_cyclic(neighs[owned_atom], num_neighs[owned_atom], owned_atom, x_updated);
        //Now update info on ghost images
        int imj_atom = sametag[owned_atom];
        while (imj_atom != -1) {
          for (int ii = 0; ii < neighs_MAX; ii++) { neighs[imj_atom][ii] = neighs[owned_atom][ii]; }
          num_neighs[imj_atom] = num_neighs[owned_atom];
          imj_atom = sametag[imj_atom];
        }
      }
    }

    // Check if the swap results in valid triangulation or not

    int flag_valid_tri = 1;

    for (int i = 0; i < 4; i++) {
      int cell = atom->map(cell_tags[i]);
      int num_neighs_cell = num_neighs[cell];
      int neigh_prev, neigh_next;
      for (int j = 0; j < num_neighs_cell; j++) {
        int curr_neigh = atom->map(neighs[cell][j]);    //local id
        if (j == 0) {
          neigh_prev = neighs[cell][num_neighs_cell - 1];
          neigh_next = neighs[cell][j + 1];
        } else if (j == num_neighs_cell - 1) {
          neigh_prev = neighs[cell][j - 1];
          neigh_next = neighs[cell][0];
        } else {
          neigh_prev = neighs[cell][j - 1];
          neigh_next = neighs[cell][j + 1];
        }
        //check the previous and next atoms for the current neigh j
        int num_neighs_curr = num_neighs[curr_neigh];
        int neigh_prev_curr, neigh_next_curr;
        int jj = 0;
        while (tag[cell] != neighs[curr_neigh][jj]) {
          jj++;
          // error check
          if (jj == num_neighs_curr) {
            error->one(FLERR, "\n tag[cell] not found in the neighs list of curr neigh \n");
          }
          //error check
        }
        if (jj == 0) {
          neigh_prev_curr = neighs[curr_neigh][num_neighs_curr - 1];
          neigh_next_curr = neighs[curr_neigh][jj + 1];
        } else if (jj == num_neighs_curr - 1) {
          neigh_prev_curr = neighs[curr_neigh][jj - 1];
          neigh_next_curr = neighs[curr_neigh][0];
        } else {
          neigh_prev_curr = neighs[curr_neigh][jj - 1];
          neigh_next_curr = neighs[curr_neigh][jj + 1];
        }
        if (neigh_prev != neigh_next_curr || neigh_next != neigh_prev_curr) {
          flag_valid_tri = 0;
          break;
        }
      }
      if (flag_valid_tri == 0) { break; }
    }

    int flag_energy_crit = 1;    //for energy criteria

    //Now get the cell_shape data for all atoms and compute energy after swap

    for (int m = 0; m < 4; m++) {
      int owned_atom = atom->map(cell_tags[m]);
      get_cell_data(vertices[owned_atom], cell_shape[owned_atom], neighs[owned_atom],
                        num_neighs[owned_atom], owned_atom, x_updated);
      int imj_atom = sametag[owned_atom];
      while (imj_atom != -1) {
        cell_shape[imj_atom][0] = cell_shape[owned_atom][0];
        cell_shape[imj_atom][1] = cell_shape[owned_atom][1];
        cell_shape[imj_atom][2] = cell_shape[owned_atom][2];
        imj_atom = sametag[imj_atom];
      }
    }

    double E_after = cell_shape[cell_i][2] + cell_shape[cell_j][2] + cell_shape[cell_k][2] +
        cell_shape[cell_l][2];

    double Delta_E = E_after - E_before;

    if (flag_valid_tri == 1) {
      //Metropolis scheme
      if (Delta_E < 0) {
        // //DEBUGGER
        // E_MC_n = E_MC_n + Delta_E;   //update energy at this MC step
        // fp[1] << n+1 << "  " << E_MC_n << endl;
        // //DEBUGGER
        continue;
      } else {
        double Prob = exp(-1.0 * Delta_E / KT);
        double rand_num = (double) rand() / RAND_MAX;    //generate a random no between 0 and 1
        if (Prob > rand_num) {
          // //DEBUGGER
          // E_MC_n = E_MC_n + Delta_E;  //Update energy at this MC step
          // fp[1] << n+1 << "  " << E_MC_n << endl;
          // //DEBUGGER
          continue;    //accept the swap with this probability
        } else {
          flag_energy_crit = 0;    //swapping attempt failed as not energetically favoured
          num_swaps_rejected += 1;
        }
      }
    }

    //Reverse swapping if not fulfilled all the criteria

    if (flag_energy_crit == 0 || flag_valid_tri == 0) {

      // //DEBUGGER
      //   fp[1] << n+1 << "  " << E_MC_n << endl;
      // //DEBUGGER

      cell_tags[0] = tag[cell_k];
      cell_tags[1] = tag[cell_l];
      cell_tags[2] = tag[cell_i];
      cell_tags[3] = tag[cell_j];

      //First delete bond between atom k and l (only if they were previously unbonded)
      if (flag_unbonded == 1) {
        //Delete bond
        for (int m = 0; m < 2; m++) {
          int owned_atom = atom->map(cell_tags[m]);
          tagint neigh_tag;
          if (m == 0) {
            neigh_tag = cell_tags[1];
          } else if (m == 1) {
            neigh_tag = cell_tags[0];
          }
          //First update info on owned atom
          remove_neigh(owned_atom, neigh_tag, neighs[owned_atom], num_neighs[owned_atom],
                       neighs_MAX);
          num_neighs[owned_atom] -= 1;
          if (num_neighs[owned_atom] == 3) { error->one(FLERR, "number of neighs fell below 4"); }
          arrange_cyclic(neighs[owned_atom], num_neighs[owned_atom], owned_atom, x_updated);
          //Now update info on its ghost images
          int imj_atom = sametag[owned_atom];
          while (imj_atom != -1) {
            for (int ii = 0; ii < neighs_MAX; ii++) {
              neighs[imj_atom][ii] = neighs[owned_atom][ii];
            }
            num_neighs[imj_atom] = num_neighs[owned_atom];
            imj_atom = sametag[imj_atom];
          }
        }
      }

      //Then Create bond between i and j
      for (int m = 2; m < 4; m++) {
        int owned_atom = atom->map(cell_tags[m]);
        tagint neigh_tag;
        if (m == 2) {
          neigh_tag = cell_tags[3];
        } else if (m == 3) {
          neigh_tag = cell_tags[2];
        }
        //Update info of owned atom
        neighs[owned_atom][num_neighs[owned_atom]] = neigh_tag;
        num_neighs[owned_atom] += 1;
        if (num_neighs[owned_atom] == neighs_MAX) {
          printf("\n Limit reached for adding new neighs to atom %d\n", tag[owned_atom]);
          error->one(FLERR, "EXITED");
        }
        arrange_cyclic(neighs[owned_atom], num_neighs[owned_atom], owned_atom, x_updated);
        //Now update info on ghost images
        int imj_atom = sametag[owned_atom];
        while (imj_atom != -1) {
          for (int ii = 0; ii < neighs_MAX; ii++) { neighs[imj_atom][ii] = neighs[owned_atom][ii]; }
          num_neighs[imj_atom] = num_neighs[owned_atom];
          imj_atom = sametag[imj_atom];
        }
      }

      // Again go back to original information
      for (int m = 0; m < 4; m++) {
        int owned_atom = atom->map(cell_tags[m]);
        get_cell_data(vertices[owned_atom], cell_shape[owned_atom], neighs[owned_atom],
                      num_neighs[owned_atom], owned_atom, x_updated);
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

  /*We have made sure that ghost atoms info is correct so don't need communication now 
  Might help save some time and make code a bit faster*/

  // // DEBUGGER
  // fp[1] << endl << endl;
  // // DEBUGGER

  /*~~~~~~~~~~~~~~~~~~~~~~ END OF MONTE CARLO ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

}

/* ---------------------------------------------------------------------- */
/*<<<<<<<<<<<<<<<<<<<<<< HELPER FUNCTIONS (BEGIN) >>>>>>>>>>>>>>>>>>>>>>>>>*/
/* ---------------------------------------------------------------------- */

/*~~~~~~~~~~~~~~ FUNCTION 1: Finding the nearest image of cell j to cell i
                         (we need to use our own x_pos) ~~~~~~~~~~~~~~~~*/

int FixTriDynamic::nearest_image(int i, int j, double **x_pos)
{

  if (j < 0) return j;

  int *sametag = atom->sametag;
  int nearest = j;
  double delx = x_pos[i][0] - x_pos[j][0];
  double dely = x_pos[i][1] - x_pos[j][1];
  double rsqmin = delx * delx + dely * dely;
  double rsq;

  while (sametag[j] >= 0) {
    j = sametag[j];
    delx = x_pos[i][0] - x_pos[j][0];
    dely = x_pos[i][1] - x_pos[j][1];
    rsq = delx * delx + dely * dely;
    if (rsq < rsqmin) {
      rsqmin = rsq;
      nearest = j;
    }
  }

  return nearest;
}

/*~~~~~~~~~~~~~~ FUNCTION 2: New arrange_cyclic for different position of atoms x_pos
                  (we need to use our own x_pos) ~~~~~~~~~~~~~~~~*/

void FixTriDynamic::arrange_cyclic(tagint *celli_neighs, int num_faces, int icell,
                                       double **x_pos)
{
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

    //No errors detected --> move ahead

    int k = nearest_image(icell, ktmp, x_pos);
    theta[j] = atan2(x_pos[k][1] - x_pos[icell][1], x_pos[k][0] - x_pos[icell][0]);

    if (isnan(theta[j])) {
      printf("current time step: %ld\n", update->ntimestep);
      printf("Cell (global): %d and neighbor (global): %d, neighbor (local): %d, closest "
             "image(local): %d\n",
             tag[icell], celli_neighs[j], ktmp, k);
      double slope = (x_pos[k][1] - x_pos[icell][1]) / (x_pos[k][0] - x_pos[icell][0]);
      printf(
          " coords of i ---> (%f, %f), coords of k ---> (%f, %f),  slope ---> %f, atan2 ---> %f\n",
          x_pos[icell][0], x_pos[icell][1], x_pos[k][0], x_pos[k][1], slope,
          atan2(x_pos[k][1] - x_pos[icell][1], x_pos[k][0] - x_pos[icell][0]));
      error->one(FLERR, "theta j in arrange cyclic function returned nan");
    }

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

/*~~~~~~~~~~~~~~~~~~~~~~~~~ FUNCTION 3 Find cell polygon area, peri and energy based on your own positions*/

void FixTriDynamic::get_cell_data(double *celli_vertices, double *celli_geom,
                                      tagint *celli_neighs, int num_faces, int icell,
                                      double **x_pos)
{
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
    int j_mu1 = nearest_image(icell, atom->map(celli_neighs[mu1]), x_pos);

    // local id of mu2
    int j_mu2 = nearest_image(icell, atom->map(celli_neighs[mu2]), x_pos);

    // Coordinates of current triangulation
    double xn[3] = {x_pos[icell][0], x_pos[j_mu1][0], x_pos[j_mu2][0]};
    double yn[3] = {x_pos[icell][1], x_pos[j_mu1][1], x_pos[j_mu2][1]};

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

/*~~~~~~~~~~~~~~ FUNCTION 4: See if one list is cyclic permutation of other or not ~~~~~~~~~~~~~~~~*/

bool FixTriDynamic::is_cyclic_perm(tagint *original, tagint *rearranged, int num_of_neighs)
{

  tagint rearranged_concat[num_of_neighs * 2];

  //Fill in the rearranged_concat list

  for (int i = 0; i < num_of_neighs; i++) {
    rearranged_concat[i] = rearranged[i];
    rearranged_concat[i + num_of_neighs] = rearranged[i];
  }

  //find the location where first element of OG list is found
  int idx;

  for (int i = 0; i < num_of_neighs; i++) {
    if (rearranged_concat[i] == original[0]) {
      idx = i;
      break;
    }
  }

  // Now see if the original list is a sublist of concatenated list or not
  for (int i = 0; i < num_of_neighs; i++) {
    if (original[i] != rearranged_concat[idx + i]) { return false; }
  }

  return true;
}

/*~~~~~~~~~~~~~~~~~~~Function 3: Get common neighs for chosen bonded pair of atoms~~~~~~~~~~~~~~~~*/

int FixTriDynamic::get_comm_neigh(tagint *common_neighs, tagint *cella_neighs, tagint *cellb_neighs,
                                  int num_cella_neighs, int num_cellb_neighs)
{

  int num_comm = 0;

  for (int n = 0; n < num_cella_neighs; n++) {
    for (int m = 0; m < num_cellb_neighs; m++) {
      if (cella_neighs[n] == cellb_neighs[m]) {
        num_comm += 1;
        if (num_comm == 3) { return num_comm; }
        common_neighs[num_comm - 1] = cella_neighs[n];
      }
    }
  }

  return num_comm;
}

/*~~~~~~~~~~~~~~~~~~~ Function 4: see if cells k and l are already bonded or not ~~~~~~~~~~~~~~~~*/

int FixTriDynamic::unbonded(tagint *cell_neighs, int num_cell_neighs, tagint tag_neigh)
{
  for (int n = 0; n < num_cell_neighs; n++) {
    if (cell_neighs[n] == tag_neigh) { return 0; }
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

void FixTriDynamic::print_neighs_list(tagint *cell_neighs, int num_cell_neighs, int cell)
{
  printf("Neighs List for local %d (global %d)--->", cell, atom->tag[cell]);
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

/*~~~~~~~~~~~~~~~~~~~~~Function 10: normalize a vector~~~~~~~~~~~*/

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
