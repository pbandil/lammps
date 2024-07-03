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

#include "fix_volvoro.h"

#include "arg_info.h"
#include "atom.h"
#include "atom_masks.h"
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
#include "region.h"
#include "respa.h"
#include "update.h"
#include "variable.h"

#include <algorithm>
#include <bits/stdc++.h>
#include <fstream>
#include <iostream>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <math.h>
#include <random>
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

FixVolVoro::FixVolVoro(LAMMPS *lmp, int narg, char **arg) :
    Fix(lmp, narg, arg), id_compute_voronoi(nullptr), total_virial(nullptr)
{
  if (narg < 6) error->all(FLERR, "Illegal fix volvoro command: not sufficient args");

  MPI_Comm_rank(world,&me);
  MPI_Comm_size(world, &nprocs);

  dynamic_group_allow = 1;
  virial_global_flag  = 1;
  virial_peratom_flag = 1;
  thermo_virial = 1;

  // Parse first two arguments: elasticity and preferred area
  Elasticity = utils::numeric(FLERR,arg[3],false,lmp);
  VolPref = utils::numeric(FLERR,arg[4],false,lmp);

  // Parse compute id
  id_compute_voronoi = utils::strdup(arg[5]);
  vcompute = modify->get_compute_by_id(id_compute_voronoi);
  if (!vcompute) error->all(FLERR,"Could not find compute ID {} for voronoi compute", id_compute_voronoi);

  // Max faces of a voronoi cell
  max_faces = utils::inumeric(FLERR,arg[6],false,lmp);

  // parse values for optional args
  flag_store_init = 0;
  id_fix_store = nullptr;

  int iarg = 7;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"store_init") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix voronoi command");
      flag_store_init = 1;

      id_fix_store = utils::strdup(arg[iarg+1]);
      fstore = modify->get_fix_by_id(id_fix_store);
      if (!fstore) error->all(FLERR,"Could not find fix ID {} for voronoi fix/store", id_fix_store);

      iarg += 2;
    } else error->all(FLERR,"Illegal fix voronoi command");
  }

  // Default nevery for now
  nevery = 1;

  // Countflag to only initialize once
  countflag = 0;

  // Initialize nmax and virial pointer
  nmax = atom->nmax;
  total_virial = nullptr;

  // Specify attributes for dumping connectivity (dtf)
  peratom_flag = 1;
  size_peratom_cols = max_faces;
  peratom_freq = 1;

  // perform initial allocation of atom-based arrays
  // register with Atom class
  if (peratom_flag) {
    FixVolVoro::grow_arrays(atom->nmax);
    atom->add_callback(Atom::GROW);
  }

}

/* ---------------------------------------------------------------------- */

FixVolVoro::~FixVolVoro()
{
  delete[] id_compute_voronoi;
  delete[] id_fix_store;
  
  memory->destroy(total_virial);

  // unregister callbacks to this fix from atom class
  if (peratom_flag) {
    atom->delete_callback(id,Atom::GROW);
  }

  if (new_fix_id && modify->nfix) modify->delete_fix(new_fix_id);
  delete [] new_fix_id;
  
}

/* ---------------------------------------------------------------------- */
// returntype classname :: functidentifier(args) }

int FixVolVoro::setmask()
{
  datamask_read = datamask_modify = 0;

  int mask = 0;
  mask |= PRE_FORCE;
  mask |= POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixVolVoro::post_constructor()
{

  // Create call to fix property/atom for storing DT faces
  new_fix_id = utils::strdup(id + std::string("_FIX_PA"));
  modify->add_fix(fmt::format("{} {} property/atom i2_dtf_{} {} ghost yes",new_fix_id, group->names[igroup],id,std::to_string(max_faces)));

  int tmp1, tmp2;
  index = atom->find_custom(utils::strdup(std::string("dtf_")+id),tmp1,tmp2);

  tagint **dtf = atom->iarray[index];
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int nall = nlocal + nghost;

  // Initialize entries to 0
  for (int i = 0; i < nall; i++) {
    for (int m = 0; m < max_faces; m++) {
      if (mask[i] & groupbit) {
        dtf[i][m] = 0;
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixVolVoro::init()
{
  // set indices and check validity of all computes and variables

}


/* ---------------------------------------------------------------------- */

void FixVolVoro::setup(int vflag)
{

  // Return if this has already been invoked
  if (countflag) return;
  countflag = 1;

  // Ensure that computes have been invoked
  modify->clearstep_compute();
  vcompute = modify->get_compute_by_id(id_compute_voronoi);
  if (!(vcompute->invoked_flag & Compute::INVOKED_PERATOM)) {
    vcompute->compute_peratom();
    vcompute->invoked_flag |= Compute::INVOKED_PERATOM;
  }

  // Pointer to dtf
  tagint **dtf = atom->iarray[index];

  // Determine the number of rows in the local array for voronoi data
  int numRows = vcompute->size_local_rows;

  // Loop through numRows and fill dtf
  for (int n = 0; n < numRows; n++) {

    // Skip external faces (box boundaries) and empty cells
    if (vcompute->array_local[n][1] == 0.0) continue;

    // Atom that owns this cell
    int i = atom->map(vcompute->array_local[n][0]);
    if (i < 0)
      error->one(FLERR,"Can't find local atom");

    // Atom that shares this cell edge
    int j = atom->map(vcompute->array_local[n][1]);
    if (j < 0) {
      error->one(FLERR,"Fix volbulk needs ghost atoms from further away");
    }
    tagint tagj = atom->tag[j];
    if (tagj == atom->tag[i]) error->one(FLERR,"Atom edge with itself");

    // Find next empty space for atom i and fill it with tagj
    for (int m = 0; m < max_faces; m++) {
        if (dtf[i][m] == 0) {
            dtf[i][m] = tagj;
            break;
        }
    }
  }

  // Confirm cyclic permutation of dtf
  int nlocal = atom->nlocal;
  for (int i = 0; i < nlocal; i++) {
    arrange_cyclic(dtf[i],i);
  }

  // Communicate dtf
  comm->forward_comm(this,max_faces);
    
  // Create memory allocations
  nmax = atom->nmax;
  memory->create(total_virial,nmax,6,"volvoro:total_virial");

  // Run single timestep
  pre_force(vflag);
  post_force(vflag);

}

/* ---------------------------------------------------------------------- */

void FixVolVoro::pre_force(int vflag)
{

  // Atom counts
  int natoms = atom->natoms;
  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int nall = nlocal + nghost;

  // Confirm cyclic permutation of dtf
  tagint **dtf = atom->iarray[index];
  for (int i = 0; i < nlocal; i++) {
    arrange_cyclic(dtf[i],i);
  }

  // Communicate dtf
  comm->forward_comm(this,max_faces);

  // Full triangulation is rebuilt if any proc has flip = 1
  // Initialize flag to 1 to begin the loop
  int flip_all = 1; 
  while (flip_all > 0) {

    // Empty pointer to edge tags
    tagint edge_tags[4] = {0,0,0,0};

    // Check for edge flips on this proc
    int flip = check_edges(edge_tags);
    if (flip == 1) flip += me;

    // Check other procs to see if a flip has occurred
    MPI_Allreduce(&flip, &flip_all, 1, MPI_INT, MPI_MAX, world);

    // If any flips, need to sync procs before next calculation
    if (flip_all > 0) {

      // Flip edge from proc with the highest rank
      int flip_rank = flip_all-1;

      // All processors take edge_tags from flip_rank
      // MPI_Barrier(world);
      MPI_Bcast(&edge_tags,4,MPI_INT,flip_rank,world);
      // MPI_Barrier(world);

      // Perform edge flipping on owned procs
      flip_edge(edge_tags);
      // MPI_Barrier(world);

      // Communicate final dtf
      comm->forward_comm(this,max_faces);

    }
  }

  // For future implementation (nevery)
  if (update->ntimestep % nevery) return;

  // Flags to atom classes
  double **x = atom->x;
  double **f = atom->f;
  int *mask = atom->mask;

  // Possibly resize arrays
  if (atom->nmax > nmax) {
    memory->destroy(total_virial);
    nmax = atom->nmax;
    memory->create(total_virial,nmax,6,"volvoro:total_virial");
  }

  // Initialize arrays to zero
  for (int i = 0; i < nall; i++) {
    for (int j = 0; j < 5; j++) {
        total_virial[i][j] = 0.0;
    }
  }

  // Reset array-atom if outputting
  if (peratom_flag) {
    for (int i = 0; i < nlocal; i++) {
      for (int n = 0; n < max_faces; n++) {
        array_atom[i][n] = 0.0;
      }
    }
  }

  // Define pointer to fix_store
  if (flag_store_init) fstore = modify->get_fix_by_id(id_fix_store);

  // Loop through local atoms:
  // 1. Apply force `on i from i'
  // 2. Apply force `on j from i'
  // 3. Apply force `on k from i'
  for (int i = 0; i < nlocal; i++) {

    // Declare variables inside loop for scope clarity
    double fx,fy;
    int j,jtmp,num_faces,jleft,jright;
    double x1,x2,y1,y2;

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // ~~~~~~~~~~~~~~~~~~~~~ Refresh vector arrays ~~~~~~~~~~~~~~~~~~~~~~ //
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

    // Local ids of atoms in current triangulation
    int DT[3] = {0,0,0};

    // Coordinates of current, previous, and next vertex
    double vert[2] = {0.0,0.0}, vert_prev[2] = {0.0,0.0}, vert_next[2] = {0.0,0.0};

    // x and y coordinates of current triangulation
    double xn[3] = {0.0,0.0,0.0}, yn[3] = {0.0,0.0,0.0};

    // Vector spanning between n+1 and n-1 vertices
    double rnu_diff[2] = {0.0,0.0};

    // Vector spanning between the vertex and the atom
    double rvec[2] = {0.0,0.0};

    // Jacobian matrix (Voigt notation): J11, J22, J12, J21
    double Jac[4] = {0.0,0.0,0.0,0.0};

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // ~~~~~~~~~~~~~~~~~~~~~~~ Begin calculations ~~~~~~~~~~~~~~~~~~~~~~~ //
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

    // Find num_faces for this atom
    for (int n = 0; n < max_faces; n++) {
        if (dtf[i][n] == 0) {
            num_faces = n;
            break;
        }
    }

    // Volume and pressure due to volume change
    double voro_volume = calc_area(dtf[i],i,num_faces);
    if (voro_volume < 0) {
      error->one(FLERR,"Calculated negative volume");
    }
    double voro_volume0;
    if (flag_store_init) {
        voro_volume0 = fstore->vector_atom[i];
    } else {
        voro_volume0 = VolPref;
    }
    double pressure = Elasticity*(voro_volume-voro_volume0)*0.5;

    // Coords of atom i
    double x0 = x[i][0];
    double y0 = x[i][1];

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // ~~~~~~~~~~~~~~~~~~~~~~ Calculate vertices ~~~~~~~~~~~~~~~~~~~~~~~~ //
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

    // First entry of DT is always i
    DT[0] = i;

    // Local id and coords of last face
    jtmp = atom->map(dtf[i][num_faces-1]);
    if (jtmp < 0) {
      error->one(FLERR,"Fix volvoro needs ghost atoms from further away");
    }
    j = domain->closest_image(i,jtmp);
    x1 = x[j][0];
    y1 = x[j][1];

    // Local id and coords of first face
    jtmp = atom->map(dtf[i][0]);
    if (jtmp < 0) {
      error->one(FLERR,"Fix volvoro needs ghost atoms from further away");
    }
    j = domain->closest_image(i,jtmp);
    DT[1] = j;
    x2 = x[j][0];
    y2 = x[j][1];

    // Circumcenter of previous vertex
    xn[0] = x0; xn[1] = x1; xn[2] = x2;
    yn[0] = y0; yn[1] = y1; yn[2] = y2;
    calc_cc(xn, yn, vert_prev);
    x1 = x2;
    y1 = y2;

    // Local id and coords of second face
    jtmp = atom->map(dtf[i][1]);
    if (jtmp < 0) {
      error->one(FLERR,"Fix volvoro needs ghost atoms from further away");
    }
    j = domain->closest_image(i,jtmp);
    DT[2] = j;
    x2 = x[j][0];
    y2 = x[j][1];

    // Circumcenter of current vertex
    xn[0] = x0; xn[1] = x1; xn[2] = x2;
    yn[0] = y0; yn[1] = y1; yn[2] = y2;
    calc_cc(xn, yn, vert);
    x1 = x2;
    y1 = y2;

    // Local id and coords of third face
    jtmp = atom->map(dtf[i][2]);
    if (jtmp < 0) {
      error->one(FLERR,"Fix volvoro needs ghost atoms from further away");
    }
    j = domain->closest_image(i,jtmp);
    x2 = x[j][0];
    y2 = x[j][1];

    // Circumcenter of next vertex
    xn[0] = x0; xn[1] = x1; xn[2] = x2;
    yn[0] = y0; yn[1] = y1; yn[2] = y2;
    calc_cc(xn, yn, vert_next);
    x1 = x2;
    y1 = y2;

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
    // ~~~~~~~~~~~~~~~~ Force calculation on cell i ~~~~~~~~~~~~~~~~~~~~~//
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

    // rnu_diff is inverted due to cross product with outward normal
    rnu_diff[0] = vert_next[1] - vert_prev[1];
    rnu_diff[1] = -(vert_next[0] - vert_prev[0]);

    // Calculate jacobian on atom i (cell owner)
    calc_jacobian(DT,i,Jac);

    // Forces on i from current vertex
    fx = -pressure*(Jac[0]*rnu_diff[0] + Jac[3]*rnu_diff[1]);
    fy = -pressure*(Jac[2]*rnu_diff[0] + Jac[1]*rnu_diff[1]);
    f[i][0] += fx;
    f[i][1] += fy;

    // // Vector for virial calculation (from vertex to atom i)
    // rvec[0] = x[i][0] - vert[0];
    // rvec[1] = x[i][1] - vert[1];

    // // Virial contributions
    // total_virial[i][0] += fx*rvec[0];
    // total_virial[i][1] += fy*rvec[1];
    // total_virial[i][3] += fx*rvec[1];

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
    // ~~~~~~~~~~~~~~~~ Force calculation on jleft ~~~~~~~~~~~~~~~~~~~~~~//
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

    jleft = DT[1];

    // Calculate jacobian on atom jleft
    calc_jacobian(DT,jleft,Jac);

    // Forces on i from current vertex
    fx = -pressure*(Jac[0]*rnu_diff[0] + Jac[3]*rnu_diff[1]);
    fy = -pressure*(Jac[2]*rnu_diff[0] + Jac[1]*rnu_diff[1]);
    f[jleft][0] += fx;
    f[jleft][1] += fy;

    // Vector for virial calculation (from vertex to atom jleft)
    // rvec[0] = x[jleft][0] - vert[0];
    // rvec[1] = x[jleft][1] - vert[1];

    // Vector for virial (from i to j)
    rvec[0] = x[i][0] - x[jleft][0];
    rvec[1] = x[i][1] - x[jleft][1];

    // Virial contributions
    total_virial[jleft][0] += fx*rvec[0];
    total_virial[jleft][1] += fy*rvec[1];
    total_virial[jleft][3] += fx*rvec[1];

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
    // ~~~~~~~~~~~~~~~~ Force calculation on jright ~~~~~~~~~~~~~~~~~~~~~//
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

    jright = DT[2];

    // Calculate jacobian on atom jright
    calc_jacobian(DT,jright,Jac);

    // Forces on i from current vertex
    fx = -pressure*(Jac[0]*rnu_diff[0] + Jac[3]*rnu_diff[1]);
    fy = -pressure*(Jac[2]*rnu_diff[0] + Jac[1]*rnu_diff[1]);
    f[jright][0] += fx;
    f[jright][1] += fy;

    // Vector for virial calculation (from vertex to atom jright)
    // rvec[0] = x[jright][0] - vert[0];
    // rvec[1] = x[jright][1] - vert[1];

    // Vector for virial (from i to j)
    rvec[0] = x[i][0] - x[jright][0];
    rvec[1] = x[i][1] - x[jright][1];

    // Virial contributions
    total_virial[jright][0] += fx*rvec[0];
    total_virial[jright][1] += fy*rvec[1];
    total_virial[jright][3] += fx*rvec[1];

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
    // ~~~~~~~~~~~~~~~~ Permute vertices cyclically ~~~~~~~~~~~~~~~~~~~~~//
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

    // vert becomes previous vert
    vert_prev[0] = vert[0];
    vert_prev[1] = vert[1];

    // Next vert becomes vert
    vert[0] = vert_next[0];
    vert[1] = vert_next[1];

    // Second entry of DT becomes jright
    DT[1] = jright;

    // Loop through faces
    for (int n = 1; n < num_faces; n++) {

        // Only need to find the index of the n+2 face
        int jnext;
        if (n + 2 == num_faces) {

            jtmp = atom->map(dtf[i][0]);
            if (jtmp < 0) {
                error->one(FLERR,"Fix volvoro needs ghost atoms from further away");
            }
            jnext = domain->closest_image(i,jtmp);

            jtmp = atom->map(dtf[i][num_faces-1]);
            DT[2] = domain->closest_image(i,jtmp);
        } else if (n + 1 == num_faces) {

            jtmp = atom->map(dtf[i][1]);
            if (jtmp < 0) {
                error->one(FLERR,"Fix volvoro needs ghost atoms from further away");
            }
            jnext = domain->closest_image(i,jtmp);

            jtmp = atom->map(dtf[i][0]);
            DT[2] = domain->closest_image(i,jtmp);
        } else {

            jtmp = atom->map(dtf[i][n+2]);
            if (jtmp < 0) {
                error->one(FLERR,"Fix volvoro needs ghost atoms from further away");
            }
            jnext = domain->closest_image(i,jtmp);

            jtmp = atom->map(dtf[i][n+1]);
            DT[2] = domain->closest_image(i,jtmp);
        }

        // Find circumcenter of next vertex
        x2 = x[jnext][0];
        y2 = x[jnext][1];
        xn[0] = x0; xn[1] = x1; xn[2] = x2;
        yn[0] = y0; yn[1] = y1; yn[2] = y2;
        calc_cc(xn, yn, vert_next);
        x1 = x2;
        y1 = y2;

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
        // ~~~~~~~~~~~~~~~~ Force calculation on cell i ~~~~~~~~~~~~~~~~~~~~~//
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

        // rnu_diff is inverted due to cross product with outward normal
        rnu_diff[0] = vert_next[1] - vert_prev[1];
        rnu_diff[1] = -(vert_next[0] - vert_prev[0]);

        // Calculate jacobian on atom i (cell owner)
        calc_jacobian(DT,i,Jac);

        // Forces on i from current vertex
        fx = -pressure*(Jac[0]*rnu_diff[0] + Jac[3]*rnu_diff[1]);
        fy = -pressure*(Jac[2]*rnu_diff[0] + Jac[1]*rnu_diff[1]);
        f[i][0] += fx;
        f[i][1] += fy;

        // // Vector for virial calculation (from vertex to atom i)
        // rvec[0] = x[i][0] - vert[0];
        // rvec[1] = x[i][1] - vert[1];

        // // Virial contributions
        // total_virial[i][0] += fx*rvec[0];
        // total_virial[i][1] += fy*rvec[1];
        // total_virial[i][3] += fx*rvec[1];

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
        // ~~~~~~~~~~~~~~~~ Force calculation on jleft ~~~~~~~~~~~~~~~~~~~~~~//
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

        jleft = DT[1];

        // Calculate jacobian on atom jleft
        calc_jacobian(DT,jleft,Jac);

        // Forces on i from current vertex
        fx = -pressure*(Jac[0]*rnu_diff[0] + Jac[3]*rnu_diff[1]);
        fy = -pressure*(Jac[2]*rnu_diff[0] + Jac[1]*rnu_diff[1]);
        f[jleft][0] += fx;
        f[jleft][1] += fy;

        // Vector for virial calculation (from vertex to atom jleft)
        // rvec[0] = x[jleft][0] - vert[0];
        // rvec[1] = x[jleft][1] - vert[1];

        // Vector for virial (from i to j)
        rvec[0] = x[i][0] - x[jleft][0];
        rvec[1] = x[i][1] - x[jleft][1];

        // Virial contributions
        total_virial[jleft][0] += fx*rvec[0];
        total_virial[jleft][1] += fy*rvec[1];
        total_virial[jleft][3] += fx*rvec[1];

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
        // ~~~~~~~~~~~~~~~~ Force calculation on jright ~~~~~~~~~~~~~~~~~~~~~//
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

        jright = DT[2];

        // Calculate jacobian on atom jleft
        calc_jacobian(DT,jright,Jac);

        // Forces on i from current vertex
        fx = -pressure*(Jac[0]*rnu_diff[0] + Jac[3]*rnu_diff[1]);
        fy = -pressure*(Jac[2]*rnu_diff[0] + Jac[1]*rnu_diff[1]);
        f[jright][0] += fx;
        f[jright][1] += fy;

        // Vector for virial calculation (from vertex to atom jright)
        // rvec[0] = x[jright][0] - vert[0];
        // rvec[1] = x[jright][1] - vert[1];

        // Vector for virial (from i to j)
        rvec[0] = x[i][0] - x[jright][0];
        rvec[1] = x[i][1] - x[jright][1];

        // Virial contributions
        total_virial[jright][0] += fx*rvec[0];
        total_virial[jright][1] += fy*rvec[1];
        total_virial[jright][3] += fx*rvec[1];

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
        // ~~~~~~~~~~~~~~~~ Permute vertices cyclically ~~~~~~~~~~~~~~~~~~~~~//
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

        // vert becomes previous vert
        vert_prev[0] = vert[0];
        vert_prev[1] = vert[1];

        // Next vert becomes vert
        vert[0] = vert_next[0];
        vert[1] = vert_next[1];

        // Second entry of DT becomes jright
        DT[1] = jright;

    }
  }

  // Reverse communication of virial contributions
  comm->reverse_comm(this,6);
 
  // Store dtf for outputting
  if (peratom_flag) {
    for (int i = 0; i < nlocal; i++) {
      for (int n = 0; n < max_faces; n++) {
        array_atom[i][n] = dtf[i][n];
      }
    }
  }

}

/* ---------------------------------------------------------------------- */

void FixVolVoro::post_force(int vflag)
{

  // virial setup
  v_init(vflag);

  // Tally virial contributions of owned atoms
  for (int i = 0; i < atom->nlocal; i++) {
    total_virial[i][0] *= -1;
    total_virial[i][1] *= -1;
    total_virial[i][3] *= -1;

    v_tally(i, total_virial[i]);
  }

}

/*------------------------------------------------------------------------*/

int FixVolVoro::check_edges(tagint *edge_tags)
{

  double **x = atom->x;
  int *mask = atom->mask;
  tagint *tag = atom->tag;

  int natoms = atom->natoms;
  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int nall = nlocal + nghost;

  tagint **dtf = atom->iarray[index];

  // Loop through all triangles and flip edges
  for (int i = 0; i < nlocal; i++) {

    // Determine the number of faces for this atom
    int num_faces;
    for (int n = 0; n < max_faces; n++) {
        if (dtf[i][n] == 0) {
            num_faces = n;
            break;
        }
    }

    // Loop through each face to count all triangles
    for (int n = 0; n < num_faces; n++) {

        // Indices of current and next neighbor
        int ind1 = n;
        int ind2 = n+1;

        // Loop back to first neighbor for final triangle
        if (ind2 == num_faces) {
            ind2 = 0;
        }

        // Index of third neighbor
        int ind3 = ind2+1;

        // Loop back to first neighbor for second to last triangle
        if (ind3 == num_faces) {
            ind3 = 0;
        }

        // Local id of ind1
        int j_ind1_tmp = atom->map(dtf[i][ind1]);
        if (j_ind1_tmp < 0) {
            error->one(FLERR,"Fix volvoro needs ghost atoms from further away");
        }
        int j_ind1 = domain->closest_image(i,j_ind1_tmp);

        // local id of ind2
        int j_ind2_tmp = atom->map(dtf[i][ind2]);
        if (j_ind2_tmp < 0) {
            error->one(FLERR,"Fix volvoro needs ghost atoms from further away");
        }
        int j_ind2 = domain->closest_image(i,j_ind2_tmp);

        // local id of ind3
        int j_ind3_tmp = atom->map(dtf[i][ind3]);
        if (j_ind3_tmp < 0) {
            error->one(FLERR,"Fix volvoro needs ghost atoms from further away");
        }
        int j_ind3 = domain->closest_image(i,j_ind3_tmp);

        // a: distance btw. i and j_ind2
        double a2 = (x[i][0] - x[j_ind2][0])*(x[i][0] - x[j_ind2][0]) + (x[i][1] - x[j_ind2][1])*(x[i][1] - x[j_ind2][1]);

        // b: distance btw. i and j_ind1
        double b2 = (x[i][0] - x[j_ind1][0])*(x[i][0] - x[j_ind1][0]) + (x[i][1] - x[j_ind1][1])*(x[i][1] - x[j_ind1][1]);

        // c: distance btw. j_ind1 and j_ind2
        double c2 = (x[j_ind2][0] - x[j_ind1][0])*(x[j_ind2][0] - x[j_ind1][0]) + (x[j_ind2][1] - x[j_ind1][1])*(x[j_ind2][1] - x[j_ind1][1]);

        // d: distance btw. i and j_ind3
        double d2 = (x[i][0] - x[j_ind3][0])*(x[i][0] - x[j_ind3][0]) + (x[i][1] - x[j_ind3][1])*(x[i][1] - x[j_ind3][1]);

        // e: distance btw. j_ind2 and j_ind3
        double e2 = (x[j_ind2][0] - x[j_ind3][0])*(x[j_ind2][0] - x[j_ind3][0]) + (x[j_ind2][1] - x[j_ind3][1])*(x[j_ind2][1] - x[j_ind3][1]);

        // External angles
        double costh1 = (b2 + c2 - a2)/(pow(b2,0.5)*pow(c2,0.5));
        double costh2 = (d2 + e2 - a2)/(pow(d2,0.5)*pow(e2,0.5));

        // Flip if costh1 + costh2 < 0
        if (costh1 + costh2 < 0.0) {

            // Store the tags for each atom in the flip to communicate
            edge_tags[0] = atom->tag[i];
            edge_tags[1] = atom->tag[j_ind1];
            edge_tags[2] = atom->tag[j_ind2];
            edge_tags[3] = atom->tag[j_ind3];

            // Return and flag edge flip
            return 1;
        }
    }
  }

  return 0;

}

/*------------------------------------------------------------------------*/

void FixVolVoro::flip_edge(tagint *edge_tags)
{

  // Skip empty arrays (nothing flipped)
  if (edge_tags[0] == 0) return;

  // Initialize pointers and needed values
  int natoms = atom->natoms;
  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int nall = nlocal + nghost;
  tagint **dtf = atom->iarray[index];

  // Tags 0 and 2 need to delete an entry of dtf
  tagint tag0 = edge_tags[0];
  tagint tag2 = edge_tags[2];

  // Tags 1 and 3 need to add an entry to dtf
  tagint tag1 = edge_tags[1];
  tagint tag3 = edge_tags[3];

  // Local ids of each tag
  int i0 = atom->map(tag0);
  int i1 = atom->map(tag1);
  int i2 = atom->map(tag2);
  int i3 = atom->map(tag3);

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
  // ~~~~~~~~~~~~~~~~~~~~~~ Remove i2 from i0 ~~~~~~~~~~~~~~~~~~~~~~~~~//
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

  if (i0 > -1 && i0 < nlocal) {

    // First remove entry tag2
    int success = 0;
    for (int m = 0; m < max_faces; m++){
        if (dtf[i0][m] == tag2) {
            dtf[i0][m] = 0;
            success = 1;
            break;
        }
    }
    if (success == 0) {
      error->one(FLERR,"Edge already deleted");
    }

    // Move zero entries to the end of dtf[i]
    int right = 0; // Points to the next position for a non-zero element
    for (int left = 0; left < max_faces; ++left) {
        if (dtf[i0][left] != 0) {
            // Swap only if left and right are different
            if (left != right) {
                std::swap(dtf[i0][left], dtf[i0][right]);
            }
            right++;
        }
    }

    // Perform cyclic arrangement
    arrange_cyclic(dtf[i0],i0);
  }

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
  // ~~~~~~~~~~~~~~~~~~~~~~ Remove i0 from i2 ~~~~~~~~~~~~~~~~~~~~~~~~~//
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

  if (i2 > -1 && i2 < nlocal) {

    // First remove entry tag0
    int success = 0;
    for (int m = 0; m < max_faces; m++){
        if (dtf[i2][m] == tag0) {
            dtf[i2][m] = 0;
            success = 1;
            break;
        }
    }
    if (success == 0) {
      error->one(FLERR,"Edge already deleted");
    }

    // Move zero entries to the end of dtf[i]
    int right = 0; // Points to the next position for a non-zero element
    for (int left = 0; left < max_faces; ++left) {
        if (dtf[i2][left] != 0) {
            // Swap only if left and right are different
            if (left != right) {
                std::swap(dtf[i2][left], dtf[i2][right]);
            }
            right++;
        }
    }

    // Perform cyclic arrangement
    arrange_cyclic(dtf[i2],i2);
  }

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
  // ~~~~~~~~~~~~~~~~~~~~~~~~~ Add i3 to i1 ~~~~~~~~~~~~~~~~~~~~~~~~~~~//
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

  if (i1 > -1 && i1 < nlocal) {

    // Add entry tag3 to atom tag1
    int index1 = max_faces+1;
    for (int m = 0; m < max_faces; m++){
        if (dtf[i1][m] == 0) {
            index1 = m;
            break;
        }
    }
    if (index1 == max_faces+1) {
        error->one(FLERR,"Max faces exceeded");
    }
    dtf[i1][index1] = tag3;

    // Perform cyclic arrangement
    arrange_cyclic(dtf[i1],i1);
  }

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
  // ~~~~~~~~~~~~~~~~~~~~~~~~~ Add i1 to i3 ~~~~~~~~~~~~~~~~~~~~~~~~~~~//
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

  if (i3 > -1 && i3 < nlocal) {

    // Add entry tag1 to atom tag3
    int index1 = max_faces+1;
    for (int m = 0; m < max_faces; m++){
        if (dtf[i3][m] == 0) {
            index1 = m;
            break;
        }
    }
    if (index1 == max_faces+1) {
        error->one(FLERR,"Max faces exceeded");
    }
    dtf[i3][index1] = tag1;

    // Perform cyclic arrangement
    arrange_cyclic(dtf[i3],i3);
  }

}

/*------------------------------------------------------------------------*/

void FixVolVoro::arrange_cyclic(tagint *tag_vec, int icell)
{
  double **x = atom->x;

  // Determine the number of faces for this atom
  int num_faces;
  for (int n = 0; n < max_faces; n++) {
      if (tag_vec[n] == 0) {
          num_faces = n;
          break;
      }
  }

  // Calculate angle for all points
  double theta[num_faces];
  int indices[num_faces];
  for (int i = 0; i < num_faces; i++) {

    // Local id of current face
    int jtmp = atom->map(tag_vec[i]);
    int j = domain->closest_image(icell,jtmp);
    theta[i] = atan2(x[j][1] - x[icell][1],x[j][0] - x[icell][0]);

    // indices array for sorting
    indices[i] = i;
  }

  // Sort indices based on minimum angle
  std::sort(indices, indices + num_faces, [&](int i, int n) { return theta[i] < theta[n]; });

  // Create dummy vector for sorting
  tagint tag_sorted[max_faces];
  for (int i = 0; i < max_faces; i++) {
    tag_sorted[i] = 0;
  }
  for (int i = 0; i < num_faces; i++) {
    tag_sorted[i] = tag_vec[indices[i]];
  }

  // Fill tag_vec with sorted values
  for (int i = 0; i < num_faces; i++) {
    tag_vec[i] = tag_sorted[i];
  }

}

/*------------------------------------------------------------------------*/

void FixVolVoro::calc_cc(double *xn, double *yn,  double *CC)
{
  double x0 = xn[0];
  double x1 = xn[1];
  double x2 = xn[2];

  double y0 = yn[0];
  double y1 = yn[1];
  double y2 = yn[2];

  double a = sqrt(pow(x2-x1,2)+pow(y2-y1,2));
  double b = sqrt(pow(x2-x0,2)+pow(y2-y0,2));
  double c = sqrt(pow(x1-x0,2)+pow(y1-y0,2));

 // Find Angles A, B and C

  double A = acos((b*b+c*c-a*a)/(2*b*c));
  double B = acos((a*a+c*c-b*b)/(2*a*c));
  double C = acos((a*a+b*b-c*c)/(2*a*b));

  CC[0] = (x0*sin(2*A)+x1*sin(2*B)+x2*sin(2*C))/(sin(2*A)+sin(2*B)+sin(2*C)); // x coord of circumcenter
  CC[1] = (y0*sin(2*A)+y1*sin(2*B)+y2*sin(2*C))/(sin(2*A)+sin(2*B)+sin(2*C)); // y coord of circumcenter
}

/*------------------------------------------------------------------------*/

void FixVolVoro::calc_jacobian(int *DT, int p, double *Jac){

  double **x = atom->x;

  // Identify j and k
  int j,k;
  if (DT[0] == p) {
      j = DT[1];
      k = DT[2];
  }
  else if (DT[1] == p){
      j = DT[0];
      k = DT[2];
  }
  else if (DT[2] == p){
      j = DT[0];
      k = DT[1];
  }

  double ri[3] = {x[p][0], x[p][1], 0};
  double rj[3] = {x[j][0], x[j][1], 0};
  double rk[3] = {x[k][0], x[k][1], 0};

  double li = sqrt(pow(rj[0]-rk[0],2)+pow(rj[1]-rk[1],2));
  double lj = sqrt(pow(ri[0]-rk[0],2)+pow(ri[1]-rk[1],2));
  double lk = sqrt(pow(ri[0]-rj[0],2)+pow(ri[1]-rj[1],2));

  double lam1 = li*li*(lj*lj+lk*lk-li*li);
  double lam2 = lj*lj*(lk*lk+li*li-lj*lj);
  double lam3 = lk*lk*(li*li+lj*lj-lk*lk);

  double clam = lam1 + lam2 + lam3;

  double dclam_dri[3] = {-4*(li*li+lk*lk-lj*lj)*(rk[0]-ri[0])+4*(li*li+lj*lj-lk*lk)*(ri[0]-rj[0]),
                         -4*(li*li+lk*lk-lj*lj)*(rk[1]-ri[1])+4*(li*li+lj*lj-lk*lk)*(ri[1]-rj[1]),
                         -4*(li*li+lk*lk-lj*lj)*(rk[2]-ri[2])+4*(li*li+lj*lj-lk*lk)*(ri[2]-rj[2])} ;

  double dlam1_dri[3] = {2*li*li*(-(rk[0]-ri[0])+(ri[0]-rj[0])),
                         2*li*li*(-(rk[1]-ri[1])+(ri[1]-rj[1])),
                         2*li*li*(-(rk[2]-ri[2])+(ri[2]-rj[2]))} ; 

  double dlam2_dri[3] = {-2*(li*li+lk*lk-2*lj*lj)*(rk[0]-ri[0])+2*lj*lj*(ri[0]-rj[0]),
                         -2*(li*li+lk*lk-2*lj*lj)*(rk[1]-ri[1])+2*lj*lj*(ri[1]-rj[1]),
                         -2*(li*li+lk*lk-2*lj*lj)*(rk[2]-ri[2])+2*lj*lj*(ri[2]-rj[2])} ;

  double dlam3_dri[3] = {2*(li*li+lj*lj-2*lk*lk)*(ri[0]-rj[0])-2*lk*lk*(rk[0]-ri[0]),
                         2*(li*li+lj*lj-2*lk*lk)*(ri[1]-rj[1])-2*lk*lk*(rk[1]-ri[1]),
                         2*(li*li+lj*lj-2*lk*lk)*(ri[2]-rj[2])-2*lk*lk*(rk[2]-ri[2])} ;

  double d1[3] = {(clam*dlam1_dri[0]-lam1*dclam_dri[0])/(clam*clam),
                  (clam*dlam1_dri[1]-lam1*dclam_dri[1])/(clam*clam),
                  (clam*dlam1_dri[2]-lam1*dclam_dri[2])/(clam*clam)} ;

  double d2[3] = {(clam*dlam2_dri[0]-lam2*dclam_dri[0])/(clam*clam),
                  (clam*dlam2_dri[1]-lam2*dclam_dri[1])/(clam*clam),
                  (clam*dlam2_dri[2]-lam2*dclam_dri[2])/(clam*clam)} ;

  double d3[3] = {(clam*dlam3_dri[0]-lam3*dclam_dri[0])/(clam*clam),
                  (clam*dlam3_dri[1]-lam3*dclam_dri[1])/(clam*clam),
                  (clam*dlam3_dri[2]-lam3*dclam_dri[2])/(clam*clam)} ;

  Jac[0] = ri[0]*d1[0]+lam1/clam+rj[0]*d2[0]+rk[0]*d3[0];
  Jac[2] = ri[0]*d1[1]+rj[0]*d2[1]+rk[0]*d3[1];

  Jac[3] = ri[1]*d1[0]+rj[1]*d2[0]+rk[1]*d3[0];
  Jac[1] = ri[1]*d1[1]+lam1/clam+rj[1]*d2[1]+rk[1]*d3[1];

}

/*------------------------------------------------------------------------*/

double FixVolVoro::calc_area(tagint *tag_vec, int icell, int num_faces){

  // Pointer to atom positions
  double **x = atom->x;

  // Coordinates of the vertices
  double vert[num_faces][2];

  // Find coordinates of each vertex
  for (int n = 0; n < num_faces; n++) {

    // Indices of current triangulation
    int nu1 = n;
    int nu2 = n+1;

    // Wrap back to first vertex for final term
    if (nu2 == num_faces) {
        nu2 = 0;
    }

    // local id of nu1
    int j_nu1 = domain->closest_image(icell,atom->map(tag_vec[nu1]));

    // local id of nu2
    int j_nu2 = domain->closest_image(icell,atom->map(tag_vec[nu2]));

    // Coordinates of current triangulation
    double xn[3] = {x[icell][0],x[j_nu1][0],x[j_nu2][0]};
    double yn[3] = {x[icell][1],x[j_nu1][1],x[j_nu2][1]};

    // Store circumcenter
    calc_cc(xn, yn, vert[n]);

  }

  double Area = 0.0;
  for (int n = 0; n < num_faces; n++) {

    // Indices of current and next vertex
    int mu1 = n;
    int mu2 = n+1;

    // Wrap back to first vertex for final term
    if (mu2 == num_faces) {
        mu2 = 0;
    }

    // Sum the area contribution
    Area += 0.5*(vert[mu1][0]*vert[mu2][1] - vert[mu1][1]*vert[mu2][0]);

  }

  return Area;

}

/*------------------------------------------------------------------------*/

int FixVolVoro::pack_forward_comm(int n, int *list, double *buf,
                                    int /*pbc_flag*/, int * /*pbc*/)

{

  int i,j,m;

  m = 0;

  tagint **dtf = atom->iarray[index];
  for (i = 0; i < n; i++) {
    j = list[i];
    for (int k = 0; k < max_faces; k++) {
        buf[m++] = ubuf(dtf[j][k]).d;
    }
  }

  return m;
  
}

void FixVolVoro::unpack_forward_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;

  tagint **dtf = atom->iarray[index];
  for (i = first; i < last; i++) {
      for (int j = 0; j < max_faces; j++) {
        dtf[i][j] = (tagint) ubuf(buf[m++]).i;
    }
  }

}

/* ---------------------------------------------------------------------- */

int FixVolVoro::pack_reverse_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;

  for (i = first; i < last; i++) {
    for (int v = 0; v < 5; v++) {
        buf[m++] = total_virial[i][v];
    }
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void FixVolVoro::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,j,m;

  m = 0;

  for (i = 0; i < n; i++) {
    j = list[i];
    for (int v = 0; v < 5; v++) {
      total_virial[j][v] += buf[m++];
    }
  }
}

/* ----------------------------------------------------------------------
   pack values in local atom-based arrays for exchange with another proc
------------------------------------------------------------------------- */

int FixVolVoro::pack_exchange(int i, double *buf)
{
  int n = 0;
  if (peratom_flag) {
    for (int m = 0; m < size_peratom_cols; m++)
      buf[n++] = array_atom[i][m];
  }
  return n;
}

/* ----------------------------------------------------------------------
   unpack values into local atom-based arrays after exchange
------------------------------------------------------------------------- */

int FixVolVoro::unpack_exchange(int nlocal, double *buf)
{
  int n = 0;
  if (peratom_flag) {
    for (int m = 0; m < size_peratom_cols; m++)
      array_atom[nlocal][m] = buf[n++];
  }
  return n;
}

/* ----------------------------------------------------------------------
   allocate local atom-based arrays
------------------------------------------------------------------------- */

void FixVolVoro::grow_arrays(int nmax)
{
  if (peratom_flag) {
    memory->grow(array_atom,nmax,size_peratom_cols,"fix_wall_gran:array_atom");
  }
}

/* ----------------------------------------------------------------------
   initialize one atom's array values, called when atom is created
------------------------------------------------------------------------- */

void FixVolVoro::set_arrays(int i)
{
  if (peratom_flag) {
    for (int m = 0; m < size_peratom_cols; m++)
      array_atom[i][m] = 0;
  }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixVolVoro::memory_usage()
{
  int nmax = atom->nmax;
  double bytes = (double)nmax*6 * sizeof(double);
  if (peratom_flag) bytes += (double)nmax * size_peratom_cols * sizeof(double);
  return bytes;
}