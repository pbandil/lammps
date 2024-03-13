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
#include "fix_spvforce.h"

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

FixSpvForce::FixSpvForce(LAMMPS *lmp, int narg, char **arg) :
    Fix(lmp, narg, arg), id_compute_voronoi(nullptr), voro_area(nullptr), voro_peri(nullptr),
    def_voro_peri(nullptr), wgn(nullptr), idregion(nullptr), region(nullptr)
{
  if (narg < 17) error->all(FLERR, "Illegal fix spvforce command: not sufficient args");

  MPI_Comm_rank(world, &me);
  MPI_Comm_size(world, &nprocs);

  // initialize Marsaglia RNG with processor-unique seed
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

  Kappa = utils::numeric(FLERR, arg[3], false, lmp);
  Apref = utils::numeric(FLERR, arg[4], false, lmp);
  Gamma = utils::numeric(FLERR, arg[5], false, lmp);
  Lambda = utils::numeric(FLERR, arg[6], false, lmp);
  alpha = utils::numeric(FLERR, arg[7], false, lmp);
  ngx = utils::numeric(FLERR, arg[8], false, lmp);
  id_compute_voronoi = utils::strdup(arg[9]);

  Jv = utils::numeric(FLERR, arg[10], false, lmp);
  Jn = utils::numeric(FLERR, arg[11], false, lmp);
  Js = utils::numeric(FLERR, arg[12], false, lmp);
  fa = utils::numeric(FLERR, arg[13], false, lmp);
  var = utils::numeric(FLERR, arg[14], false, lmp);
  gamma_R = utils::numeric(FLERR, arg[15], false, lmp);

  F11 = utils::numeric(FLERR, arg[16], false, lmp);
  F12 = 0.0;


  /*This fix takes in input as per-atom array
  produced by compute voronoi*/

  vcompute = modify->get_compute_by_id(id_compute_voronoi);
  if (!vcompute)
    error->all(FLERR, "Could not find compute ID {} for voronoi compute", id_compute_voronoi);

  //parse values for optional arguments
  /*May be used for future cell proliferation*/

  nevery = 1;    // Using default value for now

  if (narg > 17) {
    idregion = utils::strdup(arg[17]);
    region = domain->get_region_by_id(idregion);
  }

  maxatom = 1;    //previously atom->maxatom
  memory->create(voro_area, maxatom, "spvforce:voro_area");
  memory->create(voro_peri, maxatom, "spvforce:voro_peri");
  memory->create(def_voro_peri, maxatom, "spvforce:def_voro_peri");
}

/* ---------------------------------------------------------------------- */

FixSpvForce::~FixSpvForce()
{
  delete[] id_compute_voronoi;
  delete[] idregion;
  delete wgn;

  memory->destroy(voro_area);
  memory->destroy(voro_peri);
  memory->destroy(def_voro_peri);

  if (new_fix_id && modify->nfix) modify->delete_fix(new_fix_id);
  delete[] new_fix_id;

  // fclose(fp);
}

/* ---------------------------------------------------------------------- */

void FixSpvForce::post_constructor()
{

  // this is all just to make initial space

 // this is all just to make initial space

  new_fix_id = utils::strdup(id + std::string("_FIX_PA")); // This is the name of the new fix property/atom
  modify->add_fix(fmt::format("{} {} property/atom d2_pol 2 ghost yes",new_fix_id, group->names[igroup]));
  // the modify command is creating the fix proptery/atom call
  // d2_pol: d2 refers to the type, pol is the name of the variable

  int tmp1, tmp2;
  index = atom->find_custom("pol",tmp1,tmp2);

  double **pol = atom->darray[index];   //Note that it is a 2D array
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int nall = nlocal + nghost;

  for (int i = 0; i < nall; i++) {
    for (int j = 0; j < 2; j++) {
      if (mask[i] & groupbit) {
        pol[i][j] = 0.0;
      }
    }
  }
}

/* ---------------------------------------------------------------------- */
// returntype classname :: functidentifier(args) 

int FixSpvForce::setmask()
{
  datamask_read = datamask_modify = 0;

  int mask = 0;
  mask |= POST_FORCE;
  mask |= POST_FORCE_RESPA;
  mask |= MIN_POST_FORCE;
  return mask;
}

/*----------------------------------------*/

/* ---------------------------------------------------------------------- */

void FixSpvForce::init()
{
  // set indices and check validity of all computes and variables

  // set index and check validity of region

  /*For future when we include nevery and region ids*/
  if (idregion) {
    region = domain->get_region_by_id(idregion);
    if (!region) error->all(FLERR, "Region {} for fix spvforce does not exist", idregion);
  }

  if (utils::strmatch(update->integrate_style, "^respa")) {
    ilevel_respa = (dynamic_cast<Respa *>(update->integrate))->nlevels - 1;
    if (respa_level >= 0) ilevel_respa = MIN(respa_level, ilevel_respa);
  }

  //Initialize polarity vectors and shape angles

  int tmp1, tmp2;
  index = atom->find_custom("pol", tmp1, tmp2);
  double **pol = atom->darray[index];

  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int nall = nlocal + nghost;

  // Initialize polarity per atom (only for local atoms as ghost atoms will be taken care of by their processors)
  for (int i = 0; i < nlocal; i++) {
     if (mask[i] & groupbit) {
      pol[i][0] = ((double) rand() / (RAND_MAX))*(2*M_PI);       // Angle in radians from [0,2*PI)
      pol[i][1] = pol[i][0];              
     }    
  }

  // communicate these initial values to neighboring processors
  commflag = 3;
  comm->forward_comm(this, 2);  //here 2  is the size/no of columns
}

/* ---------------------------------------------------------------------- */

void FixSpvForce::setup(int vflag)
{
  if (utils::strmatch(update->integrate_style, "^verlet"))
    post_force(vflag);
  else {
    (dynamic_cast<Respa *>(update->integrate))->copy_flevel_f(ilevel_respa);
    post_force_respa(vflag, ilevel_respa, 0);
    (dynamic_cast<Respa *>(update->integrate))->copy_f_flevel(ilevel_respa);
  }
}

/* ---------------------------------------------------------------------- */

void FixSpvForce::min_setup(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */
/*<<<<<<<<<<<<<<<<<<<<<< HELPER FUNCTIONS (BEGIN) >>>>>>>>>>>>>>>>>>>>>>>>>*/
/* ---------------------------------------------------------------------- */

//Function Description: get the circumcircle of a triangle

void calc_cc(double coords_array[][2], const vector<int>& v1,  double *CC){

  double x0 = coords_array[v1[0]][0];
  double x1 = coords_array[v1[1]][0];
  double x2 = coords_array[v1[2]][0];

  double y0 = coords_array[v1[0]][1];
  double y1 = coords_array[v1[1]][1];
  double y2 = coords_array[v1[2]][1];

  double a = sqrt(pow(x2-x1,2)+pow(y2-y1,2));
  double b = sqrt(pow(x2-x0,2)+pow(y2-y0,2));
  double c = sqrt(pow(x1-x0,2)+pow(y1-y0,2));

 // Find Angles A, B and C

  double A = acos((b*b+c*c-a*a)/(2*b*c));
  double B = acos((a*a+c*c-b*b)/(2*a*c));
  double C = acos((a*a+b*b-c*c)/(2*a*b));

  CC[0] = (x0*sin(2*A)+x1*sin(2*B)+x2*sin(2*C))/(sin(2*A)+sin(2*B)+sin(2*C)); // x coord of circumcenter
  CC[1] = (y0*sin(2*A)+y1*sin(2*B)+y2*sin(2*C))/(sin(2*A)+sin(2*B)+sin(2*C)); // y coord of circumcenter
  CC[2] = sqrt(pow(CC[0]-x0,2)+pow(CC[1]-y0,2));  // radius of circumcircle
}

//Function decription: check if a point lies inside the circumcircle

int InsideCC(double **x, int i, const vector<double> &v){
    // find circum circle for the potential DT:

  double dist = sqrt(pow(x[i][0]-v[0],2) + pow(x[i][1]-v[1],2));
  if (dist < v[2]) return 1;   // introduce the tolerance here if you need to
  else return 0;
}

//Function decription: if edge is an internal/shared edge for polygon or not

int edgeshared(const vector<vector<int>> &v,int k, int kk){

int ev1,ev2;

if (kk == 0){
  ev1 = v[k][0];
  ev2 = v[k][1];
}
else if(kk == 1){
  ev1 = v[k][1];
  ev2 = v[k][2];
}
else if(kk == 2){
  ev1 = v[k][2];
  ev2 = v[k][0];
}

for (int i = 0; i < v.size(); i++){
  if (i != k){
    if (ev1 == v[i][0] && ev2 == v[i][1]) return 1;
    else if (ev1 == v[i][1] && ev2 == v[i][0]) return 1;
    else if (ev1 == v[i][1] && ev2 == v[i][2]) return 1;
    else if (ev1 == v[i][2] && ev2 == v[i][1]) return 1;
    else if (ev1 == v[i][0] && ev2 == v[i][2]) return 1;
    else if (ev1 == v[i][2] && ev2 == v[i][0]) return 1;
  }
}

return 0;
}
 
// function description: Order the vertices in CCW manner  

 void order_vertices_list (vector<int>& cvl, const vector<vector<double>> &dtcc, int cid){

 // first find the leftmost-bottommost vertex of voronoi polygon

  double ymin = INFINITY;
  int first = -1;    // initialise index for minimum y coordinate
  float epsilon = 0.0001f;
  vector<int> temp;

  for(int i = 0; i < cvl.size(); i++){
     if (ymin > dtcc[cvl[i]][1]){
      ymin = dtcc[cvl[i]][1];
      first = cvl[i];
    }
  }

  double theta[cvl.size()], theta_sorted[cvl.size()];

  // compute angles of remaining vertices w.r.t the first vertex

  for (int i = 0; i < cvl.size(); i++){
    if (cvl[i] != first){
      double x1 = dtcc[first][0];
      double y1 = dtcc[first][1];
      double x2 = dtcc[cvl[i]][0];
      double y2 = dtcc[cvl[i]][1];

      double a1 = (x2-x1)/sqrt(pow(x2-x1,2)+pow(y2-y1,2)); // x comp of vector joining both vertices
      theta[i] = acos(a1); // [0, M_PI]
    } else {
      theta[i] = -1*M_PI;
    }
    theta_sorted[i] = theta[i];
  }

  // Now sort the vertices based on their angles
  sort(theta_sorted, theta_sorted + cvl.size());

  for (int i = 0; i < cvl.size(); i++){
    for (int j = 0; j < cvl.size(); j++){
      if (theta_sorted[i] == theta[j]) temp.push_back(cvl[j]);
    }
  }

  cvl = temp;

}

// function description: get voronoi neighbors list (local ids) for an owned atom

void get_voro_neighs(int i, const vector<vector<int>> &DTmesh, const vector<int> &cvl, vector<int> &cnl){

// First traverse though the vertex list for cell i to identify the rows of corresponding DTs
  for (int j = 0; j < cvl.size(); j++){
    int id = cvl[j]; // store the index of that delaunay triangle
    for (int k = 0; k < 3; k++){
      if (DTmesh[id][k] != i && std::find(cnl.begin(), cnl.end(), DTmesh[id][k]) == cnl.end()) cnl.push_back(DTmesh[id][k]);
    }
  }
}

// helper function: finds the cross product of 2 vectors

void getCP(double *cp, double *v1, double *v2)
{
  cp[0] = v1[1] * v2[2] - v1[2] * v2[1];
  cp[1] = v1[2] * v2[0] - v1[0] * v2[2];
  cp[2] = v1[0] * v2[1] - v1[1] * v2[0];
}

// helper function: normalizes a vector

void normalize(double *v)
{
  double norm = pow(pow(v[0], 2) + pow(v[1], 2) + pow(v[2], 2), 0.5);
  v[0] = v[0] / norm;
  v[1] = v[1] / norm;
  v[2] = v[2] / norm;
}

void Jacobian(double **x, vector<int> &dt, int p, double drnu_dRi[][3], double F[][3])
{
  // identify j and k

  int j, k;
  double Jac[3][3] = {0.0};

  if (dt[0] == p) {
    j = dt[1];
    k = dt[2];
  } else if (dt[1] == p) {
    j = dt[0];
    k = dt[2];
  } else if (dt[2] == p) {
    j = dt[0];
    k = dt[1];
  }

  double ri[3] = {x[p][0], x[p][1], 0};
  double rj[3] = {x[j][0], x[j][1], 0};
  double rk[3] = {x[k][0], x[k][1], 0};

  double li = sqrt(pow(rj[0] - rk[0], 2) + pow(rj[1] - rk[1], 2));
  double lj = sqrt(pow(ri[0] - rk[0], 2) + pow(ri[1] - rk[1], 2));
  double lk = sqrt(pow(ri[0] - rj[0], 2) + pow(ri[1] - rj[1], 2));

  double lam1 = li * li * (lj * lj + lk * lk - li * li);
  double lam2 = lj * lj * (lk * lk + li * li - lj * lj);
  double lam3 = lk * lk * (li * li + lj * lj - lk * lk);

  double clam = lam1 + lam2 + lam3;

  double dclam_dri[3] = {-4 * (li * li + lk * lk - lj * lj) * (rk[0] - ri[0]) +
                             4 * (li * li + lj * lj - lk * lk) * (ri[0] - rj[0]),
                         -4 * (li * li + lk * lk - lj * lj) * (rk[1] - ri[1]) +
                             4 * (li * li + lj * lj - lk * lk) * (ri[1] - rj[1]),
                         -4 * (li * li + lk * lk - lj * lj) * (rk[2] - ri[2]) +
                             4 * (li * li + lj * lj - lk * lk) * (ri[2] - rj[2])};

  double dlam1_dri[3] = {2 * li * li * (-(rk[0] - ri[0]) + (ri[0] - rj[0])),
                         2 * li * li * (-(rk[1] - ri[1]) + (ri[1] - rj[1])),
                         2 * li * li * (-(rk[2] - ri[2]) + (ri[2] - rj[2]))};

  double dlam2_dri[3] = {
      -2 * (li * li + lk * lk - 2 * lj * lj) * (rk[0] - ri[0]) + 2 * lj * lj * (ri[0] - rj[0]),
      -2 * (li * li + lk * lk - 2 * lj * lj) * (rk[1] - ri[1]) + 2 * lj * lj * (ri[1] - rj[1]),
      -2 * (li * li + lk * lk - 2 * lj * lj) * (rk[2] - ri[2]) + 2 * lj * lj * (ri[2] - rj[2])};

  double dlam3_dri[3] = {
      2 * (li * li + lj * lj - 2 * lk * lk) * (ri[0] - rj[0]) - 2 * lk * lk * (rk[0] - ri[0]),
      2 * (li * li + lj * lj - 2 * lk * lk) * (ri[1] - rj[1]) - 2 * lk * lk * (rk[1] - ri[1]),
      2 * (li * li + lj * lj - 2 * lk * lk) * (ri[2] - rj[2]) - 2 * lk * lk * (rk[2] - ri[2])};

  double d1[3] = {(clam * dlam1_dri[0] - lam1 * dclam_dri[0]) / (clam * clam),
                  (clam * dlam1_dri[1] - lam1 * dclam_dri[1]) / (clam * clam),
                  (clam * dlam1_dri[2] - lam1 * dclam_dri[2]) / (clam * clam)};

  double d2[3] = {(clam * dlam2_dri[0] - lam2 * dclam_dri[0]) / (clam * clam),
                  (clam * dlam2_dri[1] - lam2 * dclam_dri[1]) / (clam * clam),
                  (clam * dlam2_dri[2] - lam2 * dclam_dri[2]) / (clam * clam)};

  double d3[3] = {(clam * dlam3_dri[0] - lam3 * dclam_dri[0]) / (clam * clam),
                  (clam * dlam3_dri[1] - lam3 * dclam_dri[1]) / (clam * clam),
                  (clam * dlam3_dri[2] - lam3 * dclam_dri[2]) / (clam * clam)};

  Jac[0][0] = ri[0] * d1[0] + lam1 / clam + rj[0] * d2[0] + rk[0] * d3[0];
  Jac[0][1] = ri[0] * d1[1] + rj[0] * d2[1] + rk[0] * d3[1];
  Jac[0][2] = ri[0] * d1[2] + rj[0] * d2[2] + rk[0] * d3[2];

  Jac[1][0] = ri[1] * d1[0] + rj[1] * d2[0] + rk[1] * d3[0];
  Jac[1][1] = ri[1] * d1[1] + lam1 / clam + rj[1] * d2[1] + rk[1] * d3[1];
  Jac[1][2] = ri[1] * d1[2] + rj[1] * d2[2] + rk[1] * d3[2];

  Jac[2][0] = ri[2] * d1[0] + rj[2] * d2[0] + rk[2] * d3[0];
  Jac[2][1] = ri[2] * d1[1] + rj[2] * d2[1] + rk[2] * d3[1];
  Jac[2][2] = ri[2] * d1[2] + lam1 / clam + rj[2] * d2[2] + rk[2] * d3[2];

  //Find the gradient wrt underformed cell center

  for (int I = 0; I < 3; I++) {
    for (int J = 0; J < 3; J++) {
      drnu_dRi[I][J] = F[I][0]*Jac[0][J] + F[I][1]*Jac[1][J] + F[I][2]*Jac[2][J];
    }
  }
  //printf("Jacobian: %f --> %f -->%f -->%f \n", Jac[0][0], Jac[0][1], Jac[1][0], Jac[1][1]);
  //printf("New Jac: %f --> %f -->%f -->%f \n", drnu_dRi[0][0], drnu_dRi[0][1], drnu_dRi[1][0], drnu_dRi[1][1]);
}

// helper function: multiplies a vector and a matrix

void vector_matrix(double *result, double *vec, double mat[][3])
{
  for (int i = 0; i < 3; i++) {
    result[i] = vec[0] * mat[0][i] + vec[1] * mat[1][i] + vec[2] * mat[2][i];
  }
}

/* ---------------------------------------------------------------------- */
/*<<<<<<<<<<<<<<<<<<<<<< HELPER FUNCTIONS (END) >>>>>>>>>>>>>>>>>>>>>>>>>*/
/* ---------------------------------------------------------------------- */

/*Modify The code to account for alignment: Think of total lagrangian Formulation*/

void FixSpvForce::post_force(int vflag)
{
  double **x = atom->x;
  double **f = atom->f;
  double **v = atom->v;
  int *mask = atom->mask;
  imageint *image = atom->image;
  tagint *tag = atom->tag;
  double dt = update->dt;

  if (update->ntimestep % nevery) return;

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

  // Possibly resize arrays

  if (atom->nmax > maxatom) {
    memory->destroy(voro_area);
    memory->destroy(voro_peri);
    memory->destroy(def_voro_peri);
    maxatom = atom->nmax;
    memory->create(voro_area, maxatom, "spvforce:voro_area");
    memory->create(voro_peri, maxatom, "spvforce:voro_peri");
    memory->create(def_voro_peri, maxatom, "spvforce:def_voro_peri");
  }

  // accumulate results of attributes,computes,fixes,variables to local copy
  // compute/fix/variable may invoke computes so wrap with clear/add (from fix ave atom)

  // Initialize arrays to zero
  for (int i = 0; i < nall; i++) {
    voro_area[i] = 0.0;
    voro_peri[i] = 0.0;
    def_voro_peri[i] = 0.0;
  }

  // Invoke compute
  modify->clearstep_compute();
  vcompute = modify->get_compute_by_id(id_compute_voronoi);
  if (!(vcompute->invoked_flag & Compute::INVOKED_PERATOM)) {
    vcompute->compute_peratom();
    vcompute->invoked_flag |= Compute::INVOKED_PERATOM;
  }

  // Fill voro_data and voro_area0 with values from compute voronoi
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      voro_area[i] = vcompute->array_atom[i][0];
      voro_peri[i] = vcompute->array_atom[i][2];
    }
  }

  /*Once you have voronoi information: identift atoms to delete in the next time step*/

  // forward communication of voronoi data:

  commflag = 1;
  comm->forward_comm(this, 1);

  // commflag = 2;
  // comm->forward_comm(this, 1);

   /*Construct Delaunay Triangulation for a set of nall points using Bowyer Watson Algorithm*/

 //declare dynamic containers to store details of Delaunay Triangulation

 vector<vector<int>> Del_Tri_mesh;
 vector<vector<double>> Del_Tri_cc; 

 /*Find min/max of x-y coords to find the super triangle*/ 

 double ymax = -1*INFINITY, ymin = INFINITY;
 double xmax = -1*INFINITY, xmin = INFINITY; 

 for (int i = 0; i < nall; i++){
  if (ymax < x[i][1]) ymax = x[i][1];
  if (ymin > x[i][1]) ymin = x[i][1];
  if (xmax < x[i][0]) xmax = x[i][0];
  if (xmin > x[i][0]) xmin = x[i][0];
 }

 double dmax;

 if (xmax-xmin > ymax-ymin) dmax = 3*(xmax-xmin);
 else dmax = 3*(ymax-ymin);

 double xcen = 0.5*(xmin+xmax);
 double ycen = 0.5*(ymin+ymax);

 // add super triangle to the Del_Tri_mesh

 double coords_array[nall+3][2] = {0.0};

 for (int i = 0; i < nall; i++){
  coords_array[i][0] = x[i][0];
  coords_array[i][1] = x[i][1];
 }

 coords_array[nall][0] = xcen-0.866*dmax;
 coords_array[nall][1] = ycen-0.5*dmax;
 coords_array[nall+1][0] = xcen+0.866*dmax;
 coords_array[nall+1][1] = ycen-0.5*dmax;
 coords_array[nall+2][0] = xcen;
 coords_array[nall+2][1] = ycen+dmax;

  double temp_CC_mod[3] = {0.0};

  Del_Tri_mesh.push_back({nall, nall+1, nall+2, 0});                      // add the super triangle which is flagged as intersecting (0)
  double temp_CC[3] = {0.0};
  calc_cc(coords_array, Del_Tri_mesh[0], temp_CC);
  Del_Tri_cc.push_back({temp_CC[0], temp_CC[1], temp_CC[2]});

 for (int i = 0; i < nall; i++){

  vector<vector<int>> bad_Del_Tri;
  vector<vector<int>> bound_edge;

  for (int j = 0; j < Del_Tri_mesh.size(); j++){
    if (InsideCC(x, i, Del_Tri_cc[j])) {
      bad_Del_Tri.push_back({Del_Tri_mesh[j][0], Del_Tri_mesh[j][1], Del_Tri_mesh[j][2]});
      Del_Tri_mesh[j][3] = 0;      // mark the triangle to be removed
    }  
    else Del_Tri_mesh[j][3] = 1;   // mark the triangle as valid for now
  }

  //Find the boundary edges for the corresponding polygon

  for (int k = 0; k < bad_Del_Tri.size(); k++){
    for (int kk = 0; kk < 3; kk++){
      if (!edgeshared(bad_Del_Tri,k, kk)){
        if (kk == 0) bound_edge.push_back({bad_Del_Tri[k][0], bad_Del_Tri[k][1]});
        else if(kk == 1) bound_edge.push_back({bad_Del_Tri[k][1], bad_Del_Tri[k][2]}); 
        else if(kk == 2) bound_edge.push_back({bad_Del_Tri[k][2], bad_Del_Tri[k][0]});
      }           
    }   
  }

  // for each triangle in bad triangle list remove from triangulation

  for (int l = 0; l < Del_Tri_mesh.size(); l++){
     if (Del_Tri_mesh[l][3] == 0){
        Del_Tri_mesh.erase(Del_Tri_mesh.begin() + l);       // remove triangle from triangulation list
        Del_Tri_cc.erase(Del_Tri_cc.begin() + l);           // Do the same for auxiliary info list
        l--; 
     }
  }

  for (int m=0; m < bound_edge.size(); m++){                           
    Del_Tri_mesh.push_back({i, bound_edge[m][0], bound_edge[m][1], 0});  
    calc_cc(coords_array, Del_Tri_mesh[Del_Tri_mesh.size()-1], temp_CC);
    Del_Tri_cc.push_back({temp_CC[0], temp_CC[1], temp_CC[2]});
   } 
 }

 // Clean up the triangulation mesh

  for (int i = 0; i < Del_Tri_mesh.size(); i++){
    if (Del_Tri_mesh[i][0] >= nall || Del_Tri_mesh[i][1] >= nall || Del_Tri_mesh[i][2] >= nall){
        Del_Tri_mesh.erase(Del_Tri_mesh.begin() + i);       // remove triangle from triangulation list
        Del_Tri_cc.erase(Del_Tri_cc.begin() + i);           // Do the same for auxiliary info list
        i--; 
    }
  }

  vector<vector<int>> cell_vertices_list(nall);

  for(int i = 0; i < nall; i++){
    for(int j = 0; j < Del_Tri_mesh.size(); j++){
      if(i == Del_Tri_mesh[j][0] || i == Del_Tri_mesh[j][1] || i == Del_Tri_mesh[j][2]){
        cell_vertices_list[i].push_back(j);         // store the vertex id in the row for cell i
      }
    }
  }

  /* Now, we want to order the vertices so that we are looping in a consistent 
      direction for all cells*/

  for(int i = 0; i < nall; i++){
    if(!cell_vertices_list[i].empty()) order_vertices_list(cell_vertices_list[i], Del_Tri_cc, i);
  }

  // create a neighbours list for [0,nlocal) i.e. owned atoms to store their voronoi neighbors

  vector<vector<int>> cell_neighs_list(nlocal);

 for (int i = 0; i < nlocal; i++){
      get_voro_neighs(i, Del_Tri_mesh, cell_vertices_list[i], cell_neighs_list[i]);
      cell_neighs_list[i].push_back(i);  // to add force contribution from the cell itself
  }

  /*Once you have Delaunay in undeformed config: find Lij i.e. the junction vector for each each junction*/

   double defGrad[3][3] = {0.0};
   int curr_step = update->ntimestep;
   int nsteps = update->nsteps;

   if (curr_step <= nsteps){
    defGrad[0][0] = curr_step*(F11-1.0)/nsteps + 1.0;
    //printf("curr_step = %d, def grad F1e = %f, totoal step = %d \n", curr_step, defGrad[0][0], nsteps);
   }
   else{
    defGrad[0][0] = F11;
    //printf("def grad 2nd part is: %f at current step %d\n", defGrad[0][0], curr_step);
   }
   
   defGrad[0][1] = F12;
   defGrad[1][1] = 1.0/defGrad[0][0];
   defGrad[2][2] = 1.0;
  
   vector<vector<double>> Del_Tri_cc_def;  //declare new vector to store 
   Del_Tri_cc_def = Del_Tri_cc;

   //Deform the vertex coordinates according to deformation gradient
   for (int i = 0; i < Del_Tri_cc.size(); i++){
    Del_Tri_cc_def[i][0] = defGrad[0][0]*Del_Tri_cc[i][0] + defGrad[0][1]*Del_Tri_cc[i][1];
    Del_Tri_cc_def[i][1] = defGrad[1][0]*Del_Tri_cc[i][0] + defGrad[1][1]*Del_Tri_cc[i][1];
   }

   for (int i = 0; i < nlocal; i++) {
     int num_vertices = cell_vertices_list[i].size();
     double edge = 0;
     double rmu_x, rmu_y, rmu_next_x, rmu_next_y;
     int mu, mu_next;
     for (int j = 0; j < num_vertices; j++) {
       mu = cell_vertices_list[i][j];
       if (j <= num_vertices - 2) {
         mu_next = cell_vertices_list[i][j + 1];
       } else {
         mu_next = cell_vertices_list[i][0];
       }
       rmu_x = Del_Tri_cc_def[mu][0];
       rmu_y = Del_Tri_cc_def[mu][1];
       rmu_next_x = Del_Tri_cc_def[mu_next][0];
       rmu_next_y = Del_Tri_cc_def[mu_next][1];
       edge = edge + sqrt(pow(rmu_next_x - rmu_x, 2.0) + pow(rmu_next_y - rmu_y, 2.0));
     }
     def_voro_peri[i] = edge;
     //double peri = voro_peri[i]-2*voro_area[i];
     //printf("deformed peri for cell %d --> %f --> %f\n", tag[i], def_voro_peri[i], peri);
   }

  //We need to forward communicate the new perimeters here
  commflag = 2;
  comm->forward_comm(this, 1);

  /*First do Voronoi Force Calculation*/ 

  double ngy = sqrt(1 - ngx * ngx);    // y component of groove direction
  double alph_jun;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      if (region && !region->match(x[i][0], x[i][1], x[i][2])) {
        alph_jun = 0.0;
      } else {
        alph_jun = alpha;
      }

      double F_t1[3] = {0.0};
      double F_t2[3] = {0.0};
      double F_t3[3] = {0.0};

      for (int j = 0; j < cell_neighs_list[i].size(); j++) {
        int current_cell = cell_neighs_list[i][j];
        double vertex_force_sum_t1[3] = {0.0};
        double vertex_force_sum_t2[3] = {0.0};
        double vertex_force_sum_t3[3] = {0.0};

        // First term values needed
        double area = voro_area[current_cell];
        double elasticity_area = (Kappa / 2.0) * (area - Apref);

        // Second term values needed
        //double perimeter = voro_peri[current_cell] - 2 * area;
        double perimeter = def_voro_peri[current_cell];
        double gamma_perimeter = Gamma * (perimeter);

        int num_vert = cell_vertices_list[current_cell].size();
        int vcount = 0;
        int current_vert;

        // Looping through vertices of cell j
        while (vcount < num_vert) {

          current_vert = cell_vertices_list[current_cell][vcount];

          // calculate jacobian and forces only if vertex belongs to cell i as well:
          if (std::find(cell_vertices_list[i].begin(), cell_vertices_list[i].end(),
                        current_vert) != cell_vertices_list[i].end()) {

            int pn[2] = {0};    // store previous and next vertex ids
            if (vcount == 0) {
              pn[0] = cell_vertices_list[current_cell][num_vert - 1];
              pn[1] = cell_vertices_list[current_cell][1];
            } else if (vcount == num_vert - 1) {
              pn[0] = cell_vertices_list[current_cell][vcount - 1];
              pn[1] = cell_vertices_list[current_cell][0];
            } else {
              pn[0] = cell_vertices_list[current_cell][vcount - 1];
              pn[1] = cell_vertices_list[current_cell][vcount + 1];
            }

            // First term stuff
            double rprevnext[3] = {Del_Tri_cc_def[pn[1]][0] - Del_Tri_cc_def[pn[0]][0],
                                   Del_Tri_cc_def[pn[1]][1] - Del_Tri_cc_def[pn[0]][1], 0.0};
            double cp[3] = {0};
            double N[3] = {0, 0, 1};    // normal vector to the plane of cell layer (2D)
            getCP(cp, rprevnext, N);

            // Second term stuff
            double rcurrprev[3] = {Del_Tri_cc_def[current_vert][0] - Del_Tri_cc_def[pn[0]][0],
                                   Del_Tri_cc_def[current_vert][1] - Del_Tri_cc_def[pn[0]][1], 0.0};
            double rnextcurr[3] = {Del_Tri_cc_def[pn[1]][0] - Del_Tri_cc_def[current_vert][0],
                                   Del_Tri_cc_def[pn[1]][1] - Del_Tri_cc_def[current_vert][1], 0.0};
            normalize(rcurrprev);
            normalize(rnextcurr);
            double rhatdiff_t2[3] = {rcurrprev[0] - rnextcurr[0], rcurrprev[1] - rnextcurr[1], 0.0};

            // Third term stuff

            // Find the angle between edge rcurrprev, rnextcurr and groove direction
            double s2th_curr_prev = 1 - pow(rcurrprev[0] * ngx + rcurrprev[1] * ngy, 2.0);
            double s2th_next_curr = 1 - pow(rnextcurr[0] * ngx + rnextcurr[1] * ngy, 2.0) - 1;
            double Lam_curr_prev = Lambda * (1 + alph_jun * s2th_curr_prev);
            double Lam_next_curr = Lambda * (1 + alph_jun * s2th_next_curr);

            double rhatdiff_t3[3] = {Lam_curr_prev * rcurrprev[0] - Lam_next_curr * rnextcurr[0],
                                     Lam_curr_prev * rcurrprev[1] - Lam_next_curr * rnextcurr[1],
                                     0.0};

            double drnu_dRi[3][3] = {0.0};
            Jacobian(x, Del_Tri_mesh[current_vert], i, drnu_dRi, defGrad);

            double result_t1[3] = {0};
            double result_t2[3] = {0};
            double result_t3[3] = {0};

            // Term 1 forces
            vector_matrix(result_t1, cp, drnu_dRi);
            vertex_force_sum_t1[0] += result_t1[0];
            vertex_force_sum_t1[1] += result_t1[1];
            vertex_force_sum_t1[2] += result_t1[2];

            // Term 2 forces
            vector_matrix(result_t2, rhatdiff_t2, drnu_dRi);
            vertex_force_sum_t2[0] += result_t2[0];
            vertex_force_sum_t2[1] += result_t2[1];
            vertex_force_sum_t2[2] += result_t2[2];

            // Term 3 forces
            vector_matrix(result_t3, rhatdiff_t3, drnu_dRi);
            vertex_force_sum_t3[0] += result_t3[0];
            vertex_force_sum_t3[1] += result_t3[1];
            vertex_force_sum_t3[2] += result_t3[2];
          }
          vcount++;
        }
        F_t1[0] += elasticity_area * vertex_force_sum_t1[0];
        F_t1[1] += elasticity_area * vertex_force_sum_t1[1];
        F_t1[2] += elasticity_area * vertex_force_sum_t1[2];

        F_t2[0] += gamma_perimeter * vertex_force_sum_t2[0];
        F_t2[1] += gamma_perimeter * vertex_force_sum_t2[1];
        F_t2[2] += gamma_perimeter * vertex_force_sum_t2[2];

        F_t3[0] += vertex_force_sum_t3[0];
        F_t3[1] += vertex_force_sum_t3[1];
        F_t3[2] += vertex_force_sum_t3[2];
      }
      double fx = -F_t1[0] - F_t2[0] - F_t3[0];
      double fy = -F_t1[1] - F_t2[1] - F_t3[1];
      double fz = -F_t1[2] - F_t2[2] - F_t3[2];
      f[i][0] += fx;
      f[i][1] += fy;
      f[i][2] += fz;
    }
  }

  /*Now perform force calculation for self-propelled forces*/

  int tmp1, tmp2;
  index = atom->find_custom("pol", tmp1, tmp2);
  double **pol = atom->darray[index];

  // Find shape angle (phi_s) for each owned atom at current time step
  double phi_s[nlocal] = {0.0};

  double eig_vx, eig_vy;
  double x_def, y_def;   //Cell centers need to be deformed according to def gradient to find shape 

  for (int i = 0; i < nlocal; i++) {
    double A = 0.0;
    double B = 0.0;
    double C = 0.0;
    double rnu_x, rnu_y;
    int curr_vert;
    x_def = defGrad[0][0]*x[i][0] + defGrad[0][1]*x[i][1];
    y_def = defGrad[1][0]*x[i][0] + defGrad[1][1]*x[i][1];

    for (int j = 0; j < cell_vertices_list[i].size(); j++) {
      curr_vert = cell_vertices_list[i][j];
      rnu_x = Del_Tri_cc_def[curr_vert][0];
      rnu_y = Del_Tri_cc_def[curr_vert][1];
      A = A + pow(rnu_x - x_def, 2.0);
      B = B + (rnu_x - x_def) * (rnu_y - y_def);
      C = C + pow(rnu_y - y_def, 2.0);
    }
    double lam1 = 0.5 * (A + C + sqrt(pow(A - C, 2) + 4 * B * B));
    double lam2 = 0.5 * (A + C - sqrt(pow(A - C, 2)) + 4 * B * B);
    double lamM = (lam1 > lam2) ? lam1 : lam2;

    if (B != 0) {
      eig_vx = B / sqrt(B * B + (lamM - A) * (lamM - A));
      eig_vy = (lamM - A) / sqrt(B * B + (lamM - A) * (lamM - A));
    } else if (B == 0 && lamM == A) {
      eig_vx = 1.0;
      eig_vy = 0.0;
    } else {
      eig_vx = 0.0;
      eig_vy = 1.0;
    }
    phi_s[i] = fmod(atan2(eig_vy, eig_vx), 2 * M_PI);
    //pol[i][1] = phi_s[i];
  }

//Update shape angle in polarity array and make sure it does not flip like a nematic by comparing with previous value

  // for (int i = 0; i < nlocal; i++) {
  //   if (cos(pol[i][1] - phi_s[i]) < 0) {
  //     pol[i][1] = phi_s[i] + M_PI;
  //   } else {
  //     pol[i][1] = phi_s[i];
  //   }
  //   /*For now make theta = phi_s*/
  //   //pol[i][0] = pol[i][1];
  //   //printf("Polarity cell %d-->[%f,  %f,  %f] \n", tag[i], pol[i][1], cos(pol[i][1]), sin(pol[i][1]));
  // }


  double dThetadt, dthdt_vel, dthdt_shp, dthdt_ngh;
  double tau;

  for (int i = 0; i < nlocal; i++) {

    /*Self-alignment term*/
    double phi_i = pol[i][0];    //initialise it

    if (v[i][0] != 0.0 && v[i][1] != 0.0) { phi_i = fmod(atan2(v[i][1], v[i][0]), 2 * M_PI); }
    double tau_vel = -Jv*sin(pol[i][0] - phi_i);
    //dthdt_vel = -Jv * sin(pol[i][0] - phi_i);

    /*Neighbour alignment term*/

    double tau_ngh = 0.0;
    for (int j = 0; j < cell_neighs_list[i].size() - 1; j++) {
      int neighid = cell_neighs_list[i][j];
      tau_ngh += cos(pol[i][0])*sin(pol[neighid][0]) - sin(pol[i][0])*cos(pol[neighid][0]);
    }
    tau_ngh = Jn*tau_ngh;

    // dthdt_ngh = 0.0;
    // for (int j = 0; j < cell_neighs_list[i].size() - 1; j++) {
    //   int neighid = cell_neighs_list[i][j];
    //   dthdt_ngh += -Jn * sin(pol[i][0] - pol[neighid][0]);
    // }

    /*Shape alignment term*/

    double nx = cos(pol[i][0]); //x direction of polarity
    double ny = sin(pol[i][0]); //y direction of polarity
    double px = cos(phi_s[i]); //x-eigen direction of shape
    double py = sin(phi_s[i]); //y-eigen direction of shape

    double tau_shp = -2 * Js * sin(pol[i][0] - phi_s[i]) * (nx * px + ny * py);
    // dthdt_shp = 2 * Js * (nx * px + ny * py) * (nx * py - ny * px);

    tau = tau_vel + tau_ngh + tau_shp;
    dThetadt = tau/gamma_R + wgn->gaussian(0, var);

    //dThetadt = dthdt_vel + dthdt_ngh + dthdt_shp + wgn->gaussian(0, var);

    pol[i][0] = pol[i][0] + dt * dThetadt;

    // add the polarity term to the force equation
    f[i][0] += fa * cos(pol[i][0]);
    f[i][1] += fa * sin(pol[i][0]);
    f[i][2] += fa * 0.0;
  }

  // communicate the updated polarity values for ghost atoms
  commflag = 3;
  comm->forward_comm(this, 2);
}

/* ---------------------------------------------------------------------- */

void FixSpvForce::post_force_respa(int vflag, int ilevel, int /*iloop*/)
{
  if (ilevel == ilevel_respa) post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixSpvForce::min_post_force(int vflag)
{
  post_force(vflag);
}

/*------------------------------------------------------------------------*/

int FixSpvForce::pack_forward_comm(int n, int *list, double *buf, int /*pbc_flag*/, int * /*pbc*/)

{

  int i, j, k, m;

  m = 0;

  if (commflag == 1) {
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = voro_area[j];
    }
  } else if (commflag == 2) {
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = def_voro_peri[j];
    }
  } else if (commflag == 3) {
    int tmp1, tmp2;
    index = atom->find_custom("pol",tmp1,tmp2);
    double **pol = atom->darray[index];
    for (i = 0; i < n; i++) {
      j = list[i];
      for (k = 0; k < 2; k++) { buf[m++] = pol[j][k]; }
    }
  }
  return m;
}

void FixSpvForce::unpack_forward_comm(int n, int first, double *buf)
{
  int i, j, m, last;

  m = 0;
  last = first + n;

  if (commflag == 1) {
    for (i = first; i < last; i++) { voro_area[i] = buf[m++]; }
  } 
  else if (commflag == 2) {
    for (i = first; i < last; i++) { def_voro_peri[i] = buf[m++]; }
  } 
  else if (commflag == 3) {
     int tmp1, tmp2;
    index = atom->find_custom("pol",tmp1,tmp2);
    double **pol = atom->darray[index];
    for (i = first; i < last; i++) {
      for (j = 0; j < 2; j++) { pol[i][j] = buf[m++]; }
    }
  }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixSpvForce::memory_usage()
{
  int maxatom = atom->nmax;
  double bytes = (double) maxatom * 4 * sizeof(double);
  return bytes;
}

/*--------------------------------------------------------------------
                        END OF MAIN CODE
---------------------------------------------------------------------*/
