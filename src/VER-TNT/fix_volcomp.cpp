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

#include "fix_volcomp.h"

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

FixVolComp::FixVolComp(LAMMPS *lmp, int narg, char **arg) :
    Fix(lmp, narg, arg), voro_data(nullptr), voro_area0(nullptr), id_compute_voronoi(nullptr)
{
  if (narg < 5) error->all(FLERR, "Illegal fix volcomp command: not sufficient args");

  MPI_Comm_rank(world,&me);
  MPI_Comm_size(world, &nprocs);

  dynamic_group_allow = 1;
  energy_peratom_flag = 1;
  virial_global_flag = virial_peratom_flag = 1;
  thermo_energy = thermo_virial = 1;

  respa_level_support = 1;
  ilevel_respa = 0;

  // Parse first two arguments: elasticity and preferred area
  Elasticity = utils::numeric(FLERR,arg[3],false,lmp);
  Apref = utils::numeric(FLERR,arg[4],false,lmp);

  // Parse compute id
  id_compute_voronoi = utils::strdup(arg[5]);
  vcompute = modify->get_compute_by_id(id_compute_voronoi);
  if (!vcompute) error->all(FLERR,"Could not find compute ID {} for voronoi compute", id_compute_voronoi);

  // parse values for optional args
  flag_store_init = 0;
  id_fix_store = nullptr;

  int iarg = 6;
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

  nevery = 1;

  nmax = atom->nmax;
  voro_data = nullptr;
  voro_area0 = nullptr;

}

/* ---------------------------------------------------------------------- */

FixVolComp::~FixVolComp()
{
  delete[] id_compute_voronoi;
  delete[] id_fix_store;
  
  memory->destroy(voro_data);
  if (flag_store_init) memory->destroy(voro_area0);
  
}

/* ---------------------------------------------------------------------- */
// returntype classname :: functidentifier(args) }

int FixVolComp::setmask()
{
  datamask_read = datamask_modify = 0;

  int mask = 0;
  mask |= POST_FORCE;
  mask |= POST_FORCE_RESPA;
  mask |= MIN_POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixVolComp::init()
{
  // set indices and check validity of all computes and variables

  if (utils::strmatch(update->integrate_style, "^respa")) {
    ilevel_respa = (dynamic_cast<Respa *>(update->integrate))->nlevels - 1;
    if (respa_level >= 0) ilevel_respa = MIN(respa_level, ilevel_respa);
  }
}


/* ---------------------------------------------------------------------- */

void FixVolComp::setup(int vflag)
{

  memory->create(voro_data,nmax,"volcomp:voro_data");
  if (flag_store_init) memory->create(voro_area0,nmax,"volcomp:voro_area0");

  if (utils::strmatch(update->integrate_style, "^verlet"))
    post_force(vflag);
  else {
    (dynamic_cast<Respa *>(update->integrate))->copy_flevel_f(ilevel_respa);
    post_force_respa(vflag, ilevel_respa, 0);
    (dynamic_cast<Respa *>(update->integrate))->copy_f_flevel(ilevel_respa);
  }

}

/* ---------------------------------------------------------------------- */

void FixVolComp::min_setup(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */
/*<<<<<<<<<<<<<<<<<<<<<< HELPER FUNCTIONS(BEGIN) >>>>>>>>>>>>>>>>>>>>>>>>>*/
/* ---------------------------------------------------------------------- */

//Function Description: get the circumcircle of a triangle

void FixVolComp::calc_cc(double *xn, double *yn,  double *CC)
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
  CC[2] = sqrt(pow(CC[0]-x0,2)+pow(CC[1]-y0,2));  // radius of circumcircle
}

/* ---------------------------------------------------------------------- */
//Function decription: check if a point lies inside the circumcircle
/* ---------------------------------------------------------------------- */

int InsideCC(double **x, int i, const vector<double> &v){
    // find circum circle for the potential DT:

  double dist = sqrt(pow(x[i][0]-v[0],2) + pow(x[i][1]-v[1],2));
  if (dist < v[2]) return 1;   // introduce the tolerance here if you need to
  else return 0;
}

/* ---------------------------------------------------------------------- */
//FUnction decription: if edge is an internal/shared edge for polygon or not
/* ---------------------------------------------------------------------- */

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
 
 /* ---------------------------------------------------------------------- */
// function description: Order the vertices in CCW manner 
/* ---------------------------------------------------------------------- */

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

/* ---------------------------------------------------------------------- */
// function description: compute angles of remaining vertices w.r.t the first vertex
/* ---------------------------------------------------------------------- */

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

/* ---------------------------------------------------------------------- */
// function description: get voronoi neighbors list (local ids) for an owned atom
/* ---------------------------------------------------------------------- */

void get_voro_neighs(int i, const vector<vector<int>> &DTmesh, const vector<int> &cvl, vector<int> &cnl){

// First traverse though the vertex list for cell i to identify the rows of corresponding DTs
  for (int j = 0; j < cvl.size(); j++){
    int id = cvl[j]; // store the index of that delaunay triangle
    for (int k = 0; k < 3; k++){
      if (DTmesh[id][k] != i && std::find(cnl.begin(), cnl.end(), DTmesh[id][k]) == cnl.end()) cnl.push_back(DTmesh[id][k]);
    }
  }
}

void Jacobian(double **x, vector<int> &dt, int p, double drmu_dri[][3]){
  // identify j and k

  int j,k;

  if (dt[0] == p) {
      j = dt[1];
      k = dt[2];
  }
  else if (dt[1] == p){
      j = dt[0];
      k = dt[2];
  }
  else if (dt[2] == p){
      j = dt[0];
      k = dt[1];
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

  drmu_dri[0][0] = ri[0]*d1[0]+lam1/clam+rj[0]*d2[0]+rk[0]*d3[0];
  drmu_dri[0][1] = ri[0]*d1[1]+rj[0]*d2[1]+rk[0]*d3[1];
  drmu_dri[0][2] = ri[0]*d1[2]+rj[0]*d2[2]+rk[0]*d3[2];

  drmu_dri[1][0] = ri[1]*d1[0]+rj[1]*d2[0]+rk[1]*d3[0];
  drmu_dri[1][1] = ri[1]*d1[1]+lam1/clam+rj[1]*d2[1]+rk[1]*d3[1];
  drmu_dri[1][2] = ri[1]*d1[2]+rj[1]*d2[2]+rk[1]*d3[2];

  drmu_dri[2][0] = ri[2]*d1[0]+rj[2]*d2[0]+rk[2]*d3[0];
  drmu_dri[2][1] = ri[2]*d1[1]+rj[2]*d2[1]+rk[2]*d3[1];
  drmu_dri[2][2] = ri[2]*d1[2]+lam1/clam+rj[2]*d2[2]+rk[2]*d3[2];

}

/* ---------------------------------------------------------------------- */
/*<<<<<<<<<<<<<<<<<<<<<< HELPER FUNCTIONS (END) >>>>>>>>>>>>>>>>>>>>>>>>>*/
/* ---------------------------------------------------------------------- */

void FixVolComp::post_force(int vflag)
{
  double **x = atom->x;
  double **f = atom->f;
  int *mask = atom->mask;
  // tagint *tag = atom->tag;

  int natoms = atom->natoms;
  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int nall = nlocal + nghost;

  double xn[3], yn[3];
  double x0, x1, x2, y0, y1, y2;

  if (update->ntimestep % nevery) return;

  // virial setup

  v_init(vflag);

  int me = comm->me;  //current rank value

  // Possibly resize arrays
  if (atom->nmax > nmax) {
    memory->destroy(voro_data);
    if (flag_store_init) memory->destroy(voro_area0);
    nmax = atom->nmax;
    memory->create(voro_data,nmax,"volcomp:voro_data");
    if (flag_store_init) memory->create(voro_area0,nmax,"volcomp:voro_area0");
  }

  // Initialize arrays to zero
  for (int i = 0; i < nall; i++) {
    voro_data[i] = 0.0;
    if (flag_store_init) voro_area0[i] = 0.0;
  }

  // Invoke compute
  modify->clearstep_compute();
  vcompute = modify->get_compute_by_id(id_compute_voronoi);
  if (!(vcompute->invoked_flag & Compute::INVOKED_PERATOM)) {
    vcompute->compute_peratom();
    vcompute->invoked_flag |= Compute::INVOKED_PERATOM;
  }

  // Define pointer to fix_store
  if (flag_store_init) fstore = modify->get_fix_by_id(id_fix_store);

  // Fill voro_data and voro_area0 with values from compute voronoi
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) voro_data[i] = vcompute->array_atom[i][0];
    if (flag_store_init) {
      if (mask[i] & groupbit) voro_area0[i] = fstore->vector_atom[i];
    }
  }
  
  // forward communication of voronoi data:

  commflag = 0;
  comm->forward_comm(this,1);

  // forward communication of initial cell areas:

  if (flag_store_init) {
    commflag = 1;
    comm->forward_comm(this,1);
  }

 /*Construct Delaunay Triangulation for a set of nall points using Bowyer Watson Algorithm*/

 //declare dynamic containers to store details of Delaunay Triangulation

 vector<vector<int>> Del_Tri_mesh;
 vector<vector<double>> Del_Tri_cc; 

 /*Find min/max of x-y coords to find the super triangle*/ 

 double ymax = 0.0, ymin = 0.0;
 double xmax = 0.0, xmin = 0.0; 

 for (int i = 0; i < nlocal; i++){
  xmax = MAX(xmax,x[i][0]);
  ymax = MAX(ymax,x[i][1]);
  xmin = MIN(xmin,x[i][0]);
  ymin = MIN(ymin,x[i][1]);
 }
 double xmaxall, ymaxall, xminall, yminall;
 MPI_Allreduce(&xmax, &xmaxall, 1, MPI_DOUBLE, MPI_MAX, world);
 MPI_Allreduce(&ymax, &ymaxall, 1, MPI_DOUBLE, MPI_MAX, world);
 MPI_Allreduce(&xmin, &xminall, 1, MPI_DOUBLE, MPI_MIN, world);
 MPI_Allreduce(&ymin, &yminall, 1, MPI_DOUBLE, MPI_MIN, world);

 double dmax;

 if (xmaxall-xminall > ymaxall-yminall) dmax = 3*(xmaxall-xminall);
 else dmax = 3*(ymaxall-yminall);

 double xcen = 0.5*(xminall+xmaxall); 
 double ycen = 0.5*(yminall+ymaxall);

 // Save coordinates of the super triangle

 double coords_super[3][2];
 coords_super[0][0] = xcen-0.866*dmax;
 coords_super[0][1] = ycen-0.5*dmax;
 coords_super[1][0] = xcen+0.866*dmax;
 coords_super[1][1] = ycen-0.5*dmax;
 coords_super[2][0] = xcen;
 coords_super[2][1] = ycen+dmax;

  // Initialize the mesh

  Del_Tri_mesh.push_back({nall, nall+1, nall+2, 0});
  double temp_CC[3] = {0.0};
  int id_tri;
  for (int i = 0; i < 3; i++) {
    id_tri = Del_Tri_mesh[0][i];
    if (id_tri == nall) {
      xn[i] = coords_super[0][0];
      yn[i] = coords_super[0][1];
    } else if (id_tri == nall+1) {
      xn[i] = coords_super[1][0];
      yn[i] = coords_super[1][1];
    } else if (id_tri == nall+2) {
      xn[i] = coords_super[2][0];
      yn[i] = coords_super[2][1];
    } else {
      xn[i] = x[id_tri][0];
      yn[i] = x[id_tri][1];
    }
  }
  calc_cc(xn, yn, temp_CC);
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

    for (int i = 0; i < 3; i++) {
      id_tri = Del_Tri_mesh[Del_Tri_mesh.size()-1][i];
      if (id_tri == nall) {
        xn[i] = coords_super[0][0];
        yn[i] = coords_super[0][1];
      } else if (id_tri == nall+1) {
        xn[i] = coords_super[1][0];
        yn[i] = coords_super[1][1];
      } else if (id_tri == nall+2) {
        xn[i] = coords_super[2][0];
        yn[i] = coords_super[2][1];
      } else {
        xn[i] = x[id_tri][0];
        yn[i] = x[id_tri][1];
      }
    }
    calc_cc(xn, yn, temp_CC);
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

 /*Cell center based vertex model force calculation*/
  
  // printf("peratom: %d, atom columns: %d \n",fstore->peratom_flag,fstore->size_peratom_cols);
  // for(int i = 0; i < nall; i++){
  //   printf("nlocal: %d, voro: %f \n",nlocal,fstore->vector_atom[i]);
  // }

  double rvec[3];
  for(int i = 0; i < nlocal; i++){
    if (mask[i] & groupbit){

    double F_t1[3] = {0.0}; //t1 stands for term 1 in the vertex model force equation which is the area term
    double vir[6] = {0.0};

    for(int j = 0; j < cell_neighs_list[i].size(); j++){
      int current_cell = cell_neighs_list[i][j];
      double vertex_force_sum_t1[3] = {0.0};

      // First term values needed
      double area = voro_data[current_cell];
      double elasticity_area = 0.0;

      if (flag_store_init) {
        double Apref0 = voro_area0[current_cell];
        elasticity_area = (Elasticity/2)*(area-Apref0);

        // printf("Current cell: %d, Current area: %f, preferred area: %f, elasticity: %f \n",current_cell,area,Apref0,elasticity_area);
      } else {
        elasticity_area = (Elasticity/2)*(area-Apref);
      }

      int num_vert = cell_vertices_list[current_cell].size();
      int vcount = 0;
      int current_vert;

      // Looping through vertices of cell j
      while(vcount < num_vert){

        current_vert = cell_vertices_list[current_cell][vcount];

        // calculate jacobian and forces only if vertex belongs to cell i as well:
        if (std::find(cell_vertices_list[i].begin(), cell_vertices_list[i].end(), current_vert) != cell_vertices_list[i].end()){ 

          int pn[2] = {0};    // store previous and next vertex ids
          if(vcount == 0){
            pn[0] = cell_vertices_list[current_cell][num_vert - 1];
            pn[1] = cell_vertices_list[current_cell][1];
          }
          else if(vcount == num_vert - 1){
            pn[0] = cell_vertices_list[current_cell][vcount - 1];
            pn[1] = cell_vertices_list[current_cell][0];
          }
          else{
            pn[0] = cell_vertices_list[current_cell][vcount - 1];
            pn[1] = cell_vertices_list[current_cell][vcount + 1];
          }

          // First term stuff
          double rprevnext[3] = {Del_Tri_cc[pn[1]][0]-Del_Tri_cc[pn[0]][0],
                                 Del_Tri_cc[pn[1]][1]-Del_Tri_cc[pn[0]][1],
                                 0.0};
          double cp[3] = {0};
          double N[3] = {0,0,1}; // normal vector to the plane of cell layer (2D)
          MathExtra::cross3(rprevnext,N,cp);
          
          double drmu_dri[3][3] = {0.0};
          Jacobian(x, Del_Tri_mesh[current_vert], i, drmu_dri);                      

          double result_t1[3] = {0};
          
          // Term 1 forces
          MathExtra::transpose_matvec(drmu_dri, cp, result_t1);
          vertex_force_sum_t1[0] += result_t1[0];
          vertex_force_sum_t1[1] += result_t1[1];
          vertex_force_sum_t1[2] += result_t1[2];
        }
        vcount++;
      }
      F_t1[0] += elasticity_area*vertex_force_sum_t1[0];
      F_t1[1] += elasticity_area*vertex_force_sum_t1[1]; 
      F_t1[2] += elasticity_area*vertex_force_sum_t1[2];

      // Count virial contribution
      rvec[0] = x[i][0]-x[current_cell][0];
      rvec[1] = x[i][1]-x[current_cell][1];
      rvec[2] = x[i][2]-x[current_cell][2];
      domain->minimum_image(rvec[0], rvec[1], rvec[2]);
      vir[0]  += (-elasticity_area*vertex_force_sum_t1[0]) * rvec[0];
      vir[1]  += (-elasticity_area*vertex_force_sum_t1[1]) * rvec[1];
      vir[2]  += (-elasticity_area*vertex_force_sum_t1[2]) * rvec[2];
      vir[3]  += (-elasticity_area*vertex_force_sum_t1[0]) * rvec[1];
      vir[4]  += (-elasticity_area*vertex_force_sum_t1[0]) * rvec[2];
      vir[5]  += (-elasticity_area*vertex_force_sum_t1[1]) * rvec[2];
    }
    double fx = -F_t1[0]; 
    double fy = -F_t1[1]; 
    double fz = -F_t1[2];
    f[i][0] += fx;
    f[i][1] += fy;
    f[i][2] += fz;
    if (evflag) {
      v_tally(i, vir);
    }
  }
 }
 
}

/* ---------------------------------------------------------------------- */

void FixVolComp::post_force_respa(int vflag, int ilevel, int /*iloop*/)
{
  if (ilevel == ilevel_respa) post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixVolComp::min_post_force(int vflag)
{
  post_force(vflag);
}

/*------------------------------------------------------------------------*/

int FixVolComp::pack_forward_comm(int n, int *list, double *buf,
                                    int /*pbc_flag*/, int * /*pbc*/)

{

  int i,j,m;

  m = 0;
 if(commflag == 1){
    for(i = 0; i < n; i++){
      j = list[i];
      buf[m++] = voro_area0[j];
    }
 } else {
    for(i = 0; i < n; i++){
      j = list[i];
      buf[m++] = voro_data[j];
    }
  }
  return m;
}

void FixVolComp::unpack_forward_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;

 if (commflag == 1) {
    for (i = first; i < last; i++){
      voro_area0[i] = buf[m++];
    }
 } else {
    for (i = first; i < last; i++){
      voro_data[i] = buf[m++];
    }
  }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixVolComp::memory_usage()
{
  int nmax = atom->nmax;
  double bytes = (double)nmax*4 * sizeof(double);
  // bytes += (double)nmax*3 * sizeof(double);
  // bytes += (double)nmax*4 * sizeof(int);
  return bytes;
}