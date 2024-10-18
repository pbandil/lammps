/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(tridynamic,FixTriDynamic);
// clang-format on
#else

//PRB{These Statements ensure that this class is only inlcuded once in the project}
#ifndef LMP_FIX_TRIDYNAMIC_H
#define LMP_FIX_TRIDYNAMIC_H

#include "fix.h"
#include <vector>

using namespace std;

namespace LAMMPS_NS {

class FixTriDynamic : public Fix {
 public:
  FixTriDynamic(class LAMMPS *, int, char **);
  ~FixTriDynamic() override;
  void post_constructor();
  int setmask() override;
  void init() override;
  void setup(int) override;
 // void min_setup(int) override; 

  void post_force(int) override;
  void post_force_respa(int, int, int) override;
  void min_post_force(int) override;

  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;
  int pack_exchange(int, double *) override;
  int unpack_exchange(int, double *) override;
  void grow_arrays(int) override;
  void set_arrays(int) override;
  double memory_usage() override;

  FILE *fp;  //write data for debugging

 protected:
  int me, nprocs;
  int nmax;

 private:

  int neighs_MAX;

  //Voronoi Force Data members
  class Compute *vcompute;    // To read data from compute voronoi (only for initialization)
  char *id_compute_voronoi;   // A Pointer variable

  // To store area, peri and energy (these don't need to persist accross time steps)
  double **cell_shape;    //Note that these are dynamic arrays
  double **vertices;
  double **x_updated;
  
  //Force calculation: self prpelled dynamic triangulation forces
  double ka, kp, p0;
  double Jv, Jn, Js, fa, var, eta_0, KT;
  double Num_MC_input;
  double c1,c2;
  double T1_threshold;

  //For debugging purposes
  int n_every_output;

  class RanMars *wgn;
  int seed_wgn;

  // For invoking fix property/atom
  char *new_fix_id;    //To store atom property: neighs_list
  int index;

  // Common (Dyn Tri force) Data members

  char *idregion;           // Right now for Grooves (A pointer variable)
  class Region *region;     // A pointer variable (but we don't delete it!)

  int ilevel_respa;

  int commflag;    // for communicating data of ghost atoms

  //Declare functions
  
  //functions for checking compatibility of position and topology
  int nearest_image(int, int, double **);

  // For cyclically arranging a set of points
  void arrange_cyclic(tagint *, int, int, double **);

  // For storing cell info: area, peri and energy
  void get_cell_data(double *, double *, tagint *, int, int, double **);

  // For obtaining common neighs for a pair of atoms
  int get_comm_neigh(tagint *, tagint *, tagint *, int, int);

  // See if atoms are already bonded or not

  int unbonded(tagint *, int, tagint);

  //adding and deleting neighs
  //void add_neigh(int, tagint, tagint *, int);
  void remove_neigh(int, tagint, tagint *, int, int);

  void print_neighs_list(tagint *, int, int);

  bool isConcave(double *, double *, double *, double *);
  double crossProduct(double *, double *, double *);
  void getCP(double *, double *, double *);
  void normalize(double *);
  

  //void arrange_cyclic_new(tagint *, int, int, double **);
  bool is_cyclic_perm(tagint *, tagint *, int);

  //int check_bond(int, int, tagint *, tagint *, int, int);

  //vector<vector<int>> find_triangles(tagint **, int);
};

}    // namespace LAMMPS_NS

#endif
#endif
