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
FixStyle(volvoro,FixVolVoro);
// clang-format on
#else

//PRB{These Statements ensire that this class is only inlcuded once in the project}
#ifndef LMP_FIX_VOLVORO_H
#define LMP_FIX_VOLVORO_H

#include "fix.h"

namespace LAMMPS_NS {

class FixVolVoro : public Fix {
 public:
  FixVolVoro(class LAMMPS *, int, char **);
  ~FixVolVoro() override;
  int setmask() override;
  void post_constructor();
  void init() override;
  void setup(int) override;
  void pre_force(int) override;
  void post_force(int) override;

  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;
  int pack_reverse_comm(int, int, double *) override;
  void unpack_reverse_comm(int, int *, double *) override;
  int pack_exchange(int, double *) override;
  int unpack_exchange(int, double *) override;
  void grow_arrays(int) override;
  void set_arrays(int) override;
  double memory_usage() override;
  
  FILE *fp;

 protected:
   int me, nprocs; 
   int nmax;

 private:
  
  class Compute *vcompute; // ptr to compute voronoi
  class Fix *fstore;       // ptr to fix/store

  double Elasticity, VolPref;
  int flag_store_init;
  int max_faces;
 
  char *id_compute_voronoi, *id_fix_store;

  int ilevel_respa;

  // Total virial on each cell
  double **total_virial;

  // Index of fix property/atom for storing DT faces
  char *new_fix_id;
  int index;

  // For keeping track of invoking DT creation
  int countflag;

  // For cyclically arranging a set of points
  void arrange_cyclic(tagint *, int);

  // For checking edge flips
  int check_edges(tagint *);

  // For flipping edges
  void flip_edge(tagint *);

  // For finding the circumcenter of a triangle
  void calc_cc(double *, double *, double *);

  // For calculating the jacobian
  void calc_jacobian(int *, int, double *);

  // For calculating cell volumes
  double calc_area(int *, int, int);

};

}    // namespace LAMMPS_NS

#endif
#endif