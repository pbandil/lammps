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
FixStyle(volcomp,FixVolComp);
// clang-format on
#else

//PRB{These Statements ensire that this class is only inlcuded once in the project}
#ifndef LMP_FIX_VOLCOMP_H
#define LMP_FIX_VOLCOMP_H

#include "fix.h"

namespace LAMMPS_NS {

class FixVolComp : public Fix {
 public:
  FixVolComp(class LAMMPS *, int, char **);
  ~FixVolComp() override;
  int setmask() override;
  void init() override;
  void setup(int) override;
  void min_setup(int) override;
  void post_force(int) override;
  void post_force_respa(int, int, int) override;
  void min_post_force(int) override;

  int pack_forward_comm(int, int *, double *, int, int *);
  void unpack_forward_comm(int, int, double *);
  double memory_usage();
  
  FILE *fp;

 protected:
 int me, nprocs; 
 int nmax;

 void calc_cc(double *, double *, double *);

 private:
  
  class Compute *vcompute; // ptr to compute voronoi
  class Fix *fstore;       // ptr to fix/store

  double Elasticity, Apref;
  int flag_store_init;
 
  char *id_compute_voronoi, *id_fix_store;

  int ilevel_respa;

 // To read peratom area from compute voronoi
  double *voro_data;
  double *voro_area0;

  // For communicating voro_area or voro_area0
  int commflag;

};

}    // namespace LAMMPS_NS

#endif
#endif