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
FixStyle(spvforce,FixSpvForce);
// clang-format on
#else

//PRB{These Statements ensure that this class is only inlcuded once in the project}
#ifndef LMP_FIX_SPVFORCE_H
#define LMP_FIX_SPVFORCE_H

#include "fix.h"

using namespace std;

namespace LAMMPS_NS {

class FixSpvForce : public Fix {
 public:
  FixSpvForce(class LAMMPS *, int, char **);
  ~FixSpvForce() override;
  void post_constructor();
  int setmask() override;
  void init() override;
  void setup(int) override;
  void min_setup(int) override;
  void post_force(int) override;
  void post_force_respa(int, int, int) override;
  void min_post_force(int) override;

  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;
  double memory_usage() override;

  FILE *fp;

 protected:
  int me, nprocs;

 private:

  double xlo_init, xhi_init, ylo_init, yhi_init;
  double xlo_curr, xhi_curr, ylo_curr, yhi_curr;
  double FiX, FiY;

  //Voronoi Force Data members
  double ka, kp, p0;
  class Compute *vcompute;    // To read peratom data from compute voronoi
  char *id_compute_voronoi;   // A Pointer variable
  double *voro_area, *voro_peri, *def_voro_peri;    //Note that these are dynamic arrays
  int maxatom;
  double F11;

  int initflag;

  //Self Propelled Force Data members
  double Jv, Jn, Js, fa, var, gamma_R;
  class RanMars *wgn;
  int seed_wgn;
  char *new_fix_id;    //To store an atom property: Cell Polarity
  int index;

  // Common (SPV force) Data members

  char *idregion;           // Right now for Grooves (A pointer variable)
  class Region *region;     // A pointer variable (but we don't delete it!)

  int ilevel_respa;

  int commflag;    // for communicating data of ghost atoms
};

}    // namespace LAMMPS_NS

#endif
#endif
