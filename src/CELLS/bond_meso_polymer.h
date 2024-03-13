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

#ifdef BOND_CLASS
// clang-format off
BondStyle(meso/polymer,BondMESOPolymer);
// clang-format on
#else

#ifndef LMP_BOND_MESO_POLYMER_H
#define LMP_BOND_MESO_POLYMER_H

#include "bond_meso.h"

namespace LAMMPS_NS {

class BondMESOPolymer : public BondMESO {
 public:
  BondMESOPolymer(class LAMMPS *);
  ~BondMESOPolymer() override;
  void compute(int, int) override;
  void coeff(int, char **) override;
  void init_style() override;
  void settings(int, char **) override;
  void write_restart(FILE *) override;
  void read_restart(FILE *) override;
  void write_restart_settings(FILE *) override;
  void read_restart_settings(FILE *) override;
  double single(int, double, int, int, double &) override;

 protected:
  double *b, *lamcrit, *gamma;
  int *Nmean, *Nmin, *Nmax;
  int smooth_flag, me, partial_flag;

  class RanMars *random;

  void allocate();
  void store_data();
  double store_bond(int, int, int);
};

}    // namespace LAMMPS_NS

#endif
#endif
