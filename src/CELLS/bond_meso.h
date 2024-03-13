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

#ifndef LMP_BOND_MESO_H
#define LMP_BOND_MESO_H

#include "bond.h"

#include <vector>

namespace LAMMPS_NS {

class BondMESO : public Bond {
 public:
  BondMESO(class LAMMPS *);
  ~BondMESO() override;
  void compute(int, int) override = 0;
  void coeff(int, char **) override = 0;
  void init_style() override;
//   void settings(int, char **) override;
  double equilibrium_distance(int) override;
  void write_restart(FILE *) override;
  void read_restart(FILE *) override;
  double single(int, double, int, int, double &) override = 0;

 protected:
  double r0_max_estimate;
  int store_local_freq;

  std::vector<int> leftover_iarg;

  char *id_fix_dummy, *id_fix_dummy2;
  char *id_fix_update, *id_fix_bond_history;
  char *id_fix_store_local, *id_fix_prop_atom;
  class FixStoreLocal *fix_store_local;
  class FixBondHistory *fix_bond_history;
  class FixUpdateSpecialBonds *fix_update_special_bonds;

  typedef void (BondMESO::*FnPtrPack)(int, int, int);
  FnPtrPack *pack_choice;    // ptrs to pack functions
  double *output_data;

  int prop_atom_flag, nvalues, overlay_flag;
  int index_x_ref, index_y_ref, index_z_ref;
};

}    // namespace LAMMPS_NS

#endif
