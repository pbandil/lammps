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
FixStyle(bond/dynamic,FixBondDynamic);
// clang-format on
#else

#ifndef LMP_FIX_BOND_DYNAMIC_H
#define LMP_FIX_BOND_DYNAMIC_H

#include "fix.h"

namespace LAMMPS_NS {

class FixBondDynamic : public Fix {
 public:
  FixBondDynamic(class LAMMPS *, int, char **);
  virtual ~FixBondDynamic();
  int setmask();
  void post_constructor();
  void init();
  void init_list(int, class NeighList *) override;
  void setup(int) override;
  void post_integrate();
  void post_integrate_respa(int, int);

  int pack_forward_comm(int, int *, double *, int, int *);
  void unpack_forward_comm(int, int, double *);
  double memory_usage();

 protected:
  int me, nprocs;
  int iatomtype,jatomtype;
  int atype, btype, seed;
  int imaxbond, jmaxbond, maxbond;
  double cutsq, prob_attach, prob_detach, f0, ka, kd, DT_EQ;
  double ka0, kd0, b0, b2, r_critical, r2_critical;
  int flag_bell, flag_prob, flag_rouse, flag_critical, flag_mol, flag_skip;
  int skip;

  int overflow;

  int size_bonds_and_neighbors;
  int size_bond_lists;

  int *bondcount;
  int createcount, createcounttotal;
  int breakcount, breakcounttotal;
  int nmax;
  tagint *partner, *finalpartner;
  double *distsq, probability, *probabilities;
  double cutoff;
  int **broken_partners;
  double **partners_probs, **partners_probs_f;
  int *npos, *influenced;
  int **partners_success;

  tagint **partners_possible, **partners_possible_f;

  int nbreak, maxbreak;
  int ncreate, maxcreate;
  tagint **created;
  tagint **broken;

  tagint *copy;

  class RanMars *random;
  class NeighList *list;

  int prop_atom_flag, nvalues, overlay_flag;

  int countflag, commflag;
  int nlevels_respa;
  int nangles, ndihedrals, nimpropers;

  void process_broken(int, int);
  void process_created(int, int);
  void rebuild_special_one(int);

  int dedup(int, int, tagint *);

  char *new_fix_id;
  int index;

};

}    // namespace LAMMPS_NS

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Invalid atom type in fix bond/create/dynamic command

Self-explanatory.

E: Invalid bond type in fix bond/create/dynamic command

Self-explanatory.

E: Cannot use fix bond/create/dynamic with non-molecular systems

Only systems with bonds that can be changed can be used.  Atom_style
template does not qualify.

E: Inconsistent iparam/jparam values in fix bond/create/dynamic command

If itype and jtype are the same, then their maxbond and newtype
settings must also be the same.

E: Fix bond/create/dynamic cutoff is longer than pairwise cutoff

This is not allowed because bond creation is done using the
pairwise neighbor list.

E: Fix bond/create/dynamic angle type is invalid

Self-explanatory.

E: Fix bond/create/dynamic dihedral type is invalid

Self-explanatory.

E: Fix bond/create/dynamic improper type is invalid

Self-explanatory.

E: Cannot yet use fix bond/create/dynamic with this improper style

This is a current restriction in LAMMPS.

E: Fix bond/create/dynamic needs ghost atoms from further away

This is because the fix needs to walk bonds to a certain distance to
acquire needed info, The comm_modify cutoff command can be used to
extend the communication range.

E: New bond exceeded bonds per atom in fix bond/create/dynamic

See the read_data command for info on setting the "extra bond per
atom" header value to allow for additional bonds to be formed.

E: New bond exceeded special list size in fix bond/create/dynamic

See the special_bonds extra command for info on how to leave space in
the special bonds list to allow for additional bonds to be formed.

E: Fix bond/create/dynamic induced too many angles/dihedrals/impropers per atom

See the read_data command for info on setting the "extra angle per
atom", etc header values to allow for additional angles, etc to be
formed.

E: Special list size exceeded in fix bond/create/dynamic

See the read_data command for info on setting the "extra special per
atom" header value to allow for additional special values to be
stored.

W: Fix bond/create/dynamic is used multiple times or with fix bond/break - may not work as expected

When using fix bond/create/dynamic multiple times or in combination with
fix bond/break, the individual fix instances do not share information
about changes they made at the same time step and thus it may result
in unexpected behavior.

*/
