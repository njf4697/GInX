/*@@
   @file      AST_VerticalGravity.c
   @date      June 9, 2024
   @author    Luciano Combi
   @desc
   @enddesc
@@*/

#ifndef AST_VERTICALGRAVITY_HXX
#define AST_VERTICALGRAVITY_HXX

namespace AnalyticalSpacetimeX {

#include "AST_Helpers.hxx"
#include "tensor.hxx"
#include <loop_device.hxx>

CCTK_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline void
VerticalGravity(const double *xx, double gcov[][NDIM], const double *traj_array,
                const double z0, const double g) {

  // Phi
  double Phi_grav = (xx[ZZ] - z0) * g;
  // Down metric
  gcov[TT][TT] = -(1.0 + 2.0 * Phi_grav);
  gcov[TT][XX] = 0.0;
  gcov[TT][YY] = 0.0;
  gcov[TT][ZZ] = 0.0;
  gcov[XX][XX] = 1.0;
  gcov[YY][YY] = 1.0;
  gcov[ZZ][ZZ] = 1.0;
  gcov[XX][YY] = 0.0;
  gcov[YY][ZZ] = 0.0;
  gcov[ZZ][XX] = 0.0;

  gcov[XX][ZZ] = gcov[ZZ][XX];
  gcov[ZZ][YY] = gcov[YY][ZZ];
  gcov[YY][XX] = gcov[XX][YY];
  gcov[XX][TT] = gcov[TT][XX];
  gcov[YY][TT] = gcov[TT][YY];
  gcov[ZZ][TT] = gcov[TT][ZZ];

  return;
}

} // namespace AnalyticalSpacetimeX

#endif // #ifndef AST_KERRSCHILD_HXX
