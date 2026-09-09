/*@@
   @file      AST_KerrSchild.c
   @date      May 31, 2023
   @author    Luciano Combi
   @desc
   @enddesc
@@*/

#ifndef AST_KERRSCHILD_HXX
#define AST_KERRSCHILD_HXX

#include "AST_Helpers.hxx"
#include "tensor.hxx"
#include <loop_device.hxx>

namespace AnalyticalSpacetimeX {

CCTK_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline void
KerrSchild(const double *xx, double gcov[][NDIM], const double m,
           const double a) {

  /* Mask */
  double x = xx[XX];
  double y = xx[YY];
  double z = xx[ZZ];

  double z0, x0, y0, rho02, r02, r0, hh, lt0, lx0, ly0, lz0;
  double lt, lx, ly, lz;

  // Lorentz transform
  z0 = z; // gamma * ((z ) - boostv * (t ));
  x0 = x;
  y0 = y;

  // Coordinate distance to center of black hole
  rho02 = x0 * x0 + y0 * y0 + z0 * z0;
  // Spherical auxiliary coordinate r and angle theta in BH rest frame
  r02 = 0.5 * (rho02 - a * a) +
        sqrt(0.25 * pow(rho02 - a * a, 2) + a * a * z0 * z0);
  r0 = sqrt(fmax(1e-10, r02));

  /* Apply excision */
  // double rBH_Cutoff = fabs(a) * ( 1.0 + AST_a1_buffer) + AST_cutoff_floor;
  // if ((rho0) < rBH_Cutoff) { if(z0>0) {z0 = rBH_Cutoff;} else {z0 =
  // -1.0*rBH_Cutoff;}} rho02 = x0 * x0 + y0 * y0 + z0 * z0;

  /* Apply excision */
  double rBH_Cutoff = (m + sqrt(m * m - a * a)) * 0.5;
  double a1x = 0.0;
  double a1y = 0.0;
  double a1z = 0.0;
  double a1 = a;
  double adotx_BH1 = a1x * x0 + a1y * y0 + a1z * z0;
  if (r0 < rBH_Cutoff) {
    double lx_BH1 = (r0 * x0 - a1y * z0 + a1z * y0 + adotx_BH1 * (a1x / r0)) /
                    (r0 * r0 + a1 * a1);
    double ly_BH1 = (r0 * y0 - a1z * x0 + a1x * z0 + adotx_BH1 * (a1y / r0)) /
                    (r0 * r0 + a1 * a1);
    double lz_BH1 = (r0 * z0 - a1x * y0 + a1y * x0 + adotx_BH1 * (a1z / r0)) /
                    (r0 * r0 + a1 * a1);
    double thetaBH1 = acos(lz_BH1);
    double phiBH1 = atan2(ly_BH1, lx_BH1);
    x0 = rBH_Cutoff * sin(thetaBH1) * cos(phiBH1) + a1y * cos(thetaBH1) -
         a1z * sin(phiBH1) * sin(thetaBH1);
    y0 = rBH_Cutoff * sin(thetaBH1) * sin(phiBH1) +
         a1z * sin(thetaBH1) * cos(phiBH1) - a1x * cos(thetaBH1);
    z0 = rBH_Cutoff * cos(thetaBH1) + a1x * sin(phiBH1) * sin(thetaBH1) -
         a1z * sin(thetaBH1) * cos(phiBH1);
    r0 = rBH_Cutoff;
  }

  // Coefficient H
  hh = m * r0 * r0 * r0 / (r0 * r0 * r0 * r0 + a * a * z0 * z0);

  // Components of l_a in rest frame
  lt0 = 1.0;
  lx0 = (r0 * x0 + a * y0) / (r0 * r0 + a * a);
  ly0 = (r0 * y0 - a * x0) / (r0 * r0 + a * a);
  lz0 = z0 / r0;

  // Boost to coordinates x, y, z, t
  lt = lt0; // gamma * (lt0 - boostv * lz0);
  lz = lz0; // gamma * (lz0 - boostv * lt0);
  lx = lx0;
  ly = ly0;

  // Down metric
  gcov[TT][TT] = -1.0 + 2.0 * hh * lt * lt;
  gcov[TT][XX] = 2.0 * hh * lt * lx;
  gcov[TT][YY] = 2.0 * hh * lt * ly;
  gcov[TT][ZZ] = 2.0 * hh * lt * lz;
  gcov[XX][XX] = 1.0 + 2.0 * hh * lx * lx;
  gcov[YY][YY] = 1.0 + 2.0 * hh * ly * ly;
  gcov[ZZ][ZZ] = 1.0 + 2.0 * hh * lz * lz;
  gcov[XX][YY] = 2.0 * hh * lx * ly;
  gcov[YY][ZZ] = 2.0 * hh * ly * lz;
  gcov[ZZ][XX] = 2.0 * hh * lz * lx;

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
