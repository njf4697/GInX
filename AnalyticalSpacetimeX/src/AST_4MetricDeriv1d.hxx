#ifndef AST_4METRICDERIV1D_HXX
#define AST_4METRICDERIV1D_HXX

#include "AST_Helpers.hxx"
#include "tensor.hxx"
#include <loop_device.hxx>

/* Definition of derivative */
// #define D8(comp,h) (3*(met_m4.g.comp) - 32*(met_m3.g.comp) +
// 168*((met_m2.g.comp) - 4*(met_m1.g.comp) + 4*(met_p1.g.comp) -
// (met_p2.g.comp)) + 32*(met_p3.g.comp) - 3*(met_p4.g.comp)) / (840*h)
#define D2(comp, h) ((met_p1.g.comp) - (met_m1.g.comp)) / (2 * h)

namespace AnalyticalSpacetimeX {

template <int component, typename GetMetricFunc>
CCTK_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline void
get_dmetric_1d(struct dd_sym &dmetric, const double h,
               GetMetricFunc &get_metric, const double *xx, const double *nz_m1,
               const double *nz_p1, bool &maskL) {
  struct four_metric met_m1;
  struct four_metric met_p1;

  constexpr int t_shift = component == TT;
  constexpr int x_shift = component == XX;
  constexpr int y_shift = component == YY;
  constexpr int z_shift = component == ZZ;

  const double xx_m1[NDIM] = {xx[TT] - h * t_shift, xx[XX] - h * x_shift,
                              xx[YY] - h * y_shift, xx[ZZ] - h * z_shift};
  const double xx_p1[NDIM] = {xx[TT] + h * t_shift, xx[XX] + h * x_shift,
                              xx[YY] + h * y_shift, xx[ZZ] + h * z_shift};

  get_metric(xx_m1, &met_m1, nz_m1, maskL);
  get_metric(xx_p1, &met_p1, nz_p1, maskL);

  dmetric.tt = D2(tt, h);
  dmetric.tx = D2(tx, h);
  dmetric.ty = D2(ty, h);
  dmetric.tz = D2(tz, h);
  dmetric.xx = D2(xx, h);
  dmetric.xy = D2(xy, h);
  dmetric.xz = D2(xz, h);
  dmetric.yy = D2(yy, h);
  dmetric.yz = D2(yz, h);
  dmetric.zz = D2(zz, h);
}

} // namespace AnalyticalSpacetimeX

#endif // #ifndef AST_4METRICDERIV1D_HXX
