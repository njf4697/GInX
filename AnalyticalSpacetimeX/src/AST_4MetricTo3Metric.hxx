#ifndef AST_4METRICTO3METRIC_HXX
#define AST_4METRICTO3METRIC_HXX

#include "tensor.hxx"
#include <loop_device.hxx>

namespace AnalyticalSpacetimeX {

CCTK_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline int
four_metric_to_three_metric(const struct four_metric *met,
                            struct three_metric *gam) {
  /* Check determinant first */
  gam->gxx = met->g.xx;
  gam->gxy = met->g.xy;
  gam->gxz = met->g.xz;
  gam->gyy = met->g.yy;
  gam->gyz = met->g.yz;
  gam->gzz = met->g.zz;

  CCTK_REAL det = -(gam->gxz * gam->gxz * gam->gyy) +
                  2 * gam->gxy * gam->gxz * gam->gyz -
                  gam->gxy * gam->gxy * gam->gzz +
                  gam->gxx * (-gam->gyz * gam->gyz + gam->gyy * gam->gzz);

  /* If determinant is not >0  something is wrong with the metric */
  /* This could occur during the transition to merger at certain points so here
   * we restart to Minkowski */
  if (!(det > 0)) {
    printf("det < 0: %e\n", det);
    printf("%e %e %e\n", gam->gxx, gam->gxy, gam->gxz);
    printf("%e %e %e\n", gam->gyy, gam->gyz, gam->gzz);
    det = 1.0;
    gam->gxx = 1.0;
    gam->gxy = 0.0;
    gam->gxz = 0.0;
    gam->gyy = 1.0;
    gam->gyz = 0.0;
    gam->gzz = 1.0;

    gam->betax = 0.0;
    gam->betay = 0.0;
    gam->betaz = 0.0;

    gam->alpha = 1.0;
    gam->kxx = 0.0;
    gam->kxy = 0.0;
    gam->kxz = 0.0;
    gam->kyy = 0.0;
    gam->kyz = 0.0;
    gam->kzz = 0.0;

  } else {

    /* Compute components if detg is not <0 */
    CCTK_REAL betadownx = met->g.tx;
    CCTK_REAL betadowny = met->g.ty;
    CCTK_REAL betadownz = met->g.tz;

    CCTK_REAL dbetadownxx = met->g_x.tx;
    CCTK_REAL dbetadownyx = met->g_x.ty;
    CCTK_REAL dbetadownzx = met->g_x.tz;

    CCTK_REAL dbetadownxy = met->g_y.tx;
    CCTK_REAL dbetadownyy = met->g_y.ty;
    CCTK_REAL dbetadownzy = met->g_y.tz;

    CCTK_REAL dbetadownxz = met->g_z.tx;
    CCTK_REAL dbetadownyz = met->g_z.ty;
    CCTK_REAL dbetadownzz = met->g_z.tz;

    CCTK_REAL dtgxx = met->g_t.xx;
    CCTK_REAL dtgxy = met->g_t.xy;
    CCTK_REAL dtgxz = met->g_t.xz;
    CCTK_REAL dtgyy = met->g_t.yy;
    CCTK_REAL dtgyz = met->g_t.yz;
    CCTK_REAL dtgzz = met->g_t.zz;

    CCTK_REAL dgxxx = met->g_x.xx;
    CCTK_REAL dgxyx = met->g_x.xy;
    CCTK_REAL dgxzx = met->g_x.xz;
    CCTK_REAL dgyyx = met->g_x.yy;
    CCTK_REAL dgyzx = met->g_x.yz;
    CCTK_REAL dgzzx = met->g_x.zz;

    CCTK_REAL dgxxy = met->g_y.xx;
    CCTK_REAL dgxyy = met->g_y.xy;
    CCTK_REAL dgxzy = met->g_y.xz;
    CCTK_REAL dgyyy = met->g_y.yy;
    CCTK_REAL dgyzy = met->g_y.yz;
    CCTK_REAL dgzzy = met->g_y.zz;

    CCTK_REAL dgxxz = met->g_z.xx;
    CCTK_REAL dgxyz = met->g_z.xy;
    CCTK_REAL dgxzz = met->g_z.xz;
    CCTK_REAL dgyyz = met->g_z.yy;
    CCTK_REAL dgyzz = met->g_z.yz;
    CCTK_REAL dgzzz = met->g_z.zz;

    CCTK_REAL idetgxx = -gam->gyz * gam->gyz + gam->gyy * gam->gzz;
    CCTK_REAL idetgxy = gam->gxz * gam->gyz - gam->gxy * gam->gzz;
    CCTK_REAL idetgxz = -(gam->gxz * gam->gyy) + gam->gxy * gam->gyz;
    CCTK_REAL idetgyy = -gam->gxz * gam->gxz + gam->gxx * gam->gzz;
    CCTK_REAL idetgyz = gam->gxy * gam->gxz - gam->gxx * gam->gyz;
    CCTK_REAL idetgzz = -gam->gxy * gam->gxy + gam->gxx * gam->gyy;

    CCTK_REAL invgxx = idetgxx / det;
    CCTK_REAL invgxy = idetgxy / det;
    CCTK_REAL invgxz = idetgxz / det;
    CCTK_REAL invgyy = idetgyy / det;
    CCTK_REAL invgyz = idetgyz / det;
    CCTK_REAL invgzz = idetgzz / det;

    gam->betax = betadownx * invgxx + betadowny * invgxy + betadownz * invgxz;

    gam->betay = betadownx * invgxy + betadowny * invgyy + betadownz * invgyz;

    gam->betaz = betadownx * invgxz + betadowny * invgyz + betadownz * invgzz;

    CCTK_REAL b2 = betadownx * gam->betax + betadowny * gam->betay +
                   betadownz * gam->betaz;

    gam->alpha = sqrt(fabs(b2 - met->g.tt));

    gam->kxx =
        -(-2 * dbetadownxx - gam->betax * dgxxx - gam->betay * dgxxy -
          gam->betaz * dgxxz +
          2 * (gam->betax * dgxxx + gam->betay * dgxyx + gam->betaz * dgxzx) +
          dtgxx) /
        (2. * gam->alpha);

    gam->kxy = -(-dbetadownxy - dbetadownyx + gam->betax * dgxxy -
                 gam->betaz * dgxyz + gam->betaz * dgxzy + gam->betay * dgyyx +
                 gam->betaz * dgyzx + dtgxy) /
               (2. * gam->alpha);

    gam->kxz = -(-dbetadownxz - dbetadownzx + gam->betax * dgxxz +
                 gam->betay * dgxyz - gam->betay * dgxzy + gam->betay * dgyzx +
                 gam->betaz * dgzzx + dtgxz) /
               (2. * gam->alpha);

    gam->kyy =
        -(-2 * dbetadownyy - gam->betax * dgyyx - gam->betay * dgyyy -
          gam->betaz * dgyyz +
          2 * (gam->betax * dgxyy + gam->betay * dgyyy + gam->betaz * dgyzy) +
          dtgyy) /
        (2. * gam->alpha);

    gam->kyz = -(-dbetadownyz - dbetadownzy + gam->betax * dgxyz +
                 gam->betax * dgxzy + gam->betay * dgyyz - gam->betax * dgyzx +
                 gam->betaz * dgzzy + dtgyz) /
               (2. * gam->alpha);

    gam->kzz =
        -(-2 * dbetadownzz - gam->betax * dgzzx - gam->betay * dgzzy -
          gam->betaz * dgzzz +
          2 * (gam->betax * dgxzz + gam->betay * dgyzz + gam->betaz * dgzzz) +
          dtgzz) /
        (2. * gam->alpha);
  }

  return 0;
}

} // namespace AnalyticalSpacetimeX

#endif // #ifndef AST_4METRICTO3METRIC_HXX
