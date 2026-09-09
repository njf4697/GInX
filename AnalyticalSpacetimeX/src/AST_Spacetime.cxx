#include "AST_4MetricDeriv1d.hxx"
#include "AST_4MetricTo3Metric.hxx"
#include "AST_AnalyticTrajectory.hxx"
#include "AST_Helpers.hxx"
#include "AST_KerrSchild.hxx"
#include "AST_Readtable.hxx"
#include "AST_SuperposedBBH.hxx"
#include "AST_VerticalGravity.hxx"
#include "tensor.hxx"
#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>
#include "AST_Spacetime_SetMetric.hxx"

namespace AnalyticalSpacetimeX {

enum class metric_t { superposed_bbh, kerrschild, vertical_gravity };

metric_t metricType;

/* flag indicates if we need to do extra setting metric (after
 * regridding/recovering) */
static bool is_setting_metric = true;

extern "C" void AnalyticalSpacetimeX_SwitchOnSettingMetric(CCTK_ARGUMENTS) {
  is_setting_metric = true;
}

extern "C" void AnalyticalSpacetimeX_SwitchOffSettingMetric(CCTK_ARGUMENTS) {
  is_setting_metric = false;
}

extern "C" void AnalyticalSpacetimeX_Setup(CCTK_ARGUMENTS) {
  DECLARE_CCTK_PARAMETERS;

  if (CCTK_Equals(analytical_metric_type, "SuperposedBBH")) {
    metricType = metric_t::superposed_bbh;
  } else if (CCTK_Equals(analytical_metric_type, "KerrSchild")) {
    metricType = metric_t::kerrschild;
  } else if (CCTK_Equals(analytical_metric_type, "VerticalGravity")) {
    metricType = metric_t::vertical_gravity;
  } else {
    CCTK_ERROR("analytical_metric_type not recognised!");
  }
};

void SetMetricHelper(CCTK_ARGUMENTS, const double time) {
  DECLARE_CCTK_PARAMETERS;
  DECLARE_CCTK_ARGUMENTSX_AnalyticalSpacetimeX_SetMetric;
  
  const auto superposed_bbh_func =
      [=] CCTK_DEVICE(const double *xx, struct four_metric *met,
                      const double *bbh_traj_loc, bool &maskL) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        double gcov[NDIM][NDIM];

        SuperposedBBH(xx, gcov, bbh_traj_loc, AST_adjust_mass1,
                      AST_adjust_mass2, AST_a1_buffer, AST_a2_buffer,
                      AST_cutoff_floor, maskL);
        met->g.tt = gcov[TT][TT];
        met->g.tx = gcov[TT][XX];
        met->g.ty = gcov[TT][YY];
        met->g.tz = gcov[TT][ZZ];
        met->g.xx = gcov[XX][XX];
        met->g.xy = gcov[XX][YY];
        met->g.xz = gcov[XX][ZZ];
        met->g.yy = gcov[YY][YY];
        met->g.yz = gcov[YY][ZZ];
        met->g.zz = gcov[ZZ][ZZ];
      };

  const auto kerrschild_func =
      [=] CCTK_DEVICE(const double *xx, struct four_metric *met,
                      const double *bbh_traj_loc, bool &maskL) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        double gcov[NDIM][NDIM];

        // Note mask not used below yet
        KerrSchild(xx, gcov, AST_KerrSchild__mass, AST_KerrSchild__spin);
        met->g.tt = gcov[TT][TT];
        met->g.tx = gcov[TT][XX];
        met->g.ty = gcov[TT][YY];
        met->g.tz = gcov[TT][ZZ];
        met->g.xx = gcov[XX][XX];
        met->g.xy = gcov[XX][YY];
        met->g.xz = gcov[XX][ZZ];
        met->g.yy = gcov[YY][YY];
        met->g.yz = gcov[YY][ZZ];
        met->g.zz = gcov[ZZ][ZZ];
      };
  const auto vertical_gravity_func =
      [=] CCTK_DEVICE(const double *xx, struct four_metric *met,
                      const double *bbh_traj_loc, bool &maskL) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        double gcov[NDIM][NDIM];

        // Note mask not used used below yet
        VerticalGravity(xx, gcov, bbh_traj_loc, AST_VerticalGravity_z0,
                        AST_VerticalGravity_g);
        met->g.tt = gcov[TT][TT];
        met->g.tx = gcov[TT][XX];
        met->g.ty = gcov[TT][YY];
        met->g.tz = gcov[TT][ZZ];
        met->g.xx = gcov[XX][XX];
        met->g.xy = gcov[XX][YY];
        met->g.xz = gcov[XX][ZZ];
        met->g.yy = gcov[YY][YY];
        met->g.yz = gcov[YY][ZZ];
        met->g.zz = gcov[ZZ][ZZ];
      };

  switch (metricType) {
  case metric_t::kerrschild: {
    SetMetric(CCTK_PASS_CTOC, kerrschild_func, finite_difference_h, time + AST_t0);
    break;
  }
  case metric_t::superposed_bbh: {
    SetMetric(CCTK_PASS_CTOC, superposed_bbh_func, finite_difference_h, time + AST_t0);
    break;
  }
  case metric_t::vertical_gravity: {
    SetMetric(CCTK_PASS_CTOC, vertical_gravity_func, finite_difference_h, time + AST_t0);
    break;
  }
  default:
    assert(0);
  }
}

/* Setting metric */
extern "C" void AnalyticalSpacetimeX_SetMetric(CCTK_ARGUMENTS) {
  DECLARE_CCTK_PARAMETERS;
  DECLARE_CCTK_ARGUMENTSX_AnalyticalSpacetimeX_SetMetric;

  /* Check whether we evolve the metric at this iteration*/
  if ((cctk_iteration % evolve_metric_every != 0)) {
    return;
  }

  SetMetricHelper(CCTK_PASS_CTOC, cctk_time); 
}

void GetTraj(CCTK_ARGUMENTS, const double tt, double* bbh_traj_0) {
  DECLARE_CCTK_PARAMETERS;
  DECLARE_CCTK_ARGUMENTSX_AnalyticalSpacetimeX_SetMetric;

  const double t = tt + AST_t0;

  if (traj_read_table) {
    find_traj_t(t, bbh_traj_0);
  } else {
    calc_traj(CCTK_PASS_CTOC, t, bbh_traj_0);
  }
}

template <typename GetMetricFunc>
void SetMetric(CCTK_ARGUMENTS, GetMetricFunc &get_metric, const double h, const double tt) {
  DECLARE_CCTK_PARAMETERS;
  DECLARE_CCTK_ARGUMENTSX_AnalyticalSpacetimeX_SetMetric;

  double *bbh_traj_p1 =
      (double *)amrex::The_Managed_Arena()->alloc(NTABLES * sizeof(double));
  double *bbh_traj_0 =
      (double *)amrex::The_Managed_Arena()->alloc(NTABLES * sizeof(double));
  double *bbh_traj_m1 =
      (double *)amrex::The_Managed_Arena()->alloc(NTABLES * sizeof(double));

  /* Whether we load traj from a table or we compute analytical trajectories */
  if (traj_read_table) {
    find_traj_t(tt + h, bbh_traj_p1);
    find_traj_t(tt, bbh_traj_0);
    find_traj_t(tt - h, bbh_traj_m1);
  } else {
    calc_traj(CCTK_PASS_CTOC, tt + h, bbh_traj_p1);
    calc_traj(CCTK_PASS_CTOC, tt, bbh_traj_0);
    calc_traj(CCTK_PASS_CTOC, tt - h, bbh_traj_m1);
  }

  const auto numerical_4metric =
      [=] CCTK_DEVICE(const double h, const double *xx,
                      struct four_metric *outmet, const double *nz_m1,
                      const double *nz_0, const double *nz_p1, bool &maskL)
          CCTK_ATTRIBUTE_ALWAYS_INLINE {
            get_metric(xx, outmet, nz_0, maskL);
            bool tmp{true};
            get_dmetric_1d<TT>(outmet->g_t, h, get_metric, xx, nz_m1, nz_p1, tmp);
            get_dmetric_1d<XX>(outmet->g_x, h, get_metric, xx, nz_m1, nz_p1, tmp);
            get_dmetric_1d<YY>(outmet->g_y, h, get_metric, xx, nz_m1, nz_p1, tmp);
            get_dmetric_1d<ZZ>(outmet->g_z, h, get_metric, xx, nz_m1, nz_p1, tmp);
          };

  grid.loop_all_device<0, 0, 0>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        const double xx[NDIM] = {tt, p.x, p.y, p.z};
        struct four_metric met4;
        struct three_metric met3;

        bool maskL = true;

        /* calculate 4D metric and its derivatives */
        numerical_4metric(h, xx, &met4, bbh_traj_m1, bbh_traj_0, bbh_traj_p1, maskL);

        /* transform 4D metric to 3+1 variables*/
        four_metric_to_three_metric(&met4, &met3);

	/* write to ADM variables */
        gxx(p.I) = met3.gxx;
        gxy(p.I) = met3.gxy;
        gxz(p.I) = met3.gxz;
        gyy(p.I) = met3.gyy;
        gyz(p.I) = met3.gyz;
        gzz(p.I) = met3.gzz;

        kxx(p.I) = met3.kxx;
        kxy(p.I) = met3.kxy;
        kxz(p.I) = met3.kxz;
        kyy(p.I) = met3.kyy;
        kyz(p.I) = met3.kyz;
        kzz(p.I) = met3.kzz;

        alp(p.I) = met3.alpha;

        betax(p.I) = met3.betax;
        betay(p.I) = met3.betay;
        betaz(p.I) = met3.betaz;


        if (!maskL) {aster_mask_vc(p.I) = maskL;}

      });

  amrex::Gpu::Device::streamSynchronize();
  amrex::The_Managed_Arena()->free(bbh_traj_m1);
  amrex::The_Managed_Arena()->free(bbh_traj_0);
  amrex::The_Managed_Arena()->free(bbh_traj_p1);
}

struct traj_stencil {
  double *traj_m2;
}

void GetMetricAtPoint(const double *bbh_traj_loc0, const double *bbh_traj_loc0) {
  DECLARE_CCTK_PARAMETERS;
  DECLARE_CCTK_ARGUMENTSX_AnalyticalSpacetimeX_SetMetric;
  
  const auto superposed_bbh_func =
      [=] CCTK_DEVICE(const double *xx, struct four_metric *met,
                      const double *bbh_traj_loc, bool &maskL) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        double gcov[NDIM][NDIM];

        SuperposedBBH(xx, gcov, bbh_traj_loc, AST_adjust_mass1,
                      AST_adjust_mass2, AST_a1_buffer, AST_a2_buffer,
                      AST_cutoff_floor, maskL);
        met->g.tt = gcov[TT][TT];
        met->g.tx = gcov[TT][XX];
        met->g.ty = gcov[TT][YY];
        met->g.tz = gcov[TT][ZZ];
        met->g.xx = gcov[XX][XX];
        met->g.xy = gcov[XX][YY];
        met->g.xz = gcov[XX][ZZ];
        met->g.yy = gcov[YY][YY];
        met->g.yz = gcov[YY][ZZ];
        met->g.zz = gcov[ZZ][ZZ];
      };

  const auto kerrschild_func =
      [=] CCTK_DEVICE(const double *xx, struct four_metric *met,
                      const double *bbh_traj_loc, bool &maskL) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        double gcov[NDIM][NDIM];

        // Note mask not used below yet
        KerrSchild(xx, gcov, AST_KerrSchild__mass, AST_KerrSchild__spin);
        met->g.tt = gcov[TT][TT];
        met->g.tx = gcov[TT][XX];
        met->g.ty = gcov[TT][YY];
        met->g.tz = gcov[TT][ZZ];
        met->g.xx = gcov[XX][XX];
        met->g.xy = gcov[XX][YY];
        met->g.xz = gcov[XX][ZZ];
        met->g.yy = gcov[YY][YY];
        met->g.yz = gcov[YY][ZZ];
        met->g.zz = gcov[ZZ][ZZ];
      };
  const auto vertical_gravity_func =
      [=] CCTK_DEVICE(const double *xx, struct four_metric *met,
                      const double *bbh_traj_loc, bool &maskL) CCTK_ATTRIBUTE_ALWAYS_INLINE {
        double gcov[NDIM][NDIM];

        // Note mask not used used below yet
        VerticalGravity(xx, gcov, bbh_traj_loc, AST_VerticalGravity_z0,
                        AST_VerticalGravity_g);
        met->g.tt = gcov[TT][TT];
        met->g.tx = gcov[TT][XX];
        met->g.ty = gcov[TT][YY];
        met->g.tz = gcov[TT][ZZ];
        met->g.xx = gcov[XX][XX];
        met->g.xy = gcov[XX][YY];
        met->g.xz = gcov[XX][ZZ];
        met->g.yy = gcov[YY][YY];
        met->g.yz = gcov[YY][ZZ];
        met->g.zz = gcov[ZZ][ZZ];
      };

  switch (metricType) {
  case metric_t::kerrschild: {
    SetMetric(CCTK_PASS_CTOC, kerrschild_func, finite_difference_h, time + AST_t0);
    break;
  }
  case metric_t::superposed_bbh: {
    SetMetric(CCTK_PASS_CTOC, superposed_bbh_func, finite_difference_h, time + AST_t0);
    break;
  }
  case metric_t::vertical_gravity: {
    SetMetric(CCTK_PASS_CTOC, vertical_gravity_func, finite_difference_h, time + AST_t0);
    break;
  }
  default:
    assert(0);
  }
}

} // namespace AnalyticalSpacetimeX
