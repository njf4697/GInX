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

namespace AnalyticalSpacetimeX {

void SetMetricHelper(CCTK_ARGUMENTS, const double time);

template <typename GetMetricFunc>
void SetMetric(CCTK_ARGUMENTS, GetMetricFunc &get_metric, const double h, const double tt);

void GetTraj(CCTK_ARGUMENTS, const double tt, double* out);

void GetMetricAtPoint(const double *xx, const double *bbh_traj, four_metric *met);

} // namespace AnalyticalSpacetimeX
