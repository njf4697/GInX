#include "RaytracingContainer.hxx"
#include "AST_Spacetime_SetMetric.hxx"

extern "C" void CheckParamatersRXAST(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;

  if (move_every != 0) {
    CCTK_ERROR("RaytracingXAST requires AnalyticalSpacetimeX::move_every to be zero. If fast_light is false, RaytracingX uses its own move_every=1");
  }
  if (resize_every != 0) {
    CCTK_ERROR("RaytracingXAST requires AnalyticalSpacetimeX::resize_every to be zero as it does not support it");
  }
  if (evolve_metric_every != 0) {
    CCTK_ERROR("RaytracingXAST requires AnalyticalSpacetimeX::evolve_metric_every to be zero. If fast_light is false, RaytracingX uses its own evolve_metric_every=1");
  }
}

extern "C" void CheckParamatersRXAST(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;

  *RX_t0 += AST_t0;
}

extern "C" void SetMetric(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_PARAMETERS;
  DECLARE_CCTK_ARGUMENTS;

  if (cctk_iteration > 0 && fast_light) { return; }

  AnalyticalSpacetimeX::SetMetricHelper(CCTK_PASS_CTOC, *RX_time);
}

extern "C" void SetMetricPlusHalfDeltaTime(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_PARAMETERS;
  DECLARE_CCTK_ARGUMENTS;

  if (num_photons == 0) { return; }
  if (cctk_iteration > 0 && fast_light) { return; }

  CCTK_REAL dt = (num_photons_out_of_bounds > 0) ? coarse_dt : CCTK_DELTA_TIME;

  AnalyticalSpacetimeX::SetMetricHelper(CCTK_PASS_CTOC, *RX_time + 0.5*dt);
}

extern "C" void SetMetricPlusDeltaTime(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;

  if (num_photons == 0) { return; }
  if (cctk_iteration > 0 && fast_light) { return; }
  
  CCTK_REAL dt = (num_photons_out_of_bounds > 0) ? coarse_dt : CCTK_DELTA_TIME;

  AnalyticalSpacetimeX::SetMetricHelper(CCTK_PASS_CTOC, *RX_time + dt);
}

extern "C" void SetPositions(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;

  if (cctk_iteration > 0 && fast_light) { return; }

  AnalyticalSpacetimeX::SetPositionsHelper(CCTK_PASS_CTOC, *RX_time);
}

extern "C" void RoveGrids(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_ARGUMENTSX_R_MoveGrids;
  DECLARE_CCTK_PARAMETERS;

  if (cctk_iteration > 0 && fast_light) { return; }

  MoveGridsHelper(CCTK_PASS_CTOC);
}