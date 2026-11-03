/**
 * \file RaytracingEvolution.cxx
 * \brief Available particle container implementations.
 *
 * This file includes the definitions of the thorn's scheduled functions. This
 * includes the set up, evolution, output particle data and the data clean up.
 * These functions are very similar to the functions found in 
 * ParticlesCcontainer/ParticlesEvolution.cxx, but with slight differences to
 * accomodate raytracing. Due to the similarities with
 * ParticlesCcontainer/ParticlesEvolution.cxx, any changes will be discussed in
 * comments starting with 'RaytracingX:' for clarity.
 */

#include "RaytracingX.h"
#include "RaytracingInitializers.hxx"

#include <cctk.h>

#include "BaseParticlesContainer.hxx"
#include "RaytracingContainer.hxx"
#include <AMReX_ParallelDescriptor.H>
#include <CParameters.h>
#include <AMReX_MFIter.H>

#include <cstring>
#include <driver.hxx>

#include <cctk_Arguments.h>
#include <cctk_Parameters.h>
#include <cctk_Types.h>
#include <cctk_core.h>

#include <iostream>
#include <loop_device.hxx>

#include "AST_Spacetime_SetMetric.hxx"

//RaytracingX: Uses override for particle data struct and particle container defined in RaytracingX/RaytracingContainer.hxx.
using ParticleData = RaytracingX::RaytracingPhotonsData;
using PC = RaytracingX::RaytracingParticlesContainer<ParticleData>;
std::vector<std::unique_ptr<PC>> r_photons;
long num_photons;
PhotonBounds bounds;
PhotonBounds grid_bounds;
bool camera_out_of_bounds;
bool camera_out_of_grid;
long num_photons_out_of_bounds;

/**
 * \brief Initialize particles' data
 *
 * This function initializes particles' position, velocity and energy
 * distributing the particle using the methods defined inside of the
 * RaytracingPhotonsInitializers.hxx. This initialization uses the camera
 * parameters given by RaytracingX.
 */
extern "C" void R_ParticlesContainer_setup(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;
  
  //RaytracingX: Add debug print statement.
  if (verbose)
  {
    CCTK_INFO("R_ParticlesContainer_setup");
  }

  if (!fast_light && move_every != 1) {
    CCTK_ERROR("RaytracingX requires AnalyticalSpacetimeX::move_every to be zero. If fast_light is false, RaytracingX uses its own move_every=1");
  }
  if (!fast_light && resize_every != 0) {
    CCTK_ERROR("RaytracingX requires AnalyticalSpacetimeX::resize_every to be zero as it does not support it");
  }
  if (!fast_light && evolve_metric_every != 1) {
    CCTK_ERROR("RaytracingX requires AnalyticalSpacetimeX::evolve_metric_every to be zero. If fast_light is false, RaytracingX uses its own evolve_metric_every=1");
  }

  assert(cctk_dim == 3);
  int gh[3];
  CCTK_GroupnghostzonesVN(
    cctkGH,
    3,
    gh,
    "ADMBaseX::alp");
  assert(MIN(MIN(gh[0], gh[1]), gh[2]) >= 3 && "RaytracingX requires ghost zone size of 3");

  *RX_time = 0.0;
  
  //Metric m;
  //interpolateMetricAtPoint(CCTK_PASS_CTOC, &m, 0.062791, 0.000061, -13.703892);
  //CCTK_INFO(m.to_string().c_str());

  //RaytracingX: Particle skip override moved to schedule.ccl

  const int tl = 0;
  const int gi_metric = CCTK_GroupIndex("ADMBaseX::metric");
  assert(gi_metric >= 0 && "Failed to get the metric group index");

  //RaytracingX: Passes camera information into arrays that can be used by the initialization funciton.
  CCTK_INT int_params[4];
  CCTK_REAL real_params[39];
  setup_camera_initializer_reals(CCTK_PASS_CTOC, real_params, camera_out_of_grid, grid_bounds);
  setup_camera_initializer_ints(CCTK_PASS_CTOC, int_params, camera_out_of_bounds, camera_out_of_grid);
  *RX_t0 = camera_out_of_grid ? get_starting_time(int_params, real_params, grid_bounds) : 0.0;
  real_params[38] = *RX_t0;

  for (int patch = 0; patch < CarpetX::ghext->num_patches(); ++patch)
  { 
    const auto &patchdata = CarpetX::ghext->patchdata.at(patch);
    if (r_photons.size() < CarpetX::ghext->num_patches())
    {
      r_photons.push_back(std::make_unique<PC>(patchdata.amrcore.get()));
      
      //RaytracingX: Change particle initialization function to that of the camera.
      auto &pc = r_photons.at(patch);

      pc->initialize(camera_initializer<ParticleData, PC>,
                     real_params, int_params);
    }
  }

  for (int patch = 0; patch < CarpetX::ghext->num_patches(); ++patch)
  {
    auto &pc = r_photons.at(patch);
    auto &pd = CarpetX::ghext->patchdata.at(patch);
    for (int lev = 0; lev < pd.leveldata.size(); ++lev)
    {

      const auto &ld = pd.leveldata.at(lev);
      const auto &gd_metric = *ld.groupdata.at(gi_metric);
      const amrex::MultiFab &metric = *gd_metric.mfab[tl];

      pc->Redistribute();
      pc->normalize_velocity(metric, lev);
    }
  }
}

/**
 * \brief Evolve the geodesics
 *
 * This function evolves the particles position by numerically solving the
 * geodesic equations.
 */
extern "C" void R_ParticlesContainer_redistribute(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_PARAMETERS;
  DECLARE_CCTK_ARGUMENTS;

  CCTK_Barrier(cctkGH);

  if (output_final_data) {
  for (int patch = 0; patch < CarpetX::ghext->num_patches(); ++patch)
  {
    auto &pc = r_photons.at(patch);
    auto &pd = CarpetX::ghext->patchdata.at(patch);
    for (int lev = 0; lev < pd.leveldata.size(); ++lev)
    { 
      pc->write_deleted_particle_data(lev, std::string(out_dir) + "/" + final_data_file_name);
    }
  }}

  CCTK_Barrier(cctkGH);

  for (int patch = 0; patch < CarpetX::ghext->num_patches(); ++patch)
  {
    auto &pc = r_photons.at(patch);
    auto &pd = CarpetX::ghext->patchdata.at(patch);
    pc->Redistribute();
  }
}

extern "C" void R_ParticlesContainer_redistribute_initial(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_PARAMETERS;
  DECLARE_CCTK_ARGUMENTS;

  CCTK_Barrier(cctkGH);

  for (int patch = 0; patch < CarpetX::ghext->num_patches(); ++patch)
  {
    auto &pc = r_photons.at(patch);
    auto &pd = CarpetX::ghext->patchdata.at(patch);
    pc->Redistribute();
  }

  CCTK_Barrier(cctkGH);
}

extern "C" void R_CheckCamera(CCTK_ARGUMENTS)
{ 
  DECLARE_CCTK_PARAMETERS;
  DECLARE_CCTK_ARGUMENTS;

  for (int patch = 0; patch < CarpetX::ghext->num_patches(); ++patch)
  {
    const auto &patchdata = CarpetX::ghext->patchdata.at(patch);

    const auto plo0 = patchdata.amrcore->Geom(0).ProbLoArray();
    const auto phi0 = patchdata.amrcore->Geom(0).ProbHiArray();
    const auto dx0 = patchdata.amrcore->Geom(0).CellSizeArray();

    grid_bounds.xmin() = plo0[0] + 3*dx0[0];
    grid_bounds.ymin() = plo0[1] + 3*dx0[1];
    grid_bounds.zmin() = plo0[2] + 3*dx0[2];
    grid_bounds.xmax() = phi0[0] - 3*dx0[0];
    grid_bounds.ymax() = phi0[1] - 3*dx0[1];
    grid_bounds.zmax() = phi0[2] - 3*dx0[2];

    bounds.xmin() = CCTK_ParameterSet("photon_xmin", "RaytracingX", "PRIVATE") ? MAX(grid_bounds.xmin(), photon_xmin) : grid_bounds.xmin();
    bounds.ymin() = CCTK_ParameterSet("photon_ymin", "RaytracingX", "PRIVATE") ? MAX(grid_bounds.ymin(), photon_ymin) : grid_bounds.ymin();
    bounds.zmin() = CCTK_ParameterSet("photon_zmin", "RaytracingX", "PRIVATE") ? MAX(grid_bounds.zmin(), photon_zmin) : grid_bounds.zmin();
    bounds.xmax() = CCTK_ParameterSet("photon_xmax", "RaytracingX", "PRIVATE") ? MIN(grid_bounds.xmax(), photon_xmax) : grid_bounds.xmax();
    bounds.ymax() = CCTK_ParameterSet("photon_ymax", "RaytracingX", "PRIVATE") ? MIN(grid_bounds.ymax(), photon_ymax) : grid_bounds.ymax();
    bounds.zmax() = CCTK_ParameterSet("photon_zmax", "RaytracingX", "PRIVATE") ? MIN(grid_bounds.zmax(), photon_zmax) : grid_bounds.zmax();

    camera_out_of_bounds = isOutOfBounds(camera_pos, bounds.arr);

    if (camera_out_of_bounds && !extended_camera) {
      CCTK_VERROR("Camera must be between (%f, %f, %f) and (%f, %f, %f).", UNPACK6V(grid_bounds.arr));
    }
    if (camera_out_of_grid && !camera_out_of_bounds) {
      CCTK_WARN("Camera inside grid but outside specified bounds. Using coarse dt until all particles are inside specified bounds");
    }
  }
}

/**
 * \brief Evolve the geodesics
 *
 * This function evolves the particles position by numerically solving the
 * geodesic equations.
 */
extern "C" void R_ParticlesContainer_evolvek1(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_PARAMETERS;
  DECLARE_CCTK_ARGUMENTS;

  if (num_photons == 0) { return; }

  CCTK_Barrier(cctkGH);

  CCTK_REAL dt = (num_photons_out_of_bounds > 0) ? cctk_delta_time : CCTK_DELTA_TIME;

  const int tl = 0;
  const int gi_lapse = CCTK_GroupIndex("ADMBaseX::lapse");
  const int gi_shift = CCTK_GroupIndex("ADMBaseX::shift");
  const int gi_metric = CCTK_GroupIndex("ADMBaseX::metric");
  const int gi_curv = CCTK_GroupIndex("ADMBaseX::curv");
  const int gi_rho = CCTK_GroupIndex("HydroBaseX::rho");
  assert(gi_lapse >= 0 && "Failed to get the lapse group index");
  assert(gi_shift >= 0 && "Failed to get the shift group index");
  assert(gi_metric >= 0 && "Failed to get the metric group index");
  assert(gi_curv >= 0 && "Failed to get the curvature group index");
  assert(gi_rho >= 0 && "Failed to get the density group index");

  //CCTK_Sync(cctkGH, gi_lapse);
  //CCTK_Sync(cctkGH, gi_shift);
  //CCTK_Sync(cctkGH, gi_metric);
  //CCTK_Sync(cctkGH, gi_curv);
  //CCTK_Sync(cctkGH, gi_rho);
  CCTK_Barrier(cctkGH);

  for (int patch = 0; patch < CarpetX::ghext->num_patches(); ++patch)
  {
    auto &pc = r_photons.at(patch);
    auto &pd = CarpetX::ghext->patchdata.at(patch);

    for (int lev = 0; lev < pd.leveldata.size(); ++lev)
    {
      //RaytracingX: Add density to information passed to evolution function.
      const auto &ld = pd.leveldata.at(lev);
      const auto &gd_lapse = *ld.groupdata.at(gi_lapse);
      const auto &gd_shift = *ld.groupdata.at(gi_shift);
      const auto &gd_metric = *ld.groupdata.at(gi_metric);
      const auto &gd_curv = *ld.groupdata.at(gi_curv);
      const auto &gd_rho = *ld.groupdata.at(gi_rho);
      const amrex::MultiFab &lapse = *gd_lapse.mfab[tl];
      const amrex::MultiFab &shift = *gd_shift.mfab[tl];
      const amrex::MultiFab &metric = *gd_metric.mfab[tl];
      const amrex::MultiFab &curv = *gd_curv.mfab[tl];
      const amrex::MultiFab &rho = *gd_rho.mfab[tl];

      //pc->check_horizon(lapse, lev, max_energy);

      //RaytracingX: Add density to information used in evolution function. Also uses an override for the evolution function that evolves optical depth
      // along geodesic. Information for particle output on deletion also passed.
      pc->evolve_k1(cctk_iteration, lapse, shift, metric, curv, rho, *RX_time, lev, max_energy, dt, (num_photons_out_of_bounds > 0) ? grid_bounds : bounds);
    }
  }
}

/**
 * \brief Evolve the geodesics
 *
 * This function evolves the particles position by numerically solving the
 * geodesic equations.
 */
extern "C" void R_ParticlesContainer_evolvek2(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_PARAMETERS;
  DECLARE_CCTK_ARGUMENTS;

  if (num_photons == 0) { return; }

  CCTK_Barrier(cctkGH);

  CCTK_REAL dt = (num_photons_out_of_bounds > 0) ? cctk_delta_time : CCTK_DELTA_TIME;

  const int tl = 0;
  const int gi_lapse = CCTK_GroupIndex("ADMBaseX::lapse");
  const int gi_shift = CCTK_GroupIndex("ADMBaseX::shift");
  const int gi_metric = CCTK_GroupIndex("ADMBaseX::metric");
  const int gi_curv = CCTK_GroupIndex("ADMBaseX::curv");
  const int gi_rho = CCTK_GroupIndex("HydroBaseX::rho");
  assert(gi_lapse >= 0 && "Failed to get the lapse group index");
  assert(gi_shift >= 0 && "Failed to get the shift group index");
  assert(gi_metric >= 0 && "Failed to get the metric group index");
  assert(gi_curv >= 0 && "Failed to get the curvature group index");
  assert(gi_rho >= 0 && "Failed to get the density group index");

  //CCTK_Sync(cctkGH, gi_lapse);
  //CCTK_Sync(cctkGH, gi_shift);
  //CCTK_Sync(cctkGH, gi_metric);
  //CCTK_Sync(cctkGH, gi_curv);
  //CCTK_Sync(cctkGH, gi_rho);
  CCTK_Barrier(cctkGH);

  for (int patch = 0; patch < CarpetX::ghext->num_patches(); ++patch)
  {
    auto &pc = r_photons.at(patch);
    auto &pd = CarpetX::ghext->patchdata.at(patch);

    for (int lev = 0; lev < pd.leveldata.size(); ++lev)
    {
      //RaytracingX: Add density to information passed to evolution function.
      const auto &ld = pd.leveldata.at(lev);
      const auto &gd_lapse = *ld.groupdata.at(gi_lapse);
      const auto &gd_shift = *ld.groupdata.at(gi_shift);
      const auto &gd_metric = *ld.groupdata.at(gi_metric);
      const auto &gd_curv = *ld.groupdata.at(gi_curv);
      const auto &gd_rho = *ld.groupdata.at(gi_rho);
      const amrex::MultiFab &lapse = *gd_lapse.mfab[tl];
      const amrex::MultiFab &shift = *gd_shift.mfab[tl];
      const amrex::MultiFab &metric = *gd_metric.mfab[tl];
      const amrex::MultiFab &curv = *gd_curv.mfab[tl];
      const amrex::MultiFab &rho = *gd_rho.mfab[tl];

      //pc->check_horizon(lapse, lev, max_energy);

      //RaytracingX: Add density to information used in evolution function. Also uses an override for the evolution function that evolves optical depth
      // along geodesic. Information for particle output on deletion also passed.
      pc->evolve_k2(cctk_iteration, lapse, shift, metric, curv, rho, *RX_time, lev, max_energy, dt, (num_photons_out_of_bounds > 0) ? grid_bounds : bounds);
    }
  }
}

/**
 * \brief Evolve the geodesics
 *
 * This function evolves the particles position by numerically solving the
 * geodesic equations.
 */
extern "C" void R_ParticlesContainer_evolvek3(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_PARAMETERS;
  DECLARE_CCTK_ARGUMENTS;

  if (num_photons == 0) { return; }

  CCTK_Barrier(cctkGH);

  CCTK_REAL dt = (num_photons_out_of_bounds > 0) ? cctk_delta_time : CCTK_DELTA_TIME;

  const int tl = 0;
  const int gi_lapse = CCTK_GroupIndex("ADMBaseX::lapse");
  const int gi_shift = CCTK_GroupIndex("ADMBaseX::shift");
  const int gi_metric = CCTK_GroupIndex("ADMBaseX::metric");
  const int gi_curv = CCTK_GroupIndex("ADMBaseX::curv");
  const int gi_rho = CCTK_GroupIndex("HydroBaseX::rho");
  assert(gi_lapse >= 0 && "Failed to get the lapse group index");
  assert(gi_shift >= 0 && "Failed to get the shift group index");
  assert(gi_metric >= 0 && "Failed to get the metric group index");
  assert(gi_curv >= 0 && "Failed to get the curvature group index");
  assert(gi_rho >= 0 && "Failed to get the density group index");

  //CCTK_Sync(cctkGH, gi_lapse);
  //CCTK_Sync(cctkGH, gi_shift);
  //CCTK_Sync(cctkGH, gi_metric);
  //CCTK_Sync(cctkGH, gi_curv);
  //CCTK_Sync(cctkGH, gi_rho);
  CCTK_Barrier(cctkGH);

  for (int patch = 0; patch < CarpetX::ghext->num_patches(); ++patch)
  {
    auto &pc = r_photons.at(patch);
    auto &pd = CarpetX::ghext->patchdata.at(patch);

    for (int lev = 0; lev < pd.leveldata.size(); ++lev)
    {
      //RaytracingX: Add density to information passed to evolution function.
      const auto &ld = pd.leveldata.at(lev);
      const auto &gd_lapse = *ld.groupdata.at(gi_lapse);
      const auto &gd_shift = *ld.groupdata.at(gi_shift);
      const auto &gd_metric = *ld.groupdata.at(gi_metric);
      const auto &gd_curv = *ld.groupdata.at(gi_curv);
      const auto &gd_rho = *ld.groupdata.at(gi_rho);
      const amrex::MultiFab &lapse = *gd_lapse.mfab[tl];
      const amrex::MultiFab &shift = *gd_shift.mfab[tl];
      const amrex::MultiFab &metric = *gd_metric.mfab[tl];
      const amrex::MultiFab &curv = *gd_curv.mfab[tl];
      const amrex::MultiFab &rho = *gd_rho.mfab[tl];

      //pc->check_horizon(lapse, lev, max_energy);

      //RaytracingX: Add density to information used in evolution function. Also uses an override for the evolution function that evolves optical depth
      // along geodesic. Information for particle output on deletion also passed.
      pc->evolve_k3(cctk_iteration, lapse, shift, metric, curv, rho, *RX_time, lev, max_energy, dt, (num_photons_out_of_bounds > 0) ? grid_bounds : bounds);
    }
  }
}

/**
 * \brief Evolve the geodesics
 *
 * This function evolves the particles position by numerically solving the
 * geodesic equations.
 */
extern "C" void R_ParticlesContainer_evolvek4(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_PARAMETERS;
  DECLARE_CCTK_ARGUMENTS;

  if (num_photons == 0) { return; }

  CCTK_Barrier(cctkGH);

  CCTK_REAL dt = (num_photons_out_of_bounds > 0) ? cctk_delta_time : CCTK_DELTA_TIME;

  const int tl = 0;
  const int gi_lapse = CCTK_GroupIndex("ADMBaseX::lapse");
  const int gi_shift = CCTK_GroupIndex("ADMBaseX::shift");
  const int gi_metric = CCTK_GroupIndex("ADMBaseX::metric");
  const int gi_curv = CCTK_GroupIndex("ADMBaseX::curv");
  const int gi_rho = CCTK_GroupIndex("HydroBaseX::rho");
  assert(gi_lapse >= 0 && "Failed to get the lapse group index");
  assert(gi_shift >= 0 && "Failed to get the shift group index");
  assert(gi_metric >= 0 && "Failed to get the metric group index");
  assert(gi_curv >= 0 && "Failed to get the curvature group index");
  assert(gi_rho >= 0 && "Failed to get the density group index");

  //CCTK_Sync(cctkGH, gi_lapse);
  //CCTK_Sync(cctkGH, gi_shift);
  //CCTK_Sync(cctkGH, gi_metric);
  //CCTK_Sync(cctkGH, gi_curv);
  //CCTK_Sync(cctkGH, gi_rho);
  CCTK_Barrier(cctkGH);

  for (int patch = 0; patch < CarpetX::ghext->num_patches(); ++patch)
  {
    auto &pc = r_photons.at(patch);
    auto &pd = CarpetX::ghext->patchdata.at(patch);

    for (int lev = 0; lev < pd.leveldata.size(); ++lev)
    {
      //RaytracingX: Add density to information passed to evolution function.
      const auto &ld = pd.leveldata.at(lev);
      const auto &gd_lapse = *ld.groupdata.at(gi_lapse);
      const auto &gd_shift = *ld.groupdata.at(gi_shift);
      const auto &gd_metric = *ld.groupdata.at(gi_metric);
      const auto &gd_curv = *ld.groupdata.at(gi_curv);
      const auto &gd_rho = *ld.groupdata.at(gi_rho);
      const amrex::MultiFab &lapse = *gd_lapse.mfab[tl];
      const amrex::MultiFab &shift = *gd_shift.mfab[tl];
      const amrex::MultiFab &metric = *gd_metric.mfab[tl];
      const amrex::MultiFab &curv = *gd_curv.mfab[tl];
      const amrex::MultiFab &rho = *gd_rho.mfab[tl];

      //pc->check_horizon(lapse, lev, max_energy);

      //RaytracingX: Add density to information used in evolution function. Also uses an override for the evolution function that evolves optical depth
      // along geodesic. Information for particle output on deletion also passed.
      pc->evolve_k4(cctk_iteration, lapse, shift, metric, curv, rho, *RX_time, lev, max_energy, dt, (num_photons_out_of_bounds > 0) ? grid_bounds : bounds);
    }
  }
}

extern "C" void R_SetMetric(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_PARAMETERS;
  DECLARE_CCTK_ARGUMENTS;

  if (num_photons == 0) { return; }

  AnalyticalSpacetimeX::SetMetricHelper(CCTK_PASS_CTOC, *RX_time);
}

extern "C" void R_SetMetric_plus_half_dt(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_PARAMETERS;
  DECLARE_CCTK_ARGUMENTS;

  if (num_photons == 0) { return; }

  CCTK_REAL dt = (num_photons_out_of_bounds > 0) ? cctk_delta_time : CCTK_DELTA_TIME;

  AnalyticalSpacetimeX::SetMetricHelper(CCTK_PASS_CTOC, *RX_time + 0.5*dt);
}

extern "C" void R_SetMetric_plus_dt(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_PARAMETERS;
  DECLARE_CCTK_ARGUMENTS;

  if (num_photons == 0) { return; }
  
  CCTK_REAL dt = (num_photons_out_of_bounds > 0) ? cctk_delta_time : CCTK_DELTA_TIME;

  AnalyticalSpacetimeX::SetMetricHelper(CCTK_PASS_CTOC, *RX_time + dt);
}

extern "C" void R_SetPositions(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_PARAMETERS;
  DECLARE_CCTK_ARGUMENTS;

  if (num_photons == 0) { return; }

  if (fast_light) {
    AnalyticalSpacetimeX::SetPositionsHelper(CCTK_PASS_CTOC, 0.0);
  }
  else {
    AnalyticalSpacetimeX::SetPositionsHelper(CCTK_PASS_CTOC, *RX_time);
  }
}

extern "C" void R_MoveGrids(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_PARAMETERS;
  DECLARE_CCTK_ARGUMENTS;

  if (num_photons == 0) { return; }

  MoveGridsHelper(CCTK_PASS_CTOC);
}

/**
 * \brief Evolve the geodesics
 *
 * This function evolves the particles position by numerically solving the
 * geodesic equations.
 */
extern "C" void R_ParticlesContainer_bounds_check(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_PARAMETERS;
  DECLARE_CCTK_ARGUMENTS;

  if (num_photons == 0) { return; }

  const int tl = 0;
  const int gi_lapse = CCTK_GroupIndex("ADMBaseX::lapse");
  assert(gi_lapse >= 0 && "Failed to get the lapse group index");

  // Bounds check
  const CCTK_REAL regions_x[10] = {region_1_position[0], region_2_position[0],
                                   region_3_position[0], region_4_position[0],
                                   region_5_position[0], region_6_position[0],
                                   region_7_position[0], region_8_position[0],
                                   region_9_position[0], region_10_position[0]};
  const CCTK_REAL regions_y[10] = {region_1_position[1], region_2_position[1],
                                   region_3_position[1], region_4_position[1],
                                   region_5_position[1], region_6_position[1],
                                   region_7_position[1], region_8_position[1],
                                   region_9_position[1], region_10_position[1]};
  const CCTK_REAL regions_z[10] = {region_1_position[2], region_2_position[2],
                                   region_3_position[2], region_4_position[2],
                                   region_5_position[2], region_6_position[2],
                                   region_7_position[2], region_8_position[2],
                                   region_9_position[2], region_10_position[2]};
  const CCTK_REAL regions_radius[10] = {
      region_1_radius, region_2_radius, region_3_radius, region_4_radius,
      region_5_radius, region_6_radius, region_7_radius, region_8_radius,
      region_9_radius, region_10_radius};
  //RaytracingX:: New parameter for banned region for Kerr BHs.
  const CCTK_REAL regions_a[10] = {
      region_1_a, region_2_a, region_3_a, region_4_a,
      region_5_a, region_6_a, region_7_a, region_8_a,
      region_9_a, region_10_a};

  for (int patch = 0; patch < CarpetX::ghext->num_patches(); ++patch)
  {
    auto &pc = r_photons.at(patch);
    auto &pd = CarpetX::ghext->patchdata.at(patch);
    for (int lev = 0; (lev < pd.leveldata.size()); ++lev)
    {
      const auto &ld = pd.leveldata.at(lev);
      const auto &gd_lapse = *ld.groupdata.at(gi_lapse);
      const amrex::MultiFab &lapse = *gd_lapse.mfab[tl];
      //RaytracingX: Override created for check_banned_zones(). Banned zones are now given by the outer horizon of Kerr BHs as an approximation. Information for particle output on deletion also passed.
      pc->check_banned_zones(lev, banned_regions, regions_x, regions_y,
                             regions_z, regions_radius, regions_a);
      pc->check_horizon(lapse, lev, max_energy);
    }
  }
}

/**
 * \brief Evolve the geodesics
 *
 * This function evolves the particles position by numerically solving the
 * geodesic equations.
 *
extern "C" void R_ParticlesContainer_output_final_data(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_PARAMETERS;
  DECLARE_CCTK_ARGUMENTS;

  CCTK_Barrier(cctkGH);

  if (output_final_data) {
  for (int patch = 0; patch < CarpetX::ghext->num_patches(); ++patch)
  {
    auto &pc = r_photons.at(patch);
    auto &pd = CarpetX::ghext->patchdata.at(patch);
    for (int lev = 0; lev < pd.leveldata.size(); ++lev)
    { 
      pc->write_deleted_particle_data(lev, cctk_time, std::string(out_dir) + "/" + final_data_file_name);
    }
  }}

  CCTK_Barrier(cctkGH);
}
*/

extern "C" void R_ParticlesContainer_calculate_kerr_conserved(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_PARAMETERS;
  DECLARE_CCTK_ARGUMENTS;

  if (num_photons == 0) { return; }

  const int tl = 0;
  const int gi_lapse = CCTK_GroupIndex("ADMBaseX::lapse");
  const int gi_shift = CCTK_GroupIndex("ADMBaseX::shift");
  const int gi_metric = CCTK_GroupIndex("ADMBaseX::metric");
  assert(gi_lapse >= 0 && "Failed to get the lapse group index");
  assert(gi_shift >= 0 && "Failed to get the shift group index");
  assert(gi_metric >= 0 && "Failed to get the metric group index");

  for (int patch = 0; patch < CarpetX::ghext->num_patches(); ++patch)
  {
    auto &pc = r_photons.at(patch);
    auto &pd = CarpetX::ghext->patchdata.at(patch);
    for (int lev = 0; lev < pd.leveldata.size(); ++lev)
    { 
      const auto &ld = pd.leveldata.at(lev);
      const auto &gd_lapse = *ld.groupdata.at(gi_lapse);
      const amrex::MultiFab &lapse = *gd_lapse.mfab[tl];
      const auto &gd_shift = *ld.groupdata.at(gi_shift);
      const amrex::MultiFab &shift = *gd_shift.mfab[tl];
      const auto &gd_metric = *ld.groupdata.at(gi_metric);
      const amrex::MultiFab &metric = *gd_metric.mfab[tl];
      pc->calculate_kerr_conserved_quantities(lapse, shift, metric, lev);
    }
  }
}

/**
 * \brief Print out particle data.
 */
extern "C" void R_ParticlesContainer_print(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_PARAMETERS;

  //RaytracingX: Particle skip override moved to schedule.ccl.
  const int it = cctkGH->cctk_iteration;
  
  for (int patch = 0; patch < CarpetX::ghext->num_patches(); ++patch)
  {
    auto &pc = r_photons.at(patch);
    pc->outputParticlesPlot(it, particle_plot_every, std::string(out_dir));
    pc->outputParticlesAscii(it, particle_tsv_every, std::string(out_dir));
  }
}

extern "C" int R_ParticlesContainer_final_cleanup()
{
  amrex::Gpu::Device::synchronize();
  r_photons.clear();
  return 0;
}

extern "C" void CheckRaytracingParticleNumber(CCTK_ARGUMENTS)
{
  long n_local = 0;

  for (int patch = 0; patch < CarpetX::ghext->num_patches(); ++patch) {
      auto &pc = r_photons.at(patch);
      auto &pd = CarpetX::ghext->patchdata.at(patch);

      for (int lev = 0; lev < pd.leveldata.size(); ++lev) {
          n_local += pc->NumberOfParticlesAtLevel(lev, true, true);
      }
  }

  long n_global = n_local;
  amrex::ParallelDescriptor::ReduceLongSum(n_global);

  num_photons = n_global;

  CCTK_VINFO("Found %ld particles remaining.", n_global);
}

extern "C" void CheckNumberOfOutOfBoundsParticles(CCTK_ARGUMENTS)
{
  long n_local = 0;

  for (int patch = 0; patch < CarpetX::ghext->num_patches(); ++patch) {
      auto &pc = r_photons.at(patch);
      auto &pd = CarpetX::ghext->patchdata.at(patch);

      for (int lev = 0; lev < pd.leveldata.size(); ++lev) {
          n_local += pc->num_particles_out_of_bounds(lev, bounds);
      }
  }

  long n_global = n_local;
  amrex::ParallelDescriptor::ReduceLongSum(n_global);

  num_photons_out_of_bounds = n_global;

  CCTK_VINFO("Found %ld particles remaining out of specified bounds.", num_photons_out_of_bounds);
}

extern "C" void UpdateCustomTime(CCTK_ARGUMENTS) {
  DECLARE_CCTK_PARAMETERS;
  DECLARE_CCTK_ARGUMENTS;

  if (CCTK_MyProc(cctkGH) == 0) { 
    CCTK_REAL dt = (num_photons_out_of_bounds > 0) ? cctk_delta_time : CCTK_DELTA_TIME;
    *RX_time += dt;
  }

  CCTK_VINFO("Current particle time is now %f.", *RX_time);template <typename StructType, typename ParticleTile, typename ParticleContainerClass>
void fill_photon_data_standard_camera(ParticleTile& particle_tile, CCTK_INT local_particle_id, CCTK_INT* int_params, CCTK_REAL* real_params, int local_offset, CCTK_REAL starting_time) {
  DEFINE_VARIABLES_FROM_INT_PARAMS_AND_REAL_PARAMS

  auto arrdata = particle_tile.GetStructOfArrays().realarray();
  auto ptd = particle_tile.getParticleTileData();

  //Create 4-vector \chi parallel to geodesic and fill geodesic initial conditions for each pixel (see https://arxiv.org/pdf/1410.777).
      
  //Get the index for the current pixel, defined by the offset for this processer plus the number of particles created on this tile already plus the number of particles created on previous tiles.
  int pidx = local_offset + local_particle_id;

  /**
   * Convert pixel index into 2D index with:
   * i=0,                  j=0                   -> upper-left corner
   * i=num_pixels_width-1, j=0                   -> upper-right corner
   * i=0,                  j=num_pixels_height-1 -> lower-left corner
   * i=num_pixels_width-1, j=num_pixels_height-1 -> lower-right corner
   * 
   * ex: for a 4x3 pixel image:
   * 0 1 2  3       [0, 0] [1, 0], [2, 0], [3, 0]
   * 4 5 6  7   <-> [0, 1] [1, 1], [2, 1], [3, 1]
   * 8 9 10 11      [0, 2] [1, 2], [2, 2], [3, 2]
   */
  int i = pidx % num_pixels_width;
  int j = pidx / num_pixels_width;

  //Calculate offset per pixel. The offset can be thought of as $/Delta\theta$ and $\Delta\phi$ with respect to camera facing direction,
  //but is calculated by finding a vector in the equivalent direction. This also gives the direction corresponding to the center of the pixels.
  CCTK_REAL a_adj = ((2.0 * (i + 0.5)) / num_pixels_width - 1.0) * tan(alpha_h / 2.0);  // a_{adj} = (2a-1)tan(\alpha_h/2)
  CCTK_REAL b_adj = ((2.0 * (j + 0.5)) / num_pixels_height - 1.0) * tan(alpha_v / 2.0); // b_{adj} = (2b-1)tan(\alpha_v/2)

  CCTK_REAL C = sqrt(1 + pow(b_adj, 2) + pow(a_adj, 2));

  CCTK_REAL chi[4];
  chi[0] = C * e0[0] - e1[0] + b_adj * e2[0] - a_adj * e3[0];
  chi[1] = C * e0[1] - e1[1] + b_adj * e2[1] - a_adj * e3[1];
  chi[2] = C * e0[2] - e1[2] + b_adj * e2[2] - a_adj * e3[2];
  chi[3] = C * e0[3] - e1[3] + b_adj * e2[3] - a_adj * e3[3];

  CCTK_REAL chi_lower[4];
  vectorToOneFormArr(chi_lower, chi, real_params);

  ptd.id(local_particle_id) = ParticleContainerClass::ParticleType::NextID();
  ptd.cpu(local_particle_id) = amrex::ParallelDescriptor::MyProc();

  ptd.pos(0, local_particle_id) = camera_pos[0];
  ptd.pos(1, local_particle_id) = camera_pos[1];
  ptd.pos(2, local_particle_id) = camera_pos[2];
  CCTK_REAL A = 1 / (lapse * chi[0]);

  arrdata[StructType::vx][local_particle_id] = chi_lower[1] * A; 
  arrdata[StructType::vy][local_particle_id] = chi_lower[2] * A;
  arrdata[StructType::vz][local_particle_id] = chi_lower[3] * A;
  arrdata[StructType::ln_E][local_particle_id] = 0;
  arrdata[StructType::tau][local_particle_id] = 0;
  arrdata[StructType::time][local_particle_id] = starting_time;

  //The pixel indices *should* be ints, but the data structure GInX::PhotonsData does not easily extend with integer parameters, so it is stored as a real instead. While this is hacky, it is done for clarity and ease of output. It is possible to add an integer variable at runtime, but this wouldn't be written to disk in the standard amrex particle output routine.
  arrdata[StructType::pixel_number][local_particle_id] = (CCTK_REAL) pidx;
  arrdata[StructType::deletion_reason][local_particle_id] = 0.0;

  for (int i = StructType::U0; i < StructType::n_attributes; i++) {
    arrdata[i][local_particle_id] = 0.0;
  }

  if (calculate_kerr_conserved_quantities) {
    CCTK_REAL shift_raised[3];
    oneformToVectorSpatial(shift_raised, shift_lower, real_params);
    const CCTK_REAL V_down[3] = {arrdata[StructType::vx][local_particle_id], arrdata[StructType::vy][local_particle_id], arrdata[StructType::vz][local_particle_id]};

    arrdata[StructType::U0][local_particle_id] = (lapse - (shift_raised[0]*V_down[0] + shift_raised[1]*V_down[1] + shift_raised[2]*V_down[2]));
    arrdata[StructType::U1][local_particle_id] = (camera_pos[0]*V_down[1] - camera_pos[1]*V_down[0]);
    arrdata[StructType::U2][local_particle_id] = spatialInnerProductArr(V_down, real_params);
  }
}

template <typename StructType, typename ParticleTile, typename ParticleContainerClass>
void fill_photon_data_camera_out_of_grid(ParticleTile& particle_tile, CCTK_INT local_particle_id, CCTK_INT* int_params, CCTK_REAL* real_params, int local_offset, CCTK_REAL starting_time) {
  DEFINE_VARIABLES_FROM_INT_PARAMS_AND_REAL_PARAMS

  auto arrdata = particle_tile.GetStructOfArrays().realarray();
  auto ptd = particle_tile.getParticleTileData();

  //Create 4-vector \chi parallel to geodesic and fill geodesic initial conditions for each pixel (see https://arxiv.org/pdf/1410.777).
      
  //Get the index for the current pixel, defined by the offset for this processer plus the number of particles created on this tile already plus the number of particles created on previous tiles.
  int pidx = local_offset + local_particle_id;

  /**
   * Convert pixel index into 2D index with:
   * i=0,                  j=0                   -> upper-left corner
   * i=num_pixels_width-1, j=0                   -> upper-right corner
   * i=0,                  j=num_pixels_height-1 -> lower-left corner
   * i=num_pixels_width-1, j=num_pixels_height-1 -> lower-right corner
   * 
   * ex: for a 4x3 pixel image:
   * 0 1 2  3       [0, 0] [1, 0], [2, 0], [3, 0]
   * 4 5 6  7   <-> [0, 1] [1, 1], [2, 1], [3, 1]
   * 8 9 10 11      [0, 2] [1, 2], [2, 2], [3, 2]
   */
  int i = pidx % num_pixels_width;
  int j = pidx / num_pixels_width;

  //Calculate offset per pixel. The offset can be thought of as $/Delta\theta$ and $\Delta\phi$ with respect to camera facing direction,
  //but is calculated by finding a vector in the equivalent direction. This also gives the direction corresponding to the center of the pixels.
  CCTK_REAL a_adj = ((2.0 * (i + 0.5)) / num_pixels_width - 1.0) * tan(alpha_h / 2.0);  // a_{adj} = (2a-1)tan(\alpha_h/2)
  CCTK_REAL b_adj = ((2.0 * (j + 0.5)) / num_pixels_height - 1.0) * tan(alpha_v / 2.0); // b_{adj} = (2b-1)tan(\alpha_v/2)

  CCTK_REAL ray_dir[3];
  for (int d = 0; d < 3; d++) {
    ray_dir[d] =  e1[d+1] - b_adj*e2[d+1] + a_adj*e3[d+1];
  }
  CCTK_REAL ray_inv_mag = 1.0 / sqrt(ray_dir[0]*ray_dir[0] + ray_dir[1]*ray_dir[1] + ray_dir[2]*ray_dir[2]);
  for (int d = 0; d < 3; d++) {
    ray_dir[d] *= ray_inv_mag;
  }

  const CCTK_REAL pos[3] = {camera_pos[0] + starting_time*ray_dir[0], camera_pos[1] + starting_time*ray_dir[1], camera_pos[2] + starting_time*ray_dir[2]};

  ptd.id(local_particle_id) = ParticleContainerClass::ParticleType::NextID();
  ptd.cpu(local_particle_id) = amrex::ParallelDescriptor::MyProc();

  ptd.pos(0, local_particle_id) = pos[0];
  ptd.pos(1, local_particle_id) = pos[1];
  ptd.pos(2, local_particle_id) = pos[2];

  arrdata[StructType::vx][local_particle_id] = -ray_dir[0]; 
  arrdata[StructType::vy][local_particle_id] = -ray_dir[1];
  arrdata[StructType::vz][local_particle_id] = -ray_dir[2];
  arrdata[StructType::ln_E][local_particle_id] = 0;
  arrdata[StructType::tau][local_particle_id] = 0;
  arrdata[StructType::time][local_particle_id] = starting_time;

  //The pixel indices *should* be ints, but the data structure GInX::PhotonsData does not easily extend with integer parameters, so it is stored as a real instead. While this is hacky, it is done for clarity and ease of output. It is possible to add an integer variable at runtime, but this wouldn't be written to disk in the standard amrex particle output routine.
  arrdata[StructType::pixel_number][local_particle_id] = (CCTK_REAL) pidx;
  arrdata[StructType::deletion_reason][local_particle_id] = 0.0;

  for (int i = StructType::U0; i < StructType::n_attributes; i++) {
    arrdata[i][local_particle_id] = 0.0;
  }
}
}