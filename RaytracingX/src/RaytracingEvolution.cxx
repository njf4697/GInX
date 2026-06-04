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

  assert(cctk_dim == 3);
  int gh[3];
  CCTK_GroupnghostzonesVN(
    cctkGH,
    3,
    gh,
    "ADMBaseX::alp");
  assert(MIN(MIN(gh[0], gh[1]), gh[2]) >= 5 && "RaytracingX requires ghost zone size of 5");

  //RaytracingX: Particle skip override moved to schedule.ccl

  const int tl = 0;
  const int gi_metric = CCTK_GroupIndex("ADMBaseX::metric");
  assert(gi_metric >= 0 && "Failed to get the metric group index");

  //RaytracingX: Passes camera information into arrays that can be used by the initialization funciton.
  CCTK_INT int_params[2];
  CCTK_REAL real_params[32];
  setup_camera_initializer_reals(CCTK_PASS_CTOC, real_params);
  setup_camera_initializer_ints(CCTK_PASS_CTOC, int_params);

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

  for (int patch = 0; patch < CarpetX::ghext->num_patches(); ++patch)
  {
    auto &pc = r_photons.at(patch);
    auto &pd = CarpetX::ghext->patchdata.at(patch);
    pc->Redistribute();
  }
}

/**
 * \brief Evolve the geodesics
 *
 * This function evolves the particles position by numerically solving the
 * geodesic equations.
 */
extern "C" void R_ParticlesContainer_evolve(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_PARAMETERS;
  DECLARE_CCTK_ARGUMENTS;

  if (num_photons == 0) { return; }

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
      pc->evolve(cctk_iteration, lapse, shift, metric, curv, rho, CCTK_DELTA_TIME, lev, max_energy);
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
      pc->evolve_k1(cctk_iteration, lapse, shift, metric, curv, rho, CCTK_DELTA_TIME, lev, max_energy);
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
      pc->evolve_k2(cctk_iteration, lapse, shift, metric, curv, rho, CCTK_DELTA_TIME, lev, max_energy);
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
      pc->evolve_k3(cctk_iteration, lapse, shift, metric, curv, rho, CCTK_DELTA_TIME, lev, max_energy);
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
      pc->evolve_k4(cctk_iteration, lapse, shift, metric, curv, rho, CCTK_DELTA_TIME, lev, max_energy);
    }
  }
}

extern "C" void R_SetMetric(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_PARAMETERS;
  DECLARE_CCTK_ARGUMENTS;

  if (num_photons == 0) { return; }

  AnalyticalSpacetimeX::SetMetricHelper(CCTK_PASS_CTOC, cctk_time);
}

extern "C" void R_SetMetric_plus_half_dt(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_PARAMETERS;
  DECLARE_CCTK_ARGUMENTS;

  if (num_photons == 0) { return; }

  AnalyticalSpacetimeX::SetMetricHelper(CCTK_PASS_CTOC, cctk_time + 0.5*CCTK_DELTA_TIME);
}

extern "C" void R_SetMetric_plus_dt(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_PARAMETERS;
  DECLARE_CCTK_ARGUMENTS;

  if (num_photons == 0) { return; }

  AnalyticalSpacetimeX::SetMetricHelper(CCTK_PASS_CTOC, cctk_time + CCTK_DELTA_TIME);
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
    for (int lev = 0; (lev < pd.leveldata.size()) & banned_regions; ++lev)
    {
      const auto &ld = pd.leveldata.at(lev);
      const auto &gd_lapse = *ld.groupdata.at(gi_lapse);
      const amrex::MultiFab &lapse = *gd_lapse.mfab[tl];
      //RaytracingX: Override created for check_banned_zones(). Banned zones are now given by the outer horizon of Kerr BHs as an approximation. Information for particle output on deletion also passed.
      pc->check_banned_zones(lev, banned_regions, regions_x, regions_y,
                             regions_z, regions_radius, regions_a);
      pc->check_horizon(lapse, lev, max_energy)
    }
  }
}

/**
 * \brief Evolve the geodesics
 *
 * This function evolves the particles position by numerically solving the
 * geodesic equations.
 */
extern "C" void R_ParticlesContainer_output_final_data(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_PARAMETERS;
  DECLARE_CCTK_ARGUMENTS;

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
          n_local += pc->NumberOfParticlesAtLevel(lev);
      }
  }

  long n_global = n_local;
  amrex::ParallelDescriptor::ReduceLongSum(n_global);

  num_photons = n_global;

  CCTK_VINFO("Found %ld particles remaining.", n_global);
}