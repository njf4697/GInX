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
#include "RaytracingInitializers.hxx"

#include <driver.hxx>

using ParticleData = RaytracingX::RaytracingPhotonsData;
using PC = RaytracingX::RaytracingParticlesContainer<ParticleData>;
std::vector<std::unique_ptr<PC>> r_photons;
PhotonBounds bounds;
PhotonBounds grid_bounds;
bool camera_out_of_bounds;
bool camera_out_of_grid;
CCTK_REAL coarse_dt;

extern "C" void CheckParamaters(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;

  assert(cctk_dim == 3);
  int gh[3];
  CCTK_GroupnghostzonesVN(
    cctkGH,
    3,
    gh,
    "ADMBaseX::alp");
  assert(MIN(MIN(gh[0], gh[1]), gh[2]) >= 3 && "RaytracingX requires ghost zone size of 3");
}

extern "C" void CheckCamera(CCTK_ARGUMENTS)
{ 
  DECLARE_CCTK_PARAMETERS;
  DECLARE_CCTK_ARGUMENTS;

  for (int patch = 0; patch < CarpetX::ghext->num_patches(); ++patch)
  {
    const auto &patchdata = CarpetX::ghext->patchdata.at(patch);

    const auto plo0 = patchdata.amrcore->Geom(0).ProbLoArray();
    const auto phi0 = patchdata.amrcore->Geom(0).ProbHiArray();
    const auto dx0 = patchdata.amrcore->Geom(0).CellSizeArray();

    coarse_dt = coarse_dtfac * MIN(dx0[0], MIN(dx0[1], dx0[2]));

    grid_bounds.xmin() = plo0[0] + 3*dx0[0];
    grid_bounds.ymin() = plo0[1] + 3*dx0[1];
    grid_bounds.zmin() = plo0[2] + 3*dx0[2];
    grid_bounds.xmax() = phi0[0] - 3*dx0[0];
    grid_bounds.ymax() = phi0[1] - 3*dx0[1];
    grid_bounds.zmax() = phi0[2] - 3*dx0[2];

    camera_out_of_grid = isOutOfBounds(camera_pos, grid_bounds.arr);

    bounds.xmin() = use_custom_bounds ? MAX(grid_bounds.xmin(), photon_xmin) : grid_bounds.xmin();
    bounds.ymin() = use_custom_bounds ? MAX(grid_bounds.ymin(), photon_ymin) : grid_bounds.ymin();
    bounds.zmin() = use_custom_bounds ? MAX(grid_bounds.zmin(), photon_zmin) : grid_bounds.zmin();
    bounds.xmax() = use_custom_bounds ? MIN(grid_bounds.xmax(), photon_xmax) : grid_bounds.xmax();
    bounds.ymax() = use_custom_bounds ? MIN(grid_bounds.ymax(), photon_ymax) : grid_bounds.ymax();
    bounds.zmax() = use_custom_bounds ? MIN(grid_bounds.zmax(), photon_zmax) : grid_bounds.zmax();

    if (use_custom_bounds) {
      assert(photon_xmin < photon_xmax);
      assert(photon_ymin < photon_ymax);
      assert(photon_zmin < photon_zmax);
    }

    camera_out_of_bounds = isOutOfBounds(camera_pos, bounds.arr);

    if (camera_out_of_bounds && !extended_camera) {
      CCTK_VERROR("Camera must be between (%f, %f, %f) and (%f, %f, %f).", UNPACK6V(grid_bounds.arr));
    }
    if (camera_out_of_bounds && !camera_out_of_grid) {
      CCTK_VWARN(1,"Camera inside grid but outside specified bounds. Using coarse dt until all particles are inside specified bounds");
    }
  }

  *RX_time = 0.0;
  *RX_t0 = 0.0;
  *RX_dt = 0.0;
  if (camera_out_of_grid && !fast_light) {
    *RX_time = -get_starting_time(CCTK_PASS_CTOC, grid_bounds);
  }
  if (camera_out_of_bounds) {
    get_starting_time(CCTK_PASS_CTOC, bounds); //errors if not pointing in correct direction
  }
}

extern "C" void ParticlesInit(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;

  const int tl = 0;
  const int gi_metric = CCTK_GroupIndex("ADMBaseX::metric");
  assert(gi_metric >= 0 && "Failed to get the metric group index");

  //RaytracingX: Passes camera information into arrays that can be used by the initialization funciton.
  CCTK_INT int_params[5];
  CCTK_REAL real_params[39];
  setup_camera_initializer_reals(CCTK_PASS_CTOC, real_params, camera_out_of_grid, grid_bounds);
  setup_camera_initializer_ints(CCTK_PASS_CTOC, int_params, camera_out_of_bounds, camera_out_of_grid);
  real_params[38] = get_starting_time(CCTK_PASS_CTOC, grid_bounds);

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
  CCTK_VINFO("Photon starting time is %f.", *RX_time + *RX_t0);
}

/**
 * \brief Print out particle data.
 */
extern "C" void ParticlesPrint(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_PARAMETERS;

  //RaytracingX: Particle skip override moved to schedule.ccl.
  const int it = cctkGH->cctk_iteration;
  
  for (int patch = 0; patch < CarpetX::ghext->num_patches(); ++patch)
  {
    auto &pc = r_photons.at(patch);
    pc->outputParticlesAscii(it, particle_tsv_every, std::string(out_dir));
  }
}

extern "C" void CheckParticleNumber(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_PARAMETERS;
  DECLARE_CCTK_ARGUMENTS;

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

  *num_photons = n_global;

  CCTK_VINFO("Found %ld particles remaining.", n_global);
}

extern "C" void CheckOutOfBoundsParticlesNumber(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_PARAMETERS;
  DECLARE_CCTK_ARGUMENTS;

  long n_local = 0;

  for (int patch = 0; patch < CarpetX::ghext->num_patches(); ++patch) {
      auto &pc = r_photons.at(patch);
      auto &pd = CarpetX::ghext->patchdata.at(patch);

      for (int lev = 0; lev < pd.leveldata.size(); ++lev) {
          n_local += pc->num_particles_out_of_bounds(lev, bounds);
          fprintf(stderr, "%i\n", n_local);
      }
  }

  long n_global = n_local;
  amrex::ParallelDescriptor::ReduceLongSum(n_global);

  *num_photons_out_of_bounds = n_global;

  CCTK_VINFO("Found %ld particles remaining out of specified bounds.", *num_photons_out_of_bounds);
}

/**
 * \brief Evolve the geodesics
 *
 * This function evolves the particles position by numerically solving the
 * geodesic equations.
 */
extern "C" void CheckHorizonAndBannedRegions(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_PARAMETERS;
  DECLARE_CCTK_ARGUMENTS;

  if (*num_photons == 0) { return; }

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

extern "C" void SetDeltaTime(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_PARAMETERS;
  DECLARE_CCTK_ARGUMENTS;

  *RX_dt = (*num_photons_out_of_bounds > 0) ? coarse_dt : CCTK_DELTA_TIME;
}

extern "C" void UpdateCustomTime(CCTK_ARGUMENTS) {
  DECLARE_CCTK_PARAMETERS;
  DECLARE_CCTK_ARGUMENTS;

  *RX_time += *RX_dt;

  CCTK_VINFO("Current particle time is now %f.", *RX_time + *RX_t0);
}

extern "C" void ParticlesRedistribute(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_PARAMETERS;
  DECLARE_CCTK_ARGUMENTS;

  for (int patch = 0; patch < CarpetX::ghext->num_patches(); ++patch)
  {
    auto &pc = r_photons.at(patch);
    auto &pd = CarpetX::ghext->patchdata.at(patch);
    pc->Redistribute();
  }

  CCTK_Barrier(cctkGH);
}

void ParticlesEvolveKNHelper(CCTK_ARGUMENTS, const int k) {
  DECLARE_CCTK_PARAMETERS;
  DECLARE_CCTK_ARGUMENTS;

  if (*num_photons == 0) { return; }

  CCTK_Barrier(cctkGH);

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
      pc->evolve_kn(k, cctk_iteration, lapse, shift, metric, curv, rho, *RX_time + *RX_t0, lev, max_energy, *RX_dt, (*num_photons_out_of_bounds > 0) ? grid_bounds : bounds);
    }
  }
}

/**
 * \brief Evolve the geodesics
 *
 * This function evolves the particles position by numerically solving the
 * geodesic equations.
 */
extern "C" void ParticlesEvolveK1(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_PARAMETERS;
  DECLARE_CCTK_ARGUMENTS;

  if (*num_photons == 0) { return; }

  ParticlesEvolveKNHelper(CCTK_PASS_CTOC, 1);
}

extern "C" void ParticlesEvolveK2(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_PARAMETERS;
  DECLARE_CCTK_ARGUMENTS;

  if (*num_photons == 0) { return; }

  ParticlesEvolveKNHelper(CCTK_PASS_CTOC, 2);
}

extern "C" void ParticlesEvolveK3(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_PARAMETERS;
  DECLARE_CCTK_ARGUMENTS;

  if (*num_photons == 0) { return; }

  ParticlesEvolveKNHelper(CCTK_PASS_CTOC, 3);
}

extern "C" void ParticlesEvolveK4(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_PARAMETERS;
  DECLARE_CCTK_ARGUMENTS;

  if (*num_photons == 0) { return; }

  ParticlesEvolveKNHelper(CCTK_PASS_CTOC, 4);
}

extern "C" void CalculateKerrConserved(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_PARAMETERS;
  DECLARE_CCTK_ARGUMENTS;

  if (*num_photons == 0) { return; }

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
 * \brief Evolve the geodesics
 *
 * This function evolves the particles position by numerically solving the
 * geodesic equations.
 */
extern "C" void ParticlesPrintFinalData(CCTK_ARGUMENTS)
{
  DECLARE_CCTK_PARAMETERS;
  DECLARE_CCTK_ARGUMENTS;

  for (int patch = 0; patch < CarpetX::ghext->num_patches(); ++patch)
  {
    auto &pc = r_photons.at(patch);
    auto &pd = CarpetX::ghext->patchdata.at(patch);
    for (int lev = 0; lev < pd.leveldata.size(); ++lev)
    { 
      pc->write_deleted_particle_data(lev, std::string(out_dir) + "/" + final_data_file_name);
    }
  }
  }

extern "C" int ParticlesContainerCleanUp()
{
  amrex::Gpu::Device::synchronize();
  r_photons.clear();
  return 0;
}