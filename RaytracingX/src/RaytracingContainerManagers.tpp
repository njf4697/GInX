using namespace RaytracingX;

template <typename StructType>
void RaytracingParticlesContainer<StructType>::check_horizon(
    const amrex::MultiFab &lapse,
    const int &lev,
    const CCTK_REAL max_energy)
{
    const auto plo0 = this->Geom(0).ProbLoArray();
    const auto phi0 = this->Geom(0).ProbHiArray();

    const auto dx = this->Geom(lev).CellSizeArray();

    for (GInX::ParticleIterator<StructType> pti(*this, lev); pti.isValid();
         ++pti)
    {

        const int np = pti.numParticles();

        // Get the information relate to the velocities and energy.
        auto &attribs = pti.GetAttributes();
        CCTK_REAL *AMREX_RESTRICT ln_alphaenergy = attribs[StructType::ln_alphaE].data();
        CCTK_REAL *AMREX_RESTRICT deletion_reasons = attribs[StructType::deletion_reason].data(); // RaytracingX: Add deletion reason.
        auto *AMREX_RESTRICT particles = &(pti.GetArrayOfStructs()[0]);

        // Get the array of each parameter.
        auto const lapse_array = lapse.array(pti);

        // Needed for GPU
        auto self = this;

        amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int i) noexcept
        {
            //RaytracingX: Delete particle when geodesic reaches event horizon.
            const long int i0 = get_interpolation_center(particles[i].pos(0), plo0[0], phi0[0], dx[0]);
            const long int j0 = get_interpolation_center(particles[i].pos(1), plo0[1], phi0[1], dx[1]);
            const long int k0 = get_interpolation_center(particles[i].pos(2), plo0[2], phi0[2], dx[2]);
            // Interpolate lapse & partial lapse at \vect{x}
            CCTK_REAL lapse_x;
            amrex::GpuArray<CCTK_REAL, 3> d_lapse_x;
            GInX::d_interpolate_array<5>(lapse_x, d_lapse_x, lapse_array, i0, j0, k0, particles[i].pos(0), particles[i].pos(1),
                                         particles[i].pos(2), dx, plo0);
            if (abs(exp(ln_alphaenergy[i]) / lapse_x) > max_energy) {
              particles[i].id() =-1;
              deletion_reasons[i] = -7;
            } 
        });
    }
}

/**
 * The check banned zones function check for user defined invalid particles
 * zones.
 * RaytracingX: Changed to work with spinning BHs and output data.
 *
 * @param level Adaptive Mesh Refinement level
 * @param zones Number of banned zones
 * @param x x-coordinates array for each region
 * @param y y-coordinates array for each region
 * @param z z-coordinates array for each region
 * @param radius Radius array for each region
 */
template <typename StructType>
void RaytracingParticlesContainer<StructType>::check_banned_zones(
    const int &level,
    const CCTK_INT4 &zones,
    const CCTK_REAL (&x)[10],
    const CCTK_REAL (&y)[10],
    const CCTK_REAL (&z)[10],
    const CCTK_REAL (&radius)[10],
    const CCTK_REAL (&a)[10]) // RaytracingX: Add output data.
{

    if (!zones)
    {
        return;
    }

    for (GInX::ParticleIterator<StructType> pti(*this, level);
         pti.isValid(); ++pti)
    {
        const int np = pti.numParticles();
        auto *AMREX_RESTRICT particles = &(pti.GetArrayOfStructs()[0]);

        auto &attribs = pti.GetAttributes();
        CCTK_REAL *AMREX_RESTRICT deletion_reasons = attribs[StructType::deletion_reason].data(); // RaytracingX: Add deletion reason.

        auto self = this;
        amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int i) noexcept
                           {
        for (int check = 0; check < zones; check++) {
          const CCTK_REAL dx = particles[i].pos(0) - x[check];
          const CCTK_REAL dy = particles[i].pos(1) - y[check];
          const CCTK_REAL dz = particles[i].pos(2) - z[check];

          //RaytracingX: Change to work for spinning BHs.
          const CCTK_REAL R2minusa2 = dx*dx + dy*dy + dz*dz - a[check]*a[check];
          const CCTK_REAL r = sqrt(R2minusa2 + sqrt(R2minusa2*R2minusa2+4*a[check]*a[check]*z[check]*z[check])) / 2;

          if (!(r > 0)) { CCTK_ERROR("Issue with calculating distance to banned zone."); }
          
          if (r <= (radius[check] + sqrt(radius[check]*radius[check]-4*a[check]*a[check])) / 2.0) {
            particles[i].id() = -1;
            deletion_reasons[i] = -check - 9;
          }
        } });
    }
}

/**
 * \brief Normalize the velocity accordingly to the metric on each particle
 * position.
 *
 * This function is made to normalize the velocity given a random initial data
 * using the Photons positions by using that
 *
 *  \f[
 *  P^\mu P_\mu = 0.
 *  \f]
 *
 *  or equivalently
 *
 *  \f[
 *  V^\alpha V_\alpha = V_\alpha V_\beta \gamma^{\alpha\beta} = 1.
 *  \f]
 *
 * @param metric ADM 3 dimension metric.
 * @param Current refinement level.
 */
template <typename StructType>
void RaytracingParticlesContainer<StructType>::normalize_velocity(
    const amrex::MultiFab &metric, const int level)
{

    // Get the with of the discretization on each direction.
    const auto dx = this->Geom(level).CellSizeArray();
    // Get the lower and higher value over the ParticleContainer Geometry
    const auto p_lo = this->Geom(level).ProbLoArray();
    const auto p_hi = this->Geom(level).ProbHiArray();

    for (amrex::MFIter mfi = this->MakeMFIter(level); mfi.isValid(); ++mfi)
    {

        // Get a reference to the particles
        auto &particle_tile = this->DefineAndReturnParticleTile(level, mfi);

        // Determines the current size and the required new size
        const auto current_size = particle_tile.GetArrayOfStructs().size();

        // Gets raw pointers to the two different ways particle data is stored for
        // performance reasons: Array of Struct (AoS) and Struct of Arrays (SoA)
        auto *p_struct = particle_tile.GetArrayOfStructs()().data();
        auto arrdata = particle_tile.GetStructOfArrays().realarray();

        // get the current process id
        const auto metric_array = metric.array(mfi);
        const CCTK_REAL m = this->mass;

        amrex::ParallelFor(current_size, [=] AMREX_GPU_DEVICE(int i) noexcept
        {
            // Start a for loop with Random Number evolution for the velocity
            const CCTK_REAL ratio[3] = {arrdata[StructType::vx][i],
                                        arrdata[StructType::vy][i],
                                        arrdata[StructType::vz][i]};
            const CCTK_REAL E = std::exp(arrdata[StructType::ln_alphaE][i]);
            
            // Generate a random position
            const auto &p = p_struct[i];
            
            const long int i0 = get_interpolation_center(p.pos(0), p_lo[0], p_hi[0], dx[0]);
            const long int j0 = get_interpolation_center(p.pos(1), p_lo[1], p_hi[1], dx[1]);
            const long int k0 = get_interpolation_center(p.pos(2), p_lo[2], p_hi[2], dx[2]);
            
            // Interpolate metric
            amrex::GpuArray<CCTK_REAL, 6> gamma_x;
            GInX::interpolate_array<5>(gamma_x, metric_array, i0, j0, k0, p.pos(0),
                                 p.pos(1), p.pos(2), dx, p_lo);
            
            const CCTK_REAL inv_det_gamma =
                1.0 / (gamma_x[0] * gamma_x[3] * gamma_x[5] +
                       2. * gamma_x[1] * gamma_x[2] * gamma_x[4] -
                       gamma_x[2] * gamma_x[2] * gamma_x[3] -
                       gamma_x[4] * gamma_x[4] * gamma_x[0] -
                       gamma_x[1] * gamma_x[1] * gamma_x[5]);
                
            const amrex::GpuArray<CCTK_REAL, 6> gamma_inv_x = {
                (gamma_x[3] * gamma_x[5] - gamma_x[4] * gamma_x[4]) * inv_det_gamma,
                (gamma_x[4] * gamma_x[2] - gamma_x[1] * gamma_x[5]) * inv_det_gamma,
                (gamma_x[1] * gamma_x[4] - gamma_x[2] * gamma_x[3]) * inv_det_gamma,
                (gamma_x[0] * gamma_x[5] - gamma_x[2] * gamma_x[2]) * inv_det_gamma,
                (gamma_x[2] * gamma_x[1] - gamma_x[0] * gamma_x[4]) * inv_det_gamma,
                (gamma_x[0] * gamma_x[3] - gamma_x[1] * gamma_x[1]) * inv_det_gamma};
            
            // Normalizing the velocity.
            const CCTK_REAL v_squared = ratio[0] * ratio[0] * gamma_inv_x[0] +
                                        ratio[1] * ratio[1] * gamma_inv_x[3] +
                                        ratio[2] * ratio[2] * gamma_inv_x[5] +
                                        2.0 * ratio[0] * ratio[1] * gamma_inv_x[1] +
                                        2.0 * ratio[0] * ratio[2] * gamma_inv_x[2] +
                                        2.0 * ratio[1] * ratio[2] * gamma_inv_x[4];
            
            const CCTK_REAL v = std::sqrt(v_squared);
            const CCTK_REAL alpha = std::sqrt(1. - m * m / (E * E));

            arrdata[StructType::vx][i] = ratio[0] * alpha / v;
            arrdata[StructType::vy][i] = ratio[1] * alpha / v;
            arrdata[StructType::vz][i] = ratio[2] * alpha / v; 
        });
    }
} // RaytracingParticlesContainer::normalize_velocity