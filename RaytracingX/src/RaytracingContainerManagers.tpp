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
        CCTK_REAL *AMREX_RESTRICT ln_energy = attribs[StructType::ln_E].data();
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
            GInX::interpolate_array<5>(lapse_x, lapse_array, i0, j0, k0, particles[i].pos(0), particles[i].pos(1),
                                         particles[i].pos(2), dx, plo0);
            if (ln_energy[i] > log(max_energy * lapse_x)) {
              particles[i].id() =-1;
              deletion_reasons[i] = DelReason::HORIZON;
            } 
        });
    }
}

template <typename StructType>
void RaytracingParticlesContainer<StructType>::calculate_kerr_conserved_quantities(
    const amrex::MultiFab &lapse,
    const amrex::MultiFab &shift,
    const amrex::MultiFab &metric,
    const int &lev)
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
        CCTK_REAL *AMREX_RESTRICT ln_energy = attribs[StructType::ln_E].data();
        CCTK_REAL *AMREX_RESTRICT vels_x = attribs[StructType::vx].data();
        CCTK_REAL *AMREX_RESTRICT vels_y = attribs[StructType::vy].data();
        CCTK_REAL *AMREX_RESTRICT vels_z = attribs[StructType::vz].data();
        CCTK_REAL *AMREX_RESTRICT p_0 = attribs[StructType::U0].data();
        CCTK_REAL *AMREX_RESTRICT L_z = attribs[StructType::U1].data();
        CCTK_REAL *AMREX_RESTRICT V_sqr = attribs[StructType::U2].data();
        CCTK_REAL *AMREX_RESTRICT deletion_reasons = attribs[StructType::deletion_reason].data(); // RaytracingX: Add deletion reason.
        auto *AMREX_RESTRICT particles = &(pti.GetArrayOfStructs()[0]);

        // Get the array of each parameter.
        auto const lapse_array = lapse.array(pti);
        auto const shift_array = shift.array(pti);
        auto const metric_array = metric.array(pti);

        // Needed for GPU
        auto self = this;

        amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int i) noexcept
        {   
            if (particles[i].id() == -1) { return; }

            //RaytracingX: Delete particle when geodesic reaches event horizon.
            const long int i0 = get_interpolation_center(particles[i].pos(0), plo0[0], phi0[0], dx[0]);
            const long int j0 = get_interpolation_center(particles[i].pos(1), plo0[1], phi0[1], dx[1]);
            const long int k0 = get_interpolation_center(particles[i].pos(2), plo0[2], phi0[2], dx[2]);
            // Interpolate lapse & partial lapse at \vect{x}
            CCTK_REAL lapse_x;
            GInX::interpolate_array<5>(lapse_x, lapse_array, i0, j0, k0, particles[i].pos(0), particles[i].pos(1),
                                         particles[i].pos(2), dx, plo0);
            amrex::GpuArray<CCTK_REAL, 3> shift_x;
            GInX::interpolate_array<5>(shift_x, shift_array, i0, j0, k0, particles[i].pos(0), particles[i].pos(1),
                                         particles[i].pos(2), dx, plo0);
            amrex::GpuArray<CCTK_REAL, 6> gamma_x;
            GInX::interpolate_array<5>(gamma_x, metric_array, i0, j0, k0, particles[i].pos(0), particles[i].pos(1),
                                         particles[i].pos(2), dx, plo0);
            const CCTK_REAL E = exp(ln_energy[i]);
            p_0[i] = E * (lapse_x - (shift_x[0]*vels_x[i] + shift_x[1]*vels_y[i] + shift_x[2]*vels_z[i]));
            L_z[i] = E * (particles[i].pos(0)*vels_y[i] - particles[i].pos(1)*vels_x[i]);

            const CCTK_REAL inv_det_gamma = INV_DET_GAMMA(gamma_x);
            const amrex::GpuArray<CCTK_REAL, 6> gamma_inv_x = INV_GAMMA(gamma_x, inv_det_gamma);
            
            amrex::GpuArray<CCTK_REAL, 3> V_down = {vels_x[i], vels_y[i], vels_z[i]};
            
            V_sqr[i] =  SPATIAL_INNER_PRODUCT(V_down, gamma_inv_x);
        });
    }
}

template <typename StructType>
CCTK_REAL RaytracingParticlesContainer<StructType>::calculate_dt(
    const amrex::MultiFab &lapse,
    const amrex::MultiFab &shift,
    const amrex::MultiFab &metric,
    const amrex::MultiFab &curv,
    const CCTK_REAL dtfac,
    const int &lev)
{
    const auto plo0 = this->Geom(0).ProbLoArray();
    const auto phi0 = this->Geom(0).ProbHiArray();

    const auto dx = this->Geom(lev).CellSizeArray();

    CCTK_REAL* d_min_dt = static_cast<CCTK_REAL*>(amrex::The_Managed_Arena()->alloc(sizeof(CCTK_REAL)));
    *d_min_dt = std::numeric_limits<CCTK_REAL>::max();

    for (GInX::ParticleIterator<StructType> pti(*this, lev); pti.isValid();
         ++pti)
    {   
        const int np = pti.numParticles();

        // Get the information relate to the velocities and energy.
        auto &attribs = pti.GetAttributes();
        
        CCTK_REAL *AMREX_RESTRICT vels_x = attribs[StructType::vx].data();
        CCTK_REAL *AMREX_RESTRICT vels_y = attribs[StructType::vy].data();
        CCTK_REAL *AMREX_RESTRICT vels_z = attribs[StructType::vz].data();
        CCTK_REAL *AMREX_RESTRICT lnE = attribs[StructType::ln_E].data();
        CCTK_REAL *AMREX_RESTRICT dt = attribs[StructType::dt].data();
        CCTK_REAL *AMREX_RESTRICT del_rsn = attribs[StructType::deletion_reason].data();
        auto *AMREX_RESTRICT particles = &(pti.GetArrayOfStructs()[0]);

        CCTK_REAL *min_dt = d_min_dt;

        // Get the array of each parameter.
        auto const lapse_array = lapse.array(pti);
        auto const shift_array = shift.array(pti);
        auto const metric_array = metric.array(pti);
        auto const curv_array = curv.array(pti);

        // Needed for GPU
        auto self = this;

        amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int i) noexcept
        {   
            if (particles[i].id() == -1) { return; }

            amrex::GpuArray<CCTK_REAL, 3> xvec = {particles[i].pos(0), particles[i].pos(1), particles[i].pos(2)};
            amrex::GpuArray<CCTK_REAL, 3> vvec = {vels_x[i], vels_y[i], vels_z[i]};
            amrex::GpuArray<CCTK_REAL, 3> dxvecdt = {0.0, 0.0, 0.0};
            amrex::GpuArray<CCTK_REAL, 3> dvvecdt = {0.0, 0.0, 0.0};

            const CCTK_REAL max_dx = fmax(dx[0], fmax(dx[1], dx[2]));
            CCTK_REAL lapse_x = 0.0;

            dt[i] = max_dx;

            const CCTK_REAL dt1 = get_dt(0.0, dxvecdt, dvvecdt, xvec, vvec, plo0, phi0, dx, lapse_array, shift_array, metric_array, curv_array, dtfac, 0.0, lapse_x, lev);
            //if (dt1 > max_dx) { fprintf(stderr, "dt=%f>dx=%f, E=%f/%f=%f\n", dt1, max_dx, exp(lnE[i]), lapse_x, exp(lnE[i])/lapse_x); particles[i].id() = -1; del_rsn[i] = DelReason::UNSTABLE; dt[i] = dt1; return; }
            if (exp(lnE[i]) / lapse_x > 5.0) {particles[i].id() = -1; del_rsn[i] = DelReason::UNSTABLE; dt[i] = max_dx; return;}
            const CCTK_REAL dt2 = get_dt(fmin(max_dx, dt1), dxvecdt, dvvecdt, xvec, vvec, plo0, phi0, dx, lapse_array, shift_array, metric_array, curv_array, dtfac, 0.5, lapse_x, lev);
            //if (dt2 > max_dx) { fprintf(stderr, "dt=%f>dx=%f, E=%f/%f=%f\n", dt2, max_dx, exp(lnE[i]), lapse_x, exp(lnE[i])/lapse_x); particles[i].id() = -1; del_rsn[i] = DelReason::UNSTABLE; dt[i] = dt2; return; }
            if (exp(lnE[i]) / lapse_x > 5.0) {particles[i].id() = -1; del_rsn[i] = DelReason::UNSTABLE; dt[i] = max_dx; return;}
            const CCTK_REAL dt3 = get_dt(fmin(max_dx, dt2), dxvecdt, dvvecdt, xvec, vvec, plo0, phi0, dx, lapse_array, shift_array, metric_array, curv_array, dtfac, 0.5, lapse_x, lev);
            //if (dt3 > max_dx) { fprintf(stderr, "dt=%f>dx=%f, E=%f/%f=%f\n", dt3, max_dx, exp(lnE[i]), lapse_x, exp(lnE[i])/lapse_x); particles[i].id() = -1; del_rsn[i] = DelReason::UNSTABLE; dt[i] = dt3; return; }
            if (exp(lnE[i]) / lapse_x > 5.0) {particles[i].id() = -1; del_rsn[i] = DelReason::UNSTABLE; dt[i] = max_dx; return;}
            const CCTK_REAL dt4 = get_dt(fmin(max_dx, dt3), dxvecdt, dvvecdt, xvec, vvec, plo0, phi0, dx, lapse_array, shift_array, metric_array, curv_array, dtfac, 1.0, lapse_x, lev);
            //if (dt4 > max_dx) { fprintf(stderr, "dt=%f>dx=%f, E=%f/%f=%f\n", dt4, max_dx, exp(lnE[i]), lapse_x, exp(lnE[i])/lapse_x); particles[i].id() = -1; del_rsn[i] = DelReason::UNSTABLE; dt[i] = dt4; return; }
            if (exp(lnE[i]) / lapse_x > 5.0) {particles[i].id() = -1; del_rsn[i] = DelReason::UNSTABLE; dt[i] = max_dx; return;}

            dt[i] = fmin(fmin(dt1, dt2), fmin(dt3, dt4));

            if (dt[i] < 0.00000001) {particles[i].id() = -1; del_rsn[i] = DelReason::UNSTABLE; dt[i] = max_dx; }

            amrex::Gpu::Atomic::Min(min_dt, dt[i]);
        });
    }
    amrex::Gpu::streamSynchronize();
    const CCTK_REAL dt_local = *d_min_dt;
    amrex::The_Managed_Arena()->free(d_min_dt);
    return dt_local;
}

template <typename StructType>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE CCTK_ATTRIBUTE_ALWAYS_INLINE
CCTK_REAL RaytracingParticlesContainer<StructType>::get_dt(
    const CCTK_REAL prev_dt,
    amrex::GpuArray<CCTK_REAL, 3> &dxvecdt,
    amrex::GpuArray<CCTK_REAL, 3> &dvvecdt,
    amrex::GpuArray<CCTK_REAL, 3> &xvec,
    amrex::GpuArray<CCTK_REAL, 3> &vvec,
    const amrex::GpuArray<double, 3> plo0,
    const amrex::GpuArray<double, 3> phi0,
    const amrex::GpuArray<double, 3> dx,
    const amrex::Array4<CCTK_REAL const> &lapse,
    const amrex::Array4<CCTK_REAL const> &shift,
    const amrex::Array4<CCTK_REAL const> &metric,
    const amrex::Array4<CCTK_REAL const> &curv,
    const CCTK_REAL dtfac,
    const CCTK_REAL rk4_dtfac,
    CCTK_REAL &lapse_x,
    const int &lev)
{
    xvec = {xvec[0] - rk4_dtfac*dxvecdt[0]*prev_dt, xvec[1] - rk4_dtfac*dxvecdt[1]*prev_dt, xvec[2] - rk4_dtfac*dxvecdt[2]*prev_dt};
    vvec = {vvec[0] - rk4_dtfac*dvvecdt[0]*prev_dt, vvec[1] - rk4_dtfac*dvvecdt[1]*prev_dt, vvec[2] - rk4_dtfac*dvvecdt[2]*prev_dt};

    const long int i0 = get_interpolation_center(xvec[0], plo0[0], phi0[0], dx[0]);
    const long int j0 = get_interpolation_center(xvec[1], plo0[1], phi0[1], dx[1]);
    const long int k0 = get_interpolation_center(xvec[2], plo0[2], phi0[2], dx[2]);
    
    // Interpolate lapse & partial lapse at \vect{x}
    amrex::GpuArray<CCTK_REAL, 3> d_lapse_x;
    GInX::d_interpolate_array<5>(lapse_x, d_lapse_x, lapse, i0, j0, k0, xvec[0], xvec[1],
                                 xvec[2], dx, plo0);

    // Interpolate shift & partial shift at \vect{x}
    amrex::GpuArray<CCTK_REAL, 3> shift_x;
    amrex::GpuArray<amrex::GpuArray<CCTK_REAL, 3>, 3> d_shift_x;
    GInX::d_interpolate_array<5>(shift_x, d_shift_x, shift, i0, j0, k0, xvec[0], xvec[1],
                                 xvec[2], dx, plo0);

    // Interpolate metric & partial metric at \vect{x}
    amrex::GpuArray<CCTK_REAL, 6> gamma_x;
    amrex::GpuArray<amrex::GpuArray<CCTK_REAL, 6>, 3> d_gamma_x;
    GInX::d_interpolate_array<5>(gamma_x, d_gamma_x, metric, i0, j0, k0, xvec[0], xvec[1],
                                 xvec[2], dx, plo0);

    // Interpolate Curvature at \vect{x}
    amrex::GpuArray<CCTK_REAL, 6> curv_x;
    GInX::interpolate_array<5>(curv_x, curv, i0, j0, k0, xvec[0], xvec[1], xvec[2], dx, plo0);
    
    const CCTK_REAL inv_det_gamma = INV_DET_GAMMA(gamma_x);
    const amrex::GpuArray<CCTK_REAL, 6> gamma_inv_x = INV_GAMMA(gamma_x, inv_det_gamma);
    const amrex::GpuArray<CCTK_REAL, 3> V_down = {vvec[0], vvec[1], vvec[2]};
    const amrex::GpuArray<CCTK_REAL, 3> V_up = RAISE_SPATIAL(V_down, gamma_inv_x);

    dxvecdt[0] = lapse_x*V_up[0]-shift_x[0];
    dxvecdt[1] = lapse_x*V_up[1]-shift_x[1];
    dxvecdt[2] = lapse_x*V_up[2]-shift_x[2];

    for (int i = 0; i < 3; i++)  //Uidx::vx = 3, Uidx::vx + 1 = Uidx::vy = 4, etc.
    {
        dvvecdt[i] =
            -d_lapse_x[i] +
            (VecVecMul(d_lapse_x, V_up) -
             lapse_x * VecVecMul(SMatVecMul(curv_x, V_up), V_up)) *
                V_down[i] +
            0.5 * lapse_x * VecVecMul(SMatVecMul(d_gamma_x[i], V_up), V_up) +
            VecVecMul(V_down, d_shift_x[i]);
    }


    const CCTK_REAL eps = 1e-14;
    const amrex::GpuArray<CCTK_REAL, 3> dt_vec = {dx[0] / fmax(fabs(dxvecdt[0]), eps),
                                                  dx[1] / fmax(fabs(dxvecdt[1]), eps),
                                                  dx[2] / fmax(fabs(dxvecdt[2]), eps)};
    return dtfac * fmin(fmin(1.0, dt_vec[0]), fmin(dt_vec[1], dt_vec[2]));
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
          if (radius[check] < a[check] * 2) {
            CCTK_VERROR("Banned Zone %i exceeds its maximum spin", check);
          }

          const CCTK_REAL dx = particles[i].pos(0) - x[check];
          const CCTK_REAL dy = particles[i].pos(1) - y[check];
          const CCTK_REAL dz = particles[i].pos(2) - z[check];

          //RaytracingX: Change to work for spinning BHs.
          const CCTK_REAL R2minusa2 = dx*dx + dy*dy + dz*dz - a[check]*a[check];
          const CCTK_REAL r = sqrt((R2minusa2 + sqrt(R2minusa2*R2minusa2+4*a[check]*a[check]*z[check]*z[check])) / 2);

          if (!(r > 0)) { CCTK_ERROR("Issue with calculating distance to banned zone."); }
          
          if (r <= (radius[check] + sqrt(radius[check]*radius[check]-4*a[check]*a[check])) / 2.0) {
            particles[i].id() = -1;
            deletion_reasons[i] = -check - DelReason::BANNED_REGION_OFFSET;
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
            const CCTK_REAL E = std::exp(arrdata[StructType::ln_E][i]);
            
            // Generate a random position
            const auto &p = p_struct[i];
            
            const long int i0 = get_interpolation_center(p.pos(0), p_lo[0], p_hi[0], dx[0]);
            const long int j0 = get_interpolation_center(p.pos(1), p_lo[1], p_hi[1], dx[1]);
            const long int k0 = get_interpolation_center(p.pos(2), p_lo[2], p_hi[2], dx[2]);
            
            // Interpolate metric
            amrex::GpuArray<CCTK_REAL, 6> gamma_x;
            GInX::interpolate_array<5>(gamma_x, metric_array, i0, j0, k0, p.pos(0),
                                 p.pos(1), p.pos(2), dx, p_lo);
            
            const CCTK_REAL inv_det_gamma = INV_DET_GAMMA(gamma_x);
                
            const amrex::GpuArray<CCTK_REAL, 6> gamma_inv_x = INV_GAMMA(gamma_x, inv_det_gamma);
            
            // Normalizing the velocity.
            const CCTK_REAL v_squared = SPATIAL_INNER_PRODUCT(ratio, gamma_inv_x);
            
            const CCTK_REAL v = std::sqrt(v_squared);
            const CCTK_REAL alpha = std::sqrt(1. - m * m / (E * E));

            arrdata[StructType::vx][i] = ratio[0] * alpha / v;
            arrdata[StructType::vy][i] = ratio[1] * alpha / v;
            arrdata[StructType::vz][i] = ratio[2] * alpha / v; 
        });
    }
} // RaytracingParticlesContainer::normalize_velocity