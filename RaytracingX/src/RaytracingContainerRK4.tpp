using namespace RaytracingX;

/**
 * \brief Computes the right hand side of the geodesic differential equation.
 *
 * Given differential equation \f[\frac{d}{dt}U = f\left(U, \frac{dU}{dx};
 * t\right)\f] computes
 * \f[f\left(U, \frac{dU}{dx}; t\right)\f]
 *
 * where \f$U\f$ is a vector which contains \f$(x, y, z, v_x, v_y, v_z,
 * \ln E)\f$. The differential equation for the particles' position is
 *
 * \f[\frac{d}{dt} U[i] = \alpha \gamma^{ij} U[3 + j] - \beta^i\f]
 *
 * Where \f$i,j = 0, 1, 2\f$, \f$\gamma\f$ is the induced metric, \f$\alpha\f$
 * is the lapse function and \f$\beta\f$ is the shift vector.
 *
 * For the Velocity_d the differential equation is:
 *
 * \f{eqnarray*}{
 * \frac{d}{dt}U[3 + i] &= -\partial_i\alpha + \left(\gamma^{kj} U[3 + k]
 * \partial_j\alpha - \alpha K_{jk}\gamma^{jl}\gamma^{km}U[3+l]U[3+m]\right) U[3
 * + i]\\ & +
 * \frac{1}{2}\alpha\gamma^{jl}\gamma^{km}U[3+l]U[3+m]\partial_i\gamma{jk} + U[3
 * + j] \partial_i\beta^j
 * \f}
 *
 * and finally, for the \f$ \ln \alphaE \f$ the differential equation is:
 *
 * \f[ \dfrac{d}{dt} U[6] = \alpha K_{jk}U[3 + l]U[3 + m]\gamma^{lj}\gamma^{mk}
 * - U[3+l]\gamma^{lj}\partial_j\alpha\f]
 *
 *  Where \f$i, j, k, l, m = 0, 1, 2\f$ and \f$K_{ij}\f$ is the extrinsic
 * curvature. We have been using Einstein notation.
 *
 *  RaytracingX: optical depth iteration as \f \frac{d\tau}{ds}=\kappa\rho \f so \frac{d}{dt}\tau = \kappa\rho\frac{ds}{dt}
 *
 *  @param u A GpuArray of size n_attributes + the coordinates that contains the
 * varaibles needed to evolve.
 *  @param t Current time t.
 *  @param lapse ADM lapse function.
 *  @param shift Shift vector \beta^i
 *  @param metric 3 dimensional ADM metric.
 *  @param curv Extrinsic curvature.
 *  @param rho RaytracingX: Gas density.
 *  @param dt Timestep.
 *  @param dx Spacestep
 *  @param lev AMR Level of discretization.
 *  @param plo Physical lower bounds of the whole domain.
 *  @return The right hind side of the differential equation.
 */
template <typename StructType>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE CCTK_ATTRIBUTE_ALWAYS_INLINE
amrex::GpuArray<CCTK_REAL, 9>
RaytracingParticlesContainer<StructType>::compute_rhs(
    const int iteration,
    const int index,
    const amrex::GpuArray<CCTK_REAL, 9> &u,
    const CCTK_REAL &t,
    amrex::Array4<CCTK_REAL const> const &lapse,
    const amrex::Array4<CCTK_REAL const> &shift,
    const amrex::Array4<CCTK_REAL const> &metric,
    const amrex::Array4<CCTK_REAL const> &curv,
    const amrex::Array4<CCTK_REAL const> &rho,
    const CCTK_REAL dt,
    const amrex::GpuArray<double, 3> &dx,
    const int lev,
    const amrex::GpuArray<double, 3> &plo,
    const amrex::GpuArray<double, 3> &phi,
    const CCTK_REAL max_energy, 
    const CCTK_REAL mass)
{

    // RaytracingX: Add space for optical depth variable.
    amrex::GpuArray<CCTK_REAL, 9> rhs = {0., 0., 0., 0., 0., 0., 0., 0., 0.};

    const long int i0 = get_interpolation_center(u[0], plo[0], phi[0], dx[0]);
    const long int j0 = get_interpolation_center(u[1], plo[1], phi[1], dx[1]);
    const long int k0 = get_interpolation_center(u[2], plo[2], phi[2], dx[2]);

    // Interpolate lapse & partial lapse at \vect{x}
    CCTK_REAL lapse_x;
    amrex::GpuArray<CCTK_REAL, 3> d_lapse_x;
    GInX::d_interpolate_array<5>(lapse_x, d_lapse_x, lapse, i0, j0, k0, u[0], u[1],
                                 u[2], dx, plo);

    // Interpolate shift & partial shift at \vect{x}
    amrex::GpuArray<CCTK_REAL, 3> shift_x;
    amrex::GpuArray<amrex::GpuArray<CCTK_REAL, 3>, 3> d_shift_x;
    GInX::d_interpolate_array<5>(shift_x, d_shift_x, shift, i0, j0, k0, u[0], u[1],
                                 u[2], dx, plo);

    // Interpolate metric & partial metric at \vect{x}
    amrex::GpuArray<CCTK_REAL, 6> gamma_x;
    amrex::GpuArray<amrex::GpuArray<CCTK_REAL, 6>, 3> d_gamma_x;
    GInX::d_interpolate_array<5>(gamma_x, d_gamma_x, metric, i0, j0, k0, u[0], u[1],
                                 u[2], dx, plo);

    // Interpolate Curvature at \vect{x}
    amrex::GpuArray<CCTK_REAL, 6> curv_x;
    GInX::interpolate_array<5>(curv_x, curv, i0, j0, k0, u[0], u[1], u[2], dx, plo);

    // RaytracingX: Interpolate density for optical depth calculation.
    //  Interpolate rho at \vect{x}
    CCTK_REAL rho_x;
    GInX::d_interpolate_array<5>(rho_x, rho, i0, j0, k0, u[0], u[1],
                                 u[2], dx, plo);

    //lapse_x = 1;
    //d_lapse_x[0] = 0;
    //d_lapse_x[1] = 0;
    //d_lapse_x[2] = 0;
    //shift_x[0] = 0;
    //shift_x[1] = 0;
    //shift_x[2] = 0;
    //d_shift_x[0][0] = 0;
    //d_shift_x[1][0] = 0;
    //d_shift_x[2][0] = 0;
    //d_shift_x[0][1] = 0;
    //d_shift_x[1][1] = 0;
    //d_shift_x[2][1] = 0;
    //d_shift_x[0][2] = 0;
    //d_shift_x[1][2] = 0;
    //d_shift_x[2][2] = 0;
    //gamma_x[0] = 1;
    //gamma_x[1] = 0;
    //gamma_x[2] = 0;
    //gamma_x[3] = 1;
    //gamma_x[4] = 0;
    //gamma_x[5] = 1;
    //d_gamma_x[0][0] = 0;
    //d_gamma_x[1][0] = 0;
    //d_gamma_x[2][0] = 0;
    //d_gamma_x[3][0] = 0;
    //d_gamma_x[4][0] = 0;
    //d_gamma_x[5][1] = 0;
    //d_gamma_x[0][1] = 0;
    //d_gamma_x[1][1] = 0;
    //d_gamma_x[2][1] = 0;
    //d_gamma_x[3][1] = 0;
    //d_gamma_x[4][1] = 0;
    //d_gamma_x[5][0] = 0;
    //d_gamma_x[0][2] = 0;
    //d_gamma_x[1][2] = 0;
    //d_gamma_x[2][2] = 0;
    //d_gamma_x[3][2] = 0;
    //d_gamma_x[4][2] = 0;
    //d_gamma_x[5][2] = 0;
    //curv_x[0] = 0;
    //curv_x[1] = 0;
    //curv_x[2] = 0;
    //curv_x[3] = 0;
    //curv_x[4] = 0;
    //curv_x[5] = 0;

    ASSERT_FINITE(lapse_x)
    ASSERT_FINITE1(d_lapse_x, 3)
    ASSERT_FINITE1(shift_x, 3)
    ASSERT_FINITE2(d_shift_x, 3, 3)
    ASSERT_FINITE1(gamma_x, 6)
    ASSERT_FINITE2(d_gamma_x, 6, 3)
    ASSERT_FINITE1(curv_x, 6)

    // Compute the inverse of the metric.
    const CCTK_REAL inv_det_gamma = INV_DET_GAMMA(gamma_x);

    const amrex::GpuArray<CCTK_REAL, 6> gamma_inv_x = INV_GAMMA(gamma_x, inv_det_gamma)

    const amrex::GpuArray<CCTK_REAL, 3> V_down = {u[Uidx::vx], u[Uidx::vy], u[Uidx::vz]};

    //const CCTK_REAL v_squared = V_down[0] * V_down[0] * gamma_inv_x[0] +
    //                            V_down[1] * V_down[1] * gamma_inv_x[3] +
    //                            V_down[2] * V_down[2] * gamma_inv_x[5] +
    //                            2.0 * V_down[0] * V_down[1] * gamma_inv_x[1] +
    //                            2.0 * V_down[0] * V_down[2] * gamma_inv_x[2] +
    //                            2.0 * V_down[1] * V_down[2] * gamma_inv_x[4];
//
//
    //ASSERT_FINITE(mass)
    //ASSERT_FINITE(v_squared)
    //        
    //const CCTK_REAL v = std::sqrt(v_squared);
//
    //ASSERT_FINITE(v)
//
    //const CCTK_REAL A = std::sqrt(1. - mass * mass / (exp(2*u[6])));
//
    //V_down[0] *= A / v;
    //V_down[1] *= A / v;
    //V_down[2] *= A / v; 

    // Compute the upper index velocity terms.
    amrex::GpuArray<CCTK_REAL, 3> V_up = RAISE_SPATIAL(V_down, gamma_inv_x);

    // Compute the rhs for position
    rhs[0] = lapse_x * V_up[0] - shift_x[0];
    rhs[1] = lapse_x * V_up[1] - shift_x[1];
    rhs[2] = lapse_x * V_up[2] - shift_x[2];

    // Compute the rhs for velocity
    for (int i = 0; i < 3; i++)  //Uidx::vx = 3, Uidx::vx + 1 = Uidx::vy = 4, etc.
    {
        rhs[Uidx::vx + i] =
            -d_lapse_x[i] +
            (VecVecMul(d_lapse_x, V_up) -
             lapse_x * VecVecMul(SMatVecMul(curv_x, V_up), V_up)) *
                V_down[i] +
            0.5 * lapse_x * VecVecMul(SMatVecMul(d_gamma_x[i], V_up), V_up) +
            VecVecMul(V_down, d_shift_x[i]);
    }

    // Compute the rhs for energy
    rhs[Uidx::lnE] =
        lapse_x * VecVecMul(SMatVecMul(curv_x, V_up), V_up) -
        VecVecMul(V_up, d_lapse_x);

    // RaytracingX: Add evolution for optical depth calculation.
    //  Compute the rhs for optical depth
    //TODO: implement optical depth
    //const CCTK_REAL ds = mag2_massless(dx[0], dx[1], dx[2], gamma_inv_x);
    rhs[Uidx::vx] = 0.0; //(0.4 * cgs2cactusOpacity) * (rho_x * cgs2cactusDensity) * (ds / dt);

    rhs[Uidx::del_rsn] = check_validity(rhs, u, lapse_x, max_energy, index);

    return rhs;
} // RaytracingParticlesContainer::compute_rhs

/**
 *  \brief Evolving using Runge-Kutta 4.
 *
 * We are solving the differential equation
 * \f$\frac{dU}{dt} = f\left(U, \frac{dU}{dx}, t\right)\f$ using:
 *
 *  \f[
 *  U_{n+1} = U_n + \frac{1}{6}\Delta t \left(f_1 + 2f_2 + 2f_3 + f_4\right)
 *  \f]
 *
 *  where:
 *
 *  * \f$f_1 = f(U_n, t),\f$
 *  * \f$f_2 = f\left(U_n + \frac{\Delta t}{2} f_1, t + \frac{\Delta
 * t}{2}\right),\f$
 *  * \f$f_3 = f\left(U_n + \frac{\Delta t}{2} f_2, t + \frac{\Delta
 * t}{2}\right),\f$
 *  * \f$f_4 = f(U_n + \Delta t f_3, t + \Delta t),\f$
 *
 *  While computing we are checking if the particles still in the physical domain.
 *
 *  @see compute_rhs()
 *  @param lapse ADM lapse function.
 *  @param shift ADM shift vector.
 *  @param metric ADM induced metric.
 *  @param curv Extrinsic curvature.
 *  @param rho RaytracingX: gas density
 *  @param dt Timestep.
 *  @param lev Refinement level.
 *  @param max_energy RaytracingX: Maximum energy threshold for event horizon detection.
 */
template <typename StructType>
void RaytracingParticlesContainer<StructType>::evolve_k1(
    const int iteration,
    const amrex::MultiFab &lapse,
    const amrex::MultiFab &shift,
    const amrex::MultiFab &metric,
    const amrex::MultiFab &curv,
    const amrex::MultiFab &rho,
    const CCTK_REAL &dt,
    const int &lev,
    const CCTK_REAL max_energy)
{

    const auto plo0 = this->Geom(0).ProbLoArray();
    const auto phi0 = this->Geom(0).ProbHiArray();

    const auto dx = this->Geom(lev).CellSizeArray();
    const auto plo = this->Geom(lev).ProbLoArray();

    const CCTK_REAL m = this->mass;

    for (GInX::ParticleIterator<StructType> pti(*this, lev); pti.isValid();
         ++pti)
    {
        const int np = pti.numParticles();

        // Get the information relate to the velocities and energy.
        auto &attribs = pti.GetAttributes();
        CCTK_REAL *AMREX_RESTRICT vels_x = attribs[StructType::vx].data();
        CCTK_REAL *AMREX_RESTRICT vels_y = attribs[StructType::vy].data();
        CCTK_REAL *AMREX_RESTRICT vels_z = attribs[StructType::vz].data();
        CCTK_REAL *AMREX_RESTRICT ln_energy = attribs[StructType::ln_E].data();
        CCTK_REAL *AMREX_RESTRICT tau = attribs[StructType::tau].data();                          // RaytracingX: Add optical depth.
        CCTK_REAL *AMREX_RESTRICT index = attribs[StructType::pixel_number].data();               // RaytracingX: Add pixel index.
        CCTK_REAL *AMREX_RESTRICT deletion_reasons = attribs[StructType::deletion_reason].data(); // RaytracingX: Add deletion reason.
        auto *AMREX_RESTRICT particles = &(pti.GetArrayOfStructs()[0]);

        DEFINE_RK4_VARS

        // Get the array of each parameter.
        auto const lapse_array = lapse.array(pti);
        auto const shift_array = shift.array(pti);
        auto const metric_array = metric.array(pti);
        auto const curv_array = curv.array(pti);
        auto const rho_array = rho.array(pti); // RaytracingX: Add optical depth.

        // Needed for GPU
        auto self = this;

        amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int i) noexcept
                           {
      const amrex::GpuArray<CCTK_REAL, 9> U = {
          particles[i].pos(0), particles[i].pos(1), particles[i].pos(2),
          vels_x[i],           vels_y[i],           vels_z[i],
          ln_energy[i], tau[i], 0.0}; //RaytracingX: Add density for optical depth.

      SKIP_DELETED_PARTICLES

      // f1 = rhs(u , t) for the runge kutta 4 step
      auto k =
          self->compute_rhs(iteration, index[i], U, 0.0, lapse_array, shift_array, metric_array,
                            curv_array, rho_array, dt, dx, lev, plo0, phi0, max_energy, m); //RaytracingX: Add density for optical depth.

      particles[i].pos(0) += (1. / 6.) * dt * k[Uidx::x];
      particles[i].pos(1) += (1. / 6.) * dt * k[Uidx::y];
      particles[i].pos(2) += (1. / 6.) * dt * k[Uidx::z];
      vels_x[i]           += (1. / 6.) * dt * k[Uidx::vx];
      vels_y[i]           += (1. / 6.) * dt * k[Uidx::vy];
      vels_z[i]           += (1. / 6.) * dt * k[Uidx::vz];
      ln_energy[i]        += (1. / 6.) * dt * k[Uidx::lnE];
      tau[i]              += (1. / 6.) * dt * k[Uidx::tau];

      LOAD_RK4_VARS
      });
    }
} // RaytracingParticlesContainer::evolve_k1

/**
 *  \brief Evolving using Runge-Kutta 4.
 *
 * We are solving the differential equation
 * \f$\frac{dU}{dt} = f\left(U, \frac{dU}{dx}, t\right)\f$ using:
 *
 *  \f[
 *  U_{n+1} = U_n + \frac{1}{6}\Delta t \left(f_1 + 2f_2 + 2f_3 + f_4\right)
 *  \f]
 *
 *  where:
 *
 *  * \f$f_1 = f(U_n, t),\f$
 *  * \f$f_2 = f\left(U_n + \frac{\Delta t}{2} f_1, t + \frac{\Delta
 * t}{2}\right),\f$
 *  * \f$f_3 = f\left(U_n + \frac{\Delta t}{2} f_2, t + \frac{\Delta
 * t}{2}\right),\f$
 *  * \f$f_4 = f(U_n + \Delta t f_3, t + \Delta t),\f$
 *
 *  While computing we are checking if the particles still in the physical domain.
 *
 *  @see compute_rhs()
 *  @param lapse ADM lapse function.
 *  @param shift ADM shift vector.
 *  @param metric ADM induced metric.
 *  @param curv Extrinsic curvature.
 *  @param rho RaytracingX: gas density
 *  @param dt Timestep.
 *  @param lev Refinement level.
 *  @param max_energy RaytracingX: Maximum energy threshold for event horizon detection.
 */
template <typename StructType>
void RaytracingParticlesContainer<StructType>::evolve_k2(
    const int iteration,
    const amrex::MultiFab &lapse,
    const amrex::MultiFab &shift,
    const amrex::MultiFab &metric,
    const amrex::MultiFab &curv,
    const amrex::MultiFab &rho,
    const CCTK_REAL &dt,
    const int &lev,
    const CCTK_REAL max_energy)
{

    const auto plo0 = this->Geom(0).ProbLoArray();
    const auto phi0 = this->Geom(0).ProbHiArray();

    const auto dx = this->Geom(lev).CellSizeArray();
    const auto plo = this->Geom(lev).ProbLoArray();

    const CCTK_REAL m = this->mass;

    for (GInX::ParticleIterator<StructType> pti(*this, lev); pti.isValid();
         ++pti)
    {
        const int np = pti.numParticles();

        // Get the information relate to the velocities and energy.
        auto &attribs = pti.GetAttributes();
        CCTK_REAL *AMREX_RESTRICT vels_x = attribs[StructType::vx].data();
        CCTK_REAL *AMREX_RESTRICT vels_y = attribs[StructType::vy].data();
        CCTK_REAL *AMREX_RESTRICT vels_z = attribs[StructType::vz].data();
        CCTK_REAL *AMREX_RESTRICT ln_energy = attribs[StructType::ln_E].data();
        CCTK_REAL *AMREX_RESTRICT tau = attribs[StructType::tau].data();                          // RaytracingX: Add optical depth.
        CCTK_REAL *AMREX_RESTRICT index = attribs[StructType::pixel_number].data();               // RaytracingX: Add pixel index.
        CCTK_REAL *AMREX_RESTRICT deletion_reasons = attribs[StructType::deletion_reason].data(); // RaytracingX: Add deletion reason.
        auto *AMREX_RESTRICT particles = &(pti.GetArrayOfStructs()[0]);

        DEFINE_RK4_VARS

        // Get the array of each parameter.
        auto const lapse_array = lapse.array(pti);
        auto const shift_array = shift.array(pti);
        auto const metric_array = metric.array(pti);
        auto const curv_array = curv.array(pti);
        auto const rho_array = rho.array(pti); // RaytracingX: Add optical depth.

        // Needed for GPU
        auto self = this;

        amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int i) noexcept
                           {
      SKIP_DELETED_PARTICLES
      REDEFINE_RK4_ARRAYS
      UNLOAD_RK4_VARS

      amrex::GpuArray<CCTK_REAL, 9> U_tmp = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

      U_tmp[Uidx::x] = U[Uidx::x] + 0.5 * dt * k[Uidx::x];
      U_tmp[Uidx::y] = U[Uidx::y] + 0.5 * dt * k[Uidx::y];
      U_tmp[Uidx::z] = U[Uidx::z] + 0.5 * dt * k[Uidx::z];
      U_tmp[Uidx::vx] = U[Uidx::vx] + 0.5 * dt * k[Uidx::vx];
      U_tmp[Uidx::vy] = U[Uidx::vy] + 0.5 * dt * k[Uidx::vy];
      U_tmp[Uidx::vz] = U[Uidx::vz] + 0.5 * dt * k[Uidx::vz];
      U_tmp[Uidx::lnE] = U[Uidx::lnE] + 0.5 * dt * k[Uidx::lnE];
      U_tmp[Uidx::tau] = U[Uidx::tau] + 0.5 * dt * k[Uidx::tau]; //RaytracingX: Add optical depth.
      U_tmp[Uidx::del_rsn] = k[Uidx::del_rsn];
      U_tmp[Uidx::del_rsn] = check_bounds(U_tmp, plo0, phi0);

      if (U_tmp[Uidx::del_rsn] != 0.0) {
        deletion_reasons[i] = U_tmp[Uidx::del_rsn];
        particles[i].id() = -1;
        return;
      }

      // f2 = rhs(u + 0.5 * dt * f1, t) for the runge kutta 4 step
      k = self->compute_rhs(iteration, index[i], U_tmp, 0.5 * dt, lapse_array, shift_array,
                            metric_array, curv_array, rho_array, dt, dx, lev, plo0, phi0, max_energy, m);

      particles[i].pos(0) += (1. / 3.) * dt * k[Uidx::x];
      particles[i].pos(1) += (1. / 3.) * dt * k[Uidx::y];
      particles[i].pos(2) += (1. / 3.) * dt * k[Uidx::z];
      vels_x[i]           += (1. / 3.) * dt * k[Uidx::vx];
      vels_y[i]           += (1. / 3.) * dt * k[Uidx::vy];
      vels_z[i]           += (1. / 3.) * dt * k[Uidx::vz];
      ln_energy[i]        += (1. / 3.) * dt * k[Uidx::lnE];
      tau[i]              += (1. / 3.) * dt * k[Uidx::tau];
      
      LOAD_RK4_VARS
      });
    }
} // RaytracingParticlesContainer::evolve_k2

/**
 *  \brief Evolving using Runge-Kutta 4.
 *
 * We are solving the differential equation
 * \f$\frac{dU}{dt} = f\left(U, \frac{dU}{dx}, t\right)\f$ using:
 *
 *  \f[
 *  U_{n+1} = U_n + \frac{1}{6}\Delta t \left(f_1 + 2f_2 + 2f_3 + f_4\right)
 *  \f]
 *
 *  where:
 *
 *  * \f$f_1 = f(U_n, t),\f$
 *  * \f$f_2 = f\left(U_n + \frac{\Delta t}{2} f_1, t + \frac{\Delta
 * t}{2}\right),\f$
 *  * \f$f_3 = f\left(U_n + \frac{\Delta t}{2} f_2, t + \frac{\Delta
 * t}{2}\right),\f$
 *  * \f$f_4 = f(U_n + \Delta t f_3, t + \Delta t),\f$
 *
 *  While computing we are checking if the particles still in the physical domain.
 *
 *  @see compute_rhs()
 *  @param lapse ADM lapse function.
 *  @param shift ADM shift vector.
 *  @param metric ADM induced metric.
 *  @param curv Extrinsic curvature.
 *  @param rho RaytracingX: gas density
 *  @param dt Timestep.
 *  @param lev Refinement level.
 *  @param max_energy RaytracingX: Maximum energy threshold for event horizon detection.
 */
template <typename StructType>
void RaytracingParticlesContainer<StructType>::evolve_k3(
    const int iteration,
    const amrex::MultiFab &lapse,
    const amrex::MultiFab &shift,
    const amrex::MultiFab &metric,
    const amrex::MultiFab &curv,
    const amrex::MultiFab &rho,
    const CCTK_REAL &dt,
    const int &lev,
    const CCTK_REAL max_energy)
{

    const auto plo0 = this->Geom(0).ProbLoArray();
    const auto phi0 = this->Geom(0).ProbHiArray();

    const auto dx = this->Geom(lev).CellSizeArray();
    const auto plo = this->Geom(lev).ProbLoArray();

    const CCTK_REAL m = this->mass;

    for (GInX::ParticleIterator<StructType> pti(*this, lev); pti.isValid();
         ++pti)
    {
        const int np = pti.numParticles();

        // Get the information relate to the velocities and energy.
        auto &attribs = pti.GetAttributes();
        CCTK_REAL *AMREX_RESTRICT vels_x = attribs[StructType::vx].data();
        CCTK_REAL *AMREX_RESTRICT vels_y = attribs[StructType::vy].data();
        CCTK_REAL *AMREX_RESTRICT vels_z = attribs[StructType::vz].data();
        CCTK_REAL *AMREX_RESTRICT ln_energy = attribs[StructType::ln_E].data();
        CCTK_REAL *AMREX_RESTRICT tau = attribs[StructType::tau].data();                          // RaytracingX: Add optical depth.
        CCTK_REAL *AMREX_RESTRICT index = attribs[StructType::pixel_number].data();               // RaytracingX: Add pixel index.
        CCTK_REAL *AMREX_RESTRICT deletion_reasons = attribs[StructType::deletion_reason].data(); // RaytracingX: Add deletion reason.
        auto *AMREX_RESTRICT particles = &(pti.GetArrayOfStructs()[0]);

        DEFINE_RK4_VARS

        // Get the array of each parameter.
        auto const lapse_array = lapse.array(pti);
        auto const shift_array = shift.array(pti);
        auto const metric_array = metric.array(pti);
        auto const curv_array = curv.array(pti);
        auto const rho_array = rho.array(pti); // RaytracingX: Add optical depth.

        // Needed for GPU
        auto self = this;

        amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int i) noexcept
                           {
      SKIP_DELETED_PARTICLES
      REDEFINE_RK4_ARRAYS
      UNLOAD_RK4_VARS

      amrex::GpuArray<CCTK_REAL, 9> U_tmp = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

      U_tmp[Uidx::x]   = U[Uidx::x]   + 0.5 * dt * k[Uidx::x];
      U_tmp[Uidx::y]   = U[Uidx::y]   + 0.5 * dt * k[Uidx::y];
      U_tmp[Uidx::z]   = U[Uidx::z]   + 0.5 * dt * k[Uidx::z];
      U_tmp[Uidx::vx]  = U[Uidx::vx]  + 0.5 * dt * k[Uidx::vx];
      U_tmp[Uidx::vy]  = U[Uidx::vy]  + 0.5 * dt * k[Uidx::vy];
      U_tmp[Uidx::vz]  = U[Uidx::vz]  + 0.5 * dt * k[Uidx::vz];
      U_tmp[Uidx::lnE] = U[Uidx::lnE] + 0.5 * dt * k[Uidx::lnE];
      U_tmp[Uidx::tau] = U[Uidx::tau] + 0.5 * dt * k[Uidx::tau]; //RaytracingX: Add optical depth.
      U_tmp[Uidx::del_rsn] = k[Uidx::del_rsn];
      U_tmp[Uidx::del_rsn] = check_bounds(U_tmp, plo0, phi0);

      if (U_tmp[Uidx::del_rsn] != 0.0) {
        deletion_reasons[i] = U_tmp[Uidx::del_rsn];
        particles[i].id() = -1;
        return;
      }
      
      // f3 = rhs(u + 0.5 * dt * f2, t) for the runge kutta 4 step
      k = self->compute_rhs(iteration, index[i], U_tmp, 0.5 * dt, lapse_array, shift_array,
                                metric_array, curv_array, rho_array, dt, dx, lev, plo0, phi0, max_energy, m); //RaytracingX: Add optical depth.

      particles[i].pos(0) += (1. / 3.) * dt * k[Uidx::x];
      particles[i].pos(1) += (1. / 3.) * dt * k[Uidx::y];
      particles[i].pos(2) += (1. / 3.) * dt * k[Uidx::z];
      vels_x[i]           += (1. / 3.) * dt * k[Uidx::vx];
      vels_y[i]           += (1. / 3.) * dt * k[Uidx::vy];
      vels_z[i]           += (1. / 3.) * dt * k[Uidx::vz];
      ln_energy[i]        += (1. / 3.) * dt * k[Uidx::lnE];
      tau[i]              += (1. / 3.) * dt * k[Uidx::tau];

      LOAD_RK4_VARS
      });
    }
} // RaytracingParticlesContainer::evolve_k3

/**
 *  \brief Evolving using Runge-Kutta 4.
 *
 * We are solving the differential equation
 * \f$\frac{dU}{dt} = f\left(U, \frac{dU}{dx}, t\right)\f$ using:
 *
 *  \f[
 *  U_{n+1} = U_n + \frac{1}{6}\Delta t \left(f_1 + 2f_2 + 2f_3 + f_4\right)
 *  \f]
 *
 *  where:
 *
 *  * \f$f_1 = f(U_n, t),\f$
 *  * \f$f_2 = f\left(U_n + \frac{\Delta t}{2} f_1, t + \frac{\Delta
 * t}{2}\right),\f$
 *  * \f$f_3 = f\left(U_n + \frac{\Delta t}{2} f_2, t + \frac{\Delta
 * t}{2}\right),\f$
 *  * \f$f_4 = f(U_n + \Delta t f_3, t + \Delta t),\f$
 *
 *  While computing we are checking if the particles still in the physical domain.
 *
 *  @see compute_rhs()
 *  @param lapse ADM lapse function.
 *  @param shift ADM shift vector.
 *  @param metric ADM induced metric.
 *  @param curv Extrinsic curvature.
 *  @param rho RaytracingX: gas density
 *  @param dt Timestep.
 *  @param lev Refinement level.
 *  @param max_energy RaytracingX: Maximum energy threshold for event horizon detection.
 */
template <typename StructType>
void RaytracingParticlesContainer<StructType>::evolve_k4(
    const int iteration,
    const amrex::MultiFab &lapse,
    const amrex::MultiFab &shift,
    const amrex::MultiFab &metric,
    const amrex::MultiFab &curv,
    const amrex::MultiFab &rho,
    const CCTK_REAL &dt,
    const int &lev,
    const CCTK_REAL max_energy)
{

    const auto plo0 = this->Geom(0).ProbLoArray();
    const auto phi0 = this->Geom(0).ProbHiArray();

    const auto dx = this->Geom(lev).CellSizeArray();
    const auto plo = this->Geom(lev).ProbLoArray();

    const CCTK_REAL m = this->mass;

    for (GInX::ParticleIterator<StructType> pti(*this, lev); pti.isValid();
         ++pti)
    {
        const int np = pti.numParticles();

        // Get the information relate to the velocities and energy.
        auto &attribs = pti.GetAttributes();
        CCTK_REAL *AMREX_RESTRICT vels_x = attribs[StructType::vx].data();
        CCTK_REAL *AMREX_RESTRICT vels_y = attribs[StructType::vy].data();
        CCTK_REAL *AMREX_RESTRICT vels_z = attribs[StructType::vz].data();
        CCTK_REAL *AMREX_RESTRICT ln_energy = attribs[StructType::ln_E].data();
        CCTK_REAL *AMREX_RESTRICT tau = attribs[StructType::tau].data();                          // RaytracingX: Add optical depth.
        CCTK_REAL *AMREX_RESTRICT index = attribs[StructType::pixel_number].data();               // RaytracingX: Add pixel index.
        CCTK_REAL *AMREX_RESTRICT deletion_reasons = attribs[StructType::deletion_reason].data(); // RaytracingX: Add deletion reason.
        auto *AMREX_RESTRICT particles = &(pti.GetArrayOfStructs()[0]);

        DEFINE_RK4_VARS

        // Get the array of each parameter.
        auto const lapse_array = lapse.array(pti);
        auto const shift_array = shift.array(pti);
        auto const metric_array = metric.array(pti);
        auto const curv_array = curv.array(pti);
        auto const rho_array = rho.array(pti); // RaytracingX: Add optical depth.

        // Needed for GPU
        auto self = this;

        amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int i) noexcept
                           {
      SKIP_DELETED_PARTICLES
      REDEFINE_RK4_ARRAYS
      UNLOAD_RK4_VARS

      amrex::GpuArray<CCTK_REAL, 9> U_tmp = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

      U_tmp[Uidx::x]   = U[Uidx::x]   + dt * k[Uidx::x];
      U_tmp[Uidx::y]   = U[Uidx::y]   + dt * k[Uidx::y];
      U_tmp[Uidx::z]   = U[Uidx::z]   + dt * k[Uidx::z];
      U_tmp[Uidx::vx]  = U[Uidx::vx]  + dt * k[Uidx::vx];
      U_tmp[Uidx::vy]  = U[Uidx::vy]  + dt * k[Uidx::vy];
      U_tmp[Uidx::vz]  = U[Uidx::vz]  + dt * k[Uidx::vz];
      U_tmp[Uidx::lnE] = U[Uidx::lnE] + dt * k[Uidx::lnE];
      U_tmp[Uidx::tau] = U[Uidx::tau] + dt * k[Uidx::tau]; //RaytracingX: Add optical depth.
      U_tmp[Uidx::del_rsn] = k[Uidx:del_rsn];
      U_tmp[Uidx::del_rsn] = check_bounds(U_tmp, plo0, phi0);

      if (U_tmp[Uidx::del_rsn] != 0.0) {
        deletion_reasons[i] = U_tmp[Udix::del_rsn];
        particles[i].id() = -1;
        return;
      }

      // f4 = rhs(u + dt * f3, t) for the runge kutta 4 step
      k = self->compute_rhs(iteration, index[i], U_tmp, dt, lapse_array, shift_array,
                                 metric_array, curv_array, rho_array, dt, dx, lev, plo0, phi0, max_energy, m); //RaytracingX: Add optical depth.

      // Update particles with the f3 and f4 from RK4
      particles[i].pos(0) += (1. / 6.) * dt * k[Uidx::x];
      particles[i].pos(1) += (1. / 6.) * dt * k[Uidx::y];
      particles[i].pos(2) += (1. / 6.) * dt * k[Uidx::z];
      vels_x[i]           += (1. / 6.) * dt * k[Uidx::vx];
      vels_y[i]           += (1. / 6.) * dt * k[Uidx::vy];
      vels_z[i]           += (1. / 6.) * dt * k[Uidx::vz];
      ln_energy[i]        += (1. / 6.) * dt * k[Uidx::lnE];
      tau[i]              += (1. / 6.) * dt * k[Uidx::tau];
      
      U_tmp[Uidx::x] = particles[i].pos(0);
      U_tmp[Uidx::y] = particles[i].pos(1);
      U_tmp[Uidx::z] = particles[i].pos(2);
      U_tmp[Uidx::vx] = vels_x[i];
      U_tmp[Uidx::vy] = vels_y[i];
      U_tmp[Uidx::vz] = vels_z[i];
      U_tmp[Uidx::lnE] = ln_energy[i];
      U_tmp[Uidx::tau] = tau[i];
      U_tmp[Uidx::del_rsn] = k[Uidx::del_rsn];
      U_tmp[Uidx::del_rsn] = check_bounds(U_tmp, plo0, phi0);

      if (U_tmp[Udix::del_rsn] != 0.0) {
        deletion_reasons[i] = U_tmp[Udix::del_rsn];
        particles[i].id() = -1;
        return;
      }
      });
    }
} // RaytracingParticlesContainer::evolve_k4

template <typename StructType>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE CCTK_ATTRIBUTE_ALWAYS_INLINE
CCTK_REAL RaytracingParticlesContainer<StructType>::check_validity(
    const amrex::GpuArray<CCTK_REAL, 9> rhs,
    const amrex::GpuArray<CCTK_REAL, 9> u,
    const CCTK_REAL lapse,
    const CCTK_REAL max_energy,
    const int index)
{
    CCTK_REAL deletion_reason = u[Uidx::del_rsn];

    if (abs(std::exp(u[Uidx::lnE]) / lapse) > max_energy) {
        deletion_reason = DelReason::HORIZON;
    }

    if (!(std::isfinite(rhs[Uidx::x]) &&
          std::isfinite(rhs[Uidx::y]) &&
          std::isfinite(rhs[Uidx::z]) &&
          std::isfinite(rhs[Uidx::vx]) &&
          std::isfinite(rhs[Uidx::vy]) &&
          std::isfinite(rhs[Uidx::vz]) &&
          std::isfinite(rhs[Uidx::lnE]) &&
          std::isfinite(rhs[Uidx::tau]))) {
        CCTK_VWARN(CCTK_WARN_ALERT, "RHS for particle %d invalid outside of horizon, du/dt=(x, y, z, v_x, v_y, v_z, ln_E, tau)=(%f, %f, %f, %f, %f, %f, %f, %f), u=(%f, %f, %f, %f, %f, %f, %f, %f)",
            index, 
            rhs[Uidx::x], rhs[Uidx::y], rhs[Uidx::z], rhs[Uidx::vx], rhs[Uidx::vy], rhs[Uidx::vz], rhs[Uidx::lnE], rhs[Uidx::tau],
            u[Uidx::x], u[Uidx::y], u[Uidx::z], u[Uidx::vx], u[Uidx::vy], u[Uidx::vz], u[Uidx::lnE], u[Uidx::tau]
        );
        deletion_reason = DelReason::NONFINITE;
    }

    return deletion_reason;
}

template <typename StructType>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE CCTK_ATTRIBUTE_ALWAYS_INLINE
CCTK_REAL RaytracingParticlesContainer<StructType>::check_bounds(
    const amrex::GpuArray<CCTK_REAL, 9> u,
    const amrex::GpuArray<double, 3> &plo,
    const amrex::GpuArray<double, 3> &phi)
{
    if (u[Uidx::x] > phi[0]) {
        return DelReason::XHI;
    }
    if (u[Uidx::x] < plo[0]) {
        return DelReason::XLO;
    }
    if (u[Uidx::y] > phi[1]) {
        return DelReason::YHI;
    }
    if (u[Uidx::y] < plo[1]) {
        return DelReason::YLO;
    }
    if (u[Uidx::z] > phi[2]) {
        return DelReason::ZHI;
    }
    if (u[Uidx::z] < plo[2]) {
        return DelReason::ZLO;
    }
    if (u[Uidx::tau] >= 1.0) {
        return DelReason::PHOTOSPHERE;
    }
    return u[Uidx::del_rsn];
}