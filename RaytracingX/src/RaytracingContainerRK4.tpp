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
    const amrex::GpuArray<CCTK_REAL, 8> &u,
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
    const amrex::GpuArray<double, 3> &phi
    const CCTK_REAL max_energy)
{
    // RaytracingX: Add space for optical depth variable.
    amrex::GpuArray<CCTK_REAL, 9> rhs = {0., 0., 0., 0., 0., 0., 0., 0., 0.};

    const long int i0 = get_interpolation_center(u[0], plo[0], phi[0], dx[0]);
    const long int j0 = get_interpolation_center(u[1], plo[1], phi[1], dx[1]);
    const long int k0 = get_interpolation_center(u[2], plo[2], phi[2], dx[2]);

    ASSERT_FINITE1(u, 3)

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
    amrex::GpuArray<CCTK_REAL, 3> d_rho_x;
    GInX::d_interpolate_array<5>(rho_x, d_rho_x, rho, i0, j0, k0, u[0], u[1],
                                 u[2], dx, plo);

    ASSERT_FINITE(lapse_x)
    ASSERT_FINITE1(d_lapse_x, 3)
    ASSERT_FINITE1(shift_x, 3)
    ASSERT_FINITE2(d_shift_x, 3, 3)
    ASSERT_FINITE1(gamma_x, 6)
    ASSERT_FINITE2(d_gamma_x, 6, 3)
    ASSERT_FINITE1(curv_x, 6)

    // Compute the inverse of the metric.
    const CCTK_REAL inv_det_gamma =
        1.0 / (gamma_x[0] * gamma_x[3] * gamma_x[5] +
               2. * gamma_x[1] * gamma_x[2] * gamma_x[4] -
               gamma_x[2] * gamma_x[2] * gamma_x[3] -
               gamma_x[4] * gamma_x[4] * gamma_x[0] -
               gamma_x[1] * gamma_x[1] * gamma_x[5]);

    if (!std::isfinite(inv_det_gamma))
    {
        fprintf(stderr, "the following spatial metric at (%f, %f, %f) has an invalid determinant:\n%f %f %f\n %f %f %f\n%f %f %f\n", u[0], u[1], u[2], gamma_x[0], gamma_x[1], gamma_x[2], gamma_x[1], gamma_x[3], gamma_x[4], gamma_x[2], gamma_x[4], gamma_x[5]);
    }
    ASSERT_FINITE(inv_det_gamma)

    const amrex::GpuArray<CCTK_REAL, 6> gamma_inv_x = {
        (gamma_x[3] * gamma_x[5] - gamma_x[4] * gamma_x[4]) * inv_det_gamma,
        (gamma_x[4] * gamma_x[2] - gamma_x[1] * gamma_x[5]) * inv_det_gamma,
        (gamma_x[1] * gamma_x[4] - gamma_x[2] * gamma_x[3]) * inv_det_gamma,
        (gamma_x[0] * gamma_x[5] - gamma_x[2] * gamma_x[2]) * inv_det_gamma,
        (gamma_x[2] * gamma_x[1] - gamma_x[0] * gamma_x[4]) * inv_det_gamma,
        (gamma_x[0] * gamma_x[3] - gamma_x[1] * gamma_x[1]) * inv_det_gamma};

    const amrex::GpuArray<CCTK_REAL, 3> V_down = {u[3], u[4], u[5]};

    // Compute the upper index velocity terms.
    const amrex::GpuArray<CCTK_REAL, 3> V_up = {
        gamma_inv_x[0] * u[3] + gamma_inv_x[1] * u[4] + gamma_inv_x[2] * u[5],
        gamma_inv_x[1] * u[3] + gamma_inv_x[3] * u[4] + gamma_inv_x[4] * u[5],
        gamma_inv_x[2] * u[3] + gamma_inv_x[4] * u[4] + gamma_inv_x[5] * u[5]};

    // Compute the rhs for position
    rhs[0] = lapse_x * V_up[0] - shift_x[0];
    rhs[1] = lapse_x * V_up[1] - shift_x[1];
    rhs[2] = lapse_x * V_up[2] - shift_x[2];

    // Compute the rhs for velocity
    for (int i = 0; i < 3; i++)
    {
        rhs[3 + i] =
            -d_lapse_x[i] +
            (VecVecMul(d_lapse_x, V_up) -
             lapse_x * VecVecMul(SMatVecMul(curv_x, V_up), V_up)) *
                V_down[i] +
            0.5 * lapse_x * VecVecMul(SMatVecMul(d_gamma_x[i], V_up), V_up) +
            VecVecMul(V_down, d_shift_x[i]);
    }

    // Compute the rhs for energy
    rhs[3 + StructType::ln_alphaE] =
        lapse_x * VecVecMul(SMatVecMul(curv_x, V_up), V_up) -
        VecVecMul(V_up, d_lapse_x);

    // RaytracingX: Add evolution for optical depth calculation.
    //  Compute the rhs for optical depth
    const CCTK_REAL ds = dx[0] * dx[0] * gamma_inv_x[0] +
                         dx[1] * dx[1] * gamma_inv_x[3] +
                         dx[2] * dx[2] * gamma_inv_x[5] +
                         2.0 * dx[0] * dx[1] * gamma_inv_x[1] +
                         2.0 * dx[0] * dx[2] * gamma_inv_x[2] +
                         2.0 * dx[1] * dx[2] * gamma_inv_x[4];
    rhs[3 + StructType::tau] = (0.4 * cgs2cactusOpacity) * (rho_x * cgs2cactusDensity) * (ds / dt);

    // Normalizing the velocity.
    const CCTK_REAL v_squared = rhs[0] * rhs[0] * gamma_inv_x[0] +
                                rhs[1] * rhs[1] * gamma_inv_x[3] +
                                rhs[2] * rhs[2] * gamma_inv_x[5] +
                                2.0 * rhs[0] * rhs[1] * gamma_inv_x[1] +
                                2.0 * rhs[0] * rhs[2] * gamma_inv_x[2] +
                                2.0 * rhs[1] * rhs[2] * gamma_inv_x[4];

    ASSERT_FINITE(v_squared)
    assert(v_squared >= 0);

    ASSERT_FINITE(u[3 + StructType::ln_alphaE])
    ASSERT_FINITE(std::exp(u[3 + StructType::ln_alphaE]))

    const CCTK_REAL v = std::sqrt(v_squared);

    ASSERT_FINITE(v)
    ASSERT_FINITE(this->mass)

    if (iteration >= 1170 && iteration <= 1200 && amrex::ParallelDescriptor::MyProc() == 24)
    {
        fprintf(stderr, "it: %d, pidx: %d, alphaE: %f, dx/dt=(%f, %f, %f), vup=(%f, %f, %f), vvec=(%f, %f, %f)->%f, x=(%f, %f, %f), dx=(%f, %f, %f)\nmet: %f, %f, %f, %f\n%f, %f, %f, %f\n%f, %f, %f, %f\n%f, %f, %f, %f\n",
                index,
                iteration,
                std::exp(u[3 + StructType::ln_alphaE]),
                UNPACKV(rhs),
                UNPACKV(V_up),
                UNPACKV(V_down), v,
                UNPACKV(u),
                UNPACKV(dx),
                UNPACK4M_COMP(lapse_x, shift_x, gamma_x));
    }

    const CCTK_REAL alpha_over_v = std::sqrt(1. - this->mass * this->mass / (2 * std::exp(u[3 + StructType::ln_alphaE]))) / v;

    ASSERT_FINITE(alpha_over_v)

    rhs[0] *= alpha_over_v;
    rhs[1] *= alpha_over_v;
    rhs[2] *= alpha_over_v;

    rhs[8] = check_validity(rhs, u, lapse_x, max_energy, index);

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
void RaytracingParticlesContainer<StructType>::evolve(
    const int iteration,
    const amrex::MultiFab &lapse,
    const amrex::MultiFab &shift,
    const amrex::MultiFab &metric,
    const amrex::MultiFab &curv,
    const amrex::MultiFab &rho,
    const CCTK_REAL &dt,
    const int &lev
    const CCTK_REAL max_energy)
{

    const auto plo0 = this->Geom(0).ProbLoArray();
    const auto phi0 = this->Geom(0).ProbHiArray();

    const auto dx = this->Geom(lev).CellSizeArray();
    const auto plo = this->Geom(lev).ProbLoArray();

    for (GInX::ParticleIterator<StructType> pti(*this, lev); pti.isValid();
         ++pti)
    {
        const int np = pti.numParticles();

        // Get the information relate to the velocities and energy.
        auto &attribs = pti.GetAttributes();
        CCTK_REAL *AMREX_RESTRICT vels_x = attribs[StructType::vx].data();
        CCTK_REAL *AMREX_RESTRICT vels_y = attribs[StructType::vy].data();
        CCTK_REAL *AMREX_RESTRICT vels_z = attribs[StructType::vz].data();
        CCTK_REAL *AMREX_RESTRICT ln_alphaenergy = attribs[StructType::ln_alphaE].data();
        CCTK_REAL *AMREX_RESTRICT tau = attribs[StructType::tau].data();                          // RaytracingX: Add optical depth.
        CCTK_REAL *AMREX_RESTRICT index = attribs[StructType::pixel_number].data();               // RaytracingX: Add pixel index.
        CCTK_REAL *AMREX_RESTRICT deletion_reasons = attribs[StructType::deletion_reason].data(); // RaytracingX: Add deletion reason.
        auto *AMREX_RESTRICT particles = &(pti.GetArrayOfStructs()[0]);

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
          ln_alphaenergy[i], tau[i], 0.0}; //RaytracingX: Add density for optical depth.

      SKIP_DELETED_PARTICLES

      //RaytracingX: Add optical depth.
      amrex::GpuArray<CCTK_REAL, 9> U_tmp = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

      // f1 = rhs(u , t) for the runge kutta 4 step
      auto k_odd =
          self->compute_rhs(iteration, index[i], U, 0.0, lapse_array, shift_array, metric_array,
                            curv_array, rho_array, dt, dx, lev, plo0, phi0); //RaytracingX: Add density for optical depth.

      U_tmp[0] = U[0] + 0.5 * dt * k_odd[0];
      U_tmp[1] = U[1] + 0.5 * dt * k_odd[1];
      U_tmp[2] = U[2] + 0.5 * dt * k_odd[2];
      U_tmp[3] = U[3] + 0.5 * dt * k_odd[3];
      U_tmp[4] = U[4] + 0.5 * dt * k_odd[4];
      U_tmp[5] = U[5] + 0.5 * dt * k_odd[5];
      U_tmp[6] = U[6] + 0.5 * dt * k_odd[6];
      U_tmp[7] = U[7] + 0.5 * dt * k_odd[7]; //RaytracingX: Add optical depth.
      U_tmp[8] = k_odd[8];
      U_tmp[8] = check_bounds(U_tmp, plo, phi);

      if (U_tmp[8] != 0.0) {
        deletion_reasons[i] = U_tmp[8];
        particles[i].id() = -1;
        return;
      }

      // f2 = rhs(u + 0.5 * dt * f1, t) for the runge kutta 4 step
      auto k_even =
          self->compute_rhs(iteration, index[i], U_tmp, 0.5 * dt, lapse_array, shift_array,
                            metric_array, curv_array, rho_array, dt, dx, lev, plo0, phi0);

      // Update particles with the f1 and f2 from RK4
      U_tmp[0] = U[0] + 0.5 * dt * k_even[0];
      U_tmp[1] = U[1] + 0.5 * dt * k_even[1];
      U_tmp[2] = U[2] + 0.5 * dt * k_even[2];
      U_tmp[3] = U[3] + 0.5 * dt * k_even[3];
      U_tmp[4] = U[4] + 0.5 * dt * k_even[4];
      U_tmp[5] = U[5] + 0.5 * dt * k_even[5];
      U_tmp[6] = U[6] + 0.5 * dt * k_even[6];
      U_tmp[7] = U[7] + 0.5 * dt * k_even[7]; //RaytracingX: Add optical depth.
      U_tmp[8] = k_even[8];
      U_tmp[8] = check_bounds(U_tmp, plo, phi);

      if (U_tmp[8] != 0.0) {
        deletion_reasons[i] = U_tmp[8];
        particles[i].id() = -1;
        return;
      }

      particles[i].pos(0) += (1. / 6.) * dt * (k_odd[0] + 2. * k_even[0]);
      particles[i].pos(1) += (1. / 6.) * dt * (k_odd[1] + 2. * k_even[1]);
      particles[i].pos(2) += (1. / 6.) * dt * (k_odd[2] + 2. * k_even[2]);
      vels_x[i] += (1. / 6.) * dt * (k_odd[3] + 2. * k_even[3]);
      vels_y[i] += (1. / 6.) * dt * (k_odd[4] + 2. * k_even[4]);
      vels_z[i] += (1. / 6.) * dt * (k_odd[5] + 2. * k_even[5]);
      ln_alphaenergy[i] += (1. / 6.) * dt * (k_odd[6] + 2. * k_even[6]);
      tau[i] += (1. / 6.) * dt * (k_odd[7] + 2. * k_even[7]); //RaytracingX: Add optical depth.
      
      // f3 = rhs(u + 0.5 * dt * f2, t) for the runge kutta 4 step
      k_odd = self->compute_rhs(iteration, index[i], U_tmp, 0.5 * dt, lapse_array, shift_array,
                                metric_array, curv_array, rho_array, dt, dx, lev, plo0, phi0); //RaytracingX: Add optical depth.

      U_tmp[0] = U[0] + dt * k_odd[0];
      U_tmp[1] = U[1] + dt * k_odd[1];
      U_tmp[2] = U[2] + dt * k_odd[2];
      U_tmp[3] = U[3] + dt * k_odd[3];
      U_tmp[4] = U[4] + dt * k_odd[4];
      U_tmp[5] = U[5] + dt * k_odd[5];
      U_tmp[6] = U[6] + dt * k_odd[6];
      U_tmp[7] = U[7] + dt * k_odd[7]; //RaytracingX: Add optical depth.
      U_tmp[8] = k_odd[8];
      U_tmp[8] = check_bounds(U_tmp, plo, phi);

      if (U_tmp[8] != 0.0) {
        deletion_reasons[i] = U_tmp[8];
        particles[i].id() = -1;
        return;
      }

      // f4 = rhs(u + dt * f3, t) for the runge kutta 4 step
      k_even = self->compute_rhs(iteration, index[i], U_tmp, dt, lapse_array, shift_array,
                                 metric_array, curv_array, rho_array, dt, dx, lev, plo0, phi0); //RaytracingX: Add optical depth.

      // Update particles with the f3 and f4 from RK4
      particles[i].pos(0) += (1. / 6.) * dt * (2. * k_odd[0] + k_even[0]);
      particles[i].pos(1) += (1. / 6.) * dt * (2. * k_odd[1] + k_even[1]);
      particles[i].pos(2) += (1. / 6.) * dt * (2. * k_odd[2] + k_even[2]);
      vels_x[i] += (1. / 6.) * dt * (2. * k_odd[3] + k_even[3]);
      vels_y[i] += (1. / 6.) * dt * (2. * k_odd[4] + k_even[4]);
      vels_z[i] += (1. / 6.) * dt * (2. * k_odd[5] + k_even[5]);
      ln_alphaenergy[i] += (1. / 6.) * dt * (2. * k_odd[6] + k_even[6]);
      tau[i] += (1. / 6.) * dt * (2. * k_odd[7] + k_even[7]); //RaytracingX: Add optical depth.

      U_tmp[0] = particles[i].pos(0);
      U_tmp[1] = particles[i].pos(1);
      U_tmp[2] = particles[i].pos(2);
      U_tmp[3] = vels_x[i];
      U_tmp[4] = vels_y[i];
      U_tmp[5] = vels_z[i];
      U_tmp[6] = ln_alphaenergy[i];
      U_tmp[7] = tau[i];
      U_tmp[8] = k_even[8];
      U_tmp[8] = check_bounds(U_tmp, plo, phi);

      if (U_tmp[8] != 0.0) {
        deletion_reasons[i] = U_tmp[8];
        particles[i].id() = -1;
        return;
      }
      });
    }
} // RaytracingParticlesContainer::evolve

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
    const CCTK_REAL max_energy,
    const ptclRK4data ptclRK4data)
{

    const auto plo0 = this->Geom(0).ProbLoArray();
    const auto phi0 = this->Geom(0).ProbHiArray();

    const auto dx = this->Geom(lev).CellSizeArray();
    const auto plo = this->Geom(lev).ProbLoArray();

    for (GInX::ParticleIterator<StructType> pti(*this, lev); pti.isValid();
         ++pti)
    {
        const int np = pti.numParticles();

        // Get the information relate to the velocities and energy.
        auto &attribs = pti.GetAttributes();
        CCTK_REAL *AMREX_RESTRICT vels_x = attribs[StructType::vx].data();
        CCTK_REAL *AMREX_RESTRICT vels_y = attribs[StructType::vy].data();
        CCTK_REAL *AMREX_RESTRICT vels_z = attribs[StructType::vz].data();
        CCTK_REAL *AMREX_RESTRICT ln_alphaenergy = attribs[StructType::ln_alphaE].data();
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
          ln_alphaenergy[i], tau[i], 0.0}; //RaytracingX: Add density for optical depth.

      SKIP_DELETED_PARTICLES

      //RaytracingX: Add optical depth.
      amrex::GpuArray<CCTK_REAL, 8> U_tmp = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      amrex::GpuArray<CCTK_REAL, 8> k_even = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

      // f1 = rhs(u , t) for the runge kutta 4 step
      auto k_odd =
          self->compute_rhs(iteration, index[i], U, 0.0, lapse_array, shift_array, metric_array,
                            curv_array, rho_array, dt, dx, lev, plo0, phi0); //RaytracingX: Add density for optical depth.

      U_tmp[0] = U[0] + 0.5 * dt * k_odd[0];
      U_tmp[1] = U[1] + 0.5 * dt * k_odd[1];
      U_tmp[2] = U[2] + 0.5 * dt * k_odd[2];
      U_tmp[3] = U[3] + 0.5 * dt * k_odd[3];
      U_tmp[4] = U[4] + 0.5 * dt * k_odd[4];
      U_tmp[5] = U[5] + 0.5 * dt * k_odd[5];
      U_tmp[6] = U[6] + 0.5 * dt * k_odd[6];
      U_tmp[7] = U[7] + 0.5 * dt * k_odd[7]; //RaytracingX: Add optical depth.
      U_tmp[8] = k_odd[8];
      U_tmp[8] = check_bounds(U_tmp, plo, phi);

      if (U_tmp[8] != 0.0) {
        deletion_reasons[i] = U_tmp[8];
        particles[i].id() = -1;
        return;
      }

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
    const CCTK_REAL max_energy,
    const ptclRK4data ptclRK4data)
{

    const auto plo0 = this->Geom(0).ProbLoArray();
    const auto phi0 = this->Geom(0).ProbHiArray();

    const auto dx = this->Geom(lev).CellSizeArray();
    const auto plo = this->Geom(lev).ProbLoArray();

    for (GInX::ParticleIterator<StructType> pti(*this, lev); pti.isValid();
         ++pti)
    {
        const int np = pti.numParticles();

        // Get the information relate to the velocities and energy.
        auto &attribs = pti.GetAttributes();
        CCTK_REAL *AMREX_RESTRICT vels_x = attribs[StructType::vx].data();
        CCTK_REAL *AMREX_RESTRICT vels_y = attribs[StructType::vy].data();
        CCTK_REAL *AMREX_RESTRICT vels_z = attribs[StructType::vz].data();
        CCTK_REAL *AMREX_RESTRICT ln_alphaenergy = attribs[StructType::ln_alphaE].data();
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

      // f2 = rhs(u + 0.5 * dt * f1, t) for the runge kutta 4 step
      k_even =
          self->compute_rhs(iteration, index[i], U_tmp, 0.5 * dt, lapse_array, shift_array,
                            metric_array, curv_array, rho_array, dt, dx, lev, plo0, phi0);
      CHECK_VELOCITY(i, k_even[0], k_even[1], k_even[2])

      // Update particles with the f1 and f2 from RK4
      U_tmp[0] = U[0] + 0.5 * dt * k_even[0];
      U_tmp[1] = U[1] + 0.5 * dt * k_even[1];
      U_tmp[2] = U[2] + 0.5 * dt * k_even[2];
      U_tmp[3] = U[3] + 0.5 * dt * k_even[3];
      U_tmp[4] = U[4] + 0.5 * dt * k_even[4];
      U_tmp[5] = U[5] + 0.5 * dt * k_even[5];
      U_tmp[6] = U[6] + 0.5 * dt * k_even[6];
      U_tmp[7] = U[7] + 0.5 * dt * k_even[7]; //RaytracingX: Add optical depth.
      U_tmp[8] = k_even[8];
      U_tmp[8] = check_bounds(U_tmp, plo, phi);

      if (U_tmp[8] != 0.0) {
        deletion_reasons[i] = U_tmp[8];
        particles[i].id() = -1;
        return;
      }

      particles[i].pos(0) += (1. / 6.) * dt * (k_odd[0] + 2. * k_even[0]);
      particles[i].pos(1) += (1. / 6.) * dt * (k_odd[1] + 2. * k_even[1]);
      particles[i].pos(2) += (1. / 6.) * dt * (k_odd[2] + 2. * k_even[2]);
      vels_x[i] += (1. / 6.) * dt * (k_odd[3] + 2. * k_even[3]);
      vels_y[i] += (1. / 6.) * dt * (k_odd[4] + 2. * k_even[4]);
      vels_z[i] += (1. / 6.) * dt * (k_odd[5] + 2. * k_even[5]);
      ln_alphaenergy[i] += (1. / 6.) * dt * (k_odd[6] + 2. * k_even[6]);
      tau[i] += (1. / 6.) * dt * (k_odd[7] + 2. * k_even[7]); //RaytracingX: Add optical depth.
      
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
    const CCTK_REAL max_energy,
    const ptclRK4data ptclRK4data)
{

    const auto plo0 = this->Geom(0).ProbLoArray();
    const auto phi0 = this->Geom(0).ProbHiArray();

    const auto dx = this->Geom(lev).CellSizeArray();
    const auto plo = this->Geom(lev).ProbLoArray();

    for (GInX::ParticleIterator<StructType> pti(*this, lev); pti.isValid();
         ++pti)
    {
        const int np = pti.numParticles();

        // Get the information relate to the velocities and energy.
        auto &attribs = pti.GetAttributes();
        CCTK_REAL *AMREX_RESTRICT vels_x = attribs[StructType::vx].data();
        CCTK_REAL *AMREX_RESTRICT vels_y = attribs[StructType::vy].data();
        CCTK_REAL *AMREX_RESTRICT vels_z = attribs[StructType::vz].data();
        CCTK_REAL *AMREX_RESTRICT ln_alphaenergy = attribs[StructType::ln_alphaE].data();
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
      
      // f3 = rhs(u + 0.5 * dt * f2, t) for the runge kutta 4 step
      k_odd = self->compute_rhs(iteration, index[i], U_tmp, 0.5 * dt, lapse_array, shift_array,
                                metric_array, curv_array, rho_array, dt, dx, lev, plo0, phi0); //RaytracingX: Add optical depth.
      CHECK_VELOCITY(i, k_odd[0], k_odd[1], k_odd[2])

      U_tmp[0] = U[0] + dt * k_odd[0];
      U_tmp[1] = U[1] + dt * k_odd[1];
      U_tmp[2] = U[2] + dt * k_odd[2];
      U_tmp[3] = U[3] + dt * k_odd[3];
      U_tmp[4] = U[4] + dt * k_odd[4];
      U_tmp[5] = U[5] + dt * k_odd[5];
      U_tmp[6] = U[6] + dt * k_odd[6];
      U_tmp[7] = U[7] + dt * k_odd[7]; //RaytracingX: Add optical depth.
      U_tmp[8] = k_odd[8];
      U_tmp[8] = check_bounds(U_tmp, plo, phi);

      if (U_tmp[8] != 0.0) {
        deletion_reasons[i] = U_tmp[8];
        particles[i].id() = -1;
        return;
      }

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
    const CCTK_REAL max_energy,
    const ptclRK4data ptclRK4data)
{

    const auto plo0 = this->Geom(0).ProbLoArray();
    const auto phi0 = this->Geom(0).ProbHiArray();

    const auto dx = this->Geom(lev).CellSizeArray();
    const auto plo = this->Geom(lev).ProbLoArray();

    for (GInX::ParticleIterator<StructType> pti(*this, lev); pti.isValid();
         ++pti)
    {
        const int np = pti.numParticles();

        // Get the information relate to the velocities and energy.
        auto &attribs = pti.GetAttributes();
        CCTK_REAL *AMREX_RESTRICT vels_x = attribs[StructType::vx].data();
        CCTK_REAL *AMREX_RESTRICT vels_y = attribs[StructType::vy].data();
        CCTK_REAL *AMREX_RESTRICT vels_z = attribs[StructType::vz].data();
        CCTK_REAL *AMREX_RESTRICT ln_alphaenergy = attribs[StructType::ln_alphaE].data();
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

      // f4 = rhs(u + dt * f3, t) for the runge kutta 4 step
      k_even = self->compute_rhs(iteration, index[i], U_tmp, dt, lapse_array, shift_array,
                                 metric_array, curv_array, rho_array, dt, dx, lev, plo0, phi0); //RaytracingX: Add optical depth.

      // Update particles with the f3 and f4 from RK4
      particles[i].pos(0) += (1. / 6.) * dt * (2. * k_odd[0] + k_even[0]);
      particles[i].pos(1) += (1. / 6.) * dt * (2. * k_odd[1] + k_even[1]);
      particles[i].pos(2) += (1. / 6.) * dt * (2. * k_odd[2] + k_even[2]);
      vels_x[i] += (1. / 6.) * dt * (2. * k_odd[3] + k_even[3]);
      vels_y[i] += (1. / 6.) * dt * (2. * k_odd[4] + k_even[4]);
      vels_z[i] += (1. / 6.) * dt * (2. * k_odd[5] + k_even[5]);
      ln_alphaenergy[i] += (1. / 6.) * dt * (2. * k_odd[6] + k_even[6]);
      tau[i] += (1. / 6.) * dt * (2. * k_odd[7] + k_even[7]); //RaytracingX: Add optical depth.
      
      U_tmp[0] = particles[i].pos(0);
      U_tmp[1] = particles[i].pos(1);
      U_tmp[2] = particles[i].pos(2);
      U_tmp[3] = vels_x[i];
      U_tmp[4] = vels_y[i];
      U_tmp[5] = vels_z[i];
      U_tmp[6] = ln_alphaenergy[i];
      U_tmp[7] = tau[i];
      U_tmp[8] = k_even[8];
      U_tmp[8] = check_bounds(U_tmp, plo, phi);

      if (U_tmp[8] != 0.0) {
        deletion_reasons[i] = U_tmp[8];
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
    CCTK_REAL deletion_reason = u[8];

    if (abs(std::exp(u[6]) / lapse) > max_energy) {
        deletion_reason = -7;
    }

    if (!(std::isfinite(rhs[0]) &&
          std::isfinite(rhs[1]) &&
          std::isfinite(rhs[2]) &&
          std::isfinite(rhs[3]) &&
          std::isfinite(rhs[4]) &&
          std::isfinite(rhs[5]) &&
          std::isfinite(rhs[6]) &&
          std::isfinite(rhs[7]))) {
        CCTK_VWARN(CCTK_WARN_ALERT, "RHS for particle %d invalid outside of horizon, u=(x, y, z, v_x, v_y, v_z, ln_E, tau)=(%f, %f, %f, %f, %f, %f, %f, %f), du/dt=(%f, %f, %f, %f, %f, %f, %f, %f)",
            index, 
            rhs[0], rhs[1], rhs[2], rhs[3], rhs[4], rhs[5], rhs[6], rhs[7],
            u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7]
        )
        deletion_reason = -998;
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
    if (u[0] > phi[0]) {
        return -1;
    }
    if (u[0] < plo[0]) {
        return -2;
    }
    if (u[1] > phi[1]) {
        return -3;
    }
    if (u[1] < plo[1]) {
        return -4;
    }
    if (u[2] > phi[2]) {
        return -5;
    }
    if (u[2] < plo[2]) {
        return -6;
    }
    if (u[7] >= 1.0) {
        return -8
    }
    return u[8];
}