/**
 * \file RaytracingContainer.hxx
 * 
 * \brief Defines the \ref RaytracingX::RaytracingParticlesContainer class
 * 
 * Extends from \ref GInX::BaseParticleContainer class, which itself is based on
 * \link https://amrex-codes.github.io/amrex/docs_html/Particle.html AMReX particles \endlink.
 * 
 * \see BaseParticleContainer
 * \see ParticleContainer
 * \see RaytracingContainerUtils.tpp
 * \see RaytracingContainerManagers.tpp
 * \see RaytracingContainerRK4.tpp
 */

#ifndef RAYTRACINGCONTAINER
#define RAYTRACINGCONTAINER

//RaytracingX thorn
#include "RaytracingX.h"  
#include "RaytracingContainerMacros.hxx"
//BaseParticlesContainer thorn
#include "BaseParticlesContainer.hxx"
//ParticlesContainer thorn
#include "Photons.hxx"
#include "ParticlesContainer.hxx"
//ParticlesSolverUtilities thorn
#include "Interpolator.hxx"

#include <AMReX_ParallelDescriptor.H>
#include <fstream>
#include <AMReX_Particles.H>
#include <AMReX_AmrParticles.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_MFIter.H>

namespace RaytracingX
{
    template <typename StructType>
    class RaytracingParticlesContainer : public GInX::BaseParticleContainer<RaytracingParticlesContainer<StructType>,
                                                                                   StructType>
    {

    /**
     * \brief RaytracingParticlesContainer class definition.
     *
     * The following class declares the needed functions to evolve the position and
     * velocity of the photons in the simulation for raytracing purposes. \ref compute_rhs(), \ref evolve_kn(),
     * \ref check_banned_zones(), and \ref normalize_velocity() are based on, and thus similar to those found 
     * in \ref BaseParticleContainer and/or \ref ParticleContainer.
     * 
     * \see evolve_kn()
     * \see compute_rhs()
     * \see check_horizons()
     */

    protected:
        CCTK_REAL mass = 0.;

    public:
        /**
         * \brief Constructor
         * 
         * Uses constructer from \ref BaseParticleContainer, but the template StructType
         * substitution ensures the appropriate amount of memory is allocated when the substitution 
         * \ref RaytracingPhotonsData-->StructType is made.
         */

        using Base =
            GInX::BaseParticleContainer<RaytracingParticlesContainer<StructType>,StructType>;
        using Base::Base;

        RaytracingParticlesContainer(amrex::AmrCore *amr_core, const CCTK_REAL m)
            : Base(amr_core), mass{m} { };

        ~RaytracingParticlesContainer() = default;

        /**
         * \brief Gets the index of the interpolation stencil center point
         * 
         * Gets the index of the interpolation stencil center point (see Interpolator.hxx) for
         * grid function interpolation for the appropriate dimension. This is done for each
         * dimension separately. Note that this function does not check that the returned index is valid
         * (see \ref interp_index_is_in_bounds()).
         * 
         * @param point The real-value position of the requested interpolation point in the relevant dimention
         * @param lower The real-value lower bound
         * @param upper The real-value upper bound
         * @param dx The spatial step size
         * 
         * \return The index into the grid arrays of the grid function in the relevant dimention
         * 
         * \see interp_index_is_in_bounds()
         * \see compute_rhs()
         * \see interpolate_array()
         * \see d_interpolate_array()
         */
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE CCTK_ATTRIBUTE_ALWAYS_INLINE
        static int get_interpolation_center(
            const CCTK_REAL point,
            const CCTK_REAL lower,
            const CCTK_REAL upper,
            const CCTK_REAL dx);
        
        /** \brief Checks to make sure that the index for interpolation is in bounds of the simulation
         * 
         * The interpolation index for each dimension i0, j0, k0 is checked to make sure that it lies within the grid.
         * If it is outside the grid bounds, the particle is flagged for deletion and marked as unstable (see \ref compute_rhs()). This is
         * a fallback routine to ensure no out of bounds reads are done for the grid arrays.
         * 
         * @param i0 Interpolation index into grid array for point in $x$ direction
         * @param j0 Interpolation index into grid array for point in $y$ direction
         * @param k0 Interpolation index into grid array for point in $z$ direction
         * @param lower Vector containing the lower bounds for valid indices into the grid array
         * @param upper Vector containing the upper bounds for valid indices into the grid array
         * 
         * \return Boolean, true if interpolation index is outside grid, false otherwise
         * 
         * \see compute_rhs()
         * \see interpolate_array()
         * \see d_interpolate_array()
         * \see interpolate_array()
         */
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE CCTK_ATTRIBUTE_ALWAYS_INLINE
        static bool interp_index_is_in_bounds(
            const long i0,
            const long j0,
            const long k0,
            const amrex::GpuArray<int, 3> lower,
            const amrex::GpuArray<int, 3> upper);
        
        /**
         * \brief Finds the number of particles that are inside the grid but outside the specified bounds (if applicable)
         * 
         * When \ref extended_camera is true, particles are evolved using a coarser timesteps (see \ref coarse_dtfac) if until they are all
         * inside the specified bounds set by \ref use_custom_bounds. The number of particles outside the specifed bounds is used 
         * to find when to change the time step.
         * 
         * @param lev Refinement level of current particle container
         * @param bounds A \ref PhotonBounds struct containing the custom bounds set by \ref use_custom_bounds
         * 
         * \return The number of particles that are inside the grid but outside the specified bounds
         * 
         * \see use_custom_bounds
         * \see SetDeltaTime()
         * \see CheckOutOfBoundsParticlesNumber()
         */
        long num_particles_out_of_bounds(
            const int &lev,
            const PhotonBounds &bounds);
        
        /**
         * \brief Writes data from all processors to one file
         * 
         * Data can be passed from all processors to be written in a single file for output. Used in 
         * \ref write_deleted_particle_data(). Order is not guaranteed.
         * 
         * @param filename File to write to
         * @param data String to write to file
         * 
         * \see write_deleted_particle_data()
         */
        static void write_to_one_file(
            std::string filename,
            std::string data);
        
        /**
         * \brief Write particle data to file upon particle deletion
         * 
         * A wrapper around \ref write_to_one_file() that outputs the state of the particle upon deletion. This
         * data is used to create the images.
         * 
         * @param lev Refinement level of current particle container
         * @param final_data_file_name Output filename for final particle data given by \ref final_data_file_name.
         * 
         * \see write_to_one_file()
         * \see final_data_file_name
         * \see ParticlesPrintFinalData()
         */
        void write_deleted_particle_data(
            const int &lev,
            std::string final_data_file_name);
        
        /**
         * \brief Compute right-hand side for evolution equation
         * 
         * Computes the right-hand side of the evolution equation ($k_1$ through $k_4$) for 4th-order Runge-Kutta evolution.
         * Called by \ref evolve_kn() for each $k_n$ in RK4. This method is called four times for each particle
         * in every particle container per iteration.
         * 
         * The evolution equations for geodesic evolution in 3+1 spacetime decomposition are
         * 
         * \f{eqnarray*}{
         * \frac{d}{dt} U[i] = \alpha \gamma^{ij} U[3 + j] - \beta^i\f]
         * \frac{d}{dt}U[3 + i] &= -\partial_i\alpha + \left(\gamma^{kj} U[3 + k]
         * \partial_j\alpha - \alpha K_{jk}\gamma^{jl}\gamma^{km}U[3+l]U[3+m]\right) U[3
         * + i]\\ & +
         * \frac{1}{2}\alpha\gamma^{jl}\gamma^{km}U[3+l]U[3+m]\partial_i\gamma{jk} + U[3
         * + j] \partial_i\beta^j
         * \dtfac{d}{dt} U[6] = \alpha K_{jk}U[3 + l]U[3 + m]\gamma^{lj}\gamma^{mk}
         * - U[3+l]\gamma^{lj}\partial_j\alpha
         * \f}
         * 
         * with $U={x, y, z, v_x, v_y, v_z, \ln E, \rho, delrsn}$ (using zero-indexing) with $\rho$ being added
         * for eventual implementation of optical depth calculation along geodesic and $delrsn$ being the deletion
         * reason for the particle if applicable (see \ref DelReason).
         * 
         * @param iteration Current iteration number
         * @param index Image index of current particle (see \ref fill_photon_data_standard_camera())
         * @param u Array containing the first argument of $f(U, t)$ for $k_1$, $k_2$, $k_3$, and $k_4$ defined above with $U={x, y, z, v_x, v_y, v_z, \ln E, \rho, delrsn}$
         * @param lapse The lapse variabe ($\alpha$) for the 3+1 spacetime decomposition
         * @param shift The shift variabe ($\beta^i$) for the 3+1 spacetime decomposition
         * @param metric The spatial metric variabe ($\gamma_{ij}$) for the 3+1 spacetime decomposition
         * @param curv The extrinsic curvature variabe ($K_{ij}$) for the 3+1 spacetime decomposition
         * @param rho The gas density variable used to calculate optical depth along geodesic (currently unused)
         * @param dx $\Delta x$ for refinement level of the current particle container
         * @param lev Refinement level of the current particle container
         * @param plo0 Real-valued vector of lower bounds of entire grid
         * @param phi0 Real-valued vector of upper bounds of entire grid
         * @param lower_valid_index Lower bounds of valid indices into global grid arrays
         * @param upper_valid_index upper bounds of valid indices into global grid arrays
         * 
         * \return Output vector of variables $U={x, y, z, v_x, v_y, v_z, \ln E, \rho, delrsn}$
         * 
         * \see evolve_kn()
         * \see interpolate_array()
         * \see d_interpolate_array()
         * \see check_validity()
         */
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE CCTK_ATTRIBUTE_ALWAYS_INLINE
        amrex::GpuArray<CCTK_REAL, 9> compute_rhs(
            const int iteration,
            const int index,
            const amrex::GpuArray<CCTK_REAL, 9> &u,
            amrex::Array4<CCTK_REAL const> const &lapse,
            const amrex::Array4<CCTK_REAL const> &shift,
            const amrex::Array4<CCTK_REAL const> &metric,
            const amrex::Array4<CCTK_REAL const> &curv,
            const amrex::Array4<CCTK_REAL const> &rho,
            const amrex::GpuArray<double, 3> &dx,
            const int lev,
            const amrex::GpuArray<double, 3> &plo0,
            const amrex::GpuArray<double, 3> &phi0,
            const amrex::GpuArray<int, 3> &lower_valid_index,
            const amrex::GpuArray<int, 3> &upper_valid_index);
        
        
        /**
         * \brief Evolve particles by one RK4 substep
         * 
         * The differential equation being solved is $\frac{dU}{dt} = f\left(U, t\right)$, and thus
         * for $U_n(t=t_0+\Delta t)$, $U_{n+1}=U_n+\frac{\Delta t}{6}(k_1+2k_2+3k_3+k_4)$ with
         * $k_1=f(U_n                , t_n           )$,
         * $k_2=f(U_n + k_1\Delta_t/2, t_n+\Delta t/2)$,
         * $k_3=f(U_n + k_2\Delta_t/2, t_n_\Delta t/2)$, and
         * $k_4=f(U_n + k_3\Delta_t  , t_n+\Delta t  )$.
         * 
         * In this case, U(t) is a vector of the variables $U(t)={x^i, v_i, \ln E}$ to evolve the geodesic equations
         * that the photons parametarize with $t$. These variables are functions of the 3+1 metric quantities, which 
         * themselves are time-dependant functions. Thus, new metric quantities must be found between each substep of
         * RK4 if metric is time-dependant.
         * 
         * Since new metric quantities must be found between substeps, the previous values of $k$ and the initial
         * value of $U$ for the iteration are stored in the particle data structure (see \ref RaytracingPhotonsData).
         * 
         * @param k Whether current substep is $k_1$, $k_2$, $k_3$, or $k_4$ (thus $k$ is 1, 2, 3, or 4, not zero)
         * @param iteration Current iteration number for simulation
         * @param lapse The lapse variabe ($\alpha$) for the 3+1 spacetime decomposition
         * @param shift The shift variabe ($\beta^i$) for the 3+1 spacetime decomposition
         * @param metric The spatial metric variabe ($\gamma_{ij}$) for the 3+1 spacetime decomposition
         * @param curv The extrinsic curvature variabe ($K_{ij}$) for the 3+1 spacetime decomposition
         * @param rho The gas density variable used to calculate optical depth along geodesic (currently unused)        
         * @param lev Refinement level of the current particle container
         * @param max_energy \ref max_energy parameter defining where to terminate photons if they get too close to a horizon
         * @param dt $\Delta t$ for the current iteration
         * @param bounds A \ref PhotonBounds object containing the grid bounds for the simulation
         * 
         * \see compute_rhs()
         * \see ParticlesEvolveKNHelper()
         * \see check_bounds()
         */
        void evolve_kn(
            const int k,
            const int iteration,
            const amrex::MultiFab &lapse,
            const amrex::MultiFab &shift,
            const amrex::MultiFab &metric,
            const amrex::MultiFab &curv,
            const amrex::MultiFab &rho,
            const int &lev,
            const CCTK_REAL max_energy,
            const CCTK_REAL dt,
            const PhotonBounds &bounds);
        
        /** 
         * \brief Checks bounds of current photon, including horizon
         * 
         * Checks bounds of current photon, including the horizon check (see \ref check_horizon()).
         * If the current particle is out of bounds, return the deletion reason (see \ref DelReason)
         * which flags particle for deletion in \ref evolve_kn(), otherwise return 0. Uses particle state 
         * vector $U$ discussed in \ref compute_rhs().
         * 
         * @param u Array containing the current particle's state with $U={x, y, z, v_x, v_y, v_z, \ln E, \rho, delrsn}$
         * @param plo0 Real-valued vector of lower bounds of entire grid
         * @param phi0 Real-valued vector of upper bounds of entire grid
         * @param dx $\Delta x$ for refinement level of the current particle container
         * @param max_energy \ref max_energy parameter defining where to terminate photons if they get too close to a horizon
         * @param lapse_array The lapse variabe ($\alpha$) for the 3+1 spacetime decomposition
         * @param bounds A \ref PhotonBounds object containing the grid bounds for the simulation
         * 
         * \return The deletion reason of the particle (0 if the particle should not be deleted)
         * 
         * \see evolve_kn()
         * \see check_horizon()
         * \see interpolate_array()
        */
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE CCTK_ATTRIBUTE_ALWAYS_INLINE
        CCTK_REAL check_bounds(
            const amrex::GpuArray<CCTK_REAL, 9> u,
            const amrex::GpuArray<double, 3> &plo0,
            const amrex::GpuArray<double, 3> &phi0,
            const amrex::GpuArray<double, 3> &dx,
            amrex::Array4<CCTK_REAL const> const &lapse_array,
            CCTK_REAL max_energy,
            const PhotonBounds &bounds);
        
        /**
         * \brief Checks to see that the current particle state has only finite values
         * 
         * Checks that each variable in $U={x, y, z, v_x, v_y, v_z, \ln E, \rho, delrsn}$ (see \ref compute_rhs())
         * and each value calculated by \ref compute_rhs()
         * is finite. If not, return the corresponding deletion reason (see \ref DelReason). Otherwise, return 0.
         * 
         * @param rhs Variable vector returned by \ref compute_rhs()
         * @param u Variable state vector $U={x, y, z, v_x, v_y, v_z, \ln E, \rho, delrsn}$
         * @param index Particle index in image (see \ref fill_photon_data_standard_camera())
         * 
         * \return Deletion reason for particle, 0 if particle is valid.
         */
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE CCTK_ATTRIBUTE_ALWAYS_INLINE
        CCTK_REAL check_validity(
            const amrex::GpuArray<CCTK_REAL, 9> rhs,
            const amrex::GpuArray<CCTK_REAL, 9> u,
            const int index);
        
        /**
         * \brief Check to see if each photon is too close to the horizon.
         * 
         * For a photon with four-velocity $p$, the photon's $p^t$ grows asyptotically as it approaches a horizon.
         * To detect this behavoir and ensure no photon actually falls into the horizon, define a maximum energy
         * $E_{max}$ given by the \ref max_energy parameter such that the photon is deleted if $p^0> E_{max}$.
         * Since $\ln E$ is stored instead of $p^t$, $p^t=e^{\ln E}/\alpha$. Flag the particle for deletion if 
         * $p^0> E_{max}$.
         * 
         * @param lapse The lapse variabe ($\alpha$) for the 3+1 spacetime decomposition
         * @param lev Refinement level of the current particle container
         * @param max_energy The maximum energy for the horizon check given by \ref max_energy
         * 
         * \see CheckHorizonAndBannedRegions()
         * \see max_energy
         * \see interpolate_array()
         */
        void check_horizon(
            const amrex::MultiFab &lapse,
            const int &lev,
            const CCTK_REAL max_energy);
        
        /** 
         * \brief Calculate conserved quantities of the Kerr metric
         * 
         * Calculates the quantities $p_t$, $L_z$, and $v_iv^i$ for each particle with $p$ being the photon's four momentum.
         * These values are only conserved in Kerr spacetimes. To save memory, these values are stored in the 
         * U[0], U[1], and U[2] variables (see \ref RaytracingPhotonsData) normally used as placeholders between RK4 
         * substeps (see \ref evolve_kn()), and thus is output via \ref ParticlesPrint().
         * 
         * @param lapse The lapse variabe ($\alpha$) for the 3+1 spacetime decomposition
         * @param shift The shift variabe ($\shift^i$) for the 3+1 spacetime decomposition
         * @param metric The spatial metric variabe ($\gamma_{ij}$) for the 3+1 spacetime decomposition
         * @param lev Refinement level of the current particle container
         * 
         * \see interpolate_array()
         * \see CalculateKerrConserved()
         */
        void calculate_kerr_conserved_quantities( //for definition and comments, see RaytracingContainerManagers.tpp
            const amrex::MultiFab &lapse,
            const amrex::MultiFab &shift,
            const amrex::MultiFab &metric,
            const int &lev);

        void check_banned_zones( //for definition and comments, see RaytracingContainerManagers.tpp
            const int &level, 
            const CCTK_INT4 &zones,
            const CCTK_REAL (&x)[10], 
            const CCTK_REAL (&y)[10],
            const CCTK_REAL (&z)[10],
            const CCTK_REAL (&radius)[10],
            const CCTK_REAL (&a)[10]);

        void normalize_velocity( //for definition and comments, see RaytracingContainerManagers.tpp
            const amrex::MultiFab &metric,
            const int level);

        //this evolve function is necessary as an override to extend the BaseParticlesContainer class, but should not be used.
        void evolve(
            const amrex::MultiFab &lapse,
            const amrex::MultiFab &shift,
            const amrex::MultiFab &metric,
            const amrex::MultiFab &curv, const CCTK_REAL &dt,
            const int &lev)
        {
            CCTK_ERROR("This evolve method should not be used! Use the other one.");
            return;
        }
    };
}

#include "RaytracingContainerUtils.tpp"
#include "RaytracingContainerManagers.tpp"
#include "RaytracingContainerRK4.tpp"

#endif