#ifndef RAYTRACINGCONTAINER
#define RAYTRACINGCONTAINER

#include <cctk.h>

#include "Photons.hxx"
#include "BaseParticlesContainer.hxx"
#include "ParticlesContainer.hxx"
#include "RaytracingX.h"
#include <AMReX_ParallelDescriptor.H>
#include <CParameters.h>
#include <fstream>
#include "Interpolator.hxx"
#include <AMReX_Particles.H>
#include <AMReX_AmrParticles.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_MFIter.H>

#include "RK4Macros.hxx"             

namespace RaytracingX
{

    struct RaytracingPhotonsData : public GInX::PhotonsData
    {
        enum
        {
            vx = 0,       /**< Velocity's lower index on the x direction.*/
            vy,           /**< Velocity's lower index on the y direction.*/
            vz,           /**< Velocity's lower index on the z direction.*/
            ln_alphaE,         /**< Ln Energy value.*/
            tau,          /**< Optical depth value. Also used as particle deletion code, see above macros, also iterates across banned regions*/
            pixel_number, /**< Number used to match particle to corresponding pixel in the image. Defined as a real since BaseParticleContainer does not have options for int parameters, unless defined at runtime, in which case they will not print with WriteAsciiFile*/
            deletion_reason,
            n_attributes, /**< Total number of attributes*/
        }; // enum
    };

    enum Uidx {
        x = 0,
        y,
        z,
        vx,
        vy,
        vz,
        lnaE,
        tau,
        del_rsn,
        n_attributes
    };

    struct DelReason {
        static constexpr CCTK_REAL XHI = -1;
        static constexpr CCTK_REAL XLO = -2;
        static constexpr CCTK_REAL YHI = -3;
        static constexpr CCTK_REAL YLO = -4;
        static constexpr CCTK_REAL ZHI = -5;
        static constexpr CCTK_REAL ZLO = -6;
        static constexpr CCTK_REAL HORIZON = -7;
        static constexpr CCTK_REAL PHOTOSPHERE = -8;
        static constexpr CCTK_REAL BANNED_REGION_OFFSET = -9;
        static constexpr CCTK_REAL UNSTABLE = -997;
        static constexpr CCTK_REAL NONFINITE = -998;
        static constexpr CCTK_REAL DEFAULT = -999;
    };

    template <typename StructType>
    class RaytracingParticlesContainer : public GInX::BaseParticleContainer<RaytracingParticlesContainer<StructType>,
                                                                                   StructType>
    {

    /**
     * \brief RaytracingParticlesContainer class definition.
     *
     * The following class defines the needed functions to evolve the position and
     * velocity of the photons in the simulation for raytracing purposes. Many of the methods
     * are similar to those found in ParticlesContainer/ParticlesContainer.hxx, so changes will be prefaced with 
     * 'RaytraingX:'.
     */

    protected:
        CCTK_REAL mass = 0.;

    public:
        /**
         * \brief Using BaseParticlesContainer constructor
         */

        using Base =
            GInX::BaseParticleContainer<RaytracingParticlesContainer<StructType>,StructType>;
        using Base::Base;

        RaytracingParticlesContainer(amrex::AmrCore *amr_core, const CCTK_REAL m)
            : Base(amr_core), mass{m} { };

        ~RaytracingParticlesContainer() = default;

        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE CCTK_ATTRIBUTE_ALWAYS_INLINE
        static int get_interpolation_center(
            const CCTK_REAL point,
            const CCTK_REAL lower,
            const CCTK_REAL upper,
            const CCTK_REAL dx);

        static void write_to_one_file(
            std::string filename,
            std::string data);

        void write_deleted_particle_data(
            const int &lev,
            std::string final_data_file_name);

        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE CCTK_ATTRIBUTE_ALWAYS_INLINE
        amrex::GpuArray<CCTK_REAL, 9> compute_rhs(
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
            const CCTK_REAL max_energy);

        void evolve(
            const int iteration,
            const amrex::MultiFab &lapse,
            const amrex::MultiFab &shift,
            const amrex::MultiFab &metric,
            const amrex::MultiFab &curv,
            const amrex::MultiFab &rho,
            const CCTK_REAL &dt,
            const int &lev,
            const CCTK_REAL max_energy);

        void evolve_k1(
            const int iteration,
            const amrex::MultiFab &lapse,
            const amrex::MultiFab &shift,
            const amrex::MultiFab &metric,
            const amrex::MultiFab &curv,
            const amrex::MultiFab &rho,
            const CCTK_REAL &dt,
            const int &lev,
            const CCTK_REAL max_energy,
            const ptclRK4data ptclRK4data);

        void evolve_k2(
            const int iteration,
            const amrex::MultiFab &lapse,
            const amrex::MultiFab &shift,
            const amrex::MultiFab &metric,
            const amrex::MultiFab &curv,
            const amrex::MultiFab &rho,
            const CCTK_REAL &dt,
            const int &lev,
            const CCTK_REAL max_energy,
            const ptclRK4data ptclRK4data);

        void evolve_k3(
            const int iteration,
            const amrex::MultiFab &lapse,
            const amrex::MultiFab &shift,
            const amrex::MultiFab &metric,
            const amrex::MultiFab &curv,
            const amrex::MultiFab &rho,
            const CCTK_REAL &dt,
            const int &lev,
            const CCTK_REAL max_energy,
            const ptclRK4data ptclRK4data);

        void evolve_k4(
            const int iteration,
            const amrex::MultiFab &lapse,
            const amrex::MultiFab &shift,
            const amrex::MultiFab &metric,
            const amrex::MultiFab &curv,
            const amrex::MultiFab &rho,
            const CCTK_REAL &dt,
            const int &lev,
            const CCTK_REAL max_energy,
            const ptclRK4data ptclRK4data);
        
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE CCTK_ATTRIBUTE_ALWAYS_INLINE
        CCTK_REAL check_bounds(
            const amrex::GpuArray<CCTK_REAL, 9> u,
            const amrex::GpuArray<double, 3> &plo,
            const amrex::GpuArray<double, 3> &phi);
        
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE CCTK_ATTRIBUTE_ALWAYS_INLINE
        CCTK_REAL check_validity(
            const amrex::GpuArray<CCTK_REAL, 9> rhs,
            const amrex::GpuArray<CCTK_REAL, 9> u,
            const CCTK_REAL lapse,
            const CCTK_REAL max_energy,
            const int index);
        
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE CCTK_ATTRIBUTE_ALWAYS_INLINE
        static CCTK_REAL mag2_massless(
            const CCTK_REAL x,
            const CCTK_REAL y,
            const CCTK_REAL z,
            const amrex::GpuArray<CCTK_REAL, 6> gamma)
        
        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE CCTK_ATTRIBUTE_ALWAYS_INLINE
        static amrex::GpuArray<CCTK_REAL, 3> raise_lower_spatial(
            const CCTK_REAL x,
            const CCTK_REAL y,
            const CCTK_REAL z,
            const amrex::GpuArray<CCTK_REAL, 6> gamma);

        void check_horizon(
            const amrex::MultiFab &lapse,
            const int &lev,
            const CCTK_REAL max_energy);

        void check_banned_zones(
            const int &level, 
            const CCTK_INT4 &zones,
            const CCTK_REAL (&x)[10], 
            const CCTK_REAL (&y)[10],
            const CCTK_REAL (&z)[10],
            const CCTK_REAL (&radius)[10],
            const CCTK_REAL (&a)[10]);

        void normalize_velocity(
            const amrex::MultiFab &metric,
            const int level);
        
        void redistribute_particles()
        {
            CCTK_INFO("Redistributing particles");
        } // RaytracingParticlesContainer::redistribute_particles

        //RaytracingX: evolve method override necessary because expanding virtual class from BaseParticleContainer
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