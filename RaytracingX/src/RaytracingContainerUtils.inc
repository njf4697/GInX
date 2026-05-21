template <typename StructType>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE CCTK_ATTRIBUTE_ALWAYS_INLINE
int RaytracingX::RaytracingParticlesContainer<StructType>::get_interpolation_center(const CCTK_REAL point, const CCTK_REAL lower, const CCTK_REAL upper, const CCTK_REAL dx)
{
    int i = amrex::Math::floor((amrex::Clamp(point, lower, upper - dx / 4) - lower) / dx);
    ASSERT_FINITE(i)
    return i;
}