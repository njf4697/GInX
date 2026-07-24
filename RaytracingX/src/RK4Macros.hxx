#ifndef RK4MACROS_HXX
#define RK4MACROS_HXX

#define DEFINE_RK4_VARS CCTK_REAL *AMREX_RESTRICT U0 = attribs[StructType::U0].data(); \
                        CCTK_REAL *AMREX_RESTRICT U1 = attribs[StructType::U1].data(); \
                        CCTK_REAL *AMREX_RESTRICT U2 = attribs[StructType::U2].data(); \
                        CCTK_REAL *AMREX_RESTRICT U3 = attribs[StructType::U3].data(); \
                        CCTK_REAL *AMREX_RESTRICT U4 = attribs[StructType::U4].data(); \
                        CCTK_REAL *AMREX_RESTRICT U5 = attribs[StructType::U5].data(); \
                        CCTK_REAL *AMREX_RESTRICT U6 = attribs[StructType::U6].data(); \
                        CCTK_REAL *AMREX_RESTRICT U7 = attribs[StructType::U7].data(); \
                        CCTK_REAL *AMREX_RESTRICT k0 = attribs[StructType::k0].data(); \
                        CCTK_REAL *AMREX_RESTRICT k1 = attribs[StructType::k1].data(); \
                        CCTK_REAL *AMREX_RESTRICT k2 = attribs[StructType::k2].data(); \
                        CCTK_REAL *AMREX_RESTRICT k3 = attribs[StructType::k3].data(); \
                        CCTK_REAL *AMREX_RESTRICT k4 = attribs[StructType::k4].data(); \
                        CCTK_REAL *AMREX_RESTRICT k5 = attribs[StructType::k5].data(); \
                        CCTK_REAL *AMREX_RESTRICT k6 = attribs[StructType::k6].data(); \
                        CCTK_REAL *AMREX_RESTRICT k7 = attribs[StructType::k7].data();

#define UNLOAD_RK4_VARS U[0] = U0[i]; \
                        U[1] = U1[i]; \
                        U[2] = U2[i]; \
                        U[3] = U3[i]; \
                        U[4] = U4[i]; \
                        U[5] = U5[i]; \
                        U[6] = U6[i]; \
                        U[7] = U7[i]; \
                        k[0] = k0[i]; \
                        k[1] = k1[i]; \
                        k[2] = k2[i]; \
                        k[3] = k3[i]; \
                        k[4] = k4[i]; \
                        k[5] = k5[i]; \
                        k[6] = k6[i]; \
                        k[7] = k7[i];

#define LOAD_RK4_VARS U0[i] = U[0]; \
                      U1[i] = U[1]; \
                      U2[i] = U[2]; \
                      U3[i] = U[3]; \
                      U4[i] = U[4]; \
                      U5[i] = U[5]; \
                      U6[i] = U[6]; \
                      U7[i] = U[7]; \
                      k0[i] = k[0]; \
                      k1[i] = k[1]; \
                      k2[i] = k[2]; \
                      k3[i] = k[3]; \
                      k4[i] = k[4]; \
                      k5[i] = k[5]; \
                      k6[i] = k[6]; \
                      k7[i] = k[7]; \

#define REDEFINE_RK4_ARRAYS amrex::GpuArray<CCTK_REAL, 9> U = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; \
                            amrex::GpuArray<CCTK_REAL, 9> k = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

#define SKIP_DELETED_PARTICLES if (particles[i].id() == -1) { return; }

#endif