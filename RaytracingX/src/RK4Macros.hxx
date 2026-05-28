#ifndef RK4MACROS_HXX
#define RK4MACROS_HXX

#define DEFINE_RK4_VARS auto& rk4 = this->splitRK4idx;                                   \
                        DEBUG(std::to_string(rk4.U[0]))\
                        DEBUG(std::to_string(rk4.U[1]))\
                        DEBUG(std::to_string(rk4.U[2]))\
                        DEBUG(std::to_string(rk4.U[3]))\
                        DEBUG(std::to_string(rk4.U[4]))\
                        DEBUG(std::to_string(rk4.U[5]))\
                        DEBUG(std::to_string(rk4.U[6]))\
                        DEBUG(std::to_string(rk4.U[7]))\
                        DEBUG(std::to_string(attribs.size()))\
                        CCTK_REAL *AMREX_RESTRICT U0 = attribs[rk4.U[0]].data();         \
                        CCTK_REAL *AMREX_RESTRICT U1 = attribs[rk4.U[1]].data();         \
                        CCTK_REAL *AMREX_RESTRICT U2 = attribs[rk4.U[2]].data();         \
                        CCTK_REAL *AMREX_RESTRICT U3 = attribs[rk4.U[3]].data();         \
                        CCTK_REAL *AMREX_RESTRICT U4 = attribs[rk4.U[4]].data();         \
                        CCTK_REAL *AMREX_RESTRICT U5 = attribs[rk4.U[5]].data();         \
                        CCTK_REAL *AMREX_RESTRICT U6 = attribs[rk4.U[6]].data();         \
                        CCTK_REAL *AMREX_RESTRICT U7 = attribs[rk4.U[7]].data();         \
                        CCTK_REAL *AMREX_RESTRICT U_tmp0 = attribs[rk4.U_tmp[0]].data(); \
                        CCTK_REAL *AMREX_RESTRICT U_tmp1 = attribs[rk4.U_tmp[1]].data(); \
                        CCTK_REAL *AMREX_RESTRICT U_tmp2 = attribs[rk4.U_tmp[2]].data(); \
                        CCTK_REAL *AMREX_RESTRICT U_tmp3 = attribs[rk4.U_tmp[3]].data(); \
                        CCTK_REAL *AMREX_RESTRICT U_tmp4 = attribs[rk4.U_tmp[4]].data(); \
                        CCTK_REAL *AMREX_RESTRICT U_tmp5 = attribs[rk4.U_tmp[5]].data(); \
                        CCTK_REAL *AMREX_RESTRICT U_tmp6 = attribs[rk4.U_tmp[6]].data(); \
                        CCTK_REAL *AMREX_RESTRICT U_tmp7 = attribs[rk4.U_tmp[7]].data(); \
                        CCTK_REAL *AMREX_RESTRICT k_odd0 = attribs[rk4.k_odd[0]].data();   \
                        CCTK_REAL *AMREX_RESTRICT k_odd1 = attribs[rk4.k_odd[1]].data();   \
                        CCTK_REAL *AMREX_RESTRICT k_odd2 = attribs[rk4.k_odd[2]].data();   \
                        CCTK_REAL *AMREX_RESTRICT k_odd3 = attribs[rk4.k_odd[3]].data();   \
                        CCTK_REAL *AMREX_RESTRICT k_odd4 = attribs[rk4.k_odd[4]].data();   \
                        CCTK_REAL *AMREX_RESTRICT k_odd5 = attribs[rk4.k_odd[5]].data();   \
                        CCTK_REAL *AMREX_RESTRICT k_odd6 = attribs[rk4.k_odd[6]].data();   \
                        CCTK_REAL *AMREX_RESTRICT k_odd7 = attribs[rk4.k_odd[7]].data();   \
                        CCTK_REAL *AMREX_RESTRICT k_even0 = attribs[rk4.k_even[0]].data(); \
                        CCTK_REAL *AMREX_RESTRICT k_even1 = attribs[rk4.k_even[1]].data(); \
                        CCTK_REAL *AMREX_RESTRICT k_even2 = attribs[rk4.k_even[2]].data(); \
                        CCTK_REAL *AMREX_RESTRICT k_even3 = attribs[rk4.k_even[3]].data(); \
                        CCTK_REAL *AMREX_RESTRICT k_even4 = attribs[rk4.k_even[4]].data(); \
                        CCTK_REAL *AMREX_RESTRICT k_even5 = attribs[rk4.k_even[5]].data(); \
                        CCTK_REAL *AMREX_RESTRICT k_even6 = attribs[rk4.k_even[6]].data(); \
                        CCTK_REAL *AMREX_RESTRICT k_even7 = attribs[rk4.k_even[7]].data();

#define UNLOAD_RK4_VARS U[0] = U0[i];           \
                        U[1] = U1[i];           \
                        U[2] = U2[i];           \
                        U[3] = U3[i];           \
                        U[4] = U4[i];           \
                        U[5] = U5[i];           \
                        U[6] = U6[i];           \
                        U[7] = U7[i];           \
                        U_tmp[0] = U_tmp0[i];   \
                        U_tmp[1] = U_tmp1[i];   \
                        U_tmp[2] = U_tmp2[i];   \
                        U_tmp[3] = U_tmp3[i];   \
                        U_tmp[4] = U_tmp4[i];   \
                        U_tmp[5] = U_tmp5[i];   \
                        U_tmp[6] = U_tmp6[i];   \
                        U_tmp[7] = U_tmp7[i];   \
                        k_odd[0] = k_odd0[i];   \
                        k_odd[1] = k_odd1[i];   \
                        k_odd[2] = k_odd2[i];   \
                        k_odd[3] = k_odd3[i];   \
                        k_odd[4] = k_odd4[i];   \
                        k_odd[5] = k_odd5[i];   \
                        k_odd[6] = k_odd6[i];   \
                        k_odd[7] = k_odd7[i];   \
                        k_even[0] = k_even0[i]; \
                        k_even[1] = k_even1[i]; \
                        k_even[2] = k_even2[i]; \
                        k_even[3] = k_even3[i]; \
                        k_even[4] = k_even4[i]; \
                        k_even[5] = k_even5[i]; \
                        k_even[6] = k_even6[i]; \
                        k_even[7] = k_even7[i];

#define LOAD_RK4_VARS DEBUG("8.1")\
                      U0[i] = U[0];           \
                      U1[i] = U[1];           \
                      U2[i] = U[2];           \
                      U3[i] = U[3];           \
                      U4[i] = U[4];           \
                      U5[i] = U[5];           \
                      U6[i] = U[6];           \
                      U7[i] = U[7];           \
                      DEBUG("8.2")\
                      U_tmp0[i] = U_tmp[0];   \
                      U_tmp1[i] = U_tmp[1];   \
                      U_tmp2[i] = U_tmp[2];   \
                      U_tmp3[i] = U_tmp[3];   \
                      U_tmp4[i] = U_tmp[4];   \
                      U_tmp5[i] = U_tmp[5];   \
                      U_tmp6[i] = U_tmp[6];   \
                      U_tmp7[i] = U_tmp[7];   \
                      DEBUG("8.3")\
                      k_odd0[i] = k_odd[0];   \
                      k_odd1[i] = k_odd[1];   \
                      k_odd2[i] = k_odd[2];   \
                      k_odd3[i] = k_odd[3];   \
                      k_odd4[i] = k_odd[4];   \
                      k_odd5[i] = k_odd[5];   \
                      k_odd6[i] = k_odd[6];   \
                      k_odd7[i] = k_odd[7];   \
                      DEBUG("8.4")\
                      k_even0[i] = k_even[0]; \
                      k_even1[i] = k_even[1]; \
                      k_even2[i] = k_even[2]; \
                      k_even3[i] = k_even[3]; \
                      k_even4[i] = k_even[4]; \
                      k_even5[i] = k_even[5]; \
                      k_even6[i] = k_even[6]; \
                      k_even7[i] = k_even[7]; \
                      DEBUG("8.5")

#define REDEFINE_RK4_ARRAYS amrex::GpuArray<CCTK_REAL, 9> U      = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; \
                            amrex::GpuArray<CCTK_REAL, 9> U_tmp  = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; \
                            amrex::GpuArray<CCTK_REAL, 9> k_odd  = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; \
                            amrex::GpuArray<CCTK_REAL, 9> k_even = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

#define SKIP_DELETED_PARTICLES if (particles[i].id() == -1) { return; }

#endif