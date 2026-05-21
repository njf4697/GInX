#ifndef RK4MACROS_HXX
#define RK4MACROS_HXX

#define CHECK_OUT_OF_BOUNDS_X(X) \
    if (X > boundarie_hx)        \
    {                            \
        out_of_bounds = true;    \
        deletion_reasons[i] = -1;\
    }                            \
    if (X < boundarie_lx)        \
    {                            \
        out_of_bounds = true;    \
        deletion_reasons[i] = -2;\
    }
#define CHECK_OUT_OF_BOUNDS_Y(Y) \
    if (Y > boundarie_hy)        \
    {                            \
        out_of_bounds = true;    \
        deletion_reasons[i] = -3;\
    }                            \
    if (Y < boundarie_ly)        \
    {                            \
        out_of_bounds = true;    \
        deletion_reasons[i] = -4;\
    }
#define CHECK_OUT_OF_BOUNDS_Z(Z) \
    if (Z > boundarie_hz)        \
    {                            \
        out_of_bounds = true;    \
        deletion_reasons[i] = -5;\
    }                            \
    if (Z < boundarie_lz)        \
    {                            \
        out_of_bounds = true;    \
        deletion_reasons[i] = -6;\
    }

#define CHECK_VELOCITY(I, VX, VY, VZ) \
    if (VX*VX+VY*VY+VZ*VZ>4)         \
    {                                 \
        CCTK_VWARN(CCTK_WARN_ALERT,   \
            "Particle %d has velocity (%f, %f, %f) with mag. >= 2 indicating that the evolution may be unstable! Particle deleted.\n", I, VX, VY, VZ);\
        particles[i].id() = -1;       \
        deletion_reasons[i] = -998;   \
        return;                       \
    }

#define DEFINE_RK4_VARS CCTK_REAL *AMREX_RESTRICT U0 = attribs[ptclRK4data.U[0]].data();         \
                        CCTK_REAL *AMREX_RESTRICT U1 = attribs[ptclRK4data.U[1]].data();         \
                        CCTK_REAL *AMREX_RESTRICT U2 = attribs[ptclRK4data.U[2]].data();         \
                        CCTK_REAL *AMREX_RESTRICT U3 = attribs[ptclRK4data.U[3]].data();         \
                        CCTK_REAL *AMREX_RESTRICT U4 = attribs[ptclRK4data.U[4]].data();         \
                        CCTK_REAL *AMREX_RESTRICT U5 = attribs[ptclRK4data.U[5]].data();         \
                        CCTK_REAL *AMREX_RESTRICT U6 = attribs[ptclRK4data.U[6]].data();         \
                        CCTK_REAL *AMREX_RESTRICT U7 = attribs[ptclRK4data.U[7]].data();         \
                        CCTK_REAL *AMREX_RESTRICT U_tmp0 = attribs[ptclRK4data.U_tmp[0]].data(); \
                        CCTK_REAL *AMREX_RESTRICT U_tmp1 = attribs[ptclRK4data.U_tmp[1]].data(); \
                        CCTK_REAL *AMREX_RESTRICT U_tmp2 = attribs[ptclRK4data.U_tmp[2]].data(); \
                        CCTK_REAL *AMREX_RESTRICT U_tmp3 = attribs[ptclRK4data.U_tmp[3]].data(); \
                        CCTK_REAL *AMREX_RESTRICT U_tmp4 = attribs[ptclRK4data.U_tmp[4]].data(); \
                        CCTK_REAL *AMREX_RESTRICT U_tmp5 = attribs[ptclRK4data.U_tmp[5]].data(); \
                        CCTK_REAL *AMREX_RESTRICT U_tmp6 = attribs[ptclRK4data.U_tmp[6]].data(); \
                        CCTK_REAL *AMREX_RESTRICT U_tmp7 = attribs[ptclRK4data.U_tmp[7]].data(); \
                        CCTK_REAL *AMREX_RESTRICT k_odd0 = attribs[ptclRK4data.k_odd[0]].data();   \
                        CCTK_REAL *AMREX_RESTRICT k_odd1 = attribs[ptclRK4data.k_odd[1]].data();   \
                        CCTK_REAL *AMREX_RESTRICT k_odd2 = attribs[ptclRK4data.k_odd[2]].data();   \
                        CCTK_REAL *AMREX_RESTRICT k_odd3 = attribs[ptclRK4data.k_odd[3]].data();   \
                        CCTK_REAL *AMREX_RESTRICT k_odd4 = attribs[ptclRK4data.k_odd[4]].data();   \
                        CCTK_REAL *AMREX_RESTRICT k_odd5 = attribs[ptclRK4data.k_odd[5]].data();   \
                        CCTK_REAL *AMREX_RESTRICT k_odd6 = attribs[ptclRK4data.k_odd[6]].data();   \
                        CCTK_REAL *AMREX_RESTRICT k_odd7 = attribs[ptclRK4data.k_odd[7]].data();   \
                        CCTK_REAL *AMREX_RESTRICT k_even0 = attribs[ptclRK4data.k_even[0]].data(); \
                        CCTK_REAL *AMREX_RESTRICT k_even1 = attribs[ptclRK4data.k_even[1]].data(); \
                        CCTK_REAL *AMREX_RESTRICT k_even2 = attribs[ptclRK4data.k_even[2]].data(); \
                        CCTK_REAL *AMREX_RESTRICT k_even3 = attribs[ptclRK4data.k_even[3]].data(); \
                        CCTK_REAL *AMREX_RESTRICT k_even4 = attribs[ptclRK4data.k_even[4]].data(); \
                        CCTK_REAL *AMREX_RESTRICT k_even5 = attribs[ptclRK4data.k_even[5]].data(); \
                        CCTK_REAL *AMREX_RESTRICT k_even6 = attribs[ptclRK4data.k_even[6]].data(); \
                        CCTK_REAL *AMREX_RESTRICT k_even7 = attribs[ptclRK4data.k_even[7]].data();

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

#define LOAD_RK4_VARS U0[i] = U[0];           \
                      U1[i] = U[1];           \
                      U2[i] = U[2];           \
                      U3[i] = U[3];           \
                      U4[i] = U[4];           \
                      U5[i] = U[5];           \
                      U6[i] = U[6];           \
                      U7[i] = U[7];           \
                      U_tmp0[i] = U_tmp[0];   \
                      U_tmp1[i] = U_tmp[1];   \
                      U_tmp2[i] = U_tmp[2];   \
                      U_tmp3[i] = U_tmp[3];   \
                      U_tmp4[i] = U_tmp[4];   \
                      U_tmp5[i] = U_tmp[5];   \
                      U_tmp6[i] = U_tmp[6];   \
                      U_tmp7[i] = U_tmp[7];   \
                      k_odd0[i] = k_odd[0];   \
                      k_odd1[i] = k_odd[1];   \
                      k_odd2[i] = k_odd[2];   \
                      k_odd3[i] = k_odd[3];   \
                      k_odd4[i] = k_odd[4];   \
                      k_odd5[i] = k_odd[5];   \
                      k_odd6[i] = k_odd[6];   \
                      k_odd7[i] = k_odd[7];   \
                      k_even0[i] = k_even[0]; \
                      k_even1[i] = k_even[1]; \
                      k_even2[i] = k_even[2]; \
                      k_even3[i] = k_even[3]; \
                      k_even4[i] = k_even[4]; \
                      k_even5[i] = k_even[5]; \
                      k_even6[i] = k_even[6]; \
                      k_even7[i] = k_even[7];

#define REDEFINE_RK4_ARRAYS amrex::GpuArray<CCTK_REAL, 8> U      = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; \
                            amrex::GpuArray<CCTK_REAL, 8> U_tmp  = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; \
                            amrex::GpuArray<CCTK_REAL, 8> k_odd  = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; \
                            amrex::GpuArray<CCTK_REAL, 8> k_even = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

#define SKIP_DELETED_PARTICLES if (particles[i].id() == -1) { return; }

#endif