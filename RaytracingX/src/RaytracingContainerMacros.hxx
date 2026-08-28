#ifndef RAYTRACINGCONTAINERMACROS
#define RAYTRACINGCONTAINERMACROS

#define INV_DET_GAMMA(gamma) 1.0 / (gamma[G3_xx] * gamma[G3_yy] * gamma[G3_zz] + \
                               2. * gamma[G3_xy] * gamma[G3_xz] * gamma[G3_yz] - \
                               gamma[G3_xz] * gamma[G3_xz] * gamma[G3_yy] -      \
                               gamma[G3_yz] * gamma[G3_yz] * gamma[G3_xx] -      \
                               gamma[G3_xy] * gamma[G3_xy] * gamma[G3_zz])       

#define INV_GAMMA(gamma, inv_det_gamma) { (gamma[G3_yy] * gamma[G3_zz] - gamma[G3_yz] * gamma[G3_yz]) * inv_det_gamma, \
                                          (gamma[G3_yz] * gamma[G3_xz] - gamma[G3_xy] * gamma[G3_zz]) * inv_det_gamma, \
                                          (gamma[G3_xy] * gamma[G3_yz] - gamma[G3_xz] * gamma[G3_yy]) * inv_det_gamma, \
                                          (gamma[G3_xx] * gamma[G3_zz] - gamma[G3_xz] * gamma[G3_xz]) * inv_det_gamma, \
                                          (gamma[G3_xz] * gamma[G3_xy] - gamma[G3_xx] * gamma[G3_yz]) * inv_det_gamma, \
                                          (gamma[G3_xx] * gamma[G3_yy] - gamma[G3_xy] * gamma[G3_xy]) * inv_det_gamma}


#define SPATIAL_INNER_PRODUCT(V_down, inv_gamma) V_down[0] * V_down[0] * inv_gamma[G3_xx] +       \
                                                 V_down[1] * V_down[1] * inv_gamma[G3_yy] +       \
                                                 V_down[2] * V_down[2] * inv_gamma[G3_zz] +       \
                                                 2.0 * V_down[0] * V_down[1] * inv_gamma[G3_xy] + \
                                                 2.0 * V_down[0] * V_down[2] * inv_gamma[G3_xz] + \
                                                 2.0 * V_down[1] * V_down[2] * inv_gamma[G3_yz]   
                                        
#define RAISE_SPATIAL(V_down, inv_gamma) {inv_gamma[G3_xx] * V_down[0] + inv_gamma[G3_xy] * V_down[1] + inv_gamma[G3_xz] * V_down[2], \
                                          inv_gamma[G3_xy] * V_down[0] + inv_gamma[G3_yy] * V_down[1] + inv_gamma[G3_yz] * V_down[2], \
                                          inv_gamma[G3_xz] * V_down[0] + inv_gamma[G3_yz] * V_down[1] + inv_gamma[G3_zz] * V_down[2]}

#endif