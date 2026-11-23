/**
 * \file InterpolateMetric.cxx
 * 
 * \brief Defines methods to get the value of the metric.
 * 
 * Finds metric data used for the photon initialization. Uses the struct \ref Metric defined in \ref RaytracingX.h.
 * \ref interpolateMetricAtPoint() is used most of the time, while \ref fillMetric() is only used when using the
 * \ref extended_camera parameter and the camera is positioned outside the grid.
 */

#include "RaytracingX.h"
#include <vector>
#include <array>
#include <mpi.h>

/**
 * \brief Interpolates metric at a given point
 * 
 * This function wraps
 * \link https://einsteintoolkit.org/thornguide/CarpetX/CarpetX/documentation.html#x1-290009 DriverInterpolate \endlink
 * to interpolate the metric quantities at a given point. Only used in camera initialization, but is a useful
 * debugging tool.
 * 
 * @param metric_at_point An empty \ref Metric object that will be filled in with the appropriate data.
 * @param x The $x$ position of the point at which to interpolate the metric. Usually \ref camera_pos "camera_pos[0]"
 * unless used for debugging.
 * @param y The $y$ position of the point at which to interpolate the metric. Usually \ref camera_pos "camera_pos[1]"
 * unless used for debugging.
 * @param z The $z$ position of the point at which to interpolate the metric. Usually \ref camera_pos "camera_pos[2]"
 * unless used for debugging.
 */
void interpolateMetricAtPoint(CCTK_ARGUMENTS, Metric* metric_at_point, CCTK_REAL x, CCTK_REAL y, CCTK_REAL z) {
    DECLARE_CCTK_ARGUMENTS
    DECLARE_CCTK_PARAMETERS

    if(verbose) {
      CCTK_INFO("interpolateMetricAtPoint");
    }

    //only call interpolation on one processor for the single point.
    const int nPoints = (CCTK_MyProc(cctkGH) == 0) ? 1 : 0;

    std::array<std::vector<CCTK_REAL>, 3> location_;
    location_[0].push_back(x);
    location_[1].push_back(y);
    location_[2].push_back(z);

    //only interpolating at 1 point, so only size for one point
    std::array<std::vector<CCTK_REAL>, 10> metric_;
    metric_[0].resize(1);
    metric_[1].resize(1);
    metric_[2].resize(1);
    metric_[3].resize(1);
    metric_[4].resize(1);
    metric_[5].resize(1);
    metric_[6].resize(1);
    metric_[7].resize(1);
    metric_[8].resize(1);
    metric_[9].resize(1);
    
    //interpolation coordinates
    const void *interpCoords[3] = {
        location_[0].data(), location_[1].data(), location_[2].data()};

    //interpolated variables
    const CCTK_INT nInputArrays = 10;
    const CCTK_INT inputArrayIndices[nInputArrays] = {
        CCTK_VarIndex("ADMBaseX::alp"), 
        CCTK_VarIndex("ADMBaseX::betax"), CCTK_VarIndex("ADMBaseX::betay"), CCTK_VarIndex("ADMBaseX::betaz"),
        CCTK_VarIndex("ADMBaseX::gxx"), CCTK_VarIndex("ADMBaseX::gxy"), CCTK_VarIndex("ADMBaseX::gxz"),
        CCTK_VarIndex("ADMBaseX::gyy"), CCTK_VarIndex("ADMBaseX::gyz"), CCTK_VarIndex("ADMBaseX::gzz")};

    CCTK_POINTER outputArrays[nInputArrays] = {metric_[0].data(), metric_[1].data(), metric_[2].data(), metric_[3].data(), metric_[4].data(),
                                               metric_[5].data(), metric_[6].data(), metric_[7].data(), metric_[8].data(), metric_[9].data()};

    //DriverInterpolate arguments that aren't currently implemented
    const int coordSystemHandle = 0;
    const CCTK_INT interpCoordsTypeCode = 0;
    const CCTK_INT outputArrayTypes[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    const int interpHandle = CCTK_InterpHandle("CarpetX");
    if (interpHandle < 0) {
      CCTK_ERROR("Can't get interpolation handle");
      return;
    }

    // Create parameter table for interpolation
    const int paramTableHandle = Util_TableCreate(UTIL_TABLE_FLAGS_DEFAULT);
    if (paramTableHandle < 0) {
      CCTK_VERROR("Can't create parameter table: %d", paramTableHandle);
    }

    // Set interpolation order in the parameter table
    int ierr = Util_TableSetInt(paramTableHandle, 1, "order");
    if (ierr < 0) {
      CCTK_VERROR("Can't set order in parameter table: %d", ierr);
    }

    // Perform the interpolation
    ierr = DriverInterpolate(cctkGH, 3, interpHandle, paramTableHandle,
                             coordSystemHandle, nPoints, interpCoordsTypeCode,
                             interpCoords, nInputArrays, inputArrayIndices,
                             nInputArrays, outputArrayTypes, outputArrays);

    if (ierr < 0) {
      CCTK_ERROR("Interpolation error.");
    }

    // Destroy the parameter table
    Util_TableDestroy(paramTableHandle);

    // Send data from rank 0 to all other ranks
    MPI_Datatype mpi_real = (sizeof(CCTK_REAL) == sizeof(double)) ? MPI_DOUBLE : MPI_FLOAT;
    for (int i = 0; i < 10; ++i) {
      MPI_Bcast(metric_[i].data(),
                1,
                mpi_real,
                0,
                MPI_COMM_WORLD);
    }

    //fill metric object with interpolated values
    metric_at_point->alpha = metric_[0].data()[0];
    metric_at_point->beta_xup = metric_[1].data()[0];
    metric_at_point->beta_yup = metric_[2].data()[0];
    metric_at_point->beta_zup = metric_[3].data()[0];
    metric_at_point->g_xx = metric_[4].data()[0];
    metric_at_point->g_xy = metric_[5].data()[0];
    metric_at_point->g_xz = metric_[6].data()[0];
    metric_at_point->g_yy = metric_[7].data()[0];
    metric_at_point->g_yz = metric_[8].data()[0];
    metric_at_point->g_zz = metric_[9].data()[0];

    //\beta_i = \gamma_ij*\beta^j
    metric_at_point->beta_x = metric_at_point->beta_xup*metric_at_point->g_xx + metric_at_point->beta_yup*metric_at_point->g_xy + metric_at_point->beta_zup*metric_at_point->g_xz;
    metric_at_point->beta_y = metric_at_point->beta_xup*metric_at_point->g_xy + metric_at_point->beta_yup*metric_at_point->g_yy + metric_at_point->beta_zup*metric_at_point->g_yz;
    metric_at_point->beta_z = metric_at_point->beta_xup*metric_at_point->g_xz + metric_at_point->beta_yup*metric_at_point->g_yz + metric_at_point->beta_zup*metric_at_point->g_zz;

    //g_{tt}=\alpha^2-\beta_i\beta^i
    metric_at_point->g_tt = -pow(metric_at_point->alpha,2) + metric_at_point->beta_x*metric_at_point->beta_xup + metric_at_point->beta_y*metric_at_point->beta_yup + metric_at_point->beta_z*metric_at_point->beta_zup; //g_{00}

    metric_at_point->fillInverseMetric(); //get g^{\mu\nu}
} 

/**
 * \brief Fills metric with flat spacetime values
 * 
 * @param metric_at_point An empty \ref Metric object that will be filled in with the appropriate data.
 */
void fillMetricFlat(CCTK_ARGUMENTS, Metric* metric_at_point) {
    DECLARE_CCTK_ARGUMENTS
    DECLARE_CCTK_PARAMETERS

    if(verbose) {
      CCTK_INFO("fillMetricFlat");
    }

    metric_at_point->alpha = 1.0;
    metric_at_point->beta_xup = 0.0;
    metric_at_point->beta_yup = 0.0;
    metric_at_point->beta_zup = 0.0;
    metric_at_point->g_xx = 1.0;
    metric_at_point->g_xy = 0.0;
    metric_at_point->g_xz = 0.0;
    metric_at_point->g_yy = 1.0;
    metric_at_point->g_yz = 0.0;
    metric_at_point->g_zz = 1.0;

    //\beta_i = \gamma_ij*\beta^j
    metric_at_point->beta_x = 0.0;
    metric_at_point->beta_y = 0.0;
    metric_at_point->beta_z = 0.0;

    //g_{tt}=\alpha^2-\beta_i\beta^i
    metric_at_point->g_tt = -1.0; //g_{00}

    metric_at_point->fillInverseMetric(); //get g^{\mu\nu}
}