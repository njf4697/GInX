#ifndef AST_READTABLE_HXX
#define AST_READTABLE_HXX

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
//#define H5_USE_16_API 1
#include "hdf5.h"
#include "cctk.h"
#include "cctk_Parameters.h"
#include "cctk_Arguments.h"
#include <loop_device.hxx>
#include "AST_Helpers.hxx"
#include "tensor.hxx"

namespace AnalyticalSpacetimeX {
/* Function declarations */
CCTK_HOST CCTK_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline double interpolateLin(int a, int b, double x, double *X,
                                            double *Y);
CCTK_HOST CCTK_ATTRIBUTE_ALWAYS_INLINE inline static int file_is_readable(const char *filename);
CCTK_HOST CCTK_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline double w(double x);
CCTK_HOST CCTK_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline double safe_exp_neg_inv(double x);
CCTK_HOST CCTK_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline int search_sorted_array(double x, double *arr, int N);

/* HDF5 Macros */
#define HDF5_ERROR(fn_call)                                                    \
  do {                                                                         \
    int _error_code = fn_call;                                                 \
    if (_error_code < 0) {                                                     \
      printf(                                                                  \
          "INFO (AnalyticalSpacetimeX): HDF5 call returned error code %d \n",  \
          _error_code);                                                        \
    }                                                                          \
  } while (0)

// Use these two defines to easily read in a lot of variables in the same way
// The first reads in one variable of a given type completely
#define READ_TRAJ_HDF5(NAME, VAR, TYPE, MEM)                                   \
  do {                                                                         \
    hid_t dataset;                                                             \
    HDF5_ERROR(dataset = H5Dopen1(file, NAME));                                 \
    HDF5_ERROR(H5Dread(dataset, TYPE, MEM, H5S_ALL, H5P_DEFAULT, VAR));        \
    HDF5_ERROR(H5Dclose(dataset));                                             \
  } while (0)

// The second reads a given variable into a hyperslab of the alltables_temp
// array
#define READ_TRAJTABLE_HDF5(NAME, OFF)                                         \
  do {                                                                         \
    hsize_t offset[2] = { OFF, 0 };                                            \
    H5Sselect_hyperslab(mem3, H5S_SELECT_SET, offset, NULL, var3, NULL);       \
    READ_TRAJ_HDF5(NAME, alltables_temp, H5T_NATIVE_DOUBLE, mem3);             \
  } while (0)

/*  Functions for interpolation */

/** Geoff's interpolating function **/
CCTK_HOST CCTK_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline double w(double x) {
  //# this is the function, but it doesn't deal with x <= 0 or x >= 1 properly
  //# and it overflows for small values of x
  //# use the safe version instead
  double a = exp(-1.0 / x);
  double b = exp(-1.0 / (1 - x));
  double fun = a / (a + b);
  return fun;
}

CCTK_HOST CCTK_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline double safe_exp_neg_inv(double x) {
  //# this is just a safe wrapper for the 'entire' function
  //# f(x) = {exp(-1/x) x>0, 0 otherwise}.
  //# exp(x) overflows at double precision if x > 719 ~ 700
  //# 1/700 ~ 1.4e-3
  //# So set exp(-1/x) = 0.0 for x < 1.4e-3 (this could be made marginally
  //# more precise but the numbers are already VERY small
  double tol = 1.4e-3;
  if (x > tol) {
    return exp(-1.0 / x);
  } else {
    return 0;
  }
}

CCTK_HOST CCTK_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline int search_sorted_array(double x, double *arr,
                                              int previous_max_it,
                                              int previous_min_it) {
  int N = previous_max_it;
  int zero_it = previous_min_it;

  if (x <= arr[zero_it])
    return zero_it;
  else if (x >= arr[N - 1])
    return N - 2;

  unsigned int i = ((unsigned int)N) >> 1;
  unsigned int a = zero_it;
  unsigned int b = N - 1;

  while (b - a > 1u) {
    i = (b + a) >> 1;
    if (arr[i] > x)
      b = i;
    else
      a = i;
  }

  return (int)a;
}

/* Linear interpolation function */
CCTK_HOST CCTK_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline double interpolateLin(int a, int b, int idt, double x,
                                            double *X, double *Y) {
  double xa = X[idt];
  double xb = X[idt + 1];
  double ya = Y[a];
  double yb = Y[b];

  return ya + (yb - ya) * (x - xa) / (xb - xa);
}

CCTK_HOST CCTK_ATTRIBUTE_ALWAYS_INLINE inline static int file_is_readable(const char *filename) {
  FILE *fp = NULL;
  fp = fopen(filename, "r");
  if (fp != NULL) {
    fclose(fp);
    return 1;
  }
  return 0;
}

/* Globals */
/* These quantities have a global scope in the code */
/* We load them in readtable and use them in find_traj_t */
inline int nt_tab;                 // number of points in time
inline double *restrict alltables; // array with trajectories NTABLE*nt_tab
inline double *restrict t_tab;     // array with time
// These are not used (yet)
inline int previous_max_it; // Max value where to look into table
inline int previous_min_it; // Min value where to look into table
inline double m1_tab;       // (fixed) mass of BH 1
inline double m2_tab;       // (fixed)  mass of BH 2
inline double t_merger;     // time of merger
inline double t_postmerger; // time of post-merger
inline double a_x_remnant;  // remnant spin
inline double a_y_remnant;  //
inline double a_z_remnant;  //
inline double v_x_remnant;  // remnant velocity
inline double v_y_remnant;  //
inline double v_z_remnant;  //

/* This function loads the trajectory at a given time tt in traj_array*/
/* The cost come from inspecting the table, but this should be a cheap operation
 */
CCTK_HOST CCTK_ATTRIBUTE_ALWAYS_INLINE inline int find_traj_t(double tt, double *traj_array) {
  // Get iteration given tt
  int idt = search_sorted_array(
      tt, t_tab, previous_max_it,
      previous_min_it); // I checked that this works properly.
  // Loop over all tables and do the interpolation with the whole array.
  for (int iv = 0; iv < NTABLES; iv++) {
    int idx = iv + NTABLES * idt;
    int idxp = iv + NTABLES * (idt + 1);
    traj_array[iv] = interpolateLin(idx, idxp, idt, tt, t_tab, alltables);
  }

  return idt;
}

/* Function that reads the table. Executed in a cactus wrapper */
CCTK_HOST CCTK_ATTRIBUTE_ALWAYS_INLINE inline void traj_C_ReadTable(const char *traj_table_name) {

  // extern int  nt_tab; // number of points in time
  // extern double  m1_tab; // mass of BH 1
  // extern double  m2_tab; // mass of BH 2
  // extern double   t_merger; // time of merger
  // extern double * restrict alltables; //array with trajectories NTABLE*nt_tab
  // extern double * restrict t_tab; // array with time

  printf("INFO (AnalyticalSpacetimeX): ******************************* \n");
  printf("INFO (AnalyticalSpacetimeX): Reading trajectory table file: \n");
  printf("INFO (AnalyticalSpacetimeX): %s \n", traj_table_name);
  printf("INFO (AnalyticalSpacetimeX): ******************************* \n");

  hid_t file;
  if (!file_is_readable(traj_table_name)) {

    printf("INFO (AnalyticalSpacetimeX): Could not read traj_table_name %s \n",
           traj_table_name);
  }
  HDF5_ERROR(file = H5Fopen(traj_table_name, H5F_ACC_RDONLY, H5P_DEFAULT));

  // Read size of tables
  READ_TRAJ_HDF5("nt", &nt_tab, H5T_NATIVE_INT, H5S_ALL);
  // Init previous_max/min_it
  previous_max_it = nt_tab; // MIN(nt_tab, idt + buffer_it);
  previous_min_it = 0;      // MAX(0, idt - buffer_it);

  READ_TRAJ_HDF5("m1", &m1_tab, H5T_NATIVE_DOUBLE, H5S_ALL);
  READ_TRAJ_HDF5("m2", &m2_tab, H5T_NATIVE_DOUBLE, H5S_ALL);
  READ_TRAJ_HDF5("t_postmerger", &t_postmerger, H5T_NATIVE_DOUBLE, H5S_ALL);
  READ_TRAJ_HDF5("a_x_remnant", &a_x_remnant, H5T_NATIVE_DOUBLE, H5S_ALL);
  READ_TRAJ_HDF5("a_y_remnant", &a_y_remnant, H5T_NATIVE_DOUBLE, H5S_ALL);
  READ_TRAJ_HDF5("a_z_remnant", &a_z_remnant, H5T_NATIVE_DOUBLE, H5S_ALL);
  READ_TRAJ_HDF5("v_x_remnant", &v_x_remnant, H5T_NATIVE_DOUBLE, H5S_ALL);
  READ_TRAJ_HDF5("v_y_remnant", &v_y_remnant, H5T_NATIVE_DOUBLE, H5S_ALL);
  READ_TRAJ_HDF5("v_z_remnant", &v_z_remnant, H5T_NATIVE_DOUBLE, H5S_ALL);

  // Allocate memory for tables
  double *alltables_temp;
  if (!(alltables_temp = (double *)amrex::The_Managed_Arena()->alloc(
            nt_tab * NTABLES * sizeof(double)))) {
    printf("INFO (AnalyticalSpacetimeX): Cannot allocate memory for TRAJ table "
           "\n");
  }
  if (!(t_tab = (double *)amrex::The_Managed_Arena()->alloc(nt_tab *
                                                            sizeof(double)))) {
    printf("INFO (AnalyticalSpacetimeX): Cannot allocate memory for TRAJ table "
           "\n");
  }

  // Prepare HDF5 to read hyperslabs into alltables_temp
  hsize_t table_dims[2] = { NTABLES, (hsize_t)nt_tab };
  hsize_t var3[2] = { 1, (hsize_t)nt_tab };
  hid_t mem3 = H5Screate_simple(2, table_dims, NULL);

  printf("INFO (AnalyticalSpacetimeX): We have nt=%d points in each array \n",
         nt_tab);

  // Read alltables_temp
  READ_TRAJTABLE_HDF5("x1", X1);
  READ_TRAJTABLE_HDF5("y1", Y1);
  READ_TRAJTABLE_HDF5("z1", Z1);
  READ_TRAJTABLE_HDF5("x2", X2);
  READ_TRAJTABLE_HDF5("y2", Y2);
  READ_TRAJTABLE_HDF5("z2", Z2);
  READ_TRAJTABLE_HDF5("vx1", VX1);
  READ_TRAJTABLE_HDF5("vy1", VY1);
  READ_TRAJTABLE_HDF5("vz1", VZ1);
  READ_TRAJTABLE_HDF5("vx2", VX2);
  READ_TRAJTABLE_HDF5("vy2", VY2);
  READ_TRAJTABLE_HDF5("vz2", VZ2);
  /* Spins */
  READ_TRAJTABLE_HDF5("a1x", AX1);
  READ_TRAJTABLE_HDF5("a1y", AY1);
  READ_TRAJTABLE_HDF5("a1z", AZ1);
  READ_TRAJTABLE_HDF5("a2x", AX2);
  READ_TRAJTABLE_HDF5("a2y", AY2);
  READ_TRAJTABLE_HDF5("a2z", AZ2);
  /* Masses */
  READ_TRAJTABLE_HDF5("m1_full", M1T);
  READ_TRAJTABLE_HDF5("m2_full", M2T);
  // Read additional tables and variables
  READ_TRAJ_HDF5("t", t_tab, H5T_NATIVE_DOUBLE, H5S_ALL);

  HDF5_ERROR(H5Sclose(mem3));
  HDF5_ERROR(H5Fclose(file));

  // change ordering of alltables array so that
  // the table kind is the fastest changing index
  if (!(alltables = (double *)amrex::The_Managed_Arena()->alloc(
            nt_tab * NTABLES * sizeof(double)))) {
    printf("INFO (AnalyticalSpacetimeX): Cannot allocate memory for TRAJ table "
           "\n");
  }

  for (int iv = 0; iv < NTABLES; iv++) {
    for (int k = 0; k < nt_tab; k++) {
      int indold = k + nt_tab * iv;
      int indnew = iv + NTABLES * k;
      alltables[indnew] = alltables_temp[indold];
    }
  }

  // free memory of temporary array
  amrex::The_Managed_Arena()->free(alltables_temp);
}

} // namespace AnalyticalSpacetimeX

#endif
