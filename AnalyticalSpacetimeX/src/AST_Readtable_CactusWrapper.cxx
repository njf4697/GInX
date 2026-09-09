#include <stdlib.h>
#include <cctk.h>
#include <cctk_Parameters.h>
#include <cctk_Arguments.h>
#include "AST_Helpers.hxx"
#include <cmath>
#include "indexing.hxx"
#include <loop_device.hxx>
#include "AST_Readtable.hxx"
#include "AST_AnalyticTrajectory.hxx"

namespace AnalyticalSpacetimeX {

extern "C" void AST_SetPositions(CCTK_ARGUMENTS) {

  DECLARE_CCTK_ARGUMENTSX_AST_SetPositions;
  DECLARE_CCTK_PARAMETERS;

  using namespace MoveGrids;

  CCTK_REAL tt = cctk_time;
  double *traj_array =
      (double *)amrex::The_Managed_Arena()->alloc(NTABLES * sizeof(double));
  CCTK_INT idt = 0;

  /* Whether we load traj from a table or we compute analytical trajectories */
  if (traj_read_table) {
     idt = find_traj_t(tt, traj_array);
  }
  else {
     calc_traj(CCTK_PASS_CTOC, tt, traj_array);
  }

  //CCTK_INT idt = find_traj_t(tt, traj_array);

//  fprintf(stderr, "test1");

  const bool print_traj = cctk_iteration%traj_print_every == 0 && traj_print_every > 0;

//  fprintf(stderr, "test2");

  *xbh1 = traj_array[X1];
  *ybh1 = traj_array[Y1];
  *zbh1 = traj_array[Z1];
  *xbh2 = traj_array[X2];
  *ybh2 = traj_array[Y2];
  *zbh2 = traj_array[Z2];
  *vxbh1 = traj_array[VX1] + 1e-40;
  *vybh1 = traj_array[VY1] + 1e-40;
  *vzbh1 = traj_array[VZ1] + 1e-40;
  *vxbh2 = traj_array[VX2] + 1e-40;
  *vybh2 = traj_array[VY2] + 1e-40;
  *vzbh2 = traj_array[VZ2] + 1e-40;
  *current_separation = sqrt(
      (traj_array[X1] - traj_array[X2]) * (traj_array[X1] - traj_array[X2]) +
      (traj_array[Y1] - traj_array[Y2]) * (traj_array[Y1] - traj_array[Y2]) +
      (traj_array[Z1] - traj_array[Z2]) * (traj_array[Z1] - traj_array[Z2]));

  if (cctk_iteration == 0) {
    printf("INFO (AnalyticalSpacetimeX): Initial BBH properties from table \n");
    printf("INFO (AnalyticalSpacetimeX): X1=%f \n", *xbh1);
    printf("INFO (AnalyticalSpacetimeX): Y1=%f \n", *ybh1);
    printf("INFO (AnalyticalSpacetimeX): Z1=%f \n", *zbh1);
    printf("INFO (AnalyticalSpacetimeX): X2=%f \n", *xbh2);
    printf("INFO (AnalyticalSpacetimeX): Y2=%f \n", *ybh2);
    printf("INFO (AnalyticalSpacetimeX): Z2=%f \n", *zbh2);
    printf("INFO (AnalyticalSpacetimeX): VX1=%f \n", *vxbh1);
    printf("INFO (AnalyticalSpacetimeX): VY1=%f \n", *vybh1);
    printf("INFO (AnalyticalSpacetimeX): VZ1=%f \n", *vzbh1);
    printf("INFO (AnalyticalSpacetimeX): VX2=%f \n", *vxbh2);
    printf("INFO (AnalyticalSpacetimeX): VY2=%f \n", *vybh2);
    printf("INFO (AnalyticalSpacetimeX): VZ2=%f \n", *vzbh2);
    printf("INFO (AnalyticalSpacetimeX): M1=%f \n", traj_array[M1T]);
    printf("INFO (AnalyticalSpacetimeX): M2=%f \n", traj_array[M2T]);
    printf("INFO (AnalyticalSpacetimeX): AX1=%f \n", traj_array[AX1]);
    printf("INFO (AnalyticalSpacetimeX): AY1=%f \n", traj_array[AY1]);
    printf("INFO (AnalyticalSpacetimeX): AZ1=%f \n", traj_array[AZ1]);
    printf("INFO (AnalyticalSpacetimeX): AX2=%f \n", traj_array[AX2]);
    printf("INFO (AnalyticalSpacetimeX): AY2=%f \n", traj_array[AY2]);
    printf("INFO (AnalyticalSpacetimeX): AZ2=%f \n", traj_array[AZ2]);
    *init_separation = sqrt(
        (traj_array[X1] - traj_array[X2]) * (traj_array[X1] - traj_array[X2]) +
        (traj_array[Y1] - traj_array[Y2]) * (traj_array[Y1] - traj_array[Y2]) +
        (traj_array[Z1] - traj_array[Z2]) * (traj_array[Z1] - traj_array[Z2]));
  } else if (print_traj)  {
    printf("INFO (AnalyticalSpacetimeX): BBH properties from table \n");
    printf("INFO (AnalyticalSpacetimeX): tt=%f \n", tt);
    printf("INFO (AnalyticalSpacetimeX): idt=%d \n", idt);
    printf("INFO (AnalyticalSpacetimeX): X1=%f \n", *xbh1);
    printf("INFO (AnalyticalSpacetimeX): Y1=%f \n", *ybh1);
    printf("INFO (AnalyticalSpacetimeX): Z1=%f \n", *zbh1);
    printf("INFO (AnalyticalSpacetimeX): X2=%f \n", *xbh2);
    printf("INFO (AnalyticalSpacetimeX): Y2=%f \n", *ybh2);
    printf("INFO (AnalyticalSpacetimeX): Z2=%f \n", *zbh2);
    printf("INFO (AnalyticalSpacetimeX): VX1=%f \n", *vxbh1);
    printf("INFO (AnalyticalSpacetimeX): VY1=%f \n", *vybh1);
    printf("INFO (AnalyticalSpacetimeX): VZ1=%f \n", *vzbh1);
    printf("INFO (AnalyticalSpacetimeX): VX2=%f \n", *vxbh2);
    printf("INFO (AnalyticalSpacetimeX): VY2=%f \n", *vybh2);
    printf("INFO (AnalyticalSpacetimeX): VZ2=%f \n", *vzbh2);
    printf("INFO (AnalyticalSpacetimeX): M1=%f \n", traj_array[M1T]);
    printf("INFO (AnalyticalSpacetimeX): M2=%f \n", traj_array[M2T]);
    printf("INFO (AnalyticalSpacetimeX): AX1=%f \n", traj_array[AX1]);
    printf("INFO (AnalyticalSpacetimeX): AY1=%f \n", traj_array[AY1]);
    printf("INFO (AnalyticalSpacetimeX): AZ1=%f \n", traj_array[AZ1]);
    printf("INFO (AnalyticalSpacetimeX): AX2=%f \n", traj_array[AX2]);
    printf("INFO (AnalyticalSpacetimeX): AY2=%f \n", traj_array[AY2]);
    printf("INFO (AnalyticalSpacetimeX): AZ2=%f \n", traj_array[AZ2]);
  }
}

extern "C" void traj_at_time_t_cactus_wrapper(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_traj_at_time_t_cactus_wrapper;
  DECLARE_CCTK_PARAMETERS;
  //printf("INFO (AnalyticalSpacetimeX): Executing traj_at_time_t_cactus_wrapper.. \n");
  double tt = cctk_time;
  double traj_array[NTABLES];
  int idt = find_traj_t(tt, traj_array);
  if (traj_print)  {
    printf("INFO (AnalyticalSpacetimeX): tt=%f \n", tt);
    printf("INFO (AnalyticalSpacetimeX): idt=%d \n", idt);
    printf("INFO (AnalyticalSpacetimeX): traj_array[0]=%f \n", traj_array[0]);
    printf("INFO (AnalyticalSpacetimeX): traj_array[1]=%f \n", traj_array[1]);
    printf("INFO (AnalyticalSpacetimeX): traj_array[2]=%f \n", traj_array[2]);
    printf("INFO (AnalyticalSpacetimeX): traj_array[3]=%f \n", traj_array[3]);
    printf("INFO (AnalyticalSpacetimeX): traj_array[4]=%f \n", traj_array[4]);
    printf("INFO (AnalyticalSpacetimeX): traj_array[5]=%f \n", traj_array[5]);
    printf("INFO (AnalyticalSpacetimeX): traj_array[6]=%f \n", traj_array[6]);
    printf("INFO (AnalyticalSpacetimeX): traj_array[7]=%f \n", traj_array[7]);
    printf("INFO (AnalyticalSpacetimeX): traj_array[8]=%f \n", traj_array[8]);
    printf("INFO (AnalyticalSpacetimeX): traj_array[9]=%f \n", traj_array[9]);
    printf("INFO (AnalyticalSpacetimeX): traj_array[10]=%f \n", traj_array[10]);
    printf("INFO (AnalyticalSpacetimeX): traj_array[11]=%f \n", traj_array[11]);
  }
}

extern "C" void traj_readtable_cactus_wrapper(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_traj_readtable_cactus_wrapper;
  DECLARE_CCTK_PARAMETERS;

  //printf("INFO (AnalyticalSpacetimeX): Executing traj_C_ReadTable.. \n");
  traj_C_ReadTable(traj_table_name);
}

} // namespace AnalyticalSpacetimeX
