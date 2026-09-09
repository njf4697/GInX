/* AST_AnalyticTrajectory.hxx */
/* (c) Liwei Ji, 08/15/2024 */

/**
 * \brief Set BH trajectories using analytic expressions
 */

#ifndef AST_ANALYTICTRAJECTORY_HXX
#define AST_ANALYTICTRAJECTORY_HXX

#include "AST_Helpers.hxx"

namespace AnalyticalSpacetimeX {

CCTK_HOST CCTK_ATTRIBUTE_ALWAYS_INLINE inline void calc_traj(CCTK_ARGUMENTS, double t, double *traj_array) {
  DECLARE_CCTK_PARAMETERS;

  /* orbital seperation */
  double a = 20.0;
  double m1 = 0.5;
  double m2 = 0.5;
  double m = m1 + m2;

  double omega = sqrt(m / pow(a, 3));
  double phi = omega * t;

  traj_array[X1] = (m2 / m) * a * cos(phi);
  traj_array[Y1] = (m2 / m) * a * sin(phi);
  traj_array[Z1] = 0.0;
  traj_array[X2] = -(m1 / m) * a * cos(phi);
  traj_array[Y2] = -(m1 / m) * a * sin(phi);
  traj_array[Z2] = 0.0;

  traj_array[VX1] = -omega * (m2 / m) * a * sin(phi);
  traj_array[VY1] = omega * (m2 / m) * a * cos(phi);
  traj_array[VZ1] = 0.0;
  traj_array[VX2] = omega * (m1 / m) * a * sin(phi);
  traj_array[VY2] = -omega * (m1 / m) * a * cos(phi);
  traj_array[VZ2] = 0.0;

  traj_array[AX1] = 0.0;
  traj_array[AY1] = 0.0;
  traj_array[AZ1] = 0.0;
  traj_array[AX2] = 0.0;
  traj_array[AY2] = 0.0;
  traj_array[AZ2] = 0.0;

  traj_array[M1T] = 0.5;
  traj_array[M2T] = 0.5;
}

} // namespace AnalyticalSpacetimeX

#endif // #ifndef AST_ANALYTICTRAJECTORY_HXX
