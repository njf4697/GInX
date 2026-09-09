#ifndef TENSOR_HXX
#define TENSOR_HXX

namespace AnalyticalSpacetimeX {

struct bbh_traj {
  double xi1x;
  double xi1y;
  double xi1z;
  double xi2x;
  double xi2y;
  double xi2z;
  double v1x;
  double v1y;
  double v1z;
  double v2x;
  double v2y;
  double v2z;
  double s1x;
  double s1y;
  double s1z;
  double s2x;
  double s2y;
  double s2z;
  double m1t;
  double m2t;
};

struct dd_sym {
  double tt;
  double tx;
  double ty;
  double tz;
  double xx;
  double xy;
  double xz;
  double yy;
  double yz;
  double zz;
};

struct four_metric {
  struct dd_sym g;
  struct dd_sym g_t;
  struct dd_sym g_x;
  struct dd_sym g_y;
  struct dd_sym g_z;
};

struct three_metric {
  double gxx;
  double gxy;
  double gxz;
  double gyy;
  double gyz;
  double gzz;
  double alpha;
  double betax;
  double betay;
  double betaz;
  double kxx;
  double kxy;
  double kxz;
  double kyy;
  double kyz;
  double kzz;
};

} // namespace AnalyticalSpacetimeX

#endif
