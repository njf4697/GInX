#ifndef INDEXING_HXX
#define INDEXING_HXX

#include <cctk.h>

#include <assert.h>

#include <loop_device.hxx>

namespace MoveGrids {

using namespace std;
using namespace Loop;
using namespace Arith;

CCTK_HOST CCTK_DEVICE CCTK_ATTRIBUTE_ALWAYS_INLINE inline int
index2(int const *const lsh, int const *const ash, int const i, int const j) {
  assert(lsh);
  assert(ash);
  assert(lsh[0] >= 0 && lsh[0] <= ash[0]);
  assert(lsh[1] >= 0 && lsh[1] <= ash[1]);
  assert((i >= 0) && (i < lsh[0]));
  assert((j >= 0) && (j < lsh[1]));
  return i + ash[0] * j;
}

// Get indexing information for a vector grid array
CCTK_HOST CCTK_ATTRIBUTE_ALWAYS_INLINE inline void
getvectorindex2(cGH const *const cctkGH, char const *const groupname,
                int *restrict const lsh, int *restrict const ash) {
  assert(groupname);
  assert(lsh);
  assert(ash);

  int const gi = CCTK_GroupIndex(groupname);
  assert(gi >= 0);

  {
    int const ierr = CCTK_GrouplshGI(cctkGH, 1, lsh, gi);
    assert(not ierr);
  }

  {
    int const ierr = CCTK_GroupashGI(cctkGH, 1, ash, gi);
    assert(not ierr);
  }

  cGroup groupdata;
  {
    int const ierr = CCTK_GroupData(gi, &groupdata);
    assert(not ierr);
  }
  assert(groupdata.vectorgroup);
  assert(groupdata.vectorlength >= 0);
  lsh[1] = groupdata.vectorlength;
  ash[1] = groupdata.vectorlength;
}

} // namespace MoveGrids

#endif // #ifndef INDEXING_HXX
