#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>
#include "indexing.hxx"

/* Move things around */
extern "C" void AST_MoveGrids(CCTK_ARGUMENTS)
{  
  DECLARE_CCTK_ARGUMENTSX_AST_MoveGrids;
  DECLARE_CCTK_PARAMETERS;

  using namespace MoveGrids;

  constexpr int max_num_regions = 3;

  if (cctk_iteration%move_every==0) {
  
      // Region 1
      position_x[0] = *xbh1;
      position_y[0] = *ybh1;
      position_z[0] = *zbh1;

      // Region 2
      position_x[1] = *xbh2;
      position_y[1] = *ybh2;
      position_z[1] = *zbh2;

      // Region 3 is not moved
      position_x[2] = position_x[2];
      position_y[2] = position_y[2];
      position_z[2] = position_z[2];

  } else {
 
      for (int i = 0; i < max_num_regions; i++) {

        position_x[i] = position_x[i];
        position_y[i] = position_y[i];
        position_z[i] = position_z[i];

      }
  }
} 

/* Resize things */
extern "C" void AST_ResizeGrids(CCTK_ARGUMENTS)
{  
  DECLARE_CCTK_ARGUMENTSX_AST_ResizeGrids;
  DECLARE_CCTK_PARAMETERS;

  using namespace MoveGrids;

  // Find the index of the box we are changing
  int lsh[2], ash[2], ind;
  getvectorindex2(cctkGH,"BoxInBox::radii",lsh,ash);
  
  for (int i = 0; i < ash[0]*ash[1]; i++) {

    radius[i] = radius[i];

  }

  if (cctk_iteration%resize_every==0) {
  
      // Calculate the resizing factor
      CCTK_REAL resize_factor = *current_separation / *init_separation;
      // Check if have reached the limit
      if (resize_factor < resize_factor_limit) {return;}
   
      if (resize_level_1>0){
      ind = index2(lsh,ash,resize_level_1,resize_region); 
      radius[ind] = init_radius_level_1*resize_factor;
      }
      if (resize_level_2>0){
      ind = index2(lsh,ash,resize_level_2,resize_region); 
      radius[ind] = init_radius_level_2*resize_factor;
      }
      if (resize_level_3>0){
      ind = index2(lsh,ash,resize_level_3,resize_region); 
      radius[ind] = init_radius_level_3*resize_factor;
      }
  }
}
