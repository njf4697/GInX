#ifndef AST_HELPERS_HXX
#define AST_HELPERS_HXX

/* Helpers definitions */
#define DEBUG_TABLE (0)
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

namespace AnalyticalSpacetimeX {

/* mnemonics for dimensional indices */
static constexpr int NDIM = 4;
static constexpr int TT = 0, XX = 1, YY = 2, ZZ = 3;

/* mnemonics for BH trajectory indices */
static constexpr int NTABLES = 20;
/* six position six velocities six spins, two masses */
static constexpr int X1 = 0, Y1 = 1, Z1 = 2, X2 = 3, Y2 = 4, Z2 = 5, VX1 = 6,
                     VY1 = 7, VZ1 = 8, VX2 = 9, VY2 = 10, VZ2 = 11, AX1 = 12,
                     AY1 = 13, AZ1 = 14, AX2 = 15, AY2 = 16, AZ2 = 17, M1T = 18,
                     M2T = 19;

} // namespace AnalyticalSpacetimeX

#endif // #ifndef AST_HELPERS_HXX
