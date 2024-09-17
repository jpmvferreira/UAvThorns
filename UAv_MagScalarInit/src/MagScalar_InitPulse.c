#include <math.h>

#include "cctk.h"
#include "cctk_Arguments.h"
#include "cctk_Parameters.h"

#define SMALL  (1.e-10)
#define SQR(x) ((x)*(x))


static CCTK_REAL gaussian(CCTK_REAL R, CCTK_REAL sigma) {
  return exp(-0.5 * SQR(R/sigma));
}


void MagScalar_InitPulse (CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;

  int imin[3], imax[3];

  for (int d = 0; d < 3; ++ d) {
    imin[d] = 0;
    imax[d] = cctk_lsh[d];
  }

 const CCTK_REAL t = cctk_time;

  for (int i = imin[0]; i < imax[0]; ++i) {
    for (int j = imin[1]; j < imax[1]; ++j) {
      for (int k = imin[2]; k < imax[2]; ++k) {

        const int ind = CCTK_GFINDEX3D (cctkGH, i, j, k);

        const CCTK_REAL x1 = x[ind];
        const CCTK_REAL y1 = y[ind];
        const CCTK_REAL z1 = z[ind];

        const CCTK_REAL r2 = SQR(x1) + SQR(y1) + SQR(z1);
        const CCTK_REAL r  = sqrt(r2);

        Ax[ind] = 0;
        Ay[ind] = 0;
        Az[ind] = 0;

        // TODO: general t ??
        Ex[ind] = - Amp * exp(- cAmp * (x1*x1 + y1*y1 + z1*z1) ) * y1;
        Ey[ind] =   Amp * exp(- cAmp * (x1*x1 + y1*y1 + z1*z1) ) * x1;
        Ez[ind] = 0;

        Aphi[ind] = 0;

        Zeta[ind] = 0;


        if (r > SMALL) {
          phi1[ind] = 0.5/r * ((r-t)*gaussian(r-t, 1.0) + (r+t)*gaussian(r+t, 1.0));
          phi2[ind] = 1.0/r * ((r-t)*gaussian(r-t, 0.5) + (r+t)*gaussian(r+t, 0.5));
        }
        else {
          phi1[ind] = 1. - 0.5*r2;
          phi2[ind] = 2. - 4.0*r2;
        }

        // TODO: general t ??
        Kphi1[ind] = 0;
        Kphi2[ind] = 0;


      }
    }
  }

}
