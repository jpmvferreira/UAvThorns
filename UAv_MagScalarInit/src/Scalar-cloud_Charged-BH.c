// Relevant documentation:
// GSL ODE solver: https://www.gnu.org/software/gsl/doc/html/ode-initval.html
// GSL interpolator: https://www.gnu.org/software/gsl/doc/html/interp.html

// macros
#define SQR(x) ((x)*(x))

// global imports
#include <math.h>
#include <stdio.h>
#include <gsl/gsl_odeiv2.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_spline.h>

// toolkit specific imports
#include "cctk.h"
#include "cctk_Arguments.h"
#include "cctk_Parameters.h"

// define a struct to hold the parameters
typedef struct {
  double M;
  double Q;
  double Amp;
  double R1;
  double R2;
  double c;
} params_t;

// define a strut to use as the return object of the function Get_Psi
typedef struct {
    double* R_sol;
    double* Psi_sol;
    double* dPsi_sol;
} ReturnGetPsi;

// function to reverse an array
void reverseArray(double *arr, int length) {
  int start = 0;
  int end = length - 1;
  while (start < end) {
    double temp = arr[start];
    arr[start] = arr[end];
    arr[end] = temp;
    start++;
    end--;
  }
}

// the conformal factor for a charged BH
double Psi_RN(double R, params_t p) {
  double M = p.M;
  double Q = p.Q;
  double c = p.c;

  return sqrt(c + (pow(M,2) - pow(Q,2))/(4.*c*pow(R,2)) + M/R);
}

// the derivative of the conformal factor for a charged BH
double dPsi_RN(double R, params_t p) {
  double M = p.M;
  double Q = p.Q;
  double c = p.c;

  return (-0.5*(pow(M,2) - pow(Q,2))/(c*pow(R,3)) - M/pow(R,2))/(2.*sqrt(c + (pow(M,2) - pow(Q,2))/(4.*c*pow(R,2)) + M/R));
}

// 𝜙(R)
double phi(double R, params_t p) {
  double Amp = p.Amp;
  double R1  = p.R1;
  double R2  = p.R2;

  if (R1 <= R && R <= R2) {
    return Amp*pow(R,3)*pow(1 - R/R1,3)*pow(-1 + R/R2,3);
  }

  return 0;
}

// 𝜕𝜙/𝜕R
double dphi(double R, params_t p) {
    double Amp = p.Amp;
    double R1  = p.R1;
    double R2  = p.R2;

    if (R1 <= R && R <= R2) {
      return (-3*Amp*pow(R,2)*pow(R - R1,2)*pow(R - R2,2)*(3*pow(R,2) + R1*R2 - 2*R*(R1 + R2)))/(pow(R1,3)*pow(R2,3));
    }

    return 0;
}

// define the ODE
int f(double R, const double y[], double dy[], void *params) {
  // fetch the relevant parameters
  // cannot change function signature due to GSL API, must recast pointer in correct type
  params_t p = *(params_t *)params;
  double Q   = p.Q;

  // compute the derivative of the scalar field at radius R
  double ldphi = dphi(R, p);

  dy[0] = y[1];
  dy[1] = (-2.*y[1])/R - pow(Q,2)/(4.*pow(R,4)*pow(y[0],3)) - 2.*pow(ldphi,2)*M_PI*y[0];

  return GSL_SUCCESS;
}

ReturnGetPsi Get_Psi(int N, params_t p) {
  // initial conditions
  double y[2];
  y[0] = Psi_RN(p.R2, p);
  y[1] = dPsi_RN(p.R2, p);

  // define variables
  double R2 = p.R2;   // starting radius
  double R1 = p.R1;   // final radius
  int dim   = 2;      // dimension of the system of equations

  // compute the step size
  double h = (R2-R1)/( (double) N);

  // allocate arrays to hold the numerical solution for Ψ between R1 and R2
  // we need N+1 points because of the IC's
  double* R_sol    = (double* ) malloc( (N+1) * sizeof(double));  // array for R's
  double* Psi_sol  = (double* ) malloc( (N+1) * sizeof(double));  // array for Ψ(R)
  double* dPsi_sol = (double* ) malloc( (N+1) * sizeof(double));  // array for dΨ/dR

  // store the IC's in the array
  R_sol[0]    = R2;
  Psi_sol[0]  = y[0];
  dPsi_sol[0] = y[1];

  // define the ODE system
  // args: f(t,y), jacobian, nº of equations, parameters of the system
  gsl_odeiv2_system sys = {f, NULL, dim, &p};

  // create a stepping function
  // uses a Runge-Kutta 4th order method
  gsl_odeiv2_step *step = gsl_odeiv2_step_alloc(gsl_odeiv2_step_rkf45, dim);

  // dummy arrays to provide to gsl_odeiv2_step_apply
  double dydt_out[2], y_err[2];

  // perform integration
  h = -h;
  double R = R2;
  for (int i=1; i<=N; i++) {
    int status = gsl_odeiv2_step_apply(step, R, h, y, y_err, NULL, dydt_out, &sys);

    // if there is an error, print it out
    if (status != GSL_SUCCESS) {
      printf("Error: %s\n", gsl_strerror(status));
      CCTK_ERROR("Solving the ODE returned an error, see above.");
    }

    // update current R
    R = R2 + i*h;

    // save the points in an array
    R_sol[i]    = R;
    Psi_sol[i]  = y[0];
    dPsi_sol[i] = y[1];
  }

  // free GSL ODE solver objects
  gsl_odeiv2_step_free(step);

  // reverse the arrays because GSL interpolation routines demand that the arrays are provided in increasing order of R
  reverseArray(R_sol,    N+1);
  reverseArray(Psi_sol,  N+1);
  reverseArray(dPsi_sol, N+1);

  ReturnGetPsi res = {R_sol, Psi_sol, dPsi_sol};

  return res;
}

// fetch the spacetime parameters when R < R₁
params_t Get_p_BH(params_t p, double C[2]) {
  double M  = p.M;
  double Q  = p.Q;
  double R1 = p.R1;

  // match solution at R = R₁
  // this is the first branch of solutions
  double M_BH = pow(Q,2)/(2.*pow(C[0],2)*R1) + 2.*C[1]*pow(R1,2)*(C[0] - C[1]*R1);
  double c_BH = pow(C[0],2) - pow(Q,2)/(4.*pow(C[0],2)*pow(R1,2)) - 2.*C[0]*C[1]*R1 + pow(C[1],2)*pow(R1,2);

  // if the mass of the BH is smaller than the abs value of charge or larger than ADM mass, use second branch of solutions
  if (M_BH < fabs(Q) || M_BH > M) {
    M_BH = pow(Q,2)/(2.*pow(C[0],2)*R1) - 2.*C[1]*pow(R1,2)*(C[0] + C[1]*R1);
    c_BH = pow(C[0],2) - pow(Q,2)/(4.*pow(C[0],2)*pow(R1,2)) + 2.*C[0]*C[1]*R1 + pow(C[1],2)*pow(R1,2);
  }

  // if the mass of the BH is still not valid, give an error
  // we're comparing `M_BH > 1.001*M` instead of `M_BH > M` due to numerical roundoff errors when computing M_BH
  // the pre-factor is quite arbitrary but in this case it works, and it's just 0.1% of the ADM mass as error
  if (M_BH < fabs(Q) || M_BH > 1.001*M) {
    CCTK_VERROR("M_BH is %g, which is either smaller than the absolute value of the charge Q = %g or larger than the ADM mass M = %g: no physical solutions have been found", M_BH, Q, M);
  }

  // otherwise, print information to screen with the mass of the BH and the value of c
  // TODO: find a way to only print this to screen once
  CCTK_VINFO("Mass of the BH: %g", M_BH);
  CCTK_VINFO("Parameter 'c' in the generalized solution below R1: %g", c_BH);

  // the parameters for the solution below R₁
  params_t p_BH = {
    .M   = M_BH,
    .Q   = p.Q,
    .Amp = p.Amp,
    .R1  = p.R1,
    .R2  = p.R2,
    .c   = c_BH,
  };

  return p_BH;
}

// "main" function
void Scalar_cloud_Charged_BH(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTS;
  DECLARE_CCTK_PARAMETERS;

  // spacetime parameters as seen by an observer at infinity
  params_t p = {
    .M   = M,
    .Q   = Q_BH,
    .Amp = Amp,
    .R1  = R1,
    .R2  = R2,
    .c   = 1,
  };

  // fetch interpolation function to compute Ψ(R)
  // remember that the size of the array is N_points+1, where i=0 is at R1 and i=N is at R2
  ReturnGetPsi sol = Get_Psi(N_points, p);
  double* R_sol    = sol.R_sol;
  double* Psi_sol  = sol.Psi_sol;
  double* dPsi_sol = sol.dPsi_sol;

  // fetch the spacetime parameters when R < R₁
  // obtained by assuming that the function and its first derivative are continuous
  double C[2] = {Psi_sol[0], dPsi_sol[0]};
  params_t p_BH = Get_p_BH(p, C);

  // create an interpolating function for Ψ(R₁<R<R₂)
  // uses a polynomial interpolator
  gsl_interp_accel *acc = gsl_interp_accel_alloc();                      // allocate the accelerator
  gsl_spline *spline = gsl_spline_alloc(gsl_interp_linear, N_points+1);  // allocate the interpolation object
  gsl_spline_init(spline, R_sol, Psi_sol, N_points+1);                   // initialize the interpolation object with points

  // iterate over the spatial grid
  for (CCTK_INT k = 0; k < cctk_lsh[2]; ++k) {
    for (CCTK_INT j = 0; j < cctk_lsh[1]; ++j) {
      for (CCTK_INT i = 0; i < cctk_lsh[0]; ++i) {

        const CCTK_INT ind = CCTK_GFINDEX3D(cctkGH, i, j, k);

        const CCTK_REAL x1 = x[ind] - x0;
        const CCTK_REAL y1 = y[ind] - y0;
        const CCTK_REAL z1 = z[ind] - z0;

        const CCTK_REAL R = sqrt(SQR(x1) + SQR(y1) + SQR(z1));

        CCTK_REAL lpsi;
        if (integrate_phi != 0) {
          if (R <= p.R1) {
            lpsi = Psi_RN(R, p_BH);
          }
          else if (p.R1 < R && R < p.R2) {
            lpsi = gsl_spline_eval(spline, R, acc);
          }
          else {
            lpsi = Psi_RN(R, p);
          }
        }
        else {
          lpsi = Psi_RN(R, p);
        }

        // the 3-metric
        gxx[ind] = pow(lpsi,4);
        gxy[ind] = 0;
        gxz[ind] = 0;
        gyy[ind] = pow(lpsi,4);
        gyz[ind] = 0;
        gzz[ind] = pow(lpsi,4);

        // the extrinsic curvature
        kxx[ind] = 0;
        kxy[ind] = 0;
        kxz[ind] = 0;
        kyy[ind] = 0;
        kyz[ind] = 0;
        kzz[ind] = 0;

        // the scalar field
        phi1[ind] = phi(R,p);
        phi2[ind] = 0;

        Kphi1[ind] = 0;
        Kphi2[ind] = 0;

        // the electric field
        if (convention_pi == 0) {
          Ex[ind] = pow(lpsi,-6) * p.Q/(sqrt(4.*M_PI)*pow(R,3)) * x1;
          Ey[ind] = pow(lpsi,-6) * p.Q/(sqrt(4.*M_PI)*pow(R,3)) * y1;
          Ez[ind] = pow(lpsi,-6) * p.Q/(sqrt(4.*M_PI)*pow(R,3)) * z1;
        }
        else {
          Ex[ind] = pow(lpsi,-6) * p.Q/pow(R,3) * x1;
          Ey[ind] = pow(lpsi,-6) * p.Q/pow(R,3) * y1;
          Ez[ind] = pow(lpsi,-6) * p.Q/pow(R,3) * z1;
        }

        Ax[ind] = 0;
        Ay[ind] = 0;
        Az[ind] = 0;

        Aphi[ind] = 0;

        Zeta[ind] = 0;

        // lapse
        if ( CCTK_EQUALS(initial_lapse, "psi^n") ) {
          alp[ind] = pow(lpsi, initial_lapse_psi_exponent);
        }

      }
    }
  }

  // free GSL interpolator objects
  gsl_spline_free(spline);
  gsl_interp_accel_free(acc);

  // free the solution arrays
  free(R_sol);
  free(Psi_sol);
  free(dPsi_sol);
}
