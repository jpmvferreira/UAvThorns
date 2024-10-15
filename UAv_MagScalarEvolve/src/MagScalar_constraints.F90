
#include "cctk.h"
#include "cctk_Arguments.h"
#include "cctk_Parameters.h"
#include "cctk_Functions.h"

subroutine MagScalar_constraints( CCTK_ARGUMENTS )

  implicit none
  DECLARE_CCTK_ARGUMENTS
  DECLARE_CCTK_PARAMETERS

  ! Fundamental variables
  CCTK_REAL                alph, ch, lAphi
  CCTK_REAL                lE(3), lA(3)
  CCTK_REAL                lphi1, lphi2, lKphi1, lKphi2

  ! First derivatives
  CCTK_REAL                d1_ch(3)
  CCTK_REAL                d1_lE(3,3)

  ! Auxiliary variables
  CCTK_REAL                rho_e

  ! Constraints
  CCTK_REAL                gau

  ! Misc variables
  CCTK_REAL                dx12, dy12, dz12
  CCTK_REAL                odx60, ody60, odz60
  CCTK_INT                 i, j, k, a

  dx12 = 12*CCTK_DELTA_SPACE(1)
  dy12 = 12*CCTK_DELTA_SPACE(2)
  dz12 = 12*CCTK_DELTA_SPACE(3)

  odx60 = 1 / (60 * CCTK_DELTA_SPACE(1))
  ody60 = 1 / (60 * CCTK_DELTA_SPACE(2))
  odz60 = 1 / (60 * CCTK_DELTA_SPACE(3))

  ! convert ADM variables to BSSN-like ones
  call MagScalar_adm2bssn(CCTK_PASS_FTOF)

  ! make sure there are no uninitialised values anywhere
  gc    = 0

  !$OMP PARALLEL DO COLLAPSE(3) &
  !$OMP PRIVATE(alph, ch, lAphi, lE, lA, &
  !$OMP lphi1, lphi2, lKphi1, lKphi2,&
  !$OMP d1_lE, d1_ch,&
  !$OMP rho_e, &
  !$OMP i, j, k, a)
  do k = 1+cctk_nghostzones(3), cctk_lsh(3)-cctk_nghostzones(3)
  do j = 1+cctk_nghostzones(2), cctk_lsh(2)-cctk_nghostzones(2)
  do i = 1+cctk_nghostzones(1), cctk_lsh(1)-cctk_nghostzones(1)

    !------------ Get local variables ----------
    ch      = chi(i,j,k)

    alph    = alp(i,j,k)

    lE(1)   = Ex(i,j,k)
    lE(2)   = Ey(i,j,k)
    lE(3)   = Ez(i,j,k)

    lAphi   = Aphi(i,j,k)

    lA(1)   = Ax(i,j,k)
    lA(2)   = Ay(i,j,k)
    lA(3)   = Az(i,j,k)

    lphi1   = phi1(i,j,k)
    lphi2   = phi2(i,j,k)

    lKphi1  = Kphi1(i,j,k)
    lKphi2  = Kphi2(i,j,k)
    !-------------------------------------------

    ! to get rid of "may be used uninitialized"-type of warnings
    d1_ch = 0.0d0
    d1_lE = 0.0d0

    if (derivs_order == 4) then

      !------------- Centered 1st derivatives -----------

      ! d1_ch(3)
      d1_ch(1) = (   -chi(i+2,j,k) + 8*chi(i+1,j,k)                          &
                  - 8*chi(i-1,j,k) +   chi(i-2,j,k) ) / dx12

      d1_ch(2) = (   -chi(i,j+2,k) + 8*chi(i,j+1,k)                          &
                  - 8*chi(i,j-1,k) +   chi(i,j-2,k) ) / dy12

      d1_ch(3) = (   -chi(i,j,k+2) + 8*chi(i,j,k+1)                          &
                  - 8*chi(i,j,k-1) +   chi(i,j,k-2) ) / dz12


      ! d1_lE(3,3)
      d1_lE(1,1) = (   -Ex(i+2,j,k) + 8*Ex(i+1,j,k)               &
                    - 8*Ex(i-1,j,k) +   Ex(i-2,j,k) ) / dx12
      d1_lE(2,1) = (   -Ey(i+2,j,k) + 8*Ey(i+1,j,k)               &
                    - 8*Ey(i-1,j,k) +   Ey(i-2,j,k) ) / dx12
      d1_lE(3,1) = (   -Ez(i+2,j,k) + 8*Ez(i+1,j,k)               &
                    - 8*Ez(i-1,j,k) +   Ez(i-2,j,k) ) / dx12

      d1_lE(1,2) = (   -Ex(i,j+2,k) + 8*Ex(i,j+1,k)               &
                    - 8*Ex(i,j-1,k) +   Ex(i,j-2,k) ) / dy12
      d1_lE(2,2) = (   -Ey(i,j+2,k) + 8*Ey(i,j+1,k)               &
                    - 8*Ey(i,j-1,k) +   Ey(i,j-2,k) ) / dy12
      d1_lE(3,2) = (   -Ez(i,j+2,k) + 8*Ez(i,j+1,k)               &
                    - 8*Ez(i,j-1,k) +   Ez(i,j-2,k) ) / dy12

      d1_lE(1,3) = (   -Ex(i,j,k+2) + 8*Ex(i,j,k+1)               &
                    - 8*Ex(i,j,k-1) +   Ex(i,j,k-2) ) / dz12
      d1_lE(2,3) = (   -Ey(i,j,k+2) + 8*Ey(i,j,k+1)               &
                    - 8*Ey(i,j,k-1) +   Ey(i,j,k-2) ) / dz12
      d1_lE(3,3) = (   -Ez(i,j,k+2) + 8*Ez(i,j,k+1)               &
                    - 8*Ez(i,j,k-1) +   Ez(i,j,k-2) ) / dz12


      !--------------------------------------------------

    else if (derivs_order == 6) then

      !------------ Centered 1st derivatives -----
      ! d1_ch(3)
      d1_ch(1) = (  chi(i+3,j,k) - 9*chi(i+2,j,k) + 45*chi(i+1,j,k)          &
                  - chi(i-3,j,k) + 9*chi(i-2,j,k) - 45*chi(i-1,j,k) ) * odx60

      d1_ch(2) = (  chi(i,j+3,k) - 9*chi(i,j+2,k) + 45*chi(i,j+1,k)          &
                  - chi(i,j-3,k) + 9*chi(i,j-2,k) - 45*chi(i,j-1,k) ) * ody60

      d1_ch(3) = (  chi(i,j,k+3) - 9*chi(i,j,k+2) + 45*chi(i,j,k+1)          &
                  - chi(i,j,k-3) + 9*chi(i,j,k-2) - 45*chi(i,j,k-1) ) * odz60


      ! d1_lE(3,3)
      d1_lE(1,1) = (  Ex(i+3,j,k) - 9*Ex(i+2,j,k) + 45*Ex(i+1,j,k) &
                    - Ex(i-3,j,k) + 9*Ex(i-2,j,k) - 45*Ex(i-1,j,k) ) * odx60
      d1_lE(2,1) = (  Ey(i+3,j,k) - 9*Ey(i+2,j,k) + 45*Ey(i+1,j,k) &
                    - Ey(i-3,j,k) + 9*Ey(i-2,j,k) - 45*Ey(i-1,j,k) ) * odx60
      d1_lE(3,1) = (  Ez(i+3,j,k) - 9*Ez(i+2,j,k) + 45*Ez(i+1,j,k) &
                    - Ez(i-3,j,k) + 9*Ez(i-2,j,k) - 45*Ez(i-1,j,k) ) * odx60

      d1_lE(1,2) = (  Ex(i,j+3,k) - 9*Ex(i,j+2,k) + 45*Ex(i,j+1,k) &
                    - Ex(i,j-3,k) + 9*Ex(i,j-2,k) - 45*Ex(i,j-1,k) ) * ody60
      d1_lE(2,2) = (  Ey(i,j+3,k) - 9*Ey(i,j+2,k) + 45*Ey(i,j+1,k) &
                    - Ey(i,j-3,k) + 9*Ey(i,j-2,k) - 45*Ey(i,j-1,k) ) * ody60
      d1_lE(3,2) = (  Ez(i,j+3,k) - 9*Ez(i,j+2,k) + 45*Ez(i,j+1,k) &
                    - Ez(i,j-3,k) + 9*Ez(i,j-2,k) - 45*Ez(i,j-1,k) ) * ody60

      d1_lE(1,3) = (  Ex(i,j,k+3) - 9*Ex(i,j,k+2) + 45*Ex(i,j,k+1) &
                    - Ex(i,j,k-3) + 9*Ex(i,j,k-2) - 45*Ex(i,j,k-1) ) * odz60
      d1_lE(2,3) = (  Ey(i,j,k+3) - 9*Ey(i,j,k+2) + 45*Ey(i,j,k+1) &
                    - Ey(i,j,k-3) + 9*Ey(i,j,k-2) - 45*Ey(i,j,k-1) ) * odz60
      d1_lE(3,3) = (  Ez(i,j,k+3) - 9*Ez(i,j,k+2) + 45*Ez(i,j,k+1) &
                    - Ez(i,j,k-3) + 9*Ez(i,j,k-2) - 45*Ez(i,j,k-1) ) * odz60

    else
      call CCTK_WARN(0, "derivs_order not yet implemented.")
    end if


    !------------ Definition of Maxwell sources
    rho_e = -2 * q * (2 * lphi1 * lKphi2 - 2 * lphi2 * lKphi1 + q * lAphi * (lphi1*lphi1 + lphi2*lphi2))

    !------------ Gauss constraint -------------
    gau = alph * mu_V*mu_V * lAphi - alph * rho_e

    do a = 1, 3
       gau = gau + alph * d1_lE(a,a)                         &
                 - 1.5d0 * conf_fac_exponent * alph * lE(a) * d1_ch(a) / ch
    end do

    !------------ Write to grid functions ------
    gc(i,j,k) = gau

  end do
  end do
  end do
  !$OMP END PARALLEL DO

end subroutine MagScalar_constraints
