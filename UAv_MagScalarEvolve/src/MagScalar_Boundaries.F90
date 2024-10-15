#include "cctk.h"
#include "cctk_Arguments.h"
#include "cctk_Parameters.h"
#include "cctk_Functions.h"

subroutine MagScalar_Boundaries( CCTK_ARGUMENTS )

  implicit none
  DECLARE_CCTK_ARGUMENTS
  DECLARE_CCTK_PARAMETERS
  DECLARE_CCTK_FUNCTIONS

  CCTK_INT ierr

  CCTK_INT, parameter :: one = 1

  ! The outgoing (radiative) boundary conditions are being handled from calls to
  ! the NewRad infraestructure. Here we register all BCs as 'none', which
  ! enforces all the symmetry BCs.

  ierr = Boundary_SelectGroupForBC(cctkGH, CCTK_ALL_FACES, one, -one,      &
       "ProcaBase::Ei", "none")
  if (ierr < 0)                                                            &
       call CCTK_WARN(0, "Failed to register BC for ProcaBase::Ei!")

  ierr = Boundary_SelectGroupForBC(cctkGH, CCTK_ALL_FACES, one, -one,      &
       "ProcaBase::Ai", "none")
  if (ierr < 0)                                                            &
       call CCTK_WARN(0, "Failed to register BC for ProcaBase::Ai!")

  ierr = Boundary_SelectVarForBC(cctkGH, CCTK_ALL_FACES, one, -one,        &
       "ProcaBase::Aphi", "none")
  if (ierr < 0)                                                            &
       call CCTK_WARN(0, "Failed to register BC for ProcaBase::Aphi!")

  ierr = Boundary_SelectVarForBC(cctkGH, CCTK_ALL_FACES, one, -one,        &
       "ProcaBase::Zeta", "none")
  if (ierr < 0)                                                            &
       call CCTK_WARN(0, "Failed to register BC for ProcaBase::Zeta!")

  ierr = Boundary_SelectGroupForBC(cctkGH, CCTK_ALL_FACES, one, -one,      &
       "ScalarBase::phi", "none")
  if (ierr < 0)                                                            &
       call CCTK_WARN(0, "Failed to register BC for ScalarBase::phi!")

  ierr = Boundary_SelectGroupForBC(cctkGH, CCTK_ALL_FACES, one, -one,      &
       "ScalarBase::Kphi", "none")
  if (ierr < 0)                                                            &
       call CCTK_WARN(0, "Failed to register BC for ScalarBase::Kphi!")

end subroutine MagScalar_Boundaries
!
!=============================================================================
!
subroutine MagScalar_constraints_boundaries( CCTK_ARGUMENTS )

  implicit none
  DECLARE_CCTK_ARGUMENTS
  DECLARE_CCTK_PARAMETERS
  DECLARE_CCTK_FUNCTIONS

  CCTK_INT ierr
  CCTK_INT, parameter :: one = 1
  CCTK_INT bndsize

  if (derivs_order == 6) then
     bndsize = 5
  else if (derivs_order == 4) then
     bndsize = 3
  else
     call CCTK_ERROR("derivs_order not yet implemented.")
  end if

  ierr = Boundary_SelectGroupForBC(cctkGH, CCTK_ALL_FACES, bndsize, -one, &
       "MagScalarEvolve::gauss", "flat")
  if (ierr < 0)                                                           &
       call CCTK_ERROR("Failed to register BC for MagScalarEvolve::gauss!")

end subroutine MagScalar_constraints_boundaries
