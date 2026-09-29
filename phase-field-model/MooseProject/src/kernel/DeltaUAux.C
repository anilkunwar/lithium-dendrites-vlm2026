#include "DeltaUAux.h"

registerMooseObject("newtApp", DeltaUAux);

InputParameters
DeltaUAux::validParams()
{
  InputParameters params = AuxKernel::validParams();
  params.addRequiredCoupledVar("coupled_variable", "The primary variable to compute Δu for");
  return params;
}

DeltaUAux::DeltaUAux(const InputParameters & parameters)
  : AuxKernel(parameters),
    _u(coupledValue("coupled_variable")),
    _u_old(coupledValueOld("coupled_variable"))
{
}

Real
DeltaUAux::computeValue()
{
  return _u[_qp] - _u_old[_qp];
}

