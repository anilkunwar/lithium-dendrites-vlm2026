/****************************************************************/
/* MOOSE - Multiphysics Object Oriented Simulation Environment  */
/*                                                              */
/*          All contents are licensed under LGPL V2.1           */
/*             See LICENSE for full restrictions                */
/****************************************************************/
#include "MKinetics.h"
registerMooseObject("newtApp", MKinetics);

InputParameters
MKinetics::validParams()
{
  InputParameters params = Kernel::validParams();
  params.addClassDescription("Add in Kinetics");
  params.addRequiredCoupledVar(
      "coupled_variables", "List of coupled variables");
  params.addRequiredParam<MaterialPropertyName>(
          "f_name", "Base name of the free energy function F defined in a DerivativeParsedMaterial");
  return params;
}

MKinetics::MKinetics(const InputParameters & parameters)
: DerivativeMaterialInterface<Kernel>(parameters),
  _F(getMaterialProperty<Real>("f_name")),
  _dFe(getMaterialPropertyDerivative<Real>("f_name", _var.name()))
{
  // Get the number of coupled variables
  unsigned int n = coupledComponents("coupled_variables");
  _coupled_vars.resize(n);
  _dFd_coupled.resize(n);
  
  // Initialize coupled variables and their derivatives
  for (unsigned int i = 0; i < n; ++i)
  {
    _coupled_vars[i] = coupled("coupled_variables", i);
    // Get the derivative of free energy with respect to each coupled variable
    _dFd_coupled[i] = &getMaterialPropertyDerivative<Real>("f_name", getVar("coupled_variables", i)->name());
  }
}

Real
MKinetics::computeQpResidual()
{
  return _F[_qp] * _test[_i][_qp];
}

Real
MKinetics::computeQpJacobian()
{
  return _dFe[_qp] * _phi[_j][_qp] * _test[_i][_qp];
}

Real
MKinetics::computeQpOffDiagJacobian(unsigned int jvar)
{
  // Loop through all coupled variables to find a match
  for (unsigned int i = 0; i < _coupled_vars.size(); ++i)
  {
    if (jvar == _coupled_vars[i])
    {
      return (*_dFd_coupled[i])[_qp] * _phi[_j][_qp] * _test[_i][_qp];
    }
  }
  
  return 0.0;
}
