/****************************************************************/
/* MOOSE - Multiphysics Object Oriented Simulation Environment  */
/*                                                              */
/*          All contents are licensed under LGPL V2.1           */
/*             See LICENSE for full restrictions                */
/****************************************************************/
#ifndef MKINETICS_H
#define MKINETICS_H

#include "Kernel.h"
#include "DerivativeMaterialInterface.h"

class MKinetics: public DerivativeMaterialInterface<Kernel>
{
public:
  MKinetics(const InputParameters & parameters);
  
  static InputParameters validParams();

protected:
  virtual Real computeQpResidual();
  virtual Real computeQpJacobian();
  virtual Real computeQpOffDiagJacobian(unsigned int jvar);

private:
  /// Free energy function
  const MaterialProperty<Real> & _F;
  /// Derivative of free energy with respect to the current variable
  const MaterialProperty<Real> & _dFe;
  
  /// List of coupled variable indices
  std::vector<unsigned int> _coupled_vars;
  /// List of derivatives of free energy with respect to coupled variables
  std::vector<const MaterialProperty<Real> *> _dFd_coupled;
};

#endif // KINETICS_H
