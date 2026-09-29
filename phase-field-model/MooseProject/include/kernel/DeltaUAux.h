#pragma once

#include "AuxKernel.h"

/**
 * DeltaUAux for Δu = u - u_old
 */
class DeltaUAux : public AuxKernel
{
public:
  static InputParameters validParams();
  DeltaUAux(const InputParameters & parameters);

protected:
  virtual Real computeValue() override;

private:
  const VariableValue & _u;
  const VariableValue & _u_old;
};

