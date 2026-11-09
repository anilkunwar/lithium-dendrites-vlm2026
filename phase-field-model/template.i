[GlobalParams]
  seed = 12345
[]

[Mesh]
  type = GeneratedMesh
  dim = 2
  # 0.5 um elements, no adaptivity: the interface (0.1 < eta < 0.9) spans >= 5 elements
  # for ko >= 1.5e-6 J/m (benchmarked; generate_moose_simulation_files.py uses ko >= 2e-6).
  nx = 200
  ny = 200
  xmin = 0
  xmax = 100
  ymin = 0
  ymax = 100
  elem_type = QUAD4
[]

[Variables]
  # order parameter
  [./eta]
    order = FIRST
    family = LAGRANGE
  [../]

  # solute concentration
  [./c]
    order = FIRST
    family = LAGRANGE
  [../]

  # electric overpotential
  [./pot]
    order = FIRST
    family = LAGRANGE
  [../]

[]

[AuxVariables]
  [./bounds_dummy]
    order = FIRST
    family = LAGRANGE
  [../]

  # eta=0.5 interface tip location for stopping criterion
  [./tip_coordinate]
    order = FIRST
    family = LAGRANGE
  [../]

  # Interface diagnostics: indicator of 0.1 < eta < 0.9, and x of electrolyte nodes (eta < 0.5)
  # whose minimum is the deepest point of the front.
  [./in_interface]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./trough_coordinate]
    order = FIRST
    family = LAGRANGE
  [../]

  # Residual of the c equation per node (save_in of every c kernel). It is zero
  # wherever c is solved and equals the lithium inflow where c is prescribed.
  [./c_residual]
    order = FIRST
    family = LAGRANGE
  [../]
[]

[Bounds]
  [./c_lower]
    type = ConstantBounds
    variable = bounds_dummy
    bounded_variable = c
    bound_type = lower
    bound_value = 0
  [../]
  [./eta_lower]
    type = ConstantBounds
    variable = bounds_dummy
    bounded_variable = eta
    bound_type = lower
    bound_value = 0
  [../]
  [./eta_upper]
    type = ConstantBounds
    variable = bounds_dummy
    bounded_variable = eta
    bound_type = upper
    bound_value = 1
  [../]
  # pot stays between the electrode potential and the electrolyte potential (with a margin);
  # unbounded Newton updates of pot overflow the Butler-Volmer exponentials.
  [./pot_lower]
    type = ConstantBounds
    variable = bounds_dummy
    bounded_variable = pot
    bound_type = lower
    bound_value = ${fparse $POT_LEFT - 0.1}
  [../]
  [./pot_upper]
    type = ConstantBounds
    variable = bounds_dummy
    bounded_variable = pot
    bound_type = upper
    bound_value = 0.1
  [../]
[]

[ICs]
  # Equilibrium (tanh) interface at x = 10 instead of a step. A step relaxes by moving eta on
  # the electrode side, where c ~ 0; the c >= 0 bound then clips the deficit and creates Li.
  # eta = 0.5*(1-tanh(x/(2*l))) with l = sqrt(ko/(2*W)) = 1e6*sqrt(ko/4.448e6) um (W = 2.224e6 J/m^3);
  # its 0.1-0.9 width is 4.39*l.
  [./eta]
    variable = eta
    type = FunctionIC
    function = '0.5*(1-tanh((x-10)/(2e6*sqrt($ko/4.448e6))))'
  [../]
  [./c]
    variable = c	# lithium ion concentration
    type = FunctionIC
    function = '0.5*(1+tanh((x-10)/(2e6*sqrt($ko/4.448e6))))'
  [../]
[]

[BCs]
  [./left_pot]
    type = DirichletBC
    variable = 'pot'
    boundary = 'left'
    value = $POT_LEFT
  [../]
  [./right_pot]
    type = DirichletBC
    variable = 'pot'
    boundary = 'right'
    value = 0
  [../]
  [./right_comp]	# connected to the electrolyte
    type = DirichletBC
    variable = 'c'
    boundary = 'right'
    value = 1
  [../]
[]

[Materials]
  # Lithium density c + (c_s/c_0)*eta for the balance check
  [./li_total_density]
    type = ParsedMaterial
    property_name = li_total_density
    expression = 'c+ft*eta'
    material_property_names = 'ft'
    coupled_variables = 'c eta'
  [../]


  [./scale]
    type = GenericConstantMaterial
    prop_names = 'length_scale energy_scale time_scale'
    prop_values = '1e6 6.24150943e18 1'
  [../]
  [./constants]
    type = GenericConstantMaterial
    prop_names  = 'c_s c_0 valency molar_vol Faraday RT alpha'
    prop_values = '7.64e4 1e3 1 2.2e-5 96485 2494.2 0.5'
  [../]
  [./system_constants]
    type = GenericConstantMaterial
    # Meo, Mso: mobilities D/RT with D = 3.197e-10 m^2/s in both phases, as in Hong (the Li+ flux
    # vanishes in the metal through c -> 0). With D_e = 7.5e-13 (Chen) the Li consumed in the
    # metal-side half of the interface cannot be resupplied: c is clipped at 0 and 20-60 % of the
    # deposited Li is created by the bound.
    # L1o = L_sigma (m^3/(J s)), L2o = L_eta (1/s), S1o, S2o = sigma_e, sigma_l (S/m).
    # L_eta = 1/s of Chen et al. makes the Butler-Volmer term outgrow the double well for any
    # eta > 3e-5, so the whole electrolyte turns into a dilute deposit; Hong's 0.001/s does not.
    prop_names  = 'Meo Mso L1o L2o ko S1o S2o'
    prop_values = '1.28178e-13 1.28178e-13 2.5e-6 1e-3 $ko 1e7 1.19'
  [../]
  [./L1]
    type = ParsedMaterial
    expression = '((length_scale)^3/(energy_scale*time_scale))*L1o'
    property_name = L1
    material_property_names = 'L1o length_scale energy_scale time_scale'
    outputs = exodus
  [../]
  [./L2]
    type = ParsedMaterial
    expression = 'L2o/time_scale'
    property_name = L2
    material_property_names = 'L2o time_scale'
    outputs = exodus
  [../]
  [./kappa_isotropy]
    type = ParsedMaterial
    property_name = kappa
    expression = '(energy_scale/length_scale)*ko'
    material_property_names = 'ko length_scale energy_scale'
    outputs = exodus
  [../]
  # h(eta)
  [./h]
    type = SwitchingFunctionMaterial
    h_order = HIGH
    eta = eta
    outputs = exodus
  [../]
  # g(eta)
  [./g]
    type = BarrierFunctionMaterial
    g_order = SIMPLE
    eta = eta
  [../]
  # Bulk free energy: double well only (no fl/fs, hence no chemical driving force).
  # The ion part, RT*c*(ln(c)-1), enters through the Nernst-Planck kernels below.
  [./free_energy]
    type = DerivativeParsedMaterial
    property_name = F
    coupled_variables = 'eta'
    # W = 12*gamma/delta = 2.224e6 J/m^3, in eV/um^3
    constant_names = 'W'
    constant_expressions = '2.224e6*6.24150943'
    expression = 'W*eta^2*(1-eta)^2'
    derivative_order = 2
  [../]
  # BV driving force
  [./Butlervolmer]
    type = DerivativeParsedMaterial
    expression = 'L2*dh*(exp(pot*(1-alpha)*Faraday*valency/RT)-c*exp(-pot*alpha*Faraday*valency/RT))'
    coupled_variables = 'pot c eta'
    property_name = f_bv
    material_property_names = 'L2 alpha Faraday valency RT dh:=D[h,eta]'
    outputs = exodus
    derivative_order = 1
  [../]
  # diffusion for c
  [./Deff]
    type = DerivativeParsedMaterial
    expression = '(length_scale)^2/time_scale*(Meo*h+Mso*(1-h))'
    coupled_variables = 'eta'
    property_name = Deff
    material_property_names = 'Meo Mso h(eta) length_scale time_scale'
    outputs = exodus
    derivative_order = 1
  [../]
  # Fick diffusivity of the dilute solution: D = M*RT (M = Deff above is a mobility).
  [./Dfick]
    type = DerivativeParsedMaterial
    expression = 'Deff*RT'
    coupled_variables = 'eta'
    property_name = Dfick
    material_property_names = 'Deff(eta) RT'
    outputs = exodus
    derivative_order = 1
  [../]
  [./Deffe]
    type = DerivativeParsedMaterial
    expression = '(length_scale)^2/time_scale*(Meo*h+Mso*(1-h))*c*valency*Faraday'
    coupled_variables = 'eta c'
    property_name = Deffe
    material_property_names = 'Meo Mso h(eta) length_scale time_scale valency Faraday'
    outputs = exodus
    derivative_order = 1
  [../]
  [./coupled_eta_function]
    type = ParsedMaterial
    expression = 'c_s/c_0'	# normalize
    property_name = ft
    material_property_names = 'c_s c_0'
    outputs = exodus
  [../]
  # conduction for pot
  [./ElecEff]
    type = DerivativeParsedMaterial
    expression = '(S1o*h+S2o*(1-h))/length_scale'
    coupled_variables = 'eta'
    property_name = ElecEff
    material_property_names = 'S1o S2o h(eta) length_scale'
    outputs = exodus
  [../]
  [./ChargeEff]
    type = ParsedMaterial
    # time_scale converts d(eta)/dt from model time to seconds.
    expression = 'valency*Faraday*c_s*time_scale/(length_scale^3)'
    property_name = ChargeEff
    material_property_names = 'valency Faraday length_scale c_s time_scale'
    outputs = exodus
  [../]

[]

[Kernels]
  #
  # Nernst-Planck equation for c
  #
  [./dcdt]
    type = TimeDerivative
    variable = c
    save_in = c_residual
  [../]
  # Fickian part of the Nernst-Planck flux (dilute-solution chemical potential RT*ln(c)).
  [./diff]
    type = MatDiffusion
    variable = c
    diffusivity = Dfick
    args = 'eta'
    save_in = c_residual
  [../]
  [./elec]
  	type = MatDiffusion
    variable = c
    save_in = c_residual
    v = pot
    diffusivity = Deffe
    args = 'eta c'
  [../]
  [./cSource]
  	type = CoupledSusceptibilityTimeDerivative
    variable = c
    save_in = c_residual
    v = eta
    f_name = ft
  [../]

  # Allen-Cahn Equation
  #
  [./detadt]
    type = TimeDerivative
    variable = eta
  [../]
  [./ac]
    type = AllenCahn
    variable = eta
  	f_name = F
  	mob_name = L1
  [../]
  [./ACInterface]
    type = ACInterface
    variable = eta
    kappa_name = kappa
    mob_name = L1
  [../]
  [./BV]
	type = MKinetics
	variable = eta
	f_name = f_bv
	coupled_variables = 'c pot eta'
  [../]
  [./noise_interface]
    type = LangevinNoise
    variable = eta
    multiplier = dh/deta
    amplitude = $Noise
  [../]

  # evolution of pot
  [./Cond]
    type = MatDiffusion
    variable = pot
    diffusivity = ElecEff
    args = 'eta'
  [../]
  # -nFcs(deta/dt)
  [./coupledSource]
    type = CoupledSusceptibilityTimeDerivative
    variable = pot
    v = eta
    f_name = ChargeEff
  [../]
[]


[AuxKernels]
  [./tip_coordinate]
    type = ParsedAux
    variable = tip_coordinate
    coupled_variables = 'eta'
    use_xyzt = true
    expression = 'if(eta>=0.5,x,-1)'
    execute_on = 'INITIAL TIMESTEP_END'
  [../]
  [./in_interface]
    type = ParsedAux
    variable = in_interface
    coupled_variables = 'eta'
    expression = 'if(eta>0.1&eta<0.9,1,0)'
    execute_on = 'INITIAL TIMESTEP_END'
  [../]
  [./trough_coordinate]
    type = ParsedAux
    variable = trough_coordinate
    coupled_variables = 'eta'
    use_xyzt = true
    expression = 'if(eta<0.5,x,1e3)'
    execute_on = 'INITIAL TIMESTEP_END'
  [../]
[]

[Executioner]
  type = Transient
  solve_type = 'NEWTON'
  scheme = implicit-euler
  # The default line search of vinewtonrsls stalls on the first steps of this formulation.
  line_search = none

  petsc_options_iname = '-ksp_type -pc_type -pc_factor_mat_solver_type -snes_type'
  petsc_options_value = 'preonly   lu        mumps                      vinewtonrsls'

  dtmax = 1
  end_time = 400

  [./TimeStepper]
    type = IterationAdaptiveDT
    dt = 1E-3
    growth_factor = 1.1
  [../]
[]

#
# Precondition using handcoded off-diagonal terms
#
[Preconditioning]
  [./full]
    type = SMP
    full = true
  [../]
[]

[Postprocessors]
  # Lithium balance: integral(c + ft*eta) - initial = time integral of the inflow at the right
  # boundary, taken from the residual of the c equation at the prescribed-c nodes.
  [./li_total]
    type = ElementIntegralMaterialProperty
    mat_prop = li_total_density
    execute_on = 'INITIAL TIMESTEP_END'
  [../]
  [./li_initial]
    type = ElementIntegralMaterialProperty
    mat_prop = li_total_density
    execute_on = 'INITIAL'
  [../]
  [./li_in_rate]
    type = NodalSum
    variable = c_residual
    boundary = right
    execute_on = 'INITIAL TIMESTEP_END'
    outputs = none
  [../]
  [./li_net_in]
    type = TimeIntegratedPostprocessor
    value = li_in_rate
    time_integration_scheme = implicit-euler
    execute_on = 'INITIAL TIMESTEP_END'
  [../]
  # Relative to the inventory change; the floor avoids dividing by ~0 at the start.
  [./li_balance_error]
    type = ParsedPostprocessor
    pp_names = 'li_total li_initial li_net_in'
    expression = 'abs(li_total-li_initial-li_net_in)/max(abs(li_total-li_initial),1e-3*abs(li_initial))'
    execute_on = 'INITIAL TIMESTEP_END'
  [../]

  # Dendrite tip tracking
  [./tip_x_nodal]
    type = NodalExtremeValue
    variable = tip_coordinate
    value_type = max
    execute_on = 'INITIAL TIMESTEP_END'
    force_postaux = true
  [../]

  # Front shape. For a planar front interface_width is the 0.1-0.9 width and
  # front_roughness (tip minus deepest point of the eta = 0.5 front) is ~ one element.
  [./interface_area]
    type = ElementIntegralVariablePostprocessor
    variable = in_interface
    execute_on = 'INITIAL TIMESTEP_END'
    outputs = none
  [../]
  [./interface_width]
    type = ParsedPostprocessor
    pp_names = 'interface_area'
    expression = 'interface_area/100'
    execute_on = 'INITIAL TIMESTEP_END'
  [../]
  [./trough_x_nodal]
    type = NodalExtremeValue
    variable = trough_coordinate
    value_type = min
    execute_on = 'INITIAL TIMESTEP_END'
    force_postaux = true
  [../]
  [./front_roughness]
    type = ParsedPostprocessor
    pp_names = 'tip_x_nodal trough_x_nodal'
    expression = 'tip_x_nodal-trough_x_nodal'
    execute_on = 'INITIAL TIMESTEP_END'
  [../]
  [./mean_front_x]
    type = ElementIntegralVariablePostprocessor
    variable = eta
    execute_on = 'INITIAL TIMESTEP_END'
    outputs = none
  [../]
  [./front_x]
    type = ParsedPostprocessor
    pp_names = 'mean_front_x'
    expression = 'mean_front_x/100'
    execute_on = 'INITIAL TIMESTEP_END'
  [../]

  [./tip_gap_nodal]
    type = ParsedPostprocessor
    pp_names = 'tip_x_nodal'
    expression = '100-tip_x_nodal'
    execute_on = 'INITIAL TIMESTEP_END'
  [../]
[]

[UserObjects]
  [./stop_at_right]
    type = Terminator
    expression = 'tip_x_nodal >= 0 & tip_gap_nodal <= 10'
    fail_mode = HARD
    error_level = INFO
    message = 'Nodal eta>=0.5 tip is within 10 units of x=100.'
    execute_on = TIMESTEP_END
    execution_order_group = 1
  [../]

  [./stop_at_left]
    type = Terminator
    expression = 'tip_x_nodal <= 5'
    fail_mode = HARD
    error_level = INFO
    message = 'Nodal eta>=0.5 tip has retreated to x<=5, or the solid phase has disappeared.'
    execute_on = TIMESTEP_END
    execution_order_group = 1
  [../]
[]

[Outputs]
  file_base = results/$CASE/$CASE
  # Save every accepted step so field-based checks are possible.
  [./exodus]
    type = Exodus
    time_step_interval = 10
    execute_on = 'INITIAL TIMESTEP_END FINAL'
  [../]
  [./monitor]
    type = CSV
    time_step_interval = 10
    execute_on = 'INITIAL TIMESTEP_END FINAL'
  [../]
[]