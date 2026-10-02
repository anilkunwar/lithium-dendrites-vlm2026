# hao tang Oct 2025

[GlobalParams]
  seed = 12345
[]

[Mesh]
  type = GeneratedMesh
  dim = 2
  nx = 100
  ny = 100
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
[]

[ICs]
  [./eta]
    variable = eta
    type = FunctionIC
    function = 'if(x>=0&x<=10,1,0)'
  [../]
  [./c]
    variable = c	# lithium ion concentration
    type = FunctionIC
    function = 'if(x>=0&x<=10,0,1)'
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
  # Conservation diagnostic for the equation actually solved:
  # d/dt(c + (c_s/c_0)*eta) + div(J) = 0.
  # J = -Deff*(F_cc*grad(c) + F_ceta*grad(eta)) - Deffe*grad(pot).
  # Inventories below are normalized integrals over model coordinates.
  [./li_total_density]
    type = ParsedMaterial
    property_name = li_total_density
    expression = 'c+ft*eta'
    material_property_names = 'ft'
    coupled_variables = 'c eta'
  [../]
  [./li_flux_c_coefficient]
    type = ParsedMaterial
    property_name = li_flux_c_coefficient
    expression = 'Deff*Fcc'
    material_property_names = 'Deff Fcc:=D[F,c,c]'
    coupled_variables = 'c eta'
  [../]
  [./li_flux_eta_coefficient]
    type = ParsedMaterial
    property_name = li_flux_eta_coefficient
    expression = 'Deff*Fceta'
    material_property_names = 'Deff Fceta:=D[F,c,eta]'
    coupled_variables = 'c eta'
  [../]


  [./scale]
    type = GenericConstantMaterial
    prop_names = 'length_scale energy_scale time_scale'
    prop_values = '1e6 6.24150943e18 1e3'
  [../]
  [./constants]
    type = GenericConstantMaterial
    prop_names  = 'c_s c_0 valency molar_vol Faraday RT alpha'
    prop_values = '7.69e4 1e3 1 2.2e-5 96485 2494.2 0.5'
  [../]
  [./system_constants]
    type = GenericConstantMaterial
    prop_names  = 'Meo Mso L1o L2o ko S1o S2o'
    prop_values = '1e-13 5e-13 $L1o $L2o $ko 1e7 1.19'
  [../]
  [./material_constants]
    type = GenericConstantMaterial
    prop_names  = 'fo Al As Bl Bs Cl Cs cleq cseq'	# j/mol
    prop_values = '$fo $Al $As $Bl $Bs $Cl $Cs $cleq $cseq'
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
  # free energies
  [./fl]
    type = DerivativeParsedMaterial
    property_name = fl
    material_property_names = 'fo Al Bl Cl cleq length_scale energy_scale molar_vol'
    coupled_variables = 'c'
    expression = '(energy_scale/(length_scale)^3)*(Al*(c-cleq)^2 + Bl*(c-cleq) + Cl)*fo/molar_vol'
    outputs = exodus
    derivative_order = 3
  [../]
  [./fs]
    type = DerivativeParsedMaterial
    property_name = fs
    material_property_names = 'fo As Bs Cs cseq length_scale energy_scale molar_vol'
    coupled_variables = 'c'
    expression = '(energy_scale/(length_scale)^3)*(As*(c-cseq)^2 + Bs*(c-cseq) + Cs)*fo/molar_vol'
    outputs = exodus
    derivative_order = 3
  [../]
  [./free_energy]
    type = DerivativeTwoPhaseMaterial
    property_name = F
    fa_name = fl
    fb_name = fs
    eta = eta
    coupled_variables = 'c'
    W = 2e3
    outputs = exodus
    derivative_order = 3
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
    expression = 'valency*Faraday*c_s/(length_scale^3)'
    property_name = ChargeEff
    material_property_names = 'valency Faraday length_scale c_s'
    outputs = exodus
  [../]

[]

[Kernels]
  #
  # Cahn-Hilliard Equation
  #
  [./dcdt]
    type = TimeDerivative
    variable = c
  [../]
  # Intrinsic diffusion part of equation 3 in main text.
  [./ch]
    type = CahnHilliard
    variable = c
    f_name = F
    mob_name = Deff
    coupled_variables = 'eta'
  [../]
  [./elec]
  	type = MatDiffusion
    variable = c
    v = pot
    diffusivity = Deffe
    args = 'eta c'
  [../]
  [./cSource]
  	type = CoupledSusceptibilityTimeDerivative
    variable = c
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
    coupled_variables = 'eta c'
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
[]

[Executioner]
  type = Transient
  solve_type = 'NEWTON'
  scheme = implicit-euler

  petsc_options_iname = '-ksp_type -pc_type -pc_factor_mat_solver_type -snes_type'
  petsc_options_value = 'preonly   lu        mumps                      vinewtonrsls'

  dtmax = 1
  end_time = 5E3

  [./TimeStepper]
    type = IterationAdaptiveDT
    dt = 1E-3
    growth_factor = 1.1
  [../]

[./Adaptivity]
   interval = 5
   initial_adaptivity = 4
   refine_fraction = 0.8
   coarsen_fraction = 0.1
   max_h_level = 2
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
  # Total lithium inventory:
  # M_Li = integral(c + ft*eta)dOmega
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

  # Boundary lithium inflow rate (internal dependencies)
  [./li_right_out_c]
    outputs = none
    type = SideDiffusiveFluxIntegral
    variable = c
    diffusivity = li_flux_c_coefficient
    boundary = right
    execute_on = 'INITIAL TIMESTEP_END'
  [../]

  [./li_right_out_eta]
    outputs = none
    type = SideDiffusiveFluxIntegral
    variable = eta
    diffusivity = li_flux_eta_coefficient
    boundary = right
    execute_on = 'INITIAL TIMESTEP_END'
  [../]

  [./li_right_out_electric]
    outputs = none
    type = SideDiffusiveFluxIntegral
    variable = pot
    diffusivity = Deffe
    boundary = right
    execute_on = 'INITIAL TIMESTEP_END'
  [../]

  [./li_right_in_rate]
    type = ParsedPostprocessor
    pp_names = 'li_right_out_c li_right_out_eta li_right_out_electric'
    expression = '-(li_right_out_c+li_right_out_eta+li_right_out_electric)'
    execute_on = 'INITIAL TIMESTEP_END'
  [../]

  [./li_net_in]
    type = TimeIntegratedPostprocessor
    value = li_right_in_rate
    time_integration_scheme = implicit-euler
    execute_on = 'INITIAL TIMESTEP_END'
  [../]

  [./li_balance_error]
    type = ParsedPostprocessor
    pp_names = 'li_total li_initial li_net_in'
    expression = 'li_total-li_initial-li_net_in'
    execute_on = 'INITIAL TIMESTEP_END'
  [../]

  [./li_balance_error_rel]
    type = ParsedPostprocessor
    pp_names = 'li_balance_error li_initial'
    expression = 'abs(li_balance_error)/max(abs(li_initial),1e-16)'
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
[]

[Outputs]
  file_base = results/case_003_physics/case_001
  # Save every accepted step so field-based checks are possible.
  [./exodus]
    type = Exodus
    time_step_interval = 1
    execute_on = 'INITIAL TIMESTEP_END FINAL'
  [../]
  [./monitor]
    type = CSV
    time_step_interval = 1
    execute_on = 'INITIAL TIMESTEP_END FINAL'
  [../]
[]

