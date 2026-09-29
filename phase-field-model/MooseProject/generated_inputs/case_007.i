# hao tang Oct 2025

[GlobalParams]
  seed = 12345
[]

[Mesh]
  type = GeneratedMesh
  dim = 2
  nx = 50
  ny = 50
  xmin = 0
  xmax = 50
  ymin = 0
  ymax = 50
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
  [./delta_eta]
  [../]
  [./delta_c]
  [../]
  [./delta_pot]
  [../]
[]

[ICs]
  [./eta]
    variable = eta
    type = FunctionIC
    function = 'if(x>=0&x<=5,1,0)'
  [../]
  [./c]
    variable = c	# lithium ion concentration
    type = FunctionIC
    function = 'if(x>=0&x<=5,0.2,0.8)'
  [../]
[]

[BCs]
  [./left_pot]
    type = DirichletBC
    variable = 'pot'
    boundary = 'left'
    value = -0.377853
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
    value = 0.8
  [../]
[]

[Materials]

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
    prop_values = '1e-13 5e-13 0.256068 0.178611 3.24091e-10 1e7 1.19'
  [../]
  [./material_constants]
    type = GenericConstantMaterial
    prop_names  = 'fo Al As Bl Bs Cl Cs cleq cseq'	# j/mol
    prop_values = '0.0239009 7.56687 8.4421 3.97066 1.80896 2.90918 4.32453 0.354591 0.735069'
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
    expression = 'L2*dh*(eta*exp(pot*(1-alpha)*Faraday*valency/RT)-c*exp(-pot*alpha*Faraday*valency/RT))'
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
    amplitude = 1e-3
  [../]
  
  # evolution of pot ▽(σ▽φ)
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
  [./deta]
    type = DeltaUAux
    variable = delta_eta
    coupled_variable = eta   
    execute_on = timestep_end
  [../]
  [./dc]
    type = DeltaUAux
    variable = delta_c
    coupled_variable = c    
    execute_on = timestep_end
  [../]
  [./pot]
    type = DeltaUAux
    variable = delta_pot
    coupled_variable = pot    
    execute_on = timestep_end
  [../]
[]

[Executioner]
  type = Transient
  solve_type = 'NEWTON'
  
  petsc_options_iname = '-ksp_type -pc_type -pc_factor_mat_solver_type'
  petsc_options_value = 'preonly   lu        mumps'
  
  dtmax = 100
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
  [./eta_min]
    type = NodalExtremeValue
    variable = eta
    value_type = min
    execute_on = 'INITIAL TIMESTEP_END'
  [../]
  [./eta_max]
    type = NodalExtremeValue
    variable = eta
    value_type = max
    execute_on = 'INITIAL TIMESTEP_END'
  [../]
  [./c_min]
    type = NodalExtremeValue
    variable = c
    value_type = min
    execute_on = 'INITIAL TIMESTEP_END'
  [../]
  [./c_max]
    type = NodalExtremeValue
    variable = c
    value_type = max
    execute_on = 'INITIAL TIMESTEP_END'
  [../]
  [./pot_min]
    type = NodalExtremeValue
    variable = pot
    value_type = min
    execute_on = 'INITIAL TIMESTEP_END'
  [../]
  [./pot_max]
    type = NodalExtremeValue
    variable = pot
    value_type = max
    execute_on = 'INITIAL TIMESTEP_END'
  [../]
[]

[UserObjects]
  [./eta_min_m]
    type = Terminator
    expression = 'eta_min >= 1'
    fail_mode = HARD
    error_level = INFO
    message = 'eta_min_m says this should end'
    execute_on = TIMESTEP_END
  [../]
  [./eta_max_m]
    type = Terminator
    expression = 'eta_max <= 0'
    fail_mode = HARD
    error_level = INFO
    message = 'eta_max_m says this should end'
    execute_on = TIMESTEP_END
  [../]
  [./c_min_m]
    type = Terminator
    expression = 'abs(c_min) >= 1'
    fail_mode = HARD
    error_level = INFO
    message = 'c_min_m says this should end'
    execute_on = TIMESTEP_END
  [../]
  [./c_max_m]
    type = Terminator
    expression = 'c_max <= 0'
    fail_mode = HARD
    error_level = INFO
    message = 'c_max_m says this should end'
    execute_on = TIMESTEP_END
  [../]
[]

[Outputs]
  exodus = true
  time_step_interval = 10
  file_base = results/case_007/case_007
[]
