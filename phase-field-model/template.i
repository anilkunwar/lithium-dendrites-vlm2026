# hao tang Oct 2025
#
# Li electrodeposition phase-field model (eta = 1 Li metal, eta = 0 electrolyte).
# Model units: length um, time ms, energy eV. c is the Li+ concentration
# normalized by c_0; pot is the electric potential in V.
#
# Placeholders: $POT_LEFT $Noise $kappa (J/m) $W (J/m^3) $CASE
# ($kappa and $W are derived from sigma and delta by generate_moose_simulation_files.py)
#
# ---------------------------------------------------------------------------
# Concentration equation (derived from the free energy, not ideal Nernst-Planck)
#   Total free energy density: F(c,eta) = h*fs(c) + (1-h)*fl(c) + W*g(eta)   [J/m^3]
#   Electrochemical potential per mol Li+: mu_e = (1/c_0) dF/dc + n*Faraday*pot
#   Flux (mol/m^2/s): J = -(D*C/RT) grad(mu_e), C = c_0*c
#   => dc/dt + (c_s/c_0) deta/dt = div[ M grad(dF/dc) + (D*c*n*Faraday/RT) grad(pot) ]
#      with M = D*c/(RT*c_0).
#   With fl = Al*(c-1)^2, chemical diffusivity in the liquid = D*c*2*Al/(c_0*RT).
#
# Interface (double well W*g, g = eta^2(1-eta)^2, gradient energy kappa/2|grad eta|^2)
#   sigma = sqrt(2*W*kappa)/6,  delta(10-90%) = 2.2*sqrt(2*kappa/W)
#   sigma, delta given => kappa = 3*sigma*delta/2.2,  W = 13.2*sigma/delta
#   (sigma = 0.5 J/m^2, delta = 2.5 um: kappa = 1.70e-6 J/m, W = 2.64e6 J/m^3)
# ---------------------------------------------------------------------------

[GlobalParams]
  seed = 12345
[]

[Mesh]
  # fixed mesh, dx = 0.5 um
  type = GeneratedMesh
  dim = 2
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

  # electric potential
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

  # eta>=0.5 interface location for the stopping criterion
  [./tip_coordinate]
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
[]

[ICs]
  [./eta]
    variable = eta
    type = FunctionIC
    function = 'if(x>=0&x<=10,1,0)'
  [../]
  [./c]
    variable = c    # lithium ion concentration
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
  [./right_comp]    # connected to the electrolyte
    type = DirichletBC
    variable = 'c'
    boundary = 'right'
    value = 1
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
  # De, Dl: Li+ diffusivity in electrode / electrolyte (m^2/s)
  # L1o: interface mobility (m^3/(J s)); L2o: reaction (Butler-Volmer) rate constant (1/s)
  #   L1o*kappa*curvature is the curvature-driven interface speed; with L1o ~ 0.1 it reaches
  #   ~0.4 m/s at 1 um radius, far above the transport-limited deposition speed, and smooths
  #   every finger. L2o ~ 0.4 /s lets the BV term outgrow the double well at eta ~ 1e-3, so the
  #   electrolyte metallizes. 2.5e-6 and 1e-3 follow Hong & Viswanathan 2018.
  # S1o, S2o: conductivity of Li metal / electrolyte (S/m)
  [./system_constants]
    type = GenericConstantMaterial
    prop_names  = 'De Dl L1o L2o S1o S2o'
    prop_values = '7.5e-13 3.0e-10 2.5e-6 1e-3 1e7 1.19'
  [../]
  # ko: gradient energy coefficient (J/m); Wo: double-well barrier height (J/m^3)
  # Al, As: free energy curvature (J/m^3); Al = c_0*RT/2 matches the ideal-solution
  # curvature at c = 1, so the chemical diffusivity there equals Dl.
  # As << Al: with c following the local equilibrium across the interface, the bulk terms add
  # up to ~0.5*As*Al/(As+Al) to the interface barrier (3.1e5 J/m^3 for As = Al). As = 1e4 keeps
  # it at 8e3 J/m^3, small next to W/16, so sigma and delta are set by kappa and W. As only acts
  # in the metal, where c ~ 0 and De is negligible, so it does not affect transport.
  # Bl = Bs: no chemical driving force at the equilibrium compositions; the
  # deposition driving force comes only from the overpotential (Butler-Volmer).
  [./thermo_constants]
    type = GenericConstantMaterial
    prop_names  = 'ko Wo Al As cleq cseq Bl Bs'
    prop_values = '$kappa $W 1.25e6 1e4 1 0 0 0'
  [../]

  # ---- derived parameters (converted to model units) ----
  [./L1]
    type = ParsedMaterial
    expression = '((length_scale)^3/(energy_scale*time_scale))*L1o'
    property_name = L1
    material_property_names = 'L1o length_scale energy_scale time_scale'
  [../]
  [./L2]
    type = ParsedMaterial
    expression = 'L2o/time_scale'
    property_name = L2
    material_property_names = 'L2o time_scale'
  [../]
  [./kappa_isotropy]
    type = ParsedMaterial
    property_name = kappa
    expression = '(energy_scale/length_scale)*ko'
    material_property_names = 'ko length_scale energy_scale'
  [../]
  [./barrier_height]
    type = ParsedMaterial
    property_name = W
    expression = '(energy_scale/(length_scale)^3)*Wo'
    material_property_names = 'Wo length_scale energy_scale'
  [../]

  # h(eta)
  [./h]
    type = SwitchingFunctionMaterial
    h_order = HIGH
    eta = eta
  [../]
  # g(eta)
  [./g]
    type = BarrierFunctionMaterial
    g_order = SIMPLE
    eta = eta
  [../]

  # ---- free energies ----
  [./fl]
    type = DerivativeParsedMaterial
    property_name = fl
    material_property_names = 'Al Bl cleq length_scale energy_scale'
    coupled_variables = 'c'
    expression = '(energy_scale/(length_scale)^3)*(Al*(c-cleq)^2 + Bl)'
    derivative_order = 3
  [../]
  [./fs]
    type = DerivativeParsedMaterial
    property_name = fs
    material_property_names = 'As Bs cseq length_scale energy_scale'
    coupled_variables = 'c'
    expression = '(energy_scale/(length_scale)^3)*(As*(c-cseq)^2 + Bs)'
    derivative_order = 3
  [../]
  # Written explicitly: DerivativeTwoPhaseMaterial returns a wrong d3F/dc2deta.
  [./free_energy]
    type = DerivativeParsedMaterial
    property_name = F
    coupled_variables = 'c eta'
    material_property_names = 'fl(c) fs(c) h(eta) g(eta) W'
    expression = 'h*fs + (1-h)*fl + W*g'
    derivative_order = 3
  [../]

  # BV driving force
  [./Butlervolmer]
    type = DerivativeParsedMaterial
    expression = 'L2*dh*(exp(pot*(1-alpha)*Faraday*valency/RT)-c*exp(-pot*alpha*Faraday*valency/RT))'
    coupled_variables = 'pot c eta'
    property_name = f_bv
    material_property_names = 'L2 alpha Faraday valency RT dh:=D[h,eta]'
    derivative_order = 1
  [../]

  # ---- transport of c ----
  # D(eta) in um^2/ms
  [./D]
    type = DerivativeParsedMaterial
    expression = '(length_scale)^2/time_scale*(De*h+Dl*(1-h))'
    coupled_variables = 'eta'
    property_name = D
    material_property_names = 'De Dl h(eta) length_scale time_scale'
    derivative_order = 2
  [../]
  # M = D*c/(RT*c_0) (m^5/(J s)), converted to um^5/(eV ms) so that div(M grad(dF/dc)) is in 1/ms
  [./M]
    type = DerivativeParsedMaterial
    expression = 'D*c/(RT*c_0)*(length_scale)^3/energy_scale'
    coupled_variables = 'c eta'
    property_name = M
    material_property_names = 'D(eta) RT c_0 length_scale energy_scale'
    derivative_order = 2
  [../]
  # electromigration coefficient D*c*n*Faraday/RT (um^2/(ms V))
  [./Deffe]
    type = DerivativeParsedMaterial
    expression = 'D*c*valency*Faraday/RT'
    coupled_variables = 'eta c'
    property_name = Deffe
    material_property_names = 'D(eta) valency Faraday RT'
    derivative_order = 1
  [../]
  [./coupled_eta_function]
    type = ParsedMaterial
    expression = 'c_s/c_0'  # normalize
    property_name = ft
    material_property_names = 'c_s c_0'
  [../]

  # ---- conduction for pot ----
  [./ElecEff]
    type = DerivativeParsedMaterial
    expression = '(S1o*h+S2o*(1-h))/length_scale'
    coupled_variables = 'eta'
    property_name = ElecEff
    material_property_names = 'S1o S2o h(eta) length_scale'
    derivative_order = 1
  [../]
  # n*Faraday*c_s in C/um^3; time_scale converts deta/dt from 1/ms to 1/s
  [./ChargeEff]
    type = ParsedMaterial
    expression = 'valency*Faraday*c_s/(length_scale^3)*time_scale'
    property_name = ChargeEff
    material_property_names = 'valency Faraday length_scale c_s time_scale'
  [../]

  # Li inventory density c + (c_s/c_0)*eta for the conservation check
  [./li_total_density]
    type = ParsedMaterial
    property_name = li_total_density
    expression = 'c+ft*eta'
    material_property_names = 'ft'
    coupled_variables = 'c eta'
  [../]
[]

[Kernels]
  #
  # Concentration equation
  #
  [./dcdt]
    type = TimeDerivative
    variable = c
    save_in = c_residual
  [../]
  # -div(M grad(dF/dc))
  [./ch]
    type = CahnHilliard
    variable = c
    save_in = c_residual
    f_name = F
    mob_name = M
    coupled_variables = 'eta'
  [../]
  # -div(D*c*n*Faraday/RT grad(pot))
  [./elec]
    type = MatDiffusion
    variable = c
    save_in = c_residual
    v = pot
    diffusivity = Deffe
    args = 'eta'
  [../]
  [./cSource]
    type = CoupledSusceptibilityTimeDerivative
    variable = c
    save_in = c_residual
    v = eta
    f_name = ft
  [../]

  #
  # Allen-Cahn Equation
  #
  [./detadt]
    type = TimeDerivative
    variable = eta
  [../]
  [./ac]
    type = AllenCahn
    variable = eta
    coupled_variables = 'c'
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
    coupled_variables = 'c pot'
  [../]
  [./noise_interface]
    type = LangevinNoise
    variable = eta
    multiplier = dh/deta
    amplitude = $Noise
  [../]

  #
  # Potential: div(sigma grad(pot)) = n*Faraday*c_s*deta/dt
  #
  [./Cond]
    type = MatDiffusion
    variable = pot
    diffusivity = ElecEff
    args = 'eta'
  [../]
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

  # Planar front speed v = L2o*exp(alpha*F*|POT_LEFT|/RT)*kappa/sigma: ~2e-6 m/s at -0.3 V and
  # ~2e-8 m/s at -0.1 V, so the 80 um to the stop at x = 90 takes ~40-3400 s.
  # dtmax = 100 ms moves the fastest front ~0.2 um (0.4 elements) per step.
  dtmax = 100
  end_time = 4E6

  [./TimeStepper]
    type = IterationAdaptiveDT
    dt = 1E-3
    growth_factor = 1.1
  [../]
[]

[Preconditioning]
  [./full]
    type = SMP
    full = true
  [../]
[]

[Postprocessors]
  # Lithium balance: change of integral(c + ft*eta) minus the time-integrated
  # inflow (residual at the prescribed-c nodes), relative to the inventory change.
  [./li_total]
    type = ElementIntegralMaterialProperty
    mat_prop = li_total_density
    execute_on = 'INITIAL TIMESTEP_END'
    outputs = none
  [../]
  [./li_initial]
    type = ElementIntegralMaterialProperty
    mat_prop = li_total_density
    execute_on = 'INITIAL'
    outputs = none
  [../]
  [./li_in_rate]
    type = NodalSum
    variable = c_residual
    boundary = right
    execute_on = 'TIMESTEP_END'
    outputs = none
  [../]
  [./li_in]
    type = TimeIntegratedPostprocessor
    value = li_in_rate
    time_integration_scheme = implicit-euler
    execute_on = 'TIMESTEP_END'
    outputs = none
  [../]
  [./li_balance_error]
    type = ParsedPostprocessor
    pp_names = 'li_total li_initial li_in'
    expression = 'abs(li_total-li_initial-li_in)/max(abs(li_total-li_initial),1e-3*abs(li_initial))'
    execute_on = 'TIMESTEP_END'
  [../]

  # Dendrite tip tracking
  [./tip_x]
    type = NodalExtremeValue
    variable = tip_coordinate
    value_type = max
    execute_on = 'INITIAL TIMESTEP_END'
    force_postaux = true
  [../]
[]

[UserObjects]
  [./stop_at_right]
    type = Terminator
    expression = 'tip_x >= 90'
    fail_mode = HARD
    error_level = INFO
    message = 'Nodal eta>=0.5 tip is within 10 units of x=100.'
    execute_on = TIMESTEP_END
    execution_order_group = 1
  [../]
  [./stop_at_left]
    type = Terminator
    expression = 'tip_x <= 5'
    fail_mode = HARD
    error_level = INFO
    message = 'Nodal eta>=0.5 tip has retreated to x<=5, or the solid phase has disappeared.'
    execute_on = TIMESTEP_END
    execution_order_group = 1
  [../]
[]

[Outputs]
  file_base = results/$CASE/$CASE
  # write every 10th time step
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