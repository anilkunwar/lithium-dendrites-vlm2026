# hao tang Oct 2025
#
# Li electrodeposition phase-field model (eta = 1 Li metal, eta = 0 electrolyte).
# Model units: length um, time ms, energy eV. c is the Li+ concentration
# normalized by c_0; pot is the electric potential in V.
#
# Placeholders: $POT_LEFT $Noise (dimensionless, 0-1) $SEED $kappa (J/m) $W (J/m^3) $L1o $CASE
# ($kappa and $W are derived from sigma and delta by generate_moose_simulation_files.py)
#
# ---------------------------------------------------------------------------
# Concentration equation (derived from the free energy: Nernst-Planck as one gradient flow)
#   Total free energy density: F(c,eta) = fl(c) + W*g(eta)   [J/m^3]
#   The ion free energy fl(c) is not weighted by h: a (1-h)*fl(c) term would make the AC driving
#   force h'*(fs-fl) favour the solid wherever the electrolyte is depleted, so the interface
#   would consume its own Li+ (runaway down to the c >= 0 bound). Deposition is driven by the
#   Butler-Volmer term alone, whose c factor carries the concentration dependence.
#   Electrochemical potential per mol Li+: mu_e = (1/c_0) dF/dc + n*Faraday*pot
#   Flux (mol/m^2/s): J = -(D*C/RT) grad(mu_e), C = c_0*c
#   => dc/dt + (c_s/c_0) deta/dt = div[ M grad(dF/dc) + (D*c*n*Faraday/RT) grad(pot) ]
#      with M = D*c/(RT*c_0), the same mobility as the migration term.
#   With the ideal solution fl = c_0*RT*(c*ln(c) - c), M*d2fl/dc2 = D: Fickian diffusion that
#   does not vanish at c -> 0. (A quadratic fl with this M gives D*c*grad(c): no Li+ could reach
#   the c ~ 0 metal side of the interface and the c >= 0 bound created ~9 % of the deposit.)
#
# Interface (double well W*g, g = eta^2(1-eta)^2, gradient energy kappa/2|grad eta|^2)
#   sigma = sqrt(2*W*kappa)/6,  delta(10-90%) = 2.2*sqrt(2*kappa/W)
#   sigma, delta given => kappa = 3*sigma*delta/2.2,  W = 13.2*sigma/delta
# ---------------------------------------------------------------------------

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

  # Frozen random field in [-1, 1]: local scatter of the exchange current density (e.g. a
  # non-uniform SEI). It scales the Butler-Volmer rate, see Butlervolmer.
  [./xi]
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

[Functions]
  # Equilibrium interface at x = 10. A sharp step relaxes toward this width, and every change of
  # eta must be absorbed by c with factor c_s/c_0 = 77, so c hits its bounds and the Newton
  # solve fails. eta = 0.5*(1-tanh((x-10)/(2*l))), l = sqrt(kappa/(2*W)) (in um), whose
  # 0.1-0.9 width is 4.39*l = delta.
  [./eta_ic]
    type = ParsedFunction
    expression = '0.5*(1-tanh((x-10)/(2e6*sqrt(kappa/(2*W)))))'
    symbol_names = 'kappa W'
    symbol_values = '$kappa $W'
  [../]
  # Li+ only in the electrolyte part: c = 1-h(eta_ic), h = p^3*(6p^2-15p+10)
  # ('e' is not usable as a symbol name: the parser reserves it for Euler's number.)
  [./c_ic]
    type = ParsedFunction
    expression = '1-p^3*(6*p^2-15*p+10)'
    symbol_names = 'p'
    symbol_values = 'eta_ic'
  [../]
[]

[ICs]
  [./eta]
    variable = eta
    type = FunctionIC
    function = eta_ic
  [../]
  [./c]
    variable = c    # lithium ion concentration
    type = FunctionIC
    function = c_ic
  [../]
  # seeded per node id, so the field does not depend on the number of MPI ranks
  [./xi]
    variable = xi
    type = RandomIC
    min = -1
    max = 1
    seed = $SEED
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
  # every material property is written to the exodus file (elemental averages)
  [./scale]
    type = GenericConstantMaterial
    outputs = exodus
    prop_names = 'length_scale energy_scale time_scale'
    prop_values = '1e6 6.24150943e18 1e3'
  [../]
  [./constants]
    type = GenericConstantMaterial
    outputs = exodus
    prop_names  = 'c_s c_0 valency molar_vol Faraday RT alpha'
    prop_values = '7.69e4 1e3 1 2.2e-5 96485 2494.2 0.5'
  [../]
  # De, Dl: Li+ diffusivity in electrode / electrolyte (m^2/s)
  # L1o: interface mobility (m^3/(J s)); L2o: reaction (Butler-Volmer) rate constant (1/s)
  #   L1o*kappa*curvature is the curvature-driven interface speed; with L1o ~ 0.1 it reaches
  #   ~0.4 m/s at 1 um radius, far above the transport-limited deposition speed, and smooths
  #   every finger. L2o ~ 0.4 /s lets the BV term outgrow the double well at eta ~ 1e-3, so the
  #   electrolyte metallizes. L2o = 1e-3 follows Hong & Viswanathan 2018.
  #   L1o is set per case by the generator so that the Butler-Volmer rate stays a fixed fraction
  #   r of the Allen-Cahn restoring rate: L2o*exp(alpha*F*|POT_LEFT|/RT)*1.875 = r*L1o*W.
  #   A fixed L1o (2.5e-6) gave r ~ 2 at -0.45 V: the BV term outgrew the double well and the
  #   deposit turned into a diffuse eta ~ 0.6 region (33 % of the Li created by the c >= 0 bound).
  # S1o, S2o: conductivity of Li metal / electrolyte (S/m)
  [./system_constants]
    type = GenericConstantMaterial
    outputs = exodus
    prop_names  = 'De Dl L1o L2o S1o S2o'
    # Dl = 1e-9 (upper end of liquid electrolytes, 1e-10 to 1e-9): 3e-10 could not resupply the
    # metal side of the moving interface, and the c >= 0 bound created ~5 % of the deposit.
    prop_values = '7.5e-13 1.0e-9 $L1o 1e-3 1e7 1.19'
  [../]
  # ko: gradient energy coefficient (J/m); Wo: double-well barrier height (J/m^3)
  # The ion free energy fl does not depend on eta, so the deposition driving force comes only
  # from the overpotential (Butler-Volmer).
  [./thermo_constants]
    type = GenericConstantMaterial
    outputs = exodus
    prop_names  = 'ko Wo'
    prop_values = '$kappa $W'
  [../]

  # ---- derived parameters (converted to model units) ----
  [./L1]
    type = ParsedMaterial
    outputs = exodus
    expression = '((length_scale)^3/(energy_scale*time_scale))*L1o'
    property_name = L1
    material_property_names = 'L1o length_scale energy_scale time_scale'
  [../]
  [./L2]
    type = ParsedMaterial
    outputs = exodus
    expression = 'L2o/time_scale'
    property_name = L2
    material_property_names = 'L2o time_scale'
  [../]
  [./kappa_isotropy]
    type = ParsedMaterial
    outputs = exodus
    property_name = kappa
    expression = '(energy_scale/length_scale)*ko'
    material_property_names = 'ko length_scale energy_scale'
  [../]
  [./barrier_height]
    type = ParsedMaterial
    outputs = exodus
    property_name = W
    expression = '(energy_scale/(length_scale)^3)*Wo'
    material_property_names = 'Wo length_scale energy_scale'
  [../]

  # h(eta)
  [./h]
    type = SwitchingFunctionMaterial
    outputs = exodus
    h_order = HIGH
    eta = eta
  [../]
  # g(eta)
  [./g]
    type = BarrierFunctionMaterial
    outputs = exodus
    g_order = SIMPLE
    eta = eta
  [../]

  # ---- free energies ----
  # ideal solution of Li+ (reference c = 1); plog regularizes ln(c) below c = 1e-6
  [./fl]
    type = DerivativeParsedMaterial
    outputs = exodus
    property_name = fl
    material_property_names = 'c_0 RT length_scale energy_scale'
    coupled_variables = 'c'
    expression = '(energy_scale/(length_scale)^3)*c_0*RT*(c*plog(c,1e-6) - c)'
    derivative_order = 3
  [../]
  [./free_energy]
    type = DerivativeParsedMaterial
    outputs = exodus
    property_name = F
    coupled_variables = 'c eta'
    material_property_names = 'fl(c) g(eta) W'
    expression = 'fl + W*g'
    derivative_order = 3
  [../]

  # BV driving force. The rate constant scatters as L2*(1 + Noise*xi): noise in the reaction
  # rate rather than an additive eta noise. An additive eta noise changes eta where no Li+ is
  # available, and c must absorb 77x that change (c up to 23, Li+ trapped in the metal, c >= 0
  # clipping). Here every eta change still comes with the c factor of the cathodic term.
  [./Butlervolmer]
    type = DerivativeParsedMaterial
    outputs = exodus
    constant_names = 'A_noise'
    constant_expressions = '$Noise'
    expression = 'L2*(1+A_noise*xi)*dh*(exp(pot*(1-alpha)*Faraday*valency/RT)-c*exp(-pot*alpha*Faraday*valency/RT))'
    coupled_variables = 'pot c eta xi'
    property_name = f_bv
    material_property_names = 'L2 alpha Faraday valency RT dh:=D[h,eta]'
    derivative_order = 1
  [../]

  # ---- transport of c ----
  # D(eta) in um^2/ms, linear in eta. With h(eta), D fell to 1e-3*Dl at eta = 0.95, so Li+ could
  # not reach the metal side of a moving interface and the c >= 0 bound created ~11 % of the
  # deposit at -0.35 V; linear in eta it is still 0.05*Dl there and De in the bulk metal.
  [./D]
    type = DerivativeParsedMaterial
    outputs = exodus
    expression = '(length_scale)^2/time_scale*(De*eta+Dl*(1-eta))'
    coupled_variables = 'eta'
    property_name = D
    material_property_names = 'De Dl length_scale time_scale'
    derivative_order = 2
  [../]
  # M = D*c/(RT*c_0) (m^5/(J s)), converted to um^5/(eV ms) so that div(M grad(dF/dc)) is in 1/ms
  [./M]
    type = DerivativeParsedMaterial
    outputs = exodus
    expression = 'D*c/(RT*c_0)*(length_scale)^3/energy_scale'
    coupled_variables = 'c eta'
    property_name = M
    material_property_names = 'D(eta) RT c_0 length_scale energy_scale'
    derivative_order = 2
  [../]
  # electromigration coefficient D*c*n*Faraday/RT (um^2/(ms V))
  [./Deffe]
    type = DerivativeParsedMaterial
    outputs = exodus
    expression = 'D*c*valency*Faraday/RT'
    coupled_variables = 'eta c'
    property_name = Deffe
    material_property_names = 'D(eta) valency Faraday RT'
    derivative_order = 1
  [../]
  [./coupled_eta_function]
    type = ParsedMaterial
    outputs = exodus
    expression = 'c_s/c_0'  # normalize
    property_name = ft
    material_property_names = 'c_s c_0'
  [../]

  # ---- conduction for pot ----
  # Logarithmic (geometric) mean S1o^h * S2o^(1-h), with S1o/S2o ~ 8e6:
  #   linear mean: the tail (eta ~ 0.02) is already metallic, the field vanishes ahead of the
  #     reaction zone and Li+ piles up there (c ~ 1.09 at x ~ 15);
  #   harmonic mean: the reaction zone keeps the electrolyte conductivity, so the charge source
  #     n*F*c_s*deta/dt digs a potential well below the electrode potential (0.04 V at -0.42 V)
  #     that speeds up the deposition that digs it, and the Newton solve diverges.
  #   The geometric mean is ~3e3 S/m at h = 0.5 and ~1.6 S/m at h = 0.02, avoiding both.
  [./ElecEff]
    type = DerivativeParsedMaterial
    outputs = exodus
    expression = 'exp(h*log(S1o)+(1-h)*log(S2o))/length_scale'
    coupled_variables = 'eta'
    property_name = ElecEff
    material_property_names = 'S1o S2o h(eta) length_scale'
    derivative_order = 1
  [../]
  # n*Faraday*c_s in C/um^3; time_scale converts deta/dt from 1/ms to 1/s
  [./ChargeEff]
    type = ParsedMaterial
    outputs = exodus
    expression = 'valency*Faraday*c_s/(length_scale^3)*time_scale'
    property_name = ChargeEff
    material_property_names = 'valency Faraday length_scale c_s time_scale'
  [../]

  # Li inventory density c + (c_s/c_0)*eta for the conservation check
  [./li_total_density]
    type = ParsedMaterial
    outputs = exodus
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

  # GMRES + additive Schwarz with LU on each rank's subdomain: at 8 ranks 40 s vs 52 s for a
  # global MUMPS LU (20-step benchmark), and its parallel efficiency does not drop with rank count.
  petsc_options_iname = '-ksp_type -pc_type -sub_pc_type -pc_asm_overlap -ksp_gmres_restart -snes_type'
  petsc_options_value = 'gmres     asm      lu           2               100                vinewtonrsls'

  # Front speed ~0.03 um/s at -0.1 V up to ~20 um/s (finger tips) at -0.45 V: the run to the stop
  # at x = 90 takes from a few s to ~1 h of model time. dtmax = 1000 ms keeps the slow (low
  # potential) cases to a few thousand steps; IterationAdaptiveDT cuts dt where Newton struggles.
  dtmax = 1000
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