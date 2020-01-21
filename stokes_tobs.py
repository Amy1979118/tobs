from oct2py import octave
from dolfin import *
from dolfin_adjoint import *
import numpy as np

import cplex
from cplex.exceptions import CplexError

parameters["std_out_all_processes"] = False # turn off redundant output in parallel
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
#parameters["form_compiler"]["cpp_optimize_flags"] = "-O3 -ffast-math -march=native"
parameters['form_compiler']['representation'] = 'uflacs'
parameters["ghost_mode"] = "shared_facet"

# dolfin log level
#set_log_level(LogLevel.ERROR)
#set_log_level(LogLevel.PROGRESS)
#set_log_level(LogLevel.DEBUG)
#set_log_level(LogLevel.INFO)

class Optimizer(object):

    objfun_rf = None        #Objective Functions
    iter_fobj = 0
    iter_dobj = 0
    file_out = None         #Results file output
    xi_array = None         #Design Variables
    vf_fun = None
    cst_U = []
    cst_L = []
    cst_num = 0
    rho = None

    def __init__(self, fc_xi):
        self.fc_xi = Function(fc_xi.function_space(), name="Control")
        self.nvars = len(self.fc_xi.vector())
        self.control = Control(fc_xi)

    def __check_ds_vars__(self):
        chk_var = False
        if self.xi_array is None:
            self.xi_array = np.copy(self.rho)
            chk_var = True
        else:
            xi_eval = self.xi_array - self.rho
            xi_nrm = np.linalg.norm(xi_eval)
            if xi_nrm > 1e-16:
                self.xi_array = np.copy(self.rho)
                chk_var = True      #A variavel de projeto ja foi carregada
        if chk_var is True:
            self.fc_xi.vector()[:] = self.rho
        else:
            pass
        ds_vars = self.fc_xi
        return ds_vars

    def __vf_fun_var_assem__(self):
        fc_xi_tst = TestFunction(self.fc_xi.function_space())
        self.vol_xi = assemble(fc_xi_tst * Constant(1.0) * dx)
        self.vol_sum = self.vol_xi.sum()

    def add_plot_res(self, file_out):
        self.file_out = file_out

    def add_objfun(self, AD_Obj_fx):
        self.objfun_rf = ReducedFunctional(AD_Obj_fx, self.control)

    def obj_fun(self, user_data=None):
        ds_vars = self.__check_ds_vars__()
        fval = self.objfun_rf(ds_vars) # slow because is a solve
        self.iter_fobj += 1
        return fval

    def obj_dfun(self, user_data=None):
        ds_vars = self.__check_ds_vars__()
        print('\tRecalculate Objective Function')
        self.objfun_rf(ds_vars) # slow because is a solve
        #Derivada da funcao objetivo
        print('\tEvaluating Sensibility')
#        dfval = self.objfun_rf.derivative().vector() # slow because is a solve
        dfval = self.objfun_rf.derivative() # slow because is a solve
#        dfval_viz = self.objfun_rf.derivative()
#        dfval_viz.rename("sensibility", "")
#        self.sens_viz << dfval_viz
#        dfval = dfval_viz.vector() # slow because is a solve
        #salva os arquivos de resultados
        if self.file_out is not None:
            self.file_out << self.fc_xi
        # contador do numero de iteracoes
        self.iter_dobj += 1
        return dfval

    def add_volf_constraint(self, upp, lwr):
        self.__vf_fun_var_assem__()
        self.cst_U.append(upp)
        self.cst_L.append(lwr)
        self.cst_num += 1

    def volfrac_fun(self):
        self.__check_ds_vars__()
        volume_val = float(self.vol_xi.inner(self.fc_xi.vector()))
        return volume_val/self.vol_sum

    def volfrac_dfun(self, user_data=None):
        v_df = self.vol_xi/self.vol_sum
        return v_df

    def flag_jacobian(self):
        rows = []
        for i in range(self.cst_num):
            rows += [i] * self.nvars
        cols = range(self.nvars) * self.cst_num
        return (np.array(rows, dtype=np.int), np.array(cols, dtype=np.int))

    def cst_fval(self, user_data=None):
        cst_val = np.array(self.volfrac_fun(), dtype=np.float)
        return cst_val.T

    def jacobian(self, flag=False, user_data=None):
        if flag:
            dfval = self.flag_jacobian()
        else:
            dfval = self.volfrac_dfun()
        return dfval

octave.addpath('~')
pasta = "output/"

#%% Material parameters
mu = Constant(1.0)                   # viscosity
#alphaunderbar = 2.5 * mu / (100**2)  # parameter for \alpha
#alphabar = 2.5 * mu / (0.01**2)      # parameter for \alpha
alphaunderbar = 0.0
alphabar = 1.0e5
q = 1.0 # q value that controls difficulty/discrete-valuedness of solution

def alpha(rho):
    """Inverse permeability as a function of rho, equation (40)"""
    return Constant(alphabar) + (Constant(alphaunderbar) - Constant(alphabar)) * rho * (1.0 + Constant(q)) / (rho + Constant(q))

#%% Mesh
N = 10
delta = 1.5  # The aspect ratio of the domain, 1 high and \delta wide

#mesh = Mesh(RectangleMesh(Point(0.0, 0.0), Point(delta, 1.0), int(15*N), int(10*N), diagonal="crossed"))
mesh = Mesh(RectangleMesh(Point(0.0, 0.0), Point(delta, 1.0), int(15*N+1), int(10*N+1), diagonal="left/right"))

# mesh = RectangleMesh.create([Point(0.0,0.0),Point(delta,1.0)], [int(15*N+1),int(10*N+1)], CellType.Type.quadrilateral)

mesh = Mesh(mesh)

A = FunctionSpace(mesh, "DG", 0) # control function space
U_h = VectorElement("CG", mesh.ufl_cell(), 2)
P_h = FiniteElement("CG", mesh.ufl_cell(), 1)
W = FunctionSpace(mesh, U_h*P_h)          # mixed Taylor-Hood function space
# Define the boundary condition on velocity
class InflowOutflow(UserExpression):
    def eval(self, values, x):
        values[1] = 0.0
        values[0] = 0.0
        l = 1.0/6.0
        gbar = 1.0
        if x[0] == 0.0 or x[0] == delta:
            if (1.0/4 - l/2) < x[1] < (1.0/4 + l/2):
                t = x[1] - 1.0/4
                values[0] = gbar*(1 - (2*t/l)**2)
            if (3.0/4 - l/2) < x[1] < (3.0/4 + l/2):
                t = x[1] - 3.0/4
                values[0] = gbar*(1 - (2*t/l)**2)

    def value_shape(self):
        return (2,)

def forward(rho):
    """Solve the forward problem for a given fluid distribution rho(x)."""
    w_resp = Function(W)
    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)

    F = (alpha(rho) * inner(u, v) * dx + mu*inner(grad(u)+grad(u).T, grad(v)) * dx +
         inner(grad(p), v) * dx  + inner(div(u), q) * dx)
    bc = DirichletBC(W.sub(0), InflowOutflow(degree=1), "on_boundary")
    solve(lhs(F) == rhs(F), w_resp, bcs=bc, solver_parameters={'linear_solver':'mumps'})
#    solve(F==0, w_resp, bcs=bc, solver_parameters={'nonlinear_solver':'snes'})
    return w_resp

class Distribution(UserExpression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def eval_cell(self, values, x, ufc_cell):
        values[0] = 1

    def value_shape(self):
        return ()

def cplex_optimize(prob, nvar, my_obj, my_constcoef, my_rlimits, my_ll, my_ul):
    prob.objective.set_sense(prob.objective.sense.minimize)
    my_ctype = "I"*nvar
    my_colnames = ["x"+str(item) for item in range(nvar)]
    my_sense = ["L", "G"]
    my_rownames = ["r1", "r2"]

    prob.variables.add(obj=my_obj, lb=my_ll, ub=my_ul, types=my_ctype,
                       names=my_colnames)

    rows = [cplex.SparsePair(ind=["x"+str(item) for item in range(nvar)], val = my_constcoef[0]),
            cplex.SparsePair(ind=["x"+str(item) for item in range(nvar)], val = my_constcoef[1])]

    prob.linear_constraints.add(lin_expr=rows, senses=my_sense, rhs=my_rlimits, names=my_rownames)

def helmholtz_filter(phi):
    print('\n Solve Helmholtz Filter\n')
    FS = phi.function_space()
    vH = TestFunction(FS)
#    radius = Constant(1.75*dt, name='Filter Radius')
    h = MaxCellEdgeLength(mesh)
#    h = CellDiameter(mesh)
#    radius = Constant(1.75*dt, name='Filter Radius')
#    radius = Constant(2.5*dt, name='Filter Radius')
#    radius = Constant(2.00*dt, name='Filter Radius')
#    radius = Constant(0.1, name='Filter Radius')
    radius = Constant(1.0)*h
    a_f = Function(FS, name="Filtered")

    F = (
        inner(radius*radius * grad(a_f), grad(vH))*dx
        + inner(a_f, vH)*dx
        - inner(phi, vH)*dx
    )

    dw = TrialFunction(FS)
    J = derivative(F, a_f, dw)  # Gateaux derivative in dir. of dw

    problem = NonlinearVariationalProblem(F, a_f, None, J)
    solver = NonlinearVariationalSolver(problem)

    prm = solver.parameters
    prm['nonlinear_solver'] = 'newton'
    prm['snes_solver']['linear_solver'] = 'mumps'
    prm['snes_solver']['preconditioner'] = 'default'
    prm['newton_solver']['absolute_tolerance'] = 1E-7
    prm['newton_solver']['relative_tolerance'] = 1E-16
    prm['newton_solver']['maximum_iterations'] = 20
    prm['newton_solver']['relaxation_parameter'] = 1.0
    prm['newton_solver']['linear_solver'] = 'mumps'
    prm['newton_solver']['preconditioner'] = 'default'

    solver.solve(annotate=False)

    return a_f

if __name__ == "__main__":
    controls = File(pasta + "control.pvd")
    state_file = File(pasta + "velocity.pvd")
    sens_file = File(pasta + "sensibility.pvd")
    sens_filtered_file = File(pasta + "sensibility_filtered.pvd")
    # Set initial guess
    rho = interpolate(Distribution(), A)
    rho.rename("control", "")
    print('\tStart Optimization Loop')
    #%% Opt Loop
    # Opt parameters
    iteration = 0
    max_iter = 150
    epsilons = 0.02
    opt_convergence = False
    Fobj_values = []
    Vol_constraint_values = []
    while opt_convergence is False:
        set_working_tape(Tape()) # Create new Tape for Adjoint
        print('\tForward Problem')
        w_resp = forward(rho)
        (u, p) = w_resp.split()
        u.rename("velocity", "")
        rho.rename("control", "")

        # Create Opt Object
        fval = Optimizer(rho)
        fval.rho = rho.vector()
        nvar = len(rho.vector())
        # Objective Funcion
        J = assemble((0.5*inner(alpha(rho)*u, u) + mu*inner(grad(u), grad(u)))*dx)
        fval.add_objfun(J)

        j = float(J)
        Fobj_values.append(j)        

        jd = fval.obj_dfun()
        jd.rename("sensitivity", "")

        print('\tSaving Files')
        state_file << u
        controls << rho
        sens_file << jd

#        jd = helmholtz_filter(jd)
#        sens_filtered_file << jd
        jd = np.array(jd.vector()).reshape((-1, 1))
        fval.add_volf_constraint(0.7, 0.5)
        x_L = np.ones((nvar), dtype=np.float) * 0.0
        x_U = np.ones((nvar), dtype=np.float) * 1.0
        acst_L = np.array(fval.cst_L)
        acst_U = np.array(fval.cst_U)
        cs = fval.cst_fval()
        jac = np.array(fval.jacobian()).reshape((-1, 1))
        
        vol_frac = fval.volfrac_fun()
        Vol_constraint_values.append(vol_frac)

        print('\n\tIter: %3.d \tObj Func: %.4f \tVol Frac: %.4f' %(iteration, j, vol_frac))

        ans = octave.stokes(
            nvar,
            x_L,
            x_U,
            fval.cst_num,
            acst_L,
            acst_U,
            j,
            jd,
            cs,
            jac,
            iteration,
            epsilons,
            np.array(rho.vector())
            )

        # Cplex variables
        PythonObjCoeff = ans[0][1] #because [0][0] is the design variable
        PythonConstCoeff = ans[0][2]
        PythonRelaxedLimits = ans[0][3]
        PythonLowerLimits = ans[0][4]
        PythonUpperLimits = ans[0][5]
        PythonnDesignVariables = ans[0][6]
        my_prob = cplex.Cplex()

        # cplex log
        #my_prob.set_log_stream(None)
        #my_prob.set_error_stream(None)
        #my_prob.set_warning_stream(None)
        my_prob.set_results_stream(None)

        # cplex parameters

        coef = [item[0] for item in PythonObjCoeff.tolist()]
        constcoef = PythonConstCoeff.tolist()
        rlimits = [item[0] for item in PythonRelaxedLimits.tolist()]
        ll = [item[0] for item in PythonLowerLimits.tolist()]
        ul = [item[0] for item in PythonUpperLimits.tolist()]
        cplex_optimize(my_prob, nvar, coef, constcoef, rlimits, ll, ul)

        print('\tSolve Optimization Problem')
        my_prob.solve()
        design_variables = my_prob.solution.get_values()

        rho.rename("control", "")
        print('\tUpdate Design Variables')
        rho.vector().add_local(np.array(design_variables))

        if iteration >= max_iter:
            opt_convergence = True
        else: iteration += 1
        jd_previous = jd
        print('\n----------------------------------------')

    print('\tForward Problem')
    w_resp = forward(rho)
    (u, p) = w_resp.split()
    u.rename("velocity", "")
    rho.rename("control", "")
    print('\tSaving Last Solution')
    state_file << u
    controls << rho
    
    print('\tSaving Convergence History to File')
    with open(pasta+'Fobj.txt', 'w') as f:
        for value in Fobj_values: 
            f.write('%s\n' %value)
            
    print('\tSaving Volume Fraction History to File')
    with open(pasta+'Vol_constraint.txt', 'w') as f:
        for value in Vol_constraint_values: 
            f.write('%s\n' %value)
            
    print('\n----------------------------------------')
    print('----------------------------------------')
    print('\t END OF OPTIMIZATION')
    print('----------------------------------------')
    print('----------------------------------------\n\n')
