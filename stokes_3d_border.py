from oct2py import octave
from dolfin import *
from mshr import *
from dolfin_adjoint import *
from mpi4py import MPI as bruno_mpi
import numpy as np
import os

import cplex
from cplex.exceptions import CplexError

import smooth as sm

# set_log_level(50)
# parameters["linear_algebra_backend"] = 'Eigen'
comm = bruno_mpi.COMM_WORLD
rank = comm.Get_rank()

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
        self.fc_xi = Function( fc_xi.function_space(), name="Control" )
        self.nvars = len(self.fc_xi.vector())
        self.control = Control(fc_xi)

    def __check_ds_vars__(self):
        chk_var = False
        if self.xi_array is None:
            self.xi_array = np.copy(self.rho)
            chk_var = True
        else:
            xi_eval = self.xi_array - self.rho
            xi_nrm  = np.linalg.norm(xi_eval)
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
        fc_xi_tst   = TestFunction(self.fc_xi.function_space())
        self.vol_xi  = assemble(fc_xi_tst * Constant(1.0) * dx)
        self.vol_sum = self.vol_xi.sum()

    def add_plot_res(self, file_out):
        self.file_out = file_out

    def add_objfun(self, AD_Obj_fx):
        self.objfun_rf = ReducedFunctional(AD_Obj_fx, self.control)

    def obj_fun(self, user_data=None):
        ds_vars = self.__check_ds_vars__()
        fval = self.objfun_rf(ds_vars)
        print(" fval: ", fval)
        self.iter_fobj += 1

        return fval

    def obj_dfun(self, user_data=None):
        ds_vars  = self.__check_ds_vars__()
        self.objfun_rf(ds_vars)
        #Derivada da funcao objetivo
        dfval = self.objfun_rf.derivative().vector()
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

        volume_val = float( self.vol_xi.inner( self.fc_xi.vector() ) )

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
parameters["std_out_all_processes"] = False
pasta = "output/"

mu = Constant(1.0)                   # viscosity
alphaunderbar = 2.5 * mu *1.e-4
alphabar = 2.5 * mu * 1e4
q = Constant(1.0) # q value that controls difficulty/discrete-valuedness of solution

def alpha(rho):
    """Inverse permeability as a function of rho, equation (40)"""
    return alphabar + (alphaunderbar - alphabar) * rho * (1 + q) / (rho + q)

N = 80
delta = 1.5  # The aspect ratio of the domain, 1 high and \delta wide
V = Constant(1.0/3) * delta  # want the fluid to occupy 1/3 of the domain

# mesh = Mesh(BoxMesh(Point(0.0, 0.0, 0.0), Point(delta, 1.0, 1.0), 30, 20, 20), "crossed")
box = Box(Point(0.0, 0.0, 0.0), Point(delta, 1.0, 1.0))
# plot(box)
mesh = Mesh(generate_mesh(box, N))
File("malha3d.pvd") << mesh
A = FunctionSpace(mesh, "DG", 0)        # control function space

# U_h = VectorElement("CG", mesh.ufl_cell(), 2)
eUu = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
eB = FiniteElement("Bubble", mesh.ufl_cell(), 4)
U_h = VectorElement(NodalEnrichedElement(eUu, eB))
P_h = FiniteElement("CG", mesh.ufl_cell(), 1)
W = FunctionSpace(mesh, U_h*P_h)          # mixed Taylor-Hood function space

# Define the boundary condition on velocity

class InflowOutflow(UserExpression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def eval(self, values, x):
        values[2] = 0.0
        values[1] = 0.0
        values[0] = 0.0
        l = 1.0/6.0
        gbar = 1.0

        if x[0] == 0.0 or x[0] == delta:
            if (1.0/4 - l/2) < x[1] < (1.0/4 + l/2) and (1.0/4 - l/2) < x[2] < (1.0/4 + l/2):
                t = ((x[1]-1/4)**2 + (x[2]-1/4)**2)**0.5
                values[0] = gbar*(1 - t**2)

            if (1.0/4 - l/2) < x[1] < (1.0/4 + l/2) and (3.0/4 - l/2) < x[2] < (3.0/4 + l/2):
                t = ((x[1]-1/4)**2 + (x[2]-3/4)**2)**0.5
                values[0] = gbar*(1 - t**2)

            if (3.0/4 - l/2) < x[1] < (3.0/4 + l/2) and (1.0/4 - l/2) < x[2] < (1.0/4 + l/2):
                t = ((x[1]-3/4)**2 + (x[2]-1/4)**2)**0.5
                values[0] = gbar*(1 - t**2)

            if (3.0/4 - l/2) < x[1] < (3.0/4 + l/2) and (3.0/4 - l/2) < x[2] < (3.0/4 + l/2):
                t = ((x[1]-3/4)**2 + (x[2]-3/4)**2)**0.5
                values[0] = gbar*(1 - t**2)

    def value_shape(self):
        return (3,)

def forward(rho):
    """Solve the forward problem for a given fluid distribution rho(x)."""
    w_resp = Function(W)
    (u, p) = TrialFunctions(W)
    # (u, p) = split(w_resp)
    (v, q) = TestFunctions(W)

    F = (alpha(rho) * inner(u, v) * dx + mu*inner(grad(u)+grad(u).T, grad(v)) * dx +
         inner(grad(p), v) * dx  + inner(div(u), q) * dx)
    inflow = InflowOutflow(degree=5)
    bc = DirichletBC(W.sub(0), inflow, "on_boundary")
    Jacob = derivative(F, w_resp)
    # solve(l_esq == l_dir, w_resp, bcs=bc)

    solve(lhs(F) == rhs(F), w_resp, bc, solver_parameters= {'linear_solver' : 'mumps'})


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


if __name__ == "__main__":
    rho_distrib = Distribution()
    rho = interpolate(rho_distrib, A)
    rho.rename("control", "")
    w_resp   = forward(rho)
    (u, p) = w_resp.split()

    controls = File(pasta + "control.pvd")
    state_file = File(pasta + "veloc.pvd")
    rho_viz = Function(A, name="ControlVisualisation")
    u.rename("Velocidade", "")
    state_file << u

    file_obj_fun_path = pasta + "fun_obj.txt"
    if rank ==0 :
        if os.path.exists(file_obj_fun_path):
            os.remove(file_obj_fun_path)
        with open(file_obj_fun_path, "a+") as f:
            f.write("FunObj\n")
        bruno = 0
    else:
        bruno = None
    bruno = comm.bcast(bruno, root=0)

    file_new_mesh_refined = File(pasta + "new_mesh_refined.pvd")
    file_regions = File(pasta + "regions.pvd")

    '''import pdb; pdb.set_trace()
    new_mesh_refined, domain = sm.generate_polygon_refined(rho, mesh, geo="3D")
    file_new_mesh_refined << new_mesh_refined
    file_regions << domain'''

    controls << rho

    iteration = 0
    epsilons = .2

    while True:
        set_working_tape(Tape())
        J = assemble(0.5 * inner(alpha(rho) * u, u) * dx + mu * inner(grad(u), grad(u).T) * dx)
        with open(file_obj_fun_path, "a+") as f:
            f.write(str(float(J))+"\n")

        nvar = len(rho.vector())

        fval = Optimizer(rho)
        fval.add_objfun(J)
        fval.add_volf_constraint(0.7,0.5)
        fval.rho = rho.vector()
        x_L = np.ones((nvar), dtype=np.float) * 0.0
        x_U = np.ones((nvar), dtype=np.float) * 1.0
        acst_L = np.array(fval.cst_L)
        acst_U = np.array(fval.cst_U)
        j = float(fval.obj_fun(rho.vector()))
        if iteration == 0: jd_previous = np.array(fval.obj_dfun()).reshape((-1,1))
        jd = (np.array(fval.obj_dfun()).reshape((-1,1)) + jd_previous)/2 #stabilization
        cs = fval.cst_fval()
        jac = np.array(fval.jacobian()).reshape((-1,1))

        if rank == 0:
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
        else:
            ans = None
        ans = comm.bcast(ans, root=0)

        PythonObjCoeff = ans[0][1] #because [0][0] is the design variable
        PythonConstCoeff = ans[0][2]
        PythonRelaxedLimits = ans[0][3]
        PythonLowerLimits = ans[0][4]
        PythonUpperLimits = ans[0][5]
        PythonnDesignVariables = ans[0][6]
        my_prob = cplex.Cplex()
        my_prob.parameters.mip.strategy.variableselect.set(2)
        # my_prob.parameters.mip.strategy.file.set(3) #default is 1
        # my_prob.parameters.mip.limits.treememory.set(1e+3)
        # my_prob.parameters.workmem.set(10)
        coef = [item[0] for item in PythonObjCoeff.tolist()]
        constcoef = PythonConstCoeff.tolist()
        rlimits = [item[0] for item in PythonRelaxedLimits.tolist()]
        ll = [item[0] for item in PythonLowerLimits.tolist()]
        ul = [item[0] for item in PythonUpperLimits.tolist()]
        cplex_optimize(my_prob, nvar, coef, constcoef, rlimits, ll, ul)

        my_prob.solve()
        design_variables = my_prob.solution.get_values()

        rho.rename("control", "")
        rho.vector().add_local(np.array(design_variables))
        controls << rho

        w_resp   = forward(rho)
        (u, p) = w_resp.split()
        u.rename("Velocidade", "")
        state_file << u
        if iteration == 140:
            break
        elif iteration == 6:
            #q.assign(0.01)
            pass
        iteration += 1
        jd_previous = jd

        '''new_mesh_refined, domain = sm.generate_polygon_refined(rho, mesh)
        file_new_mesh_refined << new_mesh_refined
        file_regions << domain'''

