from oct2py import octave
from dolfin import *
from dolfin_adjoint import *
import numpy as np
import os

import cplex
from cplex.exceptions import CplexError

set_log_level(50)

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
        self.fc_xi.vector()[:] = self.rho
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
        var_temp = self.objfun_rf.derivative()
        var_temp.rename("sensibility", "sensibility")
        file_sen << var_temp
        #salva os arquivos de resultados
        if self.file_out is not None:
            self.file_out << self.fc_xi
        # contador do numero de iteracoes
        self.iter_dobj += 1

        return dfval

    def volfrac_fun(self):

        ds_vars = self.__check_ds_vars__()

        fc_xi_tst = TestFunction(ds_vars.function_space())
        volume_val  = assemble(fc_xi_tst * ds_vars * dx).sum()

        return volume_val

    def volfrac_dfun(self, user_data=None):
        ds_vars = self.__check_ds_vars__()
        fc_xi_tst = TestFunction(ds_vars.function_space())
        volume_val  = np.array(assemble(fc_xi_tst * Constant(1.0) * dx))
        v_df = volume_val
        return v_df

        return (np.array(rows, dtype=np.int), np.array(cols, dtype=np.int))

    def cst_fval(self, user_data=None):
        cst_val = np.array(self.volfrac_fun(), dtype=np.float)

        return cst_val.T

octave.addpath('~')
parameters["std_out_all_processes"] = False
mu = Constant(1.0)

beta = 0.2
epsilons = 0.1
delta = 1.5
N = 80
alphabar = Constant(2.5e4)

q = Constant(1.0) # q value that controls difficulty/discrete-valuedness of solution

pasta = "output_beta"+str(beta)+"_eps"+str(epsilons)+"_delta"+str(delta)+ "iter_160" + "_N=" + str(N) +"_alphabar"+ str(float(alphabar)) + "_q="+ str(float(q)) +"bc/"

alphaunderbar = 2.5 * mu *1.e-4

def alpha(rho):
    """Inverse permeability as a function of rho, equation (40)"""
    return alphabar + (alphaunderbar - alphabar) * rho * (1 + q) / (rho + q)

mesh = Mesh(RectangleMesh(Point(0.0, 0.0), Point(delta, 1.0), int(N*delta), int(N), diagonal="right"))
A = FunctionSpace(mesh, "DG", 0)        # control function space

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
    l_esq = lhs(F)
    l_dir = rhs(F)
    solve(l_esq == l_dir, w_resp, bcs=bc)

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
    my_sense = ["L", "L"]
    my_rownames = ["r1", "r2"]

    prob.variables.add(obj=my_obj, lb=my_ll, ub=my_ul, types=my_ctype,
                       names=my_colnames)

    rows = [cplex.SparsePair(ind=["x"+str(item) for item in range(nvar)], val = my_constcoef[0]),
            cplex.SparsePair(ind=["x"+str(item) for item in range(nvar)], val = my_constcoef[1])]

    prob.linear_constraints.add(lin_expr=rows, senses=my_sense, rhs=my_rlimits, names=my_rownames)


if __name__ == "__main__":
    rho = interpolate(Distribution(), A)
    rho.rename("control", "")
    w_resp   = forward(rho)
    (u, p) = w_resp.split()

    file_sen = File(pasta + "sensibility.pvd")
    controls = File(pasta + "control.pvd")
    state_file = File(pasta + "veloc.pvd")
    rho_viz = Function(A, name="ControlVisualisation")

    file_obj_fun_path = pasta + "fun_obj.txt"
    if os.path.exists(file_obj_fun_path):
        os.remove(file_obj_fun_path)
    with open(file_obj_fun_path, "a+") as f:
        f.write("FunObj \t VolCstr(%)\n")

    controls << rho

    iteration = 0
    j_previous = 0

    while True:
        J = assemble(inner(alpha(rho) * u, u) * dx +\
                0.5*(mu * inner(grad(u)+ grad(u).T, grad(u)+ grad(u).T) ) * dx)

        nvar = len(rho.vector())

        fval = Optimizer(rho)
        fval.add_objfun(J)
        fval.vol_constraint = 1./3 * delta
        fval.rho = rho.vector()
        x_L = np.ones((nvar), dtype=np.float) * 0.0
        x_U = np.ones((nvar), dtype=np.float) * 1.0
        j = float(fval.obj_fun(rho.vector()))
        if iteration == 0: jd_previous = np.array(fval.obj_dfun()).reshape((-1,1))
        jd = (np.array(fval.obj_dfun()).reshape((-1,1)) + jd_previous)/2 #stabilization
        cs = fval.cst_fval()
        jac = np.array(fval.volfrac_dfun()).reshape((-1,1))
        with open(file_obj_fun_path, "a+") as f:
            f.write(str(float(J))+"\t" + str(cs/delta*100) + "\n")
        ans = octave.stokes(
                nvar,
                x_L,
                x_U,
                fval.cst_num,
                j,
                jd,
                cs,
                jac,
                fval.vol_constraint,
                iteration,
                epsilons,
                np.array(rho.vector()),
                beta
                )
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

        set_working_tape(Tape())
        w_resp   = forward(rho)
        (u, p) = w_resp.split()
        u.rename("Velocidade", "")
        state_file << u
        # print("Change after last iteration: {}".format(abs(np.array(design_variables).sum())))
        # print("Change limit: {}".format(nvar * 0.0010))
        # print("Change after last iteration: {}".format(abs((j - j_previous)/j)))
        # print("Change limit: {}".format(0.00010))
        '''alphabar.assign(2.5e1)
        if iteration > 10:
            alphabar.assign(2.5e2)
        if iteration > 20:
            alphabar.assign(2.5e3)
        if iteration > 30:
            alphabar.assign(2.5e4)'''
        if iteration == 160: break
        iteration += 1
        j_previous = j
        jd_previous = jd

