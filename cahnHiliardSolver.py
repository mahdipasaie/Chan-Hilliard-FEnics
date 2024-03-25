import fenics as fe
import numpy as np
from mpi4py import MPI
from tqdm import tqdm
from CahnHiliard_solo import update_solver_on_new_mesh_pf
from modAD import refine_mesh

fe.set_log_level(fe.LogLevel.ERROR)


#################### Define Parallel Variables ####################
# Get the global communicator
comm = MPI.COMM_WORLD
# Get the rank of the process
rank = comm.Get_rank()
# Get the size of the communicator (total number of processes)
size = comm.Get_size()
#############################  END  ################################


physical_parameters_dict = {
    "dy": 1E-2 ,
    "max_level": 2,
    "Nx": 1,
    "Ny": 1,
    "dt": 5E-6,
    "dy_coarse":lambda max_level, dy: 2**max_level * dy,
    "Domain": lambda Nx, Ny: [(0.0, 0.0), (Nx, Ny)],
    "M": 1,
    "A": 100,
    "kappa": 1E-2,
    "theta": 0.5,

    ###################### SOLVER PARAMETERS ######################
    
    "abs_tol_pf": 1E-8,  
    "rel_tol_pf": 1E-6,  
    "preconditioner_ns": 'ilu',  
    'maximum_iterations_ns': 50, 
    'nonlinear_solver_pf': 'snes',     # "newton" , 'snes'
    'linear_solver_pf': 'mumps',       # "mumps" , "superlu_dist", 'cg', 'gmres', 'bicgstab'
    "preconditioner_pf": 'ilu',       # 'hypre_amg', 'ilu', 'jacobi'
    'maximum_iterations_pf': 50,

    #############
     "interface_threshold_gradient": 0.0001,

}



dy = physical_parameters_dict['dy']
max_level = physical_parameters_dict['max_level']
Nx = physical_parameters_dict['Nx']
Ny = physical_parameters_dict['Ny']
dt = physical_parameters_dict['dt']



# Compute values from functions
dy_coarse = physical_parameters_dict['dy_coarse'](max_level, dy)
Domain = physical_parameters_dict['Domain'](Nx, Ny)


# Defining the mesh 
nx = (int)(Nx/ dy ) 
ny = (int)(Ny / dy ) 
nx_coarse = (int)(Nx/ dy_coarse ) 
ny_coarse = (int)(Ny / dy_coarse ) 

coarse_mesh = fe.RectangleMesh( fe.Point(0.0 , 0.0 ), fe.Point(Nx, Ny), nx_coarse, ny_coarse  )
mesh = fe.RectangleMesh( fe.Point(0.0 , 0.0 ), fe.Point(Nx, Ny), nx, ny )



####### writing to file ######## 


file = fe.XDMFFile("CahnHiliard.xdmf")


def write_simulation_data(Sol_Func, time, file, variable_names ):

    
    # Configure file parameters
    file.parameters["rewrite_function_mesh"] = True
    file.parameters["flush_output"] = True
    file.parameters["functions_share_mesh"] = True

    # Split the combined function into its components
    functions = Sol_Func.split(deepcopy=True)

    # Check if the number of variable names matches the number of functions
    if variable_names and len(variable_names) != len(functions):
        raise ValueError("The number of variable names must match the number of functions.")

    # Rename and write each function to the file
    for i, func in enumerate(functions):
        name = variable_names[i] if variable_names else f"Variable_{i}"
        func.rename(name, "solution")
        file.write(func, time)

    file.close()





##############################################################
old_solution_vector_pf = None
old_solution_vector_0_pf = None
##############################################################

########################## Initialize  problem ##############################

pf_problem_dict = update_solver_on_new_mesh_pf(mesh, physical_parameters_dict, old_solution_vector_pf= None, old_solution_vector_0_pf= None, 
                                variables_dict_pf= None)

# variables for solving the problem pf
solver_pf = pf_problem_dict["solver_pf"]
solution_vector_pf = pf_problem_dict["solution_vector_pf"]
solution_vector_pf_0 = pf_problem_dict["solution_vector_pf_0"]
spaces_pf = pf_problem_dict["spaces_pf"]
variables_dict_pf = pf_problem_dict["variables_dict_pf"]

########################## END ##############################


T = 0
for it in tqdm( range(0, 10000000) ):

    T += dt

    if it == 20 or it % 30 == 25 :

        # refining the mesh
        mesh, mesh_info = refine_mesh(physical_parameters_dict, coarse_mesh, solution_vector_pf, spaces_pf, comm )

        pf_problem_dict = update_solver_on_new_mesh_pf(mesh, physical_parameters_dict,
                            old_solution_vector_pf= old_solution_vector_pf, old_solution_vector_0_pf=old_solution_vector_0_pf, 
                            variables_dict_pf= None )
        

        # variables for solving the problem pf
        solver_pf = pf_problem_dict["solver_pf"]
        solution_vector_pf = pf_problem_dict["solution_vector_pf"]
        solution_vector_pf_0 = pf_problem_dict["solution_vector_pf_0"]
        spaces_pf = pf_problem_dict["spaces_pf"]
        variables_dict_pf = pf_problem_dict["variables_dict_pf"]


    else: 

        pf_problem_dict = update_solver_on_new_mesh_pf(mesh, physical_parameters_dict, old_solution_vector_pf= None, old_solution_vector_0_pf=None, 
                        variables_dict_pf= variables_dict_pf)
        

        # variables for solving the problem pf
        solver_pf = pf_problem_dict["solver_pf"]
        solution_vector_pf = pf_problem_dict["solution_vector_pf"]
        solution_vector_pf_0 = pf_problem_dict["solution_vector_pf_0"]
        spaces_pf = pf_problem_dict["spaces_pf"]
        variables_dict_pf = pf_problem_dict["variables_dict_pf"]


    



    # solving the problem
    solver_pf_information = solver_pf.solve()


    #definning old solution vectors
    old_solution_vector_pf = solution_vector_pf
    old_solution_vector_0_pf = solution_vector_pf

    #update the old solution vectors
    solution_vector_pf_0.assign(solution_vector_pf)


    ####### write first solution to file ########
    if it % 1 == 0: 

        variable_names_list = ["C", "mu"]  # Adjust variable names as needed
        write_simulation_data(solution_vector_pf_0, T, file, variable_names_list )







