#!/usr/bin/env python 

## \file design_optimization.py
#  \brief Python script for optimizing an airfoil shape.
#  \author A. S. Dallas, building off work by T. Economon, T. Lukaczyk, F. Palacios
#  \version 7.0.7 "Blackbird"
#
# SU2 Project Website: https://su2code.github.io
# 
# The SU2 Project is maintained by the SU2 Foundation 
# (http://su2foundation.org)
#
# Copyright 2012-2020, SU2 Contributors (cf. AUTHORS.md)
#
# SU2 is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
# 
# SU2 is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with SU2. If not, see <http://www.gnu.org/licenses/>.

import os, sys, shutil
from optparse import OptionParser
sys.path.append(os.environ['SU2_RUN'])
import SU2
import airfoilGen
import OGridGen
import pandas as pd
import numpy as np
import jax.ops as jo

# -------------------------------------------------------------------
#  Main 
# -------------------------------------------------------------------

def main():

    parser=OptionParser()
    parser.add_option("-f", "--file", dest="filename",
                      help="read config from FILE", metavar="FILE")
    parser.add_option("-r", "--name", dest="projectname", default='',
                      help="try to restart from project file NAME", metavar="NAME")
    parser.add_option("-n", "--partitions", dest="partitions", default=1,
                      help="number of PARTITIONS", metavar="PARTITIONS")
    parser.add_option("-g", "--gradient", dest="gradient", default="DISCRETE_ADJOINT",
                      help="Method for computing the GRADIENT (CONTINUOUS_ADJOINT, DISCRETE_ADJOINT, FINDIFF, NONE)", metavar="GRADIENT")
    parser.add_option("-o", "--optimization", dest="optimization", default="SLSQP",
                      help="OPTIMIZATION techique (SLSQP, CG, BFGS, POWELL)", metavar="OPTIMIZATION")
    parser.add_option("-q", "--quiet", dest="quiet", default="True",
                      help="True/False Quiet all SU2 output (optimizer output only)", metavar="QUIET")
    parser.add_option("-z", "--zones", dest="nzones", default="1",
                      help="Number of Zones", metavar="ZONES")


    (options, args)=parser.parse_args()
    
    # process inputs
    options.partitions  = int( options.partitions )
    options.quiet       = options.quiet.upper() == 'TRUE'
    options.gradient    = options.gradient.upper()
    options.nzones      = int( options.nzones )
    
    sys.stdout.write('\n-------------------------------------------------------------------------\n')
    sys.stdout.write('|    ___ _   _ ___                                                      |\n')
    sys.stdout.write('|   / __| | | |_  )   Release 7.0.7 \"Blackbird\"                         |\n')
    sys.stdout.write('|   \\__ \\ |_| |/ /                                                      |\n')
    sys.stdout.write('|   |___/\\___//___|   Design Optimization Script                       |\n')
    sys.stdout.write('|                                                                       |\n')
    sys.stdout.write('-------------------------------------------------------------------------\n')
    sys.stdout.write('| SU2 Project Website: https://su2code.github.io                        |\n')
    sys.stdout.write('|                                                                       |\n')
    sys.stdout.write('| The SU2 Project is maintained by the SU2 Foundation                   |\n')
    sys.stdout.write('| (http://su2foundation.org)                                            |\n')
    sys.stdout.write('-------------------------------------------------------------------------\n')
    sys.stdout.write('| Copyright 2012-2020, SU2 Contributors (cf. AUTHORS.md)                |\n')
    sys.stdout.write('|                                                                       |\n')
    sys.stdout.write('| SU2 is free software; you can redistribute it and/or                  |\n')
    sys.stdout.write('| modify it under the terms of the GNU Lesser General Public            |\n')
    sys.stdout.write('| License as published by the Free Software Foundation; either          |\n')
    sys.stdout.write('| version 2.1 of the License, or (at your option) any later version.    |\n')
    sys.stdout.write('|                                                                       |\n')
    sys.stdout.write('| SU2 is distributed in the hope that it will be useful,                |\n')
    sys.stdout.write('| but WITHOUT ANY WARRANTY; without even the implied warranty of        |\n')
    sys.stdout.write('| MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU      |\n')
    sys.stdout.write('| Lesser General Public License for more details.                       |\n')
    sys.stdout.write('|                                                                       |\n')
    sys.stdout.write('| You should have received a copy of the GNU Lesser General Public      |\n')
    sys.stdout.write('| License along with SU2. If not, see <http://www.gnu.org/licenses/>.   |\n')
    sys.stdout.write('-------------------------------------------------------------------------\n')

    design_optimization( options.filename    ,
                        options.projectname ,
                        options.partitions  ,
                        options.gradient    ,
                        options.optimization ,
                        options.quiet       ,
                        options.nzones      )
    
#: main()

def design_optimization( filename                           ,
                        projectname = ''                   ,
                        partitions  = 0                    ,
                        gradient    = 'CONTINUOUS_ADJOINT' ,
                        optimization = 'SLSQP'             ,
                        quiet       = False                ,
                        nzones      = 1                    ):
    # Config
    config = SU2.io.Config(filename)
    config.NUMBER_PART = partitions
    config.NZONES      = int( nzones )
    if quiet: config.CONSOLE = 'CONCISE'
    config.GRADIENT_METHOD = gradient
    
    its               = int ( config.OPT_ITERATIONS )                      # number of opt iterations
    bound_upper       = float ( config.OPT_BOUND_UPPER )                   # variable bound to be scaled by the line search
    bound_lower       = float ( config.OPT_BOUND_LOWER )                   # variable bound to be scaled by the line search
    relax_factor      = float ( config.OPT_RELAX_FACTOR )                  # line search scale
    gradient_factor   = float ( config.OPT_GRADIENT_FACTOR )               # objective function and gradient scale
    def_dv            = config.DEFINITION_DV                               # complete definition of the desing variable
    n_dv              = sum(def_dv['SIZE'])                                # number of design variables
    accu              = float ( config.OPT_ACCURACY ) * gradient_factor    # optimizer accuracy
    x0                = [0.0]*n_dv # initial design
    xb_low            = [float(bound_lower)/float(relax_factor)]*n_dv      # lower dv bound it includes the line search acceleration factor
    xb_up             = [float(bound_upper)/float(relax_factor)]*n_dv      # upper dv bound it includes the line search acceleration fa
    xb                = list(zip(xb_low, xb_up)) # design bounds
    
    # State
    state = SU2.io.State()
    state.find_files(config)

    # add restart files to state.FILES
    if config.get('TIME_DOMAIN', 'NO') == 'YES' and config.get('RESTART_SOL', 'NO') == 'YES' and gradient != 'CONTINUOUS_ADJOINT':
        restart_name = config['RESTART_FILENAME'].split('.')[0]
        restart_filename = restart_name + '_' + str(int(config['RESTART_ITER'])-1).zfill(5) + '.dat'
        if not os.path.isfile(restart_filename): # throw, if restart files does not exist
            sys.exit("Error: Restart file <" + restart_filename + "> not found.")
        state['FILES']['RESTART_FILE_1'] = restart_filename

        # use only, if time integration is second order
        if config.get('TIME_MARCHING', 'NO') == 'DUAL_TIME_STEPPING-2ND_ORDER':
            restart_filename = restart_name + '_' + str(int(config['RESTART_ITER'])-2).zfill(5) + '.dat'
            if not os.path.isfile(restart_filename): # throw, if restart files does not exist
                sys.exit("Error: Restart file <" + restart_filename + "> not found.")
            state['FILES']['RESTART_FILE_2'] =restart_filename


    # Project

    if not os.path.exists('DESIGNS'):
        os.mkdir('DESIGNS')
    
    ## User defined parameters
    maxIter             = 20            #use this many design iterations
    targetImprovement   = 0.8         #target a X change in objective function with the first step
    stepSizeDecrease    = 0.9           #decrease step size by this amount each iteration
    parameterization    = "NacaFourDigit"     #use this parameterization
    
    
    if parameterization == "NacaFourDigit":
        # [m, p, t]
        guess = np.array([1.0, 4.0, 12.0])
    elif parameterization == "BezierParsec":
        # [rle, b8, xt, yt, b15, dzte, betate, b0, b2, xc, yc, gammale, b17, zte, alphate]
        guess = np.array([.025, .015, .3, .06, .95, .001, .4, .05, .2, .3, .03, .5, .98, 0.00, .1])
    elif parameterization == "Kulfan4":
        # [au1, au2, ... ,al1, al2, ...]
        guess = np.array([.13,.18,.18,.13,.11,.11,.13,.13,.12,.11])
    
    f = open("designHistory.dat", "w")
    f.write("iter, designVariables, gradBeforeNorm, objFun \n")
    
    # change = 1.0
    for i in range(maxIter):
        
        dDir = "./DESIGNS/DSN_" + str(i).zfill(3)
        
        if not os.path.exists(dDir):
            os.mkdir(dDir)
        
        #Create the airfoil
        airfoilName = "airfoil" + str(i) + ".txt"
        jacX, jacY = airfoilGen.generateAirfoil(parameterization, guess, airfoilName)
        
        #Create the grid
        p2dName = "grid" + str(i) + ".txt"
        OGridGen.generateOGrid(airfoilName, p2dName)
        
        #Convert the grid to SU2 format
        os.system("python p3d2su2_Ogrid.py -f " + p2dName)
        
        #Move to the directory
        os.system("cp " + airfoilName + " " + dDir + "/")
        os.system("cp " + p2dName.split(".")[0] + ".su2" + " " + dDir + "/")
        
        #Run SU2 Direct
        config.MATH_PROBLEM = "DIRECT"
        config.MESH_FILENAME = p2dName.split(".")[0] + ".su2"
        config.TIME_MARCHING = 'NO' # Needed because of a bug in SU2.run.direct
        info = SU2.run.direct(config)
        state.update(info)
        os.system("cp config_CFD.cfg " + dDir + "/")
        os.system("cp forces_breakdown.dat " + dDir + "/")
        os.system("cp flow.* " + dDir + "/")
        os.system("cp restart_flow.dat " + dDir + "/restart_flow_DIRECT.dat")
        os.system("cp history_direct.csv " + dDir + "/history_direct.csv")
        
        #Prepare for adjoint run
        config.MATH_PROBLEM = "DISCRETE_ADJOINT"
        os.system("cp restart_flow.dat solution_flow.dat")
        
        #Run the adjoint
        info = SU2.run.CFD(config)
        os.system("cp config_CFD_AD.cfg " + dDir + "/")
        os.system("cp flow.vtu " + dDir + "/flow_AD.vtu")
        os.system("cp restart_adj_cd.dat " + dDir + "/restart_flow_DIRECT.dat")
        os.system("cp history.csv " + dDir + "/history_adjoint.csv")
        os.system("cp surface_adjoint.csv " + dDir + "/")
        
        #Compute the gradient
        gradX = np.array(pd.read_csv('surface_adjoint.csv')["Sensitivity_x"].values.tolist(), dtype=float)
        gradY = np.array(pd.read_csv('surface_adjoint.csv')["Sensitivity_y"].values.tolist(), dtype=float)
        designGrad = np.transpose(jacX).dot(gradX) + np.transpose(jacY).dot(gradY) 
        
        if parameterization == "BezierParsec": #don't want to update trailing edge thickness
            designGrad = jo.index_update( designGrad, 5, 0.0)
            
        f.write(str(i) + ", " + np.array_str(guess, max_line_width=1000000) + ", " + np.array_str(designGrad, max_line_width=1000000) + ", " +str(state.FUNCTIONS['DRAG']) +  "\n")
        f.flush()
        
        #Update the variables
        mag = np.linalg.norm(designGrad)
        designGrad = designGrad/mag*targetImprovement*(stepSizeDecrease**i)
        # designGrad = designGrad/mag*float(state.FUNCTIONS['DRAG'])*targetImprovement*(stepSizeDecrease**i)
        guess = np.array(guess) - designGrad
        
        
        

#: shape_optimization()


# -------------------------------------------------------------------
#  Run Main Program
# -------------------------------------------------------------------

# this is only accessed if running from command prompt
if __name__ == '__main__':
    main()

