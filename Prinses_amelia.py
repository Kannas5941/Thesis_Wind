#!/usr/bin/env python
# coding: utf-8

# In[66]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import math
from sklearn import preprocessing
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from topfarm.recorders import TopFarmListRecorder
#Pywake packages
from py_wake.examples.data.hornsrev1 import V80 # The farm is comprised of 80 V80 turbines which
from py_wake.examples.data.hornsrev1 import Hornsrev1Site #  Horns Rev 1 site,
#from py_wake.examples.data.hornsrev1 import wt_x,wt_y # HornsRev 1 coordinates
from py_wake.examples.data.lillgrund import LillgrundSite
#from py_wake.examples.data.lillgrund import wt_x,wt_y
from py_wake.examples.data.lillgrund import SWT23
from py_wake import BastankhahGaussian
from py_wake.deficit_models.gaussian import ZongGaussian 
from py_wake.deficit_models.gaussian import IEA37SimpleBastankhahGaussian, NiayifarGaussian
from py_wake.turbulence_models.stf import STF2017TurbulenceModel
from py_wake.examples.data.iea34_130rwt import IEA34_130_1WT_Surrogate
from py_wake.examples.data.iea37 import IEA37Site
from py_wake.superposition_models import MaxSum
from amalia import AmaliaSite
from amalia import wt_x,wt_y 
#Topfarm packages
from topfarm.cost_models.cost_model_wrappers import AEPMaxLoadCostModelComponent
from topfarm.cost_models.cost_model_wrappers import CostModelComponent
from topfarm.cost_models.cost_model_wrappers import AEPCostModelComponent
from topfarm.easy_drivers import EasyScipyOptimizeDriver
from topfarm import TopFarmProblem
from topfarm.plotting import NoPlot, XYPlotComp
from topfarm.constraint_components.boundary import CircleBoundaryConstraint
from topfarm.constraint_components.boundary import XYBoundaryConstraint
from topfarm.constraint_components.spacing import SpacingConstraint
from topfarm.cost_models.py_wake_wrapper import PyWakeAEPCostModelComponent


# In[67]:


# Scaled coordinates of amalia wind farm 
wt_nx = np.zeros(np.size(wt_x))
wt_ny = np.zeros(np.size(wt_y))
for i in range(len(wt_x)):
    wt_nx[i] = wt_x[i]*(1.625)
    wt_ny[i] = wt_y[i]*(1.625)
coordinates_nx = np.zeros((len(wt_nx),2))
for i in range (len(coordinates_nx)):
    coordinates_nx[i] = [wt_nx[i],wt_ny[i]]


# In[68]:


# Get wind resource data from Global wind atlas site
from py_wake.site._site import UniformWeibullSite
from py_wake.site._site import UniformWeibullSite, UniformSite
from py_wake.site.xrsite import GlobalWindAtlasSite
gw_site = GlobalWindAtlasSite(52.4935,4.5251,97.5,0.05,ti=0.1) # Get lat, Lon, hu height


# In[69]:


# Define Amalia site class 
class PAmaliasite(UniformWeibullSite):
    def __init__(self, shear=None):
        a = gw_site.ds['Weibull_A'].values
        k = gw_site.ds['Weibull_k'].values
        f = gw_site.ds['Sector_frequency'].values
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f),a,k, .1, shear=shear)
        self.initial_position = np.array([wt_nx, wt_ny]).T


# In[70]:


# Define Wind farm model with IEA3.4Mw turbine
site = PAmaliasite()
wt = IEA34_130_1WT_Surrogate()
windFarmModel = BastankhahGaussian(site, wt, 
                turbulenceModel=STF2017TurbulenceModel(addedTurbulenceSuperpositionModel=MaxSum()))
# site.plot_wd_distribution() #Just to plot wind speed distributions


# In[71]:


# # Plot for Scaled Amalia wind farm
# plt.figure()
# wt.plot(wt_nx,wt_ny)
# plt.title('Amalia Wind Farm scaled',fontsize = 12)


# In[72]:


# Main Funciton to calculate aep, area and generate convex hull vertices

def get_optimize(x,y):
    '''
    This function calcuates aep,area of a wind farm model by taking the x,y coordinates 
    as input arguments.
    Parameters
    ----------
    x[list] : X coordinates in mts
    y[list] : Y coordinates in mts
    Returns
    -------
    aep_a[int] : Annual Energy production of final wind farm layout in Gwh
    area[int] : Area of a wind farm in mt2
    con_vert[list] : A list for convex hull vertices of the wind farm 

    '''
    res = windFarmModel(x,y)
    aep_a = res.aep().sum()
    coordinates = np.zeros((len(x),2))
    for i in range (len(coordinates)):
        coordinates[i] = [x[i],y[i]]
    points = coordinates   # Wind farm points
    hull = ConvexHull(points)
    vert_indx = np.sort(hull.vertices)
    con_vert = coordinates[vert_indx]
    area = hull.volume
    return aep_a,area,con_vert,res
def get_aep(x,y):
    res = windFarmModel(x,y)
    aep_a = res.aep().sum()
    return aep_a
def get_area(x,y):
    aep_a,area,con_vert,res = get_optimize(x,y)
    print(aep_a)
    return [area,aep_a]
def get_con_vert(x,y):
    aep_a,area,con_vert,res = get_optimize(x,y)
    return con_vert
def get_loads(x,y):
    aep_a,area,con_vert,res = get_optimize(x,y)
    loads = res.loads('OneWT')['LDEL'].values
    return [aep_a,area,loads]
def get_ene_den(x,y):
    aep_a,area,con_vert,res = get_optimize(x,y)
    aep_a = aep_a*1e3 # GwH to MwH
    area = area*1e-6 # m2 to Km2
    ed = aep_a/area
    loads = res.loads('OneWT')['LDEL'].values
    return [ed,aep_a,area,loads]


# In[73]:


# Topfarm problem to setup AEP optimizations 
n_wt = 60
D = 130 # Diameter of 3.4Mw turbine
maxiter = 100
tol = 1e-6
step = 1e-3
ap=0 
min_spacing = 3*D
boundary = get_con_vert(wt_nx,wt_ny)
# boundary = site.boundary
aep_comp1 = CostModelComponent(input_keys=[('x', wt_nx),('y', wt_ny)],
                              n_wt=n_wt,
                              cost_function=get_aep,
                              objective=True,
                              maximize=True, 
                              step={'x': step, 'y': step},
                              output_keys=[('aep_a',ap)])
                              
problem1 = TopFarmProblem(design_vars={'x':wt_nx,'y':wt_ny},
                          cost_comp=aep_comp1,
                          constraints=[XYBoundaryConstraint(boundary),
                                     SpacingConstraint(min_spacing)],
                          driver=EasyScipyOptimizeDriver(optimizer='SLSQP',maxiter=maxiter,tol=tol),
                          plot_comp=XYPlotComp(),
                          expected_cost=1e-4)


# In[74]:


# cost1, state1, recorder1 = problem1.optimize()


# In[75]:


# recorder1.save('Amalia_Aep_site')


# In[76]:


# Load the results from optimization1 using pkl file 
Amalia_s1 = './recordings/Amalia_Aep_site'


# In[77]:


from topfarm.recorders import TopFarmListRecorder
recording = TopFarmListRecorder().load(Amalia_s1)


# In[78]:


# AEP values for each iteration
aep_op = recording['aep_a']


# In[79]:


# # Plot to show aep maximization for every iteration
# plt.figure()
# plt.plot(aep_op,label='AEP in Gwh')
# plt.xlabel('No of iterations')
# plt.ylabel('AEP in (Gwh)')
# plt.title('AEP maximization with boundary constraint')
# plt.legend()
# plt.show()


# In[80]:


# Inputs for next optimization by taking the maximum aep
aep_s1 = max(aep_op)
indmax = np.where(aep_op==max(aep_op))
wt_areax = recording['x'][indmax][0]
wt_areay = recording['y'][indmax][0]


# In[81]:


# XY boundary coordinates
xy_bound = recording['xy_boundary'][0]
x_bound = np.zeros(len(xy_bound))
y_bound = np.zeros(len(xy_bound))
for i in range(len(xy_bound)):
    x_bound[i] = xy_bound[i][0]
    y_bound[i] = xy_bound[i][1]


# In[82]:


# # Plot showing final optimised layout after AEP optimization
# plt.figure()
# plt.scatter(wt_areax,wt_areay,label = 'Optimised layout',marker = '*')
# plt.plot(x_bound,y_bound,label = 'farm boundary',color ='k')
# plt.title('Optimised layout for AEP optimization')
# plt.xlabel('X[m]')
# plt.ylabel('Y[m]')
# plt.legend()


# In[83]:


# Topfarm problem to setup Area optimizations with minimum area
n_wt = 60
D = 130 # Diameter of 3.4Mw turbine
maxiter = 100
tol = 1e-4
# tr = [4*D,4*D,0] # Initial conditions 
step = 1e-2
# lb = [3*D,3*D,0] # Lower bound 
# ub = [8*D,8*D,90] # Upper bound
ap=0 
min_spacing = 3*D
min_aep = 0.99*aep_s1
boundary2 = get_con_vert(wt_areax,wt_areay)
aep_comp2 = CostModelComponent(input_keys=[('x', wt_areax),('y', wt_areay)],
                              n_wt=n_wt,
                              cost_function=get_area,
                              objective=True, 
                              step={'x': step, 'y': step},
                              output_keys=[('area',0),('aep_a',ap)])
                              
problem2 = TopFarmProblem(design_vars={'x':wt_areax,'y':wt_areay},
                          cost_comp=aep_comp2,
                          constraints=[XYBoundaryConstraint(boundary2),
                                     SpacingConstraint(min_spacing)],
                          post_constraints=[('aep_a',{'lower':min_aep})],
                          driver=EasyScipyOptimizeDriver(optimizer='SLSQP',maxiter=maxiter,tol=tol),
                          expected_cost=1e2)
# print(min_aep)


# In[84]:


# cost2, state2, recorder2 = problem2.optimize()


# In[85]:


# recorder2.save('Amalia_Area_opt')


# In[86]:


min_aep = 0.99*aep_s1
# Load results from area optimization using pkl file
Amalia_s2 = './recordings/Amalia_Area_opt'


# In[87]:


recording2 = TopFarmListRecorder().load(Amalia_s2)


# In[88]:


# Getting the index of min_aep
aep_s2 = recording2['aep_a']
aep_s2 = np.round(aep_s2,3)
min_aep = round(min_aep,3)


# In[89]:


# Getting area index that satisfies both aep constraint and also have minimum area
area_s2 = recording2['area']
min_aep_indx = np.where(aep_s2 == min_aep)
min_area = min(area_s2[min_aep_indx])
min_area_indx = np.where(area_s2 == min_area)


# In[90]:


# Assigning X,Y coordinates for minimum area index
wt_loadx = recording2['x'][min_area_indx][0]
wt_loady = recording2['y'][min_area_indx][0]


# In[91]:


# Boundary at the begining of area optimization
xy_a_bound = recording2['xy_boundary'][0]
x_a_bound = np.zeros(len(xy_a_bound))
y_a_bound = np.zeros(len(xy_a_bound))
for i in range(len(xy_a_bound)):
    x_a_bound[i] = xy_a_bound[i][0]
    y_a_bound[i] = xy_a_bound[i][1]


# In[92]:


# Boundary for the optimised layout
boundary3 = get_con_vert(wt_loadx,wt_loady)
bound_lx = np.zeros(len(boundary3))
bound_ly = np.zeros(len(boundary3))
for i in range(len(boundary3)):
    bound_lx[i] = boundary3[i][0]
    bound_ly[i] = boundary3[i][1]


# In[93]:


# Scaleup the boundary3 by 5% to match maximum area criteria
from shapely import affinity
farm = Polygon(boundary3)
farm_l = affinity.scale(farm,xfact=1.05,yfact=1.05,origin=(boundary3[0][0],boundary3[0][1]))
farm_l_coord = farm_l.exterior.coords[:-1]
farm_lx = np.zeros(len(farm_l_coord))
farm_ly = np.zeros(len(farm_l_coord))
for i in range(len(farm_lx)):
    farm_lx[i] = farm_l_coord[i][0]
    farm_ly[i] = farm_l_coord[i][1]


# In[ ]:


# # Plot to show variation of AEP during Area minimization
# plt.figure()
# plt.plot(recording2['aep_a'],label ='AEP in Gwh')
# plt.xlabel('Iterations')
# plt.ylabel('AEP in (Gwh)')
# plt.title('Variation of AEP during Area minimization')
# plt.legend()
# plt.show()


# In[ ]:


# # Plot to show variation of Area during Area minimization
# plt.figure()
# plt.plot(recording2['area'],label ='Area in m2')
# plt.xlabel('Iterations')
# plt.ylabel('Area in m2')
# plt.title('Variation of Area during Area minimization')
# plt.legend()


# In[ ]:


# # Optimised wind farm layout after Area optimization
# plt.figure()
# plt.scatter(wt_loadx,wt_loady,label = 'Optimised layout',marker = 'o')
# plt.plot(x_a_bound,y_a_bound,label = 'farm boundary',color ='r')
# plt.title('Area optimization')
# plt.xlabel('X[m]')
# plt.ylabel('Y[m]')
# plt.title('Optimised wind farm layout with Area optimization ')
# plt.legend()


# In[94]:


# Define topfarm problem input parameters for AEP optimization with load constraints
n_wt = 60
D = 130 # Diameter of 3.4Mw turbine
maxiter = 50
tol = 1e-4
step = 1e-2
ap=0
min_spacing = 3*D


# In[ ]:


# # Calculate maximum loads for the wind farm layout
# load_fact = 1.03
# aep, area, nom_loads = get_loads(wt_loadx,wt_loady)
# max_loads = (nom_loads*load_fact)
# s = nom_loads.shape[0]
# load_signals = ['del_blade_flap', 'del_blade_edge', 'del_tower_bottom_fa',
#                 'del_tower_bottom_ss', 'del_tower_top_torsion']
# output_loads = np.zeros((5,n_wt))
# print(max_loads)


# In[ ]:


# # Topfarm problem to setup AEP optimizations with load constraints
# aep_comp3 = CostModelComponent(input_keys=[('x', wt_loadx),('y', wt_loady)],
#                               n_wt=n_wt,
#                               cost_function=get_loads,
#                               objective=True, 
#                               maximize= True,
#                               step={'x': step, 'y': step},
#                               output_keys=[('aep_a',0),('area',0),('loads',output_loads)])
                              
# problem3 = TopFarmProblem(design_vars={'x':wt_loadx,'y':wt_loady},
#                           cost_comp=aep_comp3,
#                           constraints=[XYBoundaryConstraint(farm_l_coord,'polygon'),
#                                      SpacingConstraint(min_spacing)],
#                           post_constraints=[('loads',{'upper':max_loads})],
#                           driver=EasyScipyOptimizeDriver(optimizer='SLSQP',maxiter=maxiter,tol=tol),
#                           expected_cost=1e-4
#                           )
# cost3,state3,recorder3 = problem3.optimize()
# recorder3.save('Load_opt_SLSQP50')


# In[ ]:


# maxiter = 40
# problem3 = TopFarmProblem(design_vars={'x':state3['x'],'y':state3['y']},
#                           cost_comp=aep_comp3,
#                           constraints=[XYBoundaryConstraint(farm_l_coord,'polygon'),
#                                      SpacingConstraint(min_spacing)],
#                           post_constraints=[('loads',{'upper':max_loads})],
#                           driver=EasyScipyOptimizeDriver(optimizer='SLSQP',maxiter=maxiter,tol=tol),
#                           expected_cost=1e-4
#                           )
# recorder3.save('Load_opt_SLSQP40')


# In[95]:


Amalia_s3 = './recordings/Load_opt_SLSQP65.pkl'


# In[96]:


recording3 = TopFarmListRecorder().load(Amalia_s3)


# In[ ]:


# plt.figure()
# plt.plot(recording3['aep_a'],label ='AEP in Gwh')
# plt.xlabel('Iterations')
# plt.ylabel('AEP in (Gwh)')
# plt.title('AEP during AEP maximization under load constraints')
# plt.legend()
# plt.show()


# In[ ]:


# plt.figure()
# plt.plot(recording3['area'],label ='Area in m2')
# plt.xlabel('Iterations')
# plt.ylabel('Area in (m2)')
# plt.title('Area during AEP opt under loads constraints')
# plt.legend()
# plt.show()


# In[ ]:


# # Plot to show Wind farm layout during different iterations
# plt.figure()
# plt.scatter(recording3['x'][0],recording3['y'][0],label='x,y 1st iteration',marker='*')
# plt.scatter(farm_lx,farm_ly,label='Farm Boundary',marker='o')
# # plt.scatter(bound_lx,bound_ly,label='Initial Boundary')
# # plt.scatter(recording3['x'][1],recording3['y'][1],label='x,y 2nd', marker='*')
# # plt.scatter(recording3['x'][2],recording3['y'][2],label='x,y 3rd', marker='_')
# plt.scatter(recording3['x'][-1],recording3['y'][-1],label='x,y 65th', marker='o')
# plt.xlabel('X[m]')
# plt.ylabel('Y[m]')
# plt.title('AEP Optimization Under Load Constraints with SLSQP')
# #plt.plot(wt_nx,wt_ny,'2b')
# #plt.plot(wt_areax,wt_areay,'2g')
# plt.legend()
# plt.show()


# In[ ]:


# # Plot to show Loads during AEP optimization under load constraints
# n_i = recording3['counter'].size
# loads = recording3['loads'].reshape((n_i,s,n_wt))
# wt = 0
# for n, ls in enumerate(load_signals):
#   plt.plot(loads[:,n,wt],label='optimised loads')
#   plt.title(ls+f' turbine {wt}')
#   plt.plot([0, n_i], [max_loads[n, wt], max_loads[n, wt]],label='Maximum loads')
#   plt.xlabel("No of iterations")
#   plt.ylabel("Loads")
#   plt.legend()
#   plt.show()


# In[97]:


# Get maximum AEP index and boundary for energy denisty optimisation
aep_max_indx = np.where(recording3['aep_a'] ==np.max(recording3['aep_a']))
wt_edx = recording3['x'][aep_max_indx][0]
wt_edy = recording3['y'][aep_max_indx][0]
boundary4 = get_con_vert(wt_edx,wt_edy)


# In[98]:


# Maximum loads for Energy density optimization
load_fact = 1.03
aep, area, nom_loads = get_loads(wt_edx,wt_edy)
max_loads = (nom_loads*load_fact)
s = nom_loads.shape[0]
load_signals = ['del_blade_flap', 'del_blade_edge', 'del_tower_bottom_fa',
                'del_tower_bottom_ss', 'del_tower_top_torsion']
output_loads = np.zeros((5,n_wt))


# In[99]:


# Topfarm problem parameters
n_wt = 60
D = 130 # Diameter of 3.4Mw turbine
maxiter = 2
tol = 1e-4
step = 1e-2
ap=0
min_spacing = 3*D


# In[108]:


# Topfarm problem to setup Energy Density optimizations with load constraints
aep_comp4 = CostModelComponent(input_keys=[('x', wt_edx),('y', wt_edy)],
                              n_wt=n_wt,
                              cost_function=get_ene_den,
                              objective=True,
                              maximize=True,
                              step={'x': step, 'y': step},
                              output_keys=[('ed',0),('aep_a',0),('area',0),('loads',output_loads)])
                              
problem4 = TopFarmProblem(design_vars={'x':wt_edx,'y':wt_edy},
                          cost_comp=aep_comp4,
                          constraints=[XYBoundaryConstraint(boundary4),
                                     SpacingConstraint(min_spacing)],
                          post_constraints=[('loads',{'upper':max_loads})],
                          driver=EasyScipyOptimizeDriver(optimizer='SLSQP',maxiter=maxiter,tol=tol),
                          expected_cost=10
                          )


# In[102]:


cost4,state4,recorder4 = problem4.optimize()
recorder4.save('ed_opt_SLSQP60_AEP1')


# In[103]:


# Amalia_s4 = './recordings/ed_opt_AEPMax2.pkl'
# recording4 = TopFarmListRecorder().load(Amalia_s4)


# In[ ]:


# get_ene_den(wt_edx,wt_edy)


# In[104]:


# plt.figure()
# plt.plot(recording4['aep_a'],label ='AEP in Gwh')
# plt.xlabel('Iterations')
# plt.ylabel('AEP(Gwh)')
# plt.title('AEP during Energy density opt under loads')
# plt.legend()
# plt.show()


# In[105]:


# plt.figure()
# plt.plot(recording4['area'],label ='Area in m2')
# plt.xlabel('Iterations')
# plt.ylabel('Area in (m2)')
# plt.title('Area during Energy density opt under loads')
# plt.legend()
# plt.show()


# In[ ]:


# plt.figure()
# plt.plot(recording4['ed'],label ='Ed in m2')
# plt.xlabel('Iterations')
# plt.ylabel('Ed in (m2)')
# plt.title('Ed during Energy density opt under loads')
# plt.legend()
# plt.show()


# In[107]:


# # Plot to show Wind farm layout during different iterations
# plt.figure()
# plt.scatter(recording4['x'][0],recording4['y'][0],label='x,y 1st iteration',marker='*')
# # plt.scatter(farm_lx,farm_ly,label='Farm Boundary',marker='o')
# # plt.scatter(bound_lx,bound_ly,label='Initial Boundary')
# # plt.scatter(recording3['x'][1],recording3['y'][1],label='x,y 2nd', marker='*')
# # plt.scatter(recording3['x'][2],recording3['y'][2],label='x,y 3rd', marker='_')
# plt.scatter(recording4['x'][-1],recording4['y'][-1],label='x,y 2nd', marker='o')
# plt.xlabel('X[m]')
# plt.ylabel('Y[m]')
# plt.title('AEP Optimization Under Load Constraints with SLSQP')
# #plt.plot(wt_nx,wt_ny,'2b')
# #plt.plot(wt_areax,wt_areay,'2g')
# plt.legend()
# plt.show()

