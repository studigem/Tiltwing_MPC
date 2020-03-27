% -------------------------------------------------------------------------
% Tiltwing MPC
% -------------------------------------------------------------------------
clc;
clear all;
close all;

BEGIN_ACADO;
% Set Problem Name
acadoSet('problemname', 'tiltwing_MPC');
% Set Differential States
DifferentialState v_x v_z theta zeta_w;
% Set Controls
Control delta_w T_w theta_ref;
% Set Disturbance
Disturbance R;
% Model Discretisation and Horizon
Ts  = 0.1;
N = 50;

% Definition of parameters
m = 1.85; %Mass of the plane [kg]
g = 9.81; %Gravity [m/s^2]
rho = 1.225; %Density of Air [kg/m^3]
S_w = 0.0750;
k_T2L = 0.3;
k = 1;
tau_0 = 0.4;
alpha = atan(v_z/v_x);
c_w = pi/40;


%% Differential Equation
dot_v_x = 1/m*(T_w*cos(zeta_w)-m*g*sin(theta)-k_T2L*T_w*sin(zeta_w)...
    +0.5*rho*(v_x^2+v_z^2)*S_w*(C_Ltotal(alpha+zeta_w)*sin(alpha)...
    -C_Dtotal(alpha+zeta_w)*cos(alpha)))+k*(theta_ref - theta)/tau_0*v_z;
dot_v_z = 1/m*(-T_w*sin(zeta_w)-m*g*cos(theta)-k_T2L*T_w*cos(zeta_w)...
    -0.5*rho*(v_x^2+v_z^2)*S_w*(C_Ltotal(alpha+zeta_w)*cos(alpha)...
    -C_Dtotal(alpha+zeta_w)*sin(alpha)))-k*(theta_ref - theta)/tau_0*v_x;
dot_theta = k*(theta_ref - theta)/tau_0;
dot_zeta_w = c_w*delta_w;

% ode
f = acado.DifferentialEquation();
f.add(dot(v_x) == dot_v_x);
f.add(dot(v_z) == dot_v_z);
f.add(dot(theta) == dot_theta);
f.add(dot(zeta_w) == dot_zeta_w);


%% Optimal Control Problem
ocp = acado.OCP(0.0, N*Ts, N);

h={v_x, v_z, theta, zeta_w};

Q = eye(1);
Q(1,1) = 10;
Q(2,2) = 10;
Q(4,4) = 1;
%The reference
r = [1.0, 0.1, 0.0, 0.0];

ocp.minimizeLSQ(Q, h, r);

ocp.subjectTo( f );

ocp.subjectTo( 0.0 <= zeta_w <= pi/2);
ocp.subjectTo( -1.0 <= delta_w <= 1.0 );
ocp.subjectTo( 0.0 <= T_w <= 30.0 );
ocp.subjectTo( R == 0.0 );
%% SETTING UP THE (SIMULATED) PROCESS
mod = acado.DifferentialEquation();
mod.linkMatlabODE('tiltwing_ode');
identity = acado.OutputFcn();
dynamicSystem = acado.DynamicSystem(mod, identity);
process = acado.Process(dynamicSystem, 'INT_RK12');

%% SETTING UP THE MPC CONTROLLER:
algo = acado.RealTimeAlgorithm(ocp, 0.02);
algo.set('MAX_NUM_ITERATIONS', 5 );            

algo.set('INTEGRATOR_TYPE', 'INT_RK45');
algo.set( 'INTEGRATOR_TOLERANCE',   1e-4);    
algo.set( 'ABSOLUTE_TOLERANCE',     1e-3 );
    
zeroReference = acado.StaticReferenceTrajectory();

controller = acado.Controller( algo,zeroReference );

%% Setting up the simulation
sim = acado.SimulationEnvironment( 0.0, 10.0, process,controller );
r = [1.0, 0.1, 0.0, 0.0];

sim.init(r);

END_ACADO;

out = tiltwing_MPC_RUN();
