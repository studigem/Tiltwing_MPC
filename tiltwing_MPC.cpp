/*
*    This file is part of ACADO Toolkit.
*
*    ACADO Toolkit -- A Toolkit for Automatic Control and Dynamic Optimization.
*    Copyright (C) 2008-2009 by Boris Houska and Hans Joachim Ferreau, K.U.Leuven.
*    Developed within the Optimization in Engineering Center (OPTEC) under
*    supervision of Moritz Diehl. All rights reserved.
*
*    ACADO Toolkit is free software; you can redistribute it and/or
*    modify it under the terms of the GNU Lesser General Public
*    License as published by the Free Software Foundation; either
*    version 3 of the License, or (at your option) any later version.
*
*    ACADO Toolkit is distributed in the hope that it will be useful,
*    but WITHOUT ANY WARRANTY; without even the implied warranty of
*    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
*    Lesser General Public License for more details.
*
*    You should have received a copy of the GNU Lesser General Public
*    License along with ACADO Toolkit; if not, write to the Free Software
*    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*
*/


/**
*    Author David Ariens, Rien Quirynen
*    Date 2009-2013
*    http://www.acadotoolkit.org/matlab 
*/

#include <acado_optimal_control.hpp>
#include <acado_toolkit.hpp>
#include <acado/utils/matlab_acado_utils.hpp>

USING_NAMESPACE_ACADO

mxArray* ModelFcn_3_f = NULL;
mxArray* ModelFcn_3_jac = NULL;
mxArray* ModelFcn_3T  = NULL;
mxArray* ModelFcn_3X  = NULL;
mxArray* ModelFcn_3XA = NULL;
mxArray* ModelFcn_3U  = NULL;
mxArray* ModelFcn_3P  = NULL;
mxArray* ModelFcn_3W  = NULL;
mxArray* ModelFcn_3DX = NULL;
unsigned int ModelFcn_3NT  = 0;
unsigned int ModelFcn_3NX  = 0;
unsigned int ModelFcn_3NXA = 0;
unsigned int ModelFcn_3NU  = 0;
unsigned int ModelFcn_3NP  = 0;
unsigned int ModelFcn_3NW  = 0;
unsigned int ModelFcn_3NDX = 0;
unsigned int jacobianNumber_3 = -1;
double* f_store_3             = NULL;
double* J_store_3             = NULL;

void clearAllGlobals3( ){ 
    if ( f_store_3 != NULL ){
        f_store_3 = NULL;
    }

    if ( J_store_3 != NULL ){
        J_store_3 = NULL;
    }

    if ( ModelFcn_3_f != NULL ){
        mxDestroyArray( ModelFcn_3_f );
        ModelFcn_3_f = NULL;
    }

    if ( ModelFcn_3T != NULL ){
        mxDestroyArray( ModelFcn_3T );
        ModelFcn_3T = NULL;
    }

    if ( ModelFcn_3X != NULL ){
        mxDestroyArray( ModelFcn_3X );
        ModelFcn_3X = NULL;
    }

    if ( ModelFcn_3XA != NULL ){
        mxDestroyArray( ModelFcn_3XA );
        ModelFcn_3XA = NULL;
    }

    if ( ModelFcn_3U != NULL ){
        mxDestroyArray( ModelFcn_3U );
        ModelFcn_3U = NULL;
    }

    if ( ModelFcn_3P != NULL ){
        mxDestroyArray( ModelFcn_3P );
        ModelFcn_3P = NULL;
    }

    if ( ModelFcn_3W != NULL ){
        mxDestroyArray( ModelFcn_3W );
        ModelFcn_3W = NULL;
    }

    if ( ModelFcn_3DX != NULL ){
        mxDestroyArray( ModelFcn_3DX );
        ModelFcn_3DX = NULL;
    }

    if ( ModelFcn_3_jac != NULL ){
        mxDestroyArray( ModelFcn_3_jac );
        ModelFcn_3_jac = NULL;
    }

    ModelFcn_3NT  = 0;
    ModelFcn_3NX  = 0;
    ModelFcn_3NXA = 0;
    ModelFcn_3NU  = 0;
    ModelFcn_3NP  = 0;
    ModelFcn_3NW  = 0;
    ModelFcn_3NDX = 0;
    jacobianNumber_3 = -1;
}

void genericODE3( double* x, double* f, void *userData ){
    unsigned int i;
    double* tt = mxGetPr( ModelFcn_3T );
    tt[0] = x[0];
    double* xx = mxGetPr( ModelFcn_3X );
    for( i=0; i<ModelFcn_3NX; ++i )
        xx[i] = x[i+1];
    double* uu = mxGetPr( ModelFcn_3U );
    for( i=0; i<ModelFcn_3NU; ++i )
        uu[i] = x[i+1+ModelFcn_3NX];
    double* pp = mxGetPr( ModelFcn_3P );
    for( i=0; i<ModelFcn_3NP; ++i )
        pp[i] = x[i+1+ModelFcn_3NX+ModelFcn_3NU];
    double* ww = mxGetPr( ModelFcn_3W );
    for( i=0; i<ModelFcn_3NW; ++i )
        ww[i] = x[i+1+ModelFcn_3NX+ModelFcn_3NU+ModelFcn_3NP];
    mxArray* FF = NULL;
    mxArray* argIn[]  = { ModelFcn_3_f,ModelFcn_3T,ModelFcn_3X,ModelFcn_3U,ModelFcn_3P,ModelFcn_3W };
    mxArray* argOut[] = { FF };

    mexCallMATLAB( 1,argOut, 6,argIn,"generic_ode" );
    double* ff = mxGetPr( *argOut );
    for( i=0; i<ModelFcn_3NX; ++i ){
        f[i] = ff[i];
    }
    mxDestroyArray( *argOut );
}

void genericJacobian3( int number, double* x, double* seed, double* f, double* df, void *userData  ){
    unsigned int i, j;
    double* ff;
    double* J;
    if (J_store_3 == NULL){
        J_store_3 = (double*) calloc ((ModelFcn_3NX+ModelFcn_3NU+ModelFcn_3NP+ModelFcn_3NW)*(ModelFcn_3NX),sizeof(double));
        f_store_3 = (double*) calloc (ModelFcn_3NX,sizeof(double));
    }
    if ( (int) jacobianNumber_3 == number){
        J = J_store_3;
        ff = f_store_3;
        for( i=0; i<ModelFcn_3NX; ++i ) {
            df[i] = 0;
            f[i] = 0;
            for (j=0; j < ModelFcn_3NX+ModelFcn_3NU+ModelFcn_3NP+ModelFcn_3NW; ++j){
                df[i] += J[(j*(ModelFcn_3NX))+i]*seed[j+1]; 
            }
        }
        for( i=0; i<ModelFcn_3NX; ++i ){
            f[i] = ff[i];
        }
    }else{
        jacobianNumber_3 = number; 
        double* tt = mxGetPr( ModelFcn_3T );
        tt[0] = x[0];
        double* xx = mxGetPr( ModelFcn_3X );
        for( i=0; i<ModelFcn_3NX; ++i )
            xx[i] = x[i+1];
        double* uu = mxGetPr( ModelFcn_3U );
        for( i=0; i<ModelFcn_3NU; ++i )
            uu[i] = x[i+1+ModelFcn_3NX];
        double* pp = mxGetPr( ModelFcn_3P );
        for( i=0; i<ModelFcn_3NP; ++i )
            pp[i] = x[i+1+ModelFcn_3NX+ModelFcn_3NU];
        double* ww = mxGetPr( ModelFcn_3W );
            for( i=0; i<ModelFcn_3NW; ++i )
        ww[i] = x[i+1+ModelFcn_3NX+ModelFcn_3NU+ModelFcn_3NP];
        mxArray* FF = NULL;
        mxArray* argIn[]  = { ModelFcn_3_jac,ModelFcn_3T,ModelFcn_3X,ModelFcn_3U,ModelFcn_3P,ModelFcn_3W };
        mxArray* argOut[] = { FF };
        mexCallMATLAB( 1,argOut, 6,argIn,"generic_jacobian" );
        unsigned int rowLen = mxGetM(*argOut);
        unsigned int colLen = mxGetN(*argOut);
        if (rowLen != ModelFcn_3NX){
            mexErrMsgTxt( "ERROR: Jacobian matrix rows do not match (should be ModelFcn_3NX). " );
        }
        if (colLen != ModelFcn_3NX+ModelFcn_3NU+ModelFcn_3NP+ModelFcn_3NW){
            mexErrMsgTxt( "ERROR: Jacobian matrix columns do not match (should be ModelFcn_3NX+ModelFcn_3NU+ModelFcn_3NP+ModelFcn_3NW). " );
        }
        J = mxGetPr( *argOut );
        memcpy(J_store_3, J, (ModelFcn_3NX+ModelFcn_3NU+ModelFcn_3NP+ModelFcn_3NW)*(ModelFcn_3NX) * sizeof ( double ));
        for( i=0; i<ModelFcn_3NX; ++i ) {
            df[i] = 0;
            f[i] = 0;
            for (j=0; j < ModelFcn_3NX+ModelFcn_3NU+ModelFcn_3NP+ModelFcn_3NW; ++j){
                df[i] += J[(j*(ModelFcn_3NX))+i]*seed[j+1];
            }
        }
        mxArray* FF2 = NULL;
        mxArray* argIn2[]  = { ModelFcn_3_f,ModelFcn_3T,ModelFcn_3X,ModelFcn_3U,ModelFcn_3P,ModelFcn_3W };
        mxArray* argOut2[] = { FF2 };
        mexCallMATLAB( 1,argOut2, 6,argIn2,"generic_ode" );
        ff = mxGetPr( *argOut2 );
        memcpy(f_store_3, ff, (ModelFcn_3NX) * sizeof ( double ));
        for( i=0; i<ModelFcn_3NX; ++i ){
            f[i] = ff[i];
        }
        mxDestroyArray( *argOut );
        mxDestroyArray( *argOut2 );
    }
}
#include <mex.h>


void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] ) 
 { 
 
    MatlabConsoleStreamBuf mybuf;
    RedirectStream redirect(std::cout, mybuf);
    clearAllStaticCounters( ); 
 
    mexPrintf("\nACADO Toolkit for Matlab - Developed by David Ariens and Rien Quirynen, 2009-2013 \n"); 
    mexPrintf("Support available at http://www.acadotoolkit.org/matlab \n \n"); 

    if (nrhs != 0){ 
      mexErrMsgTxt("This problem expects 0 right hand side argument(s) since you have defined 0 MexInput(s)");
    } 
 
    TIME autotime;
    DifferentialState v_x;
    DifferentialState v_z;
    DifferentialState theta;
    DifferentialState zeta_w;
    Control delta_w;
    Control T_w;
    Control theta_ref;
    Disturbance R;
    Function acadodata_f2;
    acadodata_f2 << v_x;
    acadodata_f2 << v_z;
    acadodata_f2 << theta;
    acadodata_f2 << zeta_w;
    DMatrix acadodata_M1;
    acadodata_M1.read( "tiltwing_MPC_data_acadodata_M1.txt" );
    DVector acadodata_v1(4);
    acadodata_v1(0) = 1;
    acadodata_v1(1) = 1.000000E-01;
    acadodata_v1(2) = 0;
    acadodata_v1(3) = 0;
    DVector acadodata_v2(4);
    acadodata_v2(0) = 1;
    acadodata_v2(1) = 1.000000E-01;
    acadodata_v2(2) = 0;
    acadodata_v2(3) = 0;
    DVector acadodata_v3(4);
    acadodata_v3(0) = 1;
    acadodata_v3(1) = 1.000000E-01;
    acadodata_v3(2) = 0;
    acadodata_v3(3) = 0;
    DifferentialEquation acadodata_f1;
    acadodata_f1 << dot(v_x) == ((((((atan(1/v_x*v_z)+zeta_w)*6.29999999999999982236e+00+2.00000000000000011102e-01)*(1.00000000000000000000e+00-1/(1.00000000000000000000e+00+exp((-2.30000000000000009992e-01+atan(1/v_x*v_z)+zeta_w)*(-4.50000000000000000000e+01))))+1/(1.00000000000000000000e+00+exp((-2.30000000000000009992e-01+atan(1/v_x*v_z)+zeta_w)*(-4.50000000000000000000e+01)))*6.99999999999999955591e-01*sin((atan(1/v_x*v_z)+zeta_w)*2.00000000000000000000e+00))*sin(atan(1/v_x*v_z))-((1.00000000000000000000e+00-1/(1.00000000000000000000e+00+exp((-2.30000000000000009992e-01+atan(1/v_x*v_z)+zeta_w)*(-4.50000000000000000000e+01))))*(1.00000000000000002082e-02+pow((atan(1/v_x*v_z)+zeta_w),2.00000000000000000000e+00))+1/(1.00000000000000000000e+00+exp((-2.30000000000000009992e-01+atan(1/v_x*v_z)+zeta_w)*(-4.50000000000000000000e+01)))*(1.00000000000000002082e-02+1.18999999999999994671e+00*pow(sin((atan(1/v_x*v_z)+zeta_w)),2.00000000000000000000e+00)))*cos(atan(1/v_x*v_z)))*(pow(v_x,2.00000000000000000000e+00)+pow(v_z,2.00000000000000000000e+00))*6.12500000000000044409e-01*7.49999999999999972244e-02-1.00000000000000005551e-01*T_w*sin(zeta_w)-1.81485000000000020748e+01*sin(theta)+T_w*cos(zeta_w))*5.40540540540540459524e-01+(-theta+theta_ref)*v_z);
    acadodata_f1 << dot(v_z) == ((((((atan(1/v_x*v_z)+zeta_w)*6.29999999999999982236e+00+2.00000000000000011102e-01)*(1.00000000000000000000e+00-1/(1.00000000000000000000e+00+exp((-2.30000000000000009992e-01+atan(1/v_x*v_z)+zeta_w)*(-4.50000000000000000000e+01))))+1/(1.00000000000000000000e+00+exp((-2.30000000000000009992e-01+atan(1/v_x*v_z)+zeta_w)*(-4.50000000000000000000e+01)))*6.99999999999999955591e-01*sin((atan(1/v_x*v_z)+zeta_w)*2.00000000000000000000e+00))*cos(atan(1/v_x*v_z))-((1.00000000000000000000e+00-1/(1.00000000000000000000e+00+exp((-2.30000000000000009992e-01+atan(1/v_x*v_z)+zeta_w)*(-4.50000000000000000000e+01))))*(1.00000000000000002082e-02+pow((atan(1/v_x*v_z)+zeta_w),2.00000000000000000000e+00))+1/(1.00000000000000000000e+00+exp((-2.30000000000000009992e-01+atan(1/v_x*v_z)+zeta_w)*(-4.50000000000000000000e+01)))*(1.00000000000000002082e-02+1.18999999999999994671e+00*pow(sin((atan(1/v_x*v_z)+zeta_w)),2.00000000000000000000e+00)))*sin(atan(1/v_x*v_z)))*(pow(v_x,2.00000000000000000000e+00)+pow(v_z,2.00000000000000000000e+00))*6.12500000000000044409e-01*7.49999999999999972244e-02+(-T_w)*sin(zeta_w)-1.00000000000000005551e-01*T_w*cos(zeta_w)-1.81485000000000020748e+01*cos(theta))*5.40540540540540459524e-01-(-theta+theta_ref)*v_x);
    acadodata_f1 << dot(theta) == (-theta+theta_ref);
    acadodata_f1 << dot(zeta_w) == 7.85398163397448278999e-02*delta_w;

    OCP ocp1(0, 5, 50);
    ocp1.minimizeLSQ(acadodata_M1, acadodata_f2, acadodata_v2);
    ocp1.subjectTo(acadodata_f1);
    ocp1.subjectTo(0.00000000000000000000e+00 <= zeta_w <= 1.57079632679489655800e+00);
    ocp1.subjectTo((-1.00000000000000000000e+00) <= delta_w <= 1.00000000000000000000e+00);
    ocp1.subjectTo(0.00000000000000000000e+00 <= T_w <= 1.50000000000000000000e+01);
    ocp1.subjectTo(R == 0.00000000000000000000e+00);


    ModelFcn_3T  = mxCreateDoubleMatrix( 1, 1,mxREAL );
    ModelFcn_3X  = mxCreateDoubleMatrix( 4, 1,mxREAL );
    ModelFcn_3XA = mxCreateDoubleMatrix( 0, 1,mxREAL );
    ModelFcn_3DX = mxCreateDoubleMatrix( 4, 1,mxREAL );
    ModelFcn_3U  = mxCreateDoubleMatrix( 3, 1,mxREAL );
    ModelFcn_3P  = mxCreateDoubleMatrix( 0, 1,mxREAL );
    ModelFcn_3W  = mxCreateDoubleMatrix( 1, 1,mxREAL );
    ModelFcn_3NT  = 1;
    ModelFcn_3NX  = 4;
    ModelFcn_3NXA = 0;
    ModelFcn_3NDX = 4;
    ModelFcn_3NP  = 0;
    ModelFcn_3NU  = 3;
    ModelFcn_3NW  = 1;
    DifferentialEquation acadodata_f3;
    ModelFcn_3_f = mxCreateString("tiltwing_ode");
    IntermediateState setc_is_3(9);
    setc_is_3(0) = autotime;
    setc_is_3(1) = v_x;
    setc_is_3(2) = v_z;
    setc_is_3(3) = theta;
    setc_is_3(4) = zeta_w;
    setc_is_3(5) = delta_w;
    setc_is_3(6) = T_w;
    setc_is_3(7) = theta_ref;
    setc_is_3(8) = R;
    ModelFcn_3_jac = NULL;
    CFunction cLinkModel_3( ModelFcn_3NX, genericODE3 ); 
    acadodata_f3 << cLinkModel_3(setc_is_3); 

    OutputFcn acadodata_f4;

    DynamicSystem dynamicsystem1( acadodata_f3,acadodata_f4 );
    Process process2( dynamicsystem1,INT_RK12 );

    RealTimeAlgorithm algo1(ocp1, 0.02);
    algo1.set( MAX_NUM_ITERATIONS, 5 );
    algo1.set( INTEGRATOR_TYPE, INT_RK45 );
    algo1.set( INTEGRATOR_TOLERANCE, 1.000000E-04 );
    algo1.set( ABSOLUTE_TOLERANCE, 1.000000E-03 );

    StaticReferenceTrajectory referencetrajectory;
    Controller controller3( algo1,referencetrajectory );

    SimulationEnvironment algo2(0, 10, process2, controller3);
     algo2.init(acadodata_v3);
    returnValue returnvalue = algo2.run();


    VariablesGrid out_processout; 
    VariablesGrid out_feedbackcontrol; 
    VariablesGrid out_feedbackparameter; 
    VariablesGrid out_states; 
    VariablesGrid out_algstates; 
    algo2.getSampledProcessOutput(out_processout);
    algo2.getProcessDifferentialStates(out_states);
    algo2.getFeedbackControl(out_feedbackcontrol);
    const char* outputFieldNames[] = {"STATES_SAMPLED", "CONTROLS", "PARAMETERS", "STATES", "ALGEBRAICSTATES", "CONVERGENCE_ACHIEVED"}; 
    plhs[0] = mxCreateStructMatrix( 1,1,6,outputFieldNames ); 
    mxArray *OutSS = NULL;
    double  *outSS = NULL;
    OutSS = mxCreateDoubleMatrix( out_processout.getNumPoints(),1+out_processout.getNumValues(),mxREAL ); 
    outSS = mxGetPr( OutSS );
    for( int i=0; i<out_processout.getNumPoints(); ++i ){ 
      outSS[0*out_processout.getNumPoints() + i] = out_processout.getTime(i); 
      for( int j=0; j<out_processout.getNumValues(); ++j ){ 
        outSS[(1+j)*out_processout.getNumPoints() + i] = out_processout(i, j); 
       } 
    } 

    mxSetField( plhs[0],0,"STATES_SAMPLED",OutSS );
    mxArray *OutS = NULL;
    double  *outS = NULL;
    OutS = mxCreateDoubleMatrix( out_states.getNumPoints(),1+out_states.getNumValues(),mxREAL ); 
    outS = mxGetPr( OutS );
    for( int i=0; i<out_states.getNumPoints(); ++i ){ 
      outS[0*out_states.getNumPoints() + i] = out_states.getTime(i); 
      for( int j=0; j<out_states.getNumValues(); ++j ){ 
        outS[(1+j)*out_states.getNumPoints() + i] = out_states(i, j); 
       } 
    } 

    mxSetField( plhs[0],0,"STATES",OutS );
    mxArray *OutC = NULL;
    double  *outC = NULL;
    OutC = mxCreateDoubleMatrix( out_feedbackcontrol.getNumPoints(),1+out_feedbackcontrol.getNumValues(),mxREAL ); 
    outC = mxGetPr( OutC );
    for( int i=0; i<out_feedbackcontrol.getNumPoints(); ++i ){ 
      outC[0*out_feedbackcontrol.getNumPoints() + i] = out_feedbackcontrol.getTime(i); 
      for( int j=0; j<out_feedbackcontrol.getNumValues(); ++j ){ 
        outC[(1+j)*out_feedbackcontrol.getNumPoints() + i] = out_feedbackcontrol(i, j); 
       } 
    } 

    mxSetField( plhs[0],0,"CONTROLS",OutC );
    mxArray *OutP = NULL;
    double  *outP = NULL;
    OutP = mxCreateDoubleMatrix( out_feedbackparameter.getNumPoints(),1+out_feedbackparameter.getNumValues(),mxREAL ); 
    outP = mxGetPr( OutP );
    for( int i=0; i<out_feedbackparameter.getNumPoints(); ++i ){ 
      outP[0*out_feedbackparameter.getNumPoints() + i] = out_feedbackparameter.getTime(i); 
      for( int j=0; j<out_feedbackparameter.getNumValues(); ++j ){ 
        outP[(1+j)*out_feedbackparameter.getNumPoints() + i] = out_feedbackparameter(i, j); 
       } 
    } 

    mxSetField( plhs[0],0,"PARAMETERS",OutP );
    mxArray *OutZ = NULL;
    double  *outZ = NULL;
    OutZ = mxCreateDoubleMatrix( out_algstates.getNumPoints(),1+out_algstates.getNumValues(),mxREAL ); 
    outZ = mxGetPr( OutZ );
    for( int i=0; i<out_algstates.getNumPoints(); ++i ){ 
      outZ[0*out_algstates.getNumPoints() + i] = out_algstates.getTime(i); 
      for( int j=0; j<out_algstates.getNumValues(); ++j ){ 
        outZ[(1+j)*out_algstates.getNumPoints() + i] = out_algstates(i, j); 
       } 
    } 

    mxSetField( plhs[0],0,"ALGEBRAICSTATES",OutZ );
    mxArray *OutConv = NULL;
    if ( returnvalue == SUCCESSFUL_RETURN ) { OutConv = mxCreateDoubleScalar( 1 ); }else{ OutConv = mxCreateDoubleScalar( 0 ); } 
    mxSetField( plhs[0],0,"CONVERGENCE_ACHIEVED",OutConv );

    clearAllGlobals3( ); 

    clearAllStaticCounters( ); 
 
} 

