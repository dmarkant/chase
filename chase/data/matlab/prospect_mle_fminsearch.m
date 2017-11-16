function [gammaOut, thetaOut] = prospect_mle_fminsearch(tbl)
init.gamma = rand(1,1); %starting parameter values
init.theta = rand(1,1); %starting parameter values
bounds.gamma = [0,3];
bounds.theta = [0,10];


xin(1) = parameter_bounder(init.gamma,0,bounds.gamma);
xin(2) = parameter_bounder(init.theta,0,bounds.theta); 


options = optimset('MaxFunEvals',20000, 'MaxIter', 20000, 'TolFun', 1.0000e-004, 'TolX',1.0000e-004 ,'LargeScale','off' );
%'Display','iter', 
bounder =1;
[xout,like,exit,output] = fminsearch(@prospect, xin,...
                                         options, tbl, bounder, bounds );
gammaOut = parameter_bounder(xout(1),1,bounds.gamma);
thetaOut = parameter_bounder(xout(2),1,bounds.theta);
                                     
                                     

    
    

