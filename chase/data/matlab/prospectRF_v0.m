function logLike = prospectRF(xin, tbl, bounder, bounds)
%power_mle the log-likelihood function of the power model
if (bounder)
   gamma = parameter_bounder(xin(1), 1, bounds.gamma); 
   theta = parameter_bounder(xin(2), 1, bounds.theta); 
else
    gamma = xin(1);
    theta = xin(2); 
end;

alpha = 1;%utility for gains
beta = 1; %utility for losses
lambda =1;%loss aversion
delta = 1;%height for weight function

uLx0 = utility( tbl.Lx0, alpha,beta,lambda  );
uLx1 = utility( tbl.Lx1, alpha,beta,lambda );
uHx0 = utility( tbl.Hx0, alpha,beta,lambda  );
uHx1 = utility( tbl.Hx1, alpha,beta,lambda );

lowOutcomes = [tbl.Lx0 tbl.Lx1];
highOutcomes = [tbl.Hx0 tbl.Hx1];

lowProb = [tbl.Lf0 tbl.Lf1];
highProb = [tbl.Hf0 tbl.Hf1];

wLow = prelec(lowProb,gamma,delta);
wHigh = prelec(highProb,gamma,delta);

prospectLow = wLow(1).*uLx0 +  wLow(2).*uLx1;
prospectHigh = wHigh(1).*uHx0 +  wHigh(2).*uHx1;

probChooseH = 1./(1+exp( -theta.*( prospectHigh - prospectLow ) ) ); 

logLike = -1.*sum(tbl.choice.*log(probChooseH) + (1-tbl.choice).*log(1-probChooseH)); 