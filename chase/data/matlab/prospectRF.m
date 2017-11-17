function logLike = prospectRF(xin, tbl, bounder, bounds)
%power_mle the log-likelihood function of the power model
if (bounder)
   gamma = parameter_bounder(xin(1), 1, bounds.gamma); 
   beta = parameter_bounder(xin(2), 1, bounds.theta); 
else
    gamma = xin(1);
    beta = xin(2); 
end

alpha = 1;%utility for gains
beta = 1; %utility for losses
lambda =1;%loss aversion
delta = 1;%height for weight function

wLow = zeros(height(tbl),2);
wHigh = zeros(height(tbl),2);
xLow = zeros(height(tbl),2);
xHigh = zeros(height(tbl),2);

for i = 1:height(tbl)
    [wLow(i,:), xLow(i,:)] = prelec([tbl.Lf0(i) tbl.Lf1(i)],[tbl.Lx0(i) tbl.Lx1(i)],gamma,delta);
    [wHigh(i,:), xHigh(i,:)]  = prelec([tbl.Hf0(i) tbl.Hf1(i)],[tbl.Hx0(i) tbl.Hx1(i)],gamma,delta);
end

uLow = zeros(height(tbl),2);
uHigh = zeros(height(tbl),2);

uLow= utility( xLow, alpha,beta,lambda  );
uHigh = utility( xHigh, alpha,beta,lambda );

%lowOutcomes = [tbl.Lx0 tbl.Lx1];
%highOutcomes = [tbl.Hx0 tbl.Hx1];

%lowProb = [tbl.Lp0 tbl.Lp1];
%highProb = [tbl.Hp0 tbl.Hp1];

%wLow = prelec(lowProb,gamma,delta);
%wHigh = prelec(highProb,gamma,delta);

prospectLow = sum(wLow.*uLow,2);
prospectHigh = sum(wHigh.*uHigh,2);

probChooseH = 1./(1+exp( -beta.*( prospectHigh - prospectLow ) ) ); 

logLike = -1.*sum(tbl.choice.*log(probChooseH) + (1-tbl.choice).*log(1-probChooseH)); 