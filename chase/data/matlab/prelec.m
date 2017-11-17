function [wout,xout] = prelec(p,x,gamma,delta)
% first identify if gains or losses
gainTest = x>0;
lossTest = x<0;
zeroTest = x==0;

if(sum(gainTest)>0 && sum(lossTest)>0) %mixed
    %gains
    theGains = x(x>0);
    probGains = p(x>0);
    [gainSort, isort] = sort(theGains); 
    psort = probGains(isort);
    wGains = weightCalc(psort,gamma,delta); 
   
    %losses
    theLosses = x(x<=0);
    probLosses = p(x<=0);
    [lossSort, isort] = sort(theLosses); 
    psort = probLosses(isort,'descend');
    wLosses = weightCalc(psort,gamma,delta); 
    
    wout = [wLosses wGains];
    xout = [lossSort gainSort];

else
    
    if(sum(gainTest)>0 && sum(lossTest)==0)%gains
        [xsort, isort] = sort(x); 
        psort = p(isort);
    else%(sum(gainTest)==0 && sum(lossTest)>0)%losses
        [xsort, isort] = sort(x,'descend'); 
        psort = p(isort); 
    end
    w = weightCalc(psort,gamma,delta);
    wout = w;
    xout = xsort;
    %w(:,2) = wr(:,2)-wr(:,1);
    %w(:,1) = wr(:,1);
    
end

function wcalc = weightCalc(psort,gamma,delta)

r = cumsum(psort,2);
wr = exp(-delta.*(-log(r)).^gamma ); 
wr(r==0)= 0;
wr(r==1) =1; 
wrTemp = zeros(1,length(wr));
wrTemp(:) = [0 wr(1:(length(wr)-1))]; 
wcalc = wr-wrTemp;
