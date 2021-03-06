clear
%get the listing of the files
listing = dir('../../data/process_recovery_sixproblems');

tbl = readtable([ '../../data/process_recovery_sixproblems/' listing(1+2).name]);
%basic stats by problem
uniqueProblems = unique(tbl.problem); 
problemStats = zeros(length(uniqueProblems),5,length(listing)-2);
pars = zeros(length(listing)-2,2);
parsRF = zeros(length(listing)-2,2);

for l = 1:(length(listing)-2)
    %read one table at a time
    tbl = readtable([ '../../data/process_recovery_sixproblems/' listing(l+2).name]);
    %get the parameters of the simulation
    temp = sscanf(listing(l+2).name, 'process_recovery_sixproblems(N=%d,minsamplesize=%d,prelec_gamma=%g,theta=%g)_iteration=%d'); 
    NSim(l) = temp(1); 
    minSampleSizeSim(l) = temp(2);
    gammaSim(l) = temp(3); 
    thetaSim(l) = temp(4);
    iterSim(l) = temp(5); 
    %basic stats by problem
    uniqueProblems = unique(tbl.problem); 
    for i = 1:length(uniqueProblems)
        propEV(i) = mean(tbl.choice(tbl.problem==uniqueProblems(i)));
        meanSample(i) = mean(tbl.samplesize(tbl.problem==uniqueProblems(i)));
        quant1Sample(i) = quantile( tbl.samplesize(tbl.problem==uniqueProblems(i)), .25 ); 
        quant2Sample(i) = quantile( tbl.samplesize(tbl.problem==uniqueProblems(i)), .50 ); 
        quant3Sample(i) = quantile( tbl.samplesize(tbl.problem==uniqueProblems(i)), .75 ); 
    end
    
    [gammaOut, thetaOut] = prospect_mle_fminsearch(tbl);
    pars(l,:) = [gammaOut,thetaOut];
    
    [gammaRFOut, thetaRFOut] = prospectRF_mle_fminsearch(tbl);
    parsRF(l,:) = [gammaRFOut,thetaRFOut];
    
    problemStats(:,:,l) = [propEV', meanSample', quant1Sample', quant2Sample', quant3Sample']; 
    
end;

for i = 1:24
   meanGamma(i) =  mean(pars((50.*i-49:50.*i),1));
   stdGamma(i) = std(pars((50.*i-49:50.*i),1));
end;

%% make doug's plots
uniqueGammaSim = unique(gammaSim);
uniqueThetaSim = unique(thetaSim); 

for i = 1:length(uniqueGammaSim)
    for j = 1:length(uniqueThetaSim)
        ii = find(gammaSim == uniqueGammaSim(i) & thetaSim ==uniqueThetaSim(j)); 
        meanGammaEst(i,j) = mean(pars(ii,1)); 
        meanBetaEst(i,j) = mean(pars(ii,2)); 
    end
end


for i = 1:length(uniqueGammaSim)
    for j = 1:length(uniqueThetaSim)
        ii = find(gammaSim == uniqueGammaSim(i) & thetaSim ==uniqueThetaSim(j)); 
        meanGammaRFEst(i,j) = mean(parsRF(ii,1)); 
        meanBetaRFEst(i,j) = mean(parsRF(ii,2)); 
    end
end

figure
subplot(2,2,1)
plot(uniqueThetaSim,meanGammaEst);
axis square
ylabel('Overweighting \leftarrow \gamma \rightarrow Underweighting','FontSize',24)
xlabel('Threshold \theta','FontSize',24)
legend('\gamma = .6','\gamma = 1.0','\gamma = 1.4','Location','northwest')
title('Objective Probabilities','FontSize',24)

subplot(2,2,2)
plot(uniqueThetaSim,meanGammaRFEst);
axis square
ylabel('Overweighting \leftarrow \gamma \rightarrow Underweighting ','FontSize',24)
xlabel('Threshold \theta','FontSize',24)
title('Relative Frequencies','FontSize',24)

subplot(2,2,3)
plot(uniqueThetaSim,meanBetaEst);
axis square
ylabel('Response Consistency \beta','FontSize',24)
xlabel('Threshold \theta','FontSize',24)
legend('\gamma = .6','\gamma = 1.0','\gamma = 1.4','Location','northwest')

subplot(2,2,4)
plot(uniqueThetaSim,meanBetaRFEst);
axis square
ylabel('Response Consistency \beta','FontSize',24)
xlabel('Threshold \theta','FontSize',24)

savefig('SixProblems.fig')
saveas(gcf,'SixProblems.png','png')

