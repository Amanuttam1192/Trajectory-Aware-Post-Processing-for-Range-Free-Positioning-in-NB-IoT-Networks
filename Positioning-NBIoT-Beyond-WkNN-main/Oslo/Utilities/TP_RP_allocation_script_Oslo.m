clear
N_points=5266;
N_runs=1000;
probability_RFP=0.7;
TP_RP_allocation=ones(N_points,N_runs);
for i = 1:N_runs
    TP_RP_allocation(randperm(N_points,round(N_points*(1-probability_RFP))),i) = 2;%RFP: 1; TP: 2;
end
clear N_points N_runs probability_RFP i
save('TP_RP_allocation_Oslo.mat')