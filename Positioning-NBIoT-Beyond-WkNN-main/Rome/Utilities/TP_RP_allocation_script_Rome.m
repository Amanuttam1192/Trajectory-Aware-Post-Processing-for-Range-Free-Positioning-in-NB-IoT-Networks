clear
N_points=2670;
N_runs=1000;
probability_RP=0.7;
TP_RP_allocation=ones(N_points,N_runs);
for i = 1:N_runs
    TP_RP_allocation(randperm(N_points,round(N_points*(1-probability_RP))),i) = 2;%RP: 1; TP: 2;
end
clear N_points N_runs probability_RP i
save('TP_RP_allocation_Rome')