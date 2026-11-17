addpath("/home/selfishmining/mdp_work/blind/ref/proportional-reward-splitting-MDP/MDPtoolbox");
P = cell(1,2); R = cell(1,2);
P{1} = sparse([0.5 0.5; 0.2 0.8]); P{2} = sparse([0.9 0.1; 0.6 0.4]);
R{1} = sparse([1 0; 0 2]); R{2} = sparse([0 0; 1 1]);
[pol, g, t] = mdp_relative_value_iteration(P, R, 1e-6);
disp(pol); disp(g); disp(version());
