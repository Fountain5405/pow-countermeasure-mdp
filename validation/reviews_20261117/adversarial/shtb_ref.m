% Review: ZP SHTB chain-quality MDP (reference InitStnum/InitStates/SolveMDP unmodified), range 15, maxForkLen 40.
addpath(fullfile(getenv("HOME"), "mdp_work/ref/MDPtoolbox"));
cd(fullfile(getenv("HOME"), "mdp_work/ref/PoWSecurity/Chain Quality/SmallestHashTieBreaking"));
global alphaPower maxForkLen range lowerBoundReward upperBoundReward;
range = 15; maxForkLen = 40; InitStnum;
alphas = str2num(getenv("ALPHAS"));
for ai = 1:numel(alphas)
  alphaPower = alphas(ai); t0 = tic;
  InitStates; SolveMDP;
  printf("RESULT alpha=%.4f lowerBoundReward(lowRou)=%.6f excess=%.2e profitable_by_Threshold=%d secs=%.0f\n", alphaPower, lowerBoundReward, lowerBoundReward-alphaPower, lowerBoundReward > alphaPower, toc(t0));
  fflush(stdout);
end
