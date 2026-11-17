% Review: run Aumayr et al. Bitcoin IC InitStates/SolveMDP unmodified, print both bisections.
p = fullfile(getenv("HOME"), "mdp_work/ref/prs/MDPtoolbox");
if ~exist(p, "dir"); p = fullfile(getenv("HOME"), "mdp_work/ref/MDPtoolbox"); end
addpath(p); disp(["toolbox: " p]);
cd(fullfile(getenv("HOME"), "mdp_work/ref/prs/incentive_compatibility/Bitcoin"));
global alphaPower gammaRatio maxForkLen lowerBoundReward upperBoundReward lowRou;
cases = str2num(getenv("CASES"));
for ci = 1:size(cases,1)
  alphaPower = cases(ci,1); gammaRatio = cases(ci,2); maxForkLen = 30;
  t0 = tic;
  InitStates; SolveMDP;
  printf("RESULT alpha=%.2f gamma=%.1f cap=%d lowerBoundRou=%.6f final_lowRou=%.6f ceiling=%.6f secs=%.0f\n", alphaPower, gammaRatio, maxForkLen, lowerBoundRou, lowRou, lowerBoundRou+0.1, toc(t0));
  fflush(stdout);
end
