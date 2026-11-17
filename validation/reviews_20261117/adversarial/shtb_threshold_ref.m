% Review: run ZP SHTB Threshold.m unmodified (its clear keeps the path; its own addpath target is absent).
addpath(fullfile(getenv("HOME"), "mdp_work/ref/MDPtoolbox"));
cd(fullfile(getenv("HOME"), "mdp_work/ref/PoWSecurity/Chain Quality/SmallestHashTieBreaking"));
Threshold
printf("RESULT Threshold.m finished: lowAlpha=%.6f highAlpha=%.6f last alphaPower=%.6f\n", lowAlpha, highAlpha, alphaPower);
