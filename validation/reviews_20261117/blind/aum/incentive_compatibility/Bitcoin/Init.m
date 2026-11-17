function [lowRou]=Init(alpha, mFL, gamma)

addpath('../../MDPtoolbox');

% alphaPower is the mining power share of the attacker
% gammaRatio is the proportion of honest miners that would mine on the
%     attacker's chain during a tie
% maxForkLen is the maximum length of a block fork
global alphaPower gammaRatio maxForkLen lowerBoundReward upperBoundReward;

maxForkLen = mFL;
gammaRatio = gamma;
alphaPower = alpha;

lowerBoundRewardZero = [];
upperBoundRewardZero = [];
InitStates;
SolveMDP;
lowerBoundRewardZero = [lowerBoundRewardZero lowerBoundReward];
upperBoundRewardZero = [upperBoundRewardZero upperBoundReward];

end
