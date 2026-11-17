addpath("/home/selfishmining/mdp_work/blind/ref/proportional-reward-splitting-MDP/MDPtoolbox");
addpath("/home/selfishmining/mdp_work/blind/work/oct");
global alphaPower maxForkLen range lowerBoundReward upperBoundReward POut RsOut RhOut Wrou;
global tillaSmaller tillaEqual tillaOneBigger numOfStates;
for maxForkLen = [10 20]
  for range = [15 10 5]
    InitStnum;
    for alphaPower = [0.3 0.4]
      InitStates;
      export_mdp(sprintf("/home/selfishmining/mdp_work/blind/work/exp/shtb_r%d_mF%d_a%.2f", range, maxForkLen, alphaPower), POut, RsOut, RhOut);
      clear POut RsOut RhOut P Wrou;
    end
  end
end
