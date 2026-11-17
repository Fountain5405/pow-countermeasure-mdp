addpath("/home/selfishmining/mdp_work/blind/work/oct");
global alphaPower maxB superOverride rou POut RsOut RhOut;
superOverride = 3;
for maxB = [12 16 20]
  for alphaPower = [0.3 0.4]
    InitStates;
    export_mdp(sprintf("/home/selfishmining/mdp_work/blind/work/exp/pop_mB%d_a%.2f", maxB, alphaPower), POut, RsOut, RhOut);
    clear POut RsOut RhOut P;
  end
end
