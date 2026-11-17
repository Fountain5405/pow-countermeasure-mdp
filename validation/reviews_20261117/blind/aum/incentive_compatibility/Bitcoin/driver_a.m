for gamma = [0 0.5 1]
  for alpha = [0.40 0.45]
    r = Init(alpha, 30, gamma);
    printf("RETURNED alpha=%.4f gamma=%.2f value=%.10f\n", alpha, gamma, r);
    fflush(stdout);
  end
end
