# Whitepaper results aggregate

### Multi-seed held-out Claude+Gemini — scorer = baseline_linear_z3
version           n          AUC      TPR@5%      TPR@1%    FPRoth@5    AUCg/oth
v11               5  0.705±0.021 0.135±0.031 0.032±0.018 0.000±0.000 0.932±0.004
v12               5  0.872±0.015 0.462±0.078 0.150±0.018 0.004±0.002 0.931±0.011
v13               5  0.875±0.016 0.523±0.085 0.256±0.161 0.007±0.002 0.940±0.006
v14               5  0.599±0.018 0.088±0.014 0.033±0.018 0.023±0.013 0.647±0.016
v14b              5  0.965±0.005 0.943±0.013 0.555±0.078 0.253±0.032 0.940±0.008
v14b_seed1        5  0.962±0.004 0.919±0.024 0.436±0.078 0.203±0.043 0.939±0.006
enron44_nosyn     5  0.658±0.016 0.081±0.018 0.022±0.010 0.000±0.000 0.943±0.003

### Multi-seed held-out Claude+Gemini — scorer = mahalanobis
version           n          AUC      TPR@5%      TPR@1%    FPRoth@5    AUCg/oth
v11               5  0.690±0.020 0.121±0.017 0.043±0.012 0.000±0.000 0.967±0.003
v12               5  0.871±0.016 0.453±0.064 0.155±0.036 0.001±0.001 0.965±0.002
v13               5  0.872±0.021 0.499±0.069 0.255±0.047 0.001±0.001 0.970±0.004
v14               5  0.610±0.016 0.137±0.015 0.060±0.013 0.010±0.003 0.667±0.015
v14b              5  0.980±0.003 0.909±0.017 0.533±0.055 0.079±0.014 0.967±0.003
v14b_seed1        5  0.976±0.005 0.867±0.037 0.493±0.052 0.038±0.017 0.967±0.007
enron44_nosyn     5  0.640±0.012 0.134±0.025 0.045±0.019 0.000±0.000 0.982±0.002

### Identity x synthetic 2x2 (held-out Claude+Gemini, mahalanobis AUC, multiseed mean)
  44 authors   no-syn   -> 0.640±0.012  (n=5, enron44_nosyn)
  44 authors   +syn     -> 0.871±0.016  (n=5, v12)
  844 authors  no-syn   -> 0.610±0.016  (n=5, v14)
  844 authors  +syn     -> 0.980±0.003  (n=5, v14b)

### Novel-vendor spot-check (Qwen-2.5-72B + DeepSeek-V3; never in train OR eval)
Pooled (mahalanobis deploy scorer), mean across probe seeds.
NOTE: AUC + AUCg/oth are the generalization evidence; FPRoth@5 is a
threshold-placement artifact when synthetics are near-separable (see writeup).
version     n          AUC      TPR@5%      TPR@1%    AUCg/oth    FPRoth@5
v12         3  0.922±0.005 0.630±0.030 0.328±0.040 0.964±0.000 0.006±0.002
v14b        3  0.996±0.001 0.986±0.008 0.917±0.036 0.966±0.003 0.579±0.206
