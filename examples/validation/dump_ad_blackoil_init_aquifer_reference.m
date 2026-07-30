%% Intended zero-flux calculation in computeInitAquifer
% MRST 2024a's utility computes this pressure and then refers to an undefined
% V_aq on its final line. This fixture records the calculation before that typo.

thisDir = fileparts(mfilename('fullpath'));
aquid = [1; 1; 2];
j = [2; 1; 3]; alpha = [0.5; 0.5; 1.0];
pW = [100; 130; 90]; bW = [1; 1.1; 0.9];
depth_conn = [100; 120; 90]; depth_aq = [80; 80; 70];
rhoWS = 1000; gravity = 9.81;
weights = alpha.*j;
head = bW.*rhoWS.*gravity.*(depth_conn-depth_aq);
M = sparse(1:3, aquid, weights, 3, 2);
r = weights.*(head-pW);
p_aq = -(M'*M)\(M'*r);
volumes = [1e6; 2e6];
save(fullfile(thisDir, 'ref_ad_blackoil_init_aquifer.mat'), '-v7', ...
     'aquid','j','alpha','pW','bW','depth_conn','depth_aq', ...
     'rhoWS','gravity','p_aq','volumes');
fprintf('Wrote intended computeInitAquifer reference to %s\n', thisDir);
