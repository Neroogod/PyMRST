%% Dump Phase 13 reference: Peng-Robinson flash (MRST compositional).

thisDir = fileparts(mfilename('fullpath'));
mrstModule add compositional ad-core ad-props

names = {'Methane', 'n-Butane', 'n-Decane'};
Tcrit = [190.564, 425.125, 617.7];
Pcrit = [4599200, 3796000, 2103000];
Vcrit = [9.86e-5, 2.55e-4, 6.098e-4];
acf   = [0.0114, 0.201, 0.4884];
mw    = [0.016043, 0.058124, 0.142285];

mixture = CompositionalMixture(names, Tcrit, Pcrit, Vcrit, acf, mw);
eos = EquationOfStateModel([], mixture, 'Peng-Robinson');
% MRST's default flash tolerance is loose: at the default setting the
% returned compositions leave a fugacity error of ~2e-2 (checked against
% the equal-fugacity condition). Tighten it so this reference is an
% actually-converged equilibrium.
eos.nonlinearTolerance = 1e-12;

n  = 12;
p  = linspace(20e5, 120e5, n)';
T  = repmat(350, n, 1);
z  = repmat([0.5, 0.3, 0.2], n, 1);

[L, x, y, Zl, Zv] = standaloneFlash(p, T, z, eos);
V = 1 - L;

K_wilson = estimateEquilibriumWilson(eos, p, T);

save(fullfile(thisDir, 'ref_p13_flash.mat'), '-v7', ...
     'p', 'T', 'z', 'V', 'x', 'y', 'Zl', 'Zv', 'K_wilson', ...
     'Tcrit', 'Pcrit', 'Vcrit', 'acf', 'mw');
fprintf('Wrote Phase-13 flash reference (V from %.4f to %.4f)\n', ...
        min(V), max(V));
