%% Dump Phase 52 reference: analytic flash derivatives, from MRST.
%
% Run in MATLAB after starting MRST. Writes ref_p52_flashderiv.mat.
%
% The flash (Phase 13) solves for the phase split on *values* - there is no
% derivative in the answer. A fully-implicit compositional flow model, whose
% primary variables are pressure and the overall composition z, needs the flash
% differentiated: dL/dp, dL/dz, dx/dp, dx/dz, dy/dp, dy/dz. Because the flash is
% an *implicit* function of (p, T, z) - defined by the equilibrium conditions,
% not a closed form - these come from the implicit function theorem, which is
% exactly what MRST's `getPhaseFractionDerivativesPTZ` computes:
%
%   ds/du = -(dF/ds)^{-1} (dF/du),
%
% with secondary s = (L, x, y), primary u = (p, T, z), and F the equilibrium
% system. This dumps those derivatives for the two-phase cells of a P-T-z grid,
% alongside the flash they were taken at (x, y, L, Z_L, Z_V), so the Python side
% can replay its own implicit-function-theorem solve on MRST's exact flash and
% compare the sensitivities directly.
%
% Column order (fullZ): [p, T, z_1, ..., z_ncomp].

mrstModule add ad-core compositional

thisDir = fileparts(mfilename('fullpath'));

names   = {'Methane', 'n-Butane', 'n-Decane'};
Tcrit   = [190.564, 425.125, 617.7];
Pcrit   = [4599200.0, 3796000.0, 2103000.0];
Vcrit   = [9.86e-5, 2.55e-4, 6.098e-4];
acf     = [0.0114, 0.201, 0.4884];
mw      = [0.016043, 0.058124, 0.142285];

mixture = CompositionalMixture(names, Tcrit, Pcrit, Vcrit, acf, mw);
eos = EquationOfStateModel([], mixture, 'Peng-Robinson');
ncomp = numel(names);

% -- a grid of conditions; keep only the genuinely two-phase ones -------------
pv = [30, 60, 100, 150]*barsa;
Tv = [340, 360, 380];
zs = [0.50, 0.30, 0.20;
      0.60, 0.25, 0.15;
      0.70, 0.20, 0.10];

np_ = numel(pv); nT = numel(Tv); nz = size(zs, 1);
n = np_*nT*nz;
[p_all, T_all] = deal(zeros(n, 1));
z_all = zeros(n, ncomp);
k = 0;
for iz = 1:nz
    for it = 1:nT
        for ip = 1:np_
            k = k + 1;
            p_all(k) = pv(ip);
            T_all(k) = Tv(it);
            z_all(k, :) = zs(iz, :);
        end
    end
end

[L, x, y, Z_L, Z_V] = standaloneFlash(p_all, T_all, z_all, eos);

% Build a state and ask MRST for the implicit-function-theorem derivatives.
state = struct();
state.pressure   = p_all;
state.T          = T_all;
state.components = z_all;
state.L   = L;
state.x   = x;
state.y   = y;
state.Z_L = Z_L;
state.Z_V = Z_V;

twoPhase = L > 0 & L < 1;
[dLc, dxc, dyc] = eos.getPhaseFractionDerivativesPTZ(state, twoPhase);

% dLc is (N2 x nprimary); dxc, dyc are cell{ncomp} of (N2 x nprimary).
nprimary = size(dLc, 2);
N2 = nnz(twoPhase);
dL = dLc;
dx = zeros(ncomp, N2, nprimary);
dy = zeros(ncomp, N2, nprimary);
for i = 1:ncomp
    dx(i, :, :) = dxc{i};
    dy(i, :, :) = dyc{i};
end

save(fullfile(thisDir, 'ref_p52_flashderiv.mat'), '-v7', ...
     'p_all', 'T_all', 'z_all', 'L', 'x', 'y', 'Z_L', 'Z_V', 'twoPhase', ...
     'dL', 'dx', 'dy', 'nprimary', ...
     'Tcrit', 'Pcrit', 'Vcrit', 'acf', 'mw');

fprintf('Wrote ref_p52_flashderiv.mat to %s\n', thisDir);
fprintf('  %d conditions, %d two-phase, nprimary = %d\n', n, N2, nprimary);
fprintf('  dL/dp range : %.3e - %.3e 1/Pa\n', min(dL(:,1)), max(dL(:,1)));
