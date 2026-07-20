%% Dump Phase 50 reference: EOS phase density and viscosity, from MRST.
%
% Run in MATLAB after starting MRST. Writes ref_p50_eosprops.mat.
%
% `eos.py` already gives what an equation of state is *for*: given a pressure, a
% temperature and an overall composition, which phases exist and what is in
% them. It returns compositions and compressibility factors and stops.
%
% A flow model cannot use any of that. Darcy's law wants a **density** and a
% **viscosity**, and neither is in a flash result. That missing half is the
% whole reason the compositional EOS has sat unused since it was ported, and it
% is what this phase adds.
%
% So: flash a real three-component mixture over a grid of pressures and
% temperatures - deliberately spanning single-phase liquid, single-phase vapour
% and the two-phase region - and dump, for both phases,
%
%   - the compressibility factors (so a disagreement in the *density* can be
%     told apart from a disagreement in the flash that fed it);
%   - the mass density;
%   - the Lohrenz-Bray-Clark viscosity, which is where the interesting mistakes
%     live: it is written in Rankine, psia and centipoise, and its constants
%     mean nothing in SI.
%
% Densities and viscosities are also dumped **with a volume shift**, because a
% shift is only legitimate if it moves the density and leaves the equilibrium
% alone - and a port that applied it in the wrong place would move both.

mrstModule add ad-core compositional

thisDir = fileparts(mfilename('fullpath'));

% -- the same mixture PyMRST's `simple_mixture()` builds ----------------------
names   = {'Methane', 'n-Butane', 'n-Decane'};
Tcrit   = [190.564, 425.125, 617.7];
Pcrit   = [4599200.0, 3796000.0, 2103000.0];
Vcrit   = [9.86e-5, 2.55e-4, 6.098e-4];
acf     = [0.0114, 0.201, 0.4884];
mw      = [0.016043, 0.058124, 0.142285];

mixture = CompositionalMixture(names, Tcrit, Pcrit, Vcrit, acf, mw);
eos = EquationOfStateModel([], mixture, 'Peng-Robinson');
pm = eos.PropertyModel;

% -- a grid of conditions, spanning liquid, vapour and the two-phase region ---
pv = [10, 30, 60, 100, 150, 250, 400]*barsa;
Tv = [300, 350, 400, 450];
zs = [0.50, 0.30, 0.20;      % rich in the heavy ends -> liquid at depth
      0.80, 0.15, 0.05;      % lighter
      0.95, 0.04, 0.01];     % nearly pure methane -> vapour

np_ = numel(pv); nT = numel(Tv); nz = size(zs, 1);
n = np_*nT*nz;

[p_all, T_all] = deal(zeros(n, 1));
z_all = zeros(n, 3);
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

% -- flash, then ask for the properties MRST derives from it ------------------
[L, x, y, Z_L, Z_V] = standaloneFlash(p_all, T_all, z_all, eos);

shift = [-0.30, -0.20, -0.10];

rho_L = pm.computeDensity(eos, p_all, x, Z_L, T_all, true);
rho_V = pm.computeDensity(eos, p_all, y, Z_V, T_all, false);
mu_L  = pm.computeViscosity(eos, p_all, x, Z_L, T_all, true);
mu_V  = pm.computeViscosity(eos, p_all, y, Z_V, T_all, false);

pm_s = pm;
pm_s.volumeShift = shift;
rho_L_shift = pm_s.computeDensity(eos, p_all, x, Z_L, T_all, true);
rho_V_shift = pm_s.computeDensity(eos, p_all, y, Z_V, T_all, false);
mu_L_shift  = pm_s.computeViscosity(eos, p_all, x, Z_L, T_all, true);
mu_V_shift  = pm_s.computeViscosity(eos, p_all, y, Z_V, T_all, false);

twoPhase = L > 0 & L < 1;

save(fullfile(thisDir, 'ref_p50_eosprops.mat'), '-v7', ...
     'p_all', 'T_all', 'z_all', 'L', 'x', 'y', 'Z_L', 'Z_V', 'twoPhase', ...
     'rho_L', 'rho_V', 'mu_L', 'mu_V', ...
     'rho_L_shift', 'rho_V_shift', 'mu_L_shift', 'mu_V_shift', 'shift', ...
     'Tcrit', 'Pcrit', 'Vcrit', 'acf', 'mw');

fprintf('Wrote ref_p50_eosprops.mat to %s\n', thisDir);
fprintf('  %d conditions, %d of them two-phase\n', n, sum(twoPhase));
fprintf('  rho_L : %.1f - %.1f kg/m3\n', min(rho_L), max(rho_L));
fprintf('  rho_V : %.1f - %.1f kg/m3\n', min(rho_V), max(rho_V));
fprintf('  mu_L  : %.4f - %.4f cP\n', min(mu_L)/(centi*poise), max(mu_L)/(centi*poise));
fprintf('  mu_V  : %.4f - %.4f cP\n', min(mu_V)/(centi*poise), max(mu_V)/(centi*poise));
