%% Dump Phase 51 reference: flash -> flow properties, from MRST.
%
% Run in MATLAB after starting MRST. Writes ref_p51_flowprops.mat.
%
% Phase 13 gave the flash (what phases exist and what is in them) and Phase 50
% gave the density and viscosity of a phase. What still stood between those and
% a flow model is `computeTwoPhaseFlowProps`: the step that turns a flash result
% into the exact quantities a *per-component* conservation equation reads -
%
%   - the **mass fractions** xM, yM (the flash speaks in moles; the equation is
%     written in mass, so `getMassFraction` converts), and
%   - the **phase saturations** sL, sV (the flash gives a *molar* liquid
%     fraction L; `computeSaturations` turns it into a *volume* fraction, which
%     with no volume shift is just `L Z_L / (L Z_L + (1-L) Z_V)`).
%
% Both are dumped for a grid spanning single-phase liquid, single-phase vapour
% and the two-phase region. The saturations are dumped **with and without a
% volume shift**, because the shift changes the volumes (hence the saturations)
% while leaving the molar flash alone - a port that applied it in the wrong
% place would move the mass fractions too, and only the coupled check notices.
%
% Following the project's "fix the inputs, compare the algorithm" contract, the
% flash (x, y, L, Z_L, Z_V) is dumped alongside so the Python side replays these
% formulas on MRST's own flash, isolating them from any flash-solver drift.

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
pm  = eos.PropertyModel;

% -- a grid of conditions, spanning liquid, vapour and the two-phase region ---
pv = [10, 30, 60, 100, 150, 250, 400]*barsa;
Tv = [300, 350, 400, 450];
zs = [0.50, 0.30, 0.20;
      0.80, 0.15, 0.05;
      0.95, 0.04, 0.01];

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

% -- flash, then derive the flow properties MRST's model would ---------------
[L, x, y, Z_L, Z_V] = standaloneFlash(p_all, T_all, z_all, eos);

xM = eos.getMassFraction(x);
yM = eos.getMassFraction(y);

rho_L = pm.computeDensity(eos, p_all, x, Z_L, T_all, true);
rho_V = pm.computeDensity(eos, p_all, y, Z_V, T_all, false);

[sL, sV] = eos.computeSaturations(p_all, T_all, rho_L, rho_V, x, y, L, Z_L, Z_V);

% -- with a volume shift: the volumes (hence saturations) move, flash does not
shift = [-0.30, -0.20, -0.10];
eos_s = eos;
eos_s.PropertyModel.volumeShift = shift;
[sL_shift, sV_shift] = eos_s.computeSaturations(p_all, T_all, rho_L, rho_V, ...
                                                x, y, L, Z_L, Z_V);

twoPhase = L > 0 & L < 1;

save(fullfile(thisDir, 'ref_p51_flowprops.mat'), '-v7', ...
     'p_all', 'T_all', 'z_all', 'L', 'x', 'y', 'Z_L', 'Z_V', 'twoPhase', ...
     'xM', 'yM', 'sL', 'sV', 'sL_shift', 'sV_shift', 'shift', ...
     'Tcrit', 'Pcrit', 'Vcrit', 'acf', 'mw');

fprintf('Wrote ref_p51_flowprops.mat to %s\n', thisDir);
fprintf('  %d conditions, %d of them two-phase\n', n, sum(twoPhase));
fprintf('  sL range : %.3f - %.3f\n', min(sL), max(sL));
fprintf('  xM(C1)   : %.4f - %.4f\n', min(xM(:,1)), max(xM(:,1)));
