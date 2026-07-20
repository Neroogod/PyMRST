% Phase 17 reference: the CO2 trapping inventory (co2lab-ve/utils/inventory)
% and injection-point ranking (co2lab-spillpoint).
%
% The VE states come from PyMRST (states_p17.mat), so MRST's
% massTrappingDistributionVEADI is fed exactly the states PyMRST computed and
% the comparison isolates the inventory arithmetic. The VE simulator itself is
% validated separately (Phase 9), as is the trap analysis (Phase 10).

mrstModule add co2lab-common co2lab-ve co2lab-spillpoint ad-core

here = fileparts(mfilename('fullpath'));
S = load(fullfile(here, 'states_p17.mat'));

%% Rebuild the same top-surface grid ---------------------------------------
nx = 60; ny = 24; nz = 5;
G = cartGrid([nx, ny, nz], [2500, 1000, 25]);
x = G.nodes.coords(:,1); y = G.nodes.coords(:,2);
G.nodes.coords(:,3) = G.nodes.coords(:,3) + ...
    (1000 - 0.02*x ...
     - 14*exp(-(((x - 900)/140).^2 + ((y - 500)/260).^2)) ...
     - 18*exp(-(((x - 1800)/150).^2 + ((y - 500)/260).^2)));
G  = computeGeometry(G);
Gt = topSurfaceGrid(G);

rock2.poro = S.poro(:);
rock2.perm = repmat(300*milli*darcy, Gt.cells.num, 1);

%% The fluid, as PyMRST has it ---------------------------------------------
fluidADI = struct();
fluidADI.res_water = S.srw;
fluidADI.res_gas   = S.srg;
fluidADI.rhoGS     = S.rhoGS;
fluidADI.bG = @(p) 1 + S.c_gas   .* (p - S.p_ref);
fluidADI.bW = @(p) 1 + S.c_water .* (p - S.p_ref);

%% PyMRST's own trap analysis, so the inventory alone is under test --------
trapstruct = struct();
trapstruct.trap_regions = S.trap_regions(:);
trapstruct.trap_z       = S.trap_z(:);

%% The inventory, state by state -------------------------------------------
nstates = size(S.p, 2);
masses  = zeros(nstates, 7);
for k = 1:nstates
    p     = S.p(:, k);
    sG    = S.sG(:, k);
    sW    = 1 - sG;
    h     = S.h(:, k);
    h_max = S.hmax(:, k);
    masses(k, :) = massTrappingDistributionVEADI( ...
        Gt, p, sG, sW, h, h_max, rock2, fluidADI, trapstruct, []);
end

% and once with sub-scale trapping and dissolution switched on
dh = ones(Gt.cells.num, 1);
masses_dh = massTrappingDistributionVEADI( ...
    Gt, S.p(:,end), S.sG(:,end), 1 - S.sG(:,end), S.h(:,end), ...
    S.hmax(:,end), rock2, fluidADI, trapstruct, dh);
masses_rs = massTrappingDistributionVEADI( ...
    Gt, S.p(:,end), S.sG(:,end), 1 - S.sG(:,end), S.h(:,end), ...
    S.hmax(:,end), rock2, fluidADI, trapstruct, [], 'rs', 0.05);

%% Injection-point ranking, on MRST's own trap analysis --------------------
res = trapAnalysis(Gt, true);
rock2.poro = S.poro(:);
[trees, trapVols] = maximizeTrapping(Gt, 'res', res, 'removeOverlap', true);
[optCell, optVol, ~, ~, optTraps] = findOptimalInjectionPoint(Gt, res);

tree_roots  = [trees.root]';
tree_values = [trees.value]';

%% Catchments on an ASYMMETRIC caprock ------------------------------------
% The symmetric caprock above is a trap for the trap analysis: mirror-image
% cells sit at bit-identical depths, and which of them survives an exact
% comparison depends on round-off in the grid construction, so MATLAB and
% Python legitimately disagree about how many summits a dome has. Offset the
% domes and the ties disappear, and the catchments must then agree exactly.
Ga = cartGrid([60, 24, 5], [2500, 1000, 25]);
xa = Ga.nodes.coords(:,1); ya = Ga.nodes.coords(:,2);
Ga.nodes.coords(:,3) = Ga.nodes.coords(:,3) + ...
    (1000 - 0.02*xa ...
     - 14*exp(-(((xa - 900)/140).^2  + ((ya - 430)/240).^2)) ...
     - 18*exp(-(((xa - 1800)/155).^2 + ((ya - 560)/275).^2)));
Gta = topSurfaceGrid(computeGeometry(Ga));
resa = trapAnalysis(Gta, true);

ref = struct('masses', masses, 'masses_dh', masses_dh(:)', ...
             'masses_rs', masses_rs(:)', ...
             'trap_vols', trapVols(:), 'tree_roots', tree_roots, ...
             'tree_values', tree_values, 'opt_cell', optCell, ...
             'opt_vol', optVol, 'opt_traps', optTraps(:), ...
             'mrst_traps', double(max(res.traps)), ...
             'trap_regions', res.trap_regions(:), ...
             'asym_traps', resa.traps(:), ...
             'asym_regions', resa.trap_regions(:), ...
             'asym_top', resa.top(:));
save(fullfile(here, 'ref_p17_co2_inventory.mat'), '-struct', 'ref', '-v7');

fprintf(['Wrote Phase-17 reference: %d states, mobile fraction %.1f%% -> ' ...
         '%.1f%%; %d traps, best tree %.3e m3\n'], nstates, ...
        100*masses(1,7)/sum(masses(1,2:end)), ...
        100*masses(end,7)/sum(masses(end,2:end)), ...
        max(res.traps), max(tree_values));
