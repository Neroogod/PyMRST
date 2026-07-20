%% Dump Phase 9 reference: co2lab-ve vertical equilibrium.
%
% Part 1: topSurfaceGrid + averageRock on a sloping aquifer (grid geometry,
%         column thickness H, top-surface depth z, upscaled rock).
% Part 2: the sharp-interface VE fluid (makeVEFluid): upscaled krG/krW and
%         capillary pressure as functions of the upscaled saturation.
%
% These are the two pieces the PyMRST VE port must reproduce; the model
% equations on top of them are the same two-phase system already validated
% in Phase 8.

thisDir = fileparts(mfilename('fullpath'));
mrstModule add co2lab-common co2lab-ve ad-core ad-props ad-blackoil
gravity reset on

%% Sloping aquifer (the geometry used by co2lab's sloping-aquifer example)
[nx, ny, nz] = deal(50, 1, 10);
G = cartGrid([nx, ny, nz], [2000, 100, 20]);
x = G.nodes.coords(:, 1);
G.nodes.coords(:, 3) = G.nodes.coords(:, 3) + 1000 - 0.04 * x;
G = computeGeometry(G);

rock.poro = repmat(0.2, G.cells.num, 1);
rock.perm = repmat(400 * milli * darcy, G.cells.num, 1);

[Gt, G] = topSurfaceGrid(G);
rock2D  = averageRock(rock, Gt);

H_ve      = Gt.cells.H;
z_ve      = Gt.cells.z;
vol_ve    = Gt.cells.volumes;      % areal footprint of each column
poro_ve   = rock2D.poro;
perm_ve   = rock2D.perm(:, 1);
ncells_ve = Gt.cells.num;

% Interior face transmissibilities of the VE grid (K*H, harmonic average),
% exactly as CO2VEBlackOilTypeModel.setupOperators builds them.
rock_tmp      = rock2D;
rock_tmp.perm = rock2D.perm .* Gt.cells.H;
Thalf         = computeTrans(Gt, rock_tmp);
cf            = Gt.cells.faces(:, 1);
Tface         = 1 ./ accumarray(cf, 1 ./ Thalf, [Gt.faces.num, 1]);
N             = Gt.faces.neighbors;
intInx        = all(N ~= 0, 2);
T_ve          = Tface(intInx);
N_ve          = N(intInx, :);

pv_ve = poreVolume(Gt, rock2D) .* Gt.cells.H;

%% Sharp-interface VE fluid
[srw, srg] = deal(0.2, 0.1);
rhoW = 1050;  rhoG = 700;
muW  = 8e-4;  muG  = 6e-5;

fluid = makeVEFluid(Gt, rock2D, 'sharp_interface_simple', ...
                    'co2_mu_ref' , muG , ...
                    'wat_mu_ref' , muW , ...
                    'co2_rho_ref', rhoG, ...
                    'wat_rho_ref', rhoW, ...
                    'co2_rho_pvt', [0, 1e7], ...   % incompressible CO2
                    'wat_rho_pvt', [0, 1e7], ...   % incompressible brine
                    'residual'   , [srw, srg], ...
                    'krmax'      , [1, 1]);

% Evaluate the upscaled properties on a sweep of upscaled saturations,
% during drainage (sGmax == sG).
sg_q  = linspace(0, 1 - srw, 41)';
p_q   = repmat(1e7, ncells_ve, 1);
krG_q = zeros(numel(sg_q), 1);
krW_q = zeros(numel(sg_q), 1);
pc_q  = zeros(numel(sg_q), 1);
for i = 1 : numel(sg_q)
   sg = repmat(sg_q(i), ncells_ve, 1);
   krG_i = fluid.krG(sg, p_q, 'sGmax', sg);
   krW_i = fluid.krW(1 - sg, p_q, 'sGmax', sg);
   pc_i  = fluid.pcWG(sg, p_q, 'sGmax', sg);
   krG_q(i) = krG_i(1);
   krW_q(i) = krW_i(1);
   pc_q(i)  = pc_i(1);
end

% ... and during imbibition (sGmax held at 0.5) to exercise residual
% trapping in krW.
sgmax_fix = 0.5;
krW_imb = zeros(numel(sg_q), 1);
for i = 1 : numel(sg_q)
   sg = repmat(min(sg_q(i), sgmax_fix), ncells_ve, 1);
   sgm = repmat(sgmax_fix, ncells_ve, 1);
   kr = fluid.krW(1 - sg, p_q, 'sGmax', sgm);
   krW_imb(i) = kr(1);
end

save(fullfile(thisDir, 'ref_p9_ve.mat'), '-v7', ...
     'H_ve', 'z_ve', 'vol_ve', 'poro_ve', 'perm_ve', 'ncells_ve', ...
     'T_ve', 'N_ve', 'pv_ve', 'sg_q', 'krG_q', 'krW_q', 'pc_q', ...
     'krW_imb', 'sgmax_fix', 'srw', 'srg', 'rhoW', 'rhoG');
fprintf('Wrote Phase-9 VE reference: %d VE cells, H = %.2f m\n', ...
        ncells_ve, H_ve(1));
