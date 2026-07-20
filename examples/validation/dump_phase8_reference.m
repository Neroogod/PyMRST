%% Dump Phase 8 reference: CO2 properties + co2lab TwoPhaseWaterGasModel.
%
% Part 1: the sampled CO2 density/viscosity tables, evaluated on a grid of
%         (p, T) points, to verify the interpolation port.
% Part 2: a 3D CO2 injection into a brine aquifer, run with co2lab's
%         TwoPhaseWaterGasModel, to verify the coupled model.

thisDir = fileparts(mfilename('fullpath'));
mrstModule add ad-core ad-blackoil ad-props co2lab-common co2lab-ve
gravity reset on

%% Part 1: property tables
co2 = CO2props('sharp_phase_boundary', false, 'assert', false);
pq = linspace(50, 350, 37) * barsa;
tq = linspace(290, 400, 23);
[PQ, TQ] = ndgrid(pq, tq);
rho_tab = co2.rho(PQ(:), TQ(:));
mu_tab  = co2.mu(PQ(:), TQ(:));

%% Part 2: 3D CO2 injection
[nx, ny, nz] = deal(20, 1, 10);
G = cartGrid([nx, ny, nz], [2000, 100, 50]);
G.nodes.coords(:, 3) = G.nodes.coords(:, 3) + 1200;
G = computeGeometry(G);

rock = makeRock(G, 100 * milli * darcy, 0.25);

T_res = 273.15 + 45;
fluid = initSimpleADIFluid('phases', 'WG', 'mu', [8e-4, 1], ...
                           'rho', [1020, 1], 'n', [2, 2]);
fluid = addSampledFluidProperties(fluid, 'G', 'pspan', [0.1, 400] * mega, ...
                                  'tspan', [278, 524], 'props', [true true false false], ...
                                  'reservoirT', T_res, ...
                                  'assert_in_range', false, ...
                                  'nan_outside_range', false);
% Reference (surface) CO2 density and the resulting b-factor / viscosity.
rhoGS = co2.rho(1.01325e5, T_res);
fluid.rhoGS = rhoGS;
fluid.rhoWS = 1020;
fluid.bG = @(p, varargin) fluid.rhoG(p) ./ rhoGS;
fluid.muG = @(p, varargin) fluid.muG(p);
cW = 4.3e-10;  pRefW = 1e7;
fluid.bW = @(p, varargin) exp(cW * (p - pRefW));
fluid.muW = @(p, varargin) 8e-4 + 0 * p;
% Corey relperms with residual saturations, normalized as in PyMRST.
srw = 0.2;  src = 0.05;  den = 1 - srw - src;
fluid.krW = @(sw) (max(min((sw - srw) / den, 1), 0)) .^ 2;
fluid.krG = @(sg) (max(min((sg - src) / den, 1), 0)) .^ 2;

% This MRST install's local "OMO modification" in SimpleWell.m makes
% injectors read fluid.rhoSinj (the injected fluid's surface density) when
% computing the connection pressure drop -- exactly the case this CO2
% script needs. Set it to [brine, CO2] surface densities.
fluid.rhoSinj = [fluid.rhoWS, rhoGS];

model = TwoPhaseWaterGasModel(G, rock, fluid, 0, 0);
model.extraStateOutput = true;

inj = 1 + nx * ny * (nz - 1);        % bottom-left cell
massRate = 0.1e9 / year;             % 0.1 Mt CO2 / year, kg/s
qGs = massRate / rhoGS;              % surface volume rate
W = addWell([], G, rock, inj, 'Type', 'rate', 'Val', qGs, ...
            'Radius', 0.15, 'Comp_i', [0, 1], 'Sign', 1);

p0 = 1e5 + 1020 * norm(gravity) * G.cells.centroids(:, 3);
state0 = struct('pressure', p0, 's', [ones(G.cells.num, 1), ...
                zeros(G.cells.num, 1)], 'sGmax', zeros(G.cells.num, 1));

dts = repmat(60 * day, 8, 1);
schedule = simpleSchedule(dts, 'W', W);

model.toleranceCNV = 1e-10;
model.toleranceMB  = 1e-12;
[wellSols, states] = simulateScheduleAD(state0, model, schedule);

ns = numel(states);  nc = G.cells.num;
p_hist  = zeros(ns, nc);
sg_hist = zeros(ns, nc);
qGs_hist = zeros(ns, 1);  bhp_hist = zeros(ns, 1);
for i = 1 : ns
   p_hist(i, :)  = states{i}.pressure;
   sg_hist(i, :) = states{i}.s(:, 2);
   qGs_hist(i) = wellSols{i}.qGs;
   bhp_hist(i) = wellSols{i}.bhp;
end
WI = W.WI;

save(fullfile(thisDir, 'ref_p8_co2.mat'), '-v7', ...
     'pq', 'tq', 'rho_tab', 'mu_tab', ...
     'p_hist', 'sg_hist', 'qGs_hist', 'bhp_hist', 'WI', 'rhoGS', ...
     'dts', 'qGs', 'T_res');
fprintf('Wrote Phase-8 CO2 reference (%d steps, max sG = %.4f)\n', ...
        ns, max(sg_hist(end, :)));
