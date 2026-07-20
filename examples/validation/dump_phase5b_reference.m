%% Dump Phase 5b reference: SPE 1 via MRST ad-blackoil (first 30 steps).
%
% Standard MRST SPE1 workflow (readEclipseDeck -> ThreePhaseBlackOilModel
% with disgas) with tightened tolerances. Well limits are stripped so no
% control switching occurs in the comparison window.

thisDir = fileparts(mfilename('fullpath'));
mrstModule add ad-core ad-blackoil ad-props deckformat
gravity reset; gravity on

fn   = fullfile(ROOTDIR, 'examples', 'data', 'SPE1', 'BENCH_SPE1.DATA');
deck = readEclipseDeck(fn);
deck = convertDeckUnits(deck);

G     = initEclipseGrid(deck);
G     = computeGeometry(G);
rock  = initEclipseRock(deck);
rock  = compressRock(rock, G.cells.indexMap);
fluid = initDeckADIFluid(deck);
% Local MICP patch in this MRST install reads fluid.rhoSinj; equal to the
% stock surface densities it reproduces stock behavior.
fluid.rhoSinj = [fluid.rhoWS, fluid.rhoOS, fluid.rhoGS];

model = selectModelFromDeck(G, rock, fluid, deck);
model.toleranceCNV = 1e-8;
model.toleranceMB  = 1e-12;

schedule = convertDeckScheduleToMRST(model, deck);
for i = 1 : numel(schedule.control)
   for k = 1 : numel(schedule.control(i).W)
      schedule.control(i).W(k).lims = [];
   end
end
N = 30;
schedule.step.val     = schedule.step.val(1 : N);
schedule.step.control = schedule.step.control(1 : N);

p0  = deck.SOLUTION.PRESSURE;
sw0 = deck.SOLUTION.SWAT;
sg0 = deck.SOLUTION.SGAS;
rs0 = deck.SOLUTION.RS;
state0 = struct('pressure', p0, 's', [sw0, 1 - sw0 - sg0, sg0], ...
                'rs', rs0, 'rv', zeros(G.cells.num, 1));

model = model.validateModel(struct('W', schedule.control(1).W, ...
                                   'bc', [], 'src', []));
model.FacilityModel.toleranceWellRate = 1e-10;
model.FacilityModel.toleranceWellBHP  = 1e-4;

[wellSols, states] = simulateScheduleAD(state0, model, schedule);

nc = G.cells.num;
p_hist  = zeros(N, nc); sw_hist = zeros(N, nc);
sg_hist = zeros(N, nc); rs_hist = zeros(N, nc);
nw = numel(schedule.control(1).W);
[qWs, qOs, qGs, bhp] = deal(zeros(N, nw));
for i = 1 : N
   p_hist(i, :)  = states{i}.pressure;
   sw_hist(i, :) = states{i}.s(:, 1);
   sg_hist(i, :) = states{i}.s(:, 3);
   rs_hist(i, :) = states{i}.rs;
   qWs(i, :) = [wellSols{i}.qWs];
   qOs(i, :) = [wellSols{i}.qOs];
   qGs(i, :) = [wellSols{i}.qGs];
   bhp(i, :) = [wellSols{i}.bhp];
end
wnames = {schedule.control(1).W.name};
isProdFirst = strcmpi(wnames{1}, 'PRODUCER');
WI = [schedule.control(1).W.WI];
dts = schedule.step.val;

save(fullfile(thisDir, 'ref_p5b_spe1.mat'), '-v7', ...
     'p_hist', 'sw_hist', 'sg_hist', 'rs_hist', ...
     'qWs', 'qOs', 'qGs', 'bhp', 'WI', 'dts', 'isProdFirst');
gravity reset; gravity off
fprintf('Wrote Phase-5b SPE1 reference (%d steps) to %s\n', N, thisDir);
