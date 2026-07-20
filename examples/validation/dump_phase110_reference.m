function dump_phase110_reference()
% Phase 110 reference: SPE1 black-oil to compositional deck conversion.

mrstModule add ad-core ad-blackoil compositional ad-props test-suite

pymrstRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
mrstRoot = fullfile(fileparts(pymrstRoot), 'mrst-2024a-Geomech');
mrstDataDirectory(fullfile(mrstRoot, 'examples', 'data'));
[G, rock, fluid, deck, state] = setupSPE1();

model = ThreePhaseBlackOilModel(G, rock, fluid, 'disgas', true);
model.extraStateOutput = true;
schedule = convertDeckScheduleToMRST(model, deck);

% Match MRST runCompositionalSPE1.m: compositional wells use a BHP producer.
schedule.control.W(2).type = 'bhp';
schedule.control.W(2).val = 200*barsa;
schedule.control.W(2).lims = [];

state0 = convertBlackOilStateToCompositional(model, state);
for i = 1:numel(schedule.control(1).W)
    schedule.control(1).W(i).components = schedule.control(1).W(i).compi(2:end);
end

nw = numel(schedule.control(1).W);
wellCells = zeros(nw, 1);
wellValues = zeros(nw, 1);
wellCompi = zeros(nw, 3);
wellComponents = zeros(nw, 2);
for i = 1:nw
    W = schedule.control(1).W(i);
    wellCells(i) = W.cells(1);
    wellValues(i) = W.val;
    wellCompi(i, :) = W.compi;
    wellComponents(i, :) = W.components;
end

scheduleControl2Type = char(schedule.control.W(2).type);
scheduleControl2Val = schedule.control.W(2).val;
scheduleControl2HasLims = ~isempty(schedule.control.W(2).lims);

pressure = state0.pressure;
s = state0.s;
components = state0.components;
x = state0.x;
y = state0.y;
L = state0.L;
T = state0.T;
initialRs = state.rs;
rhoOS = fluid.rhoOS;
rhoGS = fluid.rhoGS;
tsteps = convertDeckScheduleToMRST(model, deck).step.val;
cartDims = G.cartDims;

out = fullfile(fileparts(mfilename('fullpath')), 'ref_p110_spe1_compositional.mat');
save(out, 'cartDims', 'pressure', 's', 'components', 'x', 'y', 'L', 'T', ...
    'initialRs', 'rhoOS', 'rhoGS', 'wellCells', 'wellValues', 'wellCompi', ...
    'wellComponents', 'scheduleControl2Type', 'scheduleControl2Val', ...
    'scheduleControl2HasLims', 'tsteps');
fprintf('Wrote ref_p110_spe1_compositional.mat to %s\n', fileparts(out));
fprintf('  SPE1 converted state: %d cells, %d wells, %.0f report days\n', ...
    G.cells.num, nw, sum(tsteps)/day);
end
