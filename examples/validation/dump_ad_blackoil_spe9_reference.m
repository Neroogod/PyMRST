function dump_ad_blackoil_spe9_reference()
% Deterministic MRST fixture for PyMRST's SPE9 deck/setup regression.
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
mrst = fullfile(fileparts(root), 'mrst-2024a-Geomech');
cd(mrst); startup; mrstModule add ad-blackoil ad-core ad-props deckformat test-suite;
[G, rock, fluid, deck, state0] = setupSPE9();
model = selectModelFromDeck(G, rock, fluid, deck);
schedule = convertDeckScheduleToMRST(model, deck);
schedule1 = schedule;
schedule1.step.val = schedule.step.val(1);
schedule1.step.control = schedule.step.control(1);
[wellsol1, states1] = simulateScheduleAD(state0, model, schedule1);
state1_pressure = states1{1}.pressure;
state1_sw = states1{1}.s(:, 1);
state1_sg = states1{1}.s(:, 3);
state1_rs = states1{1}.rs;
state1_qWs = [wellsol1{1}.qWs]';
state1_qOs = [wellsol1{1}.qOs]';
state1_qGs = [wellsol1{1}.qGs]';
state1_bhp = [wellsol1{1}.bhp]';

nc = G.cells.num;
nf = G.faces.num;
nn = G.nodes.num;
cartDims = G.cartDims;
pressure = state0.pressure;
sw = state0.s(:, 1);
sg = state0.s(:, 3);
rs = state0.rs;
poro = rock.poro;
perm = rock.perm;
step_val = schedule.step.val;
step_control = schedule.step.control;
ncontrols = numel(schedule.control);
nwells = arrayfun(@(c) numel(c.W), schedule.control);
well_names = {schedule.control(1).W.name};
well_cells = arrayfun(@(w) numel(w.cells), schedule.control(1).W);
well_WI = cellfun(@(x) x, {schedule.control(1).W.WI}, 'UniformOutput', false);
well_type = cell(ncontrols, 1);
well_val = cell(ncontrols, 1);
well_bhp_limit = cell(ncontrols, 1);
for c = 1:ncontrols
    well_type{c} = {schedule.control(c).W.type};
    well_val{c} = [schedule.control(c).W.val];
    well_bhp_limit{c} = arrayfun(@(w) w.lims.bhp, schedule.control(c).W);
end
out = fullfile(root, 'examples', 'validation', 'ref_ad_blackoil_spe9.mat');
save(out, 'nc', 'nf', 'nn', 'cartDims', 'pressure', 'sw', 'sg', 'rs', ...
    'poro', 'perm', 'step_val', 'step_control', 'ncontrols', 'nwells', ...
    'well_names', 'well_cells', 'well_WI', 'well_type', 'well_val', ...
    'well_bhp_limit', 'state1_pressure', 'state1_sw', 'state1_sg', ...
    'state1_rs', 'state1_qWs', 'state1_qOs', 'state1_qGs', 'state1_bhp', '-v7');
fprintf('Saved %s\n', out);
end
