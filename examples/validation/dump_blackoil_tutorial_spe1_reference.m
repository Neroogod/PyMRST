% blackoilTutorialSPE1, as published, without the plots; saves the results.
here = fileparts(mfilename('fullpath'));
run(fullfile(here, '..', '..', '..', 'mrst-2024a-Geomech', 'startup.m'));
mrstModule add ad-props deckformat ad-core ad-blackoil test-suite
[G, rock, fluid, deck, state] = setupSPE1();
model = selectModelFromDeck(G, rock, fluid, deck);
schedule = convertDeckScheduleToMRST(model, deck);
nls = NonLinearSolver('useLinesearch', true);
[wellSols, states, report] = simulateScheduleAD(state, model, schedule, 'nonlinearsolver', nls);
load SPE1_smry
inj = find([wellSols{1}.sign] == 1);
prod = find([wellSols{1}.sign] == -1);
ind = 2:size(smry.data,2);
[qWs, qOs, qGs, bhp] = wellSolToVector(wellSols);
T = convertTo(cumsum(schedule.step.val), year);
Tcomp = smry.get(':+:+:+:+', 'YEARS', ind);
gor_ecl = convertFrom(smry.get('PRODUCER', 'WGOR', ind), 1000*ft^3/stb)';
gor_mrst = qGs(:,prod)./qOs(:,prod);
bhp_prod_ecl = convertFrom(smry.get('PRODUCER', 'WBHP', ind), psia)';
bhp_inj_ecl = convertFrom(smry.get('INJECTOR', 'WBHP', ind), psia)';
grab = @(st, f) cell2mat(cellfun(@(s) s.(f)', st, 'UniformOutput', false));
p = grab(states, 'pressure'); rs = grab(states, 'rs');
sw = cell2mat(cellfun(@(s) s.s(:, 1)', states, 'UniformOutput', false));
sg = cell2mat(cellfun(@(s) s.s(:, 3)', states, 'UniformOutput', false));
its = report.Iterations; rtime = report.ReservoirTime;
names = {wellSols{1}.name};
types = cellfun(@(ws) strjoin({ws.type}, ','), wellSols, 'UniformOutput', false);
save(fullfile(here, 'ref_blackoil_tutorial_spe1.mat'), 'p', 'rs', 'sw', 'sg', ...
     'qWs', 'qOs', 'qGs', 'bhp', 'its', 'rtime', 'T', 'Tcomp', 'gor_ecl', 'gor_mrst', ...
     'bhp_prod_ecl', 'bhp_inj_ecl', 'inj', 'prod', 'names', 'types', '-v7');
