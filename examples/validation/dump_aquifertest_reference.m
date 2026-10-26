% aquifertest in MRST 2024a without its figures (validate_ad_oo_aquifertest.py):
% both published cases simulated in full, the ECLIPSE result as
% loadAquiferEclipseResult reads it, and the relative difference the script plots.
here = fileparts(mfilename('fullpath'));
run(fullfile(here, '..', '..', '..', 'mrst-2024a-Geomech', 'startup.m'));
mrstModule add deckformat ad-props ad-core ad-blackoil
addpath(fullfile(ROOTDIR, 'modules', 'ad-blackoil', 'examples', 'utils'));
mrstVerbose true
gravity on
datadir = getDatasetPath('aquifertest', 'download', true);
out = struct();
for icase = 1 : 2
    fnroot = sprintf('2D_OILWATER_AQ_ECLIPSE_%d', icase);
    fn = fullfile(datadir, [fnroot, '.DATA']);
    deck = convertDeckUnits(readEclipseDeck(fn));
    [state0, model, schedule] = initEclipseProblemAD(deck);
    t = tic;
    [wellSols, states, schedulereport] = simulateScheduleAD(state0, model, schedule);
    r = struct();
    r.walltime = toc(t);
    r.model_class = class(model);
    r.aquifer_class = class(model.AquiferModel);
    r.p0 = state0.pressure; r.s0 = state0.s;
    r.its = cellfun(@(x) x.Iterations, schedulereport.ControlstepReports);
    r.nmini = cellfun(@(x) numel(x.StepReports), schedulereport.ControlstepReports);
    r.p = cell2mat(cellfun(@(st) st.pressure', states, 'UniformOutput', false));
    r.sw = cell2mat(cellfun(@(st) st.s(:, 1)', states, 'UniformOutput', false));
    r.aqp = cell2mat(cellfun(@(st) [st.aquiferSol.pressure], states, 'UniformOutput', false));
    r.aqv = cell2mat(cellfun(@(st) [st.aquiferSol.volume], states, 'UniformOutput', false));
    r.nwells = numel(wellSols{1});
    if r.nwells > 0
        r.bhp = cell2mat(cellfun(@(w) [w.bhp], wellSols, 'UniformOutput', false));
        r.qWs = cell2mat(cellfun(@(w) [w.qWs], wellSols, 'UniformOutput', false));
        r.qOs = cell2mat(cellfun(@(w) [w.qOs], wellSols, 'UniformOutput', false));
    end
    statesEcl = loadAquiferEclipseResult(datadir, fnroot);
    r.necl = numel(statesEcl);
    r.pecl_end = statesEcl{end}.pressure;
    r.err = (states{end}.pressure - statesEcl{end}.pressure*barsa)./(statesEcl{end}.pressure*barsa);
    out.(sprintf('case%d', icase)) = r;
    fprintf('case %d: %d steps, %d iterations, %.1f s\n', icase, numel(r.its), sum(r.its), r.walltime);
end
case1 = out.case1; case2 = out.case2;
save(fullfile(here, 'ref_aquifertest.mat'), '-v7', 'case1', 'case2');
