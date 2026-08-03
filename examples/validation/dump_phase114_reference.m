function dump_phase114_reference()
% Phase 114 reference: native SPE1 K-value compositional schedule, two steps.

mrstModule add ad-core ad-blackoil compositional ad-props test-suite

pymrstRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
mrstRoot = fullfile(fileparts(pymrstRoot), 'mrst-2024a-Geomech');
mrstDataDirectory(fullfile(mrstRoot, 'examples', 'data'));
[G, rock, fluid, deck, state] = setupSPE1();

modelBO = ThreePhaseBlackOilModel(G, rock, fluid, 'disgas', true);
modelBO.extraStateOutput = true;
schedule = convertDeckScheduleToMRST(modelBO, deck);

% Match runCompositionalSPE1: producer is converted to BHP control.
schedule.control.W(2).type = 'bhp';
schedule.control.W(2).val = 200*barsa;
schedule.control.W(2).lims = [];

state0 = convertBlackOilStateToCompositional(modelBO, state);
for i = 1:numel(schedule.control(1).W)
    schedule.control(1).W(i).components = schedule.control(1).W(i).compi(2:end);
end

% Keep the validation short and deterministic.
schedule.step.val = schedule.step.val(1:2);
schedule.step.control = schedule.step.control(1:2);

modelNat = convertBlackOilModelToCompositionalModel(modelBO, 'interpolation', 'linear');
modelNat.fluid.rhoSinj = [fluid.rhoWS, fluid.rhoOS, fluid.rhoGS];
modelNat.FacilityModel = FacilityModel(modelNat);
% This used to run at toleranceWellRate = 1e-3 with MRST's default CNV of
% 1e-3, and the reference it produced was not converged: the overall and
% natural formulations of the same problem disagreed with each other by
% 7.08e-02 bar, which is a property of where each run stopped rather than of
% the physics. Tightening below collapses that spread to 6.3e-06 bar, at the
% cost of more iterations ([4 7] and [9 10] rather than [3 4] and [4 5]).
% Without this, phase 114's natural-variable tolerances have to be sized for
% MRST's convergence slack rather than for real disagreement.
modelNat.FacilityModel.toleranceWellRate = 1e-7;
modelNat.dsMaxAbs = 0.2;
modelNat.dzMaxAbs = inf;
modelNat.incTolPressure = inf;

modelOverall = OverallCompositionCompositionalModel(G, rock, fluid, modelNat.EOSModel);
modelOverall.fluid.rhoSinj = [fluid.rhoWS, fluid.rhoOS, fluid.rhoGS];
modelOverall.FacilityModel = FacilityModel(modelOverall);
modelOverall.FacilityModel.toleranceWellRate = modelNat.FacilityModel.toleranceWellRate;
modelOverall.dsMaxAbs = modelNat.dsMaxAbs;
modelOverall.dzMaxAbs = modelNat.dzMaxAbs;
modelOverall.incTolPressure = modelNat.incTolPressure;

% Same tightening on both models - see the note above. Both are tightened and
% the solver built before either is run, so the two formulations are compared
% under identical settings.
modelNat     = tightenP114(modelNat);
modelOverall = tightenP114(modelOverall);
p114solver = NonLinearSolver('maxIterations', 200);

[wsNat, statesNat, reportNat] = simulateScheduleAD(state0, modelNat, schedule, ...
    'NonLinearSolver', p114solver);
[wsOverall, statesOverall, reportOverall] = simulateScheduleAD(state0, modelOverall, schedule, ...
    'NonLinearSolver', p114solver);

[overall_pressure, overall_components, overall_s, overall_bhp, overall_qOs, overall_qGs] = localPack(statesOverall, wsOverall);
[natural_pressure, natural_components, natural_s, natural_bhp, natural_qOs, natural_qGs] = localPack(statesNat, wsNat);

overall_iterations = reportOverall.Iterations(:);
natural_iterations = reportNat.Iterations(:);
tsteps = schedule.step.val(:);

out = fullfile(fileparts(mfilename('fullpath')), 'ref_p114_spe1_native_compositional.mat');
save(out, 'overall_pressure', 'overall_components', 'overall_s', ...
    'overall_bhp', 'overall_qOs', 'overall_qGs', 'overall_iterations', ...
    'natural_pressure', 'natural_components', 'natural_s', ...
    'natural_bhp', 'natural_qOs', 'natural_qGs', 'natural_iterations', ...
    'tsteps');
fprintf('Wrote ref_p114_spe1_native_compositional.mat to %s\n', fileparts(out));
fprintf('  Native SPE1 compositional reference: %d cells, %d report steps\n', ...
    G.cells.num, numel(tsteps));
end

function m = tightenP114(m)
    props = {'toleranceCNV', 1e-7; 'toleranceMB', 1e-11; ...
             'nonlinearTolerance', 1e-9; 'incTolComposition', 1e-9};
    for i = 1:size(props, 1)
        if isprop(m, props{i,1})
            m.(props{i,1}) = props{i,2};
        end
    end
end

function [pressure, components, sat, bhp, qOs, qGs] = localPack(states, wellSols)
    nstep = numel(states);
    nc = numel(states{1}.pressure);
    ncomp = size(states{1}.components, 2);
    nw = numel(wellSols{1});
    pressure = zeros(nstep, nc);
    components = zeros(nstep, nc, ncomp);
    sat = zeros(nstep, nc, 3);
    bhp = zeros(nstep, nw);
    qOs = zeros(nstep, nw);
    qGs = zeros(nstep, nw);
    for k = 1:nstep
        st = states{k};
        pressure(k, :) = st.pressure(:)';
        components(k, :, :) = reshape(st.components, [1, nc, ncomp]);
        sat(k, :, :) = reshape(st.s, [1, nc, 3]);
        for w = 1:nw
            bhp(k, w) = wellSols{k}(w).bhp;
            qOs(k, w) = wellSols{k}(w).qOs;
            qGs(k, w) = wellSols{k}(w).qGs;
        end
    end
end
