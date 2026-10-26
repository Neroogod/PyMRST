% Reference for NumericalPressureReductionFactors
% (validate_ad_core_numerical_pressure_reduction_factors.py): SPE9 as a
% GenericBlackOilModel with the states of dump_equations_black_oil_reference.m.
% Weights at a first iterate (no previous weights, numeric pressure) and at a
% second, perturbed iterate with AD pressure and the first weights as previous,
% which gives each weight a pressure derivative.
here = fileparts(mfilename('fullpath'));
run(fullfile(here, '..', '..', '..', 'mrst-2024a-Geomech', 'startup.m'));
mrstModule add ad-core ad-blackoil ad-props deckformat test-suite
mrstSettings('set', 'useMEX', false);
gravity reset on
[G, rock, fluid, deck, state0] = setupSPE9();
model = GenericBlackOilModel(G, rock, fluid, 'disgas', true, 'inputdata', deck);
schedule = convertDeckScheduleToMRST(model, deck);
[~, forces] = model.getDrivingForces(schedule.control(1));
model = model.validateModel(forces);
st0 = model.validateState(state0);
[model, st0] = model.updateForChangedControls(st0, forces);
nc = G.cells.num;
st = st0;
gas = (1:7:nc)';
st.s(gas, 3) = 0.05;
st.s(gas, 2) = 1 - st.s(gas, 1) - st.s(gas, 3);
st.pressure = st.pressure.*(1 + 0.01*sin((1:nc)'));
st.rs(gas) = fluid.rsSat(st.pressure(gas));
dt = 1*day;
rf = NumericalPressureReductionFactors(model);
mass0 = model.getProp(st0, 'ComponentTotalMass');
stA = st;
stA.reductionFactorProps = struct('mass0', {mass0}, 'mass', [], 'dt', dt, ...
                                  'pressure', [], 'weights', []);
wA = rf.evaluateOnDomain(model, stA);
isAD_A = cellfun(@(x) isa(x, 'ADI'), wA);
wA = [wA{:}];
stB = stA;
stB.reductionFactorProps.pressure = stA.pressure;
stB.reductionFactorProps.weights = wA;
stB.pressure = stA.pressure.*(1 + 0.002*cos((1:nc)'));
pB = stB.pressure;
stB = model.getStateAD(stB);
wB = rf.evaluateOnDomain(model, stB);
isAD_B = cellfun(@(x) isa(x, 'ADI'), wB);
wBval = cellfun(@value, wB, 'UniformOutput', false);
wBval = [wBval{:}];
dB = zeros(nc, numel(wB)); offdiag = []; njac = [];
fprintf("isAD_B %s\n", mat2str(isAD_B));
for i = 1:numel(wB)
    if isa(wB{i}, 'ADI')
        dB(:, i) = full(diag(wB{i}.jac{1}));
        offdiag(i) = nnz(wB{i}.jac{1}) - nnz(diag(wB{i}.jac{1})); %#ok<SAGROW>
        njac(i) = numel(wB{i}.jac); %#ok<SAGROW>
    end
end
p0 = st0.pressure; s0 = st0.s; rs0 = st0.rs;
p = st.pressure; s = st.s; rs = st.rs;
save(fullfile(here, 'ref_numerical_pressure_reduction_factors.mat'), '-v7', ...
     'wA', 'isAD_A', 'wBval', 'dB', 'isAD_B', 'offdiag', 'njac', 'pB', 'dt', ...
     'p0', 's0', 'rs0', 'p', 's', 'rs');
