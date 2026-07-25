%% Dump Phase 359 reference: selectModelFromDeck routing on SPE1, from MRST.
%
% Run in MATLAB after starting MRST. Writes ref_p359_selectmodel.mat.
%
% AC23 ports `selectModelFromDeck` - the factory that turns a parsed deck's
% RUNSPEC flags into the right model class and its disgas/vapoil settings. This
% pins the routing *decision* against MRST on the real SPE1 deck: the model
% class MRST picks, and the disgas/vapoil it resolves, together with the RUNSPEC
% flags and PVT-table presence that drove the choice. PyMRST rebuilds a deck
% dict from those same flags and must route to the same decision.

mrstModule add ad-core ad-props ad-blackoil deckformat

gravity reset on;
thisDir = fileparts(mfilename('fullpath'));

deck = convertDeckUnits(readEclipseDeck( ...
    fullfile(thisDir, '..', '..', '..', 'mrst-2024a-Geomech', ...
             'examples', 'data', 'SPE1', 'BENCH_SPE1.DATA')));
try; mrstSettings('set', 'useMEX', false); catch; end
[~, model] = initEclipseProblemAD(deck, 'useMex', false, ...
                                  'AutoDiffBackend', 'sparse');

model_class = class(model);
disgas = double(model.disgas);
vapoil = double(model.vapoil);

rs = deck.RUNSPEC;
check = @(name) double(isfield(rs, upper(name)) && rs.(upper(name)));
flag_water = check('water');
flag_oil   = check('oil');
flag_gas   = check('gas');
flag_disgas_rs = double(isfield(rs, 'DISGAS') && rs.DISGAS);
flag_vapoil_rs = double(isfield(rs, 'VAPOIL') && rs.VAPOIL);
flag_blackoil  = check('blackoil');
has_pvto = double(isfield(deck.PROPS, 'PVTO'));
has_pvtg = double(isfield(deck.PROPS, 'PVTG'));

save(fullfile(thisDir, 'ref_p359_selectmodel.mat'), '-v7', ...
     'model_class', 'disgas', 'vapoil', ...
     'flag_water', 'flag_oil', 'flag_gas', 'flag_disgas_rs', ...
     'flag_vapoil_rs', 'flag_blackoil', 'has_pvto', 'has_pvtg');

fprintf('Wrote ref_p359_selectmodel.mat to %s\n', thisDir);
fprintf('  class = %s, disgas = %d, vapoil = %d\n', ...
        model_class, disgas, vapoil);
fprintf('  flags: W%d O%d G%d  DISGAS%d VAPOIL%d BO%d  PVTO%d PVTG%d\n', ...
        flag_water, flag_oil, flag_gas, flag_disgas_rs, flag_vapoil_rs, ...
        flag_blackoil, has_pvto, has_pvtg);
