%% Raw WELSPECS/COMPDAT/WCON* reference for convertDeckScheduleToMRST.

mrstModule add ad-core ad-props ad-blackoil deckformat
thisDir = fileparts(mfilename('fullpath'));
G = computeGeometry(cartGrid([2, 1, 2], [20, 10, 20]));
rock = makeRock(G, 100*milli*darcy, 0.25);
fluid = initSimpleADIFluid('phases', 'WOG', 'n', [2, 2, 2], ...
    'rho', [1000, 800, 100], 'mu', [1, 5, 0.1]*centi*poise, ...
    'c', [0, 0, 0]);
model = GenericBlackOilModel(G, rock, fluid, ...
    'water', true, 'oil', true, 'gas', true);

welspecs = {
    'I', 'FIELD', 1, 1, 0, 'GAS', 0, 'STD', 'SHUT', 'YES', 0, 'SEG', 0;
    'P', 'FIELD', 2, 1, 0, 'OIL', 0, 'STD', 'SHUT', 'YES', 0, 'SEG', 0
};
compdat = {
    'I', 1, 1, 1, 2, 'OPEN', 1, 5, 0.2, -1, 0, 'Default', 'Z', -1;
    'P', 2, 1, 1, 2, 'OPEN', 1, 6, 0.2, -1, 0, 'Default', 'Z', -1
};
empty = cell(0, 0);
control = repmat(struct('WELSPECS', empty, 'COMPDAT', empty, ...
    'WCONINJE', empty, 'WCONPROD', empty), 2, 1);
control(1).WELSPECS = welspecs;
control(1).COMPDAT = compdat;
control(1).WCONINJE = {
    'I', 'GAS', 'OPEN', 'RATE', 100, 0, 300e5, 0, 0, 0, 0, 0, 0, 0
};
control(1).WCONPROD = {
    'P', 'OPEN', 'ORAT', 200, 0, 0, 0, 0, 100e5, 0, 0, 0
};
% readSCHEDULE-style controls are cumulative: geometry remains present while
% the operating records change.
control(2).WELSPECS = welspecs;
control(2).COMPDAT = compdat;
control(2).WCONINJE = {
    'I', 'GAS', 'SHUT', 'RATE', 50, 0, 300e5, 0, 0, 0, 0, 0, 0, 0
};
control(2).WCONPROD = {
    'P', 'OPEN', 'BHP', 0, 0, 0, 0, 0, 150e5, 0, 0, 0
};
scheduleDeck = struct();
scheduleDeck.step = struct('val', [10; 20], 'control', [1; 2]);
scheduleDeck.control = control;
schedule = convertDeckScheduleToMRST(model, scheduleDeck);

names = cell(2, 2); cells = cell(2, 2); types = cell(2, 2);
vals = zeros(2, 2); status = false(2, 2); compi = cell(2, 2); WI = cell(2, 2);
for c = 1:2
    for w = 1:2
        well = schedule.control(c).W(w);
        names{c, w} = well.name;
        cells{c, w} = well.cells;
        types{c, w} = well.type;
        vals(c, w) = well.val;
        status(c, w) = well.status;
        compi{c, w} = well.compi;
        WI{c, w} = well.WI;
    end
end
step_val = schedule.step.val;
step_control = schedule.step.control;
save(fullfile(thisDir, 'ref_ad_core_raw_deck_schedule.mat'), '-v7', ...
    'step_val', 'step_control', 'names', 'cells', 'types', 'vals', ...
    'status', 'compi', 'WI');
fprintf('Wrote ref_ad_core_raw_deck_schedule.mat\n');
