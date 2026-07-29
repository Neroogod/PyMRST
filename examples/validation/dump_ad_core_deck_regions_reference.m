%% Direct MRST reference for getInitializationRegionsDeck splitting.

mrstModule add ad-core ad-props ad-blackoil
thisDir = fileparts(mfilename('fullpath'));
G = computeGeometry(cartGrid([6, 1]));
rock = makeRock(G, 1, 0.2);
fluid = initSimpleADIFluid('phases', 'WO', 'n', [2, 2], ...
                           'mu', [1, 1], 'rho', [1000, 800]);
model = GenericBlackOilModel(G, rock, fluid, 'gas', false);
model = model.validateModel();
deck.SOLUTION.EQUIL = [1000, 1e7, 1100, 5, 900, 0, 0, 0, 0, 0; ...
                       1200, 1.2e7, 1300, 7, 1000, 0, 0, 0, 0, 0];
deck.REGIONS.EQLNUM = [1; 1; 1; 2; 2; 2];
deck.REGIONS.SATNUM = [1; 1; 2; 1; 1; 1];
deck.REGIONS.PVTNUM = [1; 2; 1; 1; 1; 2];
regions = getInitializationRegionsDeck(model, deck);
n = numel(regions);
cells = cell(n, 1);
contacts = cell(n, 1);
contacts_pc = cell(n, 1);
datum_depth = zeros(n, 1);
datum_pressure = zeros(n, 1);
for i = 1:n
    cells{i} = regions{i}.cells;
    contacts{i} = regions{i}.contacts;
    contacts_pc{i} = regions{i}.contacts_pc;
    datum_depth(i) = regions{i}.datum_depth;
    datum_pressure(i) = regions{i}.datum_pressure;
end
save(fullfile(thisDir, 'ref_ad_core_deck_regions.mat'), '-v7', ...
    'cells', 'contacts', 'contacts_pc', 'datum_depth', 'datum_pressure');
fprintf('Wrote ref_ad_core_deck_regions.mat\n');
