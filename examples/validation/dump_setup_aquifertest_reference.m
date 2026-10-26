% Reference for setupAquifertest (validate_setup_aquifertest.py). In MRST
% 2024a the function cannot finish: AquiferBlackOilModel does not exist. This
% records the error and what the function computes before it: grid, rock and
% processAquifer's output, for both published decks.
here = fileparts(mfilename('fullpath'));
run(fullfile(here, '..', '..', '..', 'mrst-2024a-Geomech', 'startup.m'));
mrstModule add ad-core ad-blackoil ad-props deckformat
addpath(fullfile(ROOTDIR, 'modules', 'ad-blackoil', 'examples', 'utils'));
datadir = fullfile(ROOTDIR, 'examples', 'data', 'aquifertest');
out = struct();
for icase = 1:2
    fn = fullfile(datadir, sprintf('2D_OILWATER_AQ_ECLIPSE_%d.DATA', icase));
    try
        setupAquifertest(fn);
        err = '';
    catch e
        err = e.message;
    end
    deck = convertDeckUnits(readEclipseDeck(fn));
    G = computeGeometry(initEclipseGrid(deck));
    rock = compressRock(initEclipseRock(deck), G.cells.indexMap);
    output = processAquifer(deck, G);
    r = struct('err', err, 'ncells', G.cells.num, 'perm', rock.perm, 'poro', rock.poro, ...
               'aquifers', output.aquifers, 'pressures', output.initval.pressures, ...
               'volumes', output.initval.volumes, 'C', output.aquiferprops.C);
    out.(sprintf('case%d', icase)) = r;
end
case1 = out.case1; case2 = out.case2;
save(fullfile(here, 'ref_setup_aquifertest.mat'), '-v7', 'case1', 'case2');
