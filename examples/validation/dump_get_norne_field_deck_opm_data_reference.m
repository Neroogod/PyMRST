% Reference for getNorneFieldDeckOPMData (validate_get_norne_field_deck_opm_data.py):
% a fingerprint of the deck it reads and of the OPM and ECLIPSE outputs it parses.
here = fileparts(mfilename('fullpath'));
run(fullfile(here, '..', '..', '..', 'mrst-2024a-Geomech', 'startup.m'));
mrstModule add ad-core ad-blackoil ad-props deckformat
addpath(fullfile(ROOTDIR, 'modules', 'ad-blackoil', 'examples', 'utils'));
[deck, output] = getNorneFieldDeckOPMData();
cartDims = deck.GRID.cartDims;
permx_sum = sum(deck.GRID.PERMX);
poro_sum = sum(deck.GRID.PORO);
actnum_sum = sum(deck.GRID.ACTNUM);
nctrl = numel(deck.SCHEDULE.control);
step_val = deck.SCHEDULE.step.val;
opm = output.opm;
opm_nsteps = numel(opm.wellSols);
opm_names = {opm.wellSols{1}.name};
opm_bhp = cell2mat(cellfun(@(w) [w.bhp], opm.wellSols(1:5), 'UniformOutput', false));
opm_qOs = cell2mat(cellfun(@(w) [w.qOs], opm.wellSols(1:5), 'UniformOutput', false));
opm_time = opm.time(:);
opm_nstates = numel(opm.states);
opm_p1 = opm.states{1}.pressure;
opm_s1 = opm.states{1}.s;
opm_ncells = opm.G.cells.num;
[~, opm_location] = fileparts(fileparts(opm.location));
% initEclipseGrid splits Norne's disconnected parts: output.eclipse.G is a
% grid array.
ecl = output.eclipse;
ecl_numel = numel(ecl.G);
ecl_ncells = arrayfun(@(g) g.cells.num, ecl.G);
ecl_nsteps = numel(ecl.wellSols);
ecl_bhp = cell2mat(cellfun(@(w) [w.bhp], ecl.wellSols(1:5), 'UniformOutput', false));
ecl_nstates = numel(ecl.states);
save(fullfile(here, 'ref_get_norne_field_deck_opm_data.mat'), '-v7', 'cartDims', ...
     'permx_sum', 'poro_sum', 'actnum_sum', 'nctrl', 'step_val', 'opm_nsteps', 'opm_names', ...
     'opm_bhp', 'opm_qOs', 'opm_time', 'opm_nstates', 'opm_p1', 'opm_s1', 'opm_ncells', ...
     'opm_location', 'ecl_numel', 'ecl_nsteps', 'ecl_bhp', 'ecl_nstates', 'ecl_ncells');
