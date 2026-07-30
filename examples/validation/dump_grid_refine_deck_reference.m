%% Reference data for refineDeck

thisDir = fileparts(mfilename('fullpath'));
deck.RUNSPEC.DIMENS = [2, 1, 2];
deck.GRID = cartesianGrdecl(0:2, 0:1, 0:2);
deck.GRID.PORO = [0.1; 0.2; 0.3; 0.4];
deck.SOLUTION.PRESSURE = [100; 200; 300; 400];
deck.SOLUTION.SWAT = [0.1; 0.2; 0.3; 0.4];
deck.REGIONS.SATNUM = [1; 2; 3; 4];
deck.PROPS.SWL = [0.01; 0.02; 0.03; 0.04];

control.WELSPECS = {
    'WZ', 'G', 1, 1;
    'WX', 'G', 2, 1
};
control.COMPDAT = {
    'WZ', 1, 1, 1, 2, 'OPEN', 0, 12, 0, 18, 0, 0, 'Z';
    'WX', 2, 1, 1, 1, 'OPEN', 0, 20, 0, 30, 0, 0, 'X'
};
deck.SCHEDULE.control = control;

refined = refineDeck(deck, [2, 3, 2]);
refined_dims = refined.RUNSPEC.DIMENS;
refined_grid_dims = refined.GRID.cartDims;
refined_pressure = refined.SOLUTION.PRESSURE;
refined_swat = refined.SOLUTION.SWAT;
refined_satnum = refined.REGIONS.SATNUM;
refined_swl = refined.PROPS.SWL;
refined_poro = refined.GRID.PORO;
refined_welspec_ij = cell2mat(refined.SCHEDULE.control.WELSPECS(:, 3:4));
refined_compdat_ijk = cell2mat(refined.SCHEDULE.control.COMPDAT(:, 2:5));
refined_compdat_trans = cell2mat(refined.SCHEDULE.control.COMPDAT(:, 8));
refined_compdat_kh = cell2mat(refined.SCHEDULE.control.COMPDAT(:, 10));
refined_compdat_dir = refined.SCHEDULE.control.COMPDAT(:, 13);

defaulted = refineDeck(deck, [2, 1, 1], 'default_well', true);
default_trans = strcmp(defaulted.SCHEDULE.control.COMPDAT(:, 8), '-1');
default_kh = strcmp(defaulted.SCHEDULE.control.COMPDAT(:, 10), '-1');

save(fullfile(thisDir, 'ref_grid_refine_deck.mat'), '-v7', ...
     'refined_dims', 'refined_grid_dims', 'refined_pressure', ...
     'refined_swat', 'refined_satnum', 'refined_swl', 'refined_poro', ...
     'refined_welspec_ij', 'refined_compdat_ijk', ...
     'refined_compdat_trans', 'refined_compdat_kh', ...
     'refined_compdat_dir', 'default_trans', 'default_kh');
fprintf('Wrote refineDeck reference to %s\n', thisDir);
