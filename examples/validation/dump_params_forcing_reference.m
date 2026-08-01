%% Export advanced addBC/addSource structures for PyMRST validation.

thisDir = fileparts(mfilename('fullpath'));

bc = addBC([], [2; 4], {'Pressure', 'FLUX'}, 2, 'sat', [0.2, 0.8]);
bc = addBC(bc, 6, 'pressure', 5, 'sat', [0.6, 0.4]);
bcFace = bc.face;
bcValue = bc.value;
bcSat = bc.sat;
bcTypeCode = double(strcmp(bc.type, 'pressure'));

srcEmpty = addSource([], [1; 3], 3);
srcEmptyCell = srcEmpty.cell;
srcEmptyRate = srcEmpty.rate;
srcEmptySatSize = size(srcEmpty.sat);

src = addSource([], [1; 3], [1; -1], 'sat', [0.25, 0.75]);
src = addSource(src, 5, 2, 'sat', [0.6, 0.4]);
srcCell = src.cell;
srcRate = src.rate;
srcSat = src.sat;

save(fullfile(thisDir, 'ref_params_forcing.mat'), '-v7', ...
    'bcFace', 'bcValue', 'bcSat', 'bcTypeCode', ...
    'srcEmptyCell', 'srcEmptyRate', 'srcEmptySatSize', ...
    'srcCell', 'srcRate', 'srcSat');
fprintf('Wrote BC/source reference to %s\n', thisDir);
