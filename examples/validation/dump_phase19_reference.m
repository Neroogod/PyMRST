% Phase 19 reference: the Norwegian CO2 Storage Atlas (co2lab-common/atlasgrid)
%
% Builds the Johansen formation from the NPD rasters and dumps the corner-point
% GRDECL, so the Python reader/converter can be compared node for node.

mrstModule add co2lab-common co2lab-spillpoint ad-core

here = fileparts(mfilename('fullpath'));

[meta_top,   data_top]   = readAAIGrid(fullfile(getDatasetPath('co2atlas'), 'Johansenfm_top'));
[meta_thick, data_thick] = readAAIGrid(fullfile(getDatasetPath('co2atlas'), 'Johansenfm_thickness'));

% getAtlasGrid's sign convention: top surfaces are stored as negative depths
if any(data_top(:) < 0), data_top = -data_top; end

grdecl = convertAtlasTo3D(meta_thick, meta_top, data_thick, data_top, 1);

ref = struct();
ref.cartDims = double(grdecl.cartDims(:));
ref.COORD    = grdecl.COORD(:);
ref.ZCORN    = grdecl.ZCORN(:);
ref.ACTNUM   = double(grdecl.ACTNUM(:));

% also the raw reader output, so the two stages can be told apart
ref.top_data   = data_top;
ref.thick_data = data_thick;
ref.top_meta   = [meta_top.ncols, meta_top.nrows, meta_top.cellsize, ...
                  meta_top.xllcorner, meta_top.yllcorner];

save(fullfile(here, 'ref_p19_atlas.mat'), '-struct', 'ref', '-v7');

fprintf(['Wrote Phase-19 reference: cartDims %dx%dx%d, %d active cells, ' ...
         '%d finite ZCORN\n'], grdecl.cartDims(1), grdecl.cartDims(2), ...
        grdecl.cartDims(3), sum(grdecl.ACTNUM), sum(isfinite(grdecl.ZCORN)));
