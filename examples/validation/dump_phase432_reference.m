%% Dump Phase 432 reference: convertMultiSegmentWellsDeck, from MRST.

mrstModule add ad-core

thisDir = fileparts(mfilename('fullpath'));
G = computeGeometry(cartGrid([1, 1, 3], [1, 1, 30]));

W = struct('name', 'MS1', 'refDepth', 0, 'cells', (1:3)', ...
           'WI', [11; 22; 33], 'dZ', [5; 15; 25], ...
           'r', [0.1; 0.2; 0.3], 'cstatus', true(3, 1));

compRows = num2cell([ ...
    1, 1, 1, 1,  9, 11; ...
    1, 1, 2, 1, 19, 21; ...
    1, 1, 3, 1, 29, 31]);
ws.header = {0, 0, 2, 'INC'};
ws.segments = num2cell([ ...
    2, 2, 1, 1, 10, 10, 0.2, 1e-5, 3; ...
    3, 3, 1, 2, 10, 10, 0.2, 1e-5, 3; ...
    4, 4, 1, 3, 10, 10, 0.2, 1e-5, 3]);
deck_control.COMPSEGS = {'MS1', compRows};
deck_control.WELSEGS = {'MS1', ws};

out = convertMultiSegmentWellsDeck(W, G, deck_control);
out_is_ms = out.isMS;
out_cells = out.cells;
out_wi = out.WI;
out_dz = out.dZ;
out_r = out.r;
out_perf_map = out.perf_map;
out_cells_to_nodes = out.cells_to_nodes;
out_topo = out.topo;
out_node_depth = out.nodes.depth;
out_node_vol = out.nodes.vol;
out_node_branch = out.nodes.branch_id;
out_seg_length = out.segments.length;
out_seg_diameter = out.segments.diameter;
out_seg_roughness = out.segments.roughness;

save(fullfile(thisDir, 'ref_p432_ms_well_deck.mat'), '-v7', ...
     'out_is_ms', 'out_cells', 'out_wi', 'out_dz', 'out_r', ...
     'out_perf_map', 'out_cells_to_nodes', 'out_topo', ...
     'out_node_depth', 'out_node_vol', 'out_node_branch', ...
     'out_seg_length', 'out_seg_diameter', 'out_seg_roughness');
fprintf('Wrote ref_p432_ms_well_deck.mat to %s\n', thisDir);
