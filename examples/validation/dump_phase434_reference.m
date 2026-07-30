%% Dump Phase 434: combineMSwithRegularWells, from MRST.

mrstModule add ad-core
thisDir = fileparts(mfilename('fullpath'));

regular = struct('name', {'R1'; 'R2'}, ...
                 'cells', {[1]; [2]}, ...
                 'WI', {[10]; [20]});
ms = struct('name', 'MS1', 'cells', [3; 4], ...
            'nodes', struct('depth', [0; 10]), ...
            'topo', [1, 2]);
out = combineMSwithRegularWells(regular, ms);

names = {out.name};
is_ms = [out.isMS];
wi_empty = cellfun(@isempty, {out.WI});
nodes_empty = cellfun(@isempty, {out.nodes});
topo_empty = cellfun(@isempty, {out.topo});
cells = {out.cells};

save(fullfile(thisDir, 'ref_p434_combine_wells.mat'), '-v7', ...
     'names', 'is_ms', 'wi_empty', 'nodes_empty', 'topo_empty', 'cells');
fprintf('Wrote ref_p434_combine_wells.mat to %s\n', thisDir);
