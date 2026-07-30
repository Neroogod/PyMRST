%% Reference data for processFaults and removeFaultBdryFaces

thisDir = fileparts(mfilename('fullpath'));
G = cartGrid([3, 3, 2]);

geometry.FAULTS = {
    'F_MAIN', 2, 2, 1, 2, 1, 2, 'X+';
    'F_MAIN', 2, 2, 3, 3, 1, 2, 'X+';
    'BOUND',  1, 1, 1, 3, 1, 2, 'X-'
};
geometry.MULTFLT = {
    'F_MAIN', 0.25;
    'BOUND',  0.50
};

[faults, fault_id] = processFaults(G, geometry);
fault_names = {faults.name};
fault_numf = [faults.numf]';
fault_mult = [faults.mult]';
fault_pos = cumsum([1; fault_numf]);
fault_faces = vertcat(faults.faces);
id_bound = fault_id('BOUND');
id_main = fault_id('F_MAIN');

[interior_faults, keep] = removeFaultBdryFaces(faults, G);
interior_names = {interior_faults.name};
interior_numf = [interior_faults.numf]';
interior_mult = [interior_faults.mult]';
interior_pos = cumsum([1; interior_numf]);
interior_faces = vertcat(interior_faults.faces);

save(fullfile(thisDir, 'ref_grid_fault_processing.mat'), '-v7', ...
     'fault_names', 'fault_numf', 'fault_mult', 'fault_pos', 'fault_faces', ...
     'id_bound', 'id_main', 'interior_names', 'interior_numf', ...
     'interior_mult', 'interior_pos', 'interior_faces', 'keep');
fprintf('Wrote grid fault-processing reference to %s\n', thisDir);
