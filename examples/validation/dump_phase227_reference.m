function dump_phase227_reference()
% Dump Phase 227 reference: MRST psideh hydrostatic side pressure.
%
% This isolates MRST's core psideh implementation on a small Cartesian grid.
% PyMRST compares face selection, hydrostatic pressure values, saturation, and
% selected face centroids directly against this MAT-file.

pymrstRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
mrstRoot = fullfile(fileparts(pymrstRoot), 'mrst-2024a-Geomech');
run(fullfile(mrstRoot, 'startup.m'));
mrstModule add ad-core ad-props

gravity reset on

dims = [4, 2, 3];
pdims = [40, 20, 30];
G = cartGrid(dims, pdims);
G = computeGeometry(G);

sat = [0.25, 0.75];
mu = [2.0, 4.0];
rho = [1000.0, 700.0];
fluid = struct();
fluid.properties = @(state) deal(mu, rho);
fluid.relperm = @(s) s;

side = 'zmax';
i1 = 2;                  % MATLAB/MRST uses 1-based logical ranges.
i2 = 1;
normal_range = 3;
ref_position = [0.0, 0.0, 5.0];
ref_pressure = 1.0e5;

bc = [];
bc = psideh(bc, G, side, fluid, i1, i2, ...
            'sat', sat, ...
            'range', normal_range, ...
            'ref_position', ref_position, ...
            'ref_pressure', ref_pressure);

[props_mu, props_rho] = fluid.properties(struct('s', sat));
kr = fluid.relperm(sat);
mob = kr ./ props_mu;
omega = dot(mob, props_rho) ./ sum(mob, 2);

bc_faces = bc.face;
bc_values = bc.value;
bc_sat = bc.sat;
bc_types = bc.type;
face_centroids = G.faces.centroids(bc.face, :);
gravity_vector = gravity;

out = fullfile(fileparts(mfilename('fullpath')), 'ref_p227_psideh.mat');
save(out, '-v7', ...
     'dims', 'pdims', 'side', 'i1', 'i2', 'normal_range', ...
     'sat', 'mu', 'rho', 'kr', 'omega', ...
     'ref_position', 'ref_pressure', 'gravity_vector', ...
     'bc_faces', 'bc_values', 'bc_sat', 'bc_types', 'face_centroids');

fprintf('Wrote ref_p227_psideh.mat to %s\n', fileparts(mfilename('fullpath')));
fprintf('  face ids: %s\n', mat2str(bc_faces(:)'));
fprintf('  pressure: %s\n', mat2str(bc_values(:)', 12));
end
