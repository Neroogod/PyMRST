function dump_shale_operator_pipeline_reference(output_path)
% Direct MRST fixture for EDFM/pEDFM preprocessing and operator assembly.
if nargin ~= 1, error('Expected explicit output_path.'); end
this_file = mfilename('fullpath');
pymrst_root = fileparts(fileparts(fileparts(fileparts(this_file))));
workspace_root = fileparts(pymrst_root);
mrst_root = fullfile(workspace_root, 'mrst-2024a-Geomech');
run(fullfile(mrst_root, 'startup.m'));
mrstModule add shale hfm ad-core ad-props

tol = 1e-5;
Gm = computeGeometry(cartGrid([3, 3, 3], [3, 3, 3]));
Gm.rock = makeShaleRock(Gm, 100*milli*darcy, 0.2);
planes = struct;
planes(1).points = [1.5, .75, .75; 1.5, 2.25, .75; ...
                    1.5, 2.25, 2.25; 1.5, .75, 2.25];
planes(1).aperture = .01;
planes(1).poro = .5;
planes(1).perm = 10*darcy;

[Ge, planes] = EDFMshalegrid(Gm, planes, 'Tolerance', tol, ...
    'plotgrid', false, 'fracturelist', 1);
Ge = fracturematrixShaleNNC3D(Ge, tol);
[Ge, planes] = fracturefractureShaleNNCs3D(Ge, planes, tol);
Gped = pMatFracNNCs3D(Ge, tol);
Oe = setupShaleEDFMOpsTPFA(Ge, Ge.rock, tol);
Op = setupPEDFMOpsTPFA(Gped, Gped.rock, tol);

matrix_cells = Gm.cells.num;
edfm_cells = Ge.cells.num;
edfm_nnc_cells = Ge.nnc.cells;
edfm_nnc_T = Ge.nnc.T;
edfm_N = Oe.N;
edfm_T = Oe.T;
edfm_T_all = Oe.T_all;
edfm_pv = Oe.pv;
edfm_C = Oe.C;
pedfm_N = Op.N;
pedfm_T = Op.T;
pedfm_T_all = Op.T_all;
pedfm_pv = Op.pv;
pedfm_C = Op.C;
pedfm_projected_MM = Gped.nnc.pMMneighs;
n_projected = size(pedfm_projected_MM, 1);
pedfm_projected_MF = Gped.nnc.cells(end-n_projected+1:end, :);
pedfm_projected_T = Gped.nnc.T(end-n_projected+1:end);
pedfm_projected_area = Gped.nnc.area(end-n_projected+1:end);

save(output_path, 'matrix_cells', 'edfm_cells', 'edfm_nnc_cells', ...
    'edfm_nnc_T', 'edfm_N', 'edfm_T', 'edfm_T_all', 'edfm_pv', ...
    'edfm_C', 'pedfm_N', 'pedfm_T', 'pedfm_T_all', 'pedfm_pv', ...
    'pedfm_C', 'pedfm_projected_MF', 'pedfm_projected_T', ...
    'pedfm_projected_MM', 'pedfm_projected_area', '-v7');
fprintf('Wrote shale operator-pipeline fixture to %s\n', output_path);
end
