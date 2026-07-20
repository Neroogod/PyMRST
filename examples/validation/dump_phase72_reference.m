%% Dump Phase 72 reference: vemmech deformed plotting geometry.
%
% Run in MATLAB after starting MRST (run startup.m in mrst-2024a-Geomech).
% Writes ref_p72_vemmech_plotting.mat; then run validate_phase72.py.

mrstModule add vemmech

thisDir = fileparts(mfilename('fullpath'));

G2 = computeGeometry(cartGrid([2, 1], [2, 1]));
G2 = createAugmentedGrid(G2);
x2 = G2.nodes.coords(:, 1);
y2 = G2.nodes.coords(:, 2);
u2 = [0.1*x2, -0.2*y2];
node_data2 = [(1:G2.nodes.num)', 10 + (1:G2.nodes.num)'];
G2d = G2;
G2d.nodes.coords = G2d.nodes.coords + u2;

cells2 = (1:G2.cells.num)';
nn2 = diff([G2.cells.facePos(cells2), G2.cells.facePos(cells2 + 1)], [], 2);
cn2 = G2.cells.nodes(mcolon(G2.cells.nodePos(cells2), ...
                            G2.cells.nodePos(cells2 + 1) - 1), 1);
m2 = numel(cells2);
n2 = max(nn2);
f2 = nan([n2, m2]);
present2 = false([G2.nodes.num, 1]);
present2(cn2) = true;
node_num2 = zeros([G2.nodes.num, 1]);
node_num2(present2) = 1:sum(double(present2));
off2 = reshape((0:m2 - 1).*n2, [], 1);
f2(mcolon(off2 + 1, off2 + nn2)) = node_num2(cn2);
f2 = f2.';
verts2 = find(present2);
v2 = G2d.nodes.coords(verts2, :);
cdata2 = node_data2(verts2, :);

faces2 = (1:G2.faces.num)';
shift2 = linspace(0.01, 0.05, G2.faces.num)';
nodes2 = G2.faces.nodes(mcolon(G2.faces.nodePos(faces2), ...
                               G2.faces.nodePos(faces2 + 1) - 1));
nodes2 = reshape(nodes2, 2, []).';
line_x2 = zeros(numel(faces2), 2);
line_y2 = zeros(numel(faces2), 2);
for j = 1:size(nodes2, 1)
    pos1 = G2d.nodes.coords(nodes2(j, :), 1);
    pos2 = G2d.nodes.coords(nodes2(j, :), 2);
    pos1 = pos1 + G2.faces.normals(faces2(j), 1)*shift2(j);
    pos2 = pos2 + G2.faces.normals(faces2(j), 2)*shift2(j);
    line_x2(j, :) = pos1(:).';
    line_y2(j, :) = pos2(:).';
end

G3 = computeGeometry(cartGrid([1, 1, 1], [1, 2, 3]));
x3 = G3.nodes.coords(:, 1);
y3 = G3.nodes.coords(:, 2);
z3 = G3.nodes.coords(:, 3);
u3 = [0.01*x3, -0.02*y3, 0.03*z3];
node_data3 = (1:G3.nodes.num)';
G3d = G3;
G3d.nodes.coords = G3d.nodes.coords + u3;
cells3 = (1:G3.cells.num)';
[faces3, ~] = boundaryFaces(G3, cells3);
nn3 = diff([G3.faces.nodePos(faces3), G3.faces.nodePos(faces3 + 1)], [], 2);
fn3 = G3.faces.nodes(mcolon(G3.faces.nodePos(faces3), ...
                            G3.faces.nodePos(faces3 + 1) - 1), 1);
m3 = numel(faces3);
n3 = max(nn3);
f3 = nan([n3, m3]);
present3 = false([G3.nodes.num, 1]);
present3(fn3) = true;
node_num3 = zeros([G3.nodes.num, 1]);
node_num3(present3) = 1:sum(double(present3));
off3 = reshape((0:m3 - 1).*n3, [], 1);
f3(mcolon(off3 + 1, off3 + nn3)) = node_num3(fn3);
tmp = isfinite(f3);
nnode3 = sum(tmp, 1);
ind3 = sub2ind(size(f3), nnode3, 1:size(f3, 2));
tmp_last = repmat(f3(ind3), size(f3, 1), 1);
f3(isnan(f3)) = tmp_last(isnan(f3));
f3 = f3.';
verts3 = find(present3);
v3 = G3d.nodes.coords(verts3, :);
cdata3 = node_data3(verts3, :);

save(fullfile(thisDir, 'ref_p72_vemmech_plotting.mat'), '-v7', ...
     'u2', 'node_data2', 'G2d', 'f2', 'verts2', 'v2', 'cdata2', ...
     'faces2', 'shift2', 'line_x2', 'line_y2', ...
     'u3', 'node_data3', 'G3d', 'faces3', 'f3', 'verts3', 'v3', 'cdata3');

fprintf('Wrote ref_p72_vemmech_plotting.mat to %s\n', thisDir);
