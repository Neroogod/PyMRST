%% Reference quadrature rules and polynomial dimensions from MRST's vem/utils
%
% `validate_phase403` pins `polygonInt`, which *uses* the triangle rule, so a
% compensating pair of errors in the rule's points and weights would survive
% it. This pins the rules themselves - the raw points, weights, reference
% vertices and volume - so the tables are evidence rather than an assumption
% shared by both sides of that test.
%
% `polyDim` is included for the same reason: everything that sizes a local VEM
% matrix calls it, and nothing validated it directly.

thisDir = fileparts(mfilename('fullpath'));
mrstModule add vem

out = struct();

% ---------------------------------------------------------------- polyDim
% Every (k, dim) the module actually uses, including the negative k that
% counts the (absent) interior moments of the first-order element.
ks   = -2:4;
dims = 1:3;
nk = zeros(numel(ks), numel(dims));
for a = 1:numel(ks)
    for b = 1:numel(dims)
        nk(a, b) = polyDim(ks(a), dims(b));
    end
end
out.polyDim_k    = ks(:);
out.polyDim_dim  = dims(:);
out.polyDim_vals = nk;

% ------------------------------------------------------ triangleQuadRule
for k = [1, 2, 3, 7]
    [Xq, w, V, vol] = triangleQuadRule(k);
    out.(sprintf('tri%d_Xq',  k)) = Xq;
    out.(sprintf('tri%d_w',   k)) = reshape(w, [], 1);
    out.(sprintf('tri%d_V',   k)) = V;
    out.(sprintf('tri%d_vol', k)) = vol;
end

% --------------------------------------------------- tetrahedronQuadRule
for k = [2, 3]
    [Xq, w, V, vol] = tetrahedronQuadRule(k);
    out.(sprintf('tet%d_Xq',  k)) = Xq;
    out.(sprintf('tet%d_w',   k)) = reshape(w, [], 1);
    out.(sprintf('tet%d_V',   k)) = V;
    out.(sprintf('tet%d_vol', k)) = vol;
end

% ----------------------------------------------------- squeezeBlockDiag
% Three 2x2 blocks folded each way, so the direction inference is pinned too.
blocks = {[1 2; 3 4], [5 6; 7 8], [9 10; 11 12]};
A = blkdiag(blocks{:});
out.sbd_A    = full(A);
out.sbd_down = full(squeezeBlockDiag(sparse(A), [2 2 2], size(A, 1), 2));
out.sbd_across = full(squeezeBlockDiag(sparse(A), [2 2 2], 2, size(A, 2)));

save(fullfile(thisDir, 'ref_vem_quadrature.mat'), '-struct', 'out', '-v7');
fprintf('Wrote ref_vem_quadrature.mat with %d fields\n', numel(fieldnames(out)));
