%% Dump Phase 420 reference: DG moment-fitting (fitMoments), from MRST.
%
% Run in MATLAB after starting MRST. Writes ref_p420_moment_fitting.mat.
%
% Moment-fitting yields a NON-UNIQUE minimal rule (greedy reduction + lsqnonneg),
% so PyMRST and MRST need not pick the same points. We call MRST's `fitMoments`
% directly with a deterministic candidate set (a uniform grid on [-1,1]^2) and
% the exact Legendre moments over the reference cell, and dump the resulting
% rule. PyMRST reproduces the same setup and checks both rules satisfy the same
% invariant: reproduce the moments (hence integrate every polynomial up to the
% precision exactly). The point count is recorded as a compression sanity-check.

mrstModule add dg

thisDir = fileparts(mfilename('fullpath'));
out = struct();

for k = 2:4
    basis = dgBasis(2, k, 'legendre');
    % Deterministic candidate grid on [-1,1]^2, shared with PyMRST.
    m = 2*k + 1;
    g = linspace(-1, 1, m);
    [X, Y] = ndgrid(g, g);
    x = [X(:), Y(:)];
    % Exact Legendre moments over the reference cell: M_j = integral psi_j.
    M = zeros(basis.nDof, 1);
    M(1) = 4;                                          % only the constant survives
    moments = arrayfun(@(v) v, M, 'UniformOutput', false);
    [pts, w, n] = fitMoments(x, basis, moments, 'equal', true);

    pre = sprintf('k%d_', k);
    out.([pre 'x']) = pts;
    out.([pre 'w']) = w(:);
    out.([pre 'n']) = n(1);
    out.([pre 'cand']) = x;
end

save(fullfile(thisDir, 'ref_p420_moment_fitting.mat'), '-v7', '-struct', 'out');
fprintf('Wrote ref_p420_moment_fitting.mat\n');
