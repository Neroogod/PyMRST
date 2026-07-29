%% Extract DG reference cubature tables from MRST for the PyMRST port.
%
% Dumps, for each reference element and each supported precision k, the exact
% cubature points and weights from MRST's getters. The result seeds the PyMRST
% cubature tables (constants) and the Phase-418 validation.

mrstModule add dg

thisDir = fileparts(mfilename('fullpath'));

getters = {@getLineCubaturePointsAndWeights, ...
           @getSquareCubaturePointsAndWeights, ...
           @getTriangleCubaturePointsAndWeights, ...
           @getCubeCubaturePointsAndWeights, ...
           @getTetrahedronCubaturePointsAndWeights};
names = {'line', 'square', 'triangle', 'cube', 'tet'};

out = struct();
for gNo = 1:numel(getters)
    g = getters{gNo}; nm = names{gNo};
    ks = [];
    for k = 0:25
        try
            [x, w, n, xR] = g(k);
        catch
            continue
        end
        % Store only when the point count changes (one entry per distinct rule).
        key = sprintf('%s_k%d', nm, k);
        out.([key '_x']) = x;
        out.([key '_w']) = w(:);
        ks = [ks, k]; %#ok
    end
    out.([nm '_ks']) = ks;
    out.([nm '_xR']) = xR;
end

save(fullfile(thisDir, 'ref_dg_cubature_tables.mat'), '-v7', '-struct', 'out');
fprintf('Wrote ref_dg_cubature_tables.mat\n');
