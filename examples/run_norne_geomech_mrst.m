%% MRST Norne geomechanics example matching PyMRST's run_norne_geomech.py.
%
% Run from MATLAB:
%
%   run('/Users/Shared/PyMRST_workspace/PyMRST/examples/run_norne_geomech_mrst.m')
%
% Or, from the MRST root after `startup`:
%
%   cd('/Users/Shared/PyMRST_workspace/PyMRST/examples')
%   run_norne_geomech_mrst
%
% This uses MRST's own ad-mechanics example:
%
%   runNorneExample('norne_case', 'mini Norne', ...
%                   'fluid_model', 'water', ...
%                   'method', 'fully coupled', ...
%                   'bc_case', 'bottom fixed')
%
% That is the MATLAB/MRST counterpart to:
%
%   python examples/run_norne_geomech.py

clearvars;

thisDir = fileparts(mfilename('fullpath'));
repoRoot = fullfile(thisDir, '..', '..');
mrstRoot = fullfile(repoRoot, 'mrst-2024a-Geomech');

if exist(fullfile(mrstRoot, 'startup.m'), 'file')
    old = pwd;
    cd(mrstRoot);
    startup;
    cd(old);
end

mrstModule add ad-mechanics ad-core ad-props ad-blackoil vemmech deckformat mrst-gui
gravity on;
localUsePatchedRunNorneExample();

fprintf('\n======================================================================\n');
fprintf('MRST Norne coupled flow + geomechanics example\n');
fprintf('======================================================================\n');
fprintf('Case    : mini Norne, water, fully coupled, bottom fixed\n');
fprintf('Source  : MRST modules/ad-mechanics/examples/runNorneExample.m\n\n');

t0 = tic;
[model, states, initState] = runNorneExample( ...
    'norne_case', 'mini Norne', ...
    'bc_case', 'bottom fixed', ...
    'method', 'fully coupled', ...
    'fluid_model', 'water', ...
    'nonlinearTolerance', 1e-6, ...
    'verbose', true);
elapsed = toc(t0);

G = model.G;
ns = numel(states);
top = localTopNodes(G);

fprintf('\nSimulation finished in %.1f s (%d timesteps).\n', elapsed, ns);
fprintf('\n%-6s %10s %10s %8s %10s %11s\n', ...
        'step', 'p_mean', 'p_max', 'sat_max', 'bhp_inj', 'uplift');
fprintf('%s\n', repmat('-', 1, 64));

u0 = reshape(initState.u, 3, [])';
uz0 = u0(top, 3);

for i = 1:ns
    st = states{i};
    p = st.pressure;
    sat = st.s(:, 1);
    u = reshape(st.u, 3, [])';
    uplift = -(u(top, 3) - uz0);  % z points down in MRST grids
    bhp = [st.wellSol.bhp];
    fprintf('%-6d %9.1fb %9.1fb %8.3f %9.1fb %10.3fmm\n', ...
            i - 1, mean(p)/barsa, max(p)/barsa, max(sat), ...
            bhp(1)/barsa, max(uplift)*1000);
end

last = states{end};
u = reshape(last.u, 3, [])';
uplift = -(u(top, 3) - uz0);

fprintf('\n=== Results ==========================================================\n');
fprintf('Grid      : %d cells, %d faces, %d nodes\n', ...
        G.cells.num, G.faces.num, G.nodes.num);
fprintf('Pressure  : %.1f bar -> %.1f bar mean, max %.1f bar\n', ...
        mean(initState.pressure)/barsa, mean(last.pressure)/barsa, ...
        max(last.pressure)/barsa);
fprintf('Saturation: max %.3f\n', max(last.s(:, 1)));
fprintf('Surface   : uplift %+0.3f mm, subsidence %+0.3f mm over %d top nodes\n', ...
        max(uplift)*1000, min(uplift)*1000, numel(top));
fprintf('Initial u : max |u| %.3e m\n', max(abs(initState.u)));

fprintf('\nNote:\n');
fprintf(['  This uses MRST''s one-phase water variant because the oil-water\n' ...
         '  runNorneExample path has a separate PyMRST validation script.\n' ...
         '  This wrapper applies a temporary rhoSinj compatibility patch to\n' ...
         '  MRST''s runNorneExample without editing your MRST installation.\n']);

function localUsePatchedRunNorneExample()
    src = which('runNorneExample');
    if isempty(src)
        error('Could not find MRST runNorneExample.m on the MATLAB path.');
    end

    txt = fileread(src);
    needle = '    %% Setup wells';
    patch = sprintf(['    if ~isfield(model.fluid, ''rhoSinj'')\n' ...
                     '        rhoSinj = model.fluidModel.getSurfaceDensities();\n' ...
                     '        model.fluid.rhoSinj = rhoSinj;\n' ...
                     '        model.fluidModel.fluid.rhoSinj = rhoSinj;\n' ...
                     '    end\n\n' ...
                     '%s'], needle);
    if contains(txt, 'model.fluidModel.fluid.rhoSinj = rhoSinj')
        patched = txt;
    else
        patched = strrep(txt, needle, patch);
    end
    if strcmp(patched, txt) && ~contains(txt, 'model.fluidModel.fluid.rhoSinj = rhoSinj')
        error('Could not patch runNorneExample.m: insertion point not found.');
    end

    patchedDir = fullfile(tempdir, 'pymrst_mrst_norne_patch');
    if ~exist(patchedDir, 'dir')
        mkdir(patchedDir);
    end
    patchedFile = fullfile(patchedDir, 'runNorneExample.m');
    fid = fopen(patchedFile, 'w');
    if fid < 0
        error('Could not write patched runNorneExample.m to %s', patchedFile);
    end
    cleanup = onCleanup(@() fclose(fid));
    fwrite(fid, patched);
    clear cleanup;

    addpath(patchedDir, '-begin');
    clear runNorneExample;
end

function top = localTopNodes(G)
    nb = G.faces.neighbors;
    isbd = nb(:, 1) == 0 | nb(:, 2) == 0;
    sgn = double(nb(:, 1) == 0) - double(nb(:, 2) == 0);
    nIn = bsxfun(@times, G.faces.normals, sgn ./ G.faces.areas);
    topFaces = find(isbd & nIn(:, 3) > 0.5);
    ind = mcolon(G.faces.nodePos(topFaces), G.faces.nodePos(topFaces + 1) - 1);
    top = unique(G.faces.nodes(ind));
end
