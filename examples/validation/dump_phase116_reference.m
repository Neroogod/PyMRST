function dump_phase116_reference()
% Phase 116 reference: EOSSeparator and SeparatorGroup surface streams.

pymrstRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
mrstRoot = fullfile(fileparts(pymrstRoot), 'mrst-2024a-Geomech');
run(fullfile(mrstRoot, 'startup.m'));
mrstModule add ad-core compositional deckformat ad-props

G    = cartGrid([1, 1, 1], [1, 1, 1]);
G    = computeGeometry(G);
rock = makeRock(G, 0.1*darcy, 0.3);
[mix, info] = getBenchmarkMixture('spe5');
fluid = initSimpleADIFluid('blackoil', false, 'phases', 'OG', ...
                           'n', [2, 2], 'rho', [100, 1]);
model = GenericOverallCompositionModel(G, rock, fluid, mix, 'water', false);

stream_mole = localVectorToCell(info.initial);
s = EOSSeparator('pressure', 1*atm, 'T', 300);

sg_single = SeparatorGroup(s);
sg_single.mode = 'moles';
[single_q, single_stream, single_frac, single_density] = ...
    sg_single.computeSurfaceRates(model, stream_mole);

p = [200, 175, 50, 10]*barsa;
T = [423, 400, 350, 300];
dest = [2, 3; ...
        0, 4; ...
        4, 0; ...
        0, 0];
sg_tree = SeparatorGroup(s, p, T, dest);
sg_tree.mode = 'moles';
[tree_q, tree_stream, tree_frac, tree_density] = ...
    sg_tree.computeSurfaceRates(model, stream_mole);
tree_surface_streams = sg_tree.computeSurfaceMoleRates(model, stream_mole);

single_q = localCellScalars(single_q);
single_stream = localCellScalars(single_stream);
single_frac = localPhaseFractions(single_frac);
single_density = localCellScalars(single_density);

tree_q = localCellScalars(tree_q);
tree_stream = localCellScalars(tree_stream);
tree_frac = localPhaseFractions(tree_frac);
tree_density = localCellScalars(tree_density);
tree_surface_streams = localPhaseFractions(tree_surface_streams);
tree_phase_destination = sg_tree.phaseDestination;

out = fullfile(fileparts(mfilename('fullpath')), 'ref_p116_compositional_separators.mat');
save(out, 'stream_mole', 'single_q', 'single_stream', 'single_frac', ...
    'single_density', 'tree_q', 'tree_stream', 'tree_frac', ...
    'tree_density', 'tree_surface_streams', 'tree_phase_destination');
fprintf('Wrote ref_p116_compositional_separators.mat to %s\n', fileparts(out));
end

function c = localVectorToCell(v)
    c = cell(1, numel(v));
    for i = 1:numel(v)
        c{i} = v(i);
    end
end

function x = localCellScalars(c)
    x = zeros(1, numel(c));
    for i = 1:numel(c)
        x(i) = c{i};
    end
end

function x = localPhaseFractions(c)
    nph = numel(c);
    ncomp = numel(c{1});
    x = zeros(nph, ncomp);
    for ph = 1:nph
        for comp = 1:ncomp
            x(ph, comp) = c{ph}{comp};
        end
    end
end
