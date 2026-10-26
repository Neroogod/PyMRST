% Reference for getStateFunctionGroupingDependencyGraph and buildStateFunctionDigraph
% (validate_ad_core_state_function_graph.py): a three-phase GenericBlackOilModel
% on a 3x3 grid. The dependency graph of its groupings and the digraph kept by
% the Start, Stop and Center filters, with names qualified by grouping as
% plotStateFunctionGroupings passes them. (setupStateFunctionGraph needs
% matlab_bgl's MEX, which does not run on this machine.)
here = fileparts(mfilename('fullpath'));
run(fullfile(here, '..', '..', '..', 'mrst-2024a-Geomech', 'startup.m'));
mrstModule add ad-core ad-blackoil ad-props
G = computeGeometry(cartGrid([3, 3]));
fluid = initSimpleADIFluid('phases', 'WOG', 'n', [2, 2, 2], 'rho', [1000, 800, 100], 'mu', [1, 5, 0.1]);
model = GenericBlackOilModel(G, makeRock(G, 1, 1), fluid);
model = model.validateModel();
[groups, gnames, ~] = model.getStateFunctionGroupings();
graph = getStateFunctionGroupingDependencyGraph(groups{:});
out = struct('C', graph.C, 'FunctionNames', {graph.FunctionNames}, 'Labels', {graph.Labels}, ...
             'Implementation', {graph.Implementation}, 'GroupIndex', graph.GroupIndex, ...
             'GroupTypes', graph.GroupTypes, 'GroupNames', {graph.GroupNames}, ...
             'gnames', {gnames});
full_names = cellfun(@(x, y) sprintf('%s.%s', x, y), graph.GroupNames(graph.GroupIndex), ...
                     graph.FunctionNames, 'UniformOutput', false);
filters = {{'Start', 'PhasePressures'}, {'Stop', 'FaceMobility'}, {'Center', 'Mobility'}, ...
           {'Start', 'PVTPropertyFunctions.Density'}};
for i = 1:numel(filters)
    [D, nm, cat, keep] = buildStateFunctionDigraph(graph.C, full_names, graph.GroupIndex, ...
                                                   filters{i}{:}, 'FilterNames', graph.FunctionNames);
    out.(sprintf('keep%d', i)) = keep;
    out.(sprintf('edges%d', i)) = [findnode(D, D.Edges.EndNodes(:, 1)), findnode(D, D.Edges.EndNodes(:, 2))];
    out.(sprintf('names%d', i)) = nm;
end
save(fullfile(here, 'ref_state_function_graph.mat'), '-struct', 'out', '-v7');
