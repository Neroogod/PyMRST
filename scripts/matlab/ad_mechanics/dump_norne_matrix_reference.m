function output_file = dump_norne_matrix_reference(output_dir, fluid_model, method)
%DUMP_NORNE_MATRIX_REFERENCE Export one compact MRST mini-Norne history.
%
% One case per invocation keeps long black-oil runs restartable:
%
%   dump_norne_matrix_reference('results/norne_matrix', ...
%       'oil water', 'fixed stress splitting')
%
% The output contains engineering histories rather than full field tensors.
% Use examples/validation/dump_norne_reference.m for the existing deep,
% cellwise oil-water fully-coupled fixture.

if nargin < 1 || isempty(output_dir)
    here = fileparts(mfilename('fullpath'));
    output_dir = fullfile(here, '..', '..', '..', 'examples', ...
                          'validation', 'ad_mechanics', 'norne_matrix');
end
if nargin < 2 || isempty(fluid_model), fluid_model = 'water'; end
if nargin < 3 || isempty(method), method = 'fully coupled'; end
if ~exist(output_dir, 'dir'), mkdir(output_dir); end

mrstModule add ad-mechanics ad-core ad-props ad-blackoil vemmech deckformat
gravity on

[model, states, init_state] = runNorneExample( ...
    'norne_case', 'mini Norne', 'bc_case', 'bottom fixed', ...
    'fluid_model', fluid_model, 'method', method, ...
    'splittingTolerance', 1e-3, 'verbose', false, ...
    'splittingVerbose', false);

ns = numel(states);
p_mean = zeros(ns, 1); p_max = zeros(ns, 1);
sw_mean = zeros(ns, 1); sw_max = zeros(ns, 1);
sg_mean = zeros(ns, 1); sg_max = zeros(ns, 1);
uplift_max = zeros(ns, 1); displacement_inf = zeros(ns, 1);
bhp = nan(ns, 2); qws = zeros(ns, 2); qos = zeros(ns, 2); qgs = zeros(ns, 2);
u0 = init_state.u(:);

for k = 1:ns
    state = states{k};
    p_mean(k) = mean(state.pressure);
    p_max(k) = max(state.pressure);
    sat = state.s;
    sw_mean(k) = mean(sat(:, 1)); sw_max(k) = max(sat(:, 1));
    if size(sat, 2) >= 3
        sg_mean(k) = mean(sat(:, 3)); sg_max(k) = max(sat(:, 3));
    end
    du = state.u(:) - u0;
    uu = reshape(du, 3, [])';
    uplift_max(k) = max(-uu(:, 3));
    displacement_inf(k) = max(abs(du));
    nw = min(numel(state.wellSol), 2);
    for w = 1:nw
        ws = state.wellSol(w);
        bhp(k, w) = ws.bhp;
        if isfield(ws, 'qWs') && ~isempty(ws.qWs), qws(k, w) = ws.qWs; end
        if isfield(ws, 'qOs') && ~isempty(ws.qOs), qos(k, w) = ws.qOs; end
        if isfield(ws, 'qGs') && ~isempty(ws.qGs), qgs(k, w) = ws.qGs; end
    end
end

dt = [1*day; 5*day*ones(20, 1)];
mrst_version = 'mrst-2024a-Geomech';
matlab_version = version();
slug = strrep(strrep([fluid_model, '__', method], ' ', '_'), '-', '_');
output_file = fullfile(output_dir, ['mrst_', slug, '.mat']);
save(output_file, '-v7', 'fluid_model', 'method', 'dt', ...
     'p_mean', 'p_max', 'sw_mean', 'sw_max', 'sg_mean', 'sg_max', ...
     'uplift_max', 'displacement_inf', 'bhp', 'qws', 'qos', 'qgs', ...
     'mrst_version', 'matlab_version');
fprintf('Wrote %s (%s / %s, %d report steps)\n', ...
        output_file, fluid_model, method, ns);
end
