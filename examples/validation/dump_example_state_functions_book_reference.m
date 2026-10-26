% Reference for exampleStateFunctionsBook
% (validate_ad_core_example_state_functions_book.py): the script's console
% output, recorded with diary, without its figure (plotStateFunctionGroupings)
% and TikZ output (printStateFunctionGroupingTikz); clc does not clear a diary.
here = fileparts(mfilename('fullpath'));
run(fullfile(here, '..', '..', '..', 'mrst-2024a-Geomech', 'startup.m'));
mrstModule add ad-core
addpath(fullfile(ROOTDIR, 'modules', 'ad-core', 'examples', 'book-ii', 'utils'));
out = fullfile(here, 'ref_example_state_functions_book.txt');
if exist(out, 'file'), delete(out); end
diary(out);
group = StateFunctionGrouping('mystorage'); % Store in state.mystorage
group = group.setStateFunction('A', TutorialNumber('a'));
group = group.setStateFunction('B', TutorialNumber('b'));
group = group.setStateFunction('X', TutorialNumber('x'));
group = group.setStateFunction('Y', TutorialNumber('y'));
group = group.setStateFunction('AB', TutorialProduct('A', 'B'));
group = group.setStateFunction('XY', TutorialProduct('X', 'Y'));
group = group.setStateFunction('G', TutorialAdd('AB', 'XY'));
disp(group)
state0 = struct('a', 3, 'b', 5, 'x', 7, 'y', 2); % Initial state
state = group.initStateFunctionContainer(state0); % Enable caching
disp(state.mystorage)
disp('First call:')
G = group.get([], state, 'G'); % First call, triggers execution
disp(state.mystorage)
disp('Second call:')
G = group.get([], state, 'G') % Second call, cached
state = group.initStateFunctionContainer(state0);
xy = group.get([], state, 'XY'); % First call, triggers execution
G = group.get([], state, 'G'); % Second call, cached
mrstVerbose on; % Print missing dependencies
group.checkDependencies();
group = group.setStateFunction('F', TutorialProduct('X', 'Z'));
ok = group.checkDependencies()
diary off
