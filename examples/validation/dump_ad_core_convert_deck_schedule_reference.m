%% Processed schedule reference for convertDeckScheduleToMRST.

mrstModule add ad-core ad-props ad-blackoil deckformat
thisDir = fileparts(mfilename('fullpath'));
deckPath = fullfile(thisDir, '..', '..', '..', 'mrst-2024a-Geomech', ...
    'examples', 'data', 'SPE1', 'BENCH_SPE1.DATA');

[~, model, schedule] = initEclipseProblemAD(deckPath);
model_class = class(model);
save(fullfile(thisDir, 'ref_ad_core_convert_deck_schedule.mat'), '-v7', ...
    'schedule', 'model_class');
fprintf('Wrote ref_ad_core_convert_deck_schedule.mat (%d controls, %d steps)\n', ...
    numel(schedule.control), numel(schedule.step.val));
