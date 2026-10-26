% The schedule utilities in MRST 2024a, for validate_ad_core_schedules.py:
% simpleSchedule, combineSchedules, compressSchedule, filterSchedule and
% refineSchedule. Controls carry a well named after them so their order can
% be checked.
here = fileparts(mfilename('fullpath'));
run(fullfile(here, '..', '..', '..', 'mrst-2024a-Geomech', 'startup.m'));
mrstModule add ad-core
tag = @(k) struct('W', struct('name', sprintf('c%d', k)), 'bc', [], 'src', []);
s1 = simpleSchedule([1, 2, 3]*day, 'W', struct('name', 'c1'));
s2 = simpleSchedule([4; 5]*day, 'W', struct('name', 'c2'));
s2.control = [tag(2); tag(3)]; s2.step.control = [2; 1];
s3 = simpleSchedule(6*day, 'W', struct('name', 'c4'));
comb = combineSchedules(s1, s2, s3);
pack = @(s) struct('val', s.step.val, 'control', s.step.control, ...
                   'names', {arrayfun(@(c) c.W.name, s.control, 'UniformOutput', false)});
c = pack(comb);
s4 = comb; s4.step.control = [1; 1; 3; 3; 1; 2]; s4.step.val = (1:6)'*day;
cmp = pack(compressSchedule(s4));
s5 = comb; s5.step.control = [4; 2; 4; 4; 2; 4];
flt = pack(filterSchedule(s5));
base = simpleSchedule([10, 10, 10]*day, 'W', struct('name', 'c1'));
base.control = [tag(1); tag(2)]; base.step.control = [1; 2; 2];
r1 = refineSchedule(0, [5; 5; 3; 7.0000001; 4]*day, base);
r2 = refineSchedule(2*day, [3; 4.5]*day, base);
ref = @(r) struct('val', r.step.val, 'control', r.step.control, 'repStep', r.step.repStep);
ref1 = ref(r1); ref2 = ref(r2);
simple = struct('val', s1.step.val, 'control', s1.step.control);
save(fullfile(here, 'ref_ad_core_schedules.mat'), '-v7', 'simple', 'c', 'cmp', 'flt', 'ref1', 'ref2');
fprintf('saved: refined %d and %d steps\n', numel(ref1.val), numel(ref2.val));
