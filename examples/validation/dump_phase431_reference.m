%% Dump Phase 431 reference: readSummaryLocal, from MRST SPE9 output.

mrstModule add ad-core deckformat ad-blackoil

thisDir = fileparts(mfilename('fullpath'));
mrstRoot = fileparts(which('startup'));
helperDir = tempname;
mkdir(helperDir);
copyfile(fullfile(mrstRoot, 'modules', 'deckformat', 'resultinput', ...
                  'private', 'readFieldUnFmt.m'), helperDir);
srcSummary = fullfile(mrstRoot, 'modules', 'ad-core', 'utils', ...
                      'readSummaryLocal.m');
dstSummary = fullfile(helperDir, 'readSummaryLocal.m');
txt = fileread(srcSummary);
needle = 'fprintf([''Reading info from roughly '', num2str(estNum), '' ministeps...\n''])';
txt = strrep(txt, needle, [sprintf('fseek(fid, 4, ''cof'');\n'), needle]);
fid = fopen(dstSummary, 'w');
fwrite(fid, txt);
fclose(fid);
addpath(helperDir, '-begin');
prefix = fullfile(mrstRoot, 'modules', 'ad-blackoil', 'examples', ...
                  'spe9', 'compare', 'SPE9');

keywords = {'TIME', 'WBHP', 'WOPR'};
[smry, ~] = readSummaryLocal(prefix, keywords);

out_names = smry.WGNAMES;
out_keywords = smry.KEYWORDS;
out_ninx = smry.nInx;
out_kinx = smry.kInx;
out_data = smry.data;
out_wbhp_names = smry.getNms('WBHP');
sample_name = out_wbhp_names{1};
out_sample_name = sample_name;
out_sample_keywords = smry.getKws(sample_name);
out_sample_wbhp = smry.get(sample_name, 'WBHP', 1:size(smry.data, 2));

save(fullfile(thisDir, 'ref_p431_read_summary.mat'), '-v7', ...
     'out_names', 'out_keywords', 'out_ninx', 'out_kinx', 'out_data', ...
     'out_wbhp_names', 'out_sample_name', 'out_sample_keywords', ...
     'out_sample_wbhp');
fprintf('Wrote ref_p431_read_summary.mat to %s\n', thisDir);
