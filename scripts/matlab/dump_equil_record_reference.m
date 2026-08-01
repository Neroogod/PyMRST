% Reference for validate_read_equil.py  (readEQUIL is a private/ helper, so it
% is exercised through a temporary copy on the path.)
src = fullfile(ROOTDIR, 'utils', 'inout', 'eclipse', 'deckinput_simple', 'private');
tmp = tempname; mkdir(tmp); copyfile(fullfile(src, 'readEQUIL.m'), tmp);
copyfile(fullfile(src, '..', 'readDefaultedKW.m'), tmp);
addpath(tmp);

out = struct();
cases = {'2000 300 2200 0 1800 0 1 1 0 1 0 /', ...
         '2000 300 9* /', ...
         '1234.5 1 9* /'};
names = {'full','defaulted','identity'};
for k = 1:numel(cases)
   for m = [true false]
      f = [tempname '.txt']; fid = fopen(f, 'wt');
      fprintf(fid, '%s\n', cases{k}); fclose(fid);
      fid = fopen(f, 'rt');
      e = readEQUIL(fid, m, 1);   % (fid, metric, ntequil)
      fclose(fid);
      out.([names{k} '_' char(string(m))]) = e;
   end
end
% Two regions.
f = [tempname '.txt']; fid = fopen(f, 'wt');
fprintf(fid, '2000 300 9* /\n2500 350 9* /\n'); fclose(fid);
fid = fopen(f, 'rt'); out.two = readEQUIL(fid, true, 2); fclose(fid);

save('/Users/Shared/PyMRST_workspace/PyMRST/scripts/matlab/ref_equil_record.mat', '-struct', 'out', '-v7');
fprintf('wrote ref_equil_record.mat\n');
