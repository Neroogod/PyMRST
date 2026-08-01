% Reference for validate_crossindconstruct.py
%
% The last case is the one MRST cannot do: a trailing group with nb2 = 0 makes
% its cumsum-difference construction produce mismatched arrays. Recorded as a
% failure rather than omitted.
out = struct();
cases = {{[2;1],[3;2]}, {[1],[1]}, {[3;2;1],[2;4;3]}, {[2;0;1],[3;5;2]}};
for k = 1:numel(cases)
   [c1, c2] = crossindconstruct(cases{k}{1}, cases{k}{2});
   out.(sprintf('c%d', k)) = struct('a', c1(:), 'b', c2(:));
end
try
   [c1, c2] = crossindconstruct([2;3], [4;0]);
   out.trailing_zero_ran = 1;
catch err
   out.trailing_zero_ran = 0;
   out.trailing_zero_msg = err.message;
end
save('/Users/Shared/PyMRST_workspace/PyMRST/scripts/matlab/ref_crossind.mat', '-struct', 'out', '-v7');
fprintf('wrote ref_crossind.mat\n');
