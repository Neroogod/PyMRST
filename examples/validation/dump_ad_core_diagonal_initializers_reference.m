%% Direct MRST reference for diagonal AD initializer Jacobian layouts.

mrstModule add ad-core
thisDir = fileparts(mfilename('fullpath'));

a = [1; 2];
b = [3; 4];
c = 5;
[xd1, xd2, xd3] = initVariablesAD_diagonal(a, b, c);
[xr1, xr2, xr3] = initVariablesAD_diagonalRowMajor(a, b, c);
[xo1, xo2, xo3] = initVariablesAD_oneBlock(a, c, b);

opts = struct('types', [1; 2; 1], 'useMex', false);
[xt1, xt2, xt3] = initVariablesAD_diagonal(a, c, b, opts);

jd11 = xd1.jac{1}.sparse(); jd12 = xd1.jac{2}.sparse();
jd21 = xd2.jac{1}.sparse(); jd22 = xd2.jac{2}.sparse();
jd31 = xd3.jac{1}.sparse(); jd32 = xd3.jac{2}.sparse();
jr11 = xr1.jac{1}.sparse(); jr12 = xr1.jac{2}.sparse();
jr21 = xr2.jac{1}.sparse(); jr22 = xr2.jac{2}.sparse();
jr31 = xr3.jac{1}.sparse(); jr32 = xr3.jac{2}.sparse();
jo1 = xo1.jac{1}; jo2 = xo2.jac{1}; jo3 = xo3.jac{1};
jt11 = xt1.jac{1}.sparse(); jt12 = xt1.jac{2}.sparse();
jt21 = xt2.jac{1}.sparse(); jt22 = xt2.jac{2}.sparse();
jt31 = xt3.jac{1}.sparse(); jt32 = xt3.jac{2}.sparse();

save(fullfile(thisDir, 'ref_ad_core_diagonal_initializers.mat'), '-v7', ...
    'jd11', 'jd12', 'jd21', 'jd22', 'jd31', 'jd32', ...
    'jr11', 'jr12', 'jr21', 'jr22', 'jr31', 'jr32', ...
    'jo1', 'jo2', 'jo3', ...
    'jt11', 'jt12', 'jt21', 'jt22', 'jt31', 'jt32');
fprintf('Wrote ref_ad_core_diagonal_initializers.mat\n');
