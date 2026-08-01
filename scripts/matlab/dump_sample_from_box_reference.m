% Reference output of MRST's sampleFromBox, for validate_sample_from_box.py
out = struct();

G = computeGeometry(cartGrid([4,4],[4,4]));
p = reshape(1:16, 4, 4);
out.a.q = sampleFromBox(G, p);
out.a.p = p;

G2 = computeGeometry(cartGrid([6,3],[6,3]));
p2 = reshape(1:(11*7), 11, 7);
out.b.q = sampleFromBox(G2, p2);
out.b.p = p2;

G3 = computeGeometry(cartGrid([3,3,3],[3,3,3]));
p3 = reshape(1:27, 3, 3, 3);
out.c.q = sampleFromBox(G3, p3);
out.c.p = p3;

G4 = computeGeometry(cartGrid([5,4],[5,4]));
p4 = reshape(1:(3*9), 3, 9);
c  = [1 4 7 12];
out.d.q = sampleFromBox(G4, p4, c);
out.d.p = p4;
out.d.c = c;

save('/Users/Shared/PyMRST_workspace/PyMRST/scripts/matlab/ref_sample_from_box.mat', '-struct', 'out', '-v7');
fprintf('wrote ref_sample_from_box.mat\n');
