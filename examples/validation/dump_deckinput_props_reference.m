%% Reference PROPS sections, read and unit-converted by MRST itself
%
% The Python deck reader is checked against PyMRST's bespoke SPE readers in
% tests/deckformat/. Those readers are trusted, but they are still PyMRST
% code - agreeing with them proves consistency, not correctness. This dumps
% what MRST's own readEclipseDeck + convertDeckUnits produce, so the port can
% be compared against the reference implementation rather than against a
% second port of it.

thisDir = fileparts(mfilename('fullpath'));
mrstModule add ad-core ad-props deckformat

decks = {'SPE1', 'SPE3', 'SPE9'};
out = struct();

for k = 1 : numel(decks)
   name = decks{k};
   fn   = fullfile(ROOTDIR, 'examples', 'data', name, ['BENCH_', name, '.DATA']);

   deck = convertDeckUnits(readEclipseDeck(fn));
   props = deck.PROPS;

   entry = struct();
   entry.keywords = sort(reshape(fieldnames(props), 1, []));

   % Tables arrive as cell arrays (one per region) and PVTO/PVTG as structs
   % with key/pos/data. Flatten each into plain arrays so the .mat file can
   % be read without reproducing MRST's containers on the Python side.
   for f = reshape(fieldnames(props), 1, [])
      key = f{1};
      value = props.(key);

      if iscell(value)
         if isstruct(value{1})
            % Miscible PVT: keep key, pos and data separately.
            entry.([key, '_key'])  = reshape(value{1}.key,  [], 1);
            entry.([key, '_pos'])  = reshape(value{1}.pos,  [], 1);
            entry.([key, '_data']) = value{1}.data;
         else
            entry.(key) = value{1};
            entry.([key, '_ntab']) = numel(value);
         end
      elseif isnumeric(value)
         entry.(key) = value;
      end
   end

   % TABDIMS drives how many tables each keyword owns.
   if isfield(deck.RUNSPEC, 'TABDIMS')
      entry.TABDIMS = reshape(deck.RUNSPEC.TABDIMS, 1, []);
   end
   entry.cartDims = reshape(deck.RUNSPEC.cartDims, 1, []);

   if isfield(deck, 'REGIONS')
      for f = reshape(fieldnames(deck.REGIONS), 1, [])
         entry.(['REGIONS_', f{1}]) = reshape(double(deck.REGIONS.(f{1})), [], 1);
      end
   end

   out.(name) = entry;
end

save(fullfile(thisDir, 'ref_deckinput_props.mat'), '-struct', 'out', '-v7');
fprintf('Wrote ref_deckinput_props.mat for %d decks\n', numel(decks));
