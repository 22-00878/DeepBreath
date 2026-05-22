%% =========================================================
%  TOP 50 FEATURES VIEWER + EXPORTER  (FIXED)
%  Saves: Top50_Features.csv  &  Top50_Features.mat
%% =========================================================
clc;

% ── Browse for the .mat file ─────────────────────────────
[fname, fpath] = uigetfile('*.mat', 'Select Base_ICBHI_PatientBased.mat');
if isequal(fname, 0), error('No file selected. Cancelled.'); end
full_path = fullfile(fpath, fname);
fprintf('Loading: %s\n', full_path);
load(full_path, 'sel_b', 'sel_a');

% ── Browse for OUTPUT folder ──────────────────────────────
OUT_DIR = uigetdir('', 'Select Output Folder to Save CSV and MAT');
if isequal(OUT_DIR, 0), error('No output folder selected. Cancelled.'); end

% ── Feature name map — built as CELL ARRAY first ─────────
feat_cell = [ ...
    {'ZCR'}, {'RMS'}, {'SpectralCentroid'}, {'SpectralSpread'}, ...
    {'SpectralSkewness'}, {'SpectralKurtosis'}, {'SpectralRolloff'}, ...
    arrayfun(@(i) sprintf('MFCC_%d',    i), 1:13, 'UniformOutput', false), ...
    arrayfun(@(i) sprintf('dMFCC_%d',   i), 1:13, 'UniformOutput', false), ...
    arrayfun(@(i) sprintf('ddMFCC_%d',  i), 1:13, 'UniformOutput', false), ...
    arrayfun(@(i) sprintf('BarkBand_%d',i), 1:15, 'UniformOutput', false) ...
];
% feat_cell is now 1×61 cell array of char vectors
N_NAMED = numel(feat_cell);   % = 61

% ── Build index & name arrays ─────────────────────────────
n_b  = min(50, numel(sel_b));
n_a  = min(50, numel(sel_a));

% FIXED: Broke the chained indexing into two steps
temp_b = double(sel_b(1:n_b));
idx_b  = temp_b(:);

temp_a = double(sel_a(1:n_a));
idx_a  = temp_a(:);

names_b = cell(n_b, 1);
for i = 1:n_b
    names_b{i} = get_name(idx_b(i), feat_cell, N_NAMED);
end

names_a = cell(n_a, 1);
for i = 1:n_a
    names_a{i} = get_name(idx_a(i), feat_cell, N_NAMED);
end

rank_b = (1:n_b)';
rank_a = (1:n_a)';

% ── Print Binary ──────────────────────────────────────────
fprintf('\n╔══════════════════════════════════════════════════╗\n');
fprintf('║  TOP 50 FEATURES — BINARY (Normal vs Abnormal)  ║\n');
fprintf('╚══════════════════════════════════════════════════╝\n');
fprintf('  %-4s  %-12s  %s\n', 'Rank', 'Feature Idx', 'Feature Name');
fprintf('  %s\n', repmat('─', 1, 40));
for i = 1:n_b
    fprintf('  %3d.  [%3d]         %s\n', rank_b(i), idx_b(i), names_b{i});
end

% ── Print Abnormal ────────────────────────────────────────
fprintf('\n╔══════════════════════════════════════════════════╗\n');
fprintf('║  TOP 50 FEATURES — ABNORMAL (Crackle/Wheeze)   ║\n');
fprintf('╚══════════════════════════════════════════════════╝\n');
fprintf('  %-4s  %-12s  %s\n', 'Rank', 'Feature Idx', 'Feature Name');
fprintf('  %s\n', repmat('─', 1, 40));
for i = 1:n_a
    fprintf('  %3d.  [%3d]         %s\n', rank_a(i), idx_a(i), names_a{i});
end

% ── SAVE CSV ──────────────────────────────────────────────
csv_path = fullfile(OUT_DIR, 'Top50_Features.csv');
fid = fopen(csv_path, 'w');
fprintf(fid, 'Stage,Rank,Feature_Index,Feature_Name\n');
for i = 1:n_b
    fprintf(fid, 'Binary,%d,%d,%s\n',   rank_b(i), idx_b(i), names_b{i});
end
for i = 1:n_a
    fprintf(fid, 'Abnormal,%d,%d,%s\n', rank_a(i), idx_a(i), names_a{i});
end
fclose(fid);
fprintf('\n[SAVED] CSV → %s\n', csv_path);

% ── SAVE MAT ──────────────────────────────────────────────
Top50_Binary   = table(rank_b, idx_b, names_b,  ...
    'VariableNames', {'Rank','Feature_Index','Feature_Name'});
Top50_Abnormal = table(rank_a, idx_a, names_a,  ...
    'VariableNames', {'Rank','Feature_Index','Feature_Name'});

sel_b_top50 = idx_b;
sel_a_top50 = idx_a;

mat_path = fullfile(OUT_DIR, 'Top50_Features.mat');
save(mat_path, 'Top50_Binary', 'Top50_Abnormal', ...
               'sel_b_top50', 'sel_a_top50', 'feat_cell', '-v7.3');

fprintf('[SAVED] MAT → %s\n', mat_path);
fprintf('\nDone.\n');


% ── Local Functions (MUST be at the bottom of the script) ─
function name = get_name(idx, feat_cell, N_NAMED)
    if idx >= 1 && idx <= N_NAMED
        name = feat_cell{idx};   % cell indexing with {} — always correct
    else
        name = sprintf('Padding_%d', idx);
    end
end