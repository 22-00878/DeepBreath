% =========================================================================
%  ICBHI PREPROCESSING PIPELINE
%  Steps: Resample → DC Remove → Bandpass → Kalman → Normalize → Segment
%  Processes ALL .wav + .txt pairs found in separate audio / annotation folders
% =========================================================================
clc; clear; close all;

%% ── CONFIG ────────────────────────────────────────────────────────────────
AUDIO_FOLDER      = '/MATLAB Drive/dp/ICBHI Dataset/audio';        % ← FIXED
ANNOTATION_FOLDER = '/MATLAB Drive/dp/ICBHI Dataset/annotation';   % ← FIXED
OUTPUT_FOLDER     = '/MATLAB Drive/dp/ICBHI DATA/PIPELINE_OUTPUT';  % ← FIXED: forward slashes

TARGET_FS = 4000;
BP_LOW    = 100;
BP_HIGH   = 1800;
BP_ORDER  = 4;
PEAK_NORM = 0.90;
KALMAN_Q  = 1e-5;

%% ── SETUP ─────────────────────────────────────────────────────────────────
fprintf('\n=================================================================\n');
fprintf('  ICBHI PREPROCESSING PIPELINE\n');
fprintf('=================================================================\n\n');

% Normalize separators (safety for MATLAB Online / Linux)
AUDIO_FOLDER      = strrep(AUDIO_FOLDER,      '\', '/');
ANNOTATION_FOLDER = strrep(ANNOTATION_FOLDER, '\', '/');
OUTPUT_FOLDER     = strrep(OUTPUT_FOLDER,      '\', '/');

seg_out = fullfile(OUTPUT_FOLDER, 'segments');
for d = {OUTPUT_FOLDER, seg_out}
    if ~exist(d{1},'dir'), mkdir(d{1}); end
end

% Design bandpass filter once
nyq = TARGET_FS / 2;
[bp_b, bp_a] = butter(BP_ORDER, [BP_LOW/nyq, BP_HIGH/nyq], 'bandpass');

%% ── DISCOVER ALL WAV FILES ────────────────────────────────────────────────
wav_list = dir(fullfile(AUDIO_FOLDER, '*.wav'));
fprintf('[DISCOVERY] Audio folder   : %s\n', AUDIO_FOLDER);
fprintf('[DISCOVERY] Annotation folder: %s\n', ANNOTATION_FOLDER);
fprintf('[DISCOVERY] Found %d WAV file(s)\n\n', numel(wav_list));

if isempty(wav_list)
    fprintf('[ERROR] No .wav files found in audio folder.\n');
    return;
end

label_map    = {'Normal','Crackles','Wheeze','Both'};
flag_map     = [0 0; 1 0; 0 1; 1 1];

global_rows  = {};
total_segs   = 0;
label_counts = containers.Map({'Normal','Crackles','Wheeze','Both'},{0,0,0,0});

%% =========================================================================
%  MAIN LOOP — EACH RECORDING
%% =========================================================================
for fi = 1:numel(wav_list)

    wav_name = wav_list(fi).name;
    [~, base] = fileparts(wav_name);
    wav_path  = fullfile(AUDIO_FOLDER,      wav_name);       % ← audio subfolder
    txt_path  = fullfile(ANNOTATION_FOLDER, [base '.txt']);  % ← annotation subfolder

    if ~exist(txt_path, 'file')
        fprintf('[SKIP] No annotation for %s\n', wav_name);
        continue;
    end

    fprintf('── %s\n', wav_name);

    % ── 1. Load ────────────────────────────────────────────────────────────
    [x, fs] = audioread(wav_path);
    if size(x,2) > 1, x = mean(x, 2); end
    x = double(x(:));

    % ── 2. Resample ────────────────────────────────────────────────────────
    if fs ~= TARGET_FS
        x = resample(x, TARGET_FS, fs);
    end

    % ── 3. DC Offset Removal ───────────────────────────────────────────────
    x = x - mean(x);

    % ── 4. Bandpass Filter ─────────────────────────────────────────────────
    x = filtfilt(bp_b, bp_a, x);

    % ── 5. Kalman Filter ───────────────────────────────────────────────────
    n_noise = round(0.5 * TARGET_FS);
    R_est   = var(x(1:min(n_noise, numel(x))));
    if R_est < 1e-10, R_est = 0.01; end
    x = kalman_1d(x, KALMAN_Q, R_est);

    total_dur = numel(x) / TARGET_FS;

    % ── 6. Parse Annotation ────────────────────────────────────────────────
    try
        ann = readmatrix(txt_path);
    catch
        fid2 = fopen(txt_path, 'r');
        raw  = textscan(fid2, '%f %f %f %f');
        fclose(fid2);
        ann  = [raw{1} raw{2} raw{3} raw{4}];
    end

    if isempty(ann) || size(ann,2) < 4
        fprintf('  [SKIP] Could not parse annotation: %s\n', txt_path);
        continue;
    end

    rec_segs = 0;

    for r = 1:size(ann,1)
        t1 = ann(r,1);  t2 = ann(r,2);
        cr = ann(r,3);  wh = ann(r,4);

        % Determine label
        flags = [cr wh];
        idx   = find(all(flag_map == flags, 2));
        if isempty(idx), label = 'Unknown'; else, label = label_map{idx}; end

        % Clamp to audio length
        if t1 >= total_dur, continue; end
        t2 = min(t2, total_dur);

        s1    = max(1, round(t1 * TARGET_FS));
        s2    = min(numel(x), round(t2 * TARGET_FS));
        x_seg = x(s1:s2);

        if numel(x_seg) < round(0.5 * TARGET_FS), continue; end

        % ── 7. Amplitude Normalization ─────────────────────────────────────
        pk = max(abs(x_seg));
        if pk > 0, x_seg = x_seg * (PEAK_NORM / pk); end

        % Save segment
        out_name = sprintf('%s_seg%03d_%s.wav', base, r, label);
        audiowrite(fullfile(seg_out, out_name), x_seg, TARGET_FS, 'BitsPerSample', 16);

        rec_segs   = rec_segs + 1;
        total_segs = total_segs + 1;

        if isKey(label_counts, label)
            label_counts(label) = label_counts(label) + 1;
        end

        global_rows(end+1,:) = {base, r, t1, t2, t2-t1, label, out_name}; %#ok<AGROW>
    end

    fprintf('  %d segment(s) saved\n', rec_segs);
end

%% ── CSV REPORT ────────────────────────────────────────────────────────────
if ~isempty(global_rows)
    csv_path = fullfile(OUTPUT_FOLDER, 'icbhi_pipeline_report.csv');
    fid = fopen(csv_path, 'w');
    fprintf(fid, 'recording,ann_row,t_start,t_end,duration_sec,label,filename\n');
    for i = 1:size(global_rows,1)
        fprintf(fid, '%s,%d,%.3f,%.3f,%.3f,%s,%s\n', ...
            global_rows{i,1}, global_rows{i,2}, global_rows{i,3}, ...
            global_rows{i,4}, global_rows{i,5}, global_rows{i,6}, global_rows{i,7});
    end
    fclose(fid);
    fprintf('\n[REPORT] Saved: %s\n', csv_path);
end

fprintf('\n=================================================================\n');
fprintf('  ICBHI PIPELINE COMPLETE\n');
fprintf('  Total segments : %d\n', total_segs);
fprintf('  Normal         : %d\n', label_counts('Normal'));
fprintf('  Crackles       : %d\n', label_counts('Crackles'));
fprintf('  Wheeze         : %d\n', label_counts('Wheeze'));
fprintf('  Both           : %d\n', label_counts('Both'));
fprintf('  Output         : %s\n', OUTPUT_FOLDER);
fprintf('=================================================================\n\n');

%% =========================================================================
%  LOCAL FUNCTIONS
%% =========================================================================

function x_est = kalman_1d(y, Q, R)
    N = numel(y);  x_est = zeros(size(y));
    P = 1;  x_hat = y(1);
    for t = 1:N
        P_pred   = P + Q;
        K        = P_pred / (P_pred + R);
        x_hat    = x_hat + K * (y(t) - x_hat);
        P        = (1 - K) * P_pred;
        x_est(t) = x_hat;
    end
end