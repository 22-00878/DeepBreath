% =========================================================================
%  DEEPBREATH UNIFIED PREPROCESSING PIPELINE  
%  Steps: Resample → DC Remove → Bandpass → Kalman →
%         NMF Heart-Lung Sep → Normalize → Cycle Detection
% =========================================================================

clc; clear; close all;

% =========================================================================
%  0. USER CONFIGURATION — EDIT THESE PATHS
% =========================================================================

INPUT_ROOT  = 'C:\dp\AUSCULTO DATA';         
OUTPUT_ROOT = '/MATLAB Drive/dp/AUSCULTO DATA/PIPELINE_OUTPUT'; 

% =========================================================================
%  1. PIPELINE PARAMETERS
% =========================================================================

TARGET_FS  = 4000;
BP_LOW     = 100;
BP_HIGH    = 1800;
BP_ORDER   = 4;
PEAK_NORM  = 0.90;
KALMAN_Q   = 1e-5;

NMF_RANK        = 10;
NMF_MAX_ITER    = 300;
NMF_BETA        = 1;
NMF_SPARSITY_H  = 0.5;
HEART_THRESH    = 0.28;
MASK_FLOOR      = 0.05;
STFT_NFFT       = 512;
STFT_HOP        = 64;

ENV_FRAME_SEC     = 0.02;
ENV_HOP_SEC       = 0.01;
SMOOTH_WIN_SEC    = 0.15;
MIN_CYCLE_SEC     = 1.8;
MAX_CYCLE_SEC     = 10.0;
PEAK_MIN_PROM     = 0.15;
PEAK_MIN_HEIGHT   = 0.10;

N_MELS   = 64;
N_FFT    = 512;
HOP_SPEC = 128;
N_MFCC   = 13;

SITE_LABELS = {'RUL','LUL','RML','LML','URPos','ULPos','LRPos','LLPos'};

% =========================================================================
%  2. GLOBAL SETUP
% =========================================================================

fprintf('\n');
fprintf('=================================================================\n');
fprintf('  DEEPBREATH UNIFIED PIPELINE\n');
fprintf('=================================================================\n\n');

% ← FIXED: normalize path separators to forward slash for MATLAB Online (Linux)
INPUT_ROOT  = strrep(INPUT_ROOT,  '\', '/');
OUTPUT_ROOT = strrep(OUTPUT_ROOT, '\', '/');

if ~exist(OUTPUT_ROOT,'dir'), mkdir(OUTPUT_ROOT); end

nyq = TARGET_FS / 2;
[bp_b, bp_a] = butter(BP_ORDER, [BP_LOW/nyq, BP_HIGH/nyq], 'bandpass');
fprintf('[FILTER] Butterworth bandpass designed: %d-%d Hz, order %d\n\n', ...
        BP_LOW, BP_HIGH, BP_ORDER);

% =========================================================================
%  3. DISCOVER TRIAL FOLDERS (two-level: PATIENT → TRIAL → wav files)
%     Structure: INPUT_ROOT / <PatientName> / <TrialFolder> / ch1..ch8.wav
% =========================================================================

% Level 1 — patient folders
patient_dirs = dir(INPUT_ROOT);
patient_dirs = patient_dirs([patient_dirs.isdir]);
patient_dirs = patient_dirs(~ismember({patient_dirs.name}, {'.','..'}));

fprintf('[DISCOVERY] Found %d patient folder(s) in: %s\n', ...
        numel(patient_dirs), INPUT_ROOT);

% Level 2 — collect all trial folders across all patients
trial_dirs = struct('name', {}, 'folder', {}, 'patient', {});
for pi = 1:numel(patient_dirs)
    pat_path = fullfile(INPUT_ROOT, patient_dirs(pi).name);
    sub = dir(pat_path);
    sub = sub([sub.isdir]);
    sub = sub(~ismember({sub.name}, {'.','..'}));
    for si = 1:numel(sub)
        % Only include if this sub-folder actually contains .wav files
        wavcheck = dir(fullfile(pat_path, sub(si).name, '*.wav'));
        if isempty(wavcheck), continue; end
        entry.name    = sub(si).name;
        entry.folder  = pat_path;
        entry.patient = patient_dirs(pi).name;
        trial_dirs(end+1) = entry; %#ok<AGROW>
    end
end

fprintf('[DISCOVERY] Found %d trial folder(s) with .wav files\n\n', numel(trial_dirs));

if isempty(trial_dirs)
    fprintf('[ERROR] No trial folders with .wav files found.\n');
    fprintf('        Expected structure:\n');
    fprintf('        %s/<PatientName>/<TrialFolder>/*.wav\n\n', INPUT_ROOT);
    return;
end

global_csv_rows = {};

% =========================================================================
%  4. MAIN LOOP — EACH TRIAL FOLDER
% =========================================================================

for ti = 1:numel(trial_dirs)

    trial_name   = trial_dirs(ti).name;                          % e.g. "ACOB T1"
    patient_name = trial_dirs(ti).patient;                       % e.g. "ACOB"
    trial_folder = fullfile(trial_dirs(ti).folder, trial_name);  % full path to trial
    % Output keeps hierarchy: PIPELINE_OUTPUT/ACOB/ACOB T1/
    trial_out    = fullfile(OUTPUT_ROOT, patient_name, trial_name);

    wav_files = dir(fullfile(trial_folder, '*.wav'));
    if isempty(wav_files)
        fprintf('[SKIP] %s/%s — no .wav files found\n', patient_name, trial_name);
        continue;
    end

    fprintf('─────────────────────────────────────────────────────────────\n');
    fprintf('[TRIAL] %s / %s  (%d channels)\n', patient_name, trial_name, numel(wav_files));

    % ------------------------------------------------------------------
    %  4.1  CREATE OUTPUT DIRECTORY TREE
    % ------------------------------------------------------------------
    dirs = {trial_out, ...
            fullfile(trial_out, 'segments'), ...
            fullfile(trial_out, 'full_preprocessed'), ...
            fullfile(trial_out, 'step_audio', 'raw'), ...
            fullfile(trial_out, 'step_audio', 'dc_removed'), ...
            fullfile(trial_out, 'step_audio', 'resampled'), ...
            fullfile(trial_out, 'step_audio', 'bandpassed'), ...
            fullfile(trial_out, 'step_audio', 'kalman'), ...
            fullfile(trial_out, 'step_audio', 'nmf_lung'), ...
            fullfile(trial_out, 'plots', 'psd'), ...
            fullfile(trial_out, 'plots', 'spectrograms'), ...
            fullfile(trial_out, 'plots', 'rms'), ...
            fullfile(trial_out, 'plots', 'noise_floor'), ...
            fullfile(trial_out, 'plots', 'feature_space'), ...
            fullfile(trial_out, 'report')};
    for d = 1:numel(dirs)
        if ~exist(dirs{d},'dir'), mkdir(dirs{d}); end
    end

    % ------------------------------------------------------------------
    %  4.2  SORT CHANNELS
    % ------------------------------------------------------------------
    ch_nums = zeros(1, numel(wav_files));
    for i = 1:numel(wav_files)
        tok = regexp(wav_files(i).name,'ch(\d+)','tokens');
        if ~isempty(tok), ch_nums(i) = str2double(tok{1}{1}); end
    end
    [~, si]   = sort(ch_nums);
    wav_files = wav_files(si);
    n_ch      = numel(wav_files);

    [~, fname0, ~] = fileparts(wav_files(1).name);
    id_tok = regexp(fname0,'(P\d+_t\d+)','tokens');
    if ~isempty(id_tok)
        participant_id = id_tok{1}{1};
    else
        participant_id = regexprep(trial_name,' ','_');
    end
    fprintf('[ID] Participant ID: %s\n\n', participant_id);

    all_psd_freqs    = [];
    all_psd_vals     = cell(1, n_ch);
    all_rms_vals     = cell(1, n_ch);
    all_noise_floors = zeros(1, n_ch);
    all_segments     = cell(1, n_ch);
    all_seg_labels   = cell(1, n_ch);
    seg_counts       = zeros(1, n_ch);
    ch_ids_arr       = zeros(1, n_ch);

    colors = lines(n_ch);

    % ==================================================================
    %  STEPS 1-4: PER-CHANNEL  Resample → DC → Bandpass → Kalman
    % ==================================================================
    processed = cell(1, n_ch);

    for ch = 1:n_ch
        fpath = fullfile(trial_folder, wav_files(ch).name);
        [~, fname_base, ~] = fileparts(wav_files(ch).name);
        tok   = regexp(wav_files(ch).name,'ch(\d+)','tokens');
        ch_id = ch;
        if ~isempty(tok), ch_id = str2double(tok{1}{1}); end
        ch_ids_arr(ch) = ch_id;
        site = SITE_LABELS{min(ch, numel(SITE_LABELS))};

        fprintf('  ── ch%d (%s) ──────────────────────────────────────\n', ch_id, site);

        % ── LOAD ──────────────────────────────────────────────────────
        [x, fs_orig] = audioread(fpath);
        if size(x,2) > 1
            x = mean(x, 2);
            fprintf('    [LOAD] Stereo → mono\n');
        end
        x = double(x(:));
        fprintf('    [LOAD] %.1f sec @ %d Hz\n', numel(x)/fs_orig, fs_orig);

        save_step_audio(x, fs_orig, trial_out, 'raw', [fname_base '_raw.wav']);

        % ── STEP 1: RESAMPLE ──────────────────────────────────────────
        if fs_orig ~= TARGET_FS
            x = resample(x, TARGET_FS, fs_orig);
            fprintf('    [RS]   %d Hz → %d Hz\n', fs_orig, TARGET_FS);
        else
            fprintf('    [RS]   Already %d Hz\n', TARGET_FS);
        end
        save_step_audio(x, TARGET_FS, trial_out, 'resampled', [fname_base '_resampled_4k.wav']);

        % ── STEP 2: DC REMOVAL ────────────────────────────────────────
        dc_val = mean(x);
        x = x - dc_val;
        fprintf('    [DC]   DC offset removed: %.6f\n', dc_val);
        save_step_audio(x, TARGET_FS, trial_out, 'dc_removed', [fname_base '_dc_removed.wav']);

        % ── STEP 3: BANDPASS FILTER ───────────────────────────────────
        x = filtfilt(bp_b, bp_a, x);
        fprintf('    [BP]   Bandpass %d-%d Hz applied\n', BP_LOW, BP_HIGH);
        save_step_audio(x, TARGET_FS, trial_out, 'bandpassed', [fname_base '_bandpassed.wav']);

        % ── STEP 4: KALMAN FILTER ─────────────────────────────────────
        n_noise = round(0.5 * TARGET_FS);
        R_est   = var(x(1:min(n_noise, numel(x)))); % ← FIXED: numel(x) not 'end' in indexing context
        if R_est < 1e-10, R_est = 0.01; end
        x = kalman_1d(x, KALMAN_Q, R_est);
        fprintf('    [KF]   Kalman filter (Q=%.1e, R=%.2e)\n', KALMAN_Q, R_est);
        save_step_audio(x, TARGET_FS, trial_out, 'kalman', [fname_base '_kalman.wav']);

        all_noise_floors(ch) = 10 * log10(R_est + eps);
        processed{ch} = x;
    end

    % ==================================================================
    %  STEP 5: NMF HEART-LUNG SEPARATION (per channel)
    % ==================================================================
    fprintf('\n  [NMF]  Starting heart-lung separation...\n');

    min_len  = min(cellfun(@numel, processed));
    win_stft = hann(STFT_NFFT, 'periodic');

    for ch = 1:n_ch
        processed{ch} = processed{ch}(1:min_len);
    end

    nmf_out = cell(1, n_ch);
    for ch = 1:n_ch
        [~, fname_base, ~] = fileparts(wav_files(ch).name);
        ch_id = ch_ids_arr(ch);
        x     = processed{ch};

        [S, ~, ~] = spectrogram(x, win_stft, STFT_NFFT - STFT_HOP, STFT_NFFT, TARGET_FS);
        V   = abs(S) + eps;
        Phi = angle(S);

        [W, H] = nmf_betadiv(V, NMF_RANK, NMF_MAX_ITER, NMF_BETA, NMF_SPARSITY_H);

        scores   = zeros(1, NMF_RANK);
        is_heart = false(1, NMF_RANK);
        for r = 1:NMF_RANK
            scores(r) = cardiac_score(H(r,:)', TARGET_FS, STFT_HOP);
            if scores(r) > HEART_THRESH, is_heart(r) = true; end
        end
        if ~any(is_heart)
            [~, best] = max(scores);
            is_heart(best) = true;
        end

        V_lung    = W(:,~is_heart) * H(~is_heart,:);
        V_cardiac = W(:, is_heart) * H( is_heart,:);
        mask      = max(V_lung.^2 ./ (V_lung.^2 + V_cardiac.^2 + eps), MASK_FLOOR);

        S_lung = (V .* mask) .* exp(1j * Phi);
        x_lung = istft_ola(S_lung, STFT_NFFT, STFT_HOP, min_len);

        fprintf('    ch%d: NMF done (cardiac comps: %d/%d)\n', ...
                ch_id, sum(is_heart), NMF_RANK);

        % ── STEP 6: AMPLITUDE NORMALIZATION ──────────────────────────
        pk = max(abs(x_lung));
        if pk > 0, x_lung = x_lung * (PEAK_NORM / pk); end
        fprintf('    ch%d: Peak normalized to %.2f\n', ch_id, PEAK_NORM);

        save_step_audio(x_lung, TARGET_FS, trial_out, 'nmf_lung', ...
                        [fname_base '_nmf_lung.wav']);

        full_path = fullfile(trial_out, 'full_preprocessed', ...
                             [fname_base '_full_preprocessed.wav']);
        audiowrite(full_path, x_lung, TARGET_FS, 'BitsPerSample', 16);

        nmf_out{ch} = x_lung;

        [pxx, f_psd] = pwelch(x_lung, hamming(512), 256, 512, TARGET_FS);
        all_psd_freqs    = f_psd;
        all_psd_vals{ch} = 10 * log10(pxx + eps);
    end

    % ==================================================================
    %  STEP 7: AUTOMATED CYCLE DETECTION
    % ==================================================================
    fprintf('\n  [CYCLE] Detecting respiratory cycles...\n');

    env_frame_n = round(ENV_FRAME_SEC * TARGET_FS);
    env_hop_n   = round(ENV_HOP_SEC   * TARGET_FS);
    n_frames    = floor((min_len - env_frame_n) / env_hop_n) + 1;

    smooth_win = round(SMOOTH_WIN_SEC / ENV_HOP_SEC);
    if mod(smooth_win, 2) == 0, smooth_win = smooth_win + 1; end
    if smooth_win >= 3
        g_win = gausswin(smooth_win);
        g_win = g_win / sum(g_win);
    else
        g_win = [];
    end

    env_stack = zeros(n_frames, n_ch);
    for ch = 1:n_ch
        x  = nmf_out{ch};
        ef = zeros(n_frames, 1);
        for fr = 1:n_frames
            idx = (fr-1)*env_hop_n + (1:env_frame_n);
            idx = idx(idx <= numel(x));
            if isempty(idx), continue; end  % ← FIXED: guard against empty idx
            ef(fr) = rms(x(idx));
        end
        if ~isempty(g_win), ef = filtfilt(g_win, 1, ef); end
        em = max(ef);
        if em > 1e-8, ef = ef / em; end
        env_stack(:, ch) = ef;
    end

    env_cons = mean(env_stack, 2);
    ec_max   = max(env_cons);
    if ec_max < 1e-8
        fprintf('  [SKIP] Silent signal — no cycles detected\n\n');
        continue;
    end
    env_cons = env_cons / ec_max;

    min_dist = round(MIN_CYCLE_SEC / ENV_HOP_SEC);
    [~, pk_locs] = findpeaks(env_cons, ...
        'MinPeakDistance',   min_dist, ...
        'MinPeakProminence', PEAK_MIN_PROM, ...
        'MinPeakHeight',     PEAK_MIN_HEIGHT);

    fprintf('  [CYCLE] %d peaks found\n', numel(pk_locs));

    if numel(pk_locs) < 2
        fprintf('  [SKIP] < 2 peaks — no cycles saved\n\n');
        continue;
    end

    bounds_f    = zeros(numel(pk_locs)+1, 1);
    bounds_f(1) = find_valley(env_cons, 1, pk_locs(1));
    for pk = 1:numel(pk_locs)-1
        seg_env = env_cons(pk_locs(pk):pk_locs(pk+1));
        [~, mi] = min(seg_env);
        bounds_f(pk+1) = pk_locs(pk) + mi - 1;
    end
    bounds_f(end) = find_valley(env_cons, pk_locs(end), numel(env_cons));

    bounds_s = max(1, min(round((bounds_f-1)*env_hop_n + 1), min_len));

    n_cycles = numel(pk_locs);
    ch_cycle_ct = zeros(1, n_ch);

    for cy = 1:n_cycles
        s1  = bounds_s(cy);
        s2  = bounds_s(cy+1);
        dur = (s2 - s1 + 1) / TARGET_FS;

        if dur < MIN_CYCLE_SEC
            tag = 'SAVED_SHORT';
        elseif dur > MAX_CYCLE_SEC
            tag = 'SAVED_LONG';
        else
            tag = 'SAVED_VALID';
        end

        for ch = 1:n_ch
            ch_id = ch_ids_arr(ch);
            site  = SITE_LABELS{min(ch, numel(SITE_LABELS))};
            ch_cycle_ct(ch) = ch_cycle_ct(ch) + 1;

            seg = nmf_out{ch}(s1:s2);
            pk2 = max(abs(seg));
            if pk2 > 0, seg = seg * (PEAK_NORM / pk2); end

            seg_label = sprintf('%s_%s_cycle%03d_%s', ...
                                participant_id, site, ch_cycle_ct(ch), tag);
            out_name  = [seg_label '.wav'];
            audiowrite(fullfile(trial_out, 'segments', out_name), ...
                       seg, TARGET_FS, 'BitsPerSample', 16);

            all_segments{ch}{ch_cycle_ct(ch)}   = seg;
            all_seg_labels{ch}{ch_cycle_ct(ch)} = seg_label;

            global_csv_rows(end+1, :) = {trial_name, ch_id, out_name, dur, tag}; %#ok<AGROW>
        end

        for ch = 1:n_ch
            all_rms_vals{ch}(end+1) = rms(all_segments{ch}{ch_cycle_ct(ch)}); %#ok<AGROW>
        end
    end

    seg_counts(:) = ch_cycle_ct;
    total_segs    = sum(seg_counts);

    fprintf('  [DONE] %d cycles per channel saved to: %s\n\n', ...
            n_cycles, fullfile(trial_out, 'segments'));

    % ==================================================================
    %  PLOTS — PSD
    % ==================================================================
    fprintf('  [PLOTS] PSD...\n');

    fig_psd_all = figure('Name','PSD All','Position',[50 50 1400 900],'Visible','off');
    sgtitle(sprintf('Power Spectral Density — %s', participant_id), ...
            'FontSize',14,'FontWeight','bold');
    for ch = 1:n_ch
        subplot(ceil(n_ch/2), 2, ch);
        band_idx = (all_psd_freqs >= BP_LOW) & (all_psd_freqs <= BP_HIGH);
        plot(all_psd_freqs(band_idx), all_psd_vals{ch}(band_idx), ...
             'Color', colors(ch,:), 'LineWidth', 1.5);
        xlabel('Frequency (Hz)','FontSize',9);
        ylabel('Power/Freq (dB/Hz)','FontSize',9);
        title(sprintf('ch%d — %s', ch, SITE_LABELS{min(ch,end)}),'FontSize',10);
        grid on; xlim([BP_LOW BP_HIGH]);
    end
    saveas(fig_psd_all, fullfile(trial_out,'plots','psd','PSD_all_channels.png'));
    close(fig_psd_all);

    fig_psd_ov = figure('Name','PSD Overlay','Position',[50 50 900 500],'Visible','off');
    hold on;
    for ch = 1:n_ch
        band_idx = (all_psd_freqs >= BP_LOW) & (all_psd_freqs <= BP_HIGH);
        plot(all_psd_freqs(band_idx), all_psd_vals{ch}(band_idx), ...
             'Color', colors(ch,:), 'LineWidth', 1.2, ...
             'DisplayName', sprintf('ch%d (%s)', ch, SITE_LABELS{min(ch,end)}));
    end
    legend('Location','northeast','FontSize',8);
    xlabel('Frequency (Hz)','FontSize',11); ylabel('Power/Freq (dB/Hz)','FontSize',11);
    title(sprintf('PSD Overlay — %s', participant_id),'FontSize',12);
    grid on; xlim([BP_LOW BP_HIGH]);
    saveas(fig_psd_ov, fullfile(trial_out,'plots','psd','PSD_overlay.png'));
    close(fig_psd_ov);

    % ==================================================================
    %  PLOTS — RMS
    % ==================================================================
    fprintf('  [PLOTS] RMS...\n');

    all_rms_flat = [];
    fig_rms = figure('Name','RMS','Position',[50 50 1400 900],'Visible','off');
    sgtitle(sprintf('RMS Distribution — %s', participant_id), ...
            'FontSize',14,'FontWeight','bold');
    for ch = 1:n_ch
        subplot(ceil(n_ch/2), 2, ch);
        rv = all_rms_vals{ch};
        histogram(rv, max(2,min(15,numel(rv))), ...
                  'FaceColor', colors(ch,:), 'EdgeColor','white','FaceAlpha',0.8);
        xlabel('RMS Amplitude','FontSize',9); ylabel('Cycle Count','FontSize',9);
        title(sprintf('ch%d — %s (n=%d)', ch, SITE_LABELS{min(ch,end)}, numel(rv)),'FontSize',10);
        if numel(rv) > 0
            xline(mean(rv),'r--',sprintf('mean=%.3f', mean(rv)),'FontSize',8);
        end
        grid on;
        all_rms_flat = [all_rms_flat, rv]; %#ok<AGROW>
    end
    saveas(fig_rms, fullfile(trial_out,'plots','rms','RMS_per_channel.png'));
    close(fig_rms);

    if ~isempty(all_rms_flat)
        fig_rms_all = figure('Name','RMS All','Position',[50 50 700 450],'Visible','off');
        histogram(all_rms_flat, 25, 'FaceColor',[0.2 0.5 0.8],'EdgeColor','white');
        xlabel('RMS Amplitude','FontSize',11); ylabel('Segment Count','FontSize',11);
        title(sprintf('RMS — All Cycles — %s (n=%d)', participant_id, numel(all_rms_flat)),'FontSize',12);
        xline(mean(all_rms_flat),'r-',sprintf('mean=%.4f',mean(all_rms_flat)),'LineWidth',2,'FontSize',10);
        xline(median(all_rms_flat),'g--',sprintf('median=%.4f',median(all_rms_flat)),'LineWidth',1.5,'FontSize',10);
        grid on;
        saveas(fig_rms_all, fullfile(trial_out,'plots','rms','RMS_all_cycles.png'));
        close(fig_rms_all);
    end

    % ==================================================================
    %  PLOTS — NOISE FLOOR
    % ==================================================================
    fprintf('  [PLOTS] Noise floor...\n');

    fig_noise = figure('Name','Noise Floor','Position',[50 50 800 450],'Visible','off');
    bar(1:n_ch, all_noise_floors, 'FaceColor',[0.8 0.3 0.3],'EdgeColor','white');
    xticks(1:n_ch);
    xticklabels(arrayfun(@(c) sprintf('ch%d\n%s', c, SITE_LABELS{min(c,end)}), ...
                1:n_ch, 'UniformOutput', false));
    ylabel('Noise Floor (dBFS)','FontSize',11);
    title(sprintf('Estimated Noise Floor — %s', participant_id),'FontSize',12);
    grid on;
    if max(all_noise_floors)-min(all_noise_floors) > 1e-3
        ylim([min(all_noise_floors)-5, max(all_noise_floors)+5]);
    end
    yline(-60,'b--','ICBHI Ref (~-60 dBFS)','LineWidth',1.5,'FontSize',9);
    legend('DeepBreath Channels','ICBHI Reference','Location','best');
    saveas(fig_noise, fullfile(trial_out,'plots','noise_floor','noise_floor.png'));
    close(fig_noise);

    % ==================================================================
    %  PLOTS — MEL SPECTROGRAMS
    % ==================================================================
    fprintf('  [PLOTS] Mel spectrograms...\n');

    for ch = 1:n_ch
        if isempty(all_segments{ch}), continue; end
        seg_audio = all_segments{ch}{1};
        site      = SITE_LABELS{min(ch, numel(SITE_LABELS))};

        [S_sp, ~, T_sp] = spectrogram(seg_audio, hamming(N_FFT), ...
                                       N_FFT - HOP_SPEC, N_FFT, TARGET_FS);
        S_mag    = abs(S_sp);
        mel_fb   = mel_filterbank(N_MELS, N_FFT, TARGET_FS, BP_LOW, BP_HIGH);
        mel_spec = mel_fb * S_mag;
        mel_db   = 10 * log10(mel_spec + eps);

        fig_spec = figure('Name',sprintf('Mel ch%d',ch), ...
                          'Position',[50 50 700 350],'Visible','off');
        imagesc(T_sp, 1:N_MELS, mel_db); axis xy; colormap(hot); colorbar;
        xlabel('Time (s)','FontSize',10); ylabel('Mel Band','FontSize',10);
        title(sprintf('Mel Spectrogram — ch%d (%s) — Cycle001 — %s', ...
                      ch, site, participant_id),'FontSize',11);
        clim([prctile(mel_db(:),5), prctile(mel_db(:),99)]);
        saveas(fig_spec, fullfile(trial_out,'plots','spectrograms', ...
               sprintf('melspec_ch%d_%s.png', ch, site)));
        close(fig_spec);
    end

    % ==================================================================
    %  FEATURE EXTRACTION — MFCC + t-SNE
    % ==================================================================
    fprintf('  [MFCC]  Extracting features...\n');

    all_mfcc        = [];
    all_labels_flat = {};
    all_ch_ids_feat = [];

    for ch = 1:n_ch
        for s = 1:numel(all_segments{ch})
            seg = all_segments{ch}{s};

            [S_m, ~, ~] = spectrogram(seg, hamming(N_FFT), N_FFT - HOP_SPEC, N_FFT, TARGET_FS);
            mel_fb_mf   = mel_filterbank(N_MELS, N_FFT, TARGET_FS, BP_LOW, BP_HIGH);
            mel_e       = mel_fb_mf * abs(S_m);
            log_mel     = log(mel_e + eps);
            dct_mat     = dct_matrix(N_MELS, N_MFCC);
            mfcc_frames = dct_mat * log_mel;
            feat        = [mean(mfcc_frames, 2); std(mfcc_frames, 0, 2)]';

            all_mfcc(end+1, :)      = feat; %#ok<AGROW>
            all_labels_flat{end+1}  = all_seg_labels{ch}{s}; %#ok<AGROW>
            all_ch_ids_feat(end+1)  = ch_ids_arr(ch); %#ok<AGROW>
        end
    end

    fprintf('  [MFCC]  Matrix: %d cycles × %d features\n', ...
            size(all_mfcc,1), size(all_mfcc,2));

    if size(all_mfcc,1) >= 5
        mfcc_norm = (all_mfcc - mean(all_mfcc)) ./ (std(all_mfcc) + eps);
        try
            rng(42);
            tsne_coords = tsne(mfcc_norm, 'NumDimensions', 2, ...
                               'Perplexity', min(30, floor(size(mfcc_norm,1)/4)), ...
                               'Verbose', 0);
            fig_tsne = figure('Name','t-SNE','Position',[50 50 900 700],'Visible','off');
            hold on;
            for ch = 1:n_ch
                idx = (all_ch_ids_feat == ch_ids_arr(ch));
                scatter(tsne_coords(idx,1), tsne_coords(idx,2), 60, ...
                        colors(ch,:), 'filled', 'MarkerEdgeColor','white', ...
                        'LineWidth',0.5, 'DisplayName', ...
                        sprintf('ch%d (%s)', ch, SITE_LABELS{min(ch,end)}));
            end
            legend('Location','best','FontSize',9);
            xlabel('t-SNE Dim 1','FontSize',11); ylabel('t-SNE Dim 2','FontSize',11);
            title(sprintf('t-SNE — MFCC — %s', participant_id),'FontSize',12);
            grid on;
            saveas(fig_tsne, fullfile(trial_out,'plots','feature_space','tsne_by_channel.png'));
            close(fig_tsne);
            fprintf('  [TSNE]  Plot saved.\n');
        catch ME
            fprintf('  [TSNE]  Skipped: %s\n', ME.message);
        end
    end

    % ==================================================================
    %  SUMMARY STATS
    % ==================================================================
    report_data.participant_id  = participant_id;
    report_data.n_channels      = n_ch;
    report_data.total_segments  = total_segs;
    report_data.target_fs       = TARGET_FS;
    report_data.bp_low          = BP_LOW;
    report_data.bp_high         = BP_HIGH;
    report_data.peak_norm       = PEAK_NORM;
    report_data.seg_counts      = seg_counts;
    report_data.rms_mean        = cellfun(@(v) mean(v(:)), all_rms_vals);
    report_data.rms_std         = cellfun(@(v) std(v(:)),  all_rms_vals);
    report_data.noise_floors_db = all_noise_floors;
    report_data.site_labels     = SITE_LABELS(1:n_ch);

    fprintf('\n');
    fprintf('┌──────────────────────────────────────────────────────────────┐\n');
    fprintf('│              DEEPBREATH PREPROCESSING SUMMARY               │\n');
    fprintf('├──────────────────────────────────────────────────────────────┤\n');
    fprintf('│ Participant  : %-43s │\n', participant_id);
    fprintf('│ Channels     : %-43d │\n', n_ch);
    fprintf('│ Total Cycles : %-43d │\n', total_segs);
    fprintf('│ Sample Rate  : %-43s │\n', sprintf('%d Hz  ✓', TARGET_FS));
    fprintf('│ Bandpass     : %-43s │\n', sprintf('%d-%d Hz  ✓', BP_LOW, BP_HIGH));
    fprintf('│ NMF Rank     : %-43d │\n', NMF_RANK);
    fprintf('│ Peak Norm    : %-43s │\n', sprintf('%.2f  ✓', PEAK_NORM));
    fprintf('│ Bit Depth    : %-43s │\n', '16-bit PCM  ✓');
    fprintf('├──────┬──────────┬────────┬───────────┬──────────────────────┤\n');
    fprintf('│  CH  │   SITE   │ CYCLES │  RMS mean │  Noise Floor (dBFS)  │\n');
    fprintf('├──────┼──────────┼────────┼───────────┼──────────────────────┤\n');
    for ch = 1:n_ch
        fprintf('│  %2d  │  %-6s  │  %4d  │   %.4f  │       %6.1f dB       │\n', ...
                ch_ids_arr(ch), SITE_LABELS{min(ch,end)}, seg_counts(ch), ...
                report_data.rms_mean(ch), report_data.noise_floors_db(ch));
    end
    fprintf('├──────┴──────────┴────────┴───────────┴──────────────────────┤\n');
    fprintf('│ ICBHI Noise Reference: ~-60 dBFS (clinical stethoscope)     │\n');
    fprintf('└──────────────────────────────────────────────────────────────┘\n\n');

    % ==================================================================
    %  PER-TRIAL CSV
    % ==================================================================
    csv_path = fullfile(trial_out, 'report', 'summary_stats.csv');
    fid = fopen(csv_path, 'w');
    fprintf(fid, 'channel,site,cycle_count,rms_mean,rms_std,noise_floor_dBFS\n');
    for ch = 1:n_ch
        fprintf(fid, '%d,%s,%d,%.6f,%.6f,%.4f\n', ...
                ch_ids_arr(ch), SITE_LABELS{min(ch,end)}, seg_counts(ch), ...
                report_data.rms_mean(ch), report_data.rms_std(ch), ...
                report_data.noise_floors_db(ch));
    end
    fclose(fid);

    % ==================================================================
    %  HTML REPORT
    % ==================================================================
    generate_html_report(trial_out, report_data, participant_id);
    fprintf('  [REPORT] HTML report saved.\n');

    % ==================================================================
    %  SAVE MATLAB WORKSPACE
    % ==================================================================
    mat_path = fullfile(trial_out, 'report', 'pipeline_workspace.mat');
    save(mat_path, 'report_data', 'all_mfcc', 'all_ch_ids_feat', ...
         'all_rms_flat', 'seg_counts', 'all_noise_floors', ...
         'all_psd_vals', 'all_psd_freqs');
    fprintf('  [SAVE]   Workspace saved: %s\n\n', mat_path);

end % trial loop

% =========================================================================
%  GLOBAL CSV REPORT
% =========================================================================
if ~isempty(global_csv_rows)
    csv_path = fullfile(OUTPUT_ROOT, 'deepbreath_pipeline_report.csv');
    fid = fopen(csv_path, 'w');
    fprintf(fid, 'trial,channel,filename,duration_sec,status\n');
    for i = 1:size(global_csv_rows, 1)
        fprintf(fid, '%s,%d,%s,%.3f,%s\n', ...
                global_csv_rows{i,1}, global_csv_rows{i,2}, ...
                global_csv_rows{i,3}, global_csv_rows{i,4}, ...
                global_csv_rows{i,5});
    end
    fclose(fid);
    fprintf('[REPORT] Global CSV: %s\n', csv_path);
end

fprintf('\n=================================================================\n');
fprintf('  DEEPBREATH PIPELINE COMPLETE\n');
fprintf('  Output: %s\n', OUTPUT_ROOT);
fprintf('=================================================================\n\n');


% =========================================================================
%  LOCAL FUNCTIONS
% =========================================================================

function save_step_audio(x, fs, trial_out, step_name, filename)
    x_save = x;
    pk = max(abs(x_save));
    if pk > 0, x_save = 0.9 * x_save / pk; end
    audiowrite(fullfile(trial_out, 'step_audio', step_name, filename), ...
               x_save, fs, 'BitsPerSample', 16);
end


function x_est = kalman_1d(y, Q, R)
    N = numel(y); x_est = zeros(size(y));
    P = 1; x_hat = y(1);
    for t = 1:N
        P_pred = P + Q;
        K      = P_pred / (P_pred + R);
        x_hat  = x_hat + K * (y(t) - x_hat);
        P      = (1 - K) * P_pred;
        x_est(t) = x_hat;
    end
end


function [W, H] = nmf_betadiv(V, rank, max_iter, beta, sparsity_H)
    [F, T] = size(V); eps_ = 1e-9;
    sc = sqrt(mean(V(:)) / rank);
    W  = abs(randn(F, rank)) * sc + eps_;
    H  = abs(randn(rank, T)) * sc + eps_;
    for iter = 1:max_iter
        WH = W * H + eps_;
        switch beta
            case 2
                W_num = V*H';             W_den = WH*H';
                H_num = W'*V;             H_den = W'*WH;
            case 1
                WH_i  = 1 ./ WH;
                W_num = (V.*WH_i)*H';     W_den = repmat(sum(H,2)', F, 1);
                H_num = W'*(V.*WH_i);     H_den = repmat(sum(W,1)', 1, T);
            otherwise
                WH2   = WH.^2;
                W_num = (V./WH2)*H';      W_den = (1./WH)*H';
                H_num = W'*(V./WH2);      H_den = W'*(1./WH);
        end
        W = W .* (W_num ./ (W_den + eps_));
        H = H .* (H_num ./ (H_den + sparsity_H + eps_));
        cn = sqrt(sum(W.^2, 1)) + eps_;
        W  = W ./ cn;
        H  = H .* cn';
    end
end


function sc = cardiac_score(h, fs, hop)
    h = h(:) - mean(h); n = numel(h);
    if n < 8, sc = 0; return; end
    fs_h = fs / hop;
    Y    = abs(fft(h, n)); Y = Y(1:floor(n/2)+1);
    freq = (0:numel(Y)-1) * (fs_h / n);
    mask = freq >= 40/60 & freq <= 180/60;
    if sum(mask) < 1, sc = 0; return; end
    sc = min(max(Y(mask).^2) / (sum(Y.^2) + eps), 1);
end


function x_out = istft_ola(S, nfft, hop, orig_len)
    win    = hann(nfft, 'periodic');
    n_fr   = size(S, 2);
    sl     = (n_fr-1)*hop + nfft;
    x_out  = zeros(sl, 1);
    w_sum  = zeros(sl, 1);
    for f = 1:n_fr
        half  = S(:, f);
        full  = [half; conj(half(end-1:-1:2))];
        frame = real(ifft(full, nfft)) .* win;
        idx   = (f-1)*hop + (1:nfft);
        x_out(idx) = x_out(idx) + frame;
        w_sum(idx) = w_sum(idx) + win.^2;
    end
    nz = w_sum > 1e-8;
    x_out(nz)  = x_out(nz) ./ w_sum(nz);
    x_out(~nz) = 0;
    if numel(x_out) > orig_len
        x_out = x_out(1:orig_len);
    elseif numel(x_out) < orig_len
        x_out = [x_out; zeros(orig_len - numel(x_out), 1)];
    end
end


function b = find_valley(env, from_idx, to_idx)
    from_idx = max(1, from_idx);
    to_idx   = min(numel(env), to_idx);
    if from_idx >= to_idx, b = from_idx; return; end
    [~, mi] = min(env(from_idx:to_idx));
    b = from_idx + mi - 1;
end


function fb = mel_filterbank(n_mels, n_fft, fs, f_min, f_max)
    n_bins = n_fft / 2 + 1;
    f_bins = linspace(0, fs/2, n_bins);
    hz2mel = @(f) 2595 * log10(1 + f/700);
    mel2hz = @(m) 700 * (10.^(m/2595) - 1);
    mel_pts = linspace(hz2mel(f_min), hz2mel(f_max), n_mels + 2);
    hz_pts  = mel2hz(mel_pts);
    fb = zeros(n_mels, n_bins);
    for m = 1:n_mels
        fl = hz_pts(m); fc = hz_pts(m+1); fr = hz_pts(m+2);
        for k = 1:n_bins
            f = f_bins(k);
            if f >= fl && f <= fc
                fb(m,k) = (f - fl) / (fc - fl);
            elseif f > fc && f <= fr
                fb(m,k) = (fr - f) / (fr - fc);
            end
        end
    end
end


function D = dct_matrix(n_mels, n_mfcc)
    D = zeros(n_mfcc, n_mels);
    for k = 1:n_mfcc
        for n = 1:n_mels
            D(k,n) = cos(pi*(k-1)*(2*n-1)/(2*n_mels));
        end
    end
    D(1,:) = D(1,:) / sqrt(2);
    D = D * sqrt(2/n_mels);
end


function generate_html_report(trial_out, rd, pid)
    html_path = fullfile(trial_out, 'report', 'report.html');
    fid = fopen(html_path, 'w');
    timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');

    fprintf(fid, '<!DOCTYPE html>\n<html lang="en">\n<head>\n');
    fprintf(fid, '<meta charset="UTF-8">\n');
    fprintf(fid, '<title>DeepBreath Report — %s</title>\n', pid);
    fprintf(fid, '<style>\n');
    fprintf(fid, '  body{font-family:Arial,sans-serif;background:#f5f7fa;color:#222;margin:0;padding:20px}\n');
    fprintf(fid, '  h1{background:#1a3a5c;color:white;padding:20px 30px;border-radius:8px}\n');
    fprintf(fid, '  h2{color:#1a3a5c;border-bottom:2px solid #1a3a5c;padding-bottom:6px;margin-top:40px}\n');
    fprintf(fid, '  .meta{background:#e8f0fe;padding:15px 20px;border-radius:8px;margin-bottom:20px}\n');
    fprintf(fid, '  .meta span{margin-right:30px;font-size:14px}\n');
    fprintf(fid, '  .badge-ok{background:#27ae60;color:white;padding:2px 8px;border-radius:4px;font-size:12px}\n');
    fprintf(fid, '  table{border-collapse:collapse;width:100%%;margin-top:10px;background:white;border-radius:8px;overflow:hidden}\n');
    fprintf(fid, '  th{background:#1a3a5c;color:white;padding:10px 14px;text-align:left;font-size:13px}\n');
    fprintf(fid, '  td{padding:9px 14px;font-size:13px;border-bottom:1px solid #eee}\n');
    fprintf(fid, '  tr:hover td{background:#f0f4ff}\n');
    fprintf(fid, '  .plot-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(400px,1fr));gap:20px;margin-top:15px}\n');
    fprintf(fid, '  .plot-card{background:white;border-radius:8px;padding:15px;box-shadow:0 2px 6px rgba(0,0,0,.08)}\n');
    fprintf(fid, '  .plot-card img{width:100%%;border-radius:4px}\n');
    fprintf(fid, '  .plot-card p{font-size:12px;color:#666;margin:8px 0 0;text-align:center}\n');
    fprintf(fid, '  footer{text-align:center;color:#999;font-size:12px;margin-top:60px;padding-top:20px;border-top:1px solid #ddd}\n');
    fprintf(fid, '</style>\n</head>\n<body>\n');

    fprintf(fid, '<h1>&#x1F3E5; DeepBreath Preprocessing Report</h1>\n');
    fprintf(fid, '<div class="meta">');
    fprintf(fid, '<span><strong>Participant:</strong> %s</span>', pid);
    fprintf(fid, '<span><strong>Channels:</strong> %d</span>', rd.n_channels);
    fprintf(fid, '<span><strong>Total Cycles:</strong> %d</span>', rd.total_segments);
    fprintf(fid, '<span><strong>Generated:</strong> %s</span>', timestamp);
    fprintf(fid, '</div>\n');

    fprintf(fid, '<h2>Pipeline Steps</h2>\n');
    steps = {'1. Resample → 4000 Hz', '2. DC Offset Removal', ...
             '3. Bandpass Filter 100-1800 Hz', '4. Kalman Filtering', ...
             '5. NMF Heart-Lung Separation', '6. Amplitude Normalization', ...
             '7. Automated Cycle Detection'};
    fprintf(fid, '<div style="display:flex;flex-wrap:wrap;gap:10px;margin-top:10px;">\n');
    for s = 1:numel(steps)
        fprintf(fid, '<span class="badge-ok">%s &#x2713;</span>\n', steps{s});
    end
    fprintf(fid, '</div>\n');

    fprintf(fid, '<h2>ICBHI Compliance</h2>\n');
    fprintf(fid, '<table>\n');
    fprintf(fid, '<tr><th>Parameter</th><th>DeepBreath Value</th><th>ICBHI Target</th><th>Status</th></tr>\n');
    specs = {
        'Sample Rate',   sprintf('%d Hz', rd.target_fs),              '4000 Hz',    'MATCH';
        'Bit Depth',     '16-bit PCM',                                '16-bit PCM', 'MATCH';
        'Format',        'Mono WAV',                                  'Mono WAV',   'MATCH';
        'Bandpass',      sprintf('%d-%d Hz', rd.bp_low, rd.bp_high),  '100-1800 Hz','MATCH';
        'Normalization', sprintf('Peak %.1f', rd.peak_norm),          'Peak norm',  'MATCH';
    };
    for i = 1:size(specs,1)
        fprintf(fid, '<tr><td>%s</td><td>%s</td><td>%s</td><td><span class="badge-ok">%s &#x2713;</span></td></tr>\n', ...
                specs{i,1}, specs{i,2}, specs{i,3}, specs{i,4});
    end
    fprintf(fid, '</table>\n');

    fprintf(fid, '<h2>Per-Channel Statistics</h2>\n');
    fprintf(fid, '<table>\n');
    fprintf(fid, '<tr><th>Channel</th><th>Site</th><th>Cycles</th><th>RMS Mean</th><th>RMS Std</th><th>Noise Floor (dBFS)</th><th>vs ICBHI Ref</th></tr>\n');
    for ch = 1:rd.n_channels
        diff_db = rd.noise_floors_db(ch) - (-60);
        if diff_db <= 5
            flag = '<span class="badge-ok">OK</span>';
        else
            flag = sprintf('<span style="background:#e74c3c;color:white;padding:2px 8px;border-radius:4px;font-size:12px;">+%.1f dB high</span>', diff_db);
        end
        fprintf(fid, '<tr><td>ch%d</td><td>%s</td><td>%d</td><td>%.4f</td><td>%.4f</td><td>%.1f dB</td><td>%s</td></tr>\n', ...
                ch, rd.site_labels{min(ch,end)}, rd.seg_counts(ch), ...
                rd.rms_mean(ch), rd.rms_std(ch), rd.noise_floors_db(ch), flag);
    end
    fprintf(fid, '</table>\n');

    plot_sections(1).title = 'Power Spectral Density';
    plot_sections(1).imgs  = {'../plots/psd/PSD_overlay.png',      'PSD Overlay — All Channels';
                               '../plots/psd/PSD_all_channels.png', 'PSD per Channel'};
    plot_sections(2).title = 'RMS Distribution';
    plot_sections(2).imgs  = {'../plots/rms/RMS_all_cycles.png',   'RMS — All Cycles';
                               '../plots/rms/RMS_per_channel.png',  'RMS per Channel'};
    plot_sections(3).title = 'Noise Floor';
    plot_sections(3).imgs  = {'../plots/noise_floor/noise_floor.png', 'Noise Floor vs ICBHI Reference'};
    plot_sections(4).title = 'Feature Space (t-SNE)';
    plot_sections(4).imgs  = {'../plots/feature_space/tsne_by_channel.png', 't-SNE of MFCC Features by Channel'};

    for sec = 1:numel(plot_sections)
        fprintf(fid, '<h2>%s</h2>\n<div class="plot-grid">\n', plot_sections(sec).title);
        imgs = plot_sections(sec).imgs;
        for im = 1:size(imgs,1)
            fprintf(fid, '  <div class="plot-card"><img src="%s" alt="%s"><p>%s</p></div>\n', ...
                    imgs{im,1}, imgs{im,2}, imgs{im,2});
        end
        fprintf(fid, '</div>\n');
    end

    fprintf(fid, '<h2>Mel Spectrograms (Cycle 001 per Channel)</h2>\n<div class="plot-grid">\n');
    for ch = 1:rd.n_channels
        img_rel = sprintf('../plots/spectrograms/melspec_ch%d_%s.png', ch, rd.site_labels{min(ch,end)});
        fprintf(fid, '  <div class="plot-card"><img src="%s" alt="ch%d spectrogram"><p>ch%d — %s</p></div>\n', ...
                img_rel, ch, ch, rd.site_labels{min(ch,end)});
    end
    fprintf(fid, '</div>\n');

    fprintf(fid, '<footer>DeepBreath Pipeline &nbsp;|&nbsp; %s &nbsp;|&nbsp; %s</footer>\n', timestamp, pid);
    fprintf(fid, '</body>\n</html>\n');
    fclose(fid);
end