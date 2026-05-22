%% =========================================================
%  CAPSTONE STEP 2 & 3: HARDWARE ADAPTATION & FINAL EXAM
%  STATUS: PATIENT-BASED / HONEST TRANSFER LEARNING
%  Goal: Fine-tune ICBHI AI to Custom Hardware & Test on Vault
%  Outputs: DeepBreath_Adapted_PatientBased.mat + PNG/PDF/CSV Reports
%% =========================================================
clc; clear; close all;
fprintf('====================================================\n');
fprintf(' DEEPBREATH: STEP 2 & 3 (ADAPTATION & FINAL EXAM)\n');
fprintf(' STATUS: PATIENT-BASED / HONEST EVALUATION\n');
fprintf(' MATLAB R2025b COMPATIBLE\n');
fprintf('====================================================\n\n');
%% ─── OPTIMIZED CONFIGURATION ─────────────────────────────
K_FOLDS_BIN     = 10;    
K_FOLDS_ABN     = 5;     
K_SMOTE         = 3;    
SMOTE_RATIO_BIN = 1.5;   
SMOTE_RATIO_ABN = 1.0;   
% Matching Optimized Hyperparameters for Transfer Learning
MAX_EPOCHS      = 100;    % OPTIMIZED: Increased for better settling
BATCH_SIZE      = 32;     % OPTIMIZED: Matches Step 1 generalization
LR_FINETUNE     = 5e-5;   % OPTIMIZED: Lowered to safely preserve ICBHI knowledge
LABEL_SMOOTH    = 0.05;   % OPTIMIZED: Matches Step 1 confidence sharpness
FS_TARGET       = 4000;
N_FEAT          = 104;
VAL_FRAC        = 0.15;
%% ─── 1. LOAD PRE-TRAINED ICBHI BRAIN ─────────────────────
fprintf('[INFO] Select Base_ICBHI_PatientBased.mat (From Step 1)\n');
fprintf('       DO NOT select step1_results.mat!\n\n');
[bf, bp] = uigetfile('*.mat', 'Select Base_ICBHI_PatientBased.mat');
if isequal(bf, 0), error('Cancelled.'); end
Brain = load(fullfile(bp, bf));
fprintf('[OK] Loaded Pre-Trained ICBHI Networks.\n\n');
%% ─── 2. SELECT FOLDERS ───────────────────────────────────
fprintf('[INFO] Select folder containing CUSTOM TRAINING Data (PATIENT-BASED).\n');
train_dir = uigetdir('', 'Select Patient-Based TRAINING Data Folder');
if train_dir == 0, error('Cancelled.'); end
fprintf('[INFO] Select folder containing UNTOUCHED VAULT Data (PATIENT-BASED).\n');
vault_dir = uigetdir('', 'Select Patient-Based VAULT Data Folder');
if vault_dir == 0, error('Cancelled.'); end
OUT_DIR = uigetdir('', 'Select Output folder for Final Results');
if isequal(OUT_DIR,0), OUT_DIR = pwd; end
%% ─── 3. EXTRACT FEATURES ─────────────────────────────────
fprintf('\n[STAGE 1] Extracting 104 SOTA Features from Custom Data...\n');
[X_tr_raw, Y_bin_tr, Y_abn_tr, PID_tr, Is_Aug_tr, ~] = extract_folder(train_dir, FS_TARGET, N_FEAT, true);
fprintf('[OK] Training Data Extracted: %d rows\n', size(X_tr_raw,1));
[X_vlt_raw, Y_bin_vlt, Y_abn_vlt, ~, ~, fnames_vlt] = extract_folder(vault_dir, FS_TARGET, N_FEAT, false);
fprintf('[OK] Vault Data Extracted: %d rows (No Augmentation allowed in vault)\n\n', size(X_vlt_raw,1));
% Normalize training data, apply identical parameters to vault (Zero Leakage)
orig_mask = (Is_Aug_tr == 0);
feat_mean = mean(X_tr_raw(orig_mask, :), 1);
feat_std  = std(X_tr_raw(orig_mask, :), 0, 1);
feat_std(feat_std < 1e-9) = 1;
X_tr_norm  = (X_tr_raw - feat_mean) ./ feat_std;
X_vlt_norm = (X_vlt_raw - feat_mean) ./ feat_std;
%% ─── 4. STEP 2: TRANSFER LEARNING (BINARY) ───────────────
fprintf('====================================================\n');
fprintf(' STEP 2: FINE-TUNING BINARY (Normal vs Abnormal)\n');
fprintf('====================================================\n');
Y_bin_tr_cat = removecats(categorical(Y_bin_tr));
[adapted_net_bin, adapted_gb_bin] = run_transfer_cv(...
   X_tr_norm, Y_bin_tr_cat, PID_tr, Is_Aug_tr, 'Binary Adaptation', ...
   K_FOLDS_BIN, MAX_EPOCHS, BATCH_SIZE, LR_FINETUNE, K_SMOTE, SMOTE_RATIO_BIN, VAL_FRAC, LABEL_SMOOTH, ...
   Brain.net_bin, Brain.sel_b);
%% ─── 5. STEP 2: TRANSFER LEARNING (MULTI-CLASS) ──────────
fprintf('\n====================================================\n');
fprintf(' STEP 2: FINE-TUNING MULTI-CLASS (Crackle/Wheeze/Both)\n');
fprintf('====================================================\n');
idx_sick_tr = (categorical(Y_abn_tr) ~= categorical({'Normal'}));
X_sick_tr   = X_tr_norm(idx_sick_tr, :);
Y_sick_tr   = removecats(categorical(Y_abn_tr(idx_sick_tr)));
PID_sick_tr = PID_tr(idx_sick_tr);
Aug_sick_tr = Is_Aug_tr(idx_sick_tr);
[adapted_net_abn, adapted_gb_abn] = run_transfer_cv(...
   X_sick_tr, Y_sick_tr, PID_sick_tr, Aug_sick_tr, 'Multi-Class Adaptation', ...
   K_FOLDS_ABN, MAX_EPOCHS, BATCH_SIZE, LR_FINETUNE, K_SMOTE, SMOTE_RATIO_ABN, VAL_FRAC, LABEL_SMOOTH, ...
   Brain.net_abn, Brain.sel_a);
%% ─── 6. STEP 3: THE FINAL EXAM (VAULT TESTING) ───────────
fprintf('\n====================================================\n');
fprintf(' STEP 3: THE FINAL EXAM (UNTOUCHED VAULT DATA)\n');
fprintf('====================================================\n');
% --- Vault Binary Test ---
Y_bin_vlt_cat = categorical(Y_bin_vlt);
X_vlt_bin_sel = X_vlt_norm(:, Brain.sel_b);
[~, rf_vlt_bin] = predict(adapted_gb_bin, X_vlt_bin_sel);
X_vlt_stk_bin = double([X_vlt_bin_sel, double(rf_vlt_bin)]);
[res_vlt_bin.te, res_vlt_bin.conf, probs_bin, Y_pred_vlt_bin] = evaluate_mlp(adapted_net_bin, X_vlt_stk_bin, Y_bin_vlt_cat, Brain.bin_cls);
cm_vault_bin = confusionmat(Y_bin_vlt_cat, Y_pred_vlt_bin, 'Order', Brain.bin_cls);
res_vlt_bin.Y_all_true = Y_bin_vlt_cat;
res_vlt_bin.Y_all_pred = Y_pred_vlt_bin;
res_vlt_bin.S_all = probs_bin;
% --- Vault Multi-Class Test ---
idx_sick_vlt = (categorical(Y_abn_vlt) ~= categorical({'Normal'}));
X_sick_vlt_norm = X_vlt_norm(idx_sick_vlt, :);
Y_sick_vlt_cat = categorical(Y_abn_vlt(idx_sick_vlt));
fnames_sick_vlt = fnames_vlt(idx_sick_vlt);
if sum(idx_sick_vlt) > 0
   X_sick_vlt_sel = X_sick_vlt_norm(:, Brain.sel_a);
   [~, rf_vlt_abn] = predict(adapted_gb_abn, X_sick_vlt_sel);
   X_vlt_stk_abn = double([X_sick_vlt_sel, double(rf_vlt_abn)]);
  
   [res_vlt_abn.te, res_vlt_abn.conf, probs_abn, Y_pred_vlt_abn] = evaluate_mlp(adapted_net_abn, X_vlt_stk_abn, Y_sick_vlt_cat, Brain.abn_cls);
   cm_vault_abn = confusionmat(Y_sick_vlt_cat, Y_pred_vlt_abn, 'Order', Brain.abn_cls);
  
   res_vlt_abn.Y_all_true = Y_sick_vlt_cat;
   res_vlt_abn.Y_all_pred = Y_pred_vlt_abn;
   res_vlt_abn.S_all = probs_abn;
else
   res_vlt_abn.te = NaN; res_vlt_abn.conf = NaN; cm_vault_abn = [];
   probs_abn = []; Y_pred_vlt_abn = [];
end
%% ─── 7. TABLE 8 EXPORT SUITE ─────────────────────────────
fprintf('  ╔═════════════════════════════════════════════════╗\n');
fprintf('  ║  ULTIMATE SYSTEM ACCURACY (VAULT EXAM)          ║\n');
fprintf('  ╠═════════════════════════════════════════════════╣\n');
fprintf('  ║  Binary Vault Accuracy : %6.2f%%                 ║\n', res_vlt_bin.te);
fprintf('  ║  Binary Confidence     : %6.2f%%                 ║\n', res_vlt_bin.conf);
if ~isnan(res_vlt_abn.te)
fprintf('  ║  Abnormal Vault Acc    : %6.2f%%                 ║\n', res_vlt_abn.te);
fprintf('  ║  Abnormal Confidence   : %6.2f%%                 ║\n', res_vlt_abn.conf);
else
fprintf('  ║  Abnormal Vault Acc    : N/A (No abnormal files) ║\n');
end
fprintf('  ╚═════════════════════════════════════════════════╝\n\n');
generate_table8_outputs('VAULT_BINARY', res_vlt_bin, cm_vault_bin, Brain.bin_cls, OUT_DIR);
if ~isnan(res_vlt_abn.te)
   generate_table8_outputs('VAULT_ABNORMAL', res_vlt_abn, cm_vault_abn, Brain.abn_cls, OUT_DIR);
end
export_vault_csv(fnames_vlt, Y_bin_vlt_cat, Y_pred_vlt_bin, max(probs_bin, [], 2)*100, fnames_sick_vlt, Y_sick_vlt_cat, Y_pred_vlt_abn, max(probs_abn, [], 2)*100, OUT_DIR);
bin_cls = Brain.bin_cls; abn_cls = Brain.abn_cls;
save(fullfile(OUT_DIR, 'DeepBreath_Adapted_PatientBased.mat'), 'adapted_net_bin', 'adapted_gb_bin', 'adapted_net_abn', 'adapted_gb_abn', 'feat_mean', 'feat_std', 'bin_cls', 'abn_cls', '-v7.3');
fprintf('[SUCCESS] DeepBreath_Adapted_PatientBased.mat Saved.\n');
fprintf('PIPELINE COMPLETE. System Ready for Presentation.\n');
%% =========================================================
%  CORE TRANSFER CV ENGINE
%% =========================================================
function [best_mlp_out, best_gb_out] = run_transfer_cv(X_data, Y_data, p_ids, is_aug, mode_name, K_FOLDS, MAX_EPOCHS, BATCH_SIZE, LR, K_SMOTE, SMOTE_RATIO, VAL_FRAC, LABEL_SMOOTH, pretrained_net, sel_idx)
   rng(42);
   unique_pts = unique(p_ids); unique_pts(isnan(unique_pts)|unique_pts<=0) = [];
   n_pts = numel(unique_pts); cls = categories(Y_data); numCls = numel(cls);
   cv = cvpartition(n_pts, 'KFold', min(K_FOLDS, n_pts));
  
   best_te_acc = -1; best_mlp_out = pretrained_net; best_gb_out = [];
  
   for fold = 1:cv.NumTestSets
       fprintf('  ── Fine-Tuning Fold %d / %d ──\n', fold, K_FOLDS);
       tr_mask = ismember(p_ids, unique_pts(training(cv, fold)));
       te_mask = ismember(p_ids, unique_pts(test(cv, fold))) & (is_aug == 0);
      
       X_pool = X_data(tr_mask, sel_idx); Y_pool = Y_data(tr_mask);
       X_te   = X_data(te_mask, sel_idx); Y_te   = Y_data(te_mask);
       if size(X_te, 1) < 1, continue; end
      
       rng(fold*7);
       cv_val=cvpartition(Y_pool,'HoldOut',VAL_FRAC,'Stratify',true);
       X_tr_r=X_pool(training(cv_val),:); Y_tr_r=Y_pool(training(cv_val));
       X_val_s=X_pool(test(cv_val),:);   Y_val=Y_pool(test(cv_val));
      
       miss=setdiff(cls,categories(removecats(Y_tr_r)));
       for mc=1:numel(miss)
           src=find(Y_data==miss{mc},1,'first');
           if ~isempty(src)
               X_tr_r=[X_tr_r; X_data(src,sel_idx)]; Y_tr_r=[Y_tr_r; Y_data(src)]; %#ok<AGROW>
           end
       end
      
       orig_cnts = countcats(Y_tr_r);
       cls_w = sum(orig_cnts) ./ (numCls .* orig_cnts + eps);
       cls_w = cls_w ./ min(cls_w);
      
       [X_tr_sm, Y_tr_sm] = smote_tabular(X_tr_r, Y_tr_r, K_SMOTE, SMOTE_RATIO);
      
       if numCls == 2, gb_method = 'GentleBoost'; else, gb_method = 'AdaBoostM2'; end
       gb_tmpl = templateTree('MaxNumSplits', 8, 'MinLeafSize', 3);
       gb_mdl = fitcensemble(X_tr_sm, Y_tr_sm, 'Method', gb_method, 'NumLearningCycles', 100, 'Learners', gb_tmpl, 'ClassNames', cls, 'LearnRate', 0.1);
      
       [~, gb_tr] = predict(gb_mdl, X_tr_sm);
       [~, gb_val]= predict(gb_mdl, X_val_s);
       [~, gb_te] = predict(gb_mdl, X_te);
      
       X_tr_stk = double([X_tr_sm, double(gb_tr)]);
       X_val_stk= double([X_val_s, double(gb_val)]);
       X_te_stk = double([X_te, double(gb_te)]);
      
       opts = trainingOptions('adam', 'MaxEpochs', MAX_EPOCHS, 'MiniBatchSize', BATCH_SIZE, ...
           'ValidationData',{X_val_stk,Y_val},'ValidationFrequency',20,'ValidationPatience',30,...
           'InitialLearnRate', LR, 'L2Regularization', 1e-4, 'Shuffle', 'every-epoch', 'Verbose', false, 'Plots', 'none');
      
       class_weights_vector = double(cls_w(:)');
       loss_fn = @(Yp, Yt) loss_wsce(Yp, Yt, class_weights_vector, LABEL_SMOOTH, numCls);
      
       mlp_net = trainnet(X_tr_stk, Y_tr_sm, pretrained_net, loss_fn, opts);
      
       [te_acc, te_conf, ~, Y_pred_te] = evaluate_mlp(mlp_net, X_te_stk, Y_te, cls);
      
       cm_fold = confusionmat(Y_te, Y_pred_te, 'Order', cls);
       f1_macro = mean(comp_f1(cm_fold)) * 100;
      
       fprintf('  ║  Adapted Acc: %6.2f%% | Conf: %6.2f%% | F1: %6.2f%%\n', te_acc, te_conf, f1_macro);
      
       if te_acc > best_te_acc, best_te_acc = te_acc; best_mlp_out = mlp_net; best_gb_out = gb_mdl; end
   end
end
%% =========================================================
%  HELPER: LOSS FUNCTION
%% =========================================================
function loss = loss_wsce(Yp, Yt, cls_weights, alpha, n_cls)
   Yt_data = double(extractdata(Yt)); w = cls_weights * Yt_data;
   smooth_tgt = (1 - alpha) .* Yt_data + (alpha / n_cls);
   Yp_max = max(Yp, [], 1); e = exp(Yp - Yp_max); prob = e ./ sum(e, 1) + 1e-7;
   ce = -sum(smooth_tgt .* log(prob), 1);
   w_dl = dlarray(w); loss = sum(w_dl .* ce) / sum(w_dl);
end
%% =========================================================
%  HELPER: EXTRACT FOLDER (PATIENT-BASED ID EXTRACTION)
%% =========================================================
function [X, Y_bin, Y_abn, PID, AUG, FNAMES] = extract_folder(folder_path, fs_target, n_feat, apply_aug)
   files = dir(fullfile(folder_path, '*.wav')); n_files = length(files);
   if apply_aug, est_rows = n_files * 4; else, est_rows = n_files; end
  
   X = zeros(est_rows, n_feat, 'double');
   Y_bin = strings(est_rows, 1); Y_abn = strings(est_rows, 1);
   PID = zeros(est_rows, 1); AUG = zeros(est_rows, 1); FNAMES = strings(est_rows, 1);
   valid = 0;
  
   for i = 1:n_files
       fname = lower(files(i).name); fpath = fullfile(files(i).folder, fname);
       try
           % Extracts Patient ID (e.g., 22 from P22_T3...)
           tok = regexp(fname, '^p?(\d+)_', 'tokens');
           if ~isempty(tok), pid = str2double(tok{1}{1}); else, pid = i + 100000; end
          
           if contains(fname, 'normal'), l_bin = 'Normal'; l_abn = 'Normal';
           elseif contains(fname, '_both') || endsWith(fname, 'both.wav'), l_bin = 'Abnormal'; l_abn = 'Both';
           elseif contains(fname, 'crackle'), l_bin = 'Abnormal'; l_abn = 'Crackle';
           elseif contains(fname, 'wheeze'), l_bin = 'Abnormal'; l_abn = 'Wheeze';
           else, continue; end
          
           [au, fs] = audioread(fpath);
           if size(au,2) > 1, au = mean(au,2); end
           au = double(au(:)); if numel(au) < 64, continue; end
           if fs ~= fs_target, au = resample(au, fs_target, fs); end
           max_v = max(abs(au)); if max_v == 0, continue; end
           au = au / max_v;
          
           if apply_aug
               vars = {au, au + 0.005*randn(size(au)), min(1,max(-1,au*1.5)), au*0.5};
               flags = [0, 1, 1, 1];
               for v = 1:4
                   feat = extract_104_sota(vars{v}, fs_target);
                   valid=valid+1; X(valid,:) = feat; Y_bin(valid) = l_bin; Y_abn(valid) = l_abn; PID(valid) = pid; AUG(valid) = flags(v); FNAMES(valid) = fname;
               end
           else
               feat = extract_104_sota(au, fs_target);
               valid=valid+1; X(valid,:) = feat; Y_bin(valid) = l_bin; Y_abn(valid) = l_abn; PID(valid) = pid; AUG(valid) = 0; FNAMES(valid) = fname;
           end
       catch, continue; end
   end
   X = X(1:valid, :); Y_bin = Y_bin(1:valid); Y_abn = Y_abn(1:valid); PID = PID(1:valid); AUG = AUG(1:valid); FNAMES = FNAMES(1:valid);
end
%% =========================================================
%  HELPER: 104 SOTA FEATURES & SMOTE
%% =========================================================
function feat = extract_104_sota(sig, fs)
   sig = double(sig(:)); if numel(sig) < 256, sig = [sig; zeros(256-numel(sig), 1)]; end
   zcr = mean(abs(diff(sign(sig)))) / 2; rms_eng = sqrt(mean(sig.^2));
   [Pxx, F] = periodogram(sig, [], [], fs); denom = sum(Pxx) + eps; centroid = sum(F .* Pxx) / denom; spread = sqrt(sum(((F - centroid).^2) .* Pxx) / denom);
   skewness = sum(((F - centroid).^3) .* Pxx) / (spread^3 * denom + eps); kurtosis = sum(((F - centroid).^4) .* Pxx) / (spread^4 * denom + eps);
   cum_pwr=cumsum(Pxx)/denom; ri=find(cum_pwr>=0.85,1,'first'); if isempty(ri), rolloff=F(end); else, rolloff=F(ri); end
   try mc = mfcc(sig, fs, 'NumCoeffs', 13); mfcc_m = mean(mc, 1); delta_m = mean(diff(mc, 1, 1), 1); if isempty(delta_m), delta_m = zeros(1, 13); end; dd_m = mean(diff(diff(mc, 1, 1), 1, 1), 1); if isempty(dd_m), dd_m = zeros(1, 13); end; catch, mfcc_m = zeros(1, 13); delta_m = zeros(1, 13); dd_m = zeros(1, 13); end
   band_edges = linspace(0, fs/2, 16); bark_eng = zeros(1, 15); for b = 1:15, mask = (F >= band_edges(b)) & (F < band_edges(b+1)); bark_eng(b) = log(sum(Pxx(mask)) + eps); end
   feat = [zcr, rms_eng, centroid, spread, skewness, kurtosis, rolloff, mfcc_m, delta_m, dd_m, bark_eng]; if length(feat) < 104, feat = [feat, zeros(1, 104 - length(feat))]; end; feat = feat(1:104); feat(isnan(feat) | isinf(feat)) = 0;
end
function [X_out, Y_out] = smote_tabular(X, Y, k, ratio)
   if ~iscategorical(Y), Y = categorical(Y); end; Y = removecats(Y);
   cls = categories(Y); counts = countcats(Y); target = ceil(max(counts) * ratio);
   X_acc = X; Y_acc = Y;
   for c = 1:numel(cls)
       nn = target - counts(c); if nn <= 0, continue; end
       idx_c = find(Y == cls{c}); Xm = X(idx_c, :); nm = numel(idx_c);
       Xs = zeros(nn, size(X, 2));
       if nm == 1, Xs = repmat(Xm, nn, 1);
       else
           ku = max(1, min(k, nm-1)); D = pdist2(Xm, Xm); D(1:nm+1:end) = Inf;
           for s = 1:nn
               bi = randi(nm); [~, ni] = sort(D(bi, :)); ch = ni(randi(ku));
               alpha = rand(); Xs(s, :) = Xm(bi, :) + alpha * (Xm(ch, :) - Xm(bi, :));
           end
       end
       Y_proto = Y(repmat(idx_c(1), nn, 1)); Y_proto(:) = cls{c};
       X_acc = [X_acc; Xs]; Y_acc = [Y_acc; Y_proto];
   end
   rng(42); p = randperm(size(X_acc, 1)); X_out = double(X_acc(p, :)); Y_out = Y_acc(p);
end
function [acc, conf, probs, Y_pred] = evaluate_mlp(net, X, Y, cls_list)
   raw_scores = minibatchpredict(net, double(X));
   exp_scores = exp(raw_scores - max(raw_scores, [], 2));
   probs      = exp_scores ./ sum(exp_scores, 2);
   [~, max_idx] = max(probs, [], 2);
   Y_pred = categorical(cls_list(max_idx), cls_list);
   acc  = mean(Y_pred == Y) * 100;
   conf = mean(max(probs, [], 2)) * 100;
end
function f1v = comp_f1(cm)
   nc=size(cm,1); f1v=zeros(nc,1);
   for c=1:nc
       TP=cm(c,c); FP=sum(cm(:,c))-TP; FN=sum(cm(c,:))-TP;
       p=TP/(TP+FP+eps); r=TP/(TP+FN+eps); f1v(c)=2*p*r/(p+r+eps);
   end
end
%% =========================================================
%  TABLE 8 EXPORT SUITE (PDF/PNG/CSV)
%% =========================================================
function generate_table8_outputs(name, res, cm, cls, OUT_DIR)
   nc=numel(cls); auc_v=zeros(nc,1);
   for c=1:nc
       try yb=double(res.Y_all_true==cls{c}); [~,~,~,auc_v(c)]=perfcurve(yb,res.S_all(:,c),1); catch, auc_v(c)=NaN; end
   end
  
   % 1. Save ROC Curves
   fh2=figure('Name',sprintf('%s_ROC',name),'Position',[100 50 750 560],'Color','w');
   ax2=axes(fh2); hold(ax2,'on'); clr={'#0072BD','#D95319','#77AC30','#A2142F','#7E2F8E'};
   for c=1:nc
       yb=double(res.Y_all_true==cls{c});
       try [xr,yr,~,au]=perfcurve(yb,res.S_all(:,c),1); plot(ax2,xr,yr,'Color',clr{mod(c-1,numel(clr))+1},'LineWidth',2.5, 'DisplayName',sprintf('%s  AUC=%.3f',cls{c},au)); catch, end
   end
   plot(ax2,[0 1],[0 1],'k--','LineWidth',1.5,'DisplayName','Random');
   xlabel(ax2,'False Positive Rate','FontSize',12); ylabel(ax2,'True Positive Rate','FontSize',12); title(ax2,sprintf('%s — ROC',name),'FontSize',13); legend(ax2,'Location','southeast','FontSize',10); grid(ax2,'on');
   saveas(fh2,fullfile(OUT_DIR,[name '_ROC.png'])); exportgraphics(fh2,fullfile(OUT_DIR,[name '_ROC.pdf']),'ContentType','vector');
  
   % 2. Save Confusion Matrix
   fh1=figure('Name',sprintf('%s_CM',name),'Position',[50 50 720 560],'Color','w');
   confusionchart(cm,cls,'Title',sprintf('%s Acc:%.1f%%',name,res.te),'RowSummary','row-normalized','ColumnSummary','column-normalized');
   saveas(fh1,fullfile(OUT_DIR,[name '_CM.png'])); exportgraphics(fh1,fullfile(OUT_DIR,[name '_CM.pdf']),'ContentType','vector');
  
   % 3. Save Confidence Distribution
   conf_vals=max(res.S_all,[],2)*100; correct=(res.Y_all_true==res.Y_all_pred);
   fh3=figure('Name',sprintf('%s_Conf',name),'Position',[150 100 1050 440],'Color','w');
   ax3a=subplot(1,2,1,'Parent',fh3);
   histogram(ax3a,conf_vals(correct),0:5:100, 'FaceColor',[0.18 0.65 0.30],'FaceAlpha',0.75,'EdgeColor','none','DisplayName','Correct'); hold(ax3a,'on');
   histogram(ax3a,conf_vals(~correct),0:5:100, 'FaceColor',[0.80 0.15 0.15],'FaceAlpha',0.75,'EdgeColor','none','DisplayName','Incorrect');
   xline(ax3a,mean(conf_vals(correct)),'--','Color',[0.18 0.65 0.30],'LineWidth',2, 'Label',sprintf('Correct: %.1f%%',mean(conf_vals(correct))));
   xline(ax3a,mean(conf_vals(~correct)),'--','Color',[0.80 0.15 0.15],'LineWidth',2, 'Label',sprintf('Incorrect: %.1f%%',mean(conf_vals(~correct))));
   xlabel(ax3a,'Softmax Confidence (%)','FontSize',11); ylabel(ax3a,'Count','FontSize',11); title(ax3a,sprintf('%s — Confidence Dist',name),'FontSize',12); legend(ax3a,'FontSize',9,'Location','northwest'); grid(ax3a,'on');
  
   ax3b=subplot(1,2,2,'Parent',fh3);
   cbc=cell(nc,1); lv=[]; for c=1:nc, cbc{c}=conf_vals(res.Y_all_true==cls{c}); lv=[lv; repmat(c,numel(cbc{c}),1)]; end
   if ~isempty(lv)
       boxplot(ax3b,cell2mat(cbc),lv,'Labels',cls,'Colors','bgrcm','MedianStyle','line','Widths',0.5);
   end
   ylabel(ax3b,'Softmax Confidence (%)','FontSize',11); title(ax3b,sprintf('%s — Per-Class Confidence',name),'FontSize',12); grid(ax3b,'on'); ylim(ax3b,[0 105]);
   saveas(fh3,fullfile(OUT_DIR,[name '_Confidence.png'])); exportgraphics(fh3,fullfile(OUT_DIR,[name '_Confidence.pdf']),'ContentType','vector');
end
% 4. Save Detailed CSV Output (Cycle Predictions)
function export_vault_csv(fnames_bin, Y_true_bin, Y_pred_bin, conf_bin, fnames_abn, Y_true_abn, Y_pred_abn, conf_abn, OUT_DIR)
   fid=fopen(fullfile(OUT_DIR,'DeepBreath_Vault_Results.csv'),'w');
   fprintf(fid,'VAULT BINARY RESULTS\nFile,True,Pred,Conf%%\n');
   for i=1:numel(fnames_bin), fprintf(fid,'%s,%s,%s,%.2f\n', fnames_bin(i),char(Y_true_bin(i)),char(Y_pred_bin(i)),conf_bin(i)); end
   if ~isempty(fnames_abn)
       fprintf(fid,'\nVAULT ABNORMAL RESULTS\nFile,True,Pred,Conf%%\n');
       for i=1:numel(fnames_abn), fprintf(fid,'%s,%s,%s,%.2f\n', fnames_abn(i),char(Y_true_abn(i)),char(Y_pred_abn(i)),conf_abn(i)); end
   end
   fclose(fid);
end