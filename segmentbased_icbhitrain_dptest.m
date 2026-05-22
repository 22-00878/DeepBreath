%% =========================================================
%  ABLATION STUDY: ICBHI TRAIN ONLY -> DEEPBREATH TEST ONLY
%  STATUS: DOMAIN SHIFT PROOF (Segment-Based)
%  Goal: Prove that without Fine-Tuning, custom hardware fails.
%  Architecture: ReliefF 50 + Boosted Trees 200 -> Deep MLP
%  Outputs: PNG/PDF/CSV Reports tagged with _SEGMENT
%% =========================================================
clc; clear; close all;
fprintf('╔══════════════════════════════════════════════════╗\n');
fprintf('║  ABLATION: ICBHI TRAIN -> DEEPBREATH TEST       ║\n');
fprintf('║  STATUS: SEGMENT-BASED DOMAIN SHIFT             ║\n');
fprintf('║  Architecture: SOTA Features + Trees + Deep MLP ║\n');
fprintf('╚══════════════════════════════════════════════════╝\n\n');

%% ─── OPTIMIZED CONFIGURATION (STRICTLY LOCKED) ───────────
K_SMOTE      = 5;
SMOTE_RATIO  = 1.5;   
N_RELIEF     = 50;    % LOCKED: Top 50 features
N_GB_TREES   = 200;   % LOCKED: 200 Boosted Trees
MAX_EPOCHS   = 300;   % LOCKED: 300 Epochs
BATCH_SIZE   = 32;    % LOCKED: 32 Batch Size
LR           = 0.0005;% LOCKED: 0.0005 Learning Rate
VAL_FRAC     = 0.15;  % 15% used for validation during training
LABEL_SMOOTH = 0.05;  % LOCKED: 0.05 Smoothing
FS_TARGET    = 4000;
N_FEAT       = 104;

%% ─── 1. FOLDER SELECTION ─────────────────────────────────
fprintf('[INFO] Select folder containing ICBHI TRAINING Data (processed_cycles).\n');
train_dir = uigetdir('', 'Select ICBHI TRAINING Data Folder');
if train_dir == 0, error('Cancelled.'); end

fprintf('[INFO] Select folder containing DeepBreath TESTING Data (Segment-Based Vault).\n');
test_dir = uigetdir('', 'Select DeepBreath TESTING Data Folder');
if test_dir == 0, error('Cancelled.'); end

OUT_DIR = uigetdir('', 'Select Output folder for Domain Shift Results');
if isequal(OUT_DIR,0), OUT_DIR = pwd; end

%% ─── 2. EXTRACT FEATURES ─────────────────────────────────
fprintf('\n[DATA] Extracting 104 SOTA Features from ICBHI Training Data...\n');
% apply_aug = true for ICBHI training
[X_tr_raw, Y_bin_tr, Y_abn_tr, ~, Is_Aug_tr, ~] = extract_folder(train_dir, FS_TARGET, N_FEAT, true);
fprintf('[OK] ICBHI Training Data Extracted: %d rows\n', size(X_tr_raw,1));

fprintf('\n[DATA] Extracting 104 SOTA Features from DeepBreath Testing Data...\n');
% apply_aug = false for Testing Vault
[X_te_raw, Y_bin_te, Y_abn_te, ~, ~, fnames_te] = extract_folder(test_dir, FS_TARGET, N_FEAT, false);
fprintf('[OK] DeepBreath Testing Data Extracted: %d rows\n\n', size(X_te_raw,1));

%% ─── 3. NORMALIZATION (Zero Leakage) ─────────────────────
% Calculate mean/std ONLY from original, unaugmented ICBHI training data
orig_mask = (Is_Aug_tr == 0);
feat_mean = mean(X_tr_raw(orig_mask, :), 1);
feat_std  = std(X_tr_raw(orig_mask, :), 0, 1);
feat_std(feat_std < 1e-9) = 1;

% Apply ICBHI statistics to both sets
X_tr_norm = (X_tr_raw - feat_mean) ./ feat_std;
X_te_norm = (X_te_raw - feat_mean) ./ feat_std;

%% ─── 4. TRAIN & TEST: BINARY (Normal vs Abnormal) ────────
fprintf('====================================================\n');
fprintf(' STAGE 1: DOMAIN SHIFT - BINARY CLASSIFICATION\n');
fprintf('====================================================\n');
Y_bin_tr_cat = removecats(categorical(Y_bin_tr));
Y_bin_te_cat = removecats(categorical(Y_bin_te));

[res_bin, cm_bin, sel_b] = train_and_test_domain_shift(...
    X_tr_norm, Y_bin_tr_cat, X_te_norm, Y_bin_te_cat, ...
    N_RELIEF, K_SMOTE, SMOTE_RATIO, N_GB_TREES, MAX_EPOCHS, BATCH_SIZE, LR, VAL_FRAC, LABEL_SMOOTH);

bin_cls = categories(Y_bin_tr_cat);
generate_outputs('DOMAIN_SHIFT_BINARY_SEGMENT', res_bin, cm_bin, bin_cls, OUT_DIR, 1);

%% ─── 5. TRAIN & TEST: MULTI-CLASS (Abnormal Subclasses) ──
fprintf('\n====================================================\n');
fprintf(' STAGE 2: DOMAIN SHIFT - MULTI-CLASS ABNORMAL\n');
fprintf('====================================================\n');
idx_sick_tr = (categorical(Y_abn_tr) ~= categorical({'Normal'}));
idx_sick_te = (categorical(Y_abn_te) ~= categorical({'Normal'}));

X_sick_tr = X_tr_norm(idx_sick_tr, :);
Y_sick_tr = removecats(categorical(Y_abn_tr(idx_sick_tr)));

X_sick_te = X_te_norm(idx_sick_te, :);
Y_sick_te = removecats(categorical(Y_abn_te(idx_sick_te)));
fnames_sick_te = fnames_te(idx_sick_te);

if sum(idx_sick_te) > 0
    % Using 1.0 ratio for abnormal classes to enforce 1:1:1 balance
    [res_abn, cm_abn, sel_a] = train_and_test_domain_shift(...
        X_sick_tr, Y_sick_tr, X_sick_te, Y_sick_te, ...
        N_RELIEF, K_SMOTE, 1.0, N_GB_TREES, MAX_EPOCHS, BATCH_SIZE, LR, VAL_FRAC, LABEL_SMOOTH);
    
    abn_cls = categories(Y_sick_tr);
    generate_outputs('DOMAIN_SHIFT_ABNORMAL_SEGMENT', res_abn, cm_abn, abn_cls, OUT_DIR, 2);
else
    res_abn.acc = NaN; res_abn.conf = NaN; cm_abn = [];
    fprintf('  [WARN] No Abnormal test samples found.\n');
end

%% ─── 6. FINAL CONSOLE REPORT & EXPORT ────────────────────
fprintf('\n  ╔═════════════════════════════════════════════════╗\n');
fprintf('  ║  DOMAIN SHIFT RESULTS (SEGMENT-BASED)           ║\n');
fprintf('  ╠═════════════════════════════════════════════════╣\n');
fprintf('  ║  Binary Accuracy     : %6.2f%%                 ║\n', res_bin.acc);
fprintf('  ║  Binary Confidence   : %6.2f%%                 ║\n', res_bin.conf);
if ~isnan(res_abn.acc)
fprintf('  ║  Abnormal Accuracy   : %6.2f%%                 ║\n', res_abn.acc);
fprintf('  ║  Abnormal Confidence : %6.2f%%                 ║\n', res_abn.conf);
end
fprintf('  ╚═════════════════════════════════════════════════╝\n\n');

export_vault_csv(fnames_te, Y_bin_te_cat, res_bin.pred, res_bin.conf_all, ...
                 fnames_sick_te, Y_sick_te, res_abn.pred, res_abn.conf_all, OUT_DIR);

fprintf('[SUCCESS] Segment-Based Domain Shift Experiment Complete. Results saved to %s\n', OUT_DIR);

%% =========================================================
%  CORE ENGINE: ONE-SHOT TRAIN & TEST
%% =========================================================
function [res, cm, sel_idx] = train_and_test_domain_shift(X_tr, Y_tr, X_te, Y_te, N_RELIEF, K_SMOTE, SMOTE_RATIO, N_TREES, MAX_EPOCHS, BATCH_SIZE, LR, VAL_FRAC, LABEL_SMOOTH)
    cls = categories(Y_tr); n_cls = numel(cls);
    rng(42);
    
    fprintf('  [1/4] Applying ReliefF (Top %d Features)...\n', N_RELIEF);
    try [ridx,~] = relieff(X_tr, Y_tr, 10, 'method', 'classification'); sel_idx = ridx(1:N_RELIEF); 
    catch, sel_idx = 1:N_RELIEF; end
    
    X_tr_sel = X_tr(:, sel_idx);
    X_te_sel = X_te(:, sel_idx);
    
    % Validation split (Segment-Based random partition)
    cv_val = cvpartition(Y_tr, 'HoldOut', VAL_FRAC, 'Stratify', true);
    X_tr_r = X_tr_sel(training(cv_val), :); Y_tr_r = Y_tr(training(cv_val));
    X_val  = X_tr_sel(test(cv_val), :);     Y_val  = Y_tr(test(cv_val));
    
    % Asymmetric Class Weighting Calculation
    orig_cnts = countcats(Y_tr_r); 
    cls_w = sum(orig_cnts) ./ (n_cls .* orig_cnts + eps);
    cls_w = cls_w ./ min(cls_w);
    
    fprintf('  [2/4] Applying SMOTE (Ratio %.1f)...\n', SMOTE_RATIO);
    [X_tr_sm, Y_tr_sm] = smote_tabular(X_tr_r, Y_tr_r, K_SMOTE, SMOTE_RATIO);
    
    fprintf('  [3/4] Training Boosted Ensemble (%d Trees)...\n', N_TREES);
    if n_cls == 2, gb_method = 'GentleBoost'; else, gb_method = 'AdaBoostM2'; end
    gb_tmpl = templateTree('MaxNumSplits', 8, 'MinLeafSize', 3);
    gb_mdl = fitcensemble(X_tr_sm, Y_tr_sm, 'Method', gb_method, 'NumLearningCycles', N_TREES, 'Learners', gb_tmpl, 'ClassNames', cls, 'LearnRate', 0.1);
    
    [~, gb_tr]  = predict(gb_mdl, X_tr_sm);
    [~, gb_val] = predict(gb_mdl, X_val);
    [~, gb_te]  = predict(gb_mdl, X_te_sel);
    
    X_tr_stk  = double([X_tr_sm, double(gb_tr)]);
    X_val_stk = double([X_val, double(gb_val)]);
    X_te_stk  = double([X_te_sel, double(gb_te)]);
    
    fprintf('  [4/4] Training Class-Weighted Deep MLP (%d Epochs)...\n', MAX_EPOCHS);
    net = build_mlp(size(X_tr_stk, 2), n_cls);
    opts = make_opts(LR, MAX_EPOCHS, BATCH_SIZE, X_val_stk, Y_val);
    
    class_weights_vector = double(cls_w(:)'); 
    loss_fn = @(Yp, Yt) loss_wsce(Yp, Yt, class_weights_vector, LABEL_SMOOTH, n_cls);
    net = trainnet(X_tr_stk, Y_tr_sm, net, loss_fn, opts);
    
    % Final Evaluation on Test Set
    [te_acc, te_conf, S_te, Y_pred_te] = evaluate_mlp(net, X_te_stk, Y_te, cls);
    cm = confusionmat(Y_te, Y_pred_te, 'Order', cls);
    f1_macro = mean(comp_f1(cm)) * 100;
    
    fprintf('  ║  DOMAIN SHIFT TEST -> Acc: %6.2f%% | Conf: %6.2f%% | F1: %6.2f%%\n', te_acc, te_conf, f1_macro);
    
    res.acc = te_acc; res.conf = te_conf; res.f1 = f1_macro;
    res.S_all = S_te; res.Y_all_true = Y_te; res.pred = Y_pred_te;
    res.conf_all = max(S_te, [], 2) * 100;
end

%% =========================================================
%  HELPER FUNCTIONS (Network, Loss, Metrics, Extractor)
%% =========================================================
function loss = loss_wsce(Yp, Yt, cls_weights, alpha, n_cls)
    Yt_data = double(extractdata(Yt)); w = cls_weights * Yt_data; 
    smooth_tgt = (1 - alpha) .* Yt_data + (alpha / n_cls);
    Yp_max = max(Yp, [], 1); e = exp(Yp - Yp_max); prob = e ./ sum(e, 1) + 1e-7; 
    ce = -sum(smooth_tgt .* log(prob), 1); 
    w_dl = dlarray(w); loss = sum(w_dl .* ce) / sum(w_dl);
end

function net=build_mlp(n_in,n_cls)
    layers=[featureInputLayer(n_in,'Normalization','none','Name','in'); fullyConnectedLayer(256,'Name','fc1'); batchNormalizationLayer('Name','bn1'); leakyReluLayer(0.01,'Name','lr1'); dropoutLayer(0.30,'Name','d1'); fullyConnectedLayer(128,'Name','fc2'); batchNormalizationLayer('Name','bn2'); leakyReluLayer(0.01,'Name','lr2'); dropoutLayer(0.20,'Name','d2'); fullyConnectedLayer(64,'Name','fc3'); batchNormalizationLayer('Name','bn3'); leakyReluLayer(0.01,'Name','lr3'); dropoutLayer(0.10,'Name','d3'); fullyConnectedLayer(n_cls,'Name','out')];
    net=dlnetwork(layerGraph(layers));
end

function opts=make_opts(lr,epochs,batch,X_val,Y_val)
    opts=trainingOptions('adam','MaxEpochs',epochs,'MiniBatchSize',batch,'InitialLearnRate',lr,'ValidationData',{X_val,Y_val},'ValidationFrequency',20,'ValidationPatience',30,'Shuffle','every-epoch','Verbose',false,'Plots','none');
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

function f1v=comp_f1(cm)
    nc=size(cm,1); f1v=zeros(nc,1); for c=1:nc, TP=cm(c,c); FP=sum(cm(:,c))-TP; FN=sum(cm(c,:))-TP; p=TP/(TP+FP+eps); r=TP/(TP+FN+eps); f1v(c)=2*p*r/(p+r+eps); end
end

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

%% =========================================================
%  EXPORT SUITE
%% =========================================================
function generate_outputs(name, res, cm, cls, OUT_DIR, stage)
    nc=numel(cls); auc_v=zeros(nc,1);
    for c=1:nc
        try yb=double(res.Y_all_true==cls{c}); [~,~,~,auc_v(c)]=perfcurve(yb,res.S_all(:,c),1); catch, auc_v(c)=NaN; end
    end
    
    fh2=figure('Name',sprintf('%s_ROC',name),'Position',[100 50 750 560],'Color','w');
    ax2=axes(fh2); hold(ax2,'on'); clr={'#0072BD','#D95319','#77AC30','#A2142F','#7E2F8E'};
    for c=1:nc
        yb=double(res.Y_all_true==cls{c});
        try [xr,yr,~,au]=perfcurve(yb,res.S_all(:,c),1); plot(ax2,xr,yr,'Color',clr{mod(c-1,numel(clr))+1},'LineWidth',2.5, 'DisplayName',sprintf('%s  AUC=%.3f',cls{c},au)); catch, end
    end
    plot(ax2,[0 1],[0 1],'k--','LineWidth',1.5,'DisplayName','Random');
    xlabel(ax2,'False Positive Rate','FontSize',12); ylabel(ax2,'True Positive Rate','FontSize',12); title(ax2,sprintf('%s — ROC',name),'FontSize',13); legend(ax2,'Location','southeast','FontSize',10); grid(ax2,'on');
    saveas(fh2,fullfile(OUT_DIR,[name '_ROC.png'])); exportgraphics(fh2,fullfile(OUT_DIR,[name '_ROC.pdf']),'ContentType','vector');
    
    fh1=figure('Name',sprintf('%s_CM',name),'Position',[50 50 720 560],'Color','w'); 
    confusionchart(cm,cls,'Title',sprintf('%s Acc:%.1f%%',name,res.acc),'RowSummary','row-normalized','ColumnSummary','column-normalized');
    saveas(fh1,fullfile(OUT_DIR,[name '_CM.png'])); exportgraphics(fh1,fullfile(OUT_DIR,[name '_CM.pdf']),'ContentType','vector');
    
    conf_vals=res.conf_all; correct=(res.Y_all_true==res.pred);
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

function export_vault_csv(fnames_bin, Y_true_bin, Y_pred_bin, conf_bin, fnames_abn, Y_true_abn, Y_pred_abn, conf_abn, OUT_DIR)
    fid=fopen(fullfile(OUT_DIR,'DomainShift_Results_SEGMENT.csv'),'w');
    fprintf(fid,'DOMAIN SHIFT BINARY RESULTS\nFile,True,Pred,Conf%%\n');
    for i=1:numel(fnames_bin), fprintf(fid,'%s,%s,%s,%.2f\n', fnames_bin(i),char(Y_true_bin(i)),char(Y_pred_bin(i)),conf_bin(i)); end
    if ~isempty(fnames_abn)
        fprintf(fid,'\nDOMAIN SHIFT ABNORMAL RESULTS\nFile,True,Pred,Conf%%\n');
        for i=1:numel(fnames_abn), fprintf(fid,'%s,%s,%s,%.2f\n', fnames_abn(i),char(Y_true_abn(i)),char(Y_pred_abn(i)),conf_abn(i)); end
    end
    fclose(fid);
end