%% =========================================================
%  STEP 1 — ICBHI BENCHMARK  (SEGMENT-BASED ABLATION)
%  Architecture: ReliefF + GentleBoost/AdaBoostM2 → Class-Weighted MLP
%  Target: Demonstrates artificial high scores due to Data Leakage
%
%  OUTPUTS:
%    PNG+PDF   stage1_binary_cm, stage2_abnormal_cm
%              stage1_roc, stage2_roc
%              stage1_confidence, stage2_confidence
%    CSV       DeepBreath_CV_Results_SEGMENT.csv
%    MAT       Base_ICBHI_Brain_SEGMENT.mat
%% =========================================================
clc; clear; close all;
fprintf('╔══════════════════════════════════════════════════╗\n');
fprintf('║  STEP 1 — ICBHI BENCHMARK (SEGMENT-BASED)       ║\n');
fprintf('║  ReliefF + GentleBoost → Class-Weighted MLP     ║\n');
fprintf('║  10-Fold CV with Intentional Data Leakage       ║\n');
fprintf('╚══════════════════════════════════════════════════╝\n\n');

%% ─── OPTIMIZED CONFIGURATION ─────────────────────────────
% These hyperparameters have been tuned to maximize accuracy 
% without changing the underlying architecture.
K_FOLDS      = 10;
K_SMOTE      = 5;
SMOTE_RATIO  = 1.5;   
N_RELIEF     = 50;    % INCREASED (from 40) - Allows slightly more acoustic data
N_GB_TREES   = 200;   % INCREASED (from 150) - Stronger initial ensemble guess
MAX_EPOCHS   = 300;   % INCREASED (from 200) - Longer training time
BATCH_SIZE   = 32;    % DECREASED (from 64) - Improves generalization on small data
LR           = 0.0005;% DECREASED (from 0.001) - Finer, more stable weight updates
VAL_FRAC     = 0.15;
LABEL_SMOOTH = 0.05;  % DECREASED (from 0.1) - Allows the model to be slightly more confident
FS_TARGET    = 4000;

%% ─── INTERFACE ───────────────────────────────────────────
choice = questdlg('Select Pipeline Mode:', 'Step 1 — ICBHI (SEGMENT-BASED)', '1. Extract Features', '2. Load .mat File', '2. Load .mat File');
if isempty(choice), error('Cancelled.'); end

%% =========================================================
%  BLOCK 1: FEATURE EXTRACTION 
%% =========================================================
if startsWith(choice,'1')
    N_FEAT = 104;
    wav_dir = uigetdir('','Select pre-segmented WAV folder');
    if isequal(wav_dir,0), error('Cancelled.'); end
    [out_f,out_p] = uiputfile('deepbreath_icbhi_features.mat','Save as...');
    if isequal(out_f,0), error('Cancelled.'); end
    
    files=dir(fullfile(wav_dir,'*.wav')); N_files=numel(files); est_rows=N_files*4;
    X_raw=zeros(est_rows,N_FEAT,'double');
    Y_bin_raw=strings(est_rows,1); Y_abn_raw=strings(est_rows,1);
    PID_raw=zeros(est_rows,1);     AUG_raw=zeros(est_rows,1); fname_raw=strings(est_rows,1);
    n_ok=0; tic;
    
    for fi=1:N_files
        fname=files(fi).name; fpath=fullfile(files(fi).folder,fname); fname_lc=lower(fname);
        try
            tok=regexp(fname_lc,'^(\d+)_','tokens');
            if ~isempty(tok), pid=str2double(tok{1}{1}); else, pid=fi+900000; end
            if   contains(fname_lc,'_both')||contains(fname_lc,'both_'), l_bin='Abnormal'; l_abn='Both';
            elseif contains(fname_lc,'crackle'), l_bin='Abnormal'; l_abn='Crackle';
            elseif contains(fname_lc,'wheeze'),  l_bin='Abnormal'; l_abn='Wheeze';
            elseif contains(fname_lc,'normal'),  l_bin='Normal';   l_abn='Normal';
            else, continue; end
            
            [au,fs]=audioread(fpath); if size(au,2)>1, au=mean(au,2); end
            au=double(au(:)); if numel(au)<64, continue; end
            if fs~=FS_TARGET, au=resample(au,FS_TARGET,fs); end
            mx=max(abs(au)); if mx<1e-9, continue; end; au=au/mx;
            
            variants={au, au+0.005*randn(size(au)), min(1,max(-1,au*1.5)), au*0.5};
            af=[0,1,1,1];
            
            for vi=1:4
                feat=extract_104(variants{vi},FS_TARGET); n_ok=n_ok+1;
                X_raw(n_ok,:)=feat; Y_bin_raw(n_ok)=l_bin; Y_abn_raw(n_ok)=l_abn;
                PID_raw(n_ok)=pid;  AUG_raw(n_ok)=af(vi);  fname_raw(n_ok)=fname;
            end
        catch, continue; end
    end
    X_raw=X_raw(1:n_ok,:); Y_bin_raw=Y_bin_raw(1:n_ok); Y_abn_raw=Y_abn_raw(1:n_ok);
    PID_raw=PID_raw(1:n_ok); AUG_raw=AUG_raw(1:n_ok); fname_raw=fname_raw(1:n_ok);
    
    orig_mask=(AUG_raw==0);
    feat_mean=mean(X_raw(orig_mask,:),1); feat_std=std(X_raw(orig_mask,:),0,1); feat_std(feat_std<1e-9)=1;
    X_tab_all=(X_raw-feat_mean)./feat_std;
    Y_bin_all=categorical(Y_bin_raw); Y_abn_all=categorical(Y_abn_raw);
    P_IDs_all=PID_raw(:); Is_Aug_all=AUG_raw(:); is_prenorm=true;
    save(fullfile(out_p,out_f),'X_tab_all','Y_bin_all','Y_abn_all','P_IDs_all','Is_Aug_all','feat_mean','feat_std','fname_raw','-v7.3');
    fprintf('[SAVED] %s\n\n',fullfile(out_p,out_f));
%% =========================================================
%  BLOCK 2: LOAD EXISTING .mat FILE
%% =========================================================
else
    [fl,pl]=uigetfile('*.mat','Select feature file');
    if isequal(fl,0), error('Cancelled.'); end
    DL=load(fullfile(pl,fl));
    if isfield(DL,'X_tab_all'), X_tab_all=double(DL.X_tab_all); elseif isfield(DL,'X_norm'), X_tab_all=double(DL.X_norm); end
    Y_bin_all=categorical(DL.Y_bin_all(:)); Y_abn_all=categorical(DL.Y_abn_all(:));
    if isfield(DL,'P_IDs_all'), P_IDs_all=double(DL.P_IDs_all(:)); elseif isfield(DL,'PID_raw'), P_IDs_all=double(DL.PID_raw(:)); end
    if isfield(DL,'Is_Aug_all'), Is_Aug_all=double(DL.Is_Aug_all(:)); elseif isfield(DL,'AUG_raw'), Is_Aug_all=double(DL.AUG_raw(:)); else, Is_Aug_all=zeros(size(X_tab_all,1),1); end
    col_mu=mean(abs(mean(X_tab_all,1))); col_si=abs(mean(std(X_tab_all,0,1))-1);
    is_prenorm=(col_mu<0.5)&&(col_si<0.5);
end
OUT_DIR=uigetdir('','Select OUTPUT folder');
if isequal(OUT_DIR,0), OUT_DIR=pwd; end

%% =========================================================
%  CRITICAL FIX: CONVERT TO SEGMENT-BASED SPLITTING
%% =========================================================
fprintf('[INFO] Converting to SEGMENT-BASED splitting...\n');
% By assigning a unique integer to every single row (1 to N), MATLAB 
% is forced to treat every segment as completely independent.
% This guarantees Segment-Based splitting and introduces data leakage.
P_IDs_all = (1:size(X_tab_all, 1))';

%% =========================================================
%  STAGE 1 & 2 RUN
%% =========================================================
[res_b,gb_bin,net_bin,cm_b,sel_b,norm_b]=run_cv(X_tab_all, removecats(Y_bin_all), P_IDs_all, Is_Aug_all, is_prenorm, K_FOLDS, K_SMOTE, SMOTE_RATIO, N_RELIEF, N_GB_TREES, MAX_EPOCHS, BATCH_SIZE, LR, VAL_FRAC, LABEL_SMOOTH);
bin_cls=categories(removecats(Y_bin_all));
generate_outputs('SEGMENT_BINARY',res_b,cm_b,bin_cls,K_FOLDS,OUT_DIR,1);

abn_mask = ~strcmpi(cellstr(Y_abn_all),'normal');
[res_a,gb_abn,net_abn,cm_a,sel_a,norm_a]=run_cv(X_tab_all(abn_mask,:), removecats(Y_abn_all(abn_mask)), P_IDs_all(abn_mask), Is_Aug_all(abn_mask), is_prenorm, K_FOLDS, K_SMOTE, SMOTE_RATIO, N_RELIEF, N_GB_TREES, MAX_EPOCHS, BATCH_SIZE, LR, VAL_FRAC, LABEL_SMOOTH);
abn_cls=categories(removecats(Y_abn_all(abn_mask)));
generate_outputs('SEGMENT_ABNORMAL',res_a,cm_a,abn_cls,K_FOLDS,OUT_DIR,2);

export_csv(res_b,res_a,OUT_DIR);
save(fullfile(OUT_DIR,'Base_ICBHI_Brain_SEGMENT.mat'),'gb_bin','net_bin','bin_cls','sel_b','norm_b','gb_abn','net_abn','abn_cls','sel_a','norm_a','N_RELIEF','N_GB_TREES','FS_TARGET','is_prenorm','-v7.3');
fprintf('\n[SAVED] Base_ICBHI_Brain_SEGMENT.mat\n');

%% =========================================================
%  CORE CV ENGINE
%% =========================================================
function [res,gb_fin,net_fin,cm_agg,best_sel,best_norm]=run_cv(X_data,Y_data,P_IDs,Is_Aug,is_prenorm,K_FOLDS,K_SMOTE,SMOTE_RATIO,N_RELIEF,N_TREES,MAX_EPOCHS,BATCH_SIZE,LR,VAL_FRAC,LABEL_SMOOTH)
    cls=categories(Y_data); n_cls=numel(cls);
    unique_pids=unique(P_IDs); cv_pt=cvpartition(numel(unique_pids),'KFold',K_FOLDS);
    res.tr=nan(K_FOLDS,1); res.va=nan(K_FOLDS,1); res.te=nan(K_FOLDS,1); res.conf=nan(K_FOLDS,1); res.f1=nan(K_FOLDS,1);
    Y_all_true=categorical([]); Y_all_pred=categorical([]); S_all=zeros(0,n_cls); best_te=-1; 
    
    for fold=1:K_FOLDS
        fprintf('  ── Segment Fold %d/%d ──\n',fold,K_FOLDS);
        tr_mask=ismember(P_IDs,unique_pids(training(cv_pt,fold)));
        te_mask=ismember(P_IDs,unique_pids(test(cv_pt,fold)))&(Is_Aug==0);
        X_pool=X_data(tr_mask,:); Y_pool=Y_data(tr_mask);
        X_te_r=X_data(te_mask,:); Y_te=Y_data(te_mask);
        if isempty(Y_te), continue; end
        
        if is_prenorm, X_pool_n=X_pool; X_te_n=X_te_r; feat_mu=zeros(1,size(X_pool,2)); feat_sig=ones(1,size(X_pool,2));
        else, feat_mu=mean(X_pool,1); feat_sig=std(X_pool,0,1); feat_sig(feat_sig<1e-9)=1; X_pool_n=(X_pool-feat_mu)./feat_sig; X_te_n=(X_te_r-feat_mu)./feat_sig; end
        
        try [ridx,~]=relieff(X_pool_n,Y_pool,10,'method','classification'); sel=ridx(1:N_RELIEF); catch, sel=1:N_RELIEF; end
        X_pool_s=X_pool_n(:,sel); X_te_s=X_te_n(:,sel);
        
        rng(fold*7); cv_val=cvpartition(Y_pool,'HoldOut',VAL_FRAC,'Stratify',true);
        X_tr_r=X_pool_s(training(cv_val),:); Y_tr_r=Y_pool(training(cv_val));
        X_val_s=X_pool_s(test(cv_val),:);   Y_val=Y_pool(test(cv_val));
        
        [X_tr_sm,Y_tr_sm]=smote_bal(X_tr_r,Y_tr_r,K_SMOTE,SMOTE_RATIO);
        
        gb_method=iif(n_cls==2,'GentleBoost','AdaBoostM2');
        gb_tmpl=templateTree('MaxNumSplits',8,'MinLeafSize',3);
        gb_mdl=fitcensemble(X_tr_sm,Y_tr_sm,'Method',gb_method,'NumLearningCycles',N_TREES,'Learners',gb_tmpl,'ClassNames',cls,'LearnRate',0.1);
        
        [~,gb_tr]=predict(gb_mdl,X_tr_sm); [~,gb_val2]=predict(gb_mdl,X_val_s); [~,gb_te]=predict(gb_mdl,X_te_s);
        X_tr_stk=double([X_tr_sm,double(gb_tr)]); X_val_stk=double([X_val_s,double(gb_val2)]); X_te_stk=double([X_te_s,double(gb_te)]);
        
        cls_cnts=countcats(Y_tr_sm); cls_w=sum(cls_cnts)./(n_cls.*cls_cnts+eps); cls_w=cls_w./min(cls_w);
        net=build_mlp(size(X_tr_stk,2),n_cls);
        opts=make_opts(LR,MAX_EPOCHS,BATCH_SIZE,X_val_stk,Y_val);
        
        class_weights_vector = double(cls_w(:)'); 
        loss_fn = @(Yp,Yt)loss_wsce(Yp, Yt, class_weights_vector, LABEL_SMOOTH, n_cls);
        net = trainnet(X_tr_stk, Y_tr_sm, net, loss_fn, opts);
        
        [tr_acc,~,~]=eval_net(net,X_tr_stk, Y_tr_sm,cls);
        [val_acc,~,~]=eval_net(net,X_val_stk,Y_val,cls);
        [te_acc,te_conf,S_te]=eval_net(net,X_te_stk, Y_te,cls);
        Y_pred_te=pred_lbl(net,X_te_stk,cls);
        
        cm_fold=confusionmat(Y_te,Y_pred_te,'Order',cls); fold_f1=mean(comp_f1(cm_fold))*100;
        res.tr(fold)=tr_acc; res.va(fold)=val_acc; res.te(fold)=te_acc; res.conf(fold)=te_conf; res.f1(fold)=fold_f1;
        Y_all_true=[Y_all_true;Y_te]; Y_all_pred=[Y_all_pred;Y_pred_te]; S_all=[S_all;S_te];
        
        fprintf('  ║  Train: %6.2f%%  Val: %6.2f%%  Test: %6.2f%% (F1: %.2f%%)\n',tr_acc,val_acc,te_acc,fold_f1);
        if te_acc>best_te, best_te=te_acc; gb_fin=gb_mdl; net_fin=net; best_sel=sel; best_norm=struct('mu',feat_mu,'sig',feat_sig,'pre',is_prenorm); end
    end
    cm_agg=confusionmat(Y_all_true,Y_all_pred,'Order',cls); res.S_all=S_all; res.Y_all_true=Y_all_true; res.Y_all_pred=Y_all_pred;
end

%% =========================================================
%  CUSTOM LOSS & HELPERS
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
function [acc,conf,S]=eval_net(net,X,Y,cls_list)
    raw=minibatchpredict(net,double(X)); e=exp(raw-max(raw,[],2)); S=e./sum(e,2); [~,idx]=max(S,[],2); Yp=categorical(cls_list(idx),cls_list); acc=mean(Yp==Y)*100; conf=mean(max(S,[],2))*100;
end
function Yp=pred_lbl(net,X,cls_list)
    raw=minibatchpredict(net,double(X)); [~,idx]=max(raw,[],2); Yp=categorical(cls_list(idx),cls_list);
end
function f1v=comp_f1(cm)
    nc=size(cm,1); f1v=zeros(nc,1); for c=1:nc, TP=cm(c,c); FP=sum(cm(:,c))-TP; FN=sum(cm(c,:))-TP; p=TP/(TP+FP+eps); r=TP/(TP+FN+eps); f1v(c)=2*p*r/(p+r+eps); end
end
function [X_out,Y_out]=smote_bal(X,Y,k,ratio)
    if ~iscategorical(Y), Y=categorical(Y); end; Y=removecats(Y); cls=categories(Y); cnts=countcats(Y); target=ceil(max(cnts)*ratio); X_acc=X; Y_acc=Y;
    for c=1:numel(cls)
        nn=target-cnts(c); if nn<=0, continue; end
        idx_c=find(Y==cls{c}); Xm=X(idx_c,:); nm=numel(idx_c); Xs=zeros(nn,size(X,2));
        if nm==1, Xs=repmat(Xm,nn,1); else
            ku=max(1,min(k,nm-1)); D=pdist2(Xm,Xm); D(1:nm+1:end)=Inf;
            for s=1:nn, bi=randi(nm); [~,ni]=sort(D(bi,:)); ch=ni(randi(ku)); a=rand(); Xs(s,:)=Xm(bi,:)+a*(Xm(ch,:)-Xm(bi,:)); end
        end
        Yp2=Y(repmat(idx_c(1),nn,1)); Yp2(:)=cls{c}; X_acc=[X_acc;Xs]; Y_acc=[Y_acc;Yp2]; 
    end
    rng(42); p=randperm(size(X_acc,1)); X_out=double(X_acc(p,:)); Y_out=Y_acc(p);
end
function feat=extract_104(sig,fs)
    sig=double(sig(:)); if numel(sig)<256, sig=[sig;zeros(256-numel(sig),1)]; end
    zcr=mean(abs(diff(sign(sig))))/2; rms_eng=sqrt(mean(sig.^2)); [Pxx,F]=periodogram(sig,[],[],fs); denom=sum(Pxx)+eps; centroid=sum(F.*Pxx)/denom; spread=sqrt(sum(((F-centroid).^2).*Pxx)/denom); skew3=sum(((F-centroid).^3).*Pxx)/(spread^3*denom+eps); kurt4=sum(((F-centroid).^4).*Pxx)/(spread^4*denom+eps); cum_pwr=cumsum(Pxx)/denom; ri=find(cum_pwr>=0.85,1,'first'); if isempty(ri), rolloff=F(end); else, rolloff=F(ri); end
    try mc=mfcc(sig,fs,'NumCoeffs',13); mfcc_m=mean(mc,1); delta_m=mean(diff(mc,1,1),1); if isempty(delta_m), delta_m=zeros(1,13); end; dd_m=mean(diff(diff(mc,1,1),1,1),1); if isempty(dd_m), dd_m=zeros(1,13); end
    catch, mfcc_m=zeros(1,13); delta_m=zeros(1,13); dd_m=zeros(1,13); end
    band_edges=linspace(0,fs/2,16); bark_eng=zeros(1,15); for b2=1:15, mask=(F>=band_edges(b2))&(F<band_edges(b2+1)); bark_eng(b2)=log(sum(Pxx(mask))+eps); end
    feat=[zcr,rms_eng,centroid,spread,skew3,kurt4,rolloff,mfcc_m,delta_m,dd_m,bark_eng]; if numel(feat)<104, feat=[feat,zeros(1,104-numel(feat))]; end; feat=feat(1:104); feat(isnan(feat)|isinf(feat))=0;
end
function out=iif(cond,a,b), if cond, out=a; else, out=b; end, end

%% =========================================================
%  FULL EXPORT SUITE (PDFs, PNGs, CSVs)
%% =========================================================
function generate_outputs(name,res,cm,cls,k,OUT_DIR,stage)
    nc=numel(cls);
    sens=zeros(nc,1); spec=zeros(nc,1); prec=zeros(nc,1);
    f1=zeros(nc,1);   fpr=zeros(nc,1);  fnr=zeros(nc,1);
    npv=zeros(nc,1);  mcc_v=zeros(nc,1); auc_v=zeros(nc,1);
    for c=1:nc
        TP=cm(c,c); FP=sum(cm(:,c))-TP; FN=sum(cm(c,:))-TP; TN=sum(cm(:))-TP-FP-FN;
        sens(c)=TP/(TP+FN+eps); spec(c)=TN/(TN+FP+eps); prec(c)=TP/(TP+FP+eps); npv(c)=TN/(TN+FN+eps);
        f1(c)=2*(prec(c)*sens(c))/(prec(c)+sens(c)+eps);
        fpr(c)=FP/(FP+TN+eps); fnr(c)=FN/(FN+TP+eps);
        dm=sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN));
        if dm==0, mcc_v(c)=0; else, mcc_v(c)=(TP*TN-FP*FN)/dm; end
        if isfield(res,'S_all')&&~isempty(res.S_all)
            try yb=double(res.Y_all_true==cls{c}); [~,~,~,auc_v(c)]=perfcurve(yb,res.S_all(:,c),1); catch, auc_v(c)=NaN; end
        end
    end
    supp=arrayfun(@(c)sum(res.Y_all_true==cls{c}),1:nc);
    w_f1=sum(f1.*supp')/sum(supp);
    acc_m=mean(res.te,'omitnan'); acc_s=std(res.te,'omitnan'); f1_macro=mean(f1)*100;
    
    W=68;
    fprintf('\n╔%s╗\n',repmat('═',1,W));
    fprintf('║  %s — 10-Fold SEGMENT-BASED CV\n',name);
    fprintf('╠%s╣\n',repmat('═',1,W));
    fprintf('║  %-4s %10s %10s %10s %8s %6s  ║\n','Fold','Training%','Valid%','Testing%','Conf%','F1%');
    fprintf('║  %s  ║\n',repmat('─',1,W-4));
    for i=1:k
        if isnan(res.te(i)), fprintf('║  %3d  %10s %10s %10s %8s %6s  ║\n',i,'--','--','--','--','--');
        else, fprintf('║  %3d  %10.2f %10.2f %10.2f %8.2f %6.2f  ║\n', i,res.tr(i),res.va(i),res.te(i),res.conf(i),res.f1(i)); end
    end
    fprintf('║  %s  ║\n',repmat('─',1,W-4));
    fprintf('║  MEAN %10.2f %10.2f %10.2f %8.2f %6.2f  ║\n', mean(res.tr,'omitnan'),mean(res.va,'omitnan'),acc_m,mean(res.conf,'omitnan'),mean(res.f1,'omitnan'));
    fprintf('╚%s╝\n\n',repmat('═',1,W));

    fh1=figure('Name',sprintf('Stage%d_CM',stage),'Position',[50+stage*20 50 720 560],'Color','w');
    confusionchart(cm,cls, 'Title',sprintf('%s  Acc:%.1f%%  F1:%.1f%%', name,acc_m,f1_macro), 'RowSummary','row-normalized','ColumnSummary','column-normalized','FontSize',11);
    if stage==1, b='stage1_binary_cm_SEGMENT'; else, b='stage2_abnormal_cm_SEGMENT'; end
    saveas(fh1,fullfile(OUT_DIR,[b '.png'])); exportgraphics(fh1,fullfile(OUT_DIR,[b '.pdf']),'ContentType','vector');
end
function export_csv(res_b,res_a,OUT_DIR)
    fid=fopen(fullfile(OUT_DIR,'DeepBreath_CV_Results_SEGMENT.csv'),'w');
    fprintf(fid,'Stage,Fold,Training%%,Validation%%,Testing%%,Confidence%%,F1%%\n');
    for i=1:numel(res_b.te), fprintf(fid,'Binary,%d,%.2f,%.2f,%.2f,%.2f,%.2f\n', i,res_b.tr(i),res_b.va(i),res_b.te(i),res_b.conf(i),res_b.f1(i)); end
    for i=1:numel(res_a.te), fprintf(fid,'Abnormal,%d,%.2f,%.2f,%.2f,%.2f,%.2f\n', i,res_a.tr(i),res_a.va(i),res_a.te(i),res_a.conf(i),res_a.f1(i)); end
    fclose(fid);
end