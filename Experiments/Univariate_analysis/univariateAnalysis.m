%% Magnetometers
clear all;
path = 'Path\to\the\data\';
files = dir(fullfile(path,'*.mat'));
for i = 1 : length(files)
    % Loading the data
    load(fullfile(path,files(i).name),'data_ft');
    % Preprocessing, data selection, timelock analysis
    cfg=[];
    cfg.preproc.lpfilter='yes';
    cfg.preproc.lpfreq=45;
    faceInd = find(data_ft.trialinfo);
    data_face = ft_selectdata(data_ft,'rpt',faceInd,'channel','MEGMAG');
    data_face_avg = ft_timelockanalysis(cfg,data_face);
    scbInd = find(~data_ft.trialinfo);
    data_scb = ft_selectdata(data_ft,'rpt',scbInd,'channel','MEGMAG');
    data_scb_avg = ft_timelockanalysis(cfg,data_scb);
    
    cfg = [];
    cfg.method = 'montecarlo';       
    cfg.statistic = 'ft_statfun_indepsamplesT'; 
    cfg.correctm = 'cluster';
    cfg.clusterstatistic = 'maxsum'; 
    cfg.clusteralpha = 0.05;         
    cfg.minnbchan = 2;               
    cfg.tail = 0;
    cfg.clustertail = 0;
    cfg.alpha = 0.025;
    cfg.numrandomization = 1000;     
    
    design = zeros(1,size(data_face.trial,1) + size(data_scb.trial,1));
    design(1,1:size(data_face.trial,1)) = 1;
    design(1,(size(data_face.trial,1)+1):(size(data_face.trial,1) + size(data_scb.trial,1)))= 2;
    
    cfg.design = design;             
    cfg.ivar  = 1;                   
    
    cfg_neighb        = [];
    cfg_neighb.method = 'distance';
    cfg_neighb.layout = 'neuromag306mag.lay';  
    neighbours        = ft_prepare_neighbours(cfg_neighb, data_face_avg);
    
    cfg.neighbours    = neighbours;  
    cfg.channel       = {'MEGMAG'};     
    cfg.latency       = [-0.2 0.8];     
    [stat_CBPT_MAG(i)] = ft_timelockstatistics(cfg, data_face_avg, data_scb_avg);
    sig_CBPT_MAG(i) = sum(sum(stat_CBPT_MAG(i).mask));
end

%% Gradiometers
path = 'Path\to\the\data\';
files = dir(fullfile(path,'*.mat'));
for i = 1 : length(files)
    % Loading the data
    load(fullfile(path,files(i).name),'data_ft');
    
    % Preprocessing, data selection, timelock analysis
    cfg=[];
    cfg.preproc.lpfilter='yes';
    cfg.preproc.lpfreq=45;
    faceInd = find(data_ft.trialinfo);
    data_face = ft_selectdata(data_ft,'rpt',faceInd,'channel','MEGGRAD');
    cfg_cmb.combinemethod = 'svd';
    data_face = ft_combineplanar(cfg_cmb,data_face);
    data_face_avg = ft_timelockanalysis(cfg,data_face);
    scbInd = find(~data_ft.trialinfo);
    data_scb = ft_selectdata(data_ft,'rpt',scbInd,'channel','MEGGRAD');
    data_scb = ft_combineplanar(cfg_cmb,data_scb);
    data_scb_avg = ft_timelockanalysis(cfg,data_scb);
    
    cfg = [];
    cfg.method = 'montecarlo';       
    cfg.statistic = 'ft_statfun_indepsamplesT'; 
    cfg.correctm = 'cluster';
    cfg.clusterstatistic = 'maxsum'; 
    cfg.clusteralpha = 0.05;
    cfg.minnbchan = 2;               
    cfg.tail = 0;
    cfg.clustertail = 0;
    cfg.alpha = 0.025;
    cfg.numrandomization = 100;      
    
    design = zeros(1,size(data_face.trial,1) + size(data_scb.trial,1));
    design(1,1:size(data_face.trial,1)) = 1;
    design(1,(size(data_face.trial,1)+1):(size(data_face.trial,1) + size(data_scb.trial,1)))= 2;
    
    cfg.design = design;             
    cfg.ivar  = 1;                   
    
    cfg_neighb        = [];
    cfg_neighb.method = 'distance';
    cfg_neighb.layout = 'neuromag306cmb.lay'; 
    neighbours        = ft_prepare_neighbours(cfg_neighb, data_face_avg);
    
    cfg.neighbours    = neighbours;  
    cfg.latency       = [-0.2 0.8];       
    [stat_CBPT_GRAD(i)] = ft_timelockstatistics(cfg, data_face_avg, data_scb_avg);
    sig_CBPT_GRAD(i) = sum(sum(stat_CBPT_GRAD(i).mask));
 end
