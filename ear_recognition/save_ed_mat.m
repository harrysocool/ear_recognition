    %% load pre-trained edge detection model and set opts (see edgesDemo.m)
    cd('/home/harrysocool/Github/fast-rcnn/OP_methods/edges/');
    model=load('models/forest/modelEAR0.4_2.mat'); model=model.model;
    model.opts.multiscale=0; model.opts.sharpen=2; model.opts.nThreads=4;

    %% set up opts for edgeBoxes (see edgeBoxes.m)
    opts = edgeBoxes;
    opts.alpha = .35;     % step size of sliding window search
    opts.beta  = .75;     % nms threshold for object proposals
    opts.minScore = .01;  % min score of boxes to detect
    opts.maxBoxes = 1e4;  % max number of boxes to detect

    %% detect Edge Box bounding box proposals (see edgeBoxes.m)
    % Process all images.
    fileID = fopen('/home/harrysocool/Github/fast-rcnn/ear_recognition/data_file/image_index_list.csv');
    image_list = textscan(fileID, '%s', 'Delimiter','\n');
    all_boxes = {};    
    for index = 1:length(image_list{1})
        im = imread(image_list{1}{index});
        bbs=edgeBoxes(im,model,opts);
        all_boxes{index} = double(bbs(:, 1:4));
        disp(index);
    end
    save('/home/harrysocool/Github/fast-rcnn/ear_recognition/data_file/ed_all_boxes.mat', 'all_boxes', '-v7');