imds_ts = imageDatastore('dataset/test',...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');

imds_tr = imageDatastore('dataset/train',...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');

figure
numImages = length(imds_ts.Files);
perm = randperm(numImages,25);
for i = 1:25
    subplot(5,5,i);
    imshow(imds_ts.Files{perm(i)});
    drawnow
end

layers = [
    imageInputLayer([28 28 3],"Name","imageinput")
    convolution2dLayer([5 5],64,"Name","conv_1","Padding","same")
    reluLayer("Name","relu_2")
    maxPooling2dLayer([5 5],"Name","maxpool_1","Padding","same")
    convolution2dLayer([5 5],32,"Name","conv_2","Padding","same")
    reluLayer("Name","relu_3")
    maxPooling2dLayer([5 5],"Name","maxpool_2","Padding","same")
    fullyConnectedLayer(64,"Name","fc_1")
    reluLayer("Name","relu_1")
    fullyConnectedLayer(2,"Name","fc_2")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];

options = trainingOptions('sgdm', ...
    'MaxEpochs',10,...
    'InitialLearnRate',1e-4, ...
    'Verbose',false, ...
    'Plots','training-progress');

imageAugmenter = imageDataAugmenter( ...
    'RandRotation',[-20,20], ...
    'RandXTranslation',[-3 3], ...
    'RandYTranslation',[-3 3])

imageAugmenterT = imageDataAugmenter()

imageSize = [28 28 1];
augimds = augmentedImageDatastore(imageSize,imds_tr,'DataAugmentation',imageAugmenter);
augimds_ts = augmentedImageDatastore(imageSize,imds_ts,'DataAugmentation',imageAugmenterT);


net = trainNetwork(augimds,layers,options);
analyzeNetwork(net)

YPred = classify(net,augimds_ts);
YTest = imds_ts.Labels;

accuracy = sum(YPred == YTest)/numel(YTest)

load net
figure
idx = randperm(100,20);
for i = 1:20
    j = idx(i);
    subplot(4,5,i);
    img_path = imds_ts.Files{j};
    Label = imds_ts.Labels(j);
    I_org = imread(img_path);
    I = imresize(I_org,[28,28]);
    [YPred,scores] = classify(net,I);
    imshow(I_org)
    title([char(YPred),' ' , num2str(max(scores))])
end

save net



