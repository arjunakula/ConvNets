function [net, info] = cnn_train(net, imdb, getBatch, varargin)

opts.batchSize = 256 ;
opts.numSubBatches = 1 ;
opts.train = [] ;
opts.val = [] ;
opts.numEpochs = 300 ;
opts.learningRate = 0.001 ;
opts.continue = false ;
opts.expDir = fullfile('data','exp') ;
opts.conserveMemory = false ;
opts.backPropDepth = +inf ;
opts.sync = false ;
opts.prefetch = false ;
opts.cudnn = true ;
opts.weightDecay = 0.0005 ;
opts.momentum = 0.9 ;
opts.errorFunction = 'multiclass' ;
opts.errorLabels = {} ;
opts.plotDiagnostics = false ;
opts.memoryMapFile = fullfile(tempdir, 'matconvnet.bin') ;
opts = vl_argparse(opts, varargin) ;

if ~exist(opts.expDir, 'dir'), mkdir(opts.expDir) ; end
if isempty(opts.train), opts.train = find(imdb.images.set==1) ; end
if isempty(opts.val), opts.val = find(imdb.images.set==2) ; end

if isnan(opts.train), opts.train = [] ; end


% -------------------------------------------------------------------------
%          Network initialization
% -------------------------------------------------------------------------

evaluateMode = isempty(opts.train) ;

if ~evaluateMode
  for i=1:numel(net.layers)
    if isfield(net.layers{i}, 'weights')
      J = numel(net.layers{i}.weights) ;
      for j=1:J
        net.layers{i}.momentum{j} = zeros(size(net.layers{i}.weights{j}), 'single') ;
      end
      if ~isfield(net.layers{i}, 'learningRate')
        net.layers{i}.learningRate = ones(1, J, 'single');
      end
      if ~isfield(net.layers{i}, 'weightDecay')
        net.layers{i}.weightDecay = ones(1, J, 'single');
      end
    end
  end
end

% setup error calculation function
opts.errorFunction = @error_multiclass ;
opts.errorLabels = {'top1e'};

% -------------------------------------------------------------------------
%                   Train and validate
% -------------------------------------------------------------------------
modelPath = @(ep) fullfile(opts.expDir, sprintf('net-epoch-%d.mat', ep));
modelFigPath = fullfile(opts.expDir, 'net-train.pdf') ;

for epoch=1:opts.numEpochs

  % train one epoch and validate
  learningRate = opts.learningRate(min(epoch, numel(opts.learningRate))) ;
  train = opts.train(randperm(numel(opts.train))) ; % shuffle
  val = opts.val ;

  [net,stats.train] = process_epoch(opts, getBatch, epoch, train, learningRate, imdb, net) ;
  [~,stats.val] = process_epoch(opts, getBatch, epoch, val, 0, imdb, net) ;

  % save
  if evaluateMode, sets = {'val'} ; else sets = {'train', 'val'} ; end
  for f = sets
    f = char(f) ;
    n = numel(eval(f)) ;
    info.(f).speed(epoch) = n / stats.(f)(1) * 1;
    info.(f).objective(epoch) = stats.(f)(2) / n ;
    info.(f).error(:,epoch) = stats.(f)(3:end) / n ;
  end
  if ~evaluateMode, save(modelPath(epoch), 'net', 'info') ; end
end
draw_figures(opts.numEpochs, info.train.error, info.val.error);
info.train.error
info.val.error

function [] = draw_figures(numEpochs, train_error, val_error)
x = 1:numEpochs;
plot(x,train_error,'Color',1/255*[205 0 0],'LineWidth',2);
hold on;
plot(x,val_error,'Color',1/255*[0 205 0],'LineWidth',2);
set(gca,'FontSize',14);
title('Training Error vs Validation Error', 'FontSize', 17);
legend('Training Error', 'Validation Error');
xlabel('Epoch','FontSize',16);

% -------------------------------------------------------------------------
function err = error_multiclass(opts, labels, res)
% -------------------------------------------------------------------------
predictions = gather(res(end-1).x) ;
[~,predictions] = sort(predictions, 3, 'descend') ;

% be resilient to badly formatted labels
if numel(labels) == size(predictions, 4)
  labels = reshape(labels,1,1,1,[]) ;
end

% skip null labels
mass = single(labels(:,:,1,:) > 0) ;
if size(labels,3) == 2
  % if there is a second channel in labels, used it as weights
  mass = mass .* labels(:,:,2,:) ;
  labels(:,:,2,:) = [] ;
end

error = ~bsxfun(@eq, predictions, labels) ;
err(1,1) = sum(sum(sum(mass .* error(:,:,1,:)))) ;
%err(2,1) = sum(sum(sum(mass .* min(error(:,:,1:5,:),[],3)))) ;
