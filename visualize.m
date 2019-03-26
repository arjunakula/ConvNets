function[]  = visualize()
    image_path = 'images/10.png';
    load('model.mat');
    img = imread(image_path);
    close all;
    p_img = single(img)-net.averageImage;
    net.layers = net.layers(1:end-1);
    
    res = vl_simplenn(net,p_img);
    
    for i = 1:32
        subplot(8,4,i);
        image(res(2).x(:,:,i));
        title(['Filter ',num2str(i)]);
    end
end