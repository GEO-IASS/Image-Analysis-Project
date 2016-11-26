function [training_set,test_set] = partition_data(dsObj, m, n)

    image_location = fileparts(dsObj.Files{1}); 
    disp(image_location);
    imset = imageSet(strcat(image_location,'/..'),'recursive');
    disp(imset);
    [training_set,test_set] = imset.partition(m);
    test_set = test_set.partition(n);
end