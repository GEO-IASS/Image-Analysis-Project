function [training_set,test_set] = partition_data(dsObj, m, n)

    image_location = fileparts(dsObj.Files{1}); 
    imset = imageSet(strcat(image_location,'\..'),'recursive');
    [training_set,test_set] = imset.partition(m);
    test_set = test_set.partition(n);
end