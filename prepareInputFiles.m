function [tr, test] = prepareInputFiles(obj)
loc=fileparts(obj.Files{1});
imset = imageSet(strcat(loc,'\..'),'recursive');
[tr, test] = imset.partition([0.8 0.2]);
end