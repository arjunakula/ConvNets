function [] = setup()
run(fullfile(fileparts(mfilename('fullpath')), ...
  '..', 'matlab', 'vl_setupnn.m')) ;
vl_compilenn('EnableGPU', false);