import matlab_wrapper
matlab = matlab_wrapper.MatlabSession()

matlab.eval("cd('/home/harrysocool/Github/fast-rcnn/OP_methods/edges')")
matlab.eval("addpath(genpath('/home/harrysocool/Github/fast-rcnn/OP_methods/edges'))")
# matlab.eval("toolboxCompile")
matlab.eval("res = edge_detector_demo(1,0)")
print(matlab.get('res'))