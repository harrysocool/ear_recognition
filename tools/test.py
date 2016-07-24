from pymatbridge import Matlab

# Initialise MATLAB
mlab = Matlab(matlab='/usr/local/bin/matlab', port=4000)
# Start the server
mlab.start()
# Run a test function: just adds 1 to the argument a
for i in range(10):
    print mlab.run('/home/harrysocool/Github/python-matlab-bridge/test.m', {'a': '/home/harrysocool/Github/fast-rcnn/tools', 'b':'/home/harrysocool/Github/fast-rcnn/tools'})['result']
# Stop the MATLAB server
mlab.stop()