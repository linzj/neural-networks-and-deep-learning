TENSORFLOWPATH=/media/linzj/normal/linzj/src/tensorflow
export PYTHONPATH="$TENSORFLOWPATH/bazel-bin/tensorflow/examples/learn/mnist.runfiles:$TENSORFLOWPATH/bazel-bin/tensorflow/examples/learn/mnist.runfiles/protobuf/python:$TENSORFLOWPATH/bazel-bin/tensorflow/examples/learn/mnist.runfiles/six_archive:$TENSORFLOWPATH/bazel-bin/tensorflow/examples/learn/mnist.runfiles/org_tensorflow:$TENSORFLOWPATH/bazel-bin/tensorflow/examples/learn/mnist.runfiles/protobuf"
python $*

