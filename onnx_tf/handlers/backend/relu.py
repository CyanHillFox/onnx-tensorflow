import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func
from onnx_tf.common import supports_device


@onnx_op("Relu")
@tf_func(tf.nn.relu)
class Relu(BackendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    if not supports_device("CUDA"):
      kwargs["c_last_only"] = True
    return [cls.make_tensor_from_onnx_node(node, **kwargs)]

  @classmethod
  def version_6(cls, node, **kwargs):
    if not supports_device("CUDA"):
      kwargs["c_last_only"] = True
    return [cls.make_tensor_from_onnx_node(node, **kwargs)]
