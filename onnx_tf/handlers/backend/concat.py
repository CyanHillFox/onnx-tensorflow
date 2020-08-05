import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func
from onnx_tf.common import get_data_format
from onnx_tf.common import get_perm_from_formats
from onnx_tf.common import supports_device


@onnx_op("Concat")
@tf_func(tf.concat)
class Concat(BackendHandler):

  @classmethod
  def _common(cls, node, **kwargs):
    inputs = [kwargs["tensor_dict"][inp] for inp in node.inputs]
    rank = len(inputs[0].get_shape())
    storage_format, compute_format = get_data_format(rank) if (rank >= 2 and rank <= 5) else ('', '')
    if storage_format == compute_format:
      return [cls.make_tensor_from_onnx_node(node, inputs=[inputs])]
    else:
      # Transpose from storage_format to compute_format and do concat.
      # NOTE: this solution is pretty dirty, since it is impossible to determine whether it is necessary to do transpose.
      # currently implementation only servers solving HOMA project.
      inputs = [tf.transpose(x, get_perm_from_formats(storage_format, compute_format)) for x in inputs]
      from copy import deepcopy
      attrs = deepcopy(node.attrs)
      # axis has to be converted along with layout transform.
      axis = attrs["axis"]
      axis = compute_format.index(storage_format[axis])
      attrs["axis"] = axis
      output = cls.make_tensor_from_onnx_node(node, inputs=[inputs], attrs=attrs)
      output = tf.transpose(output, get_perm_from_formats(compute_format, storage_format))
      return [output]

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_4(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_11(cls, node, **kwargs):
    return cls._common(node, **kwargs)
