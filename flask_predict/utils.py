import tensorflow as tf

def tf_load_saved_model(model_dir, tags=None, signature=None):
    """
    Creates a tf.Session with a graph defined by the contents in a saved model
    and returns tensor_dict containing references to your input placeholders and
    output tensors.
    Args:
        model_dir:
            This is the path to the SavedModel export directory
        tags: [str]
            Because a saved model can contain multiple meta graphs, each graph
            is "tagged" during export time. You must specify the metagraph you
            want to load.
        signature:
            The name of the signature you want to load. What is a signature? It
            is data structure that defines the inputs and outputs of your graph.
            You can define multiple signatures per graph, and at save time you
            must name these signatures.
    Returns:
        session:
            A tf.Session pre-populated with your model graph and your model weights
        tensor_dict:
            A dictionary of {"tensor_name": tensor} pairs that you can later use in your
            sess.run calls.
    """

    # load graph and populate session
    graph = tf.Graph()
    sess = tf.Session(graph=graph)
    tags = [tf.saved_model.tag_constants.SERVING] if tags is None else tags
    metagraph_def = tf.saved_model.loader.load(sess, tags, model_dir)

    if signature is not None:
        signature_def = metagraph_def.signature_def[signature]
    else:
        signature_def = metagraph_def.signature_def.values()[0]

    # load inputs into tensor dict
    inputs = {}
    for name, tensor_info in signature_def.inputs.items():
        inputs[name] = sess.graph.get_tensor_by_name(tensor_info.name)

    # load outputs into tensor dict
    outputs = {}
    for name, tensor_info in signature_def.outputs.items():
        outputs[name] = sess.graph.get_tensor_by_name(tensor_info.name)

    return sess, inputs, outputs


def tf_run(sess, tensor_dict, outputs, inputs):
    """
    Convenience function to run a tensorflow graph where the outputs
    are returned as a dictionary mapping from the name of the tensor to the
    output value.
    Args:
        sess: tf.Session
            The tensorflow session to run
        tensor_dict: dict
            A dictionary of the form {"tensor_name": tensorflow_tensor_op}
        outputs: list
            A list of ["tensor_name_a", "tensor_name_b", ...] that corresponds to
            the tensor op in tensor_dict that you would like to output during the sess.run call
        inputs:
            A dictionary of the form {"tensor_name": feed_value}. For each key in inputs, we will
            look up the corresponding tensor in tensor_dict and construct a feed_dict whose key is
            that tensor and whose value is the feed_value.
    Returns:
        named_outputs: dict
            A dictionary of the following form, for every "tensor_name" in the requested `outputs`
            list.
            {"tensor_name": output_value}
    """
    output_ops = []
    feed_dict = {}

    # setup feed dict
    for name, v in inputs.items():
        feed_dict[tensor_dict[name]] = v

    # setup output list
    for name in outputs:
        output_ops.append(tensor_dict[name])

    output_vals = sess.run(output_ops, feed_dict=feed_dict)
    return dict(zip(outputs, output_vals))
