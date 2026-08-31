from __future__ import annotations

import numpy as np
import onnx
from onnx import helper, numpy_helper
from onnx.utils import Extractor


def _add_int64(graph: onnx.GraphProto, name: str, value) -> str:
    graph.initializer.append(numpy_helper.from_array(np.asarray(value, dtype=np.int64), name))
    return name


def make_batch_dynamic(model: onnx.ModelProto, batch_dim: str = "batch") -> onnx.ModelProto:
    """Make an exported batch-1 graph accept a dynamic leading batch dimension.

    The rewrite handles constant ``Reshape`` targets and batch-1 constants
    concatenated along a non-batch axis. It mutates and returns ``model``.
    """

    graph = model.graph
    initializers = {value.name: value for value in graph.initializer}

    reshape_nodes: dict[str, list[onnx.NodeProto]] = {}
    for node in graph.node:
        if node.op_type != "Reshape" or node.input[1] not in initializers:
            continue
        shape = numpy_helper.to_array(initializers[node.input[1]])
        if shape.ndim == 1 and len(shape) and shape[0] == 1:
            reshape_nodes.setdefault(node.input[1], []).append(node)

    for name, nodes in reshape_nodes.items():
        shape = numpy_helper.to_array(initializers[name]).copy()
        shape[0] = 0 if np.any(shape[1:] == -1) else -1
        initializers[name].CopyFrom(numpy_helper.from_array(shape, name))
        for node in nodes:
            for attribute in node.attribute:
                if attribute.name == "allowzero":
                    attribute.i = 0

    nodes = []
    for node_index, node in enumerate(graph.node):
        if node.op_type == "Concat":
            axis = next(attribute.i for attribute in node.attribute if attribute.name == "axis")
            source = next((name for name in node.input if name not in initializers), None)
            if axis != 0 and source is not None:
                for input_index, name in enumerate(node.input):
                    if name not in initializers:
                        continue
                    value = numpy_helper.to_array(initializers[name])
                    if value.ndim < 2 or value.shape[0] != 1:
                        continue

                    prefix = f"_dynamic_batch_{node_index}_{input_index}"
                    batch_shape = f"{prefix}_batch_shape"
                    target_shape = f"{prefix}_target_shape"
                    expanded = f"{prefix}_expanded"
                    nodes.extend(
                        (
                            helper.make_node("Shape", [source], [batch_shape], name=f"{prefix}_batch", start=0, end=1),
                            helper.make_node(
                                "Concat",
                                [batch_shape, _add_int64(graph, f"{prefix}_tail", value.shape[1:])],
                                [target_shape],
                                name=f"{prefix}_target",
                                axis=0,
                            ),
                            helper.make_node("Expand", [name, target_shape], [expanded], name=f"{prefix}_expand"),
                        )
                    )
                    node.input[input_index] = expanded
        nodes.append(node)

    del graph.node[:]
    graph.node.extend(nodes)

    initializer_names = set(initializers)
    for value in (*graph.input, *graph.output):
        if value.name in initializer_names:
            continue
        dimensions = value.type.tensor_type.shape.dim
        if dimensions and dimensions[0].dim_value == 1:
            dimensions[0].dim_param = batch_dim

    del graph.value_info[:]
    return model


def _rename_value(graph: onnx.GraphProto, old: str, new: str) -> None:
    for node in graph.node:
        for names in (node.input, node.output):
            for index, name in enumerate(names):
                if name == old:
                    names[index] = new
    for values in (graph.input, graph.output, graph.value_info, graph.initializer):
        for value in values:
            if value.name == old:
                value.name = new


def _prune_initializers(graph: onnx.GraphProto) -> None:
    used = {name for node in graph.node for name in node.input}
    initializers = [value for value in graph.initializer if value.name in used]
    del graph.initializer[:]
    graph.initializer.extend(initializers)


def split_supercombo(model: onnx.ModelProto) -> tuple[onnx.ModelProto, onnx.ModelProto]:
    """Split a streaming Supercombo into dynamic-batch vision and policy graphs."""

    graph = model.graph
    producers = {output: node for node in graph.node for output in node.output}
    feature_concat = next(node for node in graph.node if "features_buffer" in node.input)
    current = producers[next(name for name in feature_concat.input if name != "features_buffer")].input[0]
    images = [value.name for value in graph.input if "img" in value.name]
    excluded = {*images, "features_buffer"}
    policy_inputs = [
        feature_concat.output[0],
        current,
        *(value.name for value in graph.input if value.name not in excluded),
    ]
    outputs = [value.name for value in graph.output]

    extractor = Extractor(model)
    vision = extractor.extract_model(images, [current])
    policies = extractor.extract_model(policy_inputs, outputs)

    policies.graph.input.remove(next(value for value in policies.graph.input if value.name == current))
    _rename_value(vision.graph, current, "features")
    _rename_value(policies.graph, feature_concat.output[0], "features")
    _add_int64(policies.graph, "_last_feature", -1)
    policies.graph.node.insert(
        0,
        helper.make_node("Gather", ["features", "_last_feature"], [current], axis=1, name="last_feature"),
    )

    vision.graph.name = "vision"
    policies.graph.name = "policies"
    policies.metadata_props.extend(model.metadata_props)
    for part in (vision, policies):
        make_batch_dynamic(part)
    features = next(value for value in policies.graph.input if value.name == "features")
    features.type.tensor_type.shape.dim[1].dim_param = "time"
    return vision, policies


def make_dense(policies: onnx.ModelProto) -> onnx.ModelProto:
    """Make a split policy graph return its dense training outputs."""

    graph = policies.graph
    initializers = {value.name: value for value in graph.initializer}
    producers = {output: node for node in graph.node for output in node.output}
    features = next(value for value in graph.input if value.name == "features")
    desire = next(value for value in graph.input if value.name == "desire_pulse")
    feature_dim = features.type.tensor_type.shape.dim[-1].dim_value
    desire_dim = desire.type.tensor_type.shape.dim[-1].dim_value
    desire_window = desire.type.tensor_type.shape.dim[1].dim_value

    temporal_selects = []
    for node in graph.node:
        if node.op_type != "Gather" or node.input[1] not in initializers:
            continue
        axis = next(attribute.i for attribute in node.attribute if attribute.name == "axis")
        index = numpy_helper.to_array(initializers[node.input[1]])
        if axis == 1 and index.ndim == 0 and index >= 0:
            temporal_selects.append(node)
    timesteps = int(numpy_helper.to_array(initializers[temporal_selects[0].input[1]])) + 1

    flat_shape = _add_int64(graph, "_dense_flat_shape", [-1, feature_dim])
    point_select = next(node for node in graph.node if node.op_type == "Gather" and node.input[0] == "features")
    for node in (point_select, *temporal_selects):
        node.op_type = "Reshape"
        node.input[:] = [node.input[0], flat_shape]
        del node.attribute[:]

    desire_reshape = next(node for node in graph.node if "desire_pulse" in node.input)
    for node in graph.node:
        producer = producers.get(node.input[0]) if node.input else None
        if node.op_type == "Reshape" and producer and any("desire" in name for name in producer.input):
            node.input[1] = _add_int64(graph, f"_dense_{node.output[0]}_shape", [-1, timesteps, feature_dim])

    desire_indices = _add_int64(
        graph,
        "_dense_desire_indices",
        np.arange(timesteps)[:, None] + np.arange(desire_window)[None],
    )
    dense_desire = "_dense_desire"
    node_index = next(index for index, node in enumerate(graph.node) if node.name == desire_reshape.name)
    graph.node.insert(
        node_index,
        helper.make_node("Gather", ["desire_pulse", desire_indices], [dense_desire], axis=1),
    )
    desire_reshape.input[0] = dense_desire
    shape_name = desire_reshape.input[1]
    initializers[shape_name].CopyFrom(
        numpy_helper.from_array(np.asarray([-1, desire_window * desire_dim], dtype=np.int64), shape_name)
    )

    output = graph.output[0]
    output_size = output.type.tensor_type.shape.dim[-1].dim_value
    flat_output = f"_{output.name}_flat"
    output_node = next(node for node in graph.node if output.name in node.output)
    output_node.output[:] = [flat_output if name == output.name else name for name in output_node.output]
    output_shape = _add_int64(graph, "_dense_output_shape", [-1, timesteps, output_size])
    graph.node.append(helper.make_node("Reshape", [flat_output, output_shape], [output.name], name="dense_outputs"))

    features.type.tensor_type.shape.dim[1].dim_value = timesteps
    desire.type.tensor_type.shape.dim[1].dim_value = desire_window + timesteps - 1
    output.CopyFrom(
        helper.make_tensor_value_info(
            output.name,
            output.type.tensor_type.elem_type,
            ["batch", timesteps, output_size],
        )
    )
    _prune_initializers(graph)
    return policies


__all__ = ["make_batch_dynamic", "make_dense", "split_supercombo"]
