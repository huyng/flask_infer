import os
import pprint
import tensorflow as tf
import json
from flask import Flask, request, Response, render_template
from utils import tf_load_saved_model, tf_run
from argparse import ArgumentParser

def ensure_json_serializable(value):
    """
    Recursively ensures all values passed in are json serializable
    """
    import numpy as np
    if isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, np.float):
        return float(value)
    elif isinstance(value, np.integer):
        return int(value)
    elif isinstance(value, dict):
        new_dict = {}
        for k,v in value.iteritems():
            new_dict[k] = ensure_json_serializable(v)
        return new_dict
    elif isinstance(value, list):
        new_list = []
        for element in value:
            new_list.append(ensure_json_serializable(element))
        return new_list
    else:
        return value


def serve(args):
    sess, inputs, outputs = tf_load_saved_model(args.model)
    tensor_dict = dict(inputs.items() + outputs.items())

    # model server
    this_dir = os.path.split(os.path.abspath(__file__))[0]
    app = Flask(__name__, template_folder=os.path.join(this_dir, "frontend"))

    @app.route("/")
    def index():
        input_keys = inputs.keys()
        output_keys = outputs.keys()
        return render_template("index.html")

    @app.route("/api", methods=["GET", "POST"])
    def api():
        if request.method == "POST":
            data = request.get_json()
        elif request.method == "GET":
            data = json.loads(request.args.get("data"))

        inputs = data.get("inputs")
        output_keys = data.get("outputs", outputs.keys())
        if inputs is None:
            return json.dumps({"error": "Missing required inputs parameter"})

        result = tf_run(sess, tensor_dict, outputs=output_keys, inputs=inputs)
        result = ensure_json_serializable(result)
        response = Response(response=json.dumps(result, indent=2),
                            status=200,
                            mimetype="application/json")

        return response

    app.run()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-m", "--model")
    args = parser.parse_args()
    serve(args)
