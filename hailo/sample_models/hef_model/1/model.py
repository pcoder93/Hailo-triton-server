#===================================================================================================
#  ?                                           ABOUT
#  @author      : pcoder93
#  @email       : pranavpune93@gmail.com
#  @repo        : hailo-triton-server
#  @createdOn   : 05-02-2025
#  @description : Sample models.py for testing BLS with triton server python backend
#                 It can be used with hailo  with OpenVino and/or CUDA ONNX Execution provider.
#===================================================================================================
import gc
import json
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import triton_python_backend_utils as pb_utils
from hailo_platform import (
    HEF,
    ConfigureParams,
    Device,
    FormatType,
    HailoStreamInterface,
    InferVStreams,
    InputVStreamParams,
    InputVStreams,
    OutputVStreamParams,
    OutputVStreams,
    VDevice,
)


class TritonPythonModel:
    @staticmethod
    def auto_complete_config(auto_complete_model_config):
        inputs = [
            {
                "name": "image_batch__0",
                "data_type": "TYPE_UINT8",
                "dims": [-1, -1, 3],
            },
        ]
        outputs = [
            {
                "name": "scores__0",
                "data_type": "TYPE_FP32",
                "dims": [1],
            },
            {
                "name": "masks__1",
                "data_type": "TYPE_UINT8",
                "dims": [-1, -1, -1],
            },
        ]
        config = auto_complete_model_config.as_dict()
        input_names = []
        output_names = []
        for ip in config["input"]:
            input_names.append(ip["name"])
        for op in config["output"]:
            output_names.append(op["name"])
        for ip in inputs:
            if ip["name"] not in input_names:
                auto_complete_model_config.add_input(ip)
        for op in outputs:
            if op["name"] not in output_names:
                auto_complete_model_config.add_output(op)

        auto_complete_model_config.set_max_batch_size(8)

        return auto_complete_model_config

    def initialize(self, args):

        self.hailo_model: bool = False
        self.onnx_model: bool = False
        self.preprocess_params = {}
        self.postprocess_params = {}
        self.onnx_params = {}
        self.ort_providers = []
        self.ort_provider_options = []

        self.logger = pb_utils.Logger
        self.model_name = args["model_name"]
        self.model_version = args["model_version"]
        model_work_dir = Path(args["model_repository"]).joinpath(self.model_version)
        self.model_config = json.loads(args["model_config"])

        self.trt_output_meta = {}
        for trt_op in self.model_config["output"]:
            self.trt_output_meta[trt_op["name"]] = {
                "shape": tuple(trt_op["dims"]),
                "dtype": trt_op["data_type"],
            }
        # params from config.pbtxt
        self.parameters = self.model_config["parameters"]
        self.load_model_config_parameters(self.parameters)
        self.logger.log(f"Loaded Params: {self.parameters}")

        # load HEF
        if len(list(model_work_dir.rglob("*.hef"))) > 0:
            self.hailo_model = True
            hef_model_path = model_work_dir.joinpath("model.hef")
            self.logger.log_verbose(f"Loading HEF from {hef_model_path}")
            if hef_model_path.exists():
                self.hef = HEF(hef_model_path.as_posix())
            else:
                raise pb_utils.TritonModelException("HEF model not found in Model Path")
            self.hef_configure_params = ConfigureParams.create_from_hef(
                hef=self.hef,
                interface=HailoStreamInterface.PCIe,
            )
            # identify HEF inputs and outputs
            self.ls_hef_inputs = self.hef.get_input_vstream_infos()
            self._log_list_details("HEF Input VStream infos", self.ls_hef_inputs)
            self.ls_hef_outputs = self.hef.get_output_vstream_infos()
            self._log_list_details("HEF Output VStream infos", self.ls_hef_outputs)

        # load onnx model head
        ls_onnx_model_paths = list(model_work_dir.rglob("*.onnx"))
        if len(ls_onnx_model_paths) > 0:
            self.onnx_model = True
            if self.hailo_model:
                onnx_model_path = model_work_dir.joinpath("model_head.onnx")
            else:
                onnx_model_path = model_work_dir.joinpath("model.onnx")
            if onnx_model_path.exists():
                self.setup_ort()
                self.ort_session = ort.InferenceSession(
                    onnx_model_path,
                    providers=self.ort_providers,
                    provider_options=self.ort_provider_options,
                    sess_options=self.ort_session_options,
                )

                # identitfy onnx inputs and outputs
                self.ort_ls_inputs = self.ort_session.get_inputs()
                self._log_list_details("ONNX inputs", self.ort_ls_inputs)
                self.ort_ls_outputs = self.ort_session.get_outputs()
                self._log_list_details("ONNX outputs", self.ort_ls_outputs)
            else:
                raise pb_utils.TritonModelException(
                    "ONNX model not found in Model Path"
                )

        self.init_metrics()

    def execute(self, requests):
        responses = []
        for request in requests:
            model_input = {
                "raw": None,
                "pre_proc": None,
                "hef_ip": None,
                "hef_op": None,
                "onnx_ip": None,
                "onnx_op": None,
            }
            model_input["raw"] = pb_utils.get_input_tensor_by_name(
                request,
                "image_batch__0",
            ).as_numpy()
            self.logger.log_verbose(
                f"New request-{str(self.model_name).strip()}-{self._log_array_details(model_input['raw'])}"
            )
            # preprocess
            start_ns = time.time_ns()
            model_input = self.preprocess(model_input)
            self.logger.log_verbose(
                f"Prepocessed batch - {self._log_array_details(model_input['pre_proc'])}"
            )
            end_ns = time.time_ns()
            self.preprocess_metric.increment(end_ns - start_ns)

            # infer hailo
            if self.hailo_model:
                with VDevice() as target:
                    network_groups = target.configure(
                        self.hef, self.hef_configure_params
                    )
                    network_group = network_groups[0]
                    network_group_params = network_group.create_params()
                    input_vstreams_params = InputVStreamParams.make(
                        network_group, format_type=FormatType.FLOAT32
                    )
                    output_vstreams_params = OutputVStreamParams.make(
                        network_group, format_type=FormatType.FLOAT32
                    )
                    model_input = self.parse_input_for_hef(model_input)
                    self._log_dict_details("HEF Input", model_input["hef_ip"])
                    # infer hef
                    start_ns = time.time_ns()
                    with InferVStreams(
                        network_group, input_vstreams_params, output_vstreams_params
                    ) as infer_pipeline:
                        with network_group.activate(network_group_params):
                            model_input["hef_op"] = infer_pipeline.infer(
                                model_input["hef_ip"]
                            )
                    end_ns = time.time_ns()
                    self.hef_infer_metric.increment(end_ns - start_ns)

                self._log_dict_details("HEF Output", model_input["hef_op"])

            # infer onnx
            if self.onnx_model:
                model_input = self.parse_input_for_onnx(model_input)
                ls_ort_output_names = [op.name for op in self.ort_ls_outputs]
                self._log_dict_details("ONNX Input", model_input["onnx_ip"])
                start_ns = time.time_ns()
                model_input["onnx_op"] = self.ort_session.run(
                    ls_ort_output_names, model_input["onnx_ip"]
                )
                end_ns = time.time_ns()
                self.onnx_infer_metric.increment(end_ns - start_ns)

            start_ns = time.time_ns()
            model_output = self.postprocess(model_input)
            self._log_dict_details("ONNX Output", model_output)
            end_ns = time.time_ns()
            self.postprocess_metric.increment(end_ns - start_ns)

            # parse ort results to trt responses
            responses.append(self.parse_to_trt_format(model_output))

        return responses

    def finalize(self):
        self.logger.log("Finalize")
        # explicity delete metrics : https://github.com/triton-inference-server/python_backend?tab=readme-ov-file#custom-metrics
        for att, obj in self.__dict__.items():
            if att.endswith(("_metric", "metrics_family")):
                self.logger.log_verbose(f"Deleting - {att} - {obj}")
        del self.preprocess_metric
        del self.preprocess_metric_family
        del self.postprocess_metric
        del self.postprocess_metric_family
        if self.hailo_model:
            del self.hef_infer_metric
            del self.hef_infer_metric_family
        if self.onnx_model:
            del self.onnx_infer_metric
            del self.onnx_infer_metric_family
        for att, obj in self.__dict__.items():
            if isinstance(obj, (ort.InferenceSession, HEF)):
                self.logger.log_verbose(f"Deleting - {att} - {obj}")
                del obj

        gc.collect()

    def load_model_config_parameters(self, d_parameters: dict):
        if "preprocess_params" in d_parameters:
            self.preprocess_params = json.loads(
                d_parameters["preprocess_params"]["string_value"]
            )
        if "postprocess_params" in d_parameters:
            self.postprocess_params = json.loads(
                d_parameters["postprocess_params"]["string_value"]
            )
        if "py_onnx" in d_parameters:
            self.onnx_params = json.loads(d_parameters["py_onnx"]["string_value"])

    def setup_ort(self):
        config_providers = (
            self.onnx_params["providers"]
            if "providers" in self.onnx_params
            else ["OpenVINOExecutionProvider", "CUDAExecutionProvider"]
        )
        config_provider_options = (
            self.onnx_params["provider_options"]
            if "provider_options" in self.onnx_params
            else [{"device_type": "GPU"}, {}]
        )
        if "providers" in self.onnx_params:
            self.logger.log(
                f"Config requested ORT providers : {len(config_providers)}:{config_providers} | {len(config_provider_options)}:{config_provider_options}"
            )
        if config_provider_options:
            if len(config_providers) != len(config_provider_options):
                raise pb_utils.TritonModelException(
                    "length of Providers does not match the list of Provider options"
                )
        self.ort_session_options = None

        for pro, opt in zip(config_providers, config_provider_options):
            if (pro in ort.get_available_providers()) and (
                pro not in self.ort_providers
            ):
                self.ort_providers.insert(0, pro)
                self.ort_provider_options.insert(0, opt)
                if pro == "OpenVINOExecutionProvider":
                    self.ort_session_options = ort.SessionOptions()
                    self.ort_session_options.graph_optimization_level = (
                        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
                    )
        self.logger.log(
            f"Using ONNX Providers: {self.ort_providers} with options : {self.ort_provider_options}"
        )

    def init_metrics(self):
        self.preprocess_metric_family = pb_utils.MetricFamily(
            name="py_requests_preprocess_latency_ns",
            description="Cumulative time spent preprocessing requests",
            kind=pb_utils.MetricFamily.COUNTER,
        )
        self.preprocess_metric = self.preprocess_metric_family.Metric(
            labels={
                "model": self.model_name,
                "version": self.model_version,
            },
        )
        if self.hailo_model:
            self.hef_infer_metric_family = pb_utils.MetricFamily(
                name="py_requests_hef_infer_latency_ns",
                description="Cumulative time spent for hef inference",
                kind=pb_utils.MetricFamily.COUNTER,
            )
            self.hef_infer_metric = self.hef_infer_metric_family.Metric(
                labels={
                    "model": self.model_name,
                    "version": self.model_version,
                },
            )
        if self.onnx_model:
            self.onnx_infer_metric_family = pb_utils.MetricFamily(
                name="py_requests_onnx_infer_latency_ns",
                description="Cumulative time spent for onnx inference",
                kind=pb_utils.MetricFamily.COUNTER,
            )
            self.onnx_infer_metric = self.onnx_infer_metric_family.Metric(
                labels={
                    "model": self.model_name,
                    "version": self.model_version,
                },
            )
        self.postprocess_metric_family = pb_utils.MetricFamily(
            name="py_requests_postprocess_latency_ns",
            description="Cumulative time spent postprocessing requests",
            kind=pb_utils.MetricFamily.COUNTER,
        )
        self.postprocess_metric = self.postprocess_metric_family.Metric(
            labels={
                "model": self.model_name,
                "version": self.model_version,
            },
        )

    def override_params(self, d_params: dict):

        ls_params = ["preprocess_params", "postprocess_params"]
        for par in ls_params:
            if par in d_params:
                old_params = getattr(self, par)
                new_params = d_params[par]
                for k in new_params:
                    if k in old_params:
                        old_params[k] = new_params[k]

    def resize_input(self, pre_proc):
        batch = None
        if all([self.hailo_model, self.onnx_model]) or self.hailo_model:
            model_input_shape = self.hef.get_input_vstream_infos()[0].shape
            b, _, _, _ = pre_proc.shape
            h, w, c = model_input_shape
        elif self.onnx_model:
            model_input_shape = self.ort_ls_inputs[0].shape[1:]
            b, _, _, _ = pre_proc.shape
            c, h, w = model_input_shape
        ls_images = []
        if pre_proc.shape[1:] != model_input_shape:
            for im in pre_proc:
                im = cv2.resize(im, (w, h), cv2.INTER_AREA)
                ls_images.append(im)
            if len(ls_images) > 0:
                batch = np.stack(ls_images, axis=0)
        else:
            batch = pre_proc
        return batch

    def resize_output(self, raw, masks, apply_mask=False):
        b, h, w, c = raw.shape
        ls_images = []
        self._log_array_details(raw)
        self._log_array_details(masks)
        for img, mask in zip(raw, masks):
            io_map = cv2.resize(mask, (w, h), cv2.INTER_AREA)
            if apply_mask:
                io_map = cv2.applyColorMap(io_map, cv2.COLORMAP_JET)
                img = cv2.addWeighted(img, 0.5, io_map, 0.5, 0.5)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                ls_images.append(img)
            else:
                ls_images.append(io_map)
        return np.stack(ls_images, axis=0)

    def preprocess(self, d_input: dict):
        """
        BHWC uint8 -> BHWC f32
        """
        preproc_batch = d_input["raw"]
        preproc_batch = self.resize_input(preproc_batch)

        if self.preprocess_params["normalize"]:
            preproc_batch = preproc_batch / 255.0
        if self.preprocess_params["channel_first"]:
            preproc_batch = preproc_batch.transpose((0, 3, 1, 2))

        preproc_batch_dtype = np.float32
        if self.onnx_model:
            if str(self.ort_ls_inputs[0]).find("float16") > 0:
                preproc_batch_dtype = np.float16

        preproc_batch = preproc_batch.astype(preproc_batch_dtype)
        d_input["pre_proc"] = preproc_batch
        return d_input

    def parse_input_for_hef(self, d_input: dict):
        input_v_name = self.hef.get_input_vstream_infos()[0].name
        d_input["hef_ip"] = {input_v_name: d_input["pre_proc"]}
        return d_input

    def parse_input_for_onnx(self, d_input: dict):
        if self.onnx_model and not self.hailo_model:
            # Assuming there is only 1 input for ONNX only models
            ort_ip = self.ort_ls_inputs[0]
            d_input["onnx_ip"] = {ort_ip.name: d_input["pre_proc"]}
        else:
            d_onnx_ip = {}
            if len(d_input["hef_op"].keys()) == len(self.ort_ls_inputs):
                for _, hef_op_val in d_input["hef_op"].items():
                    hef_op_val = hef_op_val.transpose(
                        (0, 3, 1, 2)
                    )  # always chanel 1st for ONNX input
                    for ort_ip in self.ort_ls_inputs:
                        if tuple(ort_ip.shape[1:]) == tuple(hef_op_val.shape[1:]):
                            d_onnx_ip[ort_ip.name] = hef_op_val

                d_input["onnx_ip"] = d_onnx_ip
            else:
                raise pb_utils.TritonModelException(
                    "Number of HEF outputs do NOT match number of ONNX inputs"
                )
        return d_input

    def postprocess(self, d_input: dict):
        results = {}
        raw_ip = d_input["raw"]
        model_results = None
        if d_input["onnx_op"] is not None:
            model_results = d_input["onnx_op"]
        elif d_input["hef_op"] is not None:
            model_results = [v for _, v in d_input["hef_op"].items()]

        for val in model_results:
            if tuple(val.shape[1:]) == tuple([1]):
                if self.postprocess_params["scores"]["sigmoid"]:
                    val = self.sigmoid(val)
                results["scores"] = val
            else:
                if self.postprocess_params["masks"]:
                    if self.postprocess_params["masks"]["sigmoid"]:
                        val = self.sigmoid(val)
                    if self.postprocess_params["masks"]["denormalize"]:
                        val = val * 255.0
                    val = val.astype("uint8")
                    if self.postprocess_params["masks"]["transpose"]:
                        val = val.transpose((0, 2, 3, 1))  # channel last
                    if self.postprocess_params["masks"]["resize"]:
                        val = self.resize_output(
                            raw_ip, val, self.postprocess_params["masks"]["blend"]
                        )
                    results["masks"] = val
        return results

    def parse_to_trt_format(self, onnx_output: dict) -> list:
        ls_trt_tensors = []
        trt_op_meta = self.model_config["output"]

        for op in trt_op_meta:
            if op["name"] == "scores__0":
                ls_trt_tensors.append(
                    pb_utils.Tensor(
                        "scores__0",
                        onnx_output["scores"].astype(
                            pb_utils.triton_string_to_numpy(op["data_type"])
                        ),
                    ),
                )
            elif op["name"] == "masks__1":
                ls_trt_tensors.append(
                    pb_utils.Tensor(
                        "masks__1",
                        onnx_output["masks"].astype(
                            pb_utils.triton_string_to_numpy(op["data_type"])
                        ),
                    ),
                )
        return pb_utils.InferenceResponse(output_tensors=ls_trt_tensors)

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """
        Compute sigmoid activation.

        Args:
            x: Input value.

        Returns:
            float: Sigmoid activated value.
        """
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def _log_array_details(arr):
        if isinstance(arr, np.ndarray):
            return f"shape: {arr.shape} | dtype: {arr.dtype} | min:{arr.min()} | max:{arr.max()}"

    @staticmethod
    def _log_dict_details(name, response: dict):
        logger = pb_utils.Logger
        logger.log_verbose(f"Dict of {name}")
        for key, val in response.items():
            logger.log_verbose(f"       {key} - {val.shape} | {val.dtype}")

    @staticmethod
    def log_attributes(object):
        logger = pb_utils.Logger
        for idx, ip in enumerate(object):
            logger.log_verbose(f"{idx}:[{ip}]")
            for att in dir(ip):
                if not att.startswith("__"):
                    try:
                        logger.log_verbose(f"    {att}:    {getattr(ip, att)}")
                    except Exception:
                        pass

    def _log_list_details(self, name, response: list):
        self.logger.log_verbose(f"List of {name}")
        self.log_attributes(response)
