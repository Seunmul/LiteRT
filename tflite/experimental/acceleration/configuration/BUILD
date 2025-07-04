# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# buildifier: disable=out-of-order-load

# Placeholder: load proto_library
load("@flatbuffers//:build_defs.bzl", "DEFAULT_FLATC_ARGS", "flatbuffer_android_library", "flatbuffer_cc_library", "flatbuffer_java_library")
load("@org_tensorflow//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("//tflite:build_def.bzl", "tflite_copts", "tflite_copts_warnings")
load("//tflite:special_rules.bzl", "nnapi_plugin_impl_visibility_allowlist", "tflite_portable_test_suite")
load("//tflite/core/shims:cc_library_with_tflite.bzl", "cc_library_with_tflite")
# copybara:uncomment load("//tools/build_defs/proto/cpp:cc_proto_library.bzl", "cc_proto_library")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

# Generate a C++ header containing the contents of the FlatBuffer schema
# as a char array literal.  This is potentially useful for embedding in programs
# (e.g. for JSON parsing using that schema).
genrule(
    name = "configuration_fbs_contents_cc",
    srcs = ["configuration.fbs"],
    outs = ["configuration_fbs_contents-inl.h"],
    cmd = """
      echo 'constexpr char configuration_fbs_contents[] = R"Delimiter(' > $(@)
      cat < $(<) >> $(@)
      echo ')Delimiter";' >> $(@)
    """,
    compatible_with = get_compatible_with_portable(),
)

genrule(
    name = "configuration.fbs-backwards-compat-stub",
    srcs = ["//tflite/acceleration/configuration:configuration.fbs"],
    outs = ["configuration.fbs"],
    cmd = "cp $< $@",
    compatible_with = get_compatible_with_portable(),
)

proto_library(
    name = "configuration_proto",
    srcs = ["configuration.proto"],
    exports = ["//tflite/acceleration/configuration:configuration_proto"],
    deps = ["//tflite/acceleration/configuration:configuration_proto"],
)

alias(
    name = "configuration_java_proto_lite",
    actual = "//tflite/acceleration/configuration:configuration_java_proto_lite",
)

cc_proto_library(
    name = "configuration_cc_proto",
    deps = [":configuration_proto"],
)

flatbuffer_cc_library(
    name = "configuration_fbs",
    srcs = [":configuration.fbs"],
    compatible_with = get_compatible_with_portable(),
    flatc_args = DEFAULT_FLATC_ARGS + ["--gen-compare"],
)

flatbuffer_java_library(
    name = "configuration_fbs_java",
    srcs = [":configuration.fbs"],
)

flatbuffer_android_library(
    name = "configuration_fbs_android",
    srcs = [":configuration.fbs"],
)

cc_library(
    name = "proto_to_flatbuffer",
    hdrs = ["proto_to_flatbuffer.h"],
    deps = [
        "//tflite/acceleration/configuration:proto_to_flatbuffer",
    ],
)

cc_library(
    name = "flatbuffer_to_proto",
    hdrs = ["flatbuffer_to_proto.h"],
    deps = [
        "//tflite/acceleration/configuration:flatbuffer_to_proto",
    ],
)

cc_library_with_tflite(
    name = "delegate_registry",
    hdrs = ["delegate_registry.h"],
    compatible_with = get_compatible_with_portable(),
    tflite_deps = [
        "//tflite/acceleration/configuration:delegate_registry",
    ],
)

cc_library_with_tflite(
    name = "delegate_plugin_converter",
    hdrs = ["delegate_plugin_converter.h"],
    tflite_deps = [
        "//tflite/acceleration/configuration:delegate_plugin_converter",
    ],
)

cc_library(
    name = "nnapi_plugin",
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":nnapi_plugin_impl",
    ],
)

cc_library(
    name = "nnapi_plugin_impl",
    hdrs = ["nnapi_plugin.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = nnapi_plugin_impl_visibility_allowlist(),
    deps = [
        "//tflite/core/acceleration/configuration:nnapi_plugin",
    ],
)

cc_library(
    name = "hexagon_plugin",
    srcs = ["hexagon_plugin.cc"],
    deps = [
        ":configuration_fbs",
        "//tflite/core/acceleration/configuration:delegate_registry",
        "@com_google_absl//absl/memory",
    ] + select({
        "@platforms//cpu:aarch64": [
            "//tflite/delegates/hexagon:hexagon_delegate",
        ],
        "@platforms//cpu:armv7": [
            "//tflite/delegates/hexagon:hexagon_delegate",
        ],
        "//conditions:default": [],
    }),
    alwayslink = 1,  # For registration to always run.
)

cc_library(
    name = "gpu_plugin",
    deps = [
        ":gpu_plugin_impl",
    ],
)

common_copts = tflite_copts() + tflite_copts_warnings()

cc_library(
    name = "gpu_plugin_impl",
    hdrs = ["gpu_plugin.h"],
    copts = common_copts + select({
        "@org_tensorflow//tensorflow:ios": [
            "-xobjective-c++",
        ],
        "@org_tensorflow//tensorflow:macos_arm64": [
            "-xobjective-c++",
        ],
        "//conditions:default": [],
    }),
    deps = [
        "//tflite/acceleration/configuration:gpu_plugin_impl",
    ],
)

cc_library(
    name = "xnnpack_plugin",
    deps = [
        "//tflite/acceleration/configuration:xnnpack_plugin",
    ],
)

cc_library(
    name = "coreml_plugin",
    srcs = ["coreml_plugin.cc"],
    deps = [
        ":configuration_fbs",
        "//tflite:minimal_logging",
        "//tflite/core/acceleration/configuration:delegate_registry",
        "@com_google_absl//absl/memory",
    ] + select({
        "@org_tensorflow//tensorflow:macos": [
            "//tflite/delegates/coreml:coreml_delegate",
        ],
        "@org_tensorflow//tensorflow:ios": [
            "//tflite/delegates/coreml:coreml_delegate",
        ],
        "//conditions:default": [],
    }),
    alwayslink = 1,  # For registration to always run.
)

# TODO(b/260582614): Add support for TF Lite in Play Services.
cc_library(
    name = "stable_delegate_plugin",
    hdrs = ["stable_delegate_plugin.h"],
    deps = [
        "//tflite/acceleration/configuration:stable_delegate_plugin",
    ],
)

tflite_portable_test_suite()
