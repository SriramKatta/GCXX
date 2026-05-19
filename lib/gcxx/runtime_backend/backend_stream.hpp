// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_BACKEND_BACKEND_STREAM_HPP_
#define GCXX_RUNTIME_BACKEND_BACKEND_STREAM_HPP_

#include <gcxx/internal/prologue.hpp>

#include <cstddef>

#include <gcxx/runtime_backend/backend_handles.hpp>

GCXX_NAMESPACE_MAIN_DRIVER_BEGIN

GCXX_FH auto ctxResetPersistingL2Cache() -> void {
  GCXX_SAFE_RUNTIME_CALL(CtxResetPersistingL2Cache,
                         "Failed to reset persisting L2 cache");
}

GCXX_FH auto streamAddCallback(deviceStream_t stream,
                               deviceStreamCallback_t callback, void* userData,
                               unsigned int flags) -> void {
  GCXX_SAFE_RUNTIME_CALL(StreamAddCallback, "Failed to add Stream callback",
                         stream, callback, userData, flags);
}

GCXX_FH auto streamAttachMemAsync(
  deviceStream_t stream, void* devPtr, std::size_t length = 0,
  unsigned int flags = GCXX_RUNTIME_BACKEND(MemAttachSingle)) -> void {
  GCXX_SAFE_RUNTIME_CALL(StreamAttachMemAsync,
                         "Failed to attach memory to Stream", stream, devPtr,
                         length, flags);
}

GCXX_FH auto streamBeginCapture(deviceStream_t stream,
                                deviceStreamCaptureMode_t mode) -> void {
  GCXX_SAFE_RUNTIME_CALL(StreamBeginCapture, "Failed to begin Stream Capture",
                         stream, mode);
}

GCXX_FH auto streamBeginCaptureToGraph(
  deviceStream_t stream, deviceGraph_t graph,
  const deviceGraphNode_t* dependencies,
  const deviceGraphEdgeData_t* dependencyData, std::size_t numDependencies,
  deviceStreamCaptureMode_t mode) -> void {
  GCXX_SAFE_RUNTIME_CALL(
    StreamBeginCaptureToGraph, "Failed to begin Stream Capture to graph",
    stream, graph, dependencies, dependencyData, numDependencies, mode);
}

GCXX_FH auto streamCopyAttributes(deviceStream_t dst,
                                  deviceStream_t src) -> void {
  GCXX_SAFE_RUNTIME_CALL(StreamCopyAttributes,
                         "Failed to copy Stream attributes", dst, src);
}

// TODO : needed for device side stream allocation
// GCXX_FD auto streamCreateWithFlags(unsigned int flags) -> deviceStream_t {
//   deviceStream_t stream{NULL_STREAM};
//   GCXX_SAFE_RUNTIME_CALL(StreamCreate, "Failed to Create Stream", &stream,
//                          flags);
//   return stream;
// }


GCXX_FH auto streamCreateWithPriority(unsigned int flags,
                                      int priority) -> deviceStream_t {
  deviceStream_t stream{NULL_STREAM};
  GCXX_SAFE_RUNTIME_CALL(StreamCreateWithPriority, "Failed to Create Stream",
                         &stream, flags, priority);
  return stream;
}

GCXX_FH auto streamDestroy(deviceStream_t stream) -> void {
  GCXX_SAFE_RUNTIME_CALL(StreamDestroy, "Failed to Destroy Stream", stream);
}

GCXX_FH auto streamEndCapture(deviceStream_t stream) -> deviceGraph_t {
  deviceGraph_t graph{INVALID_GRAPH};
  GCXX_SAFE_RUNTIME_CALL(StreamEndCapture, "Failed to end Stream Capture",
                         stream, &graph);
  return graph;
}

#if GCXX_CUDA_MODE
GCXX_FH auto streamGetAttribute(deviceStream_t stream,
                                deviceStreamAttrID_t attr,
                                deviceStreamAttrValue_t* valueOut) -> void {
  GCXX_SAFE_RUNTIME_CALL(StreamGetAttribute, "Failed to get Stream attribute",
                         stream, attr, valueOut);
}
#endif

GCXX_FH auto streamGetCaptureInfo(
  deviceStream_t stream, deviceStreamCaptureStatus_t* captureStatusOut,
  unsigned long long* idOut = nullptr, deviceGraph_t* graphOut = nullptr,
  const deviceGraphNode_t** dependenciesOut = nullptr,
  std::size_t* numDependenciesOut           = nullptr) -> void {
  GCXX_SAFE_RUNTIME_CALL(
    StreamGetCaptureInfo, "Failed to get Capture info of Stream", stream,
    captureStatusOut, idOut, graphOut, dependenciesOut, numDependenciesOut);
}

#if GCXX_CUDA_VERSION_GREATER_EQUAL(12, 3, 0)
GCXX_FH auto streamGetCaptureInfo_v3(
  deviceStream_t stream, deviceStreamCaptureStatus_t* captureStatusOut,
  unsigned long long* idOut = nullptr, deviceGraph_t* graphOut = nullptr,
  const deviceGraphNode_t** dependenciesOut = nullptr,
  const deviceGraphEdgeData_t** edgeDataOut = nullptr,
  std::size_t* numDependenciesOut           = nullptr) -> void {
  GCXX_SAFE_RUNTIME_CALL(StreamGetCaptureInfo_v3,
                         "Failed to get Capture info of Stream", stream,
                         captureStatusOut, idOut, graphOut, dependenciesOut,
                         edgeDataOut, numDependenciesOut);
}
#endif

GCXX_FH auto streamGetDevice(deviceStream_t stream) -> int {
  int device{};
  GCXX_SAFE_RUNTIME_CALL(StreamGetDevice, "Failed to get Stream device", stream,
                         &device);
  return device;
}

GCXX_FH auto streamGetFlags(deviceStream_t stream) -> unsigned int {
  unsigned int flags{};
  GCXX_SAFE_RUNTIME_CALL(StreamGetFlags, "Failed to get Stream flags", stream,
                         &flags);
  return flags;
}

GCXX_FH auto streamGetId(deviceStream_t stream) -> unsigned long long {
  unsigned long long streamId{};
  GCXX_SAFE_RUNTIME_CALL(StreamGetId, "Failed to get Stream Id", stream,
                         &streamId);
  return streamId;
}

GCXX_FH auto streamGetPriority(deviceStream_t stream) -> int {
  int priority{};
  GCXX_SAFE_RUNTIME_CALL(StreamGetPriority, "Failed to get Stream priority",
                         stream, &priority);
  return priority;
}

GCXX_FH auto streamIsCapturing(
  deviceStream_t stream, deviceStreamCaptureStatus_t* captureStatus) -> void {
  GCXX_SAFE_RUNTIME_CALL(StreamIsCapturing,
                         "Failed to query if Stream is capturing", stream,
                         captureStatus);
}

GCXX_FH auto streamQueryNothrow(deviceStream_t stream)
  -> driver::deviceError_t {
  return GCXX_RUNTIME_BACKEND(StreamQuery)(stream);
}

#if GCXX_CUDA_MODE
GCXX_FH auto streamSetAttribute(deviceStream_t stream,
                                deviceStreamAttrID_t attr,
                                const deviceStreamAttrValue_t* value) -> void {
  GCXX_SAFE_RUNTIME_CALL(StreamSetAttribute, "Failed to set Stream attribute",
                         stream, attr, value);
}
#endif


GCXX_FH auto streamSynchronize(deviceStream_t stream) -> void {
  GCXX_SAFE_RUNTIME_CALL(StreamSynchronize, "Failed to synchronize Stream",
                         stream);
}

GCXX_FH auto streamUpdateCaptureDependencies(
  deviceStream_t stream, deviceGraphNode_t* dependencies,
  const deviceGraphEdgeData_t* dependencyData, std::size_t numDependencies,
  unsigned int flags = 0) -> void {
  GCXX_SAFE_RUNTIME_CALL(
#if GCXX_CUDA_VERSION_LESS_THAN(13, 0, 0)
    StreamUpdateCaptureDependencies_v2,
#else
    StreamUpdateCaptureDependencies,
#endif
    "Failed to update Stream Capture dependencies", stream, dependencies,
#if !GCXX_HIP_MODE
    dependencyData,
#endif
    numDependencies, flags);
}

GCXX_FH auto StreamWaitEvent(deviceStream_t stream, deviceEvent_t event,
                             unsigned int waitFlag) -> void {
  GCXX_SAFE_RUNTIME_CALL(StreamWaitEvent,
                         "Failed to GPU Stream Wait on GPU Event", stream,
                         event, waitFlag);
}

GCXX_FH auto threadExchangeStreamCaptureMode(deviceStreamCaptureMode_t* mode)
  -> void {
  GCXX_SAFE_RUNTIME_CALL(ThreadExchangeStreamCaptureMode,
                         "Failed to exchange Stream Capture mode", mode);
}

GCXX_NAMESPACE_MAIN_DRIVER_END

#endif
