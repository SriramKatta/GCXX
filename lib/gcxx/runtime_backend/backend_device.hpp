// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_BACKEND_BACKEND_DEVICE_HPP_
#define GCXX_RUNTIME_BACKEND_BACKEND_DEVICE_HPP_

#include <gcxx/internal/prologue.hpp>

#include <cstddef>

#include <gcxx/runtime_backend/backend_handles.hpp>

GCXX_NAMESPACE_MAIN_DRIVER_BEGIN
GCXX_FH auto chooseDevice(const deviceProp_t& prop) -> int {
  int device{-1};
  GCXX_SAFE_RUNTIME_CALL(ChooseDevice, "Failed to choose device", &device,
                         &prop);
  return device;
}

GCXX_FH auto deviceFlushGPUDirectRDMAWrites() -> void {
  // TODO : to be filed later
}

GCXX_FH auto deviceGetAttribute(deviceAttribute_t attr, int device) -> int {
  int value{-1};
  GCXX_SAFE_RUNTIME_CALL(DeviceGetAttribute, "Failed to get device attribute",
                         &value, attr, device);
  return value;
}

GCXX_FH auto deviceGetByPCIBusId(const char* pciBusId) -> int {
  int device{-1};
  GCXX_SAFE_RUNTIME_CALL(DeviceGetByPCIBusId,
                         "Failed to get device by PCI bus ID", &device,
                         pciBusId);
  return device;
}

GCXX_FH auto deviceGetCacheConfig(int device) -> funcCacheConfig_t {
  funcCacheConfig_t config{};
  GCXX_SAFE_RUNTIME_CALL(DeviceGetCacheConfig,
                         "Failed to get device cache configuration", &config);
  return config;
}

GCXX_FH auto deviceGetDefaultMemoryPool(int device) -> deviceMemPool_t {
  deviceMemPool_t pool{};
  GCXX_SAFE_RUNTIME_CALL(DeviceGetDefaultMemPool,
                         "Failed to get device default memory pool", &pool,
                         device);
  return pool;
}

GCXX_FH auto deviceGetLimits(deviceLimit_t lim) -> size_t {
  size_t pVal{};
  GCXX_SAFE_RUNTIME_CALL(DeviceGetLimit, "Failed to get device limit", &pVal,
                         lim);
  return pVal;
}

GCXX_FH auto getMemPool(int device) -> deviceMemPool_t {
  deviceMemPool_t pool{};
  GCXX_SAFE_RUNTIME_CALL(DeviceGetMemPool, "Failed to get memory pool", &pool,
                         device);
  return pool;
}

GCXX_FH auto deviceGetP2PAttribute(deviceP2PAttr_t attr, int srcDevice,
                                   int dstDevice) -> int {
  int value{-1};
  GCXX_SAFE_RUNTIME_CALL(DeviceGetP2PAttribute,
                         "Failed to get device P2P attribute", &value, attr,
                         srcDevice, dstDevice);
  return value;
}

GCXX_FH auto deviceGetPCIBusId(int len, int device) -> char* {
  char* pciBusId = nullptr;
  GCXX_SAFE_RUNTIME_CALL(DeviceGetPCIBusId, "Failed to get device PCI bus ID",
                         pciBusId, len, device);
  return pciBusId;
}

GCXX_FH auto deviceGetStreamPriorityRange() -> std::pair<int, int> {
  int leastPriority{};
  int greatestPriority{};
  GCXX_SAFE_RUNTIME_CALL(DeviceGetStreamPriorityRange,
                         "Failed to get device stream priority range",
                         &leastPriority, &greatestPriority);
  return {leastPriority, greatestPriority};
}

GCXX_FH auto deviceReset() -> void {
  GCXX_SAFE_RUNTIME_CALL(DeviceReset, "Failed to reset device");
}

GCXX_FH auto deviceSetCacheConfig(funcCacheConfig_t config) -> void {
  GCXX_SAFE_RUNTIME_CALL(DeviceSetCacheConfig,
                         "Failed to set device cache configuration", config);
}

GCXX_FH auto deviceSetLimit(deviceLimit_t lim, size_t value) -> void {
  GCXX_SAFE_RUNTIME_CALL(DeviceSetLimit, "Failed to set device limit", lim,
                         value);
}

GCXX_FH auto deviceSetMemPool(int device, const deviceMemPool_t& pool) -> void {
  GCXX_SAFE_RUNTIME_CALL(DeviceSetMemPool, "Failed to set memory pool", device,
                         pool);
}

GCXX_FH auto deviceSync() -> void {
  GCXX_SAFE_RUNTIME_CALL(DeviceSynchronize, "Failed to synchronize device");
}

GCXX_FH auto deviceGet() -> int {
  int device{-1};
  GCXX_SAFE_RUNTIME_CALL(GetDevice, "Failed to get current device", &device);
  return device;
}

GCXX_FH auto deviceGetCount() -> int {
  int count{-1};
  GCXX_SAFE_RUNTIME_CALL(GetDeviceCount, "Failed to get device count", &count);
  return count;
}

GCXX_FH auto deviceGetFlags() -> unsigned int {
  unsigned int flags{};
  GCXX_SAFE_RUNTIME_CALL(GetDeviceFlags, "Failed to get device flags", &flags);
  return flags;
}

GCXX_FH auto deviceGetProp(int device) -> deviceProp_t {
  deviceProp_t prop{};
  GCXX_SAFE_RUNTIME_CALL(GetDeviceProperties, "Failed to get device properties",
                         &prop, device);
  return prop;
}

GCXX_FH auto deviceInit(int device, unsigned int deviceFlags,
                        unsigned int flags) -> void {
  GCXX_SAFE_RUNTIME_CALL(InitDevice, "Failed to initialize device", device,
                         deviceFlags, flags);
}

GCXX_FH auto ipcCloseMemHandle(void* devPtr) -> void {
  GCXX_SAFE_RUNTIME_CALL(IpcCloseMemHandle, "Failed to close IPC memory handle",
                         devPtr);
}

GCXX_FH auto ipcGetEventHandle(deviceEvent_t event) -> deviceIpcEventHandle_t {
  deviceIpcEventHandle_t ipcHandle{};
  GCXX_SAFE_RUNTIME_CALL(IpcGetEventHandle, "Failed to get IPC event handle",
                         &ipcHandle, event);
  return ipcHandle;
}

GCXX_FH auto ipcGetMemHandle(void* devPtr) -> deviceIpcMemHandle_t {
  deviceIpcMemHandle_t ipcHandle{};
  GCXX_SAFE_RUNTIME_CALL(IpcGetMemHandle, "Failed to get IPC memory handle",
                         &ipcHandle, devPtr);
  return ipcHandle;
}

GCXX_FH auto ipcOpeEventHandle(const deviceIpcEventHandle_t& ipcHandle)
  -> deviceEvent_t {
  deviceEvent_t event{};
  GCXX_SAFE_RUNTIME_CALL(IpcOpenEventHandle, "Failed to open IPC event handle",
                         &event, ipcHandle);
  return event;
}

GCXX_FH auto ipcOpenMemHandle(const deviceIpcMemHandle_t& ipcHandle,
                              unsigned int flags) -> void* {
  void* devPtr{};
  GCXX_SAFE_RUNTIME_CALL(IpcOpenMemHandle, "Failed to open IPC memory handle",
                         &devPtr, ipcHandle, flags);
  return devPtr;
}

GCXX_FH auto deviceSet(int device) -> void {
  GCXX_SAFE_RUNTIME_CALL(SetDevice, "Failed to set current device", device);
}

GCXX_FH auto deviceSetFlags(unsigned int flags) -> void {
  GCXX_SAFE_RUNTIME_CALL(SetDeviceFlags, "Failed to set current device", flags);
}

GCXX_FH auto deviceSetValid(int* dev_arr, int len) -> void {
  GCXX_SAFE_RUNTIME_CALL(SetValidDevices, "Failed to set valid devices",
                         dev_arr, len);
}

GCXX_NAMESPACE_MAIN_DRIVER_END

#endif
