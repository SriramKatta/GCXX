// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Sriram Katta
#pragma once
#ifndef GCXX_RUNTIME_GRAPH_FWD_HPP_
#define GCXX_RUNTIME_GRAPH_FWD_HPP_

// forward declarations for the graph module's public types.
#include <gcxx/internal/prologue.hpp>

GCXX_NAMESPACE_MAIN_BEGIN()

class Graph;
class GraphView;
class GraphExec;
class GraphExecView;

class GraphNodeView;
class ChildGraphNodeView;
class KernelNodeView;
class MemcpyNodeView;
class MemsetNodeView;
class HostNodeView;
class EventRecordNodeView;
class EventWaitNodeView;
class MemAllocNodeView;
class MemFreeNodeView;
class ExternalSemaphoreSignalNodeView;
class ExternalSemaphoreWaitNodeView;

struct IfNodeResult;
struct IfElseNodeResult;
struct WhileNodeResult;
struct SwitchNodeResult;

GCXX_NAMESPACE_MAIN_END()

#endif
