add_subdirectory(event_wrapper)
add_subdirectory(stream_wrapper)
add_subdirectory(graph_wrapper_simple)
add_subdirectory(graph_wrapper)

if(GCXX_CUDA_MODE)
add_subdirectory(graph_wrapper_conditional)
endif(GCXX_CUDA_MODE)

