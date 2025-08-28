#ifndef GPUCXX_RUNTIME_LAUNCH_LAUNCH_CONFIG_HPP_
#define GPUCXX_RUNTIME_LAUNCH_LAUNCH_CONFIG_HPP_

#include <gpucxx/backend/backend.hpp>

class LaunchConfig{

    private:
        dim3 gridDim{1,1,1};
        dim3 blockDim{1,1,1};
}


#endif