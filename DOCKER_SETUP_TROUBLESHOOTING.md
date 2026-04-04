# Isaac Lab Docker Setup - Complete Troubleshooting Guide

This document provides a co# Fix services.pip_archive packaging conflict  
sudo docker exec -it isaac-lab-base-gr1 bash -c "cd /workspace/isaaclab && rm -f _isaac_sim/extscache/omni.services.pip_archive-*/pip_prebundle/packaging/_structures.py && cp ./.venv/lib/python3.11/site-packages/pip/_vendor/packaging/_structures.py _isaac_sim/extscache/omni.services.pip_archive-*/pip_prebundle/packaging/"rehensive guide to setting### 9. OSQP Array API Compatibility Issue
**Problem**: `AttributeError: _ARRAY_API not found` when importing Pink IK tasks

**Root Cause**: The OSQP solver package in Isaac Sim's environment has compatibility issues with newer versions of dependent packages, specifically the `_ARRAY_API` attribute is missing from the OSQP module.

**Solution**:
Install a compatible version of OSQP that works with Isaac Sim:
```bash
# Uninstall conflicting OSQP version
sudo docker exec -it isaac-lab-base-gr1 bash -c "cd /workspace/isaaclab && ./isaaclab.sh -p -m pip uninstall osqp -y"

# Install compatible OSQP version
sudo docker exec -it isaac-lab-base-gr1 bash -c "cd /workspace/isaaclab && ./isaaclab.sh -p -m pip install osqp==0.6.3"
```

**Alternative Solution** (if the above doesn't work):
Use a different QP solver that doesn't have these compatibility issues:
```bash
# Install alternative solvers and force qpsolvers to use a different default
sudo docker exec -it isaac-lab-base-gr1 bash -c "cd /workspace/isaaclab && ./isaaclab.sh -p -m pip install cvxopt"
sudo docker exec -it isaac-lab-base-gr1 bash -c "cd /workspace/isaaclab && ./isaaclab.sh -p -m pip install quadprog"

# Or completely disable OSQP by removing it
sudo docker exec -it isaac-lab-base-gr1 bash -c "cd /workspace/isaaclab && ./isaaclab.sh -p -m pip uninstall osqp -y"
```in Docker with full GR1T2 and Pink IK support.

## 🎯 Overview

Isaac Lab Docker setup involves multiple dependencies and package conflicts that need to be resolved in a specific order. This guide documents all the issues encountered and their solutions.

## 🚨 Problems Solved

### 1. Missing Docker Image
**Problem**: `docker: Error response from daemon: pull access denied for isaac-lab-base, repository does not exist`

**Root Cause**: The Isaac Lab base Docker image needed to be built locally.

**Solution**:
```bash
cd /path/to/IsaacLab/docker
docker-compose --env-file .env.base --profile base build isaac-lab-base
```

### 2. Docker Mount Path Errors
**Problem**: `invalid mount config for type "bind": bind source path does not exist`

**Root Cause**: Mount paths were relative to the wrong directory.

**Solution**:
When running from the main IsaacLab directory, use:
```bash
--mount type=bind,src=$(pwd)/openxr,dst=/openxr
--mount type=bind,src=$(pwd)/source,dst=/workspace/isaaclab/source  
--mount type=bind,src=$(pwd)/scripts,dst=/workspace/isaaclab/scripts
```

When running from the docker directory, use:
```bash
--mount type=bind,src=$(pwd)/../openxr,dst=/openxr
--mount type=bind,src=$(pwd)/../source,dst=/workspace/isaaclab/source
--mount type=bind,src=$(pwd)/../scripts,dst=/workspace/isaaclab/scripts
```

### 3. Missing Isaac Lab Packages
**Problem**: `ModuleNotFoundError: No module named 'isaaclab'`

**Root Cause**: Isaac Lab packages were not installed in the container's Python environment.

**Solution**:
```bash
sudo docker exec -it isaac-lab-base-gr1 bash -c "cd /workspace/isaaclab && ./isaaclab.sh -p -m pip install -e source/isaaclab --no-deps"
sudo docker exec -it isaac-lab-base-gr1 bash -c "cd /workspace/isaaclab && ./isaaclab.sh -p -m pip install -e source/isaaclab_tasks --no-deps"
```

### 4. PyTorch Packaging Conflict
**Problem**: `ModuleNotFoundError: No module named 'torch._vendor.packaging._structures'`

**Root Cause**: PyTorch's vendored packaging module was missing the `_structures.py` file due to version conflicts.

**Solution**:
```bash
# Fix PyTorch packaging conflict
sudo docker exec -it isaac-lab-base-gr1 bash -c "cd /workspace/isaaclab && rm -f _isaac_sim/exts/omni.isaac.ml_archive/pip_prebundle/torch/_vendor/packaging/_structures.py && cp ./.venv/lib/python3.11/site-packages/pip/_vendor/packaging/_structures.py _isaac_sim/exts/omni.isaac.ml_archive/pip_prebundle/torch/_vendor/packaging/"

# Fix services.pip_archive packaging conflict  
    sudo docker exec -it isaac-lab-base-gr1 bash -c "cd /workspace/isaaclab && cp ./.venv/lib/python3.11/site-packages/pip/_vendor/packaging/_structures.py _isaac_sim/extscache/omni.services.pip_archive-*/pip_prebundle/packaging/"
```

### 5. Package Version Conflicts
**Problem**: Multiple package version incompatibilities causing import errors.

**Root Cause**: Newer versions of core packages were incompatible with Isaac Sim's requirements.

**Solution**:
```bash
# Downgrade packaging to be compatible with Isaac Sim
sudo docker exec -it isaac-lab-base-gr1 bash -c "cd /workspace/isaaclab && ./isaaclab.sh -p -m pip install 'packaging<24'"

# Downgrade setuptools to include pkg_resources
sudo docker exec -it isaac-lab-base-gr1 bash -c "cd /workspace/isaaclab && ./isaaclab.sh -p -m pip install 'setuptools<70' --force-reinstall"

# Downgrade isort for pink library compatibility
sudo docker exec -it isaac-lab-base-gr1 bash -c "cd /workspace/isaaclab && ./isaaclab.sh -p -m pip install 'isort<6'"
```

### 6. Missing Dependencies
**Problem**: Various missing Python packages for different components.

**Solution**:
```bash
# Install missing packages
sudo docker exec -it isaac-lab-base-gr1 bash -c "cd /workspace/isaaclab && ./isaaclab.sh -p -m pip install einops"
sudo docker exec -it isaac-lab-base-gr1 bash -c "cd /workspace/isaaclab && ./isaaclab.sh -p -m pip install flatdict==4.0.1 --no-build-isolation"
```

### 7. Wrong Pink Library Installed
**Problem**: `ModuleNotFoundError: No module named 'pink.tasks'; 'pink' is not a package`

**Root Cause**: The wrong `pink` library was installed (a code formatting tool instead of the robotics IK library).

**Solution**:
```bash
# Uninstall wrong pink library
sudo docker exec -it isaac-lab-base-gr1 bash -c "cd /workspace/isaaclab && ./isaaclab.sh -p -m pip uninstall pink -y"

# Install correct Pink IK library
sudo docker exec -it isaac-lab-base-gr1 bash -c "cd /workspace/isaaclab && ./isaaclab.sh -p -m pip install pin-pink==3.1.0"
```

### 8. Build Isolation Issues
**Problem**: `ModuleNotFoundError: No module named 'pkg_resources'`

**Root Cause**: Package build isolation prevented access to necessary build dependencies.

**Solution**:
Use the `--no-build-isolation` flag when installing problematic packages:
```bash
sudo docker exec -it isaac-lab-base-gr1 bash -c "cd /workspace/isaaclab && ./isaaclab.sh -p -m pip install flatdict==4.0.1 --no-build-isolation"
```

### 9. OSQP Array API Compatibility Issue
**Problem**: `AttributeError: _ARRAY_API not found` when importing Pink IK tasks

**Root Cause**: The OSQP solver package in Isaac Sim's environment has compatibility issues with newer versions of dependent packages.

**Solution**:
Install a compatible version of OSQP that works with Isaac Sim:
```bash
# Uninstall conflicting OSQP version
sudo docker exec -it isaac-lab-base-gr1 bash -c "cd /workspace/isaaclab && ./isaaclab.sh -p -m pip uninstall osqp -y"

# Install compatible OSQP version
sudo docker exec -it isaac-lab-base-gr1 bash -c "cd /workspace/isaaclab && ./isaaclab.sh -p -m pip install osqp==0.6.3""
```

## 🚀 Complete Working Solution

### 1. Build Isaac Lab Docker Image
```bash
cd /path/to/IsaacLab/docker
docker-compose --env-file .env.base --profile base build isaac-lab-base
```

### 2. Run Isaac Lab Container
```bash
cd /path/to/IsaacLab
sudo docker run -it --rm \
  --name isaac-lab-base-gr1 \
  --gpus all --network host \
  -e "ACCEPT_EULA=Y" \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e NVIDIA_VISIBLE_DEVICES=all \
  --mount type=bind,src=$(pwd)/openxr,dst=/openxr \
  --mount type=bind,src=$(pwd)/source,dst=/workspace/isaaclab/source \
  --mount type=bind,src=$(pwd)/scripts,dst=/workspace/isaaclab/scripts \
  -e XDG_RUNTIME_DIR=/openxr/run \
  -e XR_RUNTIME_JSON=/openxr/share/openxr/1/openxr_cloudxr.json \
  isaac-lab-base:latest
```

### 3. Fix Package Dependencies (Run these commands in order)
```bash
# Update pip and core packages
sudo docker exec -it isaac-lab-base-gr1 bash -c "cd /workspace/isaaclab && ./isaaclab.sh -p -m pip install --upgrade pip setuptools wheel"

# Downgrade conflicting packages
sudo docker exec -it isaac-lab-base-gr1 bash -c "cd /workspace/isaaclab && ./isaaclab.sh -p -m pip install 'packaging<24'"
sudo docker exec -it isaac-lab-base-gr1 bash -c "cd /workspace/isaaclab && ./isaaclab.sh -p -m pip install 'setuptools<70' --force-reinstall"
sudo docker exec -it isaac-lab-base-gr1 bash -c "cd /workspace/isaaclab && ./isaaclab.sh -p -m pip install 'isort<6'"

# Install Isaac Lab packages
sudo docker exec -it isaac-lab-base-gr1 bash -c "cd /workspace/isaaclab && ./isaaclab.sh -p -m pip install -e source/isaaclab --no-deps"
sudo docker exec -it isaac-lab-base-gr1 bash -c "cd /workspace/isaaclab && ./isaaclab.sh -p -m pip install -e source/isaaclab_tasks --no-deps"

# Fix PyTorch packaging conflict
sudo docker exec -it isaac-lab-base-gr1 bash -c "cd /workspace/isaaclab && rm -f _isaac_sim/exts/omni.isaac.ml_archive/pip_prebundle/torch/_vendor/packaging/_structures.py && cp ./.venv/lib/python3.11/site-packages/pip/_vendor/packaging/_structures.py _isaac_sim/exts/omni.isaac.ml_archive/pip_prebundle/torch/_vendor/packaging/"

# Fix services.pip_archive packaging conflict  
sudo docker exec -it isaac-lab-base-gr1 bash -c "cd /workspace/isaaclab && rm -f _isaac_sim/extscache/omni.services.pip_archive-*/pip_prebundle/packaging/_structures.py && cp ./.venv/lib/python3.11/site-packages/pip/_vendor/packaging/_structures.py _isaac_sim/extscache/omni.services.pip_archive-*/pip_prebundle/packaging/"

# Install missing dependencies
sudo docker exec -it isaac-lab-base-gr1 bash -c "cd /workspace/isaaclab && ./isaaclab.sh -p -m pip install einops"
sudo docker exec -it isaac-lab-base-gr1 bash -c "cd /workspace/isaaclab && ./isaaclab.sh -p -m pip install flatdict==4.0.1 --no-build-isolation"

# Install correct Pink IK library
sudo docker exec -it isaac-lab-base-gr1 bash -c "cd /workspace/isaaclab && ./isaaclab.sh -p -m pip uninstall pink -y"
sudo docker exec -it isaac-lab-base-gr1 bash -c "cd /workspace/isaaclab && ./isaaclab.sh -p -m pip install pin-pink==3.1.0"

# Fix OSQP compatibility issue
sudo docker exec -it isaac-lab-base-gr1 bash -c "cd /workspace/isaaclab && ./isaaclab.sh -p -m pip uninstall osqp -y"
sudo docker exec -it isaac-lab-base-gr1 bash -c "cd /workspace/isaaclab && ./isaaclab.sh -p -m pip install osqp==0.6.3"

# If OSQP still causes issues, install alternative solvers
sudo docker exec -it isaac-lab-base-gr1 bash -c "cd /workspace/isaaclab && ./isaaclab.sh -p -m pip install cvxopt quadprog"
```

## ✅ Verification

Test that everything works:
```bash
# Test basic functionality
sudo docker exec -it isaac-lab-base-gr1 bash -c "cd /workspace/isaaclab && ./isaaclab.sh -p scripts/environments/teleoperation/eval.py --headless --help"

# Test with Franka robot
sudo docker exec -it isaac-lab-base-gr1 bash -c "cd /workspace/isaaclab && ./isaaclab.sh -p scripts/environments/teleoperation/eval.py --robot franka --headless"

# Test with GR1T2 robot
sudo docker exec -it isaac-lab-base-gr1 bash -c "cd /workspace/isaaclab && ./isaaclab.sh -p scripts/environments/teleoperation/eval.py --robot gr1t2 --headless"
```

## 🎯 Available Environments

After successful setup, the following environments are available:
- ✅ `Isaac-PickPlace-Franka-custom`
- ✅ `Isaac-PickPlace-GR1T2-Abs-v0`
- ✅ `Isaac-NutPour-GR1T2-Pink-IK-Abs-v0`
- ✅ `Isaac-ExhaustPipe-GR1T2-Pink-IK-Abs-v0`
- ✅ `Isaac-PickPlace-GR1T2-WaistEnabled-Abs-v0`

## 🔧 Key Lessons Learned

1. **Build Order Matters**: Always build the base image before trying to run containers
2. **Path Awareness**: Mount paths are relative to where you run the docker command
3. **Package Dependencies**: Isaac Sim has specific version requirements that conflict with newer packages
4. **Build Isolation**: Use `--no-deps` and `--no-build-isolation` flags to avoid dependency resolution issues
5. **Library Naming**: Multiple libraries can have similar names - ensure you install the correct one
6. **Version Pinning**: Stick to exact versions specified in Isaac Lab requirements

## 🛠️ Permanent Solution: Custom Docker Image

The most reliable permanent solution is to create a custom Docker image with all fixes pre-applied. This eliminates the need to run fix commands every time you start a container.

### Create a Custom Dockerfile

Create a file called `Dockerfile.isaac-lab-fixed` in your Isaac Lab directory:

```dockerfile
FROM isaac-lab-base:latest

# Set working directory
WORKDIR /workspace/isaaclab

# Fix packaging conflicts permanently
RUN rm -f _isaac_sim/exts/omni.isaac.ml_archive/pip_prebundle/torch/_vendor/packaging/_structures.py && \
    cp ./.venv/lib/python3.11/site-packages/pip/_vendor/packaging/_structures.py _isaac_sim/exts/omni.isaac.ml_archive/pip_prebundle/torch/_vendor/packaging/ && \
    rm -f _isaac_sim/extscache/omni.services.pip_archive-*/pip_prebundle/packaging/_structures.py && \
    cp ./.venv/lib/python3.11/site-packages/pip/_vendor/packaging/_structures.py _isaac_sim/extscache/omni.services.pip_archive-*/pip_prebundle/packaging/

# Install and configure packages with proper versions
RUN ./isaaclab.sh -p -m pip install 'packaging<24' && \
    ./isaaclab.sh -p -m pip install 'setuptools<70' --force-reinstall && \
    ./isaaclab.sh -p -m pip install 'isort<6' && \
    ./isaaclab.sh -p -m pip install einops && \
    ./isaaclab.sh -p -m pip install flatdict==4.0.1 --no-build-isolation

# Install correct Pink IK library
RUN ./isaaclab.sh -p -m pip uninstall pink -y || true && \
    ./isaaclab.sh -p -m pip install pin-pink==3.1.0

# Fix OSQP compatibility
RUN ./isaaclab.sh -p -m pip uninstall osqp -y || true && \
    ./isaaclab.sh -p -m pip install osqp==0.6.3

# Install Isaac Lab packages
RUN ./isaaclab.sh -p -m pip install -e source/isaaclab --no-deps && \
    ./isaaclab.sh -p -m pip install -e source/isaaclab_tasks --no-deps

# Set default command
CMD ["/bin/bash"]
```

### Build the Fixed Image

```bash
cd /path/to/IsaacLab
docker build -f Dockerfile.isaac-lab-fixed -t isaac-lab-fixed:latest .
```

### Use the Fixed Image

```bash
sudo docker run -it --rm \
  --name isaac-lab-fixed \
  --gpus all --network host \
  -e "ACCEPT_EULA=Y" \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e NVIDIA_VISIBLE_DEVICES=all \
  --mount type=bind,src=$(pwd)/openxr,dst=/openxr \
  --mount type=bind,src=$(pwd)/source,dst=/workspace/isaaclab/source \
  --mount type=bind,src=$(pwd)/scripts,dst=/workspace/isaaclab/scripts \
  -e XDG_RUNTIME_DIR=/openxr/run \
  -e XR_RUNTIME_JSON=/openxr/share/openxr/1/openxr_cloudxr.json \
  isaac-lab-fixed:latest
```

### Alternative: Docker Compose Solution

Create a `docker-compose.fixed.yaml`:

```yaml
version: '3.8'

services:
  isaac-lab-fixed:
    build:
      context: .
      dockerfile: Dockerfile.isaac-lab-fixed
    image: isaac-lab-fixed:latest
    container_name: isaac-lab-fixed
    runtime: nvidia
    network_mode: host
    stdin_open: true
    tty: true
    environment:
      - ACCEPT_EULA=Y
      - DISPLAY=${DISPLAY}
      - NVIDIA_DRIVER_CAPABILITIES=all
      - NVIDIA_VISIBLE_DEVICES=all
      - XDG_RUNTIME_DIR=/openxr/run
      - XR_RUNTIME_JSON=/openxr/share/openxr/1/openxr_cloudxr.json
    volumes:
      - /tmp/.X11-unix:/tmp/.X11-unix
      - ./openxr:/openxr
      - ./source:/workspace/isaaclab/source
      - ./scripts:/workspace/isaaclab/scripts
```

Then run:
```bash
docker-compose -f docker-compose.fixed.yaml up -d
docker exec -it isaac-lab-fixed bash
```

This approach ensures all fixes are permanently baked into the image and you never have to run manual fix commands again.

## 📝 Notes

- This setup was tested with Isaac Lab v0.46.1 and Isaac Sim 5.0.0
- Some dependency conflicts are expected and can be safely ignored as long as functionality works
- The PyTorch packaging fix is a workaround for a known issue with Isaac Sim's bundled dependencies
- Pink IK library v3.1.0 is specifically required for Isaac Lab compatibility

## 🆘 Troubleshooting

If you encounter issues:
1. Check the terminal output for specific error messages
2. Ensure all commands are run in the correct order
3. Verify mount paths are correct for your directory structure
4. Check that the Pink IK library is properly installed with `from pink.tasks import DampingTask, FrameTask`
5. If you see OSQP errors, ensure OSQP version 0.6.3 is installed
6. Restart the container if package installations seem to hang

---

*Last updated: February 12, 2026*
*Isaac Lab Version: 0.46.1*
*Isaac Sim Version: 5.0.0*
