# Error-aware digital twin of a thermal fin in heat sink systems
 
This project constructs an error-aware digital twin of a heat sink system benchmarked using a thermal fin. It requires **Docker** to install **FEniCS** and **RBniCS** (tested on macOS) and Dockerfile is provided to set up docker environment. RBniCS is included as a Git submodule.


## 1. Clone repository (with RBniCS Submodule)
This project includes RBniCS as a Git submodule. Cloning with `--recursive` automatically downloads RBniCS.

```bash
git clone --recursive https://github.com/dkang339/dtwinfin.git
````

## 2. Build Docker image

The provided Dockerfile installs
1. FEniCS Legacy (DOLFIN)
2. RBniCS
3. Jupyter RBniCS Kernel (ipykernel)

From the project root, build the docker image with
````bash
docker build -t fenics-rbnics .
````

## 3. Run Docker container

From the project root, run
````bash
docker run -it --name rbnics -v $(pwd):/home/shared fenics-rbnics
````
where 
* `--name rbnics`: container name
* `-v $(pwd):/home/shared`: mounts your repo inside the container directory
* `fenics-rbnics`: image name

To re-enter container, run
````bash
docker exec -it rbnics bash
````

To stop container, run
````bash
docker stop rbnics
````

## 4. Use RBniCS Jupyter kernel

To run digital twin examples in `scripts/*.ipynb`:
1. Install Dev Containers extension in VS Code
2. Attach VS Code to Running Container
3. Open the notebook inside the docker environment
4. Select RBniCS Kernel
