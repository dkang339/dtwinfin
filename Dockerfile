FROM ghcr.io/scientificcomputing/fenics-gmsh:2024-05-30

# install RBnics
COPY RBniCS /home/RBniCS
RUN pip3 install --no-cache-dir -e /home/RBniCS

# add jupyter kernel
RUN pip3 install ipykernel
RUN python3 -m ipykernel install --user --name rbnics --display-name "RBniCS Kernel"

WORKDIR /home

