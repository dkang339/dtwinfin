FROM ghcr.io/scientificcomputing/fenics-gmsh:2024-05-30

COPY RBniCS /opt/RBniCS

# instalㅣ
RUN pip3 install --no-cache-dir -e /opt/RBniCS
RUN pip3 install ipykernel

WORKDIR /home/fenics

