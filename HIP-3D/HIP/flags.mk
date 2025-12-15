CC=hipcc
PGCC=hipcc
CFLAGS=-O3 -D__HIP_ROCclr__ -D__HIP_PLATFORM_AMD__ --rocm-path=${ROCM_PATH} #-D__HIP_ARCH_GFX803__=1 #--offload-arch=gfx803 -x hip
PGCCFLAGS=-O3 -x hip -D__HIP_ROCclr__ -D__HIP_PLATFORM_AMD__ --rocm-path=${ROCM_PATH} #-D__HIP_ARCH_GFX803__=1 #--offload-arch=gfx803 -x hip
# LIBS= -L${ROCM_PATH}/lib -lamdhip64
LIBS=--rocm-path=${ROCM_PATH} -L${ROCM_PATH}/lib -lamdhip64
