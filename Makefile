INPUT=in
OUTPUT=out
GPUS=0,1,2,3

build:
        docker build -t sr -f Dockerfile.gpu .

run:
        docker run -it --rm -e CUDA_VISIBLE_DEVICES=${GPUS} --runtime nvidia -v ${INPUT}:/input -v ${OUTPUT}:/output sr
