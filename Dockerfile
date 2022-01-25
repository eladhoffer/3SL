# Build: docker build -t project_name .
# Run: docker run --gpus all -it --rm project_name

FROM pytorch/pytorch

# Copy all files
ADD . /3SL
WORKDIR /3SL

RUN pip install -r requirements.txt
