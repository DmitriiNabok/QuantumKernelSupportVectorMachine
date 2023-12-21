FROM python:3.9-bullseye 

RUN pip install --upgrade pip ipython ipykernel

COPY ./requirements.txt /
RUN pip install -r /requirements.txt

# Install the custom QKSVM package from source
COPY src/qksvm/* src/qksvm/
COPY tests/* tests/
COPY setup.cfg .
COPY setup.py .
COPY README.md .
COPY LICENSE .

RUN pip install -e /
RUN pip install pytest

COPY entrypoint.sh entrypoint.sh

# At runtime, mount the connection file to /tmp/connection_file.json
ENTRYPOINT [ "./entrypoint.sh"]

# Build 
# docker build --rm --tag qksvm-docker .
