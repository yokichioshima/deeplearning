FROM python:3.10.19-slim-trixie
WORKDIR /home
COPY requirements.txt ./
COPY ./src/ .
RUN python -m pip install --upgrade pip setuptools \
    && pip3 install --no-cache-dir -r requirements.txt
CMD ["bash"]
