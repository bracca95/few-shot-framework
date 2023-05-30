FROM python:3.8.16

WORKDIR /code_protonet

COPY requirements.txt .
COPY main.py .
COPY README.md .
COPY config .
COPY src .

RUN pip install --no-cache-dir wheel
RUN pip install --no-cache-dir torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip install --no-cache-dir -r requirements.txt

CMD [ "python", "./main.py" ]