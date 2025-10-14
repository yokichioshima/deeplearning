FROM python:3.10.19-slim-trixie

COPY requirements.txt ./
COPY src/ /home/ 
RUN pip install --no-cache-dir -r requirements.txt
CMD ["bash"]
