FROM python:3.11.8-bookworm

WORKDIR /src

RUN apt-get update && apt-get install -y libgl1-mesa-glx

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD [ "python", "server.py" ]
