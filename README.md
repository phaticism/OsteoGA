# OsteoGA


## Usage
To get started, follow these steps:

### 1. Build the Docker Image

Ensure that Docker daemon is running, then execute the following command to build the image:

```bash
docker build -t be .
```

### 2. Run the application
Replace `CHOSEN_PORT` with your desired port number, and execute the following command to run the application:
```bash
docker run -dp CHOSEN_PORT:5000 --name osteoga-core be
```
Now, your application should be up and running on the specified port.

### 3. Stop the application
To stop, use:
```bash
docker stop osteoga-core
```

### 4. Remove the application
First, remove the container:
```bash
docker rm osteoga-core
```
Then the image:
```bash
docker rmi be
```
