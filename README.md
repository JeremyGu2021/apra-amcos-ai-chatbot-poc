### Development

#### build

```sh
docker build --platform linux/amd64 -t 635773515765.dkr.ecr.ap-southeast-2.amazonaws.com/apra-amcos-chatbot-sagemaker-poc:latest .
```

#### push to aws ecr

```sh
docker push 635773515765.dkr.ecr.ap-southeast-2.amazonaws.com/apra-amcos-chatbot-sagemaker-poc
```

#### testing locally (optional)

```sh
docker-compose up
```

then from other terminal

```sh
curl "http://localhost:5000/2015-03-31/functions/function/invocations" -d '{"user_message":"what is APRA AMCOS?"}'
```
