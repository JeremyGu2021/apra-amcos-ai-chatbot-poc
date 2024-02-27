FROM public.ecr.aws/lambda/python:3.10-arm64

COPY requirements.txt ${LAMBDA_TASK_ROOT}

RUN pip install -r requirements.txt

COPY main.py ${LAMBDA_TASK_ROOT}
COPY sagemaker_utils.py ${LAMBDA_TASK_ROOT}

ENV PINECONE_API_KEY '19ff39f1-6252-456e-a6ca-a257faa2df8b'
ENV OPENAI_API_KEY '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'

CMD [ "main.handler" ]