FROM python:3.10

# install libgl1-mesa-glx for opencv
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

COPY . .

# Run the init script(slow), crawling 30 dates data
# It will take about 10 minutes to finish
# It use multi-thread to crawl data
CMD python -m app.init_script --update_data --env docker && \
    python -m uvicorn app.main:app --host 0.0.0.0 --port 8090

# Run the init script without update data(faster)
# CMD python -m app.init_script --env docker && \
#     python -m uvicorn app.main:app --host 0.0.0.0 --port 8090
