gcloud dataproc jobs submit pyspark \
  --project pyspark-402405 \
  --cluster ok-py-cl \
  --region us-central1 \
  --py-files ../packages.zip \
  --files ../auto_arima.R,../darima_config.json \
  ../darima.py