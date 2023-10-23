gsutil cp install.sh gs://ok-spark-bucket/install.sh

gcloud dataproc clusters create ok-py-cl \
  --region us-central1 \
  --master-machine-type n1-standard-4 \
   --master-boot-disk-size 50 \
   --num-workers 3 \
   --worker-machine-type n1-standard-4 \
   --worker-boot-disk-size 50 \
   --image-version 1.5-debian10 \
   --scopes 'https://www.googleapis.com/auth/cloud-platform' \
   --project pyspark-402405 \
   --enable-component-gateway \
   --initialization-actions=gs://ok-spark-bucket/install.sh
