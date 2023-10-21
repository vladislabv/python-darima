# Distributed Auto-Regressiv Integrated Moving-Average Model (study Project)

The project was dedicated to dig into the distributed computing and the tools are mainly used for it. We, as students of the Applied University of Southern Westfalia, took as an example the DARIMA Models to reproduce the code and rewrite it as we this understand.

Therefore, we would like to emphasize

- We do not claim any autorship on every part of the repository, since it was inspired by https://github.com/xqnwang/darima.
- We do not claim any rights on the code, since this was in educational purpose only.

Software required
--
- Apache Spark >= 3.4.1
- R Version >= 4.2.3
- Python Version >= 3.11

Package Premises
--
- R
```R
install.packages("polynom")
install.packages("forecast")
```
- Python

`pip install -e requirements.txt`

How to get this running?
--
On a single local master node:

`python ./darima.py`

On a running (remote) master node:

Assuming that the `$SPARK_HOME` environment variable points to your local Spark installation folder, then the model can be run from the project's root directory using the following command from the terminal,

```bash
$SPARK_HOME/bin/spark-submit \
--master local[*] \
--py-files packages.zip \
--files configs/darima_config.json \
./darima.py
```

Briefly, the options supplied serve the following purposes:

- `--master local[*]` - the address of the Spark cluster to start the model on. If you have a Spark cluster in operation (either in single-executor mode locally, or something larger in the cloud) and want to send the job there, then modify this with the appropriate Spark IP - e.g. `spark://the-clusters-ip-address:7077`;
- `--files configs/darima_config.json` - the (optional) path to any config file that may be required by the DARIMA model;
- `--py-files packages.zip` - archive containing Python dependencies (modules) referenced by the model;
- `./darima.py` - the Python file containing the DARIMA model to execute.

Full details of all possible options can be found [here](http://spark.apache.org/docs/latest/submitting-applications.html). Note, that we have left some options to be defined within the model (which is actually a Spark application) - e.g. `spark.cores.max` and `spark.executor.memory` are defined in the Python script as it is felt that the job should explicitly contain the requests for the required cluster resources.

## Project Structure

The basic project structure is as follows:

```bash
root/
 |-- R/
 |   |-- auto_arima.R
 |-- configs/
 |   |-- darima_config.json
 |-- data/
 |   |-- *.csv
 |   |-- *.parquet
 |-- docs/
 |   |-- Makefile
 |   |-- conf.py
 |-- py_spark/
 |   |-- logging.py
 |   |-- spark.py
 |-- py_handlers/
 |   |-- converters.py
 |   |-- utils.py
 |-- py_handlers/
 |   |-- logging.py
 |   |-- spark.py
 |   darima.py
 |   build_dependencies.sh
 |   packages.zip
 |   requirements.txt
 |   r_requirements.txt
```

The main Python module containing the DARIMA model (which will be sent to the Spark cluster), is `./darima.py`. Any external configuration parameters required by `darima.py` are stored in JSON format in `configs/darima_config.json`. Additional modules that support this model can be kept in the `dependencies` folder. In the project's root we include `build_dependencies.sh`, which is a bash script for building these dependencies into a zip-file to be sent to the cluster (`packages.zip`).
