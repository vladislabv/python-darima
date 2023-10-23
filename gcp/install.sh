apt-get update
apt-get install -y python-pip
pip install pipenv
pip install rpy2
R -e "install.packages(c('forecast', 'polynom'), repos='https://cran.rstudio.com/')"