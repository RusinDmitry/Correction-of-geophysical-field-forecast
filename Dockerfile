FROM pytorch/pytorch

RUN pip install scikit-learn
#RUN pip install pandas
#RUN pip install netCDF4
RUN pip install matplotlib
#RUN conda install -c conda-forge wrf-python=1.3.4.1
COPY . /home

WORKDIR /home/experiments/constantBaseline


#CMD ["python", "experiments/constantBaseline/main.py"]
