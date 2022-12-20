## Presentation

This repository contains the code for our project py_docr, developed during our [Data Scientist training](https://datascientest.com/en/data-scientist-course) at [DataScientest](https://datascientest.com/).

The goal of this project is to classify documents, especially National ID cards and Passports.

This project was developed by the following team :

- Rym BEN HASSINE ([GitHub](https://github.com/) / [LinkedIn](https://www.linkedin.com/in/rym-ben-hassine-136b34109/))
- Fehrat SADOUN ([GitHub](https://github.com/) / [LinkedIn](https://www.linkedin.com/in/ferhat-sadoun-0baa54249/))
- Bogdan POSEA ([GitHub](https://github.com/) / [LinkedIn](https://www.linkedin.com/in/bogdan-posea-a9324b38/))
- Eric GASNIER ([GitHub](https://github.com/egasnier) / [LinkedIn](https://www.linkedin.com/in/ericgasnier))


You can browse and run the [notebooks](./notebooks). You will need to install the dependencies (in a dedicated environment) :

```
pip install -r requirements.txt
```



** PART NOT YET FINALISED **

## Streamlit App

To run the app :

```shell
cd streamlit_app
conda create --name my-awesome-streamlit python=3.9
conda activate my-awesome-streamlit
pip install -r requirements.txt
streamlit run app.py
```

The app should then be available at [localhost:8501](http://localhost:8501).

**Docker**

You can also run the Streamlit app in a [Docker](https://www.docker.com/) container. To do so, you will first need to build the Docker image :

```shell
cd streamlit_app
docker build -t streamlit-app .
```

You can then run the container using :

```shell
docker run --name streamlit-app -p 8501:8501 streamlit-app
```

And again, the app should then be available at [localhost:8501](http://localhost:8501).
