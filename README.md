# Overview

**Implementation Overview: Airbnb House Price Prediction with Apache Spark**

This project implements a house price prediction model for Airbnb listings, utilizing Apache Spark to handle the data processing and machine learning tasks. This implementation evolves from a lab exercise from my [Big Data Programming course](https://hub.ucd.ie/usis/!W_HU_MENU.P_PUBLISH?p_tag=MODULE&MODULE=COMP47470), solving the exercise, and adapting the original Jupyter notebook into an executable Python script which can run in a Docker container with all the required dependencies. 

**Key Updates and Enhancements:**

1. **Adaptation to Python Script:**
   - Converted the original Jupyter notebook into a standalone Python script. Although this was developed to run local clusters, it can be adapted to run on other clusters such as on Dataproc with GCP as I have implemented in my [data engineering project](https://github.com/peter716/data_engineering_credit_fraud_project)

2. **Utilization of Docker:**
   - Integrated Docker to containerize the application, ensuring consistency across different computing environments and simplifying subsequent deployment process. Docker encapsulates the application along with its dependencies, making it easy to deploy across different systems without compatibility issues. With some changes/updates, I have adapted the installation process in the lab notebook to create the Dockerfile, which is used to build the image.

3. **Incorporation of the Random Forest Model:**
   - Incorporated an additionally Random Forest algorithm, leveraging its robustness and accuracy for regression tasks. This model is known for handling overfitting better than many other models and can provide important insights into feature importance.

4. **Hyperparameter Tuning:**
   - Initiated the process of hyperparameter tuning to optimize model performance. This step is crucial for enhancing the predictive accuracy by fine-tuning the model parameters. The process is currently in progress, with the aim to systematically explore a range of parameter values.

**Instructions for Running the Application in Docker:**

To deploy and run this Spark machine learning application using Docker, follow these steps:

1. **Build the Docker Image:**
   - Execute the following command to build the Docker image from the Dockerfile in the project directory. This image will include the necessary environment, along with Apache Spark and all required Python dependencies:
     ```bash
     docker build -t test:sparkml .
     ```
   - This command creates a Docker image named `test` with the tag `sparkml`. The Dockerfile is already set up to install Spark and any other dependencies, configure the environment, and set up the entry points for the Python script.

2. **Run the Docker Container:**
   - Once the image is built, you can run the container using:
     ```bash
     docker run -it test:sparkml
     ```
   - This command starts a container where the Spark application can execute. 