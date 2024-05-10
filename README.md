# Overview

**Implementation Overview: Airbnb House Price Prediction with Apache Spark**

This project implements a house price prediction model for Airbnb listings, utilizing Apache Spark to handle the data processing and machine learning tasks. This implementation evolves from an educational lab from my Big Data Programming course, adapting the original Jupyter notebook format into a more robust form suitable for deployment and execution as a job on a distributed cluster. 

**Key Updates and Enhancements:**

1. **Adaptation to Python Script:**
   - Converted the original Jupyter notebook into a standalone Python script. This allows the model to be run as an automated job on a cluster, suitable for larger datasets and batch processing scenarios.

2. **Utilization of Docker:**
   - Integrated Docker to containerize the application, ensuring consistency across different computing environments and simplifying the deployment process. Docker encapsulates the application along with its dependencies, making it easy to deploy across different systems without compatibility issues.

3. **Incorporation of the Random Forest Model:**
   - Introduced the Random Forest algorithm, leveraging its robustness and accuracy for regression tasks. This model is known for handling overfitting better than many other models and provides important insights into feature importance.

4. **Hyperparameter Tuning:**
   - Initiated the process of hyperparameter tuning to optimize model performance. This step is crucial for enhancing the predictive accuracy by fine-tuning the model parameters. The process is currently in progress, with the aim to systematically explore a range of parameter values.

**Running the Application in Docker:**

To deploy and run this Spark machine learning application using Docker, follow these steps:

1. **Build the Docker Image:**
   - Execute the following command to build the Docker image from the Dockerfile in the project directory. This image will include the necessary environment, along with Apache Spark and all required Python dependencies:
     ```bash
     docker build -t test:sparkml .
     ```
   - This command creates a Docker image named `test` with the tag `sparkml`. The Dockerfile should be set up to install Spark and any other dependencies, configure the environment, and set up the entry points for the Python script.

2. **Run the Docker Container:**
   - Once the image is built, you can run the container using:
     ```bash
     docker run -it test:sparkml
     ```
   - This command starts a container where the Spark application can execute. The `-it` flag allocates a pseudo-TTY connected to the container’s stdin; creating an interactive bash shell in the container.

**Additional Configuration and Details:**

- **Environment Variables:**
  - If there are environment variables that need to be set for Spark or the application itself (such as database credentials or API keys), they can be specified using the `-e` option with the `docker run` command.

- **Data Volumes:**
  - For data persistence or to access datasets from the host machine, you can use Docker volumes to mount data directories from the host into the container. Add `-v /path/on/host:/path/in/container` to the `docker run` command.

- **Logging and Output:**
  - Ensure that logs and outputs are directed to stdout or are being properly collected and monitored if necessary for debugging and performance tuning.

This approach not only simplifies deployment but also aligns with best practices for developing and deploying big data applications in scalable and reproducible environments.
