# MLOps-TD
Project developed as part of the MLOps course taught by Mr. Fanilo ANDRIANASOLO : functional docker-compose app that provides an UI to do predictions on a pretrained ML model.
The model has been trained on the "penguins.csv" dataset.

To run this project locally, ensure you have Docker and Docker Compose installed, then run:

Bash

# Clone the repository
`code`
git clone <your-repo-link>

# Navigate to the root directory
`code`
cd mlops-td

# Build and start the containers
`code`
docker-compose up --build


The UI will be accessible at http://localhost:8501 and the API documentation at http://localhost:8000/docs.
