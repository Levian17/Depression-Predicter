# Depression Predicter

Depression Predictor is a tool that allows the user to input data into a graphical interface to make an AI model evaluate the probabilities of having depression based on the "Student Depression" dataset stats.

## Dataset

The dataset is explored in the 'data/data_visualization_english.ipynb' and 'data/data_visualization_spanish.ipynb' directories in high detail.

The dataset is included in the project in the data directory 'data/Student Depression Dataset.csv'. To learn more about it, check out [Kaggle](https://www.kaggle.com/datasets/hopesb/student-depression-dataset).

!(images/dataset_head.png)

## Project Structure
```
Depression-Predicter
│   LICENSE.txt
│   main.py
│   README.md
│   requirements.txt
│
├───data
│       data_visualization_english.ipynb
│       data_visualization_spanish.ipynb
│       Student Depression Dataset.csv
│
├───model_files
│       depression_model.pth
│       encoder.joblib
│       scaler.joblib
│
└───src
        GUI.py
        model_definition.py
```
## Instalation

 1. Clone the repository:
```sh
git clone https://github.com/Levian17/Depression-Predicter
cd Depression-Predicter
```

2. Install requirements:
```sh
pip install -r requirements.txt
```

## Usage
The main file allows the user to display the interface or to train and save the AI model. In its default state, the train_and_save method is commented, so it will not retrain the AI model, mainly because the project comes with a pre-trained model.

Once the interface is displayed, the user has to insert values into the necessary widgets. Once all data has been inserted, clicking the Calculate Results button will pass the data to the AI model, and the results will be displayed on the screen.

## License
This project is under the Creative Commons Attribution-NonCommercial 4.0 International License
